"""Clone management (/clones, download, delete, import), /clone-voice's meta.json and upload cap,
the prompt.pt validation, and /analyze-clips.

torch is stubbed: its load/save become pickle, and tensors are FakeTensor, so _load_prompt and
_validate_prompt run for real against a stand-in VoiceClonePromptItem. S3 is an in-memory fake.
"""

import asyncio
import contextlib
import datetime
import io
import json
import os
import pickle
import sys
import types
import unittest
import zipfile
from dataclasses import dataclass
from typing import Optional
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.device = lambda d: d
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules["torch"] = _torch

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import tts_server  # noqa: E402
from tts_server import TTSModelManager, TTSState  # noqa: E402


# ── torch / qwen_tts stand-ins ────────────────────────────────────────────────


class FakeTensor:
    def __init__(self, shape, floating=True, dtype=None):
        self.shape = tuple(shape)
        self.floating = floating
        self.dtype = dtype or ("float32" if floating else "int64")

    def is_floating_point(self):
        return self.floating

    def is_complex(self):
        return False

    def dim(self):
        return len(self.shape)

    def numel(self):
        return int(np.prod(self.shape)) if self.shape else 1

    def detach(self):
        return self

    def cpu(self):
        return self

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and (self.shape, self.dtype) == (other.shape, other.dtype)


@dataclass
class FakeItem:  # the shape of qwen_tts's VoiceClonePromptItem
    ref_code: Optional[FakeTensor]
    ref_spk_embedding: FakeTensor
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


def icl_item(text="hello there"):
    return FakeItem(FakeTensor((120, 16), floating=False), FakeTensor((1024,)), False, True, text)


def xvec_item():
    return FakeItem(None, FakeTensor((1024,)), True, False, None)


def _fake_load(buf, map_location=None, weights_only=False):
    assert weights_only, "prompt.pt must be loaded with weights_only=True"
    data = buf.read()
    if not data.startswith(b"PT"):
        raise RuntimeError("not a torch archive")
    return pickle.loads(data[2:])


def _fake_save(obj, buf):
    buf.write(b"PT" + pickle.dumps(obj))


def saved(obj) -> bytes:
    buf = io.BytesIO()
    _fake_save(obj, buf)
    return buf.getvalue()


def _to_array(t: FakeTensor):
    """tts_server._to_array for a FakeTensor: zeros of its shape, float32 or int64, and its dtype name."""
    return np.zeros(t.shape, dtype=np.float32 if t.floating else np.int64), t.dtype


def _from_array(a: np.ndarray, dtype: str) -> FakeTensor:
    return FakeTensor(a.shape, floating=a.dtype.kind == "f", dtype=dtype)


@contextlib.contextmanager
def fake_torch():
    torch = sys.modules["torch"]
    serialization = types.SimpleNamespace(safe_globals=lambda classes: contextlib.nullcontext())
    with contextlib.ExitStack() as stack:
        for name, value in (
            ("load", _fake_load), ("save", _fake_save), ("is_tensor", lambda x: isinstance(x, FakeTensor)),
            ("bool", "bool"), ("serialization", serialization),
        ):
            stack.enter_context(mock.patch.object(torch, name, value, create=True))
        stack.enter_context(mock.patch.object(tts_server, "_prompt_item_cls", lambda: FakeItem))
        stack.enter_context(mock.patch.object(tts_server, "_to_array", _to_array))
        stack.enter_context(mock.patch.object(tts_server, "_from_array", _from_array))
        yield


def packed(items) -> bytes:
    """An archive's prompt.safetensors for these prompt items (call inside fake_torch)."""
    return tts_server._prompt_to_safetensors(items)


# ── S3 stand-in ───────────────────────────────────────────────────────────────


class FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.modified: dict[str, datetime.datetime] = {}
        self.refuse_delete: set[str] = set()

    def _missing(self, op):
        return ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, op)

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body
        self.modified[Key] = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(
            seconds=len(self.modified)
        )

    def download_fileobj(self, Bucket, Key, buf):
        if Key not in self.objects:
            raise self._missing("GetObject")
        buf.write(self.objects[Key])

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self._missing("HeadObject")
        return {}

    def get_paginator(self, name):
        s3 = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for k in s3.objects if k.startswith(Prefix))
                yield {"Contents": [{"Key": k, "LastModified": s3.modified[k]} for k in keys]}

        return Paginator()

    def delete_objects(self, Bucket, Delete):
        deleted, errors = [], []
        for obj in Delete["Objects"]:
            if obj["Key"] in self.refuse_delete:
                errors.append({"Key": obj["Key"], "Code": "AccessDenied"})
            else:
                self.objects.pop(obj["Key"], None)
                deleted.append(obj)
        return {"Deleted": deleted, "Errors": errors}


def archive(prompt: Optional[bytes] = None, meta: Optional[dict] = None, extra: Optional[dict] = None) -> bytes:
    """A clone archive: *prompt* is its prompt.safetensors."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if prompt is not None:
            zf.writestr("prompt.safetensors", prompt)
        if meta is not None:
            zf.writestr("meta.json", json.dumps(meta))
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return buf.getvalue()


class _Base(unittest.TestCase):
    def setUp(self):
        self.s3 = FakeS3()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(fake_torch())
        stack.enter_context(mock.patch.object(tts_server, "_get_s3_client", lambda: self.s3))
        stack.enter_context(mock.patch.object(tts_server, "_clone_cache", tts_server.OrderedDict()))
        self.client = TestClient(tts_server.app)

    def store(self, companion_id, items, meta=None):
        self.s3.put_object(None, f"voice_clones/{companion_id}/prompt.pt", saved(items))
        if meta is not None:
            self.s3.put_object(None, f"voice_clones/{companion_id}/meta.json", json.dumps(meta).encode())


# ── prompt.pt validation ──────────────────────────────────────────────────────


class LoadPromptTests(_Base):
    def test_accepts_what_clone_voice_stores(self):
        for items in ([icl_item()], [xvec_item()]):
            out = tts_server._load_prompt(saved(items))
            self.assertEqual(len(out), 1)
            self.assertIsInstance(out[0], FakeItem)
            self.assertEqual(out[0].icl_mode, items[0].icl_mode)
            self.assertEqual(out[0].ref_text, items[0].ref_text)

    def test_xvector_prompt_drops_any_ref_code(self):
        item = xvec_item()
        item.ref_code = FakeTensor((3, 16), floating=False)
        self.assertIsNone(tts_server._load_prompt(saved([item]))[0].ref_code)

    def test_refuses_other_engines_and_malformed_prompts(self):
        bad = {
            "xtts": {"gpt_cond_latent": FakeTensor((1, 32, 1024)), "speaker_embedding": FakeTensor((1, 512, 1))},
            "miso": {"ref_audio": FakeTensor((24000,)), "ref_text": "hi", "sample_rate": 24000, "engine": "miso"},
            "two items": [icl_item(), xvec_item()],
            "bare item": icl_item(),
            "icl without text": [icl_item(text="")],
            "float ref_code": [FakeItem(FakeTensor((5, 16)), FakeTensor((8,)), False, True, "x")],
            "bool ref_code": [FakeItem(FakeTensor((5, 16), floating=False, dtype="bool"), FakeTensor((8,)), False,
                                       True, "x")],
            "no ref_code": [FakeItem(None, FakeTensor((8,)), False, True, "x")],
            "both modes": [FakeItem(None, FakeTensor((8,)), True, True, None)],
            "neither mode": [FakeItem(None, FakeTensor((8,)), False, False, None)],
            "int mode": [FakeItem(None, FakeTensor((8,)), 1, False, None)],
            "2-D embedding": [FakeItem(None, FakeTensor((2, 8)), True, False, None)],
            "int embedding": [FakeItem(None, FakeTensor((8,), floating=False), True, False, None)],
            "huge embedding": [FakeItem(None, FakeTensor((100_000,)), True, False, None)],
            "text not str": [FakeItem(None, FakeTensor((8,)), True, False, 42)],
        }
        for name, obj in bad.items():
            with self.assertRaises(ValueError, msg=name):
                tts_server._load_prompt(saved(obj))
        with self.assertRaisesRegex(ValueError, "not a Qwen clone"):
            tts_server._load_prompt(b"garbage")

    def test_stored_clone_loads_for_synthesis_and_is_cached(self):
        self.store("alice", [icl_item()])
        key = "voice_clones/alice/prompt.pt"
        prompt = tts_server._get_clone_prompt(key)
        self.assertIsInstance(prompt[0], FakeItem)
        self.s3.objects.clear()
        self.assertIs(tts_server._get_clone_prompt(key), prompt)

    def test_invalid_stored_clone_is_a_value_error(self):
        self.s3.put_object(None, "voice_clones/x/prompt.pt", saved({"gpt_cond_latent": FakeTensor((1,))}))
        with self.assertRaisesRegex(ValueError, "not a valid Qwen clone"):
            tts_server._get_clone_prompt("voice_clones/x/prompt.pt")
        with self.assertRaisesRegex(ValueError, "not found"):
            tts_server._get_clone_prompt("voice_clones/nobody/prompt.pt")


# ── GET /clones ───────────────────────────────────────────────────────────────


class ListClonesTests(_Base):
    def test_lists_meta_and_legacy_clones_newest_first(self):
        self.store("old", [xvec_item()])  # cloned before Qwen wrote meta.json
        self.store("alice", [icl_item()], {"companion_id": "alice", "name": "Alice", "created_at": 2e9})
        self.s3.put_object(None, "voice_clones/orphan/meta.json", b'{"name": "no prompt"}')
        self.s3.put_object(None, "voice_clones/bad id!/prompt.pt", b"x")
        self.s3.put_object(None, "generated/voice/a.wav", b"x")

        r = self.client.get("/clones")
        self.assertEqual(r.status_code, 200, r.text)
        clones = r.json()["clones"]
        self.assertEqual([c["companion_id"] for c in clones], ["alice", "old"])
        self.assertEqual(clones[0]["name"], "Alice")
        self.assertEqual(clones[0]["voice_clone_id"], "voice_clones/alice/prompt.pt")
        legacy = clones[1]
        self.assertEqual(
            (legacy["voice_clone_id"], legacy["name"], legacy["engine"]),
            ("voice_clones/old/prompt.pt", "old", "qwen-tts"),
        )
        self.assertIsInstance(legacy["created_at"], float)

    def test_unreadable_meta_falls_back_to_the_prompt(self):
        self.store("bob", [xvec_item()])
        self.s3.put_object(None, "voice_clones/bob/meta.json", b"[1, 2]")
        clones = self.client.get("/clones").json()["clones"]
        self.assertEqual([c["companion_id"] for c in clones], ["bob"])

    def test_storage_failure_is_500_without_internals(self):
        def boom():
            raise RuntimeError("endpoint http://minio.internal:9000 refused")

        with mock.patch.object(tts_server, "_get_s3_client", boom):
            r = self.client.get("/clones")
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("minio.internal", r.text)
        self.assertIn("request_id", r.json())


# ── download / delete ─────────────────────────────────────────────────────────


class DownloadDeleteTests(_Base):
    def test_download_zips_prompt_and_meta(self):
        self.store("alice", [icl_item()], {"companion_id": "alice", "name": "Alice", "engine": "qwen-tts"})
        r = self.client.get("/clones/alice/download")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "application/zip")
        self.assertIn('filename="alice.zip"', r.headers["content-disposition"])
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        self.assertEqual(sorted(zf.namelist()), ["meta.json", "prompt.safetensors"])
        meta = tts_server._safetensors_metadata(zf.read("prompt.safetensors"))
        self.assertEqual(
            meta,
            {"format": "qwen-clone", "format_version": "1", "engine": "qwen-tts", "x_vector_only_mode": "false",
             "icl_mode": "true", "ref_spk_embedding_dtype": "float32", "ref_text": "hello there"},
        )
        self.assertEqual(json.loads(zf.read("meta.json"))["name"], "Alice")

    def test_download_of_a_legacy_clone_carries_generated_meta(self):
        self.store("old", [xvec_item()])
        zf = zipfile.ZipFile(io.BytesIO(self.client.get("/clones/old/download").content))
        self.assertEqual(json.loads(zf.read("meta.json"))["engine"], "qwen-tts")

    def test_download_and_delete_validate_the_id_and_404(self):
        for method in ("get", "delete"):
            path = "/clones/{}/download" if method == "get" else "/clones/{}"
            self.assertEqual(getattr(self.client, method)(path.format("nobody")).status_code, 404)
            self.assertEqual(getattr(self.client, method)(path.format("a.b")).status_code, 422)
            self.assertEqual(getattr(self.client, method)(path.format("x" * 65)).status_code, 422)

    def test_delete_removes_everything_and_evicts_the_cache(self):
        self.store("alice", [icl_item()], {"name": "Alice"})
        self.store("alice2", [icl_item()])
        tts_server._get_clone_prompt("voice_clones/alice/prompt.pt")
        r = self.client.delete("/clones/alice")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"status": "deleted", "companion_id": "alice", "objects": 2})
        self.assertEqual(sorted(self.s3.objects), ["voice_clones/alice2/prompt.pt"])  # the prefix is exact
        self.assertNotIn("voice_clones/alice/prompt.pt", tts_server._clone_cache)

    def test_partial_delete_is_500(self):
        self.store("alice", [icl_item()], {"name": "Alice"})
        self.s3.refuse_delete.add("voice_clones/alice/meta.json")
        r = self.client.delete("/clones/alice")
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("deleted", r.json().get("status", ""))


# ── POST /clones/import ───────────────────────────────────────────────────────


class ImportTests(_Base):
    def _import(self, data: bytes, **form):
        return self.client.post("/clones/import", files={"archive": ("a.zip", data, "application/zip")}, data=form)

    def test_round_trip_through_download(self):
        self.store("alice", [icl_item()], {"companion_id": "alice", "name": "Alice", "num_clips": 1,
                                           "total_seconds": 7.5, "engine": "qwen-tts"})
        data = self.client.get("/clones/alice/download").content
        r = self._import(data, companion_id="alice-copy")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["voice_clone_id"], "voice_clones/alice-copy/prompt.pt")
        self.assertEqual((body["name"], body["total_seconds"], body["duration"]), ("Alice", 7.5, 7.5))
        self.assertTrue(body["imported"])
        self.assertTrue(body["has_transcript"])
        self.assertEqual(body["engine"], "qwen-tts")
        stored = tts_server._load_prompt(self.s3.objects["voice_clones/alice-copy/prompt.pt"])
        self.assertEqual(stored[0].ref_text, "hello there")
        meta = json.loads(self.s3.objects["voice_clones/alice-copy/meta.json"])
        self.assertEqual(meta["companion_id"], "alice-copy")
        # It is usable for synthesis straight away.
        self.assertIs(tts_server._get_clone_prompt(body["voice_clone_id"]), tts_server._clone_cache[body["voice_clone_id"]])

    def test_target_id_comes_from_override_then_meta_then_name(self):
        prompt = packed([xvec_item()])
        r = self._import(archive(prompt, {"companion_id": "From Meta"}))
        self.assertEqual(r.json()["companion_id"], "from-meta")
        r = self._import(archive(prompt), voice_name="My Voice!")
        self.assertEqual((r.json()["companion_id"], r.json()["name"]), ("my-voice", "My Voice!"))
        r = self._import(archive(prompt, {"companion_id": "../../etc"}))
        self.assertEqual(r.json()["companion_id"], "etc")

    def test_existing_id_is_409_unless_replace(self):
        self.store("alice", [icl_item()])
        data = archive(packed([xvec_item()]), {"companion_id": "alice"})
        self.assertEqual(self._import(data).status_code, 409)
        self.assertTrue(tts_server._load_prompt(self.s3.objects["voice_clones/alice/prompt.pt"])[0].icl_mode)
        r = self._import(data, replace="true")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(tts_server._load_prompt(self.s3.objects["voice_clones/alice/prompt.pt"])[0].x_vector_only_mode)

    def test_other_engines_archives_are_refused(self):
        from safetensors.numpy import save

        latents = {
            "gpt_cond_latent": np.zeros((1, 32, 1024), np.float32),
            "speaker_embedding": np.zeros(512, np.float32),
        }
        xtts = archive(save(latents, metadata={"engine": "xtts"}), {"companion_id": "x", "name": "X"})
        xtts_unlabelled = archive(save(latents), {"companion_id": "x"})
        miso = archive(packed([xvec_item()]), {"companion_id": "m", "engine": "miso"})
        for data, needle in ((xtts, "'xtts' clone"), (xtts_unlabelled, "ref_spk_embedding"), (miso, "miso")):
            r = self._import(data)
            self.assertEqual(r.status_code, 422, r.text)
            self.assertIn(needle, r.json()["error"])
        self.assertEqual(self.s3.objects, {})

    def test_bad_archives_are_422(self):
        cases = {
            "not a zip": b"PK not really",
            "no prompt": archive(None, {"name": "x"}),
            "garbage prompt": archive(b"not safetensors"),
            "too many entries": archive(packed([xvec_item()]), extra={f"f{i}": b"" for i in range(20)}),
            "huge meta": archive(packed([xvec_item()]), extra={"meta.json": b" " * (tts_server._MAX_META_BYTES + 1)}),
        }
        for name, data in cases.items():
            self.assertEqual(self._import(data).status_code, 422, name)
        self.assertEqual(self.s3.objects, {})

    def test_oversized_archive_is_413(self):
        with mock.patch.object(tts_server, "_MAX_ARCHIVE_BYTES", 100):
            r = self._import(archive(packed([xvec_item()]), {"name": "x" * 200}))
        self.assertEqual(r.status_code, 413)
        self.assertEqual(self.s3.objects, {})

    def test_a_legacy_prompt_pt_is_refused_unread(self):
        """GHSA-63cw-57p8-fm3p: torch.load(weights_only=True) on an upload can run code on torch < 2.10, whatever
        the allowlist."""
        legacy = io.BytesIO()
        with zipfile.ZipFile(legacy, "w") as zf:
            zf.writestr("prompt.pt", saved([xvec_item()]))
            zf.writestr("meta.json", json.dumps({"companion_id": "old"}))
        with mock.patch.object(sys.modules["torch"], "load", side_effect=AssertionError("unpickled an upload")):
            r = self._import(legacy.getvalue())
        self.assertEqual(r.status_code, 422, r.text)
        self.assertIn("legacy prompt.pt", r.json()["error"])
        self.assertEqual(self.s3.objects, {})

    def test_the_speaker_embedding_keeps_its_dtype(self):
        """numpy has no bfloat16: the archive carries float32 and names the dtype, and import casts it back."""
        item = icl_item()
        item.ref_spk_embedding = FakeTensor((1024,), dtype="bfloat16")
        self.store("bf", [item], {"companion_id": "bf"})
        r = self._import(self.client.get("/clones/bf/download").content, companion_id="bf2")
        self.assertEqual(r.status_code, 200, r.text)
        stored = tts_server._load_prompt(self.s3.objects["voice_clones/bf2/prompt.pt"])[0]
        self.assertEqual(stored.ref_spk_embedding, FakeTensor((1024,), dtype="bfloat16"))
        self.assertEqual(stored.ref_code, FakeTensor((120, 16), floating=False))

    def test_bad_prompt_metadata_and_tensors_are_422(self):
        from safetensors.numpy import save

        spk = np.zeros(1024, np.float32)
        flags = {"x_vector_only_mode": "true", "icl_mode": "false"}
        cases = {
            "no flags": save({"ref_spk_embedding": spk}),
            "bad flag": save({"ref_spk_embedding": spk}, metadata={**flags, "icl_mode": "yes"}),
            "both modes": save({"ref_spk_embedding": spk}, metadata={**flags, "icl_mode": "true"}),
            "nan": save({"ref_spk_embedding": np.full(1024, np.nan, np.float32)}, metadata=flags),
            "int embedding": save({"ref_spk_embedding": spk.astype(np.int32)}, metadata=flags),
            "odd dtype": save({"ref_spk_embedding": spk}, metadata={**flags, "ref_spk_embedding_dtype": "complex64"}),
            "icl without code": save(
                {"ref_spk_embedding": spk},
                metadata={"x_vector_only_mode": "false", "icl_mode": "true", "ref_text": "hi"},
            ),
            "float code": save(
                {"ref_spk_embedding": spk, "ref_code": np.zeros((4, 16), np.float32)},
                metadata={"x_vector_only_mode": "false", "icl_mode": "true", "ref_text": "hi"},
            ),
        }
        for name, prompt in cases.items():
            r = self._import(archive(prompt, {"companion_id": "bad"}))
            self.assertEqual(r.status_code, 422, (name, r.text))
        self.assertEqual(self.s3.objects, {})


# ── /clone-voice: meta.json and the upload cap ────────────────────────────────


def wav_bytes(seconds=4.0, sr=24000, tone=True) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    audio = (0.3 * np.sin(2 * np.pi * 220 * t) if tone else np.zeros_like(t)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class CloneVoiceMetaTests(_Base):
    def setUp(self):
        super().setUp()
        self.base = mock.Mock()
        self.base.create_voice_clone_prompt.return_value = [icl_item("the transcript")]
        mgr = TTSModelManager()
        mgr.keep_alive_gpu = mgr.keep_alive_cpu = -1
        mgr._base_model, mgr._state = self.base, TTSState.GPU
        p = mock.patch.object(tts_server, "_mgr", mgr)
        p.start()
        self.addCleanup(p.stop)

    def test_clone_writes_meta_json_and_lists(self):
        r = self.client.post(
            "/clone-voice", files={"audio": ("a.wav", wav_bytes(), "audio/wav")},
            data={"companion_id": "carol", "reference_text": "the transcript", "voice_name": "Carol"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["voice_clone_id"], "voice_clones/carol/prompt.pt")
        self.assertAlmostEqual(body["duration"], 4.0, places=2)
        meta = json.loads(self.s3.objects["voice_clones/carol/meta.json"])
        self.assertEqual((meta["name"], meta["has_transcript"], meta["engine"]), ("Carol", True, "qwen-tts"))
        self.assertEqual([c["name"] for c in self.client.get("/clones").json()["clones"]], ["Carol"])

    def test_oversized_clip_is_413_before_the_model(self):
        with mock.patch.object(tts_server, "CLIP_MAX_BYTES", 1000):
            r = self.client.post("/clone-voice", files={"audio": ("a.wav", wav_bytes(), "audio/wav")},
                                 data={"companion_id": "carol"})
        self.assertEqual(r.status_code, 413)
        self.base.create_voice_clone_prompt.assert_not_called()
        self.assertEqual(self.s3.objects, {})


# ── /analyze-clips ────────────────────────────────────────────────────────────


def speechy(sr=24000, bursts=8, amp=0.3, noise=0.001, freq=220.0, seed=0) -> np.ndarray:
    """0.5 s tone bursts separated by 0.3 s of faint noise."""
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(bursts):
        t = np.arange(int(0.5 * sr)) / sr
        parts.append(amp * np.sin(2 * np.pi * freq * t))
        parts.append(rng.normal(0, noise, int(0.3 * sr)))
    return np.concatenate(parts).astype("float32")


def encode(audio, sr=24000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class ClipMetricsTests(unittest.TestCase):
    def test_speech_with_pauses(self):
        m = tts_server._quick_clip_metrics(speechy(), 24000)
        self.assertAlmostEqual(m["raw_seconds"], 6.4, places=1)
        self.assertAlmostEqual(m["speech_seconds"], 4.0, delta=0.3)
        self.assertGreater(m["snr_db"], 40)
        self.assertLess(m["noise_floor_dbfs"], -50)
        self.assertEqual(m["flags"], [])

    def test_silence_has_no_speech(self):
        m = tts_server._quick_clip_metrics(np.zeros(24000 * 2, dtype="float32"), 24000)
        self.assertEqual((m["speech_seconds"], m["flags"]), (0.0, ["vad_no_speech"]))

    def test_continuous_signal_counts_as_speech_without_snr(self):
        t = np.arange(24000 * 3) / 24000
        m = tts_server._quick_clip_metrics((0.3 * np.sin(2 * np.pi * 200 * t)).astype("float32"), 24000)
        self.assertAlmostEqual(m["speech_seconds"], 3.0, places=1)
        self.assertIsNone(m["snr_db"])

    def test_clipping_is_measured(self):
        audio = np.clip(speechy(amp=3.0), -1, 1)
        m = tts_server._quick_clip_metrics(audio, 24000)
        self.assertGreater(m["clip_ratio"], 0.1)
        self.assertEqual(m["peak"], 1.0)


class AnalyzeClipsTests(unittest.TestCase):
    def setUp(self):
        self.base = mock.Mock()
        self.embeddings = []

        def prompt(audio, x_vector_only_mode):
            self.assertTrue(x_vector_only_mode)
            wav, sr = audio
            self.assertLessEqual(len(wav), tts_server._ANALYZE_EMBED_SECONDS * sr)
            return [types.SimpleNamespace(ref_spk_embedding=np.asarray(self.embeddings.pop(0)))]

        self.base.create_voice_clone_prompt.side_effect = prompt
        self.mgr = TTSModelManager()
        self.mgr.keep_alive_gpu = self.mgr.keep_alive_cpu = -1
        self.mgr._base_model, self.mgr._state = self.base, TTSState.GPU
        p = mock.patch.object(tts_server, "_mgr", self.mgr)
        p.start()
        self.addCleanup(p.stop)
        self.client = TestClient(tts_server.app)

    def _post(self, clips):
        return self.client.post("/analyze-clips", files=[("audio", (n, d, "audio/wav")) for n, d in clips])

    def test_ranks_clips_and_flags_the_odd_one_out(self):
        self.embeddings = [[1, 0, 0], [0.98, 0.2, 0], [0, 0, 1]]  # the third is another voice
        clips = [("good.wav", encode(speechy(seed=1))), ("good2.wav", encode(speechy(seed=2))),
                 ("other.wav", encode(speechy(seed=3, freq=330))), ("silent.wav", encode(np.zeros(48000))),
                 ("junk.wav", b"not audio at all")]
        r = self._post(clips)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual((body["engine"], body["threshold"]), ("qwen-tts", tts_server.ANALYZE_SCORE_THRESHOLD))
        self.assertIn("gpt_cond_len", body)
        by = {c["name"]: c for c in body["clips"]}
        self.assertTrue(by["good.wav"]["recommended"])
        self.assertIn("outlier", by["other.wav"]["flags"])
        self.assertFalse(by["other.wav"]["recommended"])
        self.assertIn("vad_no_speech", by["silent.wav"]["flags"])
        self.assertIsNone(by["silent.wav"]["consistency"])
        self.assertTrue(any(f.startswith("unreadable") for f in by["junk.wav"]["flags"]))
        self.assertEqual(by["junk.wav"]["score"], 0)
        self.assertEqual(sorted(body["order"]), list(range(5)))
        self.assertIn(body["order"][0], (0, 1))
        self.assertEqual(body["order"][-1], 4)  # unreadable sorts last
        self.assertEqual([body["clips"][i]["rank"] for i in body["order"]], list(range(5)))
        self.assertEqual(self.base.create_voice_clone_prompt.call_count, 3)  # not for silence or junk
        self.assertEqual(self.mgr._in_use, 0)

    def test_limits(self):
        with mock.patch.object(tts_server, "MAX_CLIPS", 2):
            self.assertEqual(self._post([(f"{i}.wav", b"x") for i in range(3)]).status_code, 413)
        with mock.patch.object(tts_server, "CLIP_MAX_BYTES", 10):
            self.assertEqual(self._post([("a.wav", encode(speechy()))]).status_code, 413)
        self.assertEqual(self.client.post("/analyze-clips").status_code, 422)
        self.base.create_voice_clone_prompt.assert_not_called()

    def test_needs_the_base_model(self):
        self.mgr._base_model = None
        r = self._post([("a.wav", encode(speechy()))])
        self.assertEqual(r.status_code, 503)
        self.assertEqual(self.mgr._in_use, 0)

    def test_embedding_oom_is_503(self):
        self.base.create_voice_clone_prompt.side_effect = RuntimeError("CUDA out of memory")
        with mock.patch.object(tts_server, "_gpu_memory_info", return_value=None):
            r = self._post([("a.wav", encode(speechy()))])
        self.assertEqual(r.status_code, 503)
        self.assertEqual(self.mgr._in_use, 0)

    def test_other_embedding_failure_is_a_flag(self):
        self.base.create_voice_clone_prompt.side_effect = ValueError("bad audio")
        r = self._post([("a.wav", encode(speechy()))])
        self.assertEqual(r.status_code, 200)
        clip = r.json()["clips"][0]
        self.assertIn("embed_failed:bad audio", clip["flags"])
        self.assertIsNone(clip["consistency"])


class ConcurrencyTests(_Base):
    def test_concurrent_imports_of_one_id_are_serialised(self):
        # import holds the id's clone lock across its existence check and upload,
        # like /clone-voice: two replace=false imports to one id give 200 and 409.
        data = archive(packed([xvec_item()]), {"companion_id": "dup"})

        async def run():
            import httpx

            transport = httpx.ASGITransport(app=tts_server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await asyncio.gather(*(
                    c.post("/clones/import", files={"archive": ("a.zip", data, "application/zip")})
                    for _ in range(2)
                ))

        real_exists = tts_server._s3_object_exists

        def slow_exists(key):
            import time

            found = real_exists(key)
            time.sleep(0.05)  # the window in which another import could slip in
            return found

        with mock.patch.object(tts_server, "_s3_object_exists", slow_exists):
            codes = sorted(r.status_code for r in asyncio.run(run()))
        self.assertEqual(codes, [200, 409])


if __name__ == "__main__":
    unittest.main()
