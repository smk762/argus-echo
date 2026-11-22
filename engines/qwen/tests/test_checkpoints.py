"""Fine-tuned checkpoints (#10): POST /admin/load, the weights /health reports, the image identity, and ``seed``.

torch and qwen_tts are faked; the manager's real load code runs on a fake Qwen3TTSModel, and the real hashing runs
on files in a temporary directory.
"""

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.device = lambda d: d
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules["torch"] = _torch

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import tts_server  # noqa: E402
from tts_server import TTSModelManager, TTSState  # noqa: E402

TOKEN = "s3cret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_model(path: Path, *, tts_model_type="custom_voice", weights=b"weights", model_type="qwen3_tts") -> Path:
    (path / "speech_tokenizer").mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": model_type, "tts_model_type": tts_model_type}))
    (path / "generation_config.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(weights)
    (path / "speech_tokenizer" / "config.json").write_text("{}")
    (path / "speech_tokenizer" / "model.safetensors").write_bytes(b"codec")
    (path / ".cache").mkdir()  # skipped: not a file the load reads
    return path


class FakeModel:
    """What Qwen3TTSModel.from_pretrained returns; generation is recorded by the test's torch."""

    def __init__(self, source, calls):
        self.source = source
        self.calls = calls

    def get_supported_speakers(self):
        return ["ryan"]

    def generate_custom_voice(self, **kw):
        self.calls.append(("generate", kw["text"]))
        return [np.zeros(10, dtype="<f4")], 24000


class CheckpointTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.root = self.tmp / "ckpt"
        self.root.mkdir()
        self.stock = {
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base": _write_model(self.tmp / "stock-base", tts_model_type="base",
                                                          weights=b"stock base"),
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": _write_model(self.tmp / "stock-cv", weights=b"stock cv"),
        }
        self.calls = []  # torch.manual_seed and generate calls, in order
        self.loads = []  # what from_pretrained was given

        torch = types.ModuleType("torch")
        torch.device = lambda d: d
        torch.bfloat16 = "bf16"
        torch.manual_seed = lambda seed: self.calls.append(("manual_seed", seed))
        torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)

        def from_pretrained(source, **kw):
            if "broken" in source:
                raise RuntimeError("Error(s) in loading state_dict")
            self.loads.append(source)
            return FakeModel(source, self.calls)

        qwen_tts = types.ModuleType("qwen_tts")
        qwen_tts.Qwen3TTSModel = types.SimpleNamespace(from_pretrained=from_pretrained)

        self.mgr = TTSModelManager()
        self.mgr.keep_alive_gpu = self.mgr.keep_alive_cpu = -1
        tts_server._hash_cache.clear()
        for patch in (
            mock.patch.dict(sys.modules, {"torch": torch, "qwen_tts": qwen_tts}),
            mock.patch.object(tts_server, "_mgr", self.mgr),
            mock.patch.object(tts_server, "CHECKPOINT_ROOT", str(self.root)),
            mock.patch.object(tts_server, "MODEL_SIZE", "0.6b"),
            mock.patch.object(tts_server, "LOAD_MODELS", "both"),
            mock.patch.object(tts_server, "DEFAULT_SPEAKER", ""),
            mock.patch.object(tts_server, "_stock_dir", lambda model_id: self.stock[model_id]),
            mock.patch.object(tts_server, "_upload_to_s3", lambda data, key: f"http://s3/{key}"),
            mock.patch.dict("os.environ", {"ARGUS_ADMIN_TOKEN": TOKEN, "ARGUS_ADMIN_INSECURE": ""}),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.client = TestClient(tts_server.app, raise_server_exceptions=False)

    def load(self, name, **kw):
        return self.client.post("/admin/load", json={"checkpoint": name}, headers=AUTH, **kw)

    def lease_once(self):
        async def run():
            async with self.mgr.use():
                pass

        asyncio.run(run())


class AdminLoadTests(CheckpointTestCase):
    def test_closed_without_a_token_and_needs_the_right_one(self):
        _write_model(self.root / "voice")
        with mock.patch.dict("os.environ", {"ARGUS_ADMIN_TOKEN": ""}):
            self.assertEqual(self.client.post("/admin/load", json={"checkpoint": "voice"}).status_code, 503)
        self.assertEqual(self.client.post("/admin/load", json={"checkpoint": "voice"}).status_code, 401)
        bad = {"Authorization": "Bearer wrong"}
        self.assertEqual(self.client.post("/admin/load", json={"checkpoint": "voice"}, headers=bad).status_code, 401)
        self.assertEqual(self.loads, [])

    def test_names_that_escape_the_root_are_refused(self):
        (self.root / "sneaky").symlink_to(self.stock["Qwen/Qwen3-TTS-12Hz-0.6B-Base"])
        linked = _write_model(self.root / "linked")
        (linked / "model.safetensors").unlink()
        (linked / "model.safetensors").symlink_to(self.tmp / "stock-cv" / "model.safetensors")
        for name in ("../stock-cv", "/etc", "a/../../stock-cv", "", "sneaky", "linked"):
            r = self.load(name)
            self.assertEqual(r.status_code, 400, (name, r.text))
        self.assertEqual(self.loads, [])
        self.assertEqual(self.mgr.state, TTSState.UNLOADED)

    def test_missing_and_wrong_format_checkpoints(self):
        self.assertEqual(self.load("nope").status_code, 404)
        (self.root / "file").write_text("x")
        self.assertEqual(self.load("file").status_code, 422)
        partial = self.root / "partial"
        partial.mkdir()
        (partial / "config.json").write_text("{}")
        r = self.load("partial")
        self.assertEqual(r.status_code, 422)
        self.assertIn("missing generation_config.json", r.json()["error"])
        self.assertIn("model.safetensors or model.safetensors.index.json", r.json()["error"])
        _write_model(self.root / "design", tts_model_type="voice_design")
        r = self.load("design")
        self.assertEqual(r.status_code, 422)
        self.assertIn("voice_design", r.json()["error"])
        _write_model(self.root / "xtts", model_type="xtts")
        self.assertEqual(self.load("xtts").status_code, 422)
        with mock.patch.object(tts_server, "CHECKPOINT_ROOT", ""):
            r = self.load("partial")
        self.assertEqual(r.status_code, 503)
        self.assertIn("TTS_CHECKPOINT_ROOT", r.json()["error"])
        self.assertEqual(self.loads, [])

    def test_load_fills_its_slot_and_reports_every_file_then_unload_returns_to_stock(self):
        ckpt = _write_model(self.root / "forge" / "run-7", weights=b"fine-tuned")
        r = self.load("forge/run-7")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["slot"], "custom_voice")
        self.assertEqual(self.loads, ["Qwen/Qwen3-TTS-12Hz-0.6B-Base", str(ckpt.resolve())])
        self.assertEqual(self.mgr.custom_model.source, str(ckpt.resolve()))
        weights = body["weights"]
        self.assertEqual(weights["checkpoint"], "forge/run-7")
        files = {f["name"]: f["sha256"] for f in weights["files"]}
        self.assertEqual(files["custom_voice/model.safetensors"], _sha(b"fine-tuned"))
        self.assertEqual(files["custom_voice/speech_tokenizer/model.safetensors"], _sha(b"codec"))
        self.assertEqual(files["base/model.safetensors"], _sha(b"stock base"))  # the other slot is pinned too
        self.assertEqual(len(files), 12)  # 6 files a slot; dotfiles skipped
        health = self.client.get("/health").json()
        self.assertEqual(health["weights"], weights)
        self.assertEqual(health["model_state"], "gpu")

        r = self.client.post("/admin/unload", headers=AUTH)
        self.assertEqual(r.json()["weights"], {"checkpoint": None, "files": None})
        self.loads.clear()
        self.lease_once()
        self.assertEqual(self.loads, list(self.stock))  # the next load is stock
        files = {f["name"]: f["sha256"] for f in self.mgr.weights["files"]}
        self.assertEqual(files["custom_voice/model.safetensors"], _sha(b"stock cv"))

    def test_a_base_checkpoint_loads_even_when_only_custom_voice_is_configured(self):
        ckpt = _write_model(self.root / "clone-ft", tts_model_type="base")
        with mock.patch.object(tts_server, "LOAD_MODELS", "custom_voice"):
            self.assertEqual(self.load("clone-ft").status_code, 200)
        self.assertEqual(self.loads, [str(ckpt.resolve()), "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"])

    def test_null_loads_the_stock_models(self):
        r = self.load(None)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIsNone(r.json()["weights"]["checkpoint"])
        self.assertEqual(self.loads, list(self.stock))

    def test_an_idle_unload_keeps_the_checkpoint(self):
        _write_model(self.root / "voice")
        self.assertEqual(self.load("voice").status_code, 200)
        asyncio.run(self.mgr.unload())  # what the eviction loop's path amounts to
        self.lease_once()
        self.assertTrue(self.loads[-1].endswith("voice"))
        self.assertEqual(self.mgr.weights["checkpoint"], "voice")

    def test_load_while_busy_is_a_409_and_changes_nothing(self):
        _write_model(self.root / "voice")
        asyncio.run(self.mgr.acquire())
        try:
            r = self.load("voice")
            self.assertEqual(r.status_code, 409)
            self.assertEqual(r.json()["status"], "busy")
            self.assertEqual(self.loads, list(self.stock))  # only the lease's own load
            self.assertIsNone(self.mgr.weights["checkpoint"])
        finally:

            async def release():
                self.mgr.release()

            asyncio.run(release())

    def test_a_failed_load_frees_everything_and_selects_stock(self):
        _write_model(self.root / "broken")
        r = self.load("broken")
        self.assertEqual(r.status_code, 500)
        self.assertIn("loading state_dict", r.json()["error"])
        self.assertIsNone(self.mgr.base_model)  # loaded before the failure, then freed
        self.assertIsNone(self.mgr.custom_model)
        self.assertEqual(self.mgr.state, TTSState.ERROR)
        self.assertEqual(self.mgr.weights, {"checkpoint": None, "files": None})

    def test_hashes_are_computed_once_per_load_not_per_request(self):
        _write_model(self.root / "voice")
        opened = []
        real = hashlib.sha256
        with mock.patch.object(tts_server.hashlib, "sha256", lambda *a: opened.append(1) or real(*a)):
            self.assertEqual(self.load("voice").status_code, 200)
            self.assertEqual(len(opened), 12)
            for _ in range(3):
                self.assertTrue(self.client.get("/health").json()["weights"]["files"])
            r = self.client.post("/synthesise", json={"text": "hello", "speaker": "ryan"})
            self.assertEqual(r.status_code, 200, r.text)
            asyncio.run(self.mgr.unload())
            self.lease_once()  # reloads the same, unchanged files
        self.assertEqual(len(opened), 12)

    def test_health_reports_the_image_identity(self):
        self.assertEqual(self.client.get("/health").json()["image"], {"version": None, "revision": None})
        with mock.patch.object(tts_server, "ECHO_VERSION", "0.2.0"), mock.patch.object(tts_server, "ECHO_REVISION", "abc"):
            self.assertEqual(self.client.get("/health").json()["image"], {"version": "0.2.0", "revision": "abc"})


class SeedTests(CheckpointTestCase):
    def test_synthesise_seeds_every_rng_before_sampling(self):
        seeded = []
        with mock.patch.object(tts_server.random, "seed", lambda s: seeded.append(("random", s))), \
                mock.patch.object(np.random, "seed", lambda s: seeded.append(("numpy", s))):
            r = self.client.post("/synthesise", json={"text": "hello there", "speaker": "ryan", "seed": 42})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["seed"], 42)
        self.assertEqual(self.calls, [("manual_seed", 42), ("generate", "hello there")])
        self.assertEqual(seeded, [("random", 42), ("numpy", 42)])

        self.calls.clear()
        self.client.post("/synthesise", json={"text": "again", "speaker": "ryan", "seed": 7})
        self.client.post("/synthesise", json={"text": "unseeded", "speaker": "ryan"})
        self.assertEqual(self.calls, [("manual_seed", 7), ("generate", "again"), ("generate", "unseeded")])

    def test_stream_seeds_each_sentence_from_the_seed(self):
        s1, s2, s3 = (
            "The first sentence is long enough.",
            "Here is the second one, also long.",
            "And a third sentence ends the text.",
        )
        r = self.client.post("/synthesise/stream", json={"text": f"{s1} {s2} {s3}", "speaker": "ryan", "seed": 100})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.content), 3 * 10 * 4)
        self.assertEqual(
            self.calls,
            [("manual_seed", 100), ("generate", s1), ("manual_seed", 101), ("generate", s2),
             ("manual_seed", 102), ("generate", s3)],
        )

    def test_out_of_range_seeds_are_refused(self):
        for seed in (-1, 2**32, "x", 1.5):
            self.assertEqual(self.client.post("/synthesise", json={"text": "hi", "seed": seed}).status_code, 422)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.loads, [])

    def test_seed_wraps_at_the_top_of_its_range(self):
        render = tts_server._seeded(lambda text: text, tts_server.SEED_MAX)
        render("a", 1)
        self.assertEqual(self.calls, [("manual_seed", 0)])


if __name__ == "__main__":
    unittest.main()
