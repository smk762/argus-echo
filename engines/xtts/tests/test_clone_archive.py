"""Clone archives (download and /clones/import) carry prompt.safetensors, never a pickle.

Runs without torch: the archive is real safetensors (safetensors.numpy) and its parsing is real, while the two
torch conversions (_to_array, _from_array) and the store's torch.save are stand-ins, so numpy arrays play the
latents. tests/test_xtts_server.py repeats the round trip with real torch where it is installed.
"""

import io
import json
import os
import pickle
import struct
import sys
import threading
import types
import zipfile

import numpy as np
import pytest
import xtts_server as xs
from fastapi.testclient import TestClient
from safetensors.numpy import save

GPT = np.random.default_rng(0).standard_normal((1, 32, 1024)).astype(np.float32)
SPK = np.random.default_rng(1).standard_normal((1, 512, 1)).astype(np.float32)


class _S3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def head_object(self, Bucket, Key):
        from botocore.exceptions import ClientError

        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}

    def download_fileobj(self, Bucket, Key, buf):
        from botocore.exceptions import ClientError

        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "GetObject")
        buf.write(self.objects[Key])


@pytest.fixture
def env(monkeypatch):
    s3 = _S3()
    torch = types.ModuleType("torch")
    torch.save = lambda obj, buf: buf.write(b"STORED" + pickle.dumps(obj))  # what the store would hold

    def no_unpickling(*args, **kwargs):
        raise AssertionError("an uploaded archive reached an unpickler")

    torch.load = no_unpickling
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(xs, "_get_s3_client", lambda: s3)
    monkeypatch.setattr(xs, "_upload_to_s3", lambda data, key, ct="": s3.objects.__setitem__(key, data))
    monkeypatch.setattr(xs, "_to_array", lambda t: t)
    monkeypatch.setattr(xs, "_from_array", lambda a: a)
    monkeypatch.setattr(xs, "_clone_cache", xs.OrderedDict())
    # A stored prompt.pt is this server's own: the download path loads it with _load_prompt (torch.load, real
    # torch only). Here a stored clone is the pickled dict the fake torch.save wrote.
    monkeypatch.setattr(xs, "_load_prompt", lambda data: pickle.loads(data.removeprefix(b"STORED")))
    client = TestClient(xs.app, raise_server_exceptions=False)
    return types.SimpleNamespace(s3=s3, client=client)


def _store(env, companion_id, meta=None):
    env.s3.objects[f"voice_clones/{companion_id}/prompt.pt"] = b"STORED" + pickle.dumps(
        {"gpt_cond_latent": GPT, "speaker_embedding": SPK}
    )
    if meta is not None:
        env.s3.objects[f"voice_clones/{companion_id}/meta.json"] = json.dumps(meta).encode()


def _zip(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _prompt(arrays=None, metadata=None) -> bytes:
    arrays = {"gpt_cond_latent": GPT, "speaker_embedding": SPK} if arrays is None else arrays
    meta = {"format": "xtts-clone", "format_version": "1", "engine": "xtts"} if metadata is None else metadata
    return save(arrays, metadata=meta)


def _import(env, data, **form):
    return env.client.post("/clones/import", files={"archive": ("a.zip", data, "application/zip")}, data=form)


def test_download_carries_prompt_safetensors_and_meta(env):
    _store(env, "alice", {"companion_id": "alice", "name": "Alice"})
    r = env.client.get("/clones/alice/download")
    assert r.status_code == 200, r.text
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(zf.namelist()) == ["meta.json", "prompt.safetensors"]
    data = zf.read("prompt.safetensors")
    assert xs._safetensors_metadata(data) == {"format": "xtts-clone", "format_version": "1", "engine": "xtts"}


def test_round_trip_through_download(env):
    _store(env, "alice", {"companion_id": "alice", "name": "Alice", "total_seconds": 7.5})
    archive = env.client.get("/clones/alice/download").content
    r = _import(env, archive, companion_id="alice-copy")
    assert r.status_code == 200, r.text
    assert (r.json()["name"], r.json()["duration"]) == ("Alice", 7.5)
    stored = pickle.loads(env.s3.objects["voice_clones/alice-copy/prompt.pt"].removeprefix(b"STORED"))
    assert np.array_equal(stored["gpt_cond_latent"], GPT) and np.array_equal(stored["speaker_embedding"], SPK)


def test_a_legacy_prompt_pt_is_refused_unread(env):
    """GHSA-63cw-57p8-fm3p: torch.load(weights_only=True) on an upload can run code on torch < 2.10."""
    marker = "/tmp/xtts_archive_pwned_marker"
    if os.path.exists(marker):
        os.unlink(marker)

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    r = _import(env, _zip({"prompt.pt": pickle.dumps(Evil()), "meta.json": json.dumps({"name": "x"})}))
    assert r.status_code == 422
    assert "legacy prompt.pt" in r.json()["error"]
    assert not os.path.exists(marker)
    assert env.s3.objects == {}


@pytest.mark.parametrize(
    "prompt, needle",
    [
        (b"\x05\x00", "truncated"),
        (struct.pack("<Q", 10**6) + b"{}", "malformed header"),
        (struct.pack("<Q", 2) + b"[]", "malformed header"),
        (_prompt(metadata={"engine": "miso"}), "'miso' clone"),
        (_prompt(arrays={"ref_audio": np.zeros(16000, np.float32)}, metadata={"engine": "xtts"}), "gpt_cond_latent"),
        (_prompt(arrays={"gpt_cond_latent": GPT}), "gpt_cond_latent"),
        (_prompt(arrays={"gpt_cond_latent": GPT.astype(np.int32), "speaker_embedding": SPK}), "float"),
        (_prompt(arrays={"gpt_cond_latent": np.zeros((0,), np.float32), "speaker_embedding": SPK}), "non-empty"),
        (_prompt(arrays={"gpt_cond_latent": np.full((2,), np.nan, np.float32), "speaker_embedding": SPK}), "finite"),
        (_prompt(arrays={"gpt_cond_latent": GPT, "speaker_embedding": SPK.reshape(512)}), "speaker_embedding"),
        (_prompt(arrays={"gpt_cond_latent": GPT, "speaker_embedding": np.ones((1, 256, 1), np.float32)}), "(1, 512, 1)"),
        (_prompt(arrays={"gpt_cond_latent": GPT[0], "speaker_embedding": SPK}), "gpt_cond_latent"),
        (_prompt(arrays={"gpt_cond_latent": GPT[..., :512].copy(), "speaker_embedding": SPK}), "(1, T, 1024)"),
        (_prompt(arrays={"gpt_cond_latent": np.ones((2, 32, 1024), np.float32), "speaker_embedding": SPK}), "(1, T"),
        (_prompt(arrays={"gpt_cond_latent": np.ones((1, 1025, 1024), np.float32), "speaker_embedding": SPK}), "at most"),
    ],
)
def test_bad_prompts_are_422_and_store_nothing(env, prompt, needle):
    r = _import(env, _zip({"prompt.safetensors": prompt}))
    assert r.status_code == 422, r.text
    assert needle in r.json()["error"]
    assert env.s3.objects == {}


def test_a_replaced_clone_is_the_newest_in_the_cache(env, monkeypatch):
    monkeypatch.setattr(xs, "_CLONE_CACHE_MAX", 2)
    archive = _zip({"prompt.safetensors": _prompt()})
    for cid in ("alice", "bob"):
        assert _import(env, archive, companion_id=cid).status_code == 200
    assert _import(env, archive, companion_id="alice", replace="true").status_code == 200
    assert _import(env, archive, companion_id="carol").status_code == 200
    assert list(xs._clone_cache) == ["voice_clones/alice/prompt.pt", "voice_clones/carol/prompt.pt"]


def test_an_archive_with_no_prompt_is_422(env):
    r = _import(env, _zip({"meta.json": "{}"}))
    assert r.status_code == 422
    assert "prompt.safetensors" in r.json()["error"]


@pytest.mark.parametrize(
    "bad",
    [
        {"name": ["x"], "num_clips": "3", "total_seconds": "long"},
        {"name": {"a": 1}, "num_clips": -2, "total_seconds": -1.5},
        {"name": 7, "num_clips": True, "total_seconds": True},
        {"name": None, "num_clips": 2.5, "total_seconds": float("nan")},
        {"name": "\ud800", "num_clips": [], "total_seconds": float("inf")},
        {"name": " ", "num_clips": 10**400, "total_seconds": 10**400},
    ],
)
def test_sidecar_fields_of_the_wrong_type_are_not_stored(env, bad):
    r = _import(env, _zip({"prompt.safetensors": _prompt(), "meta.json": json.dumps(bad)}), companion_id="bob")
    assert r.status_code == 200, r.text
    stored = json.loads(env.s3.objects["voice_clones/bob/meta.json"])
    for meta in (r.json(), stored):
        assert (meta["name"], meta["num_clips"], meta["total_seconds"]) == ("bob", 0, 0)
    assert r.json()["duration"] == 0


def test_valid_sidecar_fields_are_kept(env):
    meta = {"name": " Bob ", "num_clips": 3, "total_seconds": 7}
    r = _import(env, _zip({"prompt.safetensors": _prompt(), "meta.json": json.dumps(meta)}), companion_id="bob")
    assert (r.json()["name"], r.json()["num_clips"], r.json()["duration"]) == ("Bob", 3, 7.0)


def test_deeply_nested_json_is_not_a_server_error(env):
    nested = b"[" * 50_000
    r = _import(env, _zip({"prompt.safetensors": _prompt(), "meta.json": nested}), companion_id="bob")
    assert r.status_code == 200, r.text
    prompt = len(nested).to_bytes(8, "little") + nested
    assert _import(env, _zip({"prompt.safetensors": prompt}), companion_id="al").status_code == 422


@pytest.mark.parametrize("replace", [False, True])
def test_concurrent_imports_of_one_id_do_not_interleave(env, monkeypatch, replace):
    """The first import pauses inside its prompt upload until the second has had its chance to run."""
    first_uploading, release = threading.Event(), threading.Event()
    uploads = []

    def upload(data, key, ct=""):
        uploads.append((threading.current_thread().name, key.rsplit("/", 1)[-1]))
        if len(uploads) == 1:
            first_uploading.set()
            release.wait(10)
        env.s3.objects[key] = data

    monkeypatch.setattr(xs, "_upload_to_s3", upload)
    results = {}

    def run(name):
        try:
            results[name] = xs._import_clone(_zip({"prompt.safetensors": _prompt()}), "bob", name, replace)["name"]
        except xs.CloneExistsError:
            results[name] = "exists"

    a, b = (threading.Thread(target=run, args=(n,), name=n) for n in ("A", "B"))
    a.start()
    assert first_uploading.wait(10)
    b.start()
    b.join(0.5)  # without a lock B finishes here, between A's check and A's writes
    release.set()
    a.join(10)
    b.join(10)
    assert results == {"A": "A", "B": "B" if replace else "exists"}
    owners = ["A", "A", "B", "B"] if replace else ["A", "A"]
    assert uploads == [(n, f) for n, f in zip(owners, ["prompt.pt", "meta.json"] * 2)]
