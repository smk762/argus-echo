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
    ],
)
def test_bad_prompts_are_422_and_store_nothing(env, prompt, needle):
    r = _import(env, _zip({"prompt.safetensors": prompt}))
    assert r.status_code == 422, r.text
    assert needle in r.json()["error"]
    assert env.s3.objects == {}


def test_an_archive_with_no_prompt_is_422(env):
    r = _import(env, _zip({"meta.json": "{}"}))
    assert r.status_code == 422
    assert "prompt.safetensors" in r.json()["error"]
