"""Fine-tuned checkpoints (#10): POST /admin/load, the weights /health reports, the image identity, and ``seed``.

Torch and coqui-tts are faked per test (as in test_cpu_fallback.py); the manager's real load code runs on a fake
``Xtts``, and the real hashing runs on files in tmp_path.
"""

import asyncio
import hashlib
import json
import sys
import threading
import time
import types

import numpy as np
import pytest
import xtts_server as xs
from fastapi.testclient import TestClient

TOKEN = "s3cret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class _Tensor:
    pass


@pytest.fixture
def torch_calls(monkeypatch):
    """A fake torch that records manual_seed."""
    calls: list = []
    torch = types.ModuleType("torch")
    torch.device = lambda d: d
    torch.Tensor = _Tensor
    torch.manual_seed = lambda seed: calls.append(("manual_seed", seed))
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, is_initialized=lambda: False, empty_cache=lambda: None
    )
    torch._C = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


class _FakeXtts:
    """What Xtts.init_from_config returns: records which directory load_checkpoint read."""

    loads: list = []

    @classmethod
    def init_from_config(cls, config):
        return cls()

    def load_checkpoint(self, config, checkpoint_dir=None, checkpoint_path=None, use_deepspeed=False):
        if "broken" in str(checkpoint_dir):
            raise RuntimeError("size mismatch for gpt.text_embedding.weight")
        self.dir = str(checkpoint_dir)
        _FakeXtts.loads.append(self.dir)

    def eval(self):
        return self


def _write_checkpoint(path, *, model=b"weights", speakers=False, config=None):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config if config is not None else {"model": "xtts"}))
    (path / "model.pth").write_bytes(model)
    (path / "vocab.json").write_text("{}")
    if speakers:
        (path / "speakers_xtts.pth").write_bytes(b"speakers")
    return path


@pytest.fixture
def coqui(monkeypatch, tmp_path, torch_calls):
    """Fake coqui-tts modules; the stock model lives in tmp_path/stock and checkpoints under tmp_path/ckpt."""
    stock = _write_checkpoint(tmp_path / "stock", model=b"stock weights", speakers=True)
    root = tmp_path / "ckpt"
    root.mkdir()

    class XttsConfig:
        def load_json(self, path):
            self.path = path

    class ModelManager:
        def download_model(self, name):
            return str(stock), str(stock / "config.json"), None

    mods = {
        "TTS": types.ModuleType("TTS"),
        "TTS.tts": types.ModuleType("TTS.tts"),
        "TTS.tts.configs": types.ModuleType("TTS.tts.configs"),
        "TTS.tts.configs.xtts_config": types.SimpleNamespace(XttsConfig=XttsConfig),
        "TTS.tts.models": types.ModuleType("TTS.tts.models"),
        "TTS.tts.models.xtts": types.SimpleNamespace(Xtts=_FakeXtts),
        "TTS.utils": types.ModuleType("TTS.utils"),
        "TTS.utils.manage": types.SimpleNamespace(ModelManager=ModelManager),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(xs, "CHECKPOINT_ROOT", str(root))
    monkeypatch.setattr(xs, "DEVICE", "cpu")
    _FakeXtts.loads = []
    xs._hash_cache.clear()
    return types.SimpleNamespace(stock=stock, root=root)


@pytest.fixture
def mgr(monkeypatch, coqui):
    m = xs.XTTSModelManager()
    m.keep_alive_gpu = m.keep_alive_cpu = -1
    monkeypatch.setattr(xs, "_mgr", m)
    return m


@pytest.fixture
def client(monkeypatch, mgr):
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(xs, "_gpu_fault", None, raising=False)
    monkeypatch.setattr(xs, "_gpu_memory_info", lambda: None)
    return TestClient(xs.app, raise_server_exceptions=False)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── the admin route ──────────────────────────────────────────────────────────


def test_admin_load_is_closed_without_a_token(client, monkeypatch, coqui):
    _write_checkpoint(coqui.root / "voice")
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", "")
    monkeypatch.setenv("ARGUS_ADMIN_INSECURE", "")
    assert client.post("/admin/load", json={"checkpoint": "voice"}).status_code == 503
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", TOKEN)
    assert client.post("/admin/load", json={"checkpoint": "voice"}).status_code == 401
    bad = {"Authorization": "Bearer wrong"}
    assert client.post("/admin/load", json={"checkpoint": "voice"}, headers=bad).status_code == 401
    assert _FakeXtts.loads == []


@pytest.mark.parametrize("name", ["../stock", "/etc", "a/../../stock", "", "a\\b"])
def test_names_that_escape_the_root_are_refused(client, mgr, name):
    r = client.post("/admin/load", json={"checkpoint": name}, headers=AUTH)
    assert r.status_code == 400, r.text
    assert "checkpoint" in r.json()["error"]
    assert _FakeXtts.loads == [] and mgr.state == xs.TTSState.UNLOADED


def test_a_symlink_out_of_the_root_is_refused(client, coqui):
    (coqui.root / "sneaky").symlink_to(coqui.stock)
    r = client.post("/admin/load", json={"checkpoint": "sneaky"}, headers=AUTH)
    assert r.status_code == 400, r.text
    linked = _write_checkpoint(coqui.root / "linked")
    (linked / "model.pth").unlink()
    (linked / "model.pth").symlink_to(coqui.stock / "model.pth")
    r = client.post("/admin/load", json={"checkpoint": "linked"}, headers=AUTH)
    assert r.status_code == 400 and "links outside" in r.json()["error"]
    assert _FakeXtts.loads == []


def test_missing_and_wrong_format_checkpoints(client, coqui, monkeypatch):
    assert client.post("/admin/load", json={"checkpoint": "nope"}, headers=AUTH).status_code == 404
    (coqui.root / "file").write_text("x")
    assert client.post("/admin/load", json={"checkpoint": "file"}, headers=AUTH).status_code == 422
    partial = coqui.root / "partial"
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    r = client.post("/admin/load", json={"checkpoint": "partial"}, headers=AUTH)
    assert r.status_code == 422 and "missing model.pth, vocab.json" in r.json()["error"]
    _write_checkpoint(coqui.root / "qwen", config={"model_type": "qwen3_tts"})
    r = client.post("/admin/load", json={"checkpoint": "qwen"}, headers=AUTH)
    assert r.status_code == 422 and "not an XTTS config" in r.json()["error"]
    assert client.post("/admin/load", json={}, headers=AUTH).status_code == 422
    monkeypatch.setattr(xs, "CHECKPOINT_ROOT", "")
    r = client.post("/admin/load", json={"checkpoint": "partial"}, headers=AUTH)
    assert r.status_code == 503 and "XTTS_CHECKPOINT_ROOT" in r.json()["error"]
    assert _FakeXtts.loads == []


def test_load_reports_the_checkpoint_and_its_hashes_then_unload_returns_to_stock(client, coqui, mgr):
    ckpt = _write_checkpoint(coqui.root / "forge" / "run-7", model=b"fine-tuned")
    r = client.post("/admin/load", json={"checkpoint": "forge/run-7"}, headers=AUTH)
    assert r.status_code == 200, r.text
    assert _FakeXtts.loads == [str(ckpt.resolve())]
    weights = r.json()["weights"]
    assert weights["checkpoint"] == "forge/run-7"
    assert weights["files"] == [
        {"name": "config.json", "sha256": _sha(b'{"model": "xtts"}'), "bytes": 17},
        {"name": "model.pth", "sha256": _sha(b"fine-tuned"), "bytes": 10},
        {"name": "vocab.json", "sha256": _sha(b"{}"), "bytes": 2},
    ]
    health = client.get("/health").json()
    assert health["weights"] == weights and health["model_state"] == "cpu"

    r = client.post("/admin/unload", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["weights"] == {"checkpoint": None, "files": None}

    async def next_request():
        async with mgr.lease():
            pass

    asyncio.run(next_request())
    assert _FakeXtts.loads[-1] == str(coqui.stock)  # the next load is stock
    files = {f["name"]: f["sha256"] for f in mgr.weights["files"]}
    assert files["model.pth"] == _sha(b"stock weights")
    assert files["speakers_xtts.pth"] == _sha(b"speakers")
    assert mgr.weights["checkpoint"] is None


def test_null_loads_the_stock_weights(client, coqui):
    r = client.post("/admin/load", json={"checkpoint": None}, headers=AUTH)
    assert r.status_code == 200, r.text
    assert r.json()["weights"]["checkpoint"] is None
    assert _FakeXtts.loads == [str(coqui.stock)]


def test_an_idle_unload_keeps_the_checkpoint(client, coqui, mgr):
    _write_checkpoint(coqui.root / "voice")
    assert client.post("/admin/load", json={"checkpoint": "voice"}, headers=AUTH).status_code == 200

    async def idle_then_request():
        assert await mgr.unload()  # what the eviction loop calls
        async with mgr.lease():
            pass

    asyncio.run(idle_then_request())
    assert _FakeXtts.loads[-1].endswith("voice")
    assert mgr.weights["checkpoint"] == "voice"


def test_load_while_busy_is_a_409_and_changes_nothing(client, coqui, mgr):
    _write_checkpoint(coqui.root / "voice")
    asyncio.run(mgr.acquire())  # a request holds the model
    try:
        r = client.post("/admin/load", json={"checkpoint": "voice"}, headers=AUTH)
        assert r.status_code == 409 and r.json()["status"] == "busy"
        assert _FakeXtts.loads == [str(coqui.stock)]  # only the lease's own load
        assert mgr.weights["checkpoint"] is None
    finally:

        async def release():
            mgr.release()

        asyncio.run(release())


def test_a_failed_load_leaves_nothing_loaded_and_stock_selected(client, coqui, mgr):
    _write_checkpoint(coqui.root / "broken")
    r = client.post("/admin/load", json={"checkpoint": "broken"}, headers=AUTH)
    assert r.status_code == 500 and "size mismatch" in r.json()["error"]
    assert mgr.model is None and mgr.state == xs.TTSState.ERROR
    assert mgr.weights == {"checkpoint": None, "files": None}


def test_a_load_waits_for_a_transition_in_flight(coqui, mgr):
    _write_checkpoint(coqui.root / "voice")

    async def run():
        first = asyncio.create_task(mgr.ensure_gpu())  # a lazy stock load
        await asyncio.sleep(0)
        assert await mgr.load_weights(("voice", (coqui.root / "voice").resolve()))
        await first
        assert mgr.weights["checkpoint"] == "voice"

    asyncio.run(run())
    assert len(_FakeXtts.loads) == 2 and _FakeXtts.loads[-1].endswith("voice")


def test_hashes_are_computed_once_per_load_not_per_request(client, coqui, mgr, monkeypatch):
    _write_checkpoint(coqui.root / "voice")
    reads = []
    real = xs._sha256
    monkeypatch.setattr(xs, "_sha256", lambda p: reads.append(p.name) or real(p))
    opened = []
    real_new = hashlib.sha256
    monkeypatch.setattr(xs.hashlib, "sha256", lambda *a: opened.append(1) or real_new(*a))
    assert client.post("/admin/load", json={"checkpoint": "voice"}, headers=AUTH).status_code == 200
    assert sorted(reads) == ["config.json", "model.pth", "vocab.json"]
    for _ in range(3):
        assert client.get("/health").json()["weights"]["files"]
    assert len(reads) == 3  # /health reads what the load stored

    async def reload():
        assert await mgr.unload()
        async with mgr.lease():
            pass

    asyncio.run(reload())
    assert len(opened) == 3  # unchanged files are not read again


# ── image identity ───────────────────────────────────────────────────────────


def test_health_reports_the_image_identity(client, monkeypatch):
    assert client.get("/health").json()["image"] == {"version": None, "revision": None}
    monkeypatch.setattr(xs, "ECHO_VERSION", "0.2.0")
    monkeypatch.setattr(xs, "ECHO_REVISION", "abc123")
    assert client.get("/health").json()["image"] == {"version": "0.2.0", "revision": "abc123"}


# ── seed ─────────────────────────────────────────────────────────────────────


class _Model:
    """Speaks from torch's RNG as the fake records it."""

    def __init__(self, torch_calls):
        self.torch_calls = torch_calls
        self.speaker_manager = types.SimpleNamespace(speakers={"A": {"gpt_cond_latent": 1, "speaker_embedding": 2}})

    def parameters(self):
        return iter([types.SimpleNamespace(device="cpu")])

    def inference(self, *args, **kwargs):
        self.torch_calls.append("inference")
        return {"wav": np.zeros(240, dtype=np.float32)}

    def inference_stream(self, *args, **kwargs):
        self.torch_calls.append("inference_stream")
        yield np.zeros(4, dtype=np.float32)
        self.torch_calls.append("chunk 2")
        yield np.zeros(4, dtype=np.float32)


@pytest.fixture
def speaking(client, mgr, torch_calls, monkeypatch):
    async def load():
        mgr._model = _Model(torch_calls)
        mgr._state = xs.TTSState.CPU

    monkeypatch.setattr(mgr, "_do_load", load)
    monkeypatch.setattr(xs, "_upload_to_s3", lambda data, key: f"http://s3/{key}")
    return client


def test_synthesise_seeds_every_rng_before_sampling(speaking, torch_calls, monkeypatch):
    seeded = []
    monkeypatch.setattr(xs.random, "seed", lambda s: seeded.append(("random", s)))
    monkeypatch.setattr(np.random, "seed", lambda s: seeded.append(("numpy", s)))
    r = speaking.post("/synthesise", json={"text": "hi", "speaker": "A", "seed": 42})
    assert r.status_code == 200, r.text
    assert r.json()["seed"] == 42
    assert torch_calls == [("manual_seed", 42), "inference"]
    assert seeded == [("random", 42), ("numpy", 42)]

    torch_calls.clear()
    assert speaking.post("/synthesise", json={"text": "hi", "speaker": "A", "seed": 7}).status_code == 200
    assert speaking.post("/synthesise", json={"text": "hi", "speaker": "A"}).status_code == 200
    assert torch_calls == [("manual_seed", 7), "inference", "inference"]  # per request; unseeded seeds nothing


def test_stream_seeds_before_its_first_sample(speaking, torch_calls):
    r = speaking.post("/synthesise/stream", json={"text": "hi", "speaker": "A", "seed": 9})
    assert r.status_code == 200, r.text
    assert len(r.content) == 32
    assert torch_calls == [("manual_seed", 9), "inference_stream", "chunk 2"]


@pytest.mark.parametrize("seed", [-1, 2**32, "x", 1.5])
def test_out_of_range_seeds_are_refused(speaking, torch_calls, seed):
    assert speaking.post("/synthesise", json={"text": "hi", "seed": seed}).status_code == 422
    assert torch_calls == []


def test_a_seeded_generation_runs_alone(torch_calls):
    """Nothing else draws from the RNG between a seeded generation's seed and its last sample."""
    gate = xs._RngGate()
    events = []
    shared_in = threading.Event()
    release_shared = threading.Event()

    def unseeded():
        with gate.shared():
            events.append("shared in")
            shared_in.set()
            release_shared.wait(5)
            events.append("shared out")

    def seeded():
        with gate.exclusive():
            events.append("exclusive in")
            time.sleep(0.05)
            events.append("exclusive out")

    t1 = threading.Thread(target=unseeded)
    t1.start()
    shared_in.wait(5)
    t2 = threading.Thread(target=seeded)
    t2.start()
    time.sleep(0.05)
    t3 = threading.Thread(target=unseeded)  # arrives while the seeded one waits: it goes after it
    t3.start()
    time.sleep(0.05)
    assert events == ["shared in"]
    release_shared.set()
    for t in (t1, t2, t3):
        t.join(5)
    assert events == ["shared in", "shared out", "exclusive in", "exclusive out", "shared in", "shared out"]
