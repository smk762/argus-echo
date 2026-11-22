"""No room on the GPU (agent-composer#79/#80): a failed promotion puts the weights back on the CPU, then the request
runs there (XTTS_CPU_FALLBACK) or is a 503 with the reason. ``device: "cpu"`` never promotes, and CPU work in
flight keeps the model where it is.

Torch is faked per test, as in test_gpu_residue.py; the manager's real promote/move code runs on a fake model.
"""

import asyncio
import sys
import types

import numpy as np
import pytest
import xtts_server as xs
from fastapi.testclient import TestClient


class _OutOfMemoryError(RuntimeError):
    pass


class _Tensor:
    pass


@pytest.fixture
def cuda_calls(monkeypatch):
    calls: list = []
    torch = types.ModuleType("torch")
    torch.device = lambda d: d
    torch.Tensor = _Tensor
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        empty_cache=lambda: calls.append("empty_cache"),
        OutOfMemoryError=_OutOfMemoryError,
    )
    torch._C = types.SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: calls.append("clear_cublas"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


class _MovableModel:
    """Records moves; .to() fails like a CUDA OOM while *oom* is set."""

    def __init__(self):
        self.oom = False
        self.calls = []
        self.cached_prefix_emb = None

    def modules(self):
        yield self

    def to(self, device):
        self.calls.append(str(device))
        if self.oom:
            self.cached_prefix_emb = types.SimpleNamespace(is_cuda=True)  # what a half-done move leaves behind
            raise _OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB.")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self


@pytest.fixture
def cpu_mgr(monkeypatch, cuda_calls):
    """A manager whose load installs a model on the CPU; the real promote/move code runs."""
    m = xs.XTTSModelManager()
    model = _MovableModel()

    async def load():
        m._model = model
        m._state = xs.TTSState.CPU

    monkeypatch.setattr(m, "_do_load", load)
    monkeypatch.setattr(xs, "DEVICE", "cuda:0")
    monkeypatch.setattr(xs, "CPU_FALLBACK", True)
    monkeypatch.setattr(xs, "_free_vram_mb", lambda: 1)
    m.keep_alive_gpu = m.keep_alive_cpu = -1
    m.fake = model
    return m


# ── a failed promotion ────────────────────────────────────────────────────────


def test_oom_on_promotion_puts_the_model_back_and_runs_on_cpu(cpu_mgr, cuda_calls):
    cpu_mgr.fake.oom = True

    async def run():
        async with cpu_mgr.lease() as ran_on:
            assert ran_on == "cpu"
            assert cpu_mgr.state == xs.TTSState.CPU
        assert cpu_mgr.fake.calls == ["cuda:0", "cpu"]  # the failed move was undone
        assert cpu_mgr.fake.cached_prefix_emb is None
        assert cuda_calls == ["clear_cublas", "empty_cache"]
        cpu_mgr.fake.oom = False  # room again: the next request promotes
        async with cpu_mgr.lease() as ran_on:
            assert ran_on == "cuda:0" and cpu_mgr.state == xs.TTSState.GPU

    asyncio.run(run())


def test_oom_on_promotion_without_fallback_raises_the_reason(cpu_mgr, monkeypatch):
    monkeypatch.setattr(xs, "CPU_FALLBACK", False)
    cpu_mgr.fake.oom = True

    async def run():
        with pytest.raises(xs.GpuUnavailable, match=r"out of memory moving XTTS onto cuda:0 \(1 MiB free\)"):
            await cpu_mgr.acquire()
        assert cpu_mgr.active == 0
        assert cpu_mgr.state == xs.TTSState.CPU

    asyncio.run(run())


def test_a_refused_promotion_still_lets_the_idle_model_unload(cpu_mgr, monkeypatch):
    monkeypatch.setattr(xs, "CPU_FALLBACK", False)
    cpu_mgr.fake.oom = True
    cpu_mgr.keep_alive_cpu = 0

    async def run():
        with pytest.raises(xs.GpuUnavailable):
            await cpu_mgr.acquire()  # loaded the model, then could not promote it
        await asyncio.sleep(0.05)
        assert cpu_mgr.state == xs.TTSState.UNLOADED

    asyncio.run(run())


def test_a_non_oom_move_failure_is_undone_and_raised(cpu_mgr):
    def broken(device):
        cpu_mgr.fake.calls.append(device)
        raise RuntimeError("CUDA error: device-side assert triggered")

    cpu_mgr.fake.to = broken

    async def run():
        with pytest.raises(RuntimeError, match="device-side assert") as caught:
            await cpu_mgr.acquire()
        assert not isinstance(caught.value, xs.GpuUnavailable)
        assert cpu_mgr.fake.calls[-1] == "cpu"
        assert cpu_mgr.state == xs.TTSState.CPU and cpu_mgr.active == 0

    asyncio.run(run())


def test_wake_without_gpu_room_is_refused_and_leaves_the_model_on_cpu(cpu_mgr):
    cpu_mgr.fake.oom = True

    async def run():
        with pytest.raises(xs.GpuUnavailable):
            await cpu_mgr.ensure_gpu()
        assert cpu_mgr.state == xs.TTSState.CPU and cpu_mgr.fake.calls == ["cuda:0", "cpu"]

    asyncio.run(run())


# ── device "cpu" and CPU work in flight ───────────────────────────────────────


def test_device_cpu_does_not_promote(cpu_mgr):
    async def run():
        async with cpu_mgr.lease("cpu") as ran_on:
            assert ran_on == "cpu"
        assert cpu_mgr.fake.calls == []
        assert cpu_mgr.state == xs.TTSState.CPU

    asyncio.run(run())


def test_device_cpu_uses_a_model_already_on_the_gpu(cpu_mgr):
    async def run():
        async with cpu_mgr.lease():
            pass
        async with cpu_mgr.lease("cpu") as ran_on:
            assert ran_on == "cuda:0"
        assert cpu_mgr.fake.calls == ["cuda:0"]

    asyncio.run(run())


def test_cpu_work_in_flight_keeps_the_model_on_cpu(cpu_mgr):
    async def run():
        async with cpu_mgr.lease("cpu"):
            async with cpu_mgr.lease() as ran_on:
                assert ran_on == "cpu"  # moving it now would break the CPU run
            await cpu_mgr.ensure_gpu()  # nor does /admin/wake move it
            assert cpu_mgr.state == xs.TTSState.CPU
        assert cpu_mgr.fake.calls == []
        async with cpu_mgr.lease() as ran_on:  # the CPU work is done: the next request promotes
            assert ran_on == "cuda:0"

    asyncio.run(run())


def test_a_cpu_only_deployment_reports_cpu(cpu_mgr, monkeypatch):
    monkeypatch.setattr(xs, "DEVICE", "cpu")

    async def run():
        for device in (None, "cpu"):
            async with cpu_mgr.lease(device) as ran_on:
                assert ran_on == "cpu"
        assert cpu_mgr.fake.calls == []

    asyncio.run(run())


# ── the routes ────────────────────────────────────────────────────────────────


class _NoGpuMgr:
    state = xs.TTSState.CPU
    active = 0
    _last_used = 0.0

    async def acquire(self, device=None):
        raise xs.GpuUnavailable("CUDA out of memory moving XTTS onto cuda:0 (1 MiB free)")

    def lease(self, device=None):
        import contextlib

        @contextlib.asynccontextmanager
        async def cm():
            await self.acquire(device)
            yield  # pragma: no cover

        return cm()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(xs, "_gpu_fault", None, raising=False)
    monkeypatch.setattr(xs, "_gpu_memory_info", lambda: None)
    return TestClient(xs.app, raise_server_exceptions=False)


@pytest.mark.parametrize("route", ["/synthesise", "/synthesise/stream"])
def test_synthesis_without_gpu_room_is_a_503_with_the_reason(client, monkeypatch, route):
    monkeypatch.setattr(xs, "_mgr", _NoGpuMgr())
    r = client.post(route, json={"text": "hi", "language": "en"})
    assert r.status_code == 503, r.text
    assert "CUDA out of memory moving XTTS onto cuda:0" in r.json()["error"]
    assert r.headers["retry-after"] == str(xs.OOM_RETRY_AFTER_S)


class _CpuMgr:
    """Records the device each lease asked for and reports it ran on the CPU."""

    state = xs.TTSState.CPU
    _last_used = 0.0

    def __init__(self, model=None):
        self.seen = []
        self.active = 0
        self.model = model

    async def acquire(self, device=None):
        self.seen.append(device)
        self.active += 1
        return "cpu"

    def release(self):
        self.active -= 1

    def lease(self, device=None):
        import contextlib

        @contextlib.asynccontextmanager
        async def cm():
            ran_on = await self.acquire(device)
            try:
                yield ran_on
            finally:
                self.release()

        return cm()


def test_synthesise_passes_the_device_and_reports_where_it_ran(client, monkeypatch):
    mgr = _CpuMgr()
    monkeypatch.setattr(xs, "_mgr", mgr)
    monkeypatch.setattr(xs, "_resolve_latents", lambda req: (None, None))
    monkeypatch.setattr(xs, "_run_inference", lambda *a: [0.0] * 240)
    monkeypatch.setattr(xs, "_upload_to_s3", lambda data, key: f"http://s3/{key}")
    r = client.post("/synthesise", json={"text": "hi", "language": "en", "device": "cpu"})
    assert r.status_code == 200, r.text
    assert mgr.seen == ["cpu"]
    assert r.json()["device"] == "cpu"
    assert client.post("/synthesise", json={"text": "hi", "device": "cuda:7"}).status_code == 422
    assert mgr.seen == ["cpu"]  # refused before any lease


class _StreamModel:
    def parameters(self):
        return iter([types.SimpleNamespace(device="cpu")])

    def inference_stream(self, *args, **kwargs):
        yield np.zeros(4, dtype=np.float32)


def test_stream_passes_the_device_and_says_where_it_runs(client, monkeypatch, cuda_calls):
    mgr = _CpuMgr(_StreamModel())
    monkeypatch.setattr(xs, "_mgr", mgr)
    monkeypatch.setattr(xs, "_resolve_latents", lambda req: (None, None))
    r = client.post("/synthesise/stream", json={"text": "hi", "device": "cpu"})
    assert r.status_code == 200, r.text
    assert r.headers["x-xtts-device"] == "cpu"
    assert len(r.content) == 16
    assert mgr.seen == ["cpu"]
