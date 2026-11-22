"""#22: what a demote to the CPU leaves on the GPU, and refusing to start with an unusable S3 endpoint.

Torch is faked per test (the helpers import it when called), so no GPU or model is needed.
"""

import asyncio
import sys
import types

import pytest
import xtts_server as xs


def _fake_torch(calls: list, initialized: bool = True) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.device = lambda d: d
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: initialized,
        empty_cache=lambda: calls.append("empty_cache"),
    )
    torch._C = types.SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: calls.append("clear_cublas"))
    return torch


class _Tensor:
    def __init__(self, is_cuda: bool):
        self.is_cuda = is_cuda


class _Module:
    def __init__(self, *children, prefix=None):
        self._children = children
        self.cached_prefix_emb = prefix

    def modules(self):
        yield self
        for child in self._children:
            yield from child.modules()


class _Model(_Module):
    def __init__(self, *children):
        super().__init__(*children)
        self.moved = []

    def cpu(self):
        self.moved.append("cpu")
        return self


def test_demote_drops_the_cuda_prefix_and_releases_the_cache(monkeypatch):
    calls: list = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls))
    gpt_inference = _Module(prefix=_Tensor(is_cuda=True))
    model = _Model(_Module(gpt_inference))
    mgr = xs.XTTSModelManager()
    mgr._model = model
    mgr._move_to_device("cpu")
    assert model.moved == ["cpu"]
    assert gpt_inference.cached_prefix_emb is None, "Module.cpu() does not move a plain tensor attribute"
    assert calls == ["clear_cublas", "empty_cache"], "cuBLAS's workspace pins its allocator segment"


def test_a_cpu_prefix_is_kept():
    cpu_prefix = _Tensor(is_cuda=False)
    module = _Module(prefix=cpu_prefix)
    xs._drop_cuda_attributes(_Model(module))
    assert module.cached_prefix_emb is cpu_prefix


def test_unload_releases_the_cache(monkeypatch):
    calls: list = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls))
    mgr = xs.XTTSModelManager()
    mgr._model = _Model()
    mgr._state = xs.TTSState.CPU
    assert asyncio.run(mgr.unload())
    assert mgr.model is None
    assert calls == ["clear_cublas", "empty_cache"]


def test_no_cuda_context_means_nothing_to_release(monkeypatch):
    calls: list = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls, initialized=False))
    xs._release_cuda_cache()
    assert calls == []


def _s3(monkeypatch, **overrides):
    values = {"S3_ACCESS_KEY_ID": "k", "S3_SECRET_ACCESS_KEY": "s", "S3_ENDPOINT_URL": "https://s3.example",
              "S3_BUCKET_NAME": "voice-xtts", **overrides}
    for name, value in values.items():
        monkeypatch.setattr(xs, name, value)
    xs._require_s3_credentials()


def test_a_usable_s3_configuration_passes(monkeypatch):
    _s3(monkeypatch)
    _s3(monkeypatch, S3_ENDPOINT_URL="http://minio:9000")


@pytest.mark.parametrize("endpoint", ["", "   ", "s3.example", "ftp://s3.example", "https://"])
def test_an_empty_or_unusable_endpoint_is_refused_at_startup(monkeypatch, endpoint):
    # morpheus, 2026-11-22: an empty S3_ENDPOINT_URL passed startup and failed each /synthesise after rendering.
    with pytest.raises(RuntimeError, match="S3_ENDPOINT_URL"):
        _s3(monkeypatch, S3_ENDPOINT_URL=endpoint)


def test_an_empty_bucket_is_refused(monkeypatch):
    with pytest.raises(RuntimeError, match="S3_BUCKET_NAME"):
        _s3(monkeypatch, S3_BUCKET_NAME="")


def test_gpu_block_says_whether_a_context_is_held(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([], initialized=True))
    monkeypatch.setattr(xs, "_gpu_memory_info", lambda: {"reserved_mb": 0.0})
    monkeypatch.setattr(xs, "_free_vram_mb", lambda: None)
    block = xs._resident_gpu()
    assert block["cuda_context"] is True
    assert block["reserved_mb"] == 0.0
