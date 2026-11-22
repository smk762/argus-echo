"""#22: what a demote to the CPU leaves on the GPU, and refusing to start with an unusable S3 endpoint.

Torch is faked per test (the helpers import it when called), so no GPU or model is needed.
"""

import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.device = lambda d: d
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules["torch"] = _torch

import tts_server  # noqa: E402
from tts_server import TTSModelManager  # noqa: E402


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


class _Module:
    """Stands in for an nn.Module: records where it was moved."""

    def __init__(self, device="cuda:0"):
        self.device = device
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        self.device = device
        return self


class _SpeechTokenizer:
    """qwen-tts's Qwen3TTSTokenizer: a plain object (not a Module) holding its own module and a cached device."""

    def __init__(self):
        self.model = _Module()
        self.device = "cuda:0"


class _Wrapper:
    """Qwen3TTSModel: a plain wrapper; the TTS module is ``.model``, which carries ``.speech_tokenizer``."""

    def __init__(self):
        self.model = _Module()
        self.model.speech_tokenizer = _SpeechTokenizer()
        self.device = "cuda:0"


class SpeechTokenizerMovesWithTheModelTest(unittest.TestCase):
    def _manager(self) -> TTSModelManager:
        mgr = TTSModelManager()
        mgr._base_model = _Wrapper()
        mgr._custom_model = _Wrapper()
        return mgr

    def test_demote_moves_each_speech_tokenizer_to_the_cpu(self):
        mgr = self._manager()
        with mock.patch.dict(sys.modules, {"torch": _fake_torch([])}):
            mgr._move_to_device("cpu")
        for wrapper in (mgr._base_model, mgr._custom_model):
            tokenizer = wrapper.model.speech_tokenizer
            self.assertEqual(tokenizer.model.moves, ["cpu"])
            self.assertEqual(tokenizer.model.device, "cpu")
            self.assertEqual(tokenizer.device, "cpu", "the tokenizer places its inputs on its cached device")

    def test_promote_brings_it_back(self):
        mgr = self._manager()
        with mock.patch.dict(sys.modules, {"torch": _fake_torch([])}):
            mgr._move_to_device("cpu")
            mgr._move_to_device("cuda:0")
        tokenizer = mgr._custom_model.model.speech_tokenizer
        self.assertEqual(tokenizer.model.moves, ["cpu", "cuda:0"])
        self.assertEqual(tokenizer.device, "cuda:0")

    def test_a_model_without_a_speech_tokenizer_still_moves(self):
        mgr = TTSModelManager()
        wrapper = _Wrapper()
        del wrapper.model.speech_tokenizer
        mgr._custom_model = wrapper
        with mock.patch.dict(sys.modules, {"torch": _fake_torch([])}):
            mgr._move_to_device("cpu")
        self.assertEqual(wrapper.model.moves, ["cpu"])

    def test_demote_clears_the_cublas_workspaces_before_emptying_the_cache(self):
        calls: list = []
        with mock.patch.dict(sys.modules, {"torch": _fake_torch(calls)}):
            self._manager()._move_to_device("cpu")
        self.assertEqual(calls, ["clear_cublas", "empty_cache"])

    def test_unload_releases_the_cache_too(self):
        calls: list = []
        mgr = self._manager()
        with mock.patch.dict(sys.modules, {"torch": _fake_torch(calls)}):
            mgr._free_models()
        self.assertIsNone(mgr._base_model)
        self.assertEqual(calls, ["clear_cublas", "empty_cache"])

    def test_no_cuda_context_means_nothing_to_release(self):
        calls: list = []
        with mock.patch.dict(sys.modules, {"torch": _fake_torch(calls, initialized=False)}):
            tts_server._release_cuda_cache()
        self.assertEqual(calls, [])


class S3EndpointRefusedAtStartupTest(unittest.TestCase):
    def _check(self, **overrides):
        values = {"S3_ACCESS_KEY_ID": "k", "S3_SECRET_ACCESS_KEY": "s", "S3_ENDPOINT_URL": "https://s3.example",
                  "S3_BUCKET_NAME": "voice-qwen", **overrides}
        with mock.patch.multiple(tts_server, **values):
            tts_server._require_s3_credentials()

    def test_a_usable_configuration_passes(self):
        self._check()
        self._check(S3_ENDPOINT_URL="http://minio:9000")

    def test_an_empty_or_unusable_endpoint_is_refused(self):
        for bad in ("", "   ", "s3.example", "ftp://s3.example", "https://"):
            with self.subTest(endpoint=bad), self.assertRaisesRegex(RuntimeError, "S3_ENDPOINT_URL"):
                self._check(S3_ENDPOINT_URL=bad)

    def test_an_empty_bucket_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "S3_BUCKET_NAME"):
            self._check(S3_BUCKET_NAME=" ")


class HealthNamesTheCudaContextTest(unittest.TestCase):
    def test_gpu_block_says_whether_a_context_is_held(self):
        with mock.patch.dict(sys.modules, {"torch": _fake_torch([], initialized=False)}):
            self.assertIs(tts_server._resident_gpu()["cuda_context"], False)


if __name__ == "__main__":
    unittest.main()
