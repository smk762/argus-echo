"""Generator edge cases: an utterance that ends on its first frame, checkpoints
that are pickles, and backbone configs that must not drop attention weights."""

import os
import pickle
import tempfile
import unittest
from unittest import mock

import torch

import generator as gen_mod
import models
from test_utterance_shaping import K, _stub_generator


class _EosFirstModel:
    def reset_caches(self):
        pass

    def generate_frame(self, tokens, mask, pos, temperature, topk):
        return torch.zeros(1, K, dtype=torch.long)


class EmptyGenerationTest(unittest.TestCase):
    def test_eos_on_first_frame_returns_empty_clip(self):
        gen = _stub_generator()
        gen._model = _EosFirstModel()
        audio = gen.generate("...", 0, [], max_audio_length_ms=2000)
        self.assertEqual(tuple(audio.shape), (0,))
        self.assertEqual(list(gen.generate_stream("...", 0, [], max_audio_length_ms=2000)), [])

    def test_budget_under_one_frame_returns_empty_clip(self):
        gen = _stub_generator()
        audio = gen.generate("hi", 0, [], max_audio_length_ms=40)
        self.assertEqual(tuple(audio.shape), (0,))


_ran: list[str] = []


def _payload(marker: str) -> None:
    _ran.append(marker)


class _Boom:
    def __reduce__(self):
        return (_payload, ("pwned",))


class _TinyModel(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2))


class CheckpointLoadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(gen_mod, "Model", _TinyModel)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pickled_checkpoint_cannot_run_code(self):
        path = os.path.join(self.tmp.name, "model.pt")
        with open(path, "wb") as f:
            pickle.dump({"w": _Boom()}, f)
        _ran.clear()
        with self.assertRaises(Exception):
            gen_mod._load_model(path, None, device="cpu", dtype=torch.float32)
        self.assertEqual(_ran, [])

    def test_plain_tensor_checkpoint_still_loads(self):
        path = os.path.join(self.tmp.name, "model.pt")
        torch.save({"state_dict": {"module.w": torch.ones(2)}}, path)
        model = gen_mod._load_model(path, None, device="cpu", dtype=torch.float32)
        self.assertEqual(model.w.tolist(), [1.0, 1.0])
        self.assertFalse(model.training)


class BackboneConfigTest(unittest.TestCase):
    def test_no_attention_dropout(self):
        for name, build in models.FLAVORS.items():
            with mock.patch.object(models.llama3_2, "llama3_2") as llama:
                build()
            self.assertEqual(llama.call_args.kwargs["attn_dropout"], 0.0, name)


if __name__ == "__main__":
    unittest.main()
