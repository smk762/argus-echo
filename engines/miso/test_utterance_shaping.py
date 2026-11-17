"""Length budget (#3), request context (#4) and edge fades (#5) for Miso."""

import base64
import contextlib
import threading
import unittest
from unittest import mock

import numpy as np
import torch
from fastapi.testclient import TestClient

import generator as gen_mod
import miso_server as srv
from generator import Generator, Segment

K = 4  # codebooks in the stub model
FRAME = 1920  # samples per 80 ms frame at 24 kHz


class _NeverEndingModel:
    """generate_frame never returns the all-zero end-of-speech frame."""

    def __init__(self):
        self.frames = 0

    def reset_caches(self):
        self.frames = 0

    def generate_frame(self, tokens, mask, pos, temperature, topk):
        self.frames += 1
        return torch.ones(1, K, dtype=torch.long)


class _Tokenizer:
    def encode(self, text):
        return list(range(len(text.split()) + 2))


class _Mimi:
    def decode(self, codes):  # (1, K, n) → (1, 1, n * FRAME)
        return torch.full((1, 1, codes.shape[-1] * FRAME), 0.1)

    def streaming(self, batch_size):
        return contextlib.nullcontext()


def _stub_generator():
    """A real Generator whose model, tokenizers and codec are stubs."""
    gen = Generator.__new__(Generator)
    gen._model = _NeverEndingModel()
    gen._text_tokenizer = _Tokenizer()
    gen._audio_tokenizer = _Mimi()
    gen._frame_size = K + 1
    gen._watermarker = None
    gen.sample_rate = 24000
    gen.device = torch.device("cpu")
    return gen


def _words(n):
    return " ".join(["word"] * n)


class LengthBudgetTest(unittest.TestCase):
    def test_budget_scales_with_words_and_respects_ceiling(self):
        self.assertAlmostEqual(srv._max_audio_ms_for("Greetings."), (1.5 + 1 / 1.6) * 1000)
        self.assertAlmostEqual(srv._max_audio_ms_for(_words(40)), (1.5 + 40 / 1.6) * 1000)
        self.assertEqual(srv._max_audio_ms_for(_words(200)), srv.MAX_AUDIO_MS)
        self.assertEqual(srv._max_audio_ms_for(""), srv._max_audio_ms_for("one"))

    def test_digit_strings_count_per_digit(self):
        self.assertEqual(srv._word_count("Call 555-123-4567 now"), 2 + 10)
        self.assertEqual(srv._word_count("in 2024"), 1 + 4)

    def test_joined_and_unspaced_text_is_not_one_word(self):
        self.assertEqual(srv._word_count("state-of-the-art"), 4)
        self.assertEqual(srv._word_count("go to example.com/docs/setup"), 6)
        self.assertEqual(srv._word_count("don't stop"), 2)
        self.assertEqual(srv._word_count("今天天气很好"), 6)
        self.assertEqual(srv._word_count("$5 off"), 3)
        self.assertEqual(srv._word_count("..."), 1)

    def test_never_ending_generation_stops_at_budget_on_both_paths(self):
        gen = _stub_generator()
        no_watermark = lambda wm, audio, sr, key: (audio, sr)  # noqa: E731
        with mock.patch.object(srv._mgr, "_generator", gen), mock.patch.object(
            srv, "_resolve_context", return_value=(0, [], {})
        ), mock.patch.object(gen_mod, "watermark", no_watermark), mock.patch.object(
            srv, "logger"
        ) as log:
            for n_words in (1, 5, 20, 60):
                req = srv.SynthesiseRequest(text=_words(n_words))
                budget = srv._max_frames(srv._max_audio_ms_for(req.text))

                audio = srv._generate_audio(req)
                self.assertEqual(gen._model.frames, budget, n_words)
                self.assertEqual(audio.size, budget * FRAME, n_words)

                chunks = list(srv._generate_stream_chunks(req, threading.Event()))
                self.assertEqual(gen._model.frames, budget, n_words)
                self.assertEqual(sum(c.size for c in chunks), budget * FRAME, n_words)

        capped = [c for c in log.warning.call_args_list if c.args[0] == "miso_generation_capped"]
        self.assertEqual(len(capped), 8)  # 4 lengths × 2 paths
        self.assertEqual(capped[0].kwargs["words"], 1)
        self.assertTrue(capped[-1].kwargs["at_ceiling"])  # 60 words hits the 30 s ceiling

    def test_end_of_speech_is_not_logged_as_capped(self):
        with mock.patch.object(srv, "logger") as log:
            self.assertFalse(srv._log_if_capped("a b c", 4000, 10 * FRAME, "synthesise"))
        log.warning.assert_not_called()


class _ChunkGenerator:
    """Yields fixed chunks of constant, full-level audio; generate() returns all of it."""

    def __init__(self, sizes, level=0.1):
        self.chunks = [np.full(n, level, dtype=np.float32) for n in sizes]

    def generate(self, **kwargs):
        return torch.from_numpy(np.concatenate(self.chunks))

    def generate_stream(self, **kwargs):
        for c in self.chunks:
            yield torch.from_numpy(c)


class EdgeFadeTest(unittest.TestCase):
    def _both(self, sizes):
        gen = _ChunkGenerator(sizes)
        req = srv.SynthesiseRequest(text="hello there")
        with mock.patch.object(srv._mgr, "_generator", gen), mock.patch.object(
            srv, "_resolve_context", return_value=(0, [], {})
        ):
            whole = srv._generate_audio(req)
            chunks = list(srv._generate_stream_chunks(req, threading.Event()))
        return whole, chunks

    def test_endpoints_start_at_zero_and_end_near_silence(self):
        whole, chunks = self._both([9600, 28800, 500, 1000])
        ten_ms = 240
        for audio in (whole, np.concatenate(chunks)):
            self.assertEqual(audio[0], 0.0)
            self.assertEqual(audio[-1], 0.0)
            # Speech stopped at 0.1 (≈ -20 dBFS); a clean decay is ≈ 0.001.
            self.assertLess(float(np.sqrt(np.mean(audio[-ten_ms:] ** 2))), 0.003)
            # The middle is untouched.
            self.assertTrue(np.all(audio[1000:-1000] == np.float32(0.1)))

    def test_stream_equals_faded_whole_and_only_tail_is_held_back(self):
        whole, chunks = self._both([9600, 28800, 500, 1000])
        np.testing.assert_array_equal(np.concatenate(chunks), whole)
        hold = srv._ms_to_samples(srv.FADE_OUT_MS)
        self.assertEqual([c.size for c in chunks], [9600 - hold, 28800, 500, 1000, hold])

    def test_utterance_shorter_than_the_fades(self):
        whole, chunks = self._both([50, 30])
        np.testing.assert_array_equal(np.concatenate(chunks), whole)
        self.assertEqual(whole.size, 80)
        self.assertEqual(whole[0], 0.0)
        self.assertEqual(whole[-1], 0.0)


def _pcm_b64(samples):
    return base64.b64encode(np.asarray(samples, dtype="<f4").tobytes()).decode()


class _TokenCounter:
    max_seq_len = 2048
    _text_tokenizer = _Tokenizer()  # "[speaker] text" → words + 3


class RequestContextTest(unittest.TestCase):
    def setUp(self):
        self.ref = Segment(speaker=3, text="reference words", audio=torch.zeros(FRAME * 10))
        self.clone = mock.patch.object(srv, "_resolve_context", return_value=(3, [self.ref], {}))
        self.clone.start()
        self.addCleanup(self.clone.stop)

    def _prepare(self, context, gen=None):
        req = srv.SynthesiseRequest(text="the next sentence", context=context)
        with mock.patch.object(srv._mgr, "_generator", gen or _TokenCounter()):
            return srv._prepare_generation(req)

    def test_no_context_is_unchanged(self):
        speaker, context, *_ = self._prepare(None, gen=object())  # generator not consulted
        self.assertEqual(speaker, 3)
        self.assertEqual(context, [self.ref])

    def test_reference_first_then_request_segments_in_order(self):
        older = np.full(FRAME * 3, 0.2, dtype=np.float32)
        newer = np.full(FRAME * 2, -0.3, dtype=np.float32)
        _, context, *_ = self._prepare(
            [{"text": "first one", "audio": _pcm_b64(older)}, {"text": "second one", "audio": _pcm_b64(newer)}]
        )
        self.assertIs(context[0], self.ref)
        self.assertEqual([s.text for s in context[1:]], ["first one", "second one"])
        self.assertEqual({s.speaker for s in context}, {3})
        np.testing.assert_array_equal(context[2].audio.numpy(), newer)

    def test_budget_drops_oldest_request_segment_never_the_reference(self):
        gen = _TokenCounter()
        budget = srv._max_frames(srv._max_audio_ms_for("the next sentence"))
        ref_len = srv._segment_prompt_len(self.ref, gen)
        target_len = srv._text_token_count(gen, "the next sentence", 3)
        seg_len = 5 + 50 + 1  # "[3] first one" tokens + 50 frames + EOS
        # Room for exactly one request segment.
        gen.max_seq_len = budget + srv.CONTEXT_TOKEN_MARGIN + ref_len + target_len + seg_len + 1
        audio = _pcm_b64(np.zeros(FRAME * 50, dtype=np.float32))
        with mock.patch.object(srv, "logger") as log:
            _, context, *_ = self._prepare(
                [{"text": "first one", "audio": audio}, {"text": "second one", "audio": audio}], gen=gen
            )
        self.assertEqual([s.text for s in context], ["reference words", "second one"])
        trimmed = [c for c in log.warning.call_args_list if c.args[0] == "miso_context_trimmed"]
        self.assertEqual(trimmed[0].kwargs["dropped"], 1)

        gen.max_seq_len -= 2  # now neither fits; the reference stays
        _, context, *_ = self._prepare(
            [{"text": "first one", "audio": audio}, {"text": "second one", "audio": audio}], gen=gen
        )
        self.assertEqual(context, [self.ref])

    def test_line_wrapped_base64_is_accepted(self):
        samples = np.full(FRAME, 0.25, dtype=np.float32)
        wrapped = base64.encodebytes(samples.astype("<f4").tobytes()).decode()
        self.assertIn("\n", wrapped)
        _, context, *_ = self._prepare([{"text": "first one", "audio": wrapped}])
        np.testing.assert_array_equal(context[1].audio.numpy(), samples)

    def test_reference_too_long_for_prompt_is_a_value_error(self):
        gen = _TokenCounter()
        gen.max_seq_len = 10  # the reference alone overflows
        with self.assertRaises(ValueError):
            self._prepare([{"text": "a", "audio": _pcm_b64(np.zeros(FRAME))}], gen=gen)

    def test_bad_context_is_a_value_error(self):
        ok = _pcm_b64(np.zeros(FRAME))
        too_long = _pcm_b64(np.zeros(int(srv.CONTEXT_MAX_SECONDS * 24000) + 1))
        half = _pcm_b64(np.zeros(int(srv.CONTEXT_MAX_SECONDS * 24000) // 2 + FRAME))
        cases = {
            "not base64": [{"text": "a", "audio": "!!!not-base64!!!"}],
            "odd bytes": [{"text": "a", "audio": base64.b64encode(b"\x00" * 6).decode()}],
            "empty": [{"text": "a", "audio": ""}],
            "nan": [{"text": "a", "audio": _pcm_b64([0.0, float("nan")])}],
            "s16 as f32": [{"text": "a", "audio": _pcm_b64([1e6])}],
            "no text": [{"text": "  ", "audio": ok}],
            "too many": [{"text": "a", "audio": ok}] * (srv.CONTEXT_MAX_SEGMENTS + 1),
            "too long": [{"text": "a", "audio": too_long}],
            "too long in total": [{"text": "a", "audio": half}, {"text": "b", "audio": half}],
        }
        for name, context in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self._prepare(context)


class RequestContextHttpTest(unittest.TestCase):
    def test_endpoints_return_422_for_malformed_context(self):
        client = TestClient(srv.app)
        bad = {"text": "hello", "context": [{"text": "a", "audio": "%%%"}]}
        wrong_shape = {"text": "hello", "context": [{"words": "a"}]}
        with mock.patch.object(srv._mgr, "ensure_ready", mock.AsyncMock()), mock.patch.object(
            srv._mgr, "_generator", _TokenCounter()
        ):
            for path in ("/synthesise", "/synthesise/stream"):
                for body in (bad, wrong_shape):
                    with self.subTest(path=path, body=body):
                        self.assertEqual(client.post(path, json=body).status_code, 422)


if __name__ == "__main__":
    unittest.main()
