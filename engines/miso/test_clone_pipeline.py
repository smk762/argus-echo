"""Clone building and import: VAD spans, the auto-transcript, clip counts,
speech metrics, the prompt-length check and malformed archives. The STT engine
(MISO_STT_URL) is stubbed with httpx.MockTransport."""

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from unittest import mock

import httpx
import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient

import miso_server as srv
from generator import Segment
from test_utterance_shaping import _stub_generator

SR = 24000


def _archive(prompt: dict, meta=None) -> bytes:
    buf = io.BytesIO()
    pt = io.BytesIO()
    torch.save(prompt, pt)
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("prompt.pt", pt.getvalue())
        if meta is not None:
            zf.writestr("meta.json", meta if isinstance(meta, (bytes, str)) else json.dumps(meta))
    return buf.getvalue()


class _LocalStorageCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.storage = srv.LocalStorage(self.tmp.name, "http://x")
        patcher = mock.patch.object(srv, "_storage", self.storage)
        patcher.start()
        self.addCleanup(patcher.stop)
        srv._clone_cache.clear()
        self.addCleanup(srv._clone_cache.clear)


class VadSpansTest(unittest.TestCase):
    def test_close_segments_are_merged_not_duplicated(self):
        ts = [{"start": 0.5, "end": 1.0}, {"start": 1.1, "end": 1.6}]
        self.assertEqual(srv._padded_speech_spans(ts, 0.1, SR, 3 * SR), [(int(0.4 * SR), int(1.7 * SR))])

    def test_distant_segments_stay_apart_and_clamp_to_the_clip(self):
        ts = [{"start": 0.05, "end": 0.5}, {"start": 2.0, "end": 2.95}]
        self.assertEqual(srv._padded_speech_spans(ts, 0.1, SR, 3 * SR), [(0, int(0.6 * SR)), (int(1.9 * SR), 3 * SR)])

    def test_preprocess_keeps_each_sample_once(self):
        ramp = torch.linspace(-0.5, 0.5, 3 * SR)
        ts = [{"start": 0.5, "end": 1.0}, {"start": 1.1, "end": 1.6}]
        with mock.patch.object(srv, "_load_audio_24k_mono", return_value=ramp), mock.patch.object(
            srv, "_get_vad_model", return_value=None
        ), mock.patch("silero_vad.get_speech_timestamps", return_value=ts), mock.patch.object(
            srv, "REF_PREPROCESS", True
        ):
            _samples, report = srv._preprocess_reference("clip.wav")
        self.assertIn("vad_trim", report["applied"])
        self.assertEqual(report["kept_seconds"], 1.3)  # 0.4 s … 1.7 s, not 0.7 s + 0.7 s


class QuickMetricsTest(unittest.TestCase):
    def _metrics(self, timestamps):
        wav = torch.zeros(1, 2 * 16000)
        with mock.patch("torchaudio.load", return_value=(wav, 16000)), mock.patch.object(
            srv, "_get_vad_model", return_value=None
        ), mock.patch("silero_vad.get_speech_timestamps", **timestamps):
            return srv._quick_clip_metrics("clip.wav")

    def test_no_speech_reports_zero_speech_seconds(self):
        m = self._metrics({"return_value": []})
        self.assertIn("vad_no_speech", m["flags"])
        self.assertEqual(m["speech_seconds"], 0.0)
        self.assertEqual(m["raw_seconds"], 2.0)

    def test_vad_failure_keeps_the_raw_length(self):
        m = self._metrics({"side_effect": RuntimeError("vad broke")})
        self.assertEqual(m["speech_seconds"], 2.0)


class FitContextTest(unittest.TestCase):
    def test_overlong_reference_is_refused_without_request_context(self):
        gen = _stub_generator()
        # 150 s of reference audio is 1875 frames: over the 2048 budget with any generation room.
        base = [Segment(speaker=0, text="reference", audio=torch.zeros(150 * SR))]
        with self.assertRaisesRegex(ValueError, "too long for the prompt"):
            srv._fit_context(base, [], "hello there", 0, 20_000, gen)

    def test_short_reference_passes_through(self):
        gen = _stub_generator()
        base = [Segment(speaker=0, text="reference", audio=torch.zeros(5 * SR))]
        self.assertEqual(srv._fit_context(base, [], "hello there", 0, 20_000, gen), base)


class CloneVoiceTest(_LocalStorageCase):
    def setUp(self):
        super().setUp()
        self.client = TestClient(srv.app)

    def _post(self, files=None, **data):
        files = files or [
            ("audio", ("a.wav", b"clip-a", "audio/wav")),
            ("audio", ("b.wav", b"clip-b", "audio/wav")),
        ]
        return self.client.post("/clone-voice", files=files, data={"companion_id": "alice", **data})

    def _stt(self, handler):
        """Point MISO_STT_URL at a stubbed STT engine; records each request it gets."""
        seen = []

        def record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request, len(seen) - 1)

        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(srv, "STT_URL", "http://stt:8000"))
        stack.enter_context(mock.patch.object(srv, "_STT_TRANSPORT", httpx.MockTransport(record)))
        self.addCleanup(stack.close)
        return seen

    def _preprocess(self, *clips):
        """Each upload's bytes name the tensor its preprocessing returns; b"broken" fails."""

        def preprocess(path):
            with open(path, "rb") as f:
                raw = f.read()
            if raw == b"broken":
                raise RuntimeError("cannot decode")
            return clips[int(raw.decode().rsplit("-", 1)[1])], {"applied": [], "warnings": []}

        patcher = mock.patch.object(srv, "_preprocess_reference", preprocess)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_transcript_and_clip_count_follow_the_stored_reference(self):
        self._preprocess(torch.full((3 * SR,), 0.25), torch.full((3 * SR,), -0.5))
        texts = ["first clip says this.", "second clip says that."]
        seen = self._stt(lambda req, i: httpx.Response(200, json={"text": f" {texts[i]} ", "language": "en"}))
        r = self._post(
            files=[
                ("audio", ("bad.wav", b"broken", "audio/wav")),
                ("audio", ("a.wav", b"clip-0", "audio/wav")),
                ("audio", ("b.wav", b"clip-1", "audio/wav")),
            ],
            language="en-GB",
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["num_clips"], 2)  # bad.wav is not in the clone
        self.assertTrue(body["has_transcript"])
        self.assertEqual(body["reference_text_source"], "stt")
        self.assertEqual(body["reference_text_stt"], {"language": "en"})

        # One request per stored clip, in clip order, never the dropped upload.
        self.assertEqual(len(seen), 2)
        for req, level in zip(seen, (0.25, -0.5)):
            self.assertEqual(str(req.url), "http://stt:8000/transcribe")
            form = req.read()
            self.assertIn(b'name="language"\r\n\r\nen\r\n', form)
            self.assertIn(b'name="audio"; filename="clip', form)
            wav = form[form.index(b"RIFF") :]
            heard, rate = sf.read(io.BytesIO(wav[: wav.rindex(b"\r\n--")]), dtype="float32")
            self.assertEqual(rate, SR)
            self.assertEqual(len(heard), 3 * SR)
            np.testing.assert_allclose(heard, level, atol=1e-3)

        stored = torch.load(io.BytesIO(self.storage.get("voice_clones/alice/prompt.pt")), weights_only=True)
        self.assertEqual(stored["ref_audio"].shape[-1], 6 * SR)
        self.assertEqual(stored["ref_text"], "first clip says this. second clip says that.")
        meta = json.loads(self.storage.get("voice_clones/alice/meta.json"))
        self.assertEqual(meta["num_clips"], 2)
        self.assertEqual(meta["reference_text_source"], "stt")

    def test_stt_engine_and_model_are_recorded_when_returned(self):
        self._preprocess(torch.full((3 * SR,), 0.25))
        self._stt(lambda req, i: httpx.Response(200, json={"text": "hi", "engine": "whisper", "model": "large-v3"}))
        r = self._post(files=[("audio", ("a.wav", b"clip-0", "audio/wav"))])
        self.assertEqual(r.status_code, 200, r.text)
        meta = json.loads(self.storage.get("voice_clones/alice/meta.json"))
        self.assertEqual(meta["reference_text_stt"], {"language": "en", "engine": "whisper", "model": "large-v3"})

    def test_only_the_audio_kept_under_the_cap_is_transcribed(self):
        self._preprocess(torch.full((3 * SR,), 0.25), torch.full((3 * SR,), 0.25), torch.full((3 * SR,), 0.25))
        seen = self._stt(lambda req, i: httpx.Response(200, json={"text": f"clip {i}"}))
        # 3 + 3 + 3 s capped at 6.2 s: the third clip keeps 0.2 s, too little to transcribe.
        with mock.patch.object(srv, "REF_MAX_SECONDS", 6.2):
            r = self._post(
                files=[("audio", (f"{i}.wav", f"clip-{i}".encode(), "audio/wav")) for i in range(3)],
            )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(seen), 2)
        stored = torch.load(io.BytesIO(self.storage.get("voice_clones/alice/prompt.pt")), weights_only=True)
        self.assertEqual(stored["ref_text"], "clip 0 clip 1")

    def test_caller_transcript_skips_the_stt_engine(self):
        self._preprocess(torch.full((3 * SR,), 0.25))
        seen = self._stt(lambda req, i: httpx.Response(500))
        r = self._post(files=[("audio", ("a.wav", b"clip-0", "audio/wav"))], reference_text="my own words")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(seen, [])
        self.assertEqual(r.json()["reference_text_source"], "request")
        self.assertNotIn("reference_text_stt", r.json())

    def test_without_stt_url_the_clone_has_no_transcript(self):
        self._preprocess(torch.full((3 * SR,), 0.25))
        with mock.patch.object(srv, "STT_URL", ""):
            r = self._post(files=[("audio", ("a.wav", b"clip-0", "audio/wav"))])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse(r.json()["has_transcript"])
        self.assertIsNone(r.json()["reference_text_source"])

    def test_stt_failures_refuse_the_clone(self):
        def timeout(req, i):
            raise httpx.ReadTimeout("slow", request=req)

        def refused(req, i):
            raise httpx.ConnectError("connection refused", request=req)

        cases = {
            "busy": (lambda req, i: httpx.Response(503, json={"error": "busy"}, headers={"Retry-After": "7"}), 503, "7"),
            "unreachable": (refused, 503, "5"),
            "timeout": (timeout, 504, None),
            "server error": (lambda req, i: httpx.Response(500, text="boom"), 502, None),
            "bad language": (lambda req, i: httpx.Response(422, json={"error": "Unknown language"}), 502, None),
            "not json": (lambda req, i: httpx.Response(200, text="<html>"), 502, None),
            "no text": (lambda req, i: httpx.Response(200, json={"segments": []}), 502, None),
            "all empty": (lambda req, i: httpx.Response(200, json={"text": "  "}), 502, None),
            "second clip fails": (
                lambda req, i: httpx.Response(200, json={"text": "ok"}) if i == 0 else httpx.Response(500),
                502,
                None,
            ),
        }
        self._preprocess(torch.full((3 * SR,), 0.25), torch.full((3 * SR,), 0.25))
        for name, (handler, status, retry_after) in cases.items():
            with self.subTest(name):
                stack = contextlib.ExitStack()
                with stack:
                    stack.enter_context(mock.patch.object(srv, "STT_URL", "http://stt:8000"))
                    stack.enter_context(
                        mock.patch.object(
                            srv, "_STT_TRANSPORT", httpx.MockTransport(_counting(handler))
                        )
                    )
                    r = self._post(
                        files=[("audio", (f"{i}.wav", f"clip-{i}".encode(), "audio/wav")) for i in range(2)],
                        companion_id=f"x-{name.replace(' ', '-')}",
                    )
                self.assertEqual(r.status_code, status, r.text)
                self.assertIn("Speech-to-text", r.json()["error"])
                self.assertEqual(r.headers.get("Retry-After"), retry_after)
        # A refused clone stores nothing.
        self.assertEqual(self.storage.list_keys("voice_clones/"), [])


def _counting(handler):
    calls = []

    def wrapped(request):
        calls.append(request)
        return handler(request, len(calls) - 1)

    return wrapped


class ImportCloneTest(_LocalStorageCase):
    def test_non_object_meta_is_ignored(self):
        for meta in ("[1, 2]", '"text"', "7", "null"):
            out = srv._import_clone(_archive({"ref_audio": torch.zeros(SR)}, meta), "", "Voice", replace=True)
            self.assertEqual(out["companion_id"], "voice", meta)

    def test_non_string_meta_ids_are_ignored(self):
        out = srv._import_clone(_archive({"ref_audio": torch.zeros(SR)}, {"companion_id": 7, "name": ["x"]}), "", "")
        self.assertEqual(out["companion_id"], "imported-voice")

    def test_malformed_prompt_fields_are_validation_errors(self):
        for prompt in (
            {"ref_audio": torch.zeros(SR), "speaker": None},
            {"ref_audio": torch.zeros(SR), "speaker": [1]},
            {"ref_audio": torch.zeros(SR), "sample_rate": "fast"},
            {"ref_audio": torch.zeros(SR), "sample_rate": 0},
            {"ref_audio": torch.tensor(0.5)},
            {"ref_audio": torch.zeros(0)},
        ):
            with self.assertRaises(ValueError, msg=str(prompt)):
                srv._import_clone(_archive(prompt), "bob", "")
        self.assertEqual(self.storage.list_keys("voice_clones/"), [])

    def test_fallback_duration_uses_the_clones_sample_rate(self):
        out = srv._import_clone(_archive({"ref_audio": torch.zeros(32000), "sample_rate": 16000}), "bob", "")
        self.assertEqual(out["total_seconds"], 2.0)

    def test_existing_clone_is_not_replaced_without_replace(self):
        first = srv._import_clone(_archive({"ref_audio": torch.ones(SR)}, {"companion_id": "alice", "name": "A"}), "", "")
        self.assertEqual(first["companion_id"], "alice")
        other = _archive({"ref_audio": torch.zeros(2 * SR)}, {"companion_id": "alice", "name": "Other"})
        for override in ("", "alice"):  # the id from the archive, or named by the caller
            with self.assertRaises(srv.CloneExistsError):
                srv._import_clone(other, override, "")
        kept = torch.load(io.BytesIO(self.storage.get("voice_clones/alice/prompt.pt")), weights_only=True)
        self.assertEqual(kept["ref_audio"].shape[-1], SR)
        self.assertEqual(json.loads(self.storage.get("voice_clones/alice/meta.json"))["name"], "A")

        replaced = srv._import_clone(other, "", "", replace=True)
        self.assertEqual(replaced["companion_id"], "alice")
        kept = torch.load(io.BytesIO(self.storage.get("voice_clones/alice/prompt.pt")), weights_only=True)
        self.assertEqual(kept["ref_audio"].shape[-1], 2 * SR)

    def test_import_route_answers_409_then_takes_replace(self):
        client = TestClient(srv.app)
        archive = _archive({"ref_audio": torch.ones(SR)}, {"companion_id": "alice"})
        files = {"archive": ("v.zip", archive, "application/zip")}
        self.assertEqual(client.post("/clones/import", files=files).status_code, 200)
        r = client.post("/clones/import", files=files)
        self.assertEqual(r.status_code, 409, r.text)
        self.assertIn("replace=true", r.json()["error"])
        self.assertEqual(client.post("/clones/import", files=files, data={"replace": "true"}).status_code, 200)


class ListClonesTest(_LocalStorageCase):
    def test_malformed_sidecars_do_not_break_the_listing(self):
        self.storage.put("voice_clones/a/meta.json", json.dumps({"companion_id": "a", "created_at": 5}).encode())
        self.storage.put("voice_clones/b/meta.json", b"[1, 2]")
        self.storage.put("voice_clones/c/meta.json", json.dumps({"companion_id": "c", "created_at": "2026"}).encode())
        self.storage.put("voice_clones/d/meta.json", json.dumps({"companion_id": "d", "created_at": 9}).encode())
        self.assertEqual([c["companion_id"] for c in srv._list_clones()], ["d", "a", "c"])


if __name__ == "__main__":
    unittest.main()
