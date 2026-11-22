"""Clone building and import: VAD spans, the auto-transcript, clip counts,
speech metrics, the prompt-length check and malformed archives."""

import io
import json
import tempfile
import unittest
import zipfile
from unittest import mock

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

    def test_transcript_and_clip_count_follow_the_stored_reference(self):
        good = torch.full((3 * SR,), 0.25)
        transcribed = []

        def preprocess(path):
            with open(path, "rb") as f:
                if f.read() == b"broken":
                    raise RuntimeError("cannot decode")
            return good, {"applied": [], "warnings": []}

        def transcribe(wav_bytes):
            transcribed.append(wav_bytes)
            return "what the reference says"

        with mock.patch.object(srv, "_preprocess_reference", preprocess), mock.patch.object(
            srv, "_transcribe_reference", transcribe
        ), mock.patch.object(srv, "WHISPER_URL", "http://whisper"):
            r = self.client.post(
                "/clone-voice",
                files=[
                    ("audio", ("bad.wav", b"broken", "audio/wav")),
                    ("audio", ("a.wav", b"clip-a", "audio/wav")),
                    ("audio", ("b.wav", b"clip-b", "audio/wav")),
                ],
                data={"companion_id": "alice"},
            )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["num_clips"], 2)  # bad.wav is not in the clone
        self.assertTrue(body["has_transcript"])

        # The transcriber heard the stored reference: both good clips, never the dropped upload.
        self.assertEqual(len(transcribed), 1)
        heard, rate = sf.read(io.BytesIO(transcribed[0]), dtype="float32")
        self.assertEqual(rate, SR)
        self.assertEqual(len(heard), 6 * SR)
        np.testing.assert_allclose(heard, 0.25, atol=1e-3)
        stored = torch.load(io.BytesIO(self.storage.get("voice_clones/alice/prompt.pt")), weights_only=True)
        self.assertEqual(stored["ref_audio"].shape[-1], len(heard))
        self.assertEqual(stored["ref_text"], "what the reference says")
        self.assertEqual(json.loads(self.storage.get("voice_clones/alice/meta.json"))["num_clips"], 2)


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
