"""The Whisper engine with a fake model and a fake VAD: no weights, no GPU, no faster-whisper."""

import asyncio
import io
import json
import os
import sys
import shutil
import subprocess
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from argus_cortex.gpu_wire import ResidentHealth  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

import stt_server  # noqa: E402


class FakeBackend:
    """Records calls; 'transcribes' to the number of samples it was given."""

    instances = []

    def __init__(self):
        self.calls = []
        self.events = []
        FakeBackend.instances.append(self)

    def load(self):
        self.events.append("load")

    def to_cpu(self):
        self.events.append("to_cpu")

    def to_device(self):
        self.events.append("to_device")

    def free(self):
        self.events.append("free")

    def transcribe(self, samples, sample_rate, language, word_timestamps, vad_filter=False):
        self.calls.append(
            {"n": len(samples), "rate": sample_rate, "language": language, "words": word_timestamps, "vad": vad_filter}
        )
        dur = len(samples) / sample_rate
        seg = {"start": 0.0, "end": round(dur, 3), "text": f"heard {len(samples)} samples"}
        if word_timestamps:
            seg["words"] = [{"start": 0.0, "end": round(dur, 3), "word": " heard", "probability": 0.9}]
        return {
            "text": seg["text"],
            "language": language or "en",
            "language_probability": None if language else 0.98,
            "duration": round(dur, 3),
            "segments": [seg],
        }


def wav(seconds=1.0, rate=16000, channels=1, fmt="WAV", subtype="PCM_16", amp=0.1):
    t = np.arange(int(seconds * rate)) / rate
    tone = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    buf = io.BytesIO()
    sf.write(buf, data, rate, format=fmt, subtype=subtype)
    return buf.getvalue()


class EngineTest(unittest.TestCase):
    def setUp(self):
        FakeBackend.instances.clear()
        patches = [
            mock.patch.object(stt_server, "backend_factory", FakeBackend),
            mock.patch.object(stt_server, "mgr", stt_server.ModelManager()),
            mock.patch.object(stt_server, "_slots", asyncio.Semaphore(1)),
            mock.patch.object(stt_server, "_intake", asyncio.Semaphore(stt_server.MAX_DECODES)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(stt_server.app)

    def post(self, data=None, **form):
        files = {"audio": ("clip.wav", wav() if data is None else data, "audio/wav")}
        return self.client.post("/transcribe", files=files, data={k: str(v) for k, v in form.items()})

    @property
    def backend(self):
        return FakeBackend.instances[-1]


class TranscribeTest(EngineTest):
    def test_returns_the_contract_shape_and_detects_language_when_absent(self):
        r = self.post()
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(set(body), {"text", "language", "language_probability", "duration", "segments"})
        self.assertEqual(body["language"], "en")
        self.assertEqual(body["duration"], 1.0)
        self.assertEqual(set(body["segments"][0]), {"start", "end", "text"})
        self.assertIsNone(self.backend.calls[0]["language"])

    def test_language_is_normalised_and_passed_through(self):
        r = self.post(language="en-US")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.backend.calls[0]["language"], "en")
        self.assertIsNone(r.json()["language_probability"])

    def test_unknown_language_is_422_before_the_model_loads(self):
        r = self.post(language="klingon")
        self.assertEqual(r.status_code, 422)
        self.assertEqual(FakeBackend.instances, [])

    def test_word_timestamps(self):
        r = self.post(word_timestamps="true")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(self.backend.calls[0]["words"])
        self.assertIn("words", r.json()["segments"][0])

    def test_stereo_is_averaged_to_mono_and_the_rate_kept(self):
        r = self.post(wav(seconds=0.5, rate=48000, channels=2))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(
            self.backend.calls[0], {"n": 24000, "rate": 48000, "language": None, "words": False, "vad": False}
        )

    def test_flac_and_float_wav_are_read(self):
        for data in (wav(fmt="FLAC", subtype="PCM_16"), wav(subtype="FLOAT")):
            self.assertEqual(self.post(data).status_code, 200)

    def test_not_wav_or_flac_is_400(self):
        r = self.post(b"ID3\x04\x00\x00" + b"\x00" * 200)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(FakeBackend.instances, [])

    def test_other_formats_are_decoded_by_ffmpeg(self):
        # agent-composer's sidecar decoded anything; rag-chat forwards the browser's webm/opus recording as is.
        decoded = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(stt_server, "ffmpeg_decode", return_value=decoded) as dec:
            r = self.post(b"\x1aE\xdf\xa3" + b"\x00" * 200)  # an EBML (webm) header
        self.assertEqual(r.status_code, 200, r.text)
        dec.assert_called_once()
        self.assertEqual(self.backend.calls[0]["n"], 16000)
        self.assertEqual(self.backend.calls[0]["rate"], 16000)

    def test_wav_does_not_go_through_ffmpeg(self):
        with mock.patch.object(stt_server, "ffmpeg_decode") as dec:
            self.assertEqual(self.post().status_code, 200)
        dec.assert_not_called()

    def test_input_formats_none_means_wav_or_flac_only(self):
        with mock.patch.object(stt_server, "INPUT_FORMATS", ""), mock.patch.object(stt_server, "ffmpeg_decode") as dec:
            r = self.post(b"\x1aE\xdf\xa3" + b"\x00" * 200)
        self.assertEqual(r.status_code, 400)
        dec.assert_not_called()
        self.assertEqual(FakeBackend.instances, [])

    def test_language_as_a_query_parameter(self):
        # agent-composer's sidecar took ?language=, and rag-chat sends it there.
        files = {"audio": ("clip.wav", wav(), "audio/wav")}
        r = self.client.post("/transcribe?language=de-DE", files=files)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.backend.calls[0]["language"], "de")

    def test_the_form_field_wins_over_the_query_parameter(self):
        files = {"audio": ("clip.wav", wav(), "audio/wav")}
        r = self.client.post("/transcribe?language=de", files=files, data={"language": "fr"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.backend.calls[0]["language"], "fr")

    def test_an_unknown_query_language_is_422(self):
        files = {"audio": ("clip.wav", wav(), "audio/wav")}
        self.assertEqual(self.client.post("/transcribe?language=klingon", files=files).status_code, 422)

    def test_upload_over_the_cap_is_413_and_never_loads(self):
        with mock.patch.object(stt_server, "MAX_UPLOAD_BYTES", 1000):
            r = self.post(wav(seconds=1.0))
        self.assertEqual(r.status_code, 413)
        self.assertEqual(FakeBackend.instances, [])

    def test_audio_over_the_duration_cap_is_413(self):
        with mock.patch.object(stt_server, "MAX_SECONDS", 0.5):
            r = self.post(wav(seconds=1.0))
        self.assertEqual(r.status_code, 413)

    def test_busy_is_503_with_retry_after(self):
        with mock.patch.object(stt_server, "_slots", asyncio.Semaphore(0)), mock.patch.object(
            stt_server, "QUEUE_TIMEOUT_S", 0.01
        ):
            r = self.post()
        self.assertEqual(r.status_code, 503)
        self.assertIn("Retry-After", r.headers)

    def test_a_full_decode_intake_is_503_and_decodes_nothing(self):
        # Shared with /transcriptions: at most STT_MAX_DECODES decodes (maybe ffmpeg processes) at once.
        with mock.patch.object(stt_server, "_intake", asyncio.Semaphore(0)), mock.patch.object(
            stt_server, "QUEUE_TIMEOUT_S", 0.01
        ), mock.patch.object(stt_server, "decode_audio_bytes") as dec:
            r = self.post()
        self.assertEqual(r.status_code, 503)
        self.assertIn("Retry-After", r.headers)
        self.assertIn("decode", r.json()["error"])
        dec.assert_not_called()
        self.assertEqual(FakeBackend.instances, [])

    def test_the_decode_slot_is_given_back(self):
        with mock.patch.object(stt_server, "_intake", asyncio.Semaphore(1)):
            self.assertEqual(self.post(b"not audio at all").status_code, 400)
            self.assertEqual(self.post().status_code, 200)

    def test_model_load_failure_is_503(self):
        class Broken(FakeBackend):
            def load(self):
                raise RuntimeError("CUDA failed with error out of memory")

        with mock.patch.object(stt_server, "backend_factory", Broken):
            r = self.post()
        self.assertEqual(r.status_code, 503)
        self.assertEqual(stt_server.mgr.state, stt_server.State.ERROR)


class HealthTest(EngineTest):
    def test_health_is_resident_health_and_advertises_stt(self):
        body = self.client.get("/health").json()
        ResidentHealth.model_validate(body)
        self.assertEqual(body["capabilities"], ["stt"])
        self.assertEqual(body["engine"], "whisper")  # clients key on it (docs/api.md)
        self.assertEqual(body["model_state"], "unloaded")
        self.assertEqual(FakeBackend.instances, [])  # health loads nothing


class LifecycleTest(EngineTest):
    def test_load_evict_wake_unload(self):
        async def go():
            m = stt_server.mgr
            m.keep_alive_gpu = m.keep_alive_cpu = -1  # no timers
            with mock.patch.object(stt_server, "DEVICE", "cuda"):
                async with m.lease():
                    self.assertEqual(m.state, stt_server.State.GPU)
                    self.assertFalse(await m.evict())  # a held lease defers it
                    self.assertFalse(await m.unload())
                self.assertTrue(await m.evict())
                self.assertEqual(m.state, stt_server.State.CPU)
                await m.ensure_loaded()
                self.assertEqual(m.state, stt_server.State.GPU)
                self.assertTrue(await m.unload())
            self.assertEqual(self.backend.events, ["load", "to_cpu", "to_device", "free"])
            self.assertEqual(m.state, stt_server.State.UNLOADED)

        asyncio.run(go())

    def test_idle_gpu_moves_to_ram_then_frees(self):
        async def go():
            m = stt_server.mgr
            m.keep_alive_gpu, m.keep_alive_cpu = 0.01, 0.01
            with mock.patch.object(stt_server, "DEVICE", "cuda"):
                async with m.lease():
                    pass
                for _ in range(100):
                    if m.state == stt_server.State.UNLOADED:
                        break
                    await asyncio.sleep(0.01)
            self.assertEqual(m.state, stt_server.State.UNLOADED)
            self.assertEqual(self.backend.events, ["load", "to_cpu", "free"])

        asyncio.run(go())


class AdminAuthTest(EngineTest):
    ROUTES = ("/admin/evict", "/admin/unload", "/admin/wake")

    def test_closed_when_token_unset(self):
        with mock.patch.dict("os.environ", {"ARGUS_ADMIN_TOKEN": "", "ARGUS_ADMIN_INSECURE": ""}):
            for route in self.ROUTES:
                self.assertEqual(self.client.post(route).status_code, 503, route)

    def test_insecure_opens_when_token_unset(self):
        with mock.patch.dict("os.environ", {"ARGUS_ADMIN_TOKEN": "", "ARGUS_ADMIN_INSECURE": "1"}):
            for route in self.ROUTES:
                self.assertEqual(self.client.post(route).status_code, 200, route)

    def test_token_required_when_set(self):
        with mock.patch.dict("os.environ", {"ARGUS_ADMIN_TOKEN": "s3cret"}):
            for route in self.ROUTES:
                self.assertEqual(self.client.post(route).status_code, 401, route)
                ok = {"Authorization": "Bearer s3cret"}
                self.assertEqual(self.client.post(route, headers=ok).status_code, 200, route)


class LanguageTest(unittest.TestCase):
    def test_normalise(self):
        self.assertIsNone(stt_server.normalise_language(None))
        self.assertIsNone(stt_server.normalise_language("  "))
        self.assertEqual(stt_server.normalise_language("EN"), "en")
        self.assertEqual(stt_server.normalise_language("pt_BR"), "pt")
        self.assertEqual(stt_server.normalise_language("haw"), "haw")
        with self.assertRaises(stt_server.LanguageError):
            stt_server.normalise_language("xx")


class EndpointerTest(unittest.TestCase):
    def make(self, **kw):
        args = {"threshold": 0.5, "min_speech": 3, "min_silence": 4, "max_frames": 50, **kw}
        return stt_server.Endpointer(**args)

    def run_probs(self, ep, probs):
        return [(i, e) for i, p in enumerate(probs) if (e := ep.feed(p))]

    def test_speech_then_silence(self):
        events = self.run_probs(self.make(), [0.1] * 5 + [0.9] * 10 + [0.1] * 6)
        self.assertEqual(events, [(7, "start"), (18, "end")])

    def test_a_blip_shorter_than_min_speech_is_not_an_utterance(self):
        self.assertEqual(self.run_probs(self.make(), [0.9, 0.9, 0.1, 0.9, 0.1] * 3), [])

    def test_a_pause_shorter_than_min_silence_does_not_end_it(self):
        events = self.run_probs(self.make(), [0.9] * 5 + [0.1] * 3 + [0.9] * 5 + [0.1] * 4)
        self.assertEqual([e for _, e in events], ["start", "end"])

    def test_hysteresis_band_neither_extends_nor_resets_silence(self):
        # 0.4 sits between neg_threshold (0.35) and threshold: it does not count as silence
        events = self.run_probs(self.make(), [0.9] * 3 + [0.4] * 10)
        self.assertEqual([e for _, e in events], ["start"])

    def test_max_frames_forces_an_end(self):
        events = self.run_probs(self.make(max_frames=10), [0.9] * 20)
        self.assertEqual([e for _, e in events], ["start", "end", "start", "end"])


def loud_vad():
    """Fake VAD: a frame is speech when it is loud."""
    return lambda frame: 0.95 if float(np.abs(frame).mean()) > 0.05 else 0.02


def pcm(seconds, amp):
    n = int(seconds * 16000)
    t = np.arange(n) / 16000
    return (amp * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()


class StreamTest(EngineTest):
    def setUp(self):
        super().setUp()
        for p in (
            mock.patch.object(stt_server, "vad_factory", loud_vad),
            mock.patch.object(stt_server, "STREAM_PARTIAL_S", 0.5),
            mock.patch.object(stt_server, "STREAM_MIN_SILENCE_MS", 300),
        ):
            p.start()
            self.addCleanup(p.stop)

    def collect(self, ws, until):
        out = []
        while True:
            msg = ws.receive_json()
            out.append(msg)
            if msg["type"] in until:
                return out

    def test_speech_then_silence_gives_partials_then_a_final(self):
        with self.client.websocket_connect("/transcribe/stream") as ws:
            ready = ws.receive_json()
            self.assertEqual((ready["type"], ready["sample_rate"], ready["audio_format"]), ("ready", 16000, "pcm_s16le"))
            ws.send_text(json.dumps({"type": "start", "language": "en", "word_timestamps": True}))
            ws.send_bytes(pcm(1.0, 0.0))
            for _ in range(10):  # 1.5 s of speech in 150 ms messages
                ws.send_bytes(pcm(0.15, 0.5))
            ws.send_bytes(pcm(0.6, 0.0))
            msgs = self.collect(ws, {"final"})
        partials = [m for m in msgs if m["type"] == "partial"]
        final = msgs[-1]
        self.assertGreaterEqual(len(partials), 1)
        self.assertAlmostEqual(final["start"], 1.0 - 0.2, delta=0.05)  # 200 ms of pre-roll before the onset
        self.assertGreater(final["end"], 2.5)
        self.assertEqual(final["language"], "en")
        self.assertGreaterEqual(final["segments"][0]["start"], final["start"])  # stream-relative times
        self.assertIn("words", final["segments"][0])
        self.assertTrue(all(c["language"] == "en" and c["words"] for c in self.backend.calls))

    def test_end_flushes_the_open_utterance_and_closes(self):
        with self.client.websocket_connect("/transcribe/stream") as ws:
            ws.receive_json()
            ws.send_bytes(pcm(0.4, 0.5)[:-3])  # an odd byte count is carried over, not refused
            ws.send_text(json.dumps({"type": "end"}))
            msgs = self.collect(ws, {"final"})
        self.assertEqual(msgs[-1]["type"], "final")

    def test_silence_only_sends_nothing_and_loads_nothing(self):
        with self.client.websocket_connect("/transcribe/stream") as ws:
            ws.receive_json()
            ws.send_bytes(pcm(2.0, 0.0))
            ws.send_text(json.dumps({"type": "end"}))
            with self.assertRaises(WebSocketDisconnect):
                ws.receive_json()
        self.assertEqual(FakeBackend.instances, [])

    def test_bad_language_and_bad_control_messages_are_errors(self):
        for msg in ({"type": "start", "language": "klingon"}, {"type": "nope"}):
            with self.client.websocket_connect("/transcribe/stream") as ws:
                ws.receive_json()
                ws.send_text(json.dumps(msg))
                self.assertEqual(ws.receive_json()["type"], "error")

    def test_oversized_frame_is_refused(self):
        with mock.patch.object(stt_server, "STREAM_MAX_MESSAGE_BYTES", 1024):
            with self.client.websocket_connect("/transcribe/stream") as ws:
                ws.receive_json()
                ws.send_bytes(b"\x00" * 2048)
                self.assertEqual(ws.receive_json()["type"], "error")

    def test_session_limit(self):
        with mock.patch.object(stt_server, "STREAM_MAX_SESSIONS", 1):
            with self.client.websocket_connect("/transcribe/stream") as ws:
                ws.receive_json()
                with self.client.websocket_connect("/transcribe/stream") as ws2:
                    self.assertEqual(ws2.receive_json()["type"], "error")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class RealFfmpegTranscribeTest(EngineTest):
    def test_a_browser_webm_opus_recording_is_transcribed(self):
        webm = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=1",
             "-ar", "48000", "-c:a", "libopus", "-f", "webm", "pipe:1"],
            capture_output=True, check=True,
        ).stdout  # fmt: skip
        r = self.client.post("/transcribe", files={"audio": ("rec.webm", webm, "audio/webm")})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertAlmostEqual(self.backend.calls[0]["n"], 16000, delta=1600)
        self.assertEqual(self.backend.calls[0]["rate"], 16000)
