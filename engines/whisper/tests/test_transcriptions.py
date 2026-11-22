"""POST /transcriptions (agent-composer's JSON route for kimini-api), with the fake model: no weights, no GPU.

Ports voice-stack/tests/test_whisper_audio_url.py from agent-composer: the SSRF policy on audio_url, the download
and base64 caps, and the generic error bodies, against echo's fetch (argus_cortex.net.afetch) and caps.
"""

import asyncio
import base64
import contextlib
import ipaddress
import os
import shutil
import socket
import subprocess
import sys
import time
import unittest
from unittest import mock

import httpx
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from test_stt_server import EngineTest, FakeBackend, wav  # noqa: E402

import stt_server  # noqa: E402

HOSTS = {
    "public.example": ["93.184.216.34"],
    "other.example": ["93.184.216.35"],
    "internal.example": ["10.0.0.5"],
    "mixed.example": ["93.184.216.34", "127.0.0.1"],
    "dual.example": ["2606:2800:220:1:248:1893:25c8:1946", "93.184.216.34"],
    "minio": ["172.18.0.4"],
}


def resolve(host, port):
    try:
        return [str(ipaddress.ip_address(host.strip("[]")))]
    except ValueError:
        pass
    if host not in HOSTS:
        raise socket.gaierror("no such host")
    return HOSTS[host]


def b64(data):
    return base64.b64encode(data).decode()


class TranscriptionsTest(EngineTest):
    def setUp(self):
        super().setUp()
        self.seen = []
        self.upstream = lambda r: httpx.Response(200, content=wav())
        for p in (
            mock.patch.object(stt_server, "audio_url_resolver", resolve),
            mock.patch.object(stt_server, "audio_url_transport", httpx.MockTransport(self._handle)),
            mock.patch.object(stt_server, "AUDIO_URL_ALLOWED_HOSTS", ()),
            mock.patch.object(stt_server, "AUDIO_URL_MAX_REDIRECTS", 0),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _handle(self, request):
        self.seen.append(request)
        return self.upstream(request)

    def tx(self, **body):
        return self.client.post("/transcriptions", json=body)

    def url(self, url, **body):
        return self.tx(audio_url=url, **body)

    def assert_nothing_loaded(self):
        self.assertEqual(FakeBackend.instances, [])


class Base64Test(TranscriptionsTest):
    def test_returns_composers_shape_with_vad_on(self):
        r = self.tx(audio_base64=b64(wav()))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"text": "heard 16000 samples", "language": "en", "duration_seconds": 1.0, "confidence": 0.98})
        self.assertEqual(self.backend.calls[0], {"n": 16000, "rate": 16000, "language": None, "words": False, "vad": True})

    def test_vad_follows_the_setting(self):
        with mock.patch.object(stt_server, "VAD_FILTER", False):
            self.assertEqual(self.tx(audio_base64=b64(wav())).status_code, 200)
        self.assertFalse(self.backend.calls[0]["vad"])

    def test_language_is_normalised_and_confidence_is_1(self):
        r = self.tx(audio_base64=b64(wav()), language="en-US")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.backend.calls[0]["language"], "en")
        self.assertEqual(r.json()["confidence"], 1.0)

    def test_data_uri_and_kiminis_nulls_are_accepted(self):
        r = self.tx(audio_base64="data:audio/wav;base64," + b64(wav()), audio_url=None, language=None)
        self.assertEqual(r.status_code, 200, r.text)

    def test_flac_is_read(self):
        self.assertEqual(self.tx(audio_base64=b64(wav(fmt="FLAC", subtype="PCM_16"))).status_code, 200)

    def test_bad_base64_is_400(self):
        r = self.tx(audio_base64="not base64 at all!!")
        self.assertEqual(r.status_code, 400)
        self.assert_nothing_loaded()

    def test_both_is_422_and_neither_is_400(self):
        self.assertEqual(self.tx(audio_base64=b64(wav()), audio_url="http://public.example/a.wav").status_code, 422)
        self.assertEqual(self.tx(language="en").status_code, 400)
        self.assertEqual(self.tx(audio_base64="").status_code, 400)
        self.assertEqual(self.seen, [])
        self.assert_nothing_loaded()

    def test_malformed_body_is_422(self):
        for content in (b"{not json", b"[1, 2]", b'{"audio_base64": 5}'):
            r = self.client.post("/transcriptions", content=content, headers={"Content-Type": "application/json"})
            self.assertEqual(r.status_code, 422, content)

    def test_unknown_language_is_422_before_anything_loads(self):
        self.assertEqual(self.tx(audio_base64=b64(wav()), language="klingon").status_code, 422)
        self.assert_nothing_loaded()

    def test_base64_over_the_cap_is_413_before_decoding(self):
        with mock.patch.object(stt_server, "MAX_UPLOAD_BYTES", 1000), mock.patch("base64.b64decode") as decode:
            r = self.tx(audio_base64="data:audio/wav;base64," + b64(b"z" * 10_000))
        self.assertEqual(r.status_code, 413)
        decode.assert_not_called()
        self.assert_nothing_loaded()

    def test_base64_decoding_past_the_cap_is_413(self):
        with mock.patch.object(stt_server, "MAX_UPLOAD_BYTES", 1000):
            r = self.tx(audio_base64=b64(b"z" * 1001))
        self.assertEqual(r.status_code, 413)

    def test_a_body_over_the_cap_is_413_before_parsing(self):
        with mock.patch.object(stt_server, "MAX_UPLOAD_BYTES", 1000), mock.patch.object(
            stt_server.TranscriptionsRequest, "model_validate_json"
        ) as parse:
            r = self.tx(audio_base64="A" * 200_000)
        self.assertEqual(r.status_code, 413)
        parse.assert_not_called()

    def test_audio_over_the_duration_cap_is_413(self):
        with mock.patch.object(stt_server, "MAX_SECONDS", 0.5):
            self.assertEqual(self.tx(audio_base64=b64(wav(seconds=1.0))).status_code, 413)
        self.assert_nothing_loaded()

    def test_busy_is_503_with_retry_after(self):
        with mock.patch.object(stt_server, "_slots", asyncio.Semaphore(0)), mock.patch.object(
            stt_server, "QUEUE_TIMEOUT_S", 0.01
        ):
            r = self.tx(audio_base64=b64(wav()))
        self.assertEqual(r.status_code, 503)
        self.assertIn("Retry-After", r.headers)

    def test_model_load_failure_is_503(self):
        class Broken(FakeBackend):
            def load(self):
                raise RuntimeError("CUDA failed with error out of memory")

        with mock.patch.object(stt_server, "backend_factory", Broken):
            self.assertEqual(self.tx(audio_base64=b64(wav())).status_code, 503)


def _fake_ffmpeg(samples):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=np.zeros(samples, dtype="<i2").tobytes(), stderr=b"")

    return run, calls


class FfmpegTest(TranscriptionsTest):
    MP3ISH = b"ID3\x04\x00\x00" + b"\x00" * 200

    def test_non_wav_goes_through_ffmpeg_reading_only_a_pipe(self):
        run, calls = _fake_ffmpeg(8000)
        with mock.patch.object(stt_server.subprocess, "run", run):
            r = self.tx(audio_base64=b64(self.MP3ISH))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.backend.calls[0]["n"], 8000)
        self.assertEqual(self.backend.calls[0]["rate"], 16000)
        cmd, kw = calls[0]
        self.assertEqual(cmd[cmd.index("-protocol_whitelist") + 1], "pipe")
        self.assertEqual(cmd[cmd.index("-format_whitelist") + 1], stt_server.INPUT_FORMATS)
        self.assertEqual(cmd[cmd.index("-i") + 1], "pipe:0")
        self.assertNotIn("hls", stt_server.INPUT_FORMATS.split(","))
        self.assertEqual(kw["input"], self.MP3ISH)
        self.assertEqual(kw["timeout"], stt_server.DECODE_TIMEOUT)

    def test_wav_never_reaches_ffmpeg(self):
        with mock.patch.object(stt_server.subprocess, "run") as run:
            self.assertEqual(self.tx(audio_base64=b64(wav())).status_code, 200)
        run.assert_not_called()

    def test_ffmpeg_output_past_the_duration_cap_is_413(self):
        run, _ = _fake_ffmpeg(16000 + 100)
        with mock.patch.object(stt_server.subprocess, "run", run), mock.patch.object(stt_server, "MAX_SECONDS", 1.0):
            self.assertEqual(self.tx(audio_base64=b64(self.MP3ISH)).status_code, 413)
        self.assert_nothing_loaded()

    def test_ffmpeg_failure_or_absence_is_400(self):
        failed = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"Invalid data")  # noqa: E731
        for run in (failed, mock.Mock(side_effect=FileNotFoundError("ffmpeg"))):
            with mock.patch.object(stt_server.subprocess, "run", run):
                self.assertEqual(self.tx(audio_base64=b64(self.MP3ISH)).status_code, 400)
        self.assert_nothing_loaded()

    def test_formats_none_accepts_only_wav_and_flac(self):
        with mock.patch.object(stt_server, "INPUT_FORMATS", ""), mock.patch.object(stt_server.subprocess, "run") as run:
            self.assertEqual(self.tx(audio_base64=b64(self.MP3ISH)).status_code, 400)
        run.assert_not_called()


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class RealFfmpegTest(TranscriptionsTest):
    def test_mp3_is_decoded(self):
        mp3 = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=1",
             "-ar", "22050", "-f", "mp3", "pipe:1"],
            capture_output=True, check=True,
        ).stdout  # fmt: skip
        r = self.tx(audio_base64=b64(mp3))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertAlmostEqual(self.backend.calls[0]["n"], 16000, delta=1600)

    def test_a_playlist_is_refused_and_nothing_is_fetched(self):
        playlist = b"#EXTM3U\n#EXTINF:1,\nhttp://127.0.0.1:9/x.mp3\n#EXTINF:1,\nfile:///etc/passwd\n"
        self.assertEqual(self.tx(audio_base64=b64(playlist)).status_code, 400)
        self.assert_nothing_loaded()


class AudioUrlTest(TranscriptionsTest):
    def test_fetches_pinned_to_the_checked_address(self):
        r = self.url("http://public.example/a.wav?sig=1", language="en")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["text"], "heard 16000 samples")
        self.assertEqual(self.seen[0].url.host, "93.184.216.34")
        self.assertEqual(self.seen[0].url.query, b"sig=1")
        self.assertEqual(self.seen[0].headers["host"], "public.example")
        self.assertTrue(self.backend.calls[0]["vad"])

    def test_refusals_are_400_and_fetch_nothing(self):
        urls = [
            "file:///etc/passwd",
            "ftp://public.example/a.wav",
            "gopher://public.example/",
            "http://127.0.0.1:6333/collections",
            "http://[::1]/a.wav",
            "http://[::ffff:127.0.0.1]/a.wav",
            "http://169.254.169.254/latest/meta-data/",
            "http://100.64.0.1/a.wav",
            "http://192.168.1.198/a.wav",
            "http://[fe80::1]/a.wav",
            "http://224.0.0.1/a.wav",
            "http://0.0.0.0/a.wav",
            "http://internal.example/a.wav",  # DNS resolving to a private address
            "http://mixed.example/a.wav",  # one public and one loopback address
            "http://nxdomain.example/a.wav",
            "http://user:pw@public.example/a.wav",
            "http://public.example:99999/a.wav",
            "http:///nohost",
        ]
        for url in urls:
            r = self.url(url)
            self.assertEqual(r.status_code, 400, url)
            self.assertEqual(r.json(), {"error": stt_server.AUDIO_URL_REJECTED}, url)
        self.assertEqual(self.seen, [])
        self.assert_nothing_loaded()

    def test_allowlisted_host_may_be_private_but_only_on_its_port(self):
        with mock.patch.object(stt_server, "AUDIO_URL_ALLOWED_HOSTS", ("minio:9000", "internal.example")):
            self.assertEqual(self.url("http://minio:9000/voice/a.wav").status_code, 200)
            self.assertEqual(self.url("http://internal.example/a.wav").status_code, 200)
            self.assertEqual(self.url("http://minio:6333/").status_code, 400)
        self.assertEqual([r.url.host for r in self.seen], ["172.18.0.4", "10.0.0.5"])  # still pinned

    def test_a_redirect_is_not_followed_by_default(self):
        self.upstream = lambda r: httpx.Response(302, headers={"Location": "http://127.0.0.1/"})
        r = self.url("http://public.example/a.wav")
        self.assertEqual(r.status_code, 502)
        self.assertEqual(r.json(), {"error": stt_server.AUDIO_URL_FAILED})
        self.assertEqual(len(self.seen), 1)

    def test_a_followed_redirect_is_checked_again(self):
        def upstream(r):
            if r.headers["host"] == "public.example":
                return httpx.Response(302, headers={"Location": self.target})
            return httpx.Response(200, content=wav())

        self.upstream = upstream
        with mock.patch.object(stt_server, "AUDIO_URL_MAX_REDIRECTS", 2):
            for self.target in ("http://127.0.0.1/", "http://internal.example/a.wav", "http://169.254.169.254/"):
                self.seen.clear()
                r = self.url("http://public.example/a.wav")
                self.assertEqual(r.status_code, 400, self.target)
                self.assertEqual(len(self.seen), 1, self.target)  # the private hop was never requested
            self.target = "http://other.example/b.wav"
            self.seen.clear()
            self.assertEqual(self.url("http://public.example/a.wav").status_code, 200)
            self.assertEqual([r.url.host for r in self.seen], ["93.184.216.34", "93.184.216.35"])

    def test_upstream_failures_are_a_generic_502(self):
        def refused(request):
            raise httpx.ConnectError("[Errno 111] Connection refused to 10.0.0.9:22")

        for upstream in (lambda r: httpx.Response(404), lambda r: httpx.Response(500), refused):
            self.upstream = upstream
            r = self.url("http://public.example/a.wav")
            self.assertEqual(r.status_code, 502)
            self.assertEqual(r.json(), {"error": stt_server.AUDIO_URL_FAILED})
        self.assert_nothing_loaded()

    def test_falls_back_across_resolved_addresses(self):
        def upstream(request):
            if request.url.host.startswith("2606:"):
                raise httpx.ConnectError("Network is unreachable")
            return httpx.Response(200, content=wav())

        self.upstream = upstream
        self.assertEqual(self.url("http://dual.example/a.wav").status_code, 200)
        self.assertEqual([r.url.host for r in self.seen], ["2606:2800:220:1:248:1893:25c8:1946", "93.184.216.34"])
        self.assertTrue(all(r.headers["host"] == "dual.example" for r in self.seen))

    def test_over_the_cap_by_content_length_is_413(self):
        self.upstream = lambda r: httpx.Response(200, content=b"x" * 5000)
        with mock.patch.object(stt_server, "MAX_UPLOAD_BYTES", 1000):
            self.assertEqual(self.url("http://public.example/a.wav").status_code, 413)
        self.assert_nothing_loaded()

    def test_streamed_past_the_cap_is_413_and_stops_reading(self):
        pulled = []

        async def chunks():
            for _ in range(100):
                pulled.append(1)
                yield b"x" * 500

        self.upstream = lambda r: httpx.Response(200, content=chunks())
        with mock.patch.object(stt_server, "MAX_UPLOAD_BYTES", 1000):
            self.assertEqual(self.url("http://public.example/a.wav").status_code, 413)
        self.assertLess(len(pulled), 100)
        self.assert_nothing_loaded()

    def test_at_the_cap_is_fetched(self):
        clip = wav(seconds=0.25)
        self.upstream = lambda r: httpx.Response(200, content=clip)
        with mock.patch.object(stt_server, "MAX_UPLOAD_BYTES", len(clip)):
            self.assertEqual(self.url("http://public.example/a.wav").status_code, 200)

    def test_a_fetched_non_audio_body_is_400(self):
        self.upstream = lambda r: httpx.Response(200, content=b"<html>not audio</html>")
        with mock.patch.object(stt_server, "INPUT_FORMATS", ""):
            self.assertEqual(self.url("http://public.example/a.wav").status_code, 400)
        self.assert_nothing_loaded()


class CortexReasonsTest(TranscriptionsTest):
    """fetch_audio_url tells a failed fetch (502) and an oversized one (413) from a refused URL (400) by the start of
    argus-cortex's UrlRefused message, which has no structured reason. Each case here makes the real afetch raise
    one of those messages, so a cortex release that rewords one fails this test instead of turning 502s into 400s."""

    def reason_and_status(self, upstream, **patches):
        self.upstream = upstream
        log = mock.Mock()
        multiple = mock.patch.multiple(stt_server, **patches) if patches else contextlib.nullcontext()
        with mock.patch.object(stt_server, "logger", log), multiple:
            r = self.url("http://public.example/a.wav")
        reasons = [c.kwargs["reason"] for c in log.warning.call_args_list if c.args[0] == "stt_audio_url_refused"]
        self.assertEqual(len(reasons), 1, log.warning.call_args_list)
        return reasons[0], r

    @staticmethod
    def streamed(*chunks, **headers):
        async def body():
            for chunk in chunks:
                yield chunk

        return lambda r: httpx.Response(200, headers=headers, content=body())

    def test_every_upstream_failure_reason_is_still_a_502(self):
        def read_timeout(request):
            raise httpx.ReadTimeout("timed out")

        def connect_refused(request):
            raise httpx.ConnectError("refused")

        def slow(request):
            time.sleep(0.05)
            return httpx.Response(200, content=wav())

        redirect = lambda r: httpx.Response(302, headers={"Location": "http://public.example/b.wav"})  # noqa: E731
        cases = [
            ("could not fetch", read_timeout, {}),
            ("could not connect", connect_refused, {}),
            ("the fetch took longer", slow, {"DOWNLOAD_TIMEOUT": 0.01}),
            ("more than", redirect, {"AUDIO_URL_MAX_REDIRECTS": 1}),
            ("the response has an unsupported", self.streamed(b"x", **{"Content-Encoding": "br"}), {}),
            ("the response body could not", self.streamed(b"not gzip", **{"Content-Encoding": "gzip"}), {}),
        ]
        # "a redirect without" (a Location header) can't be reached through httpx, whose is_redirect needs one.
        triggered = {"a redirect without"}
        for prefix, upstream, patches in cases:
            with self.subTest(prefix):
                reason, r = self.reason_and_status(upstream, **patches)
                self.assertTrue(reason.startswith(prefix), reason)
                self.assertEqual(r.status_code, 502, reason)
                self.assertEqual(r.json(), {"error": stt_server.AUDIO_URL_FAILED})
                triggered.add(prefix)
        self.assertEqual(triggered, set(stt_server._UPSTREAM_FAILURES))
        self.assert_nothing_loaded()

    def test_the_size_cap_reason_is_still_a_413(self):
        reason, r = self.reason_and_status(self.streamed(b"x" * 600, b"x" * 600), MAX_UPLOAD_BYTES=1000)
        self.assertTrue(reason.startswith("the response is larger than"), reason)
        self.assertEqual(r.status_code, 413)

    def test_a_refused_url_is_still_a_400(self):
        log = mock.Mock()
        with mock.patch.object(stt_server, "logger", log):
            r = self.url("http://internal.example/a.wav")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.seen, [])


class IntakeTest(TranscriptionsTest):
    """At most STT_MAX_DECODES requests fetch and decode at once; the rest wait, then 503."""

    def test_a_full_intake_is_503_and_fetches_nothing(self):
        for send in (lambda: self.url("http://public.example/a.wav"), lambda: self.tx(audio_base64=b64(wav()))):
            # A fresh semaphore each time: one that made a request wait is bound to that request's event loop.
            with mock.patch.object(stt_server, "_intake", asyncio.Semaphore(0)), mock.patch.object(
                stt_server, "QUEUE_TIMEOUT_S", 0.01
            ):
                r = send()
            self.assertEqual(r.status_code, 503)
        self.assertIn("Retry-After", r.headers)
        self.assertIn("decode", r.json()["error"])
        self.assertEqual(self.seen, [])
        self.assert_nothing_loaded()

    def test_the_slot_is_given_back_however_the_request_ends(self):
        self.upstream = lambda r: httpx.Response(404)
        with mock.patch.object(stt_server, "_intake", asyncio.Semaphore(1)), mock.patch.object(
            stt_server, "QUEUE_TIMEOUT_S", 0.5
        ):
            self.assertEqual(self.url("http://public.example/a.wav").status_code, 502)
            self.assertEqual(self.tx(audio_base64="!!!").status_code, 400)
            with mock.patch.object(stt_server, "MAX_SECONDS", 0.1):
                self.assertEqual(self.tx(audio_base64=b64(wav())).status_code, 413)
            self.assertEqual(self.tx(audio_base64=b64(wav())).status_code, 200)


class WavFallbackTest(TranscriptionsTest):
    """A WAV or FLAC that open_audio refuses (a rate outside ALLOWED_RATES) still gets ffmpeg's try, as
    agent-composer's sidecar decoded it. ADPCM, a-law and mu-law WAVs open_audio reads itself."""

    ODD_RATE = wav(rate=12000)

    def test_an_odd_rate_wav_goes_to_ffmpeg(self):
        self.assertNotIn(12000, stt_server.ALLOWED_RATES)
        with mock.patch.object(stt_server, "ffmpeg_decode", return_value=np.zeros(16000, np.float32)) as dec:
            r = self.tx(audio_base64=b64(self.ODD_RATE))
        self.assertEqual(r.status_code, 200, r.text)
        dec.assert_called_once()

    def test_not_with_formats_none(self):
        with mock.patch.object(stt_server, "INPUT_FORMATS", ""), mock.patch.object(stt_server, "ffmpeg_decode") as dec:
            self.assertEqual(self.tx(audio_base64=b64(self.ODD_RATE)).status_code, 400)
        dec.assert_not_called()

    def test_not_past_the_duration_cap(self):
        with mock.patch.object(stt_server, "MAX_SECONDS", 0.5), mock.patch.object(stt_server, "ffmpeg_decode") as dec:
            self.assertEqual(self.tx(audio_base64=b64(wav(seconds=1.0))).status_code, 413)
        dec.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_an_odd_rate_wav_is_decoded_by_the_real_ffmpeg(self):
        r = self.tx(audio_base64=b64(self.ODD_RATE))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.backend.calls[0]["rate"], 16000)
        self.assertAlmostEqual(self.backend.calls[0]["n"], 16000, delta=1600)


class WhisperBackendTest(unittest.TestCase):
    def test_vad_filter_reaches_faster_whisper(self):
        seen = {}

        class Model:
            def transcribe(self, audio, **kw):
                seen.update(kw)
                return iter(()), mock.Mock(language="en", language_probability=0.5)

        b = stt_server.WhisperBackend()
        b._model = Model()
        b.transcribe(np.zeros(16000, dtype=np.float32), 16000, None, False, True)
        self.assertTrue(seen["vad_filter"])
        b.transcribe(np.zeros(16000, dtype=np.float32), 16000, None, False)
        self.assertFalse(seen["vad_filter"])


if __name__ == "__main__":
    unittest.main()
