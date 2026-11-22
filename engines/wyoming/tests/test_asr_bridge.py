"""asr_bridge: Wyoming Transcribe + audio in, one POST /transcribe out, a Transcript back.
The whisper engine is an httpx.MockTransport; nothing listens.

Run from engines/wyoming/: python -m pytest tests -q
"""

import io
import json
import os
import sys
import unittest
import wave
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asr_bridge  # noqa: E402
import httpx  # noqa: E402
from wyoming.asr import Transcribe, Transcript  # noqa: E402
from wyoming.audio import AudioChunk, AudioStart, AudioStop  # noqa: E402
from wyoming.info import Describe, Info  # noqa: E402

_RealAsyncClient = httpx.AsyncClient


def _engine(handler, seen: list):
    """Patch httpx.AsyncClient so every client the bridge opens talks to ``handler``."""

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def client(**kwargs):
        kwargs["transport"] = httpx.MockTransport(record)
        return _RealAsyncClient(**kwargs)

    return mock.patch.object(asr_bridge.httpx, "AsyncClient", client)


def _ok(text="hello there", language="en"):
    return lambda request: httpx.Response(200, json={"text": f" {text} ", "language": language, "segments": []})


def _form(request: httpx.Request) -> tuple[dict, bytes]:
    """Split a multipart request into its text fields and the audio file's bytes."""
    ctype = request.headers["content-type"]
    boundary = ctype.split("boundary=")[1].encode()
    fields, audio = {}, b""
    for part in request.content.split(b"--" + boundary):
        if b"\r\n\r\n" not in part:
            continue
        head, body = part.split(b"\r\n\r\n", 1)
        body = body[:-2] if body.endswith(b"\r\n") else body
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        if name == "audio":
            audio = body
        else:
            fields[name] = body.decode()
    return fields, audio


def _handler():
    handler = asr_bridge.AsrEventHandler.__new__(asr_bridge.AsrEventHandler)
    handler._reset()
    handler._language = None
    handler.write_event = mock.AsyncMock()
    return handler


def _written(handler):
    return [c.args[0] for c in handler.write_event.call_args_list]


async def _utterance(handler, audio: bytes, rate=16000, width=2, channels=1, language=None, chunk=3200):
    await handler.handle_event(Transcribe(language=language).event())
    await handler.handle_event(AudioStart(rate=rate, width=width, channels=channels).event())
    for i in range(0, len(audio), chunk):
        await handler.handle_event(
            AudioChunk(rate=rate, width=width, channels=channels, audio=audio[i : i + chunk]).event()
        )
    await handler.handle_event(AudioStop().event())


class TranscribeTests(unittest.IsolatedAsyncioTestCase):
    async def test_buffers_to_audio_stop_and_posts_one_wav(self):
        seen, handler = [], _handler()
        pcm = b"\x01\x00" * 16000  # one second
        with _engine(_ok(), seen):
            await _utterance(handler, pcm, language="en")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].url.path, "/transcribe")
        fields, audio = _form(seen[0])
        self.assertEqual(fields, {"language": "en"})
        with wave.open(io.BytesIO(audio)) as wav:
            self.assertEqual((wav.getframerate(), wav.getsampwidth(), wav.getnchannels()), (16000, 2, 1))
            self.assertEqual(wav.readframes(wav.getnframes()), pcm)
        events = _written(handler)
        self.assertEqual(len(events), 1)
        self.assertTrue(Transcript.is_type(events[0].type))
        transcript = Transcript.from_event(events[0])
        self.assertEqual((transcript.text, transcript.language), ("hello there", "en"))

    async def test_other_formats_are_converted_to_16k_mono_s16(self):
        seen, handler = [], _handler()
        stereo_48k = b"\x10\x00\x10\x00" * 48000  # one second, 48 kHz stereo s16
        with _engine(_ok(), seen):
            await _utterance(handler, stereo_48k, rate=48000, channels=2, chunk=9600)
        _, audio = _form(seen[0])
        with wave.open(io.BytesIO(audio)) as wav:
            self.assertEqual((wav.getframerate(), wav.getsampwidth(), wav.getnchannels()), (16000, 2, 1))
            self.assertAlmostEqual(wav.getnframes(), 16000, delta=16)

    async def test_no_language_means_detect_and_reports_what_was_heard(self):
        seen, handler = [], _handler()
        with _engine(_ok("bonjour", "fr"), seen), mock.patch.object(asr_bridge, "DEFAULT_LANGUAGE", None):
            await _utterance(handler, b"\x00\x00" * 1600)
        fields, _ = _form(seen[0])
        self.assertNotIn("language", fields)
        self.assertEqual(Transcript.from_event(_written(handler)[0]).language, "fr")

    async def test_language_is_normalised_and_unknown_ones_are_detected(self):
        self.assertEqual(asr_bridge.normalise_language("en-US"), "en")
        self.assertEqual(asr_bridge.normalise_language("en_GB"), "en")
        self.assertEqual(asr_bridge.normalise_language(" DE "), "de")
        self.assertIsNone(asr_bridge.normalise_language("xx-YY"))
        self.assertIsNone(asr_bridge.normalise_language(""))
        self.assertIsNone(asr_bridge.normalise_language(None))

    async def test_a_transcribe_applies_to_one_utterance(self):
        seen, handler = [], _handler()
        with _engine(_ok(), seen), mock.patch.object(asr_bridge, "DEFAULT_LANGUAGE", None):
            await _utterance(handler, b"\x00\x00" * 1600, language="de")
            await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
            await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00\x00" * 1600).event())
            await handler.handle_event(AudioStop().event())
        self.assertEqual(_form(seen[0])[0], {"language": "de"})
        self.assertEqual(_form(seen[1])[0], {})
        self.assertEqual(len(_form(seen[1])[1]) - 44, 3200)  # the second clip holds only its own audio


class FailureTests(unittest.IsolatedAsyncioTestCase):
    async def _transcript_for(self, engine):
        seen, handler = [], _handler()
        with _engine(engine, seen):
            await _utterance(handler, b"\x00\x00" * 1600, language="en")
        events = _written(handler)
        self.assertEqual(len(events), 1)
        self.assertTrue(Transcript.is_type(events[0].type))
        return Transcript.from_event(events[0]).text

    async def test_engine_errors_give_an_empty_transcript(self):
        for status in (400, 413, 422, 500, 503):
            with self.subTest(status=status), self.assertLogs("wyoming_asr", "ERROR") as logs:
                engine = lambda r, s=status: httpx.Response(s, json={"error": "nope"}, headers={"Retry-After": "5"})  # noqa: E731
                self.assertEqual(await self._transcript_for(engine), "")
            self.assertIn(str(status), logs.output[0])
            self.assertIn("nope", logs.output[0])

    async def test_unreachable_and_timed_out_engines_give_an_empty_transcript(self):
        def refuse(request):
            raise httpx.ConnectError("connection refused", request=request)

        def slow(request):
            raise httpx.ReadTimeout("read timed out", request=request)

        for engine, words in ((refuse, "unreachable"), (slow, "timed out")):
            with self.subTest(words), self.assertLogs("wyoming_asr", "ERROR") as logs:
                self.assertEqual(await self._transcript_for(engine), "")
            self.assertIn(words, logs.output[0])

    async def test_a_body_without_text_gives_an_empty_transcript(self):
        for body in (b"not json", json.dumps({"segments": []}).encode(), b"[]"):
            with self.subTest(body=body), self.assertLogs("wyoming_asr", "ERROR"):
                self.assertEqual(await self._transcript_for(lambda r, b=body: httpx.Response(200, content=b)), "")

    async def test_no_audio_answers_without_calling_the_engine(self):
        seen, handler = [], _handler()
        with _engine(_ok(), seen):
            await handler.handle_event(Transcribe().event())
            await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
            await handler.handle_event(AudioStop().event())
        self.assertEqual(seen, [])
        self.assertEqual(Transcript.from_event(_written(handler)[0]).text, "")

    async def test_audio_past_the_cap_is_dropped(self):
        seen, handler = [], _handler()
        with _engine(_ok(), seen), mock.patch.object(asr_bridge, "MAX_BYTES", 6400):
            await _utterance(handler, b"\x00\x00" * 16000)
        self.assertLessEqual(len(_form(seen[0])[1]) - 44, 6400)

    async def test_timeouts_are_bounded(self):
        captured = {}

        def client(**kwargs):
            captured.update(kwargs)
            kwargs["transport"] = httpx.MockTransport(_ok())
            return _RealAsyncClient(**kwargs)

        with mock.patch.object(asr_bridge.httpx, "AsyncClient", client):
            await asr_bridge.transcribe(b"\x00\x00", None)
        timeout = captured["timeout"]
        self.assertEqual(timeout.connect, asr_bridge.CONNECT_TIMEOUT)
        self.assertEqual(timeout.read, asr_bridge.REQUEST_TIMEOUT)


class DescribeTests(unittest.IsolatedAsyncioTestCase):
    async def test_describe_advertises_an_asr_program(self):
        handler = _handler()
        await handler.handle_event(Describe().event())
        info = Info.from_event(_written(handler)[0])
        self.assertEqual(info.tts, [])
        self.assertEqual(len(info.asr), 1)
        program = info.asr[0]
        self.assertEqual(program.name, "whisper")
        self.assertIn("MIT", program.attribution.name)
        self.assertIn("en", program.models[0].languages)
        self.assertIn("yue", program.models[0].languages)
        self.assertEqual(program.models[0].name, "large-v3-turbo")


if __name__ == "__main__":
    unittest.main()
