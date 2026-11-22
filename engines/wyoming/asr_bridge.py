"""Wyoming ASR bridge -> argus-echo's stt contract (the whisper engine).

Speaks the Wyoming protocol on a TCP port, so Home Assistant's voice pipeline (or
any Wyoming client) can use echo's speech-to-text engine as an ``asr`` service:

  client: Transcribe (optional language) -> AudioStart -> AudioChunk... -> AudioStop
  bridge: Transcript (the text)

The bridge buffers the audio until ``AudioStop``, converts it to 16 kHz mono
signed 16-bit (Whisper's rate; most clients already send exactly that), wraps it
in one WAV and posts it to the engine's ``POST /transcribe`` with the language
from ``Transcribe`` (or ``WYOMING_ASR_LANGUAGE``; neither means detect). The
client does the endpointing (Home Assistant's own VAD), so the engine's
streaming route is not needed here.

A failure (engine down, busy, a rejected clip) is logged and answered with an
empty ``Transcript``, which a client treats as "nothing heard", rather than a
dropped connection.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import wave
from functools import partial
from typing import Optional

import httpx
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger("wyoming_asr")

STT_URL = (os.getenv("STT_URL") or "http://whisper:8031").rstrip("/")
DEFAULT_LANGUAGE = os.getenv("WYOMING_ASR_LANGUAGE", "").strip() or None
PROGRAM_NAME = os.getenv("WYOMING_ASR_PROGRAM_NAME", "whisper").strip() or "whisper"
PROGRAM_DESC = os.getenv("WYOMING_ASR_PROGRAM_DESC", "Whisper large-v3-turbo (faster-whisper) on argus-echo")
MODEL_NAME = os.getenv("WYOMING_ASR_MODEL_NAME", "large-v3-turbo").strip() or "large-v3-turbo"
ATTRIBUTION_NAME = os.getenv("WYOMING_ASR_ATTRIBUTION_NAME", "OpenAI Whisper (MIT), via faster-whisper (MIT)")
ATTRIBUTION_URL = os.getenv("WYOMING_ASR_ATTRIBUTION_URL", "https://github.com/SYSTRAN/faster-whisper")

# The engine queues for a slot (STT_QUEUE_TIMEOUT_S), may load the model, then transcribes:
# a generous read timeout, but a short connect so a down engine is reported at once.
REQUEST_TIMEOUT = float(os.getenv("WYOMING_ASR_TIMEOUT", "120"))
CONNECT_TIMEOUT = float(os.getenv("WYOMING_ASR_CONNECT_TIMEOUT", "5"))
# The engine's own duration cap is STT_MAX_SECONDS (600 by default); stop buffering past ours.
MAX_SECONDS = float(os.getenv("WYOMING_ASR_MAX_SECONDS", "300"))

RATE = 16000
WIDTH = 2
CHANNELS = 1
MAX_BYTES = int(MAX_SECONDS * RATE * WIDTH * CHANNELS)

# Whisper's languages, as the whisper engine accepts them (kept in step with its LANGUAGES by hand:
# an engine must not import another's code).
LANGUAGES = (
    "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha haw he hi hr ht hu hy "
    "id is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru "
    "sa sd si sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo zh yue"
).split()


def _attribution() -> Attribution:
    return Attribution(name=ATTRIBUTION_NAME, url=ATTRIBUTION_URL)


def build_info() -> Info:
    return Info(
        asr=[
            AsrProgram(
                name=PROGRAM_NAME,
                description=PROGRAM_DESC,
                attribution=_attribution(),
                installed=True,
                version=None,
                models=[
                    AsrModel(
                        name=MODEL_NAME,
                        description=f"Whisper {MODEL_NAME}",
                        attribution=_attribution(),
                        installed=True,
                        version=None,
                        languages=list(LANGUAGES),
                    )
                ],
                supports_transcript_streaming=False,
            )
        ]
    )


def to_wav(pcm: bytes) -> bytes:
    """16 kHz mono s16le PCM -> a WAV file."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(WIDTH)
        wav.setframerate(RATE)
        wav.writeframes(pcm)
    return out.getvalue()


class SttError(RuntimeError):
    pass


def normalise_language(raw: Optional[str]) -> Optional[str]:
    """A client's language as Whisper knows it (``en-US`` -> ``en``), or None (detect) for one it doesn't.

    The engine would refuse an unknown code with a 422; detecting is the better answer to a voice
    pipeline configured for a language Whisper lacks."""
    if raw is None or not raw.strip():
        return None
    code = raw.strip().lower().replace("_", "-")
    if code not in LANGUAGES:
        code = code.split("-")[0]
    if code not in LANGUAGES:
        _LOGGER.warning("language %r is not one Whisper knows: detecting instead", raw.strip()[:16])
        return None
    return code


async def transcribe(pcm: bytes, language: Optional[str]) -> tuple[str, Optional[str]]:
    """POST one clip to the engine's /transcribe; return (text, language). Raises SttError on any failure."""
    data = {"language": language} if language else {}
    files = {"audio": ("audio.wav", to_wav(pcm), "audio/wav")}
    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{STT_URL}/transcribe", data=data, files=files)
    except httpx.TimeoutException as exc:
        raise SttError(f"{STT_URL} timed out ({type(exc).__name__})") from exc
    except httpx.HTTPError as exc:
        raise SttError(f"{STT_URL} unreachable: {exc}") from exc
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error")
        except Exception:  # noqa: BLE001 — not JSON
            detail = None
        retry = resp.headers.get("retry-after")
        raise SttError(
            f"{STT_URL}/transcribe answered {resp.status_code}: {detail or resp.text[:200]!r}"
            + (f" (retry after {retry}s)" if retry else "")
        )
    try:
        body = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise SttError(f"{STT_URL}/transcribe sent a body that is not JSON") from exc
    text = body.get("text") if isinstance(body, dict) else None
    if not isinstance(text, str):
        raise SttError(f"{STT_URL}/transcribe sent no text")
    detected = body.get("language")
    return text.strip(), detected if isinstance(detected, str) and detected else language


class AsrEventHandler(AsyncEventHandler):
    """Handles one Wyoming client connection: any number of Transcribe/audio rounds."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reset()
        self._language: Optional[str] = None

    def _reset(self) -> None:
        self._converter = AudioChunkConverter(rate=RATE, width=WIDTH, channels=CHANNELS)
        self._audio = bytearray()
        self._overflow = False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(build_info().event())
            return True

        if Transcribe.is_type(event.type):
            self._language = Transcribe.from_event(event).language or None
            return True

        if AudioStart.is_type(event.type):
            self._reset()
            return True

        if AudioChunk.is_type(event.type):
            if self._overflow:
                return True
            chunk = self._converter.convert(AudioChunk.from_event(event))
            if len(self._audio) + len(chunk.audio) > MAX_BYTES:
                _LOGGER.warning("audio over %.0f s: the rest of this utterance is dropped", MAX_SECONDS)
                self._overflow = True
                return True
            self._audio.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            await self._finish()
            return True

        return True

    async def _finish(self) -> None:
        pcm = bytes(self._audio)
        language = normalise_language(self._language or DEFAULT_LANGUAGE)
        self._reset()
        self._language = None  # a Transcribe applies to the one utterance that follows it
        text, heard = "", language
        if pcm:
            try:
                text, heard = await transcribe(pcm, language)
            except SttError as exc:
                _LOGGER.error("transcription failed: %s", exc)
        else:
            _LOGGER.warning("AudioStop with no audio: empty transcript")
        _LOGGER.debug("transcript (%s): %r", heard or "detect", text)
        await self.write_event(Transcript(text=text, language=heard).event())


async def main() -> None:
    parser = argparse.ArgumentParser(description="Wyoming ASR bridge for argus-echo's stt engine")
    parser.add_argument("--uri", default=os.getenv("WYOMING_URI", "tcp://0.0.0.0:10300"))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    _LOGGER.info("Wyoming %s ASR bridge listening on %s -> %s", PROGRAM_NAME, args.uri, STT_URL)

    server = AsyncServer.from_uri(args.uri)
    await server.run(partial(AsrEventHandler))


if __name__ == "__main__":
    asyncio.run(main())
