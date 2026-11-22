"""The ``stt`` contract: what a speech-to-text engine speaks.

The Whisper engine (``engines/whisper/``) serves it. The full reference, with every field and status code, is the
"Speech to text" section of ``docs/api.md``.

- ``POST /transcribe`` is ``multipart/form-data``: one ``audio`` file (WAV or FLAC) plus the optional
  :class:`TranscribeOptions` fields. It answers a :class:`Transcription`.
- ``WS /transcribe/stream`` takes raw PCM (:data:`STREAM_SAMPLE_RATE`, :data:`STREAM_AUDIO_FORMAT`, mono) as
  binary frames, cuts it into utterances with voice activity detection, and sends :class:`StreamPartial` while an
  utterance is open and a :class:`StreamFinal` when it ends. The client may open with a :class:`StreamStart` and
  sends :class:`StreamEnd` to flush the last utterance.
- ``POST /transcriptions`` is agent-composer's JSON route, kept so kimini-api moves to echo unchanged: a
  :class:`TranscriptionsRequest` (``audio_url`` fetched by the engine under an SSRF policy, or ``audio_base64``)
  answered by a :class:`TranscriptionsResult`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TRANSCRIBE_PATH = "/transcribe"
STREAM_PATH = "/transcribe/stream"
TRANSCRIPTIONS_PATH = "/transcriptions"

#: The PCM layout ``/transcribe/stream`` takes: 16 kHz mono little-endian int16, the rate Whisper runs at.
STREAM_SAMPLE_RATE = 16000
STREAM_AUDIO_FORMAT = "pcm_s16le"
STREAM_CHANNELS = 1


class TranscribeOptions(BaseModel):
    """The form fields of ``POST /transcribe`` beside the ``audio`` file (and the options of a stream)."""

    model_config = ConfigDict(extra="forbid")

    language: str | None = Field(
        None,
        description="ISO 639-1 code of the speech (``en``; a region suffix such as ``en-US`` is dropped). "
        "Detected from the audio when absent.",
    )
    word_timestamps: bool = Field(False, description="Also time each word (``Segment.words``).")


class Word(BaseModel):
    """One word of a segment, with ``word_timestamps``."""

    start: float = Field(ge=0, description="Seconds from the start of the audio.")
    end: float = Field(ge=0)
    word: str = Field(description="The word as transcribed, with its leading space if it had one.")
    probability: float = Field(ge=0, le=1)


class Segment(BaseModel):
    """A stretch of the transcript with its time span."""

    start: float = Field(ge=0, description="Seconds from the start of the audio.")
    end: float = Field(ge=0)
    text: str
    words: list[Word] | None = Field(None, description="Present only when ``word_timestamps`` was asked for.")


class Transcription(BaseModel):
    """The answer of ``POST /transcribe``."""

    text: str = Field(description="The whole transcript: the segments' text joined, stripped. Empty for silence.")
    language: str = Field(description="The language used: the one asked for, or the detected one.")
    language_probability: float | None = Field(
        None, ge=0, le=1, description="Confidence of the detection; null when the language was given."
    )
    duration: float = Field(ge=0, description="Length of the audio in seconds.")
    segments: list[Segment] = Field(default_factory=list)


# ── POST /transcriptions (agent-composer's route) ────────────────────────────


class TranscriptionsRequest(BaseModel):
    """The JSON body of ``POST /transcriptions``: exactly one of ``audio_url`` and ``audio_base64``.

    The engine ignores unknown fields, as composer's sidecar did. Both given is a ``422``, neither a ``400``."""

    model_config = ConfigDict(extra="ignore")

    audio_url: str | None = Field(
        None,
        description="An http(s) URL the engine downloads. Every address its host resolves to must be public "
        "unless the host is in ``STT_AUDIO_URL_ALLOWED_HOSTS``; redirects are not followed by default.",
    )
    audio_base64: str | None = Field(
        None, description="The audio, standard base64; a ``data:...;base64,`` prefix is accepted."
    )
    language: str | None = Field(
        None, description="As :attr:`TranscribeOptions.language` (``en-US`` becomes ``en``); detected when absent."
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> TranscriptionsRequest:
        if bool(self.audio_url) == bool(self.audio_base64):
            raise ValueError("give exactly one of audio_url and audio_base64")
        return self


class TranscriptionsResult(BaseModel):
    """The answer of ``POST /transcriptions`` (composer's shape, not :class:`Transcription`)."""

    text: str = Field(description="The whole transcript. Empty for silence.")
    language: str = Field(description="The language used: the one asked for, or the detected one.")
    duration_seconds: float = Field(ge=0, description="Length of the audio in seconds.")
    confidence: float = Field(
        ge=0, le=1, description="Whisper's language-detection probability; 1.0 when the language was given."
    )


# ── Streaming messages (JSON text frames) ────────────────────────────────────


class StreamStart(BaseModel):
    """Client, optional, before any audio: the stream's options."""

    type: Literal["start"] = "start"
    language: str | None = None
    word_timestamps: bool = False


class StreamEnd(BaseModel):
    """Client: no more audio. The server finishes the open utterance, sends its final and closes."""

    type: Literal["end"] = "end"


class StreamReady(BaseModel):
    """Server, first message: the PCM layout it expects and its endpointing settings."""

    type: Literal["ready"] = "ready"
    sample_rate: int = STREAM_SAMPLE_RATE
    audio_format: str = STREAM_AUDIO_FORMAT
    channels: int = STREAM_CHANNELS
    min_silence_ms: int = Field(description="Silence that ends an utterance.")
    max_utterance_s: float = Field(description="An utterance this long is ended even without silence.")


class StreamPartial(BaseModel):
    """Server: the open utterance transcribed so far. Later partials replace earlier ones."""

    type: Literal["partial"] = "partial"
    start: float = Field(ge=0, description="Seconds from the start of the stream at which the utterance began.")
    text: str


class StreamFinal(Transcription):
    """Server: one finished utterance. Segment and word times are seconds from the start of the stream."""

    type: Literal["final"] = "final"
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class StreamError(BaseModel):
    """Server: the stream failed; the socket closes after it."""

    type: Literal["error"] = "error"
    error: str
