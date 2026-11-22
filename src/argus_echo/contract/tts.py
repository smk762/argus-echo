"""The ``tts`` contract: what every text-to-speech engine speaks.

Each engine (Miso, XTTS-v2, Qwen3-TTS) is its own server with its own Python environment, so they share this
contract rather than code. A client written against it can switch engines by changing a URL. The full reference,
with every route and status code, is ``docs/api.md``.

- ``POST /synthesise`` takes a :class:`SynthesiseRequest` and returns ``audio_url`` (a WAV in the engine's storage).
- ``POST /synthesise/stream`` takes the same body and streams raw PCM. The format is in the response headers
  (:class:`StreamFormat`): 24 kHz mono little-endian float32 on every engine today.
- ``GET /speakers``, ``GET /clones``, ``POST /clone-voice`` and the ``/clones/*`` routes manage voices.
- ``GET /health`` is argus-cortex's :class:`~argus_cortex.gpu_wire.ResidentHealth`, plus engine fields
  (``engine``, ``model``, ``device``). ``POST /admin/{unload,evict,wake}`` take
  ``Authorization: Bearer $ARGUS_ADMIN_TOKEN`` and are closed (503) while it is unset.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

SAMPLE_RATE_HEADER = "X-Sample-Rate"
AUDIO_FORMAT_HEADER = "X-Audio-Format"
CHANNELS_HEADER = "X-Channels"

#: Bytes per sample for each ``X-Audio-Format`` an engine may send.
SAMPLE_WIDTH = {"pcm_f32le": 4, "pcm_s16le": 2}


class SynthesiseRequest(BaseModel):
    """The body of ``POST /synthesise`` and ``POST /synthesise/stream``.

    These fields are accepted by every engine. An engine ignores the ones it has no use for rather than refusing
    them (Miso and XTTS ignore ``voice_description``, ``instruction`` and ``emotion``), so one request works
    everywhere. Engine-specific fields (Miso's ``temperature``, ``topk`` and ``context``) pass through as extras.
    """

    model_config = ConfigDict(extra="allow")

    text: str = Field(min_length=1, description="What to say.")
    voice_clone_id: str | None = Field(None, description="A stored clone (``GET /clones``); wins over ``speaker``.")
    speaker: str | None = Field(None, description="A built-in speaker (``GET /speakers``).")
    language: str = Field("en", description="ISO language code. Miso speaks English only.")
    voice_description: str | None = Field(None, description="Qwen3-TTS: describe a voice to design.")
    instruction: str | None = Field(None, description="Qwen3-TTS: delivery instruction for a built-in speaker.")
    emotion: str = Field("neutral", description="Qwen3-TTS: delivery hint.")
    speed: float | None = Field(None, description="Speaking-rate multiplier. XTTS applies it; Miso and Qwen ignore it.")


class StreamFormat(BaseModel):
    """The PCM layout of a ``/synthesise/stream`` response, read from its headers."""

    model_config = ConfigDict(frozen=True)

    sample_rate: int = Field(gt=0)
    audio_format: str
    channels: int = Field(gt=0)

    @property
    def sample_width(self) -> int:
        """Bytes per sample. Raises ``KeyError`` for a format this package does not know."""
        return SAMPLE_WIDTH[self.audio_format]

    def seconds(self, n_bytes: int) -> float:
        """Duration of *n_bytes* of audio in this format."""
        return n_bytes / (self.sample_width * self.channels * self.sample_rate)

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> StreamFormat:
        """Parse the stream headers. Raises ``ValueError`` if one is missing or malformed."""
        lower = {k.lower(): v for k, v in headers.items()}
        try:
            return cls(
                sample_rate=int(lower[SAMPLE_RATE_HEADER.lower()]),
                audio_format=lower[AUDIO_FORMAT_HEADER.lower()],
                channels=int(lower[CHANNELS_HEADER.lower()]),
            )
        except KeyError as exc:
            raise ValueError(f"missing stream header {exc.args[0]!r}") from None
