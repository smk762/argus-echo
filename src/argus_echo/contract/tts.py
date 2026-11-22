"""The ``tts`` contract: what every text-to-speech engine speaks.

Each engine (Miso, XTTS-v2, Qwen3-TTS) is its own server with its own Python environment, so they share this
contract rather than code. A client written against it can switch engines by changing a URL. The full reference,
with every route and status code, is ``docs/api.md``.

- ``POST /synthesise`` takes a :class:`SynthesiseRequest` and returns ``audio_url`` (a WAV in the engine's storage).
- ``POST /synthesise/stream`` takes the same body and streams raw PCM. The format is in the response headers
  (:class:`StreamFormat`): 24 kHz mono little-endian float32 on every engine today.
- ``GET /speakers``, ``GET /clones``, ``POST /clone-voice`` and the ``/clones/*`` routes manage voices.
- ``GET /health`` is argus-cortex's :class:`~argus_cortex.gpu_wire.ResidentHealth`, plus engine fields
  (``engine``, ``model``, ``device``), the image identity (:class:`ImageIdentity`, as ``image``) and, on engines
  that load checkpoints, the loaded weights (:class:`Weights`, as ``weights``). ``POST /admin/{unload,evict,wake}``
  and ``POST /admin/load`` (:class:`LoadCheckpointRequest`; XTTS and Qwen) take
  ``Authorization: Bearer $ARGUS_ADMIN_TOKEN`` and are closed (503) while it is unset.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SAMPLE_RATE_HEADER = "X-Sample-Rate"
AUDIO_FORMAT_HEADER = "X-Audio-Format"
CHANNELS_HEADER = "X-Channels"

#: Bytes per sample for each ``X-Audio-Format`` an engine may send.
SAMPLE_WIDTH = {"pcm_f32le": 4, "pcm_s16le": 2}

#: The largest ``seed`` (numpy's legacy seed range, so every engine can seed every RNG it uses).
SEED_MAX = 2**32 - 1

#: Loads a fine-tuned checkpoint (XTTS, Qwen); see :class:`LoadCheckpointRequest`.
LOAD_PATH = "/admin/load"


class SynthesiseRequest(BaseModel):
    """The body of ``POST /synthesise`` and ``POST /synthesise/stream``.

    These fields are accepted by every engine. An engine ignores the ones it has no use for rather than refusing
    them (Miso and XTTS ignore ``voice_description``, ``instruction`` and ``emotion``; Miso and Qwen ignore
    ``device``), so one request works everywhere. Engine-specific fields (Miso's ``temperature``, ``topk`` and
    ``context``) pass through as extras.
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
    device: Literal["cpu"] | None = Field(
        None,
        description=(
            'XTTS: "cpu" runs without promoting the model to the GPU (a client sends it when argus-iris refuses its '
            "GPU lease). Miso and Qwen ignore it."
        ),
    )
    seed: int | None = Field(
        None,
        ge=0,
        le=SEED_MAX,
        description=(
            "Seeds the sampler (python, numpy, torch CPU and CUDA) for this request, on every engine. What it makes "
            "reproducible differs by engine and route: see docs/api.md."
        ),
    )


class LoadCheckpointRequest(BaseModel):
    """The body of ``POST /admin/load`` (XTTS, Qwen): a checkpoint directory under the engine's checkpoint root
    (``XTTS_CHECKPOINT_ROOT``, ``TTS_CHECKPOINT_ROOT``), or ``None`` for the stock weights."""

    checkpoint: str | None


class WeightFile(BaseModel):
    """One file the loaded weights were read from, hashed once when they were loaded."""

    model_config = ConfigDict(extra="allow")

    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int | None = Field(None, ge=0)


class Weights(BaseModel):
    """``GET /health``'s ``weights``: the selected checkpoint (``None`` = stock) and the files its last load read.
    ``files`` is ``None`` until those weights have been loaded."""

    checkpoint: str | None = None
    files: list[WeightFile] | None = None


class ImageIdentity(BaseModel):
    """``GET /health``'s ``image``: the release and git revision baked into the engine image (``None`` for a local
    build). A running container can't know its own digest; whoever deployed it can."""

    version: str | None = None
    revision: str | None = None


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
