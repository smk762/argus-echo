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

import hashlib
from collections.abc import Iterable, Mapping
from typing import Literal
from urllib.parse import unquote

from pydantic import BaseModel, ConfigDict, Field

SAMPLE_RATE_HEADER = "X-Sample-Rate"
AUDIO_FORMAT_HEADER = "X-Audio-Format"
CHANNELS_HEADER = "X-Channels"
#: On a stream (XTTS, Qwen): the checkpoint generating it, percent-encoded; empty for stock weights.
CHECKPOINT_HEADER = "X-Echo-Checkpoint"
#: On a stream (XTTS, Qwen): the fingerprint of the weights generating it (see :func:`weights_fingerprint`).
WEIGHTS_HEADER = "X-Echo-Weights"
#: On a stream (Qwen): the model slot generating it, ``base`` or ``custom_voice``.
MODEL_HEADER = "X-Echo-Model"

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


def weights_fingerprint(files: Iterable[WeightFile | Mapping[str, str]], slot: str | None = None) -> str:
    """The fingerprint of a set of weight files, as the engines compute it (docs/api.md, "Weights fingerprint").

    One line ``<name>:<sha256>`` per file, ``name`` exactly as ``/health`` lists it (Qwen's slot prefix kept), the
    lines sorted byte-wise as UTF-8, each ending in ``\\n`` (the last one too); the sha256 of those bytes, lowercase
    hex. *slot* keeps only the files named ``<slot>/...`` (Qwen's per-model fingerprint)."""
    pairs = [(f.name, f.sha256) if isinstance(f, WeightFile) else (f["name"], f["sha256"]) for f in files]
    if slot is not None:
        pairs = [(n, s) for n, s in pairs if n.startswith(f"{slot}/")]
    return hashlib.sha256(b"".join(sorted(f"{n}:{s}\n".encode() for n, s in pairs))).hexdigest()


class ModelWeights(BaseModel):
    """One loaded model in Qwen's ``weights.models``: the checkpoint it was loaded from (``None`` = stock) and its
    fingerprint (:func:`weights_fingerprint` over its ``<slot>/`` files)."""

    checkpoint: str | None = None
    fingerprint: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class Weights(BaseModel):
    """``GET /health``'s ``weights``: the selected checkpoint (``None`` = stock), the files its last load read and
    their :func:`weights_fingerprint`. ``files`` and ``fingerprint`` are ``None`` until those weights have been
    loaded (engines load lazily). Qwen adds ``models``, per loaded slot (``base``, ``custom_voice``)."""

    checkpoint: str | None = None
    fingerprint: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    files: list[WeightFile] | None = None
    models: dict[str, ModelWeights] | None = None


class SynthesiseResponse(BaseModel):
    """The body of a ``POST /synthesise`` answer. ``checkpoint`` and ``weights_fingerprint`` name the weights that
    generated the utterance, read under its model lease (XTTS, Qwen; Miso reports none); Qwen adds ``model``, the
    slot that spoke it. Pin these, not a ``/health`` read, since another client can load other weights between."""

    model_config = ConfigDict(extra="allow")

    audio_url: str
    duration: float | None = None
    duration_seconds: float | None = None
    format: str | None = None
    seed: int | None = None
    device: str | None = None
    model: str | None = None
    checkpoint: str | None = None
    weights_fingerprint: str | None = None


def stream_identity(headers: Mapping[str, str]) -> dict[str, str | None]:
    """``{"model", "checkpoint", "weights_fingerprint"}`` from a ``/synthesise/stream`` response's headers (``None``
    where absent or empty; the checkpoint is percent-decoded)."""
    lower = {k.lower(): v for k, v in headers.items()}
    checkpoint = lower.get(CHECKPOINT_HEADER.lower()) or None
    return {
        "model": lower.get(MODEL_HEADER.lower()) or None,
        "checkpoint": unquote(checkpoint) if checkpoint else None,
        "weights_fingerprint": lower.get(WEIGHTS_HEADER.lower()) or None,
    }


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
