"""The HTTP contracts argus-echo engines speak, one module per capability.

- :mod:`argus_echo.contract.tts`: text to speech (Miso, XTTS-v2, Qwen3-TTS).
- :mod:`argus_echo.contract.stt`: speech to text (the Whisper engine).

An engine names the capabilities it serves in ``GET /health`` (``capabilities``); one that names none is a ``tts``
engine. The TTS names are re-exported here so ``from argus_echo.contract import SynthesiseRequest`` keeps working.
"""

from __future__ import annotations

from argus_echo.contract.tts import (
    AUDIO_FORMAT_HEADER,
    CHANNELS_HEADER,
    CHECKPOINT_HEADER,
    LOAD_PATH,
    MODEL_HEADER,
    SAMPLE_RATE_HEADER,
    SAMPLE_WIDTH,
    SEED_MAX,
    WEIGHTS_HEADER,
    ImageIdentity,
    LoadCheckpointRequest,
    ModelWeights,
    StreamFormat,
    SynthesiseRequest,
    SynthesiseResponse,
    WeightFile,
    Weights,
    stream_identity,
    weights_fingerprint,
)

#: The capabilities an engine may advertise in ``GET /health``.
CAPABILITIES = ("tts", "stt")

#: What an engine that advertises nothing is taken to serve.
DEFAULT_CAPABILITIES = ("tts",)

__all__ = [
    "AUDIO_FORMAT_HEADER",
    "CAPABILITIES",
    "CHANNELS_HEADER",
    "CHECKPOINT_HEADER",
    "DEFAULT_CAPABILITIES",
    "LOAD_PATH",
    "MODEL_HEADER",
    "SAMPLE_RATE_HEADER",
    "SAMPLE_WIDTH",
    "SEED_MAX",
    "WEIGHTS_HEADER",
    "ImageIdentity",
    "LoadCheckpointRequest",
    "ModelWeights",
    "StreamFormat",
    "SynthesiseRequest",
    "SynthesiseResponse",
    "WeightFile",
    "Weights",
    "stream_identity",
    "weights_fingerprint",
]
