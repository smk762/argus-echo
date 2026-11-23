# syntax=docker/dockerfile:1
#
# argus-echo's Qwen3-TTS engine. Build from the repo root:
#   docker build -f engines/qwen/Dockerfile engines/qwen
FROM python:3.12-slim

WORKDIR /app

# sox is required by qwen-tts for audio processing
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends sox libsox-dev

# Every package's exact version (#32): all three installs below are constrained by it, so a revision implies its
# library stack. See the file's header for how it was made and how to change it.
COPY constraints.txt /tmp/constraints.txt

# torch + torchaudio with CUDA support (build arg selects wheel index)
ARG TTS_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt torch torchaudio --index-url "${TTS_TORCH_INDEX_URL}"

# FastAPI + runtime deps; argus-cortex for the admin bearer check.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt fastapi "uvicorn[standard]" structlog python-multipart \
        "argus-cortex[server]>=0.14,<0.15"

# TTS-specific packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt qwen-tts soundfile boto3 safetensors

# Bake both model variants into the image so first-run is instant.
ARG TTS_MODEL_SIZE=0.6b
RUN python -c "\
from qwen_tts import Qwen3TTSModel; \
size='${TTS_MODEL_SIZE}'.upper(); \
Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{size}-Base'); \
Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice')"

COPY tts_server.py .

# The image's identity, reported in /health as `image` (null when unset, e.g. a local build). release.yml passes the
# release tag and commit; a container can't read its own digest. Last, so a new revision rebuilds only this layer.
ARG ECHO_VERSION=
ARG ECHO_REVISION=
ENV ECHO_VERSION=${ECHO_VERSION} \
    ECHO_REVISION=${ECHO_REVISION}

EXPOSE 8031

ENV TTS_MODEL_SIZE=${TTS_MODEL_SIZE}
ENV TTS_DEVICE=cuda:0

CMD ["uvicorn", "tts_server:app", "--host", "0.0.0.0", "--port", "8031"]
