# syntax=docker/dockerfile:1
#
# argus-echo's Miso TTS 8B engine (same :8031 HTTP contract as the other engines).
# compose.yaml builds this; ./run_server.sh runs the same server in a uv venv on
# the host instead. The model is NOT baked in (it is ~33 GB F32) —
# mount an HF cache volume so first run downloads once and later runs reuse it.
FROM python:3.11-slim

WORKDIR /app

# ffmpeg: torchaudio decode of uploaded mp3/m4a/ogg reference clips.
# libsndfile1: soundfile WAV encode. git: the silentcipher git dependency.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git build-essential curl

# uv for the lockfile-driven env (pins torch==2.9.1 + the Miso deps).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Resolve the environment from the repo's pyproject/uv.lock first (better layer
# caching), then add the server runtime deps.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python 3.11 --no-install-project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install setuptools fastapi "uvicorn[standard]" python-multipart structlog \
        boto3 silero-vad pyloudnorm noisereduce httpx

# App code (generator.py carries the low-RAM meta-load + ungated-tokenizer patch).
COPY generator.py models.py moshi_compat.py watermarking.py miso_server.py ./

# Ungated Llama-3.2 tokenizer mirror (meta-llama is gated); override if you have
# Meta access. HF cache + model dir are mounted at runtime (see compose).
ENV MISO_TTS_TOKENIZER=unsloth/Llama-3.2-1B \
    MISO_DEVICE=cuda \
    MISO_PORT=8031 \
    HF_HOME=/hf-cache \
    NO_TORCH_COMPILE=1 \
    PYTHONUNBUFFERED=1

# The image's identity, reported in /health as `image` (null when unset, e.g. a local build). release.yml passes the
# release tag and commit; a container can't read its own digest. Last, so a new revision rebuilds only this layer.
ARG ECHO_VERSION=
ARG ECHO_REVISION=
ENV ECHO_VERSION=${ECHO_VERSION} \
    ECHO_REVISION=${ECHO_REVISION}

EXPOSE 8031

# Invoke the venv interpreter directly rather than `uv run` — the server deps
# (fastapi/uvicorn/…) are added via `uv pip install` and are NOT in pyproject,
# so a `uv run` lockfile re-sync could evict them.
CMD ["/app/.venv/bin/python", "miso_server.py"]
