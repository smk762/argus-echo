#!/usr/bin/env bash
# Launch the Miso TTS sidecar (drop-in for the XTTS-v2 contract) on the host, without Docker.
#
#   ./run_server.sh              # foreground
#   nohup ./run_server.sh >/var/log/miso_server.log 2>&1 &   # background
#
# Then point your XTTS client's TTS URL at  http://<host>:${MISO_PORT:-8034}
# Settings come from ./run.env (gitignored): HF_TOKEN, HF_HOME and any MISO_* variable.
set -euo pipefail

cd "$(dirname "$0")"

# HF cache, token, and Miso server config.
set -a
# shellcheck disable=SC1091
source ./run.env
set +a

export PYTHONUNBUFFERED=1

# The server deps are not in pyproject/uv.lock (Dockerfile.miso adds them with
# `uv pip install`), so layer them on here or a fresh venv can't import fastapi.
exec uv run \
    --with setuptools --with fastapi --with "uvicorn[standard]" \
    --with python-multipart --with structlog --with boto3 --with silero-vad \
    --with pyloudnorm --with noisereduce --with httpx \
    python miso_server.py
