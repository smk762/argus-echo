# syntax=docker/dockerfile:1
#
# argus-echo's Coqui XTTS-v2 engine: zero-shot voice cloning with real streaming
# output, behind the same HTTP contract as the other engines. Build from the repo root:
#   docker build -f engines/xtts/Dockerfile engines/xtts
#
# The XTTS-v2 weights are under the Coqui Public Model License (non-commercial);
# building this image accepts it (COQUI_TOS_AGREED below).
FROM python:3.11-slim

WORKDIR /app

# Build/runtime libs: espeak-ng (XTTS phonemiser), libsndfile (soundfile),
# ffmpeg (torchaudio decode of uploaded mp3/m4a/ogg reference clips),
# git (some TTS deps), build-essential (native wheels).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        espeak-ng libsndfile1 ffmpeg git build-essential

# Every package's exact version (#32): both installs below are constrained by it, so a revision implies its
# library stack. See the file's header for how it was made and how to change it.
COPY constraints.txt /tmp/constraints.txt

# torch + torchaudio with CUDA support (build arg selects wheel index).
ARG XTTS_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt torch torchaudio --index-url "${XTTS_TORCH_INDEX_URL}"

# FastAPI runtime + the maintained Coqui fork (`coqui-tts`, not the archived
# `TTS`). boto3 for MinIO uploads; soundfile for WAV encode.
#
# transformers is pinned to the last 4.x release: coqui-tts has no upper bound
# and would otherwise pull transformers 5.x, which (a) removed isin_mps_friendly
# that XTTS imports and (b) broke XTTS streaming (do_stream). 4.57.5 satisfies
# both the import and our inference_stream() path.
#
# silero-vad + pyloudnorm + noisereduce power the reference-clip preprocessor
# (VAD silence trimming, LUFS loudness normalisation, and stationary denoise of
# HVAC hum seeded from the VAD gaps). The silero-vad wheel bundles its model, so
# no network/torch.hub fetch is needed at runtime.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt fastapi "uvicorn[standard]" structlog python-multipart \
        coqui-tts "transformers==4.57.5" boto3 soundfile safetensors \
        silero-vad pyloudnorm noisereduce "argus-cortex[server]>=0.14,<0.15"

# Accept the Coqui Public Model License non-interactively so the model can be
# fetched at build time and at runtime without a TTY prompt.
ENV COQUI_TOS_AGREED=1
ENV XTTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2

# Bake the XTTS-v2 weights into the image so first-run is instant.
RUN python -c "\
from TTS.utils.manage import ModelManager; \
ModelManager().download_model('${XTTS_MODEL}')"

COPY xtts_server.py .

# The image's identity, reported in /health as `image` (null when unset, e.g. a local build). release.yml passes the
# release tag and commit; a container can't read its own digest. Last, so a new revision rebuilds only this layer.
ARG ECHO_VERSION=
ARG ECHO_REVISION=
ENV ECHO_VERSION=${ECHO_VERSION} \
    ECHO_REVISION=${ECHO_REVISION}

EXPOSE 8031

ENV XTTS_DEVICE=cuda:0

CMD ["uvicorn", "xtts_server:app", "--host", "0.0.0.0", "--port", "8031"]
