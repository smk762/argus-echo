# syntax=docker/dockerfile:1
#
# argus-echo's speech-to-text engine: faster-whisper running Whisper large-v3-turbo (CTranslate2).
# Build from the repo root:
#   docker build -f engines/whisper/Dockerfile engines/whisper
#
# Whisper's weights and faster-whisper are MIT; the CTranslate2 conversion used here
# (mobiuslabsgmbh/faster-whisper-large-v3-turbo) is MIT too.
FROM python:3.11-slim

WORKDIR /app

# POST /transcriptions decodes audio that isn't WAV/FLAC with the ffmpeg CLI (pipe-only, demuxer allowlist).
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install huggingface_hub
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# CTranslate2 finds cuBLAS and cuDNN in the nvidia-* wheels.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

# The weights, pinned to one commit of the repo and baked in so first use needs no network.
ARG STT_MODEL=mobiuslabsgmbh/faster-whisper-large-v3-turbo
ARG STT_MODEL_REVISION=0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${STT_MODEL}', revision='${STT_MODEL_REVISION}', local_dir='/models/whisper')"
ENV STT_MODEL=${STT_MODEL} STT_MODEL_REVISION=${STT_MODEL_REVISION} STT_MODEL_PATH=/models/whisper

# VRAM cap: CTranslate2's caching allocator keeps at most 200 MiB of freed blocks cached
# (bin_growth,min_bin,max_bin,max_cached_bytes), so the footprint stays near the working set.
ENV CT2_CUDA_ALLOCATOR=cub_caching CT2_CUDA_CACHING_ALLOCATOR_CONFIG=4,3,12,209715200

COPY stt_server.py .

EXPOSE 8031
ENV STT_DEVICE=cuda

CMD ["uvicorn", "stt_server:app", "--host", "0.0.0.0", "--port", "8031"]
