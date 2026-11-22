# syntax=docker/dockerfile:1
# argus-echo's Wyoming bridges: speak the Wyoming protocol (Home Assistant's voice
# pipeline, argus-mask's wyoming:// URLs) and forward each request to an engine:
# tts_bridge.py to POST /synthesise/stream, asr_bridge.py to POST /transcribe.
# One image, two commands. CPU only, no torch. Build from the repo root:
#   docker build -f engines/wyoming/Dockerfile engines/wyoming
#
# Python 3.11: wyoming's resampling uses the stdlib `audioop`, removed in 3.13.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY tts_bridge.py asr_bridge.py ./

ENV PYTHONUNBUFFERED=1
# 10200: the TTS bridge (the default command). 10300: the ASR bridge, run with
#   command: ["python", "asr_bridge.py"]
# which forwards to a speech-to-text engine's POST /transcribe (STT_URL).
EXPOSE 10200 10300

CMD ["python", "tts_bridge.py"]
