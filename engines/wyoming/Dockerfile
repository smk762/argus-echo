# syntax=docker/dockerfile:1
# argus-echo's Wyoming TTS bridge: speaks the Wyoming protocol (Home Assistant's
# voice pipeline, argus-mask's wyoming:// URLs) and forwards each request to an
# engine's POST /synthesise/stream. CPU only, no torch. Build from the repo root:
#   docker build -f engines/wyoming/Dockerfile engines/wyoming
#
# Python 3.11: wyoming's resampling uses the stdlib `audioop`, removed in 3.13.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY tts_bridge.py ./

ENV PYTHONUNBUFFERED=1
EXPOSE 10200

CMD ["python", "tts_bridge.py"]
