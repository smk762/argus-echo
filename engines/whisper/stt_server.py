"""
argus-echo's speech-to-text engine: faster-whisper (Whisper large-v3-turbo on CTranslate2), FastAPI, port 8031.

Serves the ``stt`` contract (docs/api.md, "Speech to text"):

  POST /transcribe          — multipart WAV/FLAC in, {text, language, duration, segments} out
  POST /transcriptions      — JSON {audio_url | audio_base64, language} in, {text, language, duration_seconds,
                              confidence} out (imported from agent-composer's whisper sidecar)
  WS   /transcribe/stream   — 16 kHz s16le PCM in, partial and final transcripts out, cut by Silero VAD
  GET  /health              — argus-cortex ResidentHealth + engine fields, capabilities ["stt"]
  POST /admin/{evict,unload,wake}

GPU memory lifecycle, as on the TTS engines: unloaded -> gpu on first use; after STT_KEEP_ALIVE_GPU idle
seconds the weights move to host RAM (CTranslate2's unload_model(to_cpu=True)), after a further
STT_KEEP_ALIVE_CPU they are freed. Work holds a lease (ModelManager.lease) for its whole run, and demotion
or unload waits for every lease to drop. A negative keep-alive pins that stage. VRAM is bounded by
CTranslate2's caching allocator (CT2_CUDA_CACHING_ALLOCATOR_CONFIG, set in the Dockerfile) and by
transcribing one request at a time (STT_CONCURRENCY); gpu-tenant.yaml declares the measured peak to iris.

Uploads are read through argus_cortex.limits (read_upload, then open_audio), so the byte cap, the format
(WAV or FLAC, sniffed from the magic bytes), the channel count, the sample rate and the duration are checked
before any sample is decoded. /transcriptions puts its fetched or base64-decoded bytes through the same caps;
audio that is not WAV or FLAC is decoded by the ffmpeg CLI, reading only from a pipe and only through a demuxer in
STT_INPUT_FORMATS, and stopped at STT_MAX_SECONDS.

audio_url (/transcriptions) is fetched with argus_cortex.net.afetch: http(s) only, no credentials in the URL, every
address the host resolves to must be global (no loopback, private, link-local/metadata, CGNAT, reserved, multicast
or unspecified, IPv4 embedded in IPv6 included) unless the host is in STT_AUDIO_URL_ALLOWED_HOSTS, the connection is
pinned to the checked address, a redirect is followed only up to STT_AUDIO_URL_MAX_REDIRECTS (default 0) and each
hop is checked again, the body is capped at STT_MAX_UPLOAD_MB as it streams, and the whole fetch at
STT_DOWNLOAD_TIMEOUT. Failures answer one generic 400 or 502.

Env vars:
  STT_MODEL             — Hugging Face repo of the CTranslate2 weights (default mobiuslabsgmbh/faster-whisper-large-v3-turbo)
  STT_MODEL_REVISION    — the repo commit the weights are pinned to (default: the one baked into the image)
  STT_MODEL_PATH        — a local directory of weights; wins over STT_MODEL (the image sets it)
  STT_DEVICE            — "cuda" (default) or "cpu"
  STT_COMPUTE_TYPE      — CTranslate2 compute type (default float16 on cuda, int8 on cpu)
  STT_BEAM_SIZE         — beam width (default 5)
  STT_PRELOAD           — "1" to load at startup
  STT_KEEP_ALIVE_GPU    — idle seconds on the GPU before moving to RAM (default 300)
  STT_KEEP_ALIVE_CPU    — idle seconds in RAM before freeing (default 600)
  STT_CONCURRENCY       — transcriptions run at once (default 1)
  STT_QUEUE_TIMEOUT_S   — how long a request waits for a slot before 503 (default 30)
  STT_MAX_UPLOAD_MB     — upload cap (default 50)
  STT_MAX_SECONDS       — longest audio accepted (default 600)
  STT_STREAM_MAX_SESSIONS, STT_STREAM_MIN_SILENCE_MS, STT_STREAM_MAX_UTTERANCE_S, STT_STREAM_PARTIAL_S,
  STT_VAD_THRESHOLD     — streaming; see docs/api.md
  STT_VAD_FILTER        — "1" (default) runs faster-whisper's Silero VAD pre-filter on /transcriptions (as composer's
                          sidecar did); /transcribe never uses it
  STT_AUDIO_URL_ALLOWED_HOSTS — `host` or `host:port` entries /transcriptions may fetch although they resolve to a
                          private address (e.g. `minio:9000`). Default empty
  STT_AUDIO_URL_MAX_REDIRECTS — redirects an audio_url fetch follows, each re-checked (default 0: none)
  STT_DOWNLOAD_TIMEOUT  — seconds for a whole audio_url fetch (default 30)
  STT_DECODE_TIMEOUT    — seconds for one ffmpeg decode (default 120)
  STT_MAX_DECODES       — /transcriptions requests that may fetch and decode at once (default 2); the rest wait
                          up to STT_QUEUE_TIMEOUT_S, then 503. Bounds the bytes held and the ffmpeg processes run
  STT_INPUT_FORMATS     — ffmpeg demuxers /transcriptions may decode non-WAV/FLAC audio with (comma list; default
                          plain audio containers, no playlists). "none" accepts WAV/FLAC only
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import os
import subprocess
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Callable, Optional, Protocol

import numpy as np
import structlog
from argus_cortex.auth import bearer
from argus_cortex.limits import (
    DEFAULT_SAMPLE_RATES,
    BadInput,
    TooLarge,
    decode_base64,
    open_audio,
    read_body,
    read_upload,
)
from argus_cortex.net import UrlRefused, afetch
from fastapi import Depends, FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.requests import HTTPConnection, Request

logger = structlog.get_logger()


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


# The weights: the CTranslate2 conversion of openai/whisper-large-v3-turbo that faster-whisper's own
# "large-v3-turbo" alias downloads (MIT, like the original), pinned to one commit.
MODEL_REPO = os.getenv("STT_MODEL", "mobiuslabsgmbh/faster-whisper-large-v3-turbo")
MODEL_REVISION = os.getenv("STT_MODEL_REVISION", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf")
MODEL_PATH = os.getenv("STT_MODEL_PATH", "")
DEVICE = os.getenv("STT_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "float16" if DEVICE.startswith("cuda") else "int8")
BEAM_SIZE = _env_int("STT_BEAM_SIZE", 5)
PRELOAD = os.getenv("STT_PRELOAD", "0") == "1"
KEEP_ALIVE_GPU = _env_float("STT_KEEP_ALIVE_GPU", 300)
KEEP_ALIVE_CPU = _env_float("STT_KEEP_ALIVE_CPU", 600)
CONCURRENCY = max(1, _env_int("STT_CONCURRENCY", 1))
QUEUE_TIMEOUT_S = _env_float("STT_QUEUE_TIMEOUT_S", 30)
MAX_DECODES = max(1, _env_int("STT_MAX_DECODES", 2))
MAX_UPLOAD_BYTES = int(_env_float("STT_MAX_UPLOAD_MB", 50) * 1024 * 1024)
MAX_SECONDS = _env_float("STT_MAX_SECONDS", 600)
STREAM_MAX_SESSIONS = max(1, _env_int("STT_STREAM_MAX_SESSIONS", 4))
STREAM_MIN_SILENCE_MS = _env_int("STT_STREAM_MIN_SILENCE_MS", 600)
STREAM_MAX_UTTERANCE_S = _env_float("STT_STREAM_MAX_UTTERANCE_S", 30)
STREAM_PARTIAL_S = _env_float("STT_STREAM_PARTIAL_S", 1.0)
VAD_THRESHOLD = _env_float("STT_VAD_THRESHOLD", 0.5)
# /transcriptions (agent-composer's route). Its byte and duration caps are MAX_UPLOAD_BYTES and MAX_SECONDS.
VAD_FILTER = os.getenv("STT_VAD_FILTER", "1") == "1"
AUDIO_URL_ALLOWED_HOSTS = tuple(h.strip().lower() for h in os.getenv("STT_AUDIO_URL_ALLOWED_HOSTS", "").split(",") if h.strip())
AUDIO_URL_MAX_REDIRECTS = max(0, _env_int("STT_AUDIO_URL_MAX_REDIRECTS", 0))
DOWNLOAD_TIMEOUT = _env_float("STT_DOWNLOAD_TIMEOUT", 30)
DECODE_TIMEOUT = _env_float("STT_DECODE_TIMEOUT", 120)
# Demuxers ffmpeg may pick (it probes the content, whatever the URL says). Plain audio/AV containers only: a playlist
# or concat demuxer (hls, concat, ...) would open other resources named inside the audio. Names are ffmpeg's.
_INPUT_FORMATS_RAW = os.getenv("STT_INPUT_FORMATS", "wav,w64,mp3,ogg,flac,webm,matroska,mp4,m4a,aac,aiff,caf,amr,au")
INPUT_FORMATS = "" if _INPUT_FORMATS_RAW.strip().lower() == "none" else _INPUT_FORMATS_RAW.replace(" ", "")

SAMPLE_RATE = 16000  # Whisper's rate; uploads at other rates are resampled
ALLOWED_RATES = tuple(sorted({*DEFAULT_SAMPLE_RATES, 88200, 96000}))
VAD_FRAME = 512  # Silero's window at 16 kHz (32 ms)
STREAM_MAX_MESSAGE_BYTES = 1 << 20  # one binary frame; a second of audio is 32 KB
CAPABILITIES = ["stt"]

# Whisper's languages (faster_whisper.tokenizer._LANGUAGE_CODES), so a bad code is a 422 before the model loads.
LANGUAGES = frozenset(
    "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha haw he hi hr ht hu hy "
    "id is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru "
    "sa sd si sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo zh yue".split()
)


class LanguageError(ValueError):
    pass


def normalise_language(raw: Optional[str]) -> Optional[str]:
    """``None``/blank -> None (detect); ``en-US`` -> ``en``. Raises LanguageError for a code Whisper lacks."""
    if raw is None or not raw.strip():
        return None
    code = raw.strip().lower().replace("_", "-")
    if code not in LANGUAGES:
        code = code.split("-")[0]
    if code not in LANGUAGES:
        raise LanguageError(f"language {raw.strip()[:16]!r} is not one Whisper knows (an ISO 639-1 code such as 'en')")
    return code


# ── The model ─────────────────────────────────────────────────────────────────


class Backend(Protocol):
    """What the server needs from a model. ``WhisperBackend`` is the real one; tests pass a fake."""

    def load(self) -> None: ...  # onto DEVICE
    def to_cpu(self) -> None: ...
    def to_device(self) -> None: ...
    def free(self) -> None: ...
    def transcribe(
        self, samples: np.ndarray, sample_rate: int, language: Optional[str], word_timestamps: bool, vad_filter: bool = False
    ) -> dict: ...


def resample(samples: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 at *rate* -> *target*, through PyAV's resampler (faster-whisper's own decoder uses it)."""
    if rate == target:
        return np.ascontiguousarray(samples, dtype=np.float32)
    import av

    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples, dtype=np.float32)[None, :], format="flt", layout="mono")
    frame.sample_rate = rate
    resampler = av.AudioResampler(format="flt", layout="mono", rate=target)
    out = [f.to_ndarray().reshape(-1) for f in [*resampler.resample(frame), *resampler.resample(None)]]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, dtype=np.float32)


class WhisperBackend:
    def __init__(self) -> None:
        self._model = None

    def _weights(self) -> str:
        if MODEL_PATH:
            return MODEL_PATH
        from huggingface_hub import snapshot_download

        return snapshot_download(MODEL_REPO, revision=MODEL_REVISION)

    def load(self) -> None:
        from faster_whisper import WhisperModel

        path = self._weights()
        logger.info("stt_loading", model=MODEL_REPO, revision=MODEL_REVISION, device=DEVICE, compute_type=COMPUTE_TYPE)
        self._model = WhisperModel(path, device=DEVICE, compute_type=COMPUTE_TYPE, num_workers=CONCURRENCY)

    def to_cpu(self) -> None:
        if self._model is not None and DEVICE != "cpu":
            self._model.model.unload_model(to_cpu=True)

    def to_device(self) -> None:
        if self._model is not None and not self._model.model.model_is_loaded:
            self._model.model.load_model()

    def free(self) -> None:
        self._model = None
        gc.collect()

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, language: Optional[str], word_timestamps: bool, vad_filter: bool = False
    ) -> dict:
        audio = resample(samples, sample_rate)
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=BEAM_SIZE,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            condition_on_previous_text=False,
        )
        out = []
        for s in segments:  # a generator: the decoding happens here
            seg: dict[str, Any] = {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
            if word_timestamps:
                seg["words"] = [
                    {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word, "probability": round(w.probability, 4)}
                    for w in (s.words or [])
                ]
            out.append(seg)
        return {
            "text": " ".join(s["text"] for s in out if s["text"]).strip(),
            "language": info.language,
            "language_probability": None if language else round(float(info.language_probability), 4),
            "duration": round(len(samples) / sample_rate, 3),
            "segments": out,
        }


#: Builds the backend on first load. Tests replace it.
backend_factory: Callable[[], Backend] = WhisperBackend


class State(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    CPU = "cpu"
    GPU = "gpu"
    ERROR = "error"


class ModelManager:
    """unloaded -> gpu <-> cpu -> unloaded, with leases and keep-alive timers (as the TTS engines)."""

    def __init__(self) -> None:
        self.keep_alive_gpu = KEEP_ALIVE_GPU
        self.keep_alive_cpu = KEEP_ALIVE_CPU
        self.state = State.UNLOADED
        self.error: Optional[str] = None
        self.backend: Optional[Backend] = None
        self.last_used = 0.0
        self.active = 0
        self._lock = asyncio.Lock()
        self._evict_task: Optional[asyncio.Task] = None
        self._transition: Optional[asyncio.Task] = None

    @property
    def resident_device(self) -> bool:
        return DEVICE.startswith("cuda")

    async def _shielded(self, fn: Callable[[], None], ok: State) -> None:
        """Run a load or move in a thread as its own task: a cancelled caller can't strand LOADING."""

        async def run() -> None:
            try:
                await asyncio.to_thread(fn)
            except BaseException as exc:
                self.state = State.ERROR
                self.error = str(exc) or type(exc).__name__
                logger.error("stt_transition_failed", error=self.error)
                raise
            self.state, self.error = ok, None
            if self.active == 0 and not self._lock.locked():
                self.last_used = time.time()
                self._arm_evict()

        task = asyncio.create_task(run())
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        self._transition = task
        await asyncio.shield(task)

    async def _await_transition(self) -> None:
        task = self._transition
        if task is not None and not task.done():
            with contextlib.suppress(Exception):  # its own handler records the failure in self.state
                await asyncio.shield(task)

    async def _ensure_locked(self) -> None:
        await self._await_transition()
        if self.state == State.GPU and self.backend is not None:
            return
        if self.state in (State.UNLOADED, State.ERROR) or self.backend is None:
            self.state = State.LOADING
            self.backend = backend_factory()
            await self._shielded(self.backend.load, State.GPU)
        elif self.state == State.CPU:
            await self._shielded(self.backend.to_device, State.GPU)
        if self.state != State.GPU or self.backend is None:
            raise RuntimeError(f"stt model not ready (state {self.state.value})")

    async def ensure_loaded(self) -> None:
        """Load now without a lease (preload, /admin/wake)."""
        async with self._lock:
            await self._ensure_locked()
            self.last_used = time.time()
            if self.active == 0:
                self._arm_evict()

    async def acquire(self) -> Backend:
        async with self._lock:
            await self._ensure_locked()
            self.active += 1
            self.last_used = time.time()
            self._cancel_evict()
            return self.backend

    def release(self) -> None:
        self.active = max(0, self.active - 1)
        self.last_used = time.time()
        if self.active == 0:
            self._arm_evict()

    @asynccontextmanager
    async def lease(self):
        backend = await self.acquire()
        try:
            yield backend
        finally:
            self.release()

    async def evict(self) -> bool:
        """GPU -> host RAM. False while a lease is held or when there is nothing on the GPU."""
        async with self._lock:
            if self.active > 0:
                return False
            await self._await_transition()
            if self.state != State.GPU or not self.resident_device or self.backend is None:
                return False
            await self._shielded(self.backend.to_cpu, State.CPU)
        if self.active == 0:
            self._arm_evict()
        return True

    async def unload(self, force: bool = False) -> bool:
        async with self._lock:
            if self.active > 0 and not force:
                return False
            await self._await_transition()
            self._cancel_evict()
            if self.backend is not None:
                await asyncio.to_thread(self.backend.free)
            self.backend = None
            self.state = State.UNLOADED
            logger.info("stt_unloaded")
            return True

    def _cancel_evict(self) -> None:
        task = self._evict_task
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()

    def _arm_evict(self) -> None:
        self._cancel_evict()
        self._evict_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        try:
            if self.state == State.GPU:
                if self.keep_alive_gpu < 0:
                    return
                if self.keep_alive_gpu > 0:
                    await asyncio.sleep(self.keep_alive_gpu)
                    if time.time() - self.last_used < self.keep_alive_gpu:
                        return
                if self.state == State.GPU and self.resident_device and not await self.evict():
                    return
            elif self.state != State.CPU:
                return
            if self.active > 0 or self.state not in (State.CPU, State.GPU):
                return
            if self.keep_alive_cpu < 0:
                return
            if self.keep_alive_cpu > 0:
                since = time.time()
                await asyncio.sleep(self.keep_alive_cpu)
                if self.last_used > since:
                    return
            await self.unload()
        except asyncio.CancelledError:
            pass


mgr = ModelManager()
_slots = asyncio.Semaphore(CONCURRENCY)
# /transcriptions' fetch and decode, before it takes a _slots permit: each holds up to MAX_UPLOAD_BYTES of audio, its
# decoded samples and maybe an ffmpeg process, so they are bounded too.
_intake = asyncio.Semaphore(MAX_DECODES)


class Busy(Exception):
    pass


async def run_transcription(
    samples: np.ndarray, rate: int, language: Optional[str], word_timestamps: bool, *, vad_filter: bool = False
) -> dict:
    """Wait (bounded) for a slot, take a lease and transcribe in a thread."""
    try:
        await asyncio.wait_for(_slots.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise Busy() from None
    try:
        async with mgr.lease() as backend:
            return await asyncio.to_thread(backend.transcribe, samples, rate, language, word_timestamps, vad_filter)
    finally:
        _slots.release()


# ── App ───────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRELOAD:
        await mgr.ensure_loaded()
    yield
    await mgr.unload(force=True)


app = FastAPI(title="argus-echo — Whisper STT", version="1.0.0", lifespan=lifespan)


def _error(status: int, message: str, retry_after: Optional[int] = None) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    return JSONResponse(status_code=status, content={"error": message}, headers=headers)


def _model_failure(exc: RuntimeError) -> Optional[JSONResponse]:
    text = str(exc).lower()
    if "out of memory" in text or "cuda" in text or "not ready" in text:
        logger.error("stt_model_error", error=str(exc))
        return _error(503, "the model is unavailable (loading failed or out of GPU memory)", retry_after=30)
    return None


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    word_timestamps: bool = Form(False),
):
    try:
        lang = normalise_language(language)
    except LanguageError as exc:
        return _error(422, str(exc))
    if mgr.state == State.LOADING:
        return _error(503, "model loading", retry_after=10)
    try:
        data = await read_upload(audio, MAX_UPLOAD_BYTES, what="audio")
        clip = await asyncio.to_thread(
            open_audio, data, max_seconds=MAX_SECONDS, allowed_rates=ALLOWED_RATES, max_bytes=MAX_UPLOAD_BYTES, mono=True
        )
    except TooLarge as exc:
        return _error(413, str(exc))
    except BadInput as exc:
        return _error(400, str(exc))
    del data
    try:
        return await run_transcription(clip.samples, clip.sample_rate, lang, word_timestamps)
    except Busy:
        return _error(503, "busy: every transcription slot is taken", retry_after=5)
    except RuntimeError as exc:
        failure = _model_failure(exc)
        if failure is None:
            raise
        return failure


# ── POST /transcriptions (from a URL or base64; agent-composer's JSON route) ──

AUDIO_URL_REJECTED = "audio_url is not an allowed http(s) URL"
AUDIO_URL_FAILED = "Could not download audio_url"
#: Tests replace these: a resolver (host, port) -> addresses for afetch, and an httpx transport.
audio_url_resolver: Optional[Callable[[str, int], list[str]]] = None
audio_url_transport: Any = None
# argus_cortex.net's UrlRefused reasons for a fetch that was allowed but failed (502); every other refusal is a 400.
# cortex gives no structured reason, so these match its messages: test_transcriptions.py's
# CortexReasonsTest triggers each one through the real afetch, so a reworded message fails CI, not production.
_UPSTREAM_FAILURES = (
    "could not fetch",
    "could not connect",
    "the fetch took longer",
    "more than",  # ... redirects
    "a redirect without",
    "the response has an unsupported",
    "the response body could not",
)


class TranscriptionsRequest(BaseModel):
    """composer's body; unknown fields are ignored, as composer did."""

    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    language: Optional[str] = None


class AudioURLError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _max_body_bytes() -> int:
    """The JSON body: base64 of MAX_UPLOAD_BYTES plus room for the URL, the language and the JSON around them."""
    return 4 * ((MAX_UPLOAD_BYTES + 2) // 3) + 64 * 1024


async def fetch_audio_url(url: str) -> bytes:
    """The body of *url* under argus-cortex's SSRF policy, capped at MAX_UPLOAD_BYTES. AudioURLError otherwise:
    413 over the cap, 400 for a URL that may not be fetched, 502 when the fetch itself fails. The reason is logged;
    the client gets a generic message (no resolved address, no upstream status)."""
    try:
        reply = await afetch(
            url,
            allow_hosts=AUDIO_URL_ALLOWED_HOSTS,
            timeout=DOWNLOAD_TIMEOUT,
            deadline=DOWNLOAD_TIMEOUT,
            max_bytes=MAX_UPLOAD_BYTES,
            max_redirects=AUDIO_URL_MAX_REDIRECTS,
            resolve=audio_url_resolver,
            transport=audio_url_transport,
        )
    except UrlRefused as exc:
        reason = str(exc)
        logger.warning("stt_audio_url_refused", reason=reason)
        if reason.startswith("the response is larger than"):
            raise AudioURLError(413, f"audio is larger than {MAX_UPLOAD_BYTES} bytes") from None
        # Allowed, but the fetch failed: unreachable, too slow, redirected, or a body that can't be read. (Prefixes:
        # a reason that names the host starts with it, and a host has no spaces.)
        if reason.startswith(_UPSTREAM_FAILURES):
            raise AudioURLError(502, AUDIO_URL_FAILED) from None
        raise AudioURLError(400, AUDIO_URL_REJECTED) from None
    if reply.status != 200:
        logger.warning("stt_audio_url_fetch_failed", status=reply.status)
        raise AudioURLError(502, AUDIO_URL_FAILED)
    return reply.body


def _is_wav_or_flac(data: bytes) -> bool:
    return (data[:4] == b"RIFF" and data[8:12] == b"WAVE") or data[:4] == b"fLaC"


def ffmpeg_decode(data: bytes) -> np.ndarray:
    """Any audio ffmpeg can demux with a format in INPUT_FORMATS -> 16 kHz mono float32, at most MAX_SECONDS.

    The bytes go in on stdin and the only protocol ffmpeg may open is ``pipe``, so nothing in them (a playlist, a
    reference) can make it read a file or the network. Output stops just past MAX_SECONDS (``-t``), so a long or
    lying stream costs no more than the cap. TooLarge past the cap, BadInput when ffmpeg can't decode it."""
    limit = int(MAX_SECONDS * SAMPLE_RATE)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "pipe", "-format_whitelist", INPUT_FORMATS,
        "-i", "pipe:0", "-map", "0:a:0", "-t", f"{MAX_SECONDS + 0.1:.3f}",
        "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1",
    ]  # fmt: skip
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, timeout=DECODE_TIMEOUT, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("stt_decode_failed", error=str(exc)[:300])
        raise BadInput("could not decode the audio") from None
    if proc.returncode != 0:
        logger.warning("stt_decode_failed", code=proc.returncode, error=proc.stderr.decode(errors="replace")[-300:])
        raise BadInput("could not decode the audio")
    samples = np.frombuffer(proc.stdout[: len(proc.stdout) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    if len(samples) > limit:
        raise TooLarge(f"audio at {SAMPLE_RATE} Hz", limit, "frames")
    if len(samples) == 0:
        raise BadInput("audio has no samples")
    return samples


def decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """WAV/FLAC through open_audio (as /transcribe); anything else through ffmpeg when INPUT_FORMATS allows it.

    A WAV or FLAC that open_audio refuses (a sample rate outside ALLOWED_RATES, say 12 kHz) goes to ffmpeg
    too when INPUT_FORMATS allows it, as agent-composer's sidecar decoded it. TooLarge (over the duration cap) is
    final: ffmpeg would stop at the same cap."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise TooLarge("audio", MAX_UPLOAD_BYTES)
    if _is_wav_or_flac(data) or not INPUT_FORMATS:
        try:
            clip = open_audio(
                data, max_seconds=MAX_SECONDS, allowed_rates=ALLOWED_RATES, max_bytes=MAX_UPLOAD_BYTES, mono=True
            )
        except BadInput as exc:
            if not (INPUT_FORMATS and _is_wav_or_flac(data)):
                raise
            logger.info("stt_open_audio_refused_trying_ffmpeg", reason=str(exc)[:200])
            return ffmpeg_decode(data), SAMPLE_RATE
        return clip.samples, clip.sample_rate
    return ffmpeg_decode(data), SAMPLE_RATE


@app.post("/transcriptions")
async def transcriptions(request: Request):
    """Transcribe from a URL or base64 (agent-composer's whisper sidecar route, imported): a JSON body with ``audio_url``
    (fetched here) or ``audio_base64``, and ``{text, language, duration_seconds, confidence}`` back."""
    try:
        body = await read_body(request, _max_body_bytes(), what="request body")
    except TooLarge as exc:
        return _error(413, str(exc))
    except BadInput as exc:
        return _error(400, str(exc))
    try:
        req = TranscriptionsRequest.model_validate_json(body)
    except ValidationError:
        return _error(422, "the body must be a JSON object with audio_url or audio_base64 (strings) and language")
    del body
    if req.audio_url and req.audio_base64:
        return _error(422, "Provide audio_url or audio_base64, not both")
    if not req.audio_url and not req.audio_base64:
        return _error(400, "audio_url or audio_base64 is required")
    try:
        lang = normalise_language(req.language)
    except LanguageError as exc:
        return _error(422, str(exc))
    if mgr.state == State.LOADING:
        return _error(503, "model loading", retry_after=10)

    # At most MAX_DECODES requests fetch and decode at once (each holds its audio, its samples and maybe an ffmpeg
    # process); the rest wait like a transcription slot does, then 503.
    try:
        await asyncio.wait_for(_intake.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        return _error(503, "busy: every decode slot is taken", retry_after=5)
    try:
        if req.audio_url:
            data = await fetch_audio_url(req.audio_url)
        else:
            data = await asyncio.to_thread(decode_base64, req.audio_base64, MAX_UPLOAD_BYTES, what="audio_base64")
        req = None  # drop the base64 text before decoding
        samples, rate = await asyncio.to_thread(decode_audio_bytes, data)
    except AudioURLError as exc:
        return _error(exc.status, exc.message)
    except TooLarge as exc:
        return _error(413, str(exc))
    except BadInput as exc:
        return _error(400, str(exc))
    finally:
        _intake.release()
    del data
    try:
        result = await run_transcription(samples, rate, lang, False, vad_filter=VAD_FILTER)
    except Busy:
        return _error(503, "busy: every transcription slot is taken", retry_after=5)
    except RuntimeError as exc:
        failure = _model_failure(exc)
        if failure is None:
            raise
        return failure
    prob = result.get("language_probability")
    return {
        "text": result["text"],
        "language": result["language"] or "en",
        "duration_seconds": round(float(result["duration"]), 3),
        # faster-whisper's language probability, as composer reported it (1.0 when the language was given).
        "confidence": round(float(prob), 4) if prob is not None else 1.0,
    }


# ── Streaming ─────────────────────────────────────────────────────────────────


class Endpointer:
    """Cuts a stream of per-frame speech probabilities into utterances.

    ``feed`` returns "start" on the frame that opens an utterance (after ``min_speech`` speech frames),
    "end" on the frame that closes one (``min_silence`` frames below the lower threshold, or ``max_frames``
    in the utterance), else None. Hysteresis as Silero's own: speech above ``threshold``, silence below
    ``threshold - 0.15``."""

    def __init__(self, *, threshold: float, min_speech: int, min_silence: int, max_frames: int) -> None:
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.01)
        self.min_speech = max(1, min_speech)
        self.min_silence = max(1, min_silence)
        self.max_frames = max(self.min_speech + 1, max_frames)
        self.in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._frames = 0

    def feed(self, prob: float) -> Optional[str]:
        if not self.in_speech:
            self._speech_run = self._speech_run + 1 if prob >= self.threshold else 0
            if self._speech_run >= self.min_speech:
                self.in_speech, self._silence_run, self._frames = True, 0, self._speech_run
                return "start"
            return None
        self._frames += 1
        if prob < self.neg_threshold:
            self._silence_run += 1
        elif prob >= self.threshold:
            self._silence_run = 0
        if self._silence_run >= self.min_silence or self._frames >= self.max_frames:
            self.in_speech, self._speech_run = False, 0
            return "end"
        return None


class SileroStream:
    """Silero VAD (the ONNX model faster-whisper ships) run one 512-sample frame at a time, keeping the
    recurrent state and the 64-sample context between frames, as faster_whisper.vad does within one call."""

    _session = None

    def __init__(self) -> None:
        if SileroStream._session is None:
            from faster_whisper.vad import get_vad_model

            SileroStream._session = get_vad_model().session
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame.reshape(1, -1)], axis=1).astype(np.float32)
        out, self.h, self.c = self._session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = frame.reshape(1, -1)[:, -64:]
        return float(np.asarray(out).reshape(-1)[0])


#: Builds a per-stream VAD (a callable: one float32 frame of VAD_FRAME samples -> speech probability). Tests replace it.
vad_factory: Callable[[], Callable[[np.ndarray], float]] = SileroStream


def _shift(result: dict, offset: float) -> dict:
    for seg in result.get("segments", []):
        seg["start"] = round(seg["start"] + offset, 3)
        seg["end"] = round(seg["end"] + offset, 3)
        for w in seg.get("words") or []:
            w["start"] = round(w["start"] + offset, 3)
            w["end"] = round(w["end"] + offset, 3)
    return result


_sessions = 0


@app.websocket("/transcribe/stream")
async def transcribe_stream(ws: WebSocket):
    global _sessions
    await ws.accept()
    if _sessions >= STREAM_MAX_SESSIONS:
        await ws.send_json({"type": "error", "error": "busy: too many open streams"})
        await ws.close(code=1013)
        return
    _sessions += 1
    try:
        await _stream(ws)
    except WebSocketDisconnect:
        pass
    finally:
        _sessions -= 1


async def _stream(ws: WebSocket) -> None:
    frame_s = VAD_FRAME / SAMPLE_RATE
    endpointer = Endpointer(
        threshold=VAD_THRESHOLD,
        min_speech=round(0.25 / frame_s),
        min_silence=round(STREAM_MIN_SILENCE_MS / 1000 / frame_s),
        max_frames=round(STREAM_MAX_UTTERANCE_S / frame_s),
    )
    # The frames that confirmed the onset, plus 200 ms before it, so the first word is not clipped.
    preroll_frames = endpointer.min_speech + round(0.2 / frame_s)
    vad = vad_factory()
    language: Optional[str] = None
    word_timestamps = False
    started = False
    pending = b""
    frames_seen = 0
    recent: list[np.ndarray] = []  # frames before speech, kept as pre-roll
    utterance: list[np.ndarray] = []
    utt_start_frame = 0
    last_partial_frames = 0

    await ws.send_json(
        {
            "type": "ready",
            "sample_rate": SAMPLE_RATE,
            "audio_format": "pcm_s16le",
            "channels": 1,
            "min_silence_ms": STREAM_MIN_SILENCE_MS,
            "max_utterance_s": STREAM_MAX_UTTERANCE_S,
        }
    )

    async def send_error(message: str) -> None:
        await ws.send_json({"type": "error", "error": message})
        await ws.close(code=1011)

    async def run(samples: list[np.ndarray]) -> Optional[dict]:
        try:
            return await run_transcription(np.concatenate(samples), SAMPLE_RATE, language, word_timestamps)
        except Busy:
            await send_error("busy: every transcription slot is taken")
        except RuntimeError as exc:
            if _model_failure(exc) is None:
                raise
            await send_error("the model is unavailable (loading failed or out of GPU memory)")
        return None

    async def finish() -> bool:
        nonlocal utterance
        start = utt_start_frame * frame_s
        end = (utt_start_frame + len(utterance)) * frame_s
        result = await run(utterance)
        utterance = []
        if result is None:
            return False
        await ws.send_json({"type": "final", "start": round(start, 3), "end": round(end, 3), **_shift(result, start)})
        return True

    while True:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            return
        if msg.get("text") is not None:
            try:
                ctl = json.loads(msg["text"])
                kind = ctl.get("type")
            except (ValueError, AttributeError):
                await send_error("text frames must be JSON control messages")
                return
            if kind == "start" and not started:
                try:
                    language = normalise_language(ctl.get("language"))
                except LanguageError as exc:
                    await send_error(str(exc))
                    return
                word_timestamps = bool(ctl.get("word_timestamps", False))
                started = True
                continue
            if kind == "end":
                if utterance and not await finish():
                    return
                await ws.close(code=1000)
                return
            await send_error(f"unknown or misplaced control message {str(kind)[:32]!r}")
            return
        data = msg.get("bytes") or b""
        if len(data) > STREAM_MAX_MESSAGE_BYTES:
            await send_error(f"a binary frame is at most {STREAM_MAX_MESSAGE_BYTES} bytes")
            return
        started = True
        pending += data
        n = len(pending) // (2 * VAD_FRAME)
        if not n:
            continue
        pcm = np.frombuffer(pending[: n * 2 * VAD_FRAME], dtype="<i2").astype(np.float32) / 32768.0
        pending = pending[n * 2 * VAD_FRAME :]
        for frame in pcm.reshape(n, VAD_FRAME):
            event = endpointer.feed(vad(frame))
            if endpointer.in_speech or event == "end":
                if event == "start":
                    utterance = [*recent, frame]
                    utt_start_frame = frames_seen - len(recent)
                    recent = []
                    last_partial_frames = 0
                else:
                    utterance.append(frame)
            else:
                recent.append(frame)
                if len(recent) > preroll_frames:
                    recent.pop(0)
            frames_seen += 1
            if event == "end":
                if not await finish():
                    return
        if endpointer.in_speech and STREAM_PARTIAL_S > 0 and utterance:
            frames_in = len(utterance)
            if (frames_in - last_partial_frames) * frame_s >= STREAM_PARTIAL_S:
                last_partial_frames = frames_in
                result = await run(utterance)
                if result is None:
                    return
                await ws.send_json({"type": "partial", "start": round(utt_start_frame * frame_s, 3), "text": result["text"]})


# ── Health and admin (argus-cortex resident contract) ────────────────────────


def _idle_s() -> Optional[float]:
    if not mgr.last_used or mgr.active:
        return None
    return round(time.time() - mgr.last_used, 1)


def _gpu_block() -> dict:
    free = None
    if DEVICE.startswith("cuda"):
        with contextlib.suppress(Exception):  # NVML is optional; /health omits the figure without it
            from argus_cortex.gpu import free_vram_mb

            free = free_vram_mb()
    return {
        "loaded": mgr.state in (State.CPU, State.GPU),
        "model_state": mgr.state.value,
        "idle_s": _idle_s(),
        "free_vram_mb": free,
    }


@app.get("/health")
async def health():
    if mgr.state == State.LOADING:
        return JSONResponse(status_code=503, content={"status": "loading"})
    body = {
        "status": "ok",
        "engine": "whisper",
        "model": MODEL_REPO,
        "revision": MODEL_REVISION,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "capabilities": CAPABILITIES,
        "model_state": mgr.state.value,
        "busy": mgr.active > 0,
        "idle_s": _idle_s(),
        "keep_alive_gpu": KEEP_ALIVE_GPU,
        "keep_alive_cpu": KEEP_ALIVE_CPU,
        "streams": _sessions,
        "gpu": _gpu_block(),
    }
    if mgr.state == State.ERROR:
        body["error"] = mgr.error
    return body


def _admin_insecure() -> bool:
    return os.environ.get("ARGUS_ADMIN_INSECURE", "").strip().lower() in ("1", "true", "yes")


_admin_closed = bearer("ARGUS_ADMIN_TOKEN", open_when_unset=False)
_admin_open = bearer("ARGUS_ADMIN_TOKEN", open_when_unset=True)


async def _require_admin(conn: HTTPConnection) -> None:
    await (_admin_open if _admin_insecure() else _admin_closed)(conn)


if not os.environ.get("ARGUS_ADMIN_TOKEN", "").strip():
    logger.warning(
        "admin_token_unset",
        admin_routes="open (ARGUS_ADMIN_INSECURE)" if _admin_insecure() else "refused with 503",
        hint="set ARGUS_ADMIN_TOKEN to the suite's shared value",
    )


def _busy() -> JSONResponse:
    return JSONResponse(status_code=409, content={"status": "busy", "active": mgr.active, "model_state": mgr.state.value})


@app.post("/admin/evict", dependencies=[Depends(_require_admin)])
async def admin_evict():
    was_gpu = mgr.state == State.GPU
    if not await mgr.evict() and mgr.active:
        return _busy()
    return {"status": "evicted", "unloaded": was_gpu and mgr.state == State.CPU, "model_state": mgr.state.value, "gpu": _gpu_block()}


@app.post("/admin/unload", dependencies=[Depends(_require_admin)])
async def admin_unload():
    was_loaded = mgr.state in (State.CPU, State.GPU)
    if not await mgr.unload():
        return _busy()
    return {"status": "unloaded", "unloaded": was_loaded, "model_state": mgr.state.value, "gpu": _gpu_block()}


@app.post("/admin/wake", dependencies=[Depends(_require_admin)])
async def admin_wake():
    await mgr.ensure_loaded()
    return {"status": "ready", "unloaded": None, "model_state": mgr.state.value, "gpu": _gpu_block()}
