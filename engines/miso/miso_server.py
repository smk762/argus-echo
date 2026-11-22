"""
Miso TTS server — FastAPI, port 8034 (drop-in for the XTTS-v2 sidecar).

Wraps MisoLabs/MisoTTS (8B, Sesame-CSM-style, Mimi 24 kHz) behind the same
HTTP contract as an XTTS-v2 sidecar, so a client that already speaks it (a
chat app's voice proxy, a Wyoming bridge) only needs its TTS URL pointed here.

Endpoints (mirroring the XTTS sidecar):
  POST /synthesise          — JSON → audio uploaded to storage, returns {audio_url,...}
  POST /synthesise/stream   — JSON → raw 24 kHz mono float32 PCM (pcm_f32le)
  POST /analyze-clips       — multipart clips → per-clip quality score + ordering
  POST /clone-voice         — multipart clips → one-shot clone (ref audio+text), stored
  GET  /clones              — list stored clones (meta.json sidecars)
  GET  /clones/{id}/download — clone prompt.safetensors + meta.json as a zip
  DELETE /clones/{id}       — remove stored clone + evict in-memory cache
  POST /clones/import       — import a downloaded voice archive (zip)
  GET  /speakers            — built-in speaker ids (Miso is clone-first)
  GET  /health              — readiness + model/device/GPU state
  GET  /media/{key}         — serve generated audio / clone artifacts (local backend)
  POST /admin/{evict,unload,wake} — bearer ARGUS_ADMIN_TOKEN (closed when unset)

A Miso "voice clone" is one-shot: we store the (preprocessed) reference audio
at 24 kHz mono + its transcript + speaker id as prompt.pt, then pass it as a
conditioning Segment to generate() at synth time. This mirrors XTTS's
`voice_clones/{companion_id}/{prompt.pt,meta.json}` storage layout so /clones,
download and import keep working. Stored prompt.pt objects are only ever written
by this server; the clone *archive* (download/import) carries the prompt as
prompt.safetensors instead, so uploaded bytes never reach a pickle loader.

Native sample rate: 24 kHz (Mimi) — identical to XTTS, so no resample needed in
the Wyoming bridge or browser PCM path.

Env vars:
  MISO_MODEL            — HF repo id or local path (default MisoLabs/MisoTTS)
  MISO_DEVICE           — "cuda" (default) or "cpu"
  MISO_PORT             — listen port (default 8034)
  MISO_PRELOAD          — "1" to load at startup, "0" (default) lazy on first use
  MISO_MAX_AUDIO_MS     — per-utterance ceiling (default 30000)
  MISO_MIN_WORDS_PER_S  — slowest reading rate the length budget allows for
                          (default 1.6). Each utterance may run for
                          MISO_LENGTH_PAD_S + words / this, capped by
                          MISO_MAX_AUDIO_MS. Lower it if slow voices get cut off.
  MISO_LENGTH_PAD_S     — fixed seconds added to that budget (default 1.5)
  MISO_FADE_IN_MS       — fade-in at the start of an utterance (default 5, 0 = off)
  MISO_FADE_OUT_MS      — fade-out at the end of an utterance (default 30, 0 = off).
                          /synthesise/stream holds this much back from each chunk
                          until the next chunk or the end of the stream.
  MISO_CONTEXT_MAX_SEGMENTS — max request `context` segments (default 2)
  MISO_CONTEXT_MAX_SECONDS  — max total request context audio (default 20)
  MISO_TEMPERATURE      — default sampling temperature (default 0.9).
                          A request `temperature`, then a clone meta.json
                          `temperature`, override this. Clamped to 0.1–1.5.
  MISO_TOPK             — default top-k (default 50). Same override order
                          (`topk` on the request, then meta.json). Clamped to 1–200.
  MISO_DEFAULT_SPEAKER  — integer speaker id for non-clone synth (default 0)
  MISO_STREAM_CHUNK_MS  — PCM chunk granularity for /synthesise/stream (default 240)
  MISO_REF_MAX_SECONDS  — cap on stored reference audio (default 20)
  MISO_MAX_CLIP_MB      — largest reference clip /clone-voice and /analyze-clips
                          accept, per clip (default 50); larger is 413
  MISO_MAX_ARCHIVE_MB   — largest zip /clones/import accepts (default 64); larger is 413
  MISO_MAX_CLIPS        — most clips one /clone-voice or /analyze-clips takes (default
                          10). With MISO_MAX_CLIP_MB it caps the whole upload body,
                          which is refused before Starlette spools it to disk.
  ARGUS_ADMIN_TOKEN     — /admin/* needs `Authorization: Bearer <token>`; unset = closed (503)
                          unless ARGUS_ADMIN_INSECURE=1.
                          Unset (default) leaves them open, as argus-iris expects.
  MISO_WHISPER_URL      — optional STT to auto-transcribe clone refs (default "")
  Idle lifecycle (unloaded → cpu → gpu, like the XTTS and ModernBERT sidecars):
  MISO_KEEP_ALIVE_GPU   — idle s on GPU before leaving it (default 300).
                          -1 = stay on GPU for ever, 0 = leave right after each request.
  MISO_KEEP_ALIVE_CPU   — idle s parked on CPU before a full unload (default 0).
                          0 = skip the CPU stage: drop straight from GPU to
                          unloaded, freeing system RAM (the 8B model holds ~17 GB
                          of RSS on CPU). -1 = keep it on CPU for ever.
                          With MISO_DEVICE=cpu the model is always on CPU and the
                          two values add up to its idle time before unloading.
  MISO_OOM_RETRY_AFTER_S — Retry-After on a 503 for a CUDA OOM (default 30).
                          A failed load keeps no partial copy in RAM.

  GPU lock watchdog (one generation at a time):
  MISO_GEN_TIMEOUT_PAD_S, MISO_GEN_TIMEOUT_FACTOR — a generation is abandoned
                          after PAD + FACTOR × its audio budget in s (30, 6)
  MISO_LOCK_WAIT_S      — wait for the GPU before 503 busy (default 120, 0 = for ever)
  MISO_STREAM_STALL_S   — stop a stream whose client stops reading (default 30)
  MISO_STREAM_CHUNK_TIMEOUT_S — end a stream when no chunk comes for this long (60)
  MISO_STUCK_AFTER_S    — lock held this long → /health 503 "stuck"
                          (default: longest possible generation + stall + 60)
  MISO_STUCK_EXIT       — "1" (default) exits the process when stuck, so the
                          container's restart policy brings it back

  Storage (audio_url + clone persistence):
  MISO_STORAGE_BACKEND  — "local" (default) or "s3"
  MISO_DATA_DIR         — local backend root (default ./miso-data)
  MISO_PUBLIC_URL       — public base for local audio_url (default http://<host>:<port>)
  S3_ENDPOINT_URL / S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY /
  S3_BUCKET_NAME / S3_PUBLIC_URL  — used when MISO_STORAGE_BACKEND=s3
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import ctypes
import functools
import gc
import io
import json
import math
import os
import re
import socket
import tempfile
import time
import traceback
import uuid
import zipfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import structlog
from argus_cortex.auth import bearer
from starlette.requests import HTTPConnection
from argus_cortex.limits import BadInput, TooLarge, read_upload
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

logger = structlog.get_logger("miso_server")

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_SOURCE = os.getenv("MISO_MODEL", "MisoLabs/MisoTTS")
DEVICE = os.getenv("MISO_DEVICE", "cuda")
PORT = int(os.getenv("MISO_PORT", "8034"))
PRELOAD = os.getenv("MISO_PRELOAD", "0") == "1"
MAX_AUDIO_MS = float(os.getenv("MISO_MAX_AUDIO_MS", "30000"))
TEMPERATURE = float(os.getenv("MISO_TEMPERATURE", "0.9"))
TOPK = int(os.getenv("MISO_TOPK", "50"))
# Per-request and per-clone overrides are clamped into these ranges so a typo
# cannot push sampling into a regime that clips, rambles, or collapses.
TEMPERATURE_MIN = 0.1
TEMPERATURE_MAX = 1.5
TOPK_MIN = 1
TOPK_MAX = 200
DEFAULT_SPEAKER = int(os.getenv("MISO_DEFAULT_SPEAKER", "0"))
STREAM_CHUNK_MS = int(os.getenv("MISO_STREAM_CHUNK_MS", "240"))
# Incremental decode granularity for /synthesise/stream. Each Miso frame is 80 ms
# of audio; we decode the first few frames quickly for low first-byte latency,
# then larger batches to amortise the per-decode overhead.
STREAM_FIRST_FRAMES = int(os.getenv("MISO_STREAM_FIRST_FRAMES", "5"))
STREAM_DECODE_FRAMES = int(os.getenv("MISO_STREAM_DECODE_FRAMES", "12"))
REF_MAX_SECONDS = float(os.getenv("MISO_REF_MAX_SECONDS", "20"))
# Upload caps. A clip is cut to REF_MAX_SECONDS after decoding, but callers send
# whole recordings to pick from: 50 MB is ~4 min of 48 kHz 16-bit stereo WAV and
# far longer in mp3/ogg. A clone archive holds prompt.safetensors (at most
# _MAX_PROMPT_BYTES, ~2 MB for 20 s of 24 kHz float32) and meta.json.
MAX_CLIP_BYTES = int(float(os.getenv("MISO_MAX_CLIP_MB", "50")) * 1024 * 1024)
MAX_ARCHIVE_BYTES = int(float(os.getenv("MISO_MAX_ARCHIVE_MB", "64")) * 1024 * 1024)
MAX_CLIPS = int(os.getenv("MISO_MAX_CLIPS", "10"))
# Starlette spools every multipart file to a temp file before a handler runs,
# so read_upload's caps bound memory, not disk (smk762/voluptas#30). _UploadBodyCap
# caps the whole body of each upload route first: its files plus form fields
# and multipart framing (_FORM_SLACK).
_FORM_SLACK = 1024 * 1024
_UPLOAD_BODY_CAPS = {
    "/analyze-clips": lambda: MAX_CLIPS * MAX_CLIP_BYTES + _FORM_SLACK,
    "/clone-voice": lambda: MAX_CLIPS * MAX_CLIP_BYTES + _FORM_SLACK,
    "/clones/import": lambda: MAX_ARCHIVE_BYTES + _FORM_SLACK,
}
# Length budget (#3): a generation may run for LENGTH_PAD_S plus the time a slow
# reader (MIN_WORDS_PER_S) needs for the text, never more than MAX_AUDIO_MS. This
# stops a short sentence that misses its end-of-speech frame running on for 30 s.
MIN_WORDS_PER_S = float(os.getenv("MISO_MIN_WORDS_PER_S", "1.6"))
LENGTH_PAD_S = float(os.getenv("MISO_LENGTH_PAD_S", "1.5"))
# Edge fades (#5), in ms. 0 disables a fade.
FADE_IN_MS = float(os.getenv("MISO_FADE_IN_MS", "5"))
FADE_OUT_MS = float(os.getenv("MISO_FADE_OUT_MS", "30"))
# Request context (#4): previous utterances a client sends back for continuity.
CONTEXT_MAX_SEGMENTS = int(os.getenv("MISO_CONTEXT_MAX_SEGMENTS", "2"))
CONTEXT_MAX_SECONDS = float(os.getenv("MISO_CONTEXT_MAX_SECONDS", "20"))
FRAME_MS = 80  # one Miso/Mimi frame
# Head-room on the prompt-length estimate (Mimi frame rounding, tokenizer drift).
CONTEXT_TOKEN_MARGIN = 8
WHISPER_URL = os.getenv("MISO_WHISPER_URL", "").rstrip("/")
# Idle lifecycle: gpu → cpu → unloaded. -1 keeps the
# model in that stage for ever, 0 leaves it at once. CPU defaults to 0 because an
# 8B model parked on CPU holds ~17 GB of RSS, which starves a shared host.
KEEP_ALIVE_GPU = float(os.getenv("MISO_KEEP_ALIVE_GPU", "300"))
KEEP_ALIVE_CPU = float(os.getenv("MISO_KEEP_ALIVE_CPU", "0"))
# Retry-After (s) on a 503 for a GPU OOM while loading or generating.
OOM_RETRY_AFTER_S = int(os.getenv("MISO_OOM_RETRY_AFTER_S", "30"))
# GPU-lock watchdog (#9). Generation runs at about a quarter of real time, so a
# generation may take MISO_GEN_TIMEOUT_PAD_S + MISO_GEN_TIMEOUT_FACTOR × its
# audio budget before it is abandoned.
GEN_TIMEOUT_PAD_S = float(os.getenv("MISO_GEN_TIMEOUT_PAD_S", "30"))
GEN_TIMEOUT_FACTOR = float(os.getenv("MISO_GEN_TIMEOUT_FACTOR", "6"))
# How long a request waits for the GPU before answering 503 busy (0 = for ever).
LOCK_WAIT_S = float(os.getenv("MISO_LOCK_WAIT_S", "120"))
# A stream stops generating when its client takes no chunk for this long.
STREAM_STALL_S = float(os.getenv("MISO_STREAM_STALL_S", "30"))
# A stream gives up when the model produces no chunk for this long.
STREAM_CHUNK_TIMEOUT_S = float(os.getenv("MISO_STREAM_CHUNK_TIMEOUT_S", "60"))
STUCK_EXIT = os.getenv("MISO_STUCK_EXIT", "1") == "1"
STUCK_CHECK_S = 15.0


def _generation_deadline_s(max_audio_ms: float) -> float:
    """Wall-clock limit for one generation with this audio budget."""
    return GEN_TIMEOUT_PAD_S + GEN_TIMEOUT_FACTOR * max_audio_ms / 1000.0


# Past any legitimate hold: the longest generation, then a stalled hand-off.
STUCK_AFTER_S = float(os.getenv("MISO_STUCK_AFTER_S", "0")) or (
    _generation_deadline_s(MAX_AUDIO_MS) + STREAM_STALL_S + 60.0
)

SAMPLE_RATE = 24000  # Mimi codec output; confirmed from generator.sample_rate at load
ANALYZE_SCORE_THRESHOLD = float(os.getenv("MISO_ANALYZE_THRESHOLD", "60"))
REF_PREPROCESS = os.getenv("MISO_PREPROCESS", "1") == "1"

# Miso supports English only.
SUPPORTED_LANGS = ["en"]

_CLONE_CACHE_MAX = 50


# ── Storage backends ──────────────────────────────────────────────────────────
# The XTTS contract uploads WAVs and clone artifacts to MinIO and returns public
# URLs. The default is a local filesystem backend that this service serves at
# GET /media/<key>, so no object store is needed. Switch to S3 (same layout)
# with MISO_STORAGE_BACKEND=s3.

STORAGE_BACKEND = os.getenv("MISO_STORAGE_BACKEND", "local").strip().lower()
DATA_DIR = os.path.abspath(os.getenv("MISO_DATA_DIR", os.path.join(os.getcwd(), "miso-data")))


def _default_public_url() -> str:
    explicit = os.getenv("MISO_PUBLIC_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    host = os.getenv("MISO_PUBLIC_HOST", "").strip()
    if not host:
        try:
            host = socket.gethostbyname(socket.gethostname())
        except Exception:
            host = "127.0.0.1"
    return f"http://{host}:{PORT}"


class Storage:
    """put/get/list/exists over an opaque object key (XTTS S3 key layout)."""

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        raise NotImplementedError

    def get(self, key: str) -> bytes:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def list_keys(self, prefix: str) -> list[str]:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def url_for(self, key: str) -> str:
        raise NotImplementedError


class LocalStorage(Storage):
    def __init__(self, root: str, public_url: str) -> None:
        self.root = root
        self.public_url = public_url.rstrip("/")
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        # GET /media/{key:path} passes a client-chosen key, so every key is
        # confined to the root here: no NUL or backslash, and no `..` that
        # climbs out once normalised. A leading "/" is taken as relative.
        if "\x00" in key or "\\" in key:
            raise ValueError("invalid key")
        safe = os.path.normpath(key).lstrip("/")
        if safe == ".." or safe.startswith("../"):
            raise ValueError("invalid key")
        path = os.path.join(self.root, safe)
        if os.path.commonpath([self.root, path]) != self.root:
            raise ValueError("invalid key")
        return path

    @staticmethod
    def _open_tmp(tmp: str):
        """Open the temp file, creating its directory. A concurrent delete()
        prunes empty parents, so it can remove the directory between makedirs
        and open; make it again rather than fail the write."""
        for attempt in range(3):
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            try:
                return open(tmp, "wb")
            except FileNotFoundError:
                if attempt == 2:
                    raise

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        tmp = f"{path}.tmp-{uuid.uuid4().hex}"
        try:
            with self._open_tmp(tmp) as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            # Don't leave the partial file: list_keys() hides it and nothing else removes it.
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise
        return self.url_for(key)

    def get(self, key: str) -> bytes:
        with open(self._path(key), "rb") as f:
            return f.read()

    def exists(self, key: str) -> bool:
        return os.path.isfile(self._path(key))

    def list_keys(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        root = base if os.path.isdir(base) else os.path.dirname(base)
        out: list[str] = []
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if name.endswith(".tmp") or ".tmp-" in name:
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                if rel.startswith(prefix):
                    out.append(rel)
        return out

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if not os.path.isfile(path):
            return False
        os.remove(path)
        parent = os.path.dirname(path)
        while parent.startswith(self.root) and parent != self.root:
            try:
                os.rmdir(parent)
                parent = os.path.dirname(parent)
            except OSError:
                break
        return True

    def url_for(self, key: str) -> str:
        return f"{self.public_url}/media/{key}"


class S3Storage(Storage):
    def __init__(self) -> None:
        self.endpoint = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
        # No default credentials: a deployment that forgot them must not start
        # on MinIO's well-known root account.
        self.access_key = os.getenv("S3_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("S3_SECRET_ACCESS_KEY", "").strip()
        if not (self.access_key and self.secret_key):
            raise RuntimeError(
                "MISO_STORAGE_BACKEND=s3 needs S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY; there are no defaults"
            )
        self.bucket = os.getenv("S3_BUCKET_NAME", "voice")
        self.public_url = os.getenv("S3_PUBLIC_URL", "http://localhost:9000/voice").rstrip("/")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
            )
        return self._client

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return self.url_for(key)

    def get(self, key: str) -> bytes:
        buf = io.BytesIO()
        self.client.download_fileobj(self.bucket, key, buf)
        return buf.getvalue()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def delete(self, key: str) -> bool:
        if not self.exists(key):
            return False
        self.client.delete_object(Bucket=self.bucket, Key=key)
        return True

    def url_for(self, key: str) -> str:
        return f"{self.public_url}/{key}"


def _make_storage() -> Storage:
    if STORAGE_BACKEND == "s3":
        logger.info("storage_backend", backend="s3")
        return S3Storage()
    store = LocalStorage(DATA_DIR, _default_public_url())
    logger.info("storage_backend", backend="local", root=DATA_DIR, public_url=store.public_url)
    return store


_storage = _make_storage()


# ── Model lifecycle ───────────────────────────────────────────────────────────


class TTSState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    CPU = "cpu"
    GPU = "gpu"
    ERROR = "error"


class MisoLoadError(RuntimeError):
    """The model could not be made ready. `oom` → a CUDA out-of-memory (retryable)."""

    def __init__(self, message: str, oom: bool = False) -> None:
        super().__init__(message)
        self.oom = oom


def _is_oom(exc: BaseException) -> bool:
    with contextlib.suppress(Exception):  # classification only: anything torch can't inspect is not an OOM
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    return "out of memory" in str(exc).lower()


def _rss_mb() -> Optional[float]:
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1024**2, 1)
    except Exception:
        return None


def _mem_fields() -> dict:
    """RSS and VRAM figures for lifecycle log lines."""
    fields: dict = {"rss_mb": _rss_mb()}
    gpu = _gpu_memory_info()
    if gpu:
        fields["vram_allocated_mb"] = gpu["allocated_mb"]
        fields["vram_reserved_mb"] = gpu["reserved_mb"]
    return fields


def _empty_cuda_cache() -> None:
    """Return every free cached CUDA segment to the driver.

    cuBLAS keeps a workspace (~8 MB) from the caching allocator for as long as
    the process lives. It is usually carved out of a large segment that the
    model load allocated, so it pins that whole segment: empty_cache() alone
    left ~2 GB reserved after unload (smk762/voluptas#26). Clearing the workspaces first
    lets empty_cache() release it; cuBLAS allocates a new one on its next call.
    Only call this when no GPU work is running."""
    with contextlib.suppress(Exception):  # best-effort cache release; there may be no CUDA context
        import torch

        if not torch.cuda.is_available():
            return
        clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if clear is not None:
            torch.cuda.synchronize()
            clear()
        torch.cuda.empty_cache()


def _release_host_memory() -> None:
    """Free dropped tensors and hand the heap back to the OS."""
    gc.collect()
    _empty_cuda_cache()
    with contextlib.suppress(Exception):  # glibc keeps freed arenas mapped, so RSS would not fall without malloc_trim; elsewhere it is absent
        _libc().malloc_trim(0)


@functools.lru_cache(maxsize=1)
def _libc() -> ctypes.CDLL:
    return ctypes.CDLL("libc.so.6")


class MisoModelManager:
    """Lazy three-state lifecycle (unloaded → cpu → gpu), mirroring the XTTS
    sidecar. Miso is a single graph, so device moves are plain .to().

    Idle: MISO_KEEP_ALIVE_GPU s after the last request the model leaves the GPU,
    either parked on CPU for MISO_KEEP_ALIVE_CPU s or (CPU = 0) unloaded at once.
    A request is "in flight" from hold() until it releases the GPU lock; the
    evict loop never demotes or unloads under one."""

    def __init__(self) -> None:
        self._state = TTSState.UNLOADED
        self._error: Optional[str] = None
        self._generator = None
        self._last_used = 0.0
        self._load_lock = asyncio.Lock()
        self._gen_lock = asyncio.Lock()  # serialise GPU generation (single GPU)
        self._evict_task: Optional[asyncio.Task] = None
        # Requests between ensure_ready() and run_locked(): the GPU lock does not
        # cover them yet, but the evict loop must not pull the model away.
        self._holds = 0
        # Who holds gen_lock, for logs and /health (#9).
        self._busy_since: Optional[float] = None
        self._busy_request: Optional[str] = None
        self._busy_endpoint: Optional[str] = None

    @property
    def state(self) -> TTSState:
        return self._state

    @property
    def generator(self):
        return self._generator

    @property
    def gen_lock(self) -> asyncio.Lock:
        return self._gen_lock

    @property
    def error(self) -> Optional[str]:
        return self._error

    @asynccontextmanager
    async def hold(self):
        """Mark a request in flight; the idle evict waits until it ends."""
        self._holds += 1
        try:
            yield
        finally:
            self._holds -= 1
            self.touch()

    def _in_use(self) -> bool:
        return self._holds > 0 or self._gen_lock.locked()

    def _ready_for_generation(self) -> bool:
        if self._generator is None:
            return False
        # Demoted after ensure_ready(): an 8B generation on CPU would crawl.
        return not (DEVICE.startswith("cuda") and self._state == TTSState.CPU)

    def busy_info(self) -> Optional[dict]:
        """The request holding the GPU lock and for how long, or None."""
        if self._busy_since is None:
            return None
        return {
            "request_id": self._busy_request,
            "endpoint": self._busy_endpoint,
            "busy_s": round(time.monotonic() - self._busy_since, 1),
        }

    async def acquire_gen(self, request_id: str, endpoint: str, **log_fields) -> bool:
        """Wait up to MISO_LOCK_WAIT_S for the GPU lock. False → caller answers 503.

        On True the caller must hand the lock to run_locked() without awaiting
        anything in between; run_locked() releases it when the work returns."""
        t0 = time.monotonic()
        task = asyncio.ensure_future(self._gen_lock.acquire())
        try:
            done, _ = await asyncio.wait({task}, timeout=LOCK_WAIT_S if LOCK_WAIT_S > 0 else None)
        except BaseException:  # request cancelled while waiting
            self._abandon_acquire(task)
            raise
        if not done:
            self._abandon_acquire(task)
            logger.warning(
                "miso_gen_lock_busy", request_id=request_id, endpoint=endpoint,
                waited_s=round(time.monotonic() - t0, 1), holder=self.busy_info(),
            )
            return False
        if not self._ready_for_generation():
            # /admin/unload or /admin/evict ran between ensure_ready() and here.
            # Don't run on a None generator, or an 8B generation on CPU.
            self._gen_lock.release()
            self.touch()
            logger.warning(
                "miso_not_ready_after_lock", request_id=request_id, endpoint=endpoint,
                model_state=self._state.value,
            )
            return False
        self._busy_since = time.monotonic()
        self._busy_request = request_id
        self._busy_endpoint = endpoint
        logger.info(
            "miso_gen_lock_acquired", request_id=request_id, endpoint=endpoint,
            wait_s=round(self._busy_since - t0, 2), **log_fields,
        )
        return True

    def _abandon_acquire(self, task: asyncio.Future) -> None:
        # If the acquire wins the race with cancel(), give the lock straight back.
        task.cancel()

        def _give_back(t: asyncio.Future) -> None:
            if not t.cancelled() and t.exception() is None:
                self._gen_lock.release()
                self.touch()  # the evict loop may have skipped while we held it

        task.add_done_callback(_give_back)

    def run_locked(self, fn, *args) -> asyncio.Future:
        """Run fn(*args) in a worker thread that owns the GPU lock.

        The lock is released when the thread returns, whatever becomes of the
        request, so a cancelled or abandoned client cannot leak it. A thread that
        never returns keeps it; /health and the stuck watchdog report that.

        The caller gets a shield over the work: cancelling it (wait_for, a
        disconnect) must not mark the work done, or the lock would be released
        while the thread is still generating on the GPU."""
        work = asyncio.get_running_loop().run_in_executor(None, fn, *args)

        def _done(f: asyncio.Future) -> None:
            if not f.cancelled():
                f.exception()  # mark retrieved; a request that timed out won't read it
            self._release_gen()

        work.add_done_callback(_done)
        fut = asyncio.shield(work)
        # The /stream caller never awaits its future: don't log its exception as unretrieved.
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())
        return fut

    def _release_gen(self) -> None:
        held = round(time.monotonic() - self._busy_since, 2) if self._busy_since else None
        logger.info("miso_gen_lock_released", request_id=self._busy_request, endpoint=self._busy_endpoint, held_s=held)
        self._busy_since = self._busy_request = self._busy_endpoint = None
        self._gen_lock.release()
        self.touch()

    def _load(self) -> None:
        from generator import load_miso_8b

        logger.info("loading_miso", model=MODEL_SOURCE, device=DEVICE, **_mem_fields())
        gen = load_miso_8b(DEVICE, model_path_or_repo_id=MODEL_SOURCE)
        self._generator = gen
        global SAMPLE_RATE
        SAMPLE_RATE = int(getattr(gen, "sample_rate", SAMPLE_RATE))
        logger.info("miso_loaded", sample_rate=SAMPLE_RATE, **_mem_fields())

    def _move(self, device: str) -> None:
        import torch

        gen = self._generator
        if gen is None:
            return
        gen._model.to(device=torch.device(device))
        # Keep auxiliary modules (mimi codec, watermarker) on the same device.
        for attr in ("_audio_tokenizer", "_watermarker"):
            mod = getattr(gen, attr, None)
            if mod is not None and hasattr(mod, "to"):
                try:
                    mod.to(torch.device(device))
                except Exception:
                    logger.warning("aux_module_move_failed", module=attr, device=device, exc_info=True)
        gen.device = torch.device(device)
        if device == "cpu":
            _empty_cuda_cache()

    async def ensure_ready(self) -> None:
        """Load or promote the model to MISO_DEVICE. Raises MisoLoadError.

        A failed load or promotion keeps nothing: no partial copy stays in RAM
        or VRAM, so a GPU OOM can simply be retried later."""
        async with self._load_lock:
            failure: Optional[tuple[str, bool, str]] = None
            if self._state in (TTSState.UNLOADED, TTSState.ERROR):
                self._state = TTSState.LOADING
                try:
                    await asyncio.to_thread(self._load)
                    self._state = TTSState.GPU if DEVICE.startswith("cuda") else TTSState.CPU
                    self._error = None
                except Exception as exc:
                    failure = (str(exc), _is_oom(exc), "miso_load_failed")
                    # The traceback's frames still reference whatever load_miso_8b
                    # had built; clear them so that copy can be freed.
                    traceback.clear_frames(exc.__traceback__)
            elif self._state == TTSState.CPU and DEVICE.startswith("cuda"):
                try:
                    await asyncio.to_thread(self._move, DEVICE)
                    self._state = TTSState.GPU
                    logger.info("miso_promoted_gpu", **_mem_fields())
                except Exception as exc:
                    # Weights may now be split across devices; start over instead.
                    failure = (str(exc), _is_oom(exc), "miso_promote_failed")
                    traceback.clear_frames(exc.__traceback__)
            if failure is not None:
                # Raised outside the except block so no __context__ keeps the
                # original exception (and its frames) alive.
                message, oom, event = failure
                self._generator = None
                # gc + malloc_trim over a ~17 GB heap: keep it off the event loop.
                await asyncio.to_thread(_release_host_memory)
                self._state = TTSState.UNLOADED if oom else TTSState.ERROR
                self._error = message
                logger.error(event, error=message, oom=oom, **_mem_fields())
                raise MisoLoadError(message, oom=oom)
            self._last_used = time.monotonic()
            self._arm_evict()

    async def demote_to_cpu(self, if_idle: bool = False) -> None:
        async with self._load_lock:
            if self._state != TTSState.GPU or not DEVICE.startswith("cuda"):
                return
            if if_idle and self._in_use():
                return  # a request arrived while the evict loop waited for the lock
            failure: Optional[str] = None
            # Never move weights while a generation (incl. a stream) is running.
            async with self._gen_lock:
                try:
                    await asyncio.to_thread(self._move, "cpu")
                except Exception as exc:
                    # Weights may now be split across devices, as after a failed
                    # promotion in ensure_ready(): drop the model, don't keep "gpu".
                    failure = str(exc)
                    traceback.clear_frames(exc.__traceback__)
                    self._generator = None
            if failure is not None:
                await asyncio.to_thread(_release_host_memory)
                self._state = TTSState.UNLOADED
                self._error = failure
                logger.error("miso_demote_failed", error=failure, **_mem_fields())
                return
            self._state = TTSState.CPU
            logger.info("miso_demoted_cpu", keep_alive_cpu=KEEP_ALIVE_CPU, **_mem_fields())

    async def unload(self, if_idle: bool = False) -> None:
        async with self._load_lock:
            if if_idle and self._in_use():
                return  # a request arrived while the evict loop waited for the lock
            # The evict loop itself calls unload(); don't cancel ourselves.
            if (
                self._evict_task
                and not self._evict_task.done()
                and self._evict_task is not asyncio.current_task()
            ):
                self._evict_task.cancel()
            if self._generator is None and self._state == TTSState.UNLOADED:
                return
            from_state = self._state.value
            before = _mem_fields()
            # Wait for any in-flight generation before dropping the model.
            async with self._gen_lock:
                self._generator = None
            await asyncio.to_thread(_release_host_memory)
            self._state = TTSState.UNLOADED
            after = _mem_fields()
            logger.info(
                "miso_unloaded", from_state=from_state,
                rss_mb_before=before.get("rss_mb"), **after,
            )

    def touch(self) -> None:
        self._last_used = time.monotonic()
        self._arm_evict()

    def _arm_evict(self) -> None:
        if self._evict_task and not self._evict_task.done():
            self._evict_task.cancel()
        self._evict_task = None
        if KEEP_ALIVE_GPU < 0 or self._state == TTSState.UNLOADED:
            return  # pinned to the GPU, or nothing to evict
        self._evict_task = asyncio.create_task(self._evict_loop())

    def _idle_s(self) -> float:
        return time.monotonic() - self._last_used

    async def _evict_loop(self) -> None:
        # Whoever is using the model re-arms this on release (hold() exit,
        # _release_gen), so a busy check just ends the loop. Re-arming here
        # would spin when a keep-alive is 0.
        try:
            await asyncio.sleep(max(KEEP_ALIVE_GPU, 0))
            if self._in_use() or self._idle_s() < KEEP_ALIVE_GPU:
                return
            # Shielded: a re-arm cancels this task, and a transition cut off
            # half-way would leave _state out of step with the weights.
            if KEEP_ALIVE_CPU == 0:
                await asyncio.shield(self.unload(if_idle=True))  # skip the CPU stage: free the RAM too
                return
            await asyncio.shield(self.demote_to_cpu(if_idle=True))
            if KEEP_ALIVE_CPU < 0:
                return  # parked on CPU for ever
            await asyncio.sleep(KEEP_ALIVE_CPU)
            if self._in_use() or self._idle_s() < KEEP_ALIVE_GPU + KEEP_ALIVE_CPU:
                return
            await asyncio.shield(self.unload(if_idle=True))
        except asyncio.CancelledError:
            pass
        except Exception:
            # Nothing awaits this task; log instead of dying with an unretrieved exception.
            logger.error("miso_evict_failed", state=self._state.value, exc_info=True)


_mgr = MisoModelManager()
_clone_cache: OrderedDict = OrderedDict()


def _gpu_memory_info() -> dict | None:
    with contextlib.suppress(Exception):  # /health reports memory when it can; a failed probe omits it
        import torch

        if torch.cuda.is_available():
            return {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
                "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1),
            }
    return None


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return slug or f"voice-{int(time.time())}"


# Companion ids become storage path segments (`voice_clones/{id}/...`), so they
# are restricted to a strict allowlist: no '/', '\\', '.', '..' or empty ids
# that could escape the clone namespace or address every clone at once.
_COMPANION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CLONE_KEY_RE = re.compile(r"^voice_clones/([A-Za-z0-9_-]{1,64})/prompt\.pt$")


def _is_valid_companion_id(companion_id) -> bool:
    return isinstance(companion_id, str) and bool(_COMPANION_ID_RE.fullmatch(companion_id))


def _clone_prompt_key(companion_id: str) -> str:
    """Map a companion id or full `voice_clones/{id}/prompt.pt` key to its key.

    Anything else (traversal, other prefixes, unsafe characters) is rejected so
    callers can't steer storage reads onto arbitrary objects.
    """
    cid = (companion_id or "").strip()
    if "/" in cid:  # already a full key
        if not _CLONE_KEY_RE.fullmatch(cid):
            raise ValueError(f"Invalid voice clone id: {companion_id!r}")
        return cid
    if not _is_valid_companion_id(cid):
        raise ValueError(f"Invalid voice clone id: {companion_id!r}")
    return f"voice_clones/{cid}/prompt.pt"


def _load_clone(clone_id: str):
    """Load a stored Miso clone {ref_audio, ref_text, speaker} by its key."""
    key = _clone_prompt_key(clone_id)
    if key in _clone_cache:
        _clone_cache.move_to_end(key)
        return _clone_cache[key]

    import torch

    try:
        raw = _storage.get(key)
    except Exception as exc:
        raise ValueError(f"Voice clone not found: {clone_id}") from exc
    # weights_only=True: the prompt is tensors + str/int, so never run the full
    # pickle unpickler on stored bytes (that would be code execution).
    try:
        prompt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Invalid or untrusted voice clone: {clone_id}") from exc
    if not (isinstance(prompt, dict) and torch.is_tensor(prompt.get("ref_audio"))):
        raise ValueError(f"Voice clone is malformed (no ref_audio tensor): {clone_id}")
    if not isinstance(prompt.get("ref_text", ""), str):
        raise ValueError(f"Voice clone is malformed (ref_text): {clone_id}")
    if not isinstance(prompt.get("speaker", 0), int):
        raise ValueError(f"Voice clone is malformed (speaker): {clone_id}")

    _clone_cache[key] = prompt
    if len(_clone_cache) > _CLONE_CACHE_MAX:
        _clone_cache.popitem(last=False)
    return prompt


def _clamp_temperature(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("temperature must be finite")
    return min(TEMPERATURE_MAX, max(TEMPERATURE_MIN, number))


def _clamp_topk(value: int) -> int:
    number = int(value)
    return min(TOPK_MAX, max(TOPK_MIN, number))


def _coerce_temperature(value) -> float | None:
    """Return a clamped temperature, or None when the value is absent or unusable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return _clamp_temperature(value)
    except (TypeError, ValueError):
        return None


def _coerce_topk(value) -> int | None:
    """Return a clamped top-k, or None when the value is absent or unusable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return _clamp_topk(int(round(number)))


# Env defaults, clamped once at import (fallbacks only for a NaN/inf env value).
_DEFAULT_TEMPERATURE = _coerce_temperature(TEMPERATURE) or 0.9
_DEFAULT_TOPK = _coerce_topk(TOPK) or 50


def _sampling_params(req, meta: dict | None) -> tuple[float, int]:
    """Resolve temperature and top-k.

    Request fields win, then the clone's meta.json, then MISO_TEMPERATURE /
    MISO_TOPK. Out-of-range numbers are clamped; junk in meta.json is ignored
    so a hand-edited sidecar cannot fail synthesis.
    """
    meta = meta or {}
    # Coerced values are clamped to >= the minimums, so `or` only skips None.
    temperature = (
        _coerce_temperature(getattr(req, "temperature", None))
        or _coerce_temperature(meta.get("temperature"))
        or _DEFAULT_TEMPERATURE
    )
    topk = (
        _coerce_topk(getattr(req, "topk", None))
        or _coerce_topk(meta.get("topk"))
        or _DEFAULT_TOPK
    )
    return temperature, topk


def _sampling_meta_fields(src: dict | None) -> dict:
    """Copy valid per-voice sampling defaults so import does not drop them."""
    src = src or {}
    kept: dict = {}
    temperature = _coerce_temperature(src.get("temperature"))
    if temperature is not None:
        kept["temperature"] = temperature
    topk = _coerce_topk(src.get("topk"))
    if topk is not None:
        kept["topk"] = topk
    return kept


def _clone_meta_key(clone_id: str) -> str:
    prompt_key = _clone_prompt_key(clone_id)
    if prompt_key.endswith("prompt.pt"):
        return prompt_key[: -len("prompt.pt")] + "meta.json"
    parent, _, _ = prompt_key.rpartition("/")
    return f"{parent}/meta.json" if parent else "meta.json"


def _load_clone_meta(clone_id: str) -> dict:
    """Read a clone's meta.json. Missing or unreadable sidecars yield {}."""
    key = _clone_meta_key(clone_id)
    try:
        if not _storage.exists(key):
            return {}
        parsed = json.loads(_storage.get(key).decode("utf-8"))
    except Exception as exc:
        logger.warning("clone_meta_unreadable", key=key, error=str(exc))
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ── Length budget (#3) ───────────────────────────────────────────────────────


# Han, kana and Hangul are written without spaces; each character is about a
# syllable aloud, so each counts as a word.
_CJK = "぀-ヿ㐀-䶿一-鿿가-힯豈-﫿"
# One spoken unit each: a letter run (apostrophes kept inside, so "don't" is
# one), a single digit, a CJK character, or a symbol read as a word.
_SPOKEN_UNIT_RE = re.compile(
    rf"[{_CJK}]|(?:(?![{_CJK}])[^\W\d_])+(?:['’](?:(?![{_CJK}])[^\W\d_])+)*|\d|[$€£%&@#+=]"
)


def _word_count(text: str) -> int:
    """Spoken-word estimate. Letter runs split on any non-letter, so
    "state-of-the-art" and URLs count each part; each digit counts as a word
    ("555-123-4567" is ~10 words aloud, not 1); unspaced CJK counts per character."""
    return max(1, len(_SPOKEN_UNIT_RE.findall(text or "")))


def _max_audio_ms_for(text: str) -> float:
    """Longest audio this text may produce: a slow reading plus a pad, at most
    MISO_MAX_AUDIO_MS. Words, not characters, so numbers written out as words
    stay proportionate; digit strings count roughly one word per digit."""
    if MIN_WORDS_PER_S <= 0:
        return MAX_AUDIO_MS
    seconds = LENGTH_PAD_S + _word_count(text) / MIN_WORDS_PER_S
    return min(MAX_AUDIO_MS, seconds * 1000.0)


def _max_frames(max_audio_ms: float) -> int:
    # Same arithmetic as Generator.generate / generate_stream.
    return int(max_audio_ms / FRAME_MS)


def _frame_samples() -> int:
    return SAMPLE_RATE * FRAME_MS // 1000


def _log_if_capped(text: str, max_audio_ms: float, n_samples: int, endpoint: str) -> bool:
    """Log when a generation ran to its budget instead of ending on its own."""
    frames = round(n_samples / _frame_samples())
    if frames < _max_frames(max_audio_ms):
        return False
    logger.warning(
        "miso_generation_capped",
        endpoint=endpoint,
        words=_word_count(text),
        budget_s=round(max_audio_ms / 1000.0, 2),
        audio_s=round(n_samples / SAMPLE_RATE, 2),
        at_ceiling=max_audio_ms >= MAX_AUDIO_MS,
    )
    return True


# ── Edge fades (#5) ───────────────────────────────────────────────────────────
# The model's end-of-speech frame often lands before the last sound has
# decayed, so utterances stop dead (a click, or a clipped last word) and start
# on a step. Both paths fade the edges with the same per-sample gains, so a
# concatenated stream equals the faded /synthesise output.


def _ms_to_samples(ms: float) -> int:
    return max(0, int(round(ms * SAMPLE_RATE / 1000.0)))


def _fade_in(audio, start: int, fade_len: int):
    """Raised-cosine fade-in indexed by absolute sample position, so a stream
    can apply it chunk by chunk. Sample 0 gets gain 0."""
    import numpy as np

    if fade_len <= 0 or start >= fade_len or audio.size == 0:
        return audio
    end = min(fade_len, start + audio.size)
    idx = np.arange(start, end, dtype=np.float64)
    gain = (0.5 - 0.5 * np.cos(np.pi * idx / fade_len)).astype(np.float32)
    out = audio.copy()
    out[: end - start] *= gain
    return out


def _fade_out(audio, fade_len: int):
    """Fade the last min(len, fade_len) samples to exactly 0.

    Squared raised cosine: smooth at both ends like a plain one, but it is
    already below -24 dB over the final third, so the last ~10 ms of a 30 ms
    fade sits near silence even when speech stops at full level."""
    import numpy as np

    n = min(audio.size, fade_len)
    if n <= 0:
        return audio
    t = np.arange(1, n + 1, dtype=np.float64) / n
    gain = (((1.0 + np.cos(np.pi * t)) / 2.0) ** 2).astype(np.float32)
    out = audio.copy()
    out[-n:] *= gain
    return out


def _apply_fades(audio):
    """Fade a whole utterance in and out."""
    audio = _fade_in(audio, 0, _ms_to_samples(FADE_IN_MS))
    return _fade_out(audio, _ms_to_samples(FADE_OUT_MS))


class _EdgeFader:
    """Streaming form of _apply_fades.

    push() fades in by absolute position and holds back the last fade-out
    length of audio, since any chunk might be the last. finish() fades that tail
    out. Chunk boundaries are unchanged apart from the held-back tail."""

    def __init__(self) -> None:
        import numpy as np

        self._fade_in_len = _ms_to_samples(FADE_IN_MS)
        self._fade_out_len = _ms_to_samples(FADE_OUT_MS)
        self._emitted = 0
        self._tail = np.zeros(0, dtype=np.float32)

    def push(self, chunk):
        import numpy as np

        buf = np.concatenate([self._tail, chunk]) if self._tail.size else chunk
        cut = max(0, buf.size - self._fade_out_len)
        out, self._tail = buf[:cut], buf[cut:]
        out = _fade_in(out, self._emitted, self._fade_in_len)
        self._emitted += out.size
        return out

    def finish(self):
        import numpy as np

        tail = _fade_in(self._tail, self._emitted, self._fade_in_len)
        tail = _fade_out(tail, self._fade_out_len)
        self._emitted += tail.size
        self._tail = np.zeros(0, dtype=np.float32)
        return tail


# ── Request context (#4) ──────────────────────────────────────────────────────
# A client may send back its previous utterance(s) (text + the audio Miso
# returned) so the next sentence keeps the same delivery. Stateless: nothing is
# kept between requests. Order is [clone reference, *request context], and
# when the prompt would not fit, the oldest request segments are dropped first.

# Context audio far outside [-1, 1] means the client sent the wrong format
# (e.g. s16 bytes), not quiet or slightly hot speech.
_CONTEXT_MAX_ABS = 4.0


def _decode_context_audio(b64: str, index: int):
    """base64 pcm_f32le (24 kHz mono) → float32 numpy array, or ValueError."""
    import base64
    import binascii

    import numpy as np

    where = f"context[{index}].audio"
    max_samples = int(CONTEXT_MAX_SECONDS * SAMPLE_RATE)
    # Line-wrapped base64 (MIME style, `base64` CLI) is still valid input.
    b64 = "".join((b64 or "").split())
    # Refuse oversized payloads before decoding them.
    if len(b64 or "") > 4 * math.ceil(max_samples * 4 / 3) + 16:
        raise ValueError(f"{where} is longer than {CONTEXT_MAX_SECONDS:g} s")
    try:
        raw = base64.b64decode(b64 or "", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{where} is not valid base64: {exc}") from exc
    if not raw:
        raise ValueError(f"{where} is empty")
    if len(raw) % 4:
        raise ValueError(f"{where} must be pcm_f32le (a multiple of 4 bytes); got {len(raw)} bytes")
    samples = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{where} contains NaN or infinite samples")
    if float(np.max(np.abs(samples))) > _CONTEXT_MAX_ABS:
        raise ValueError(f"{where} is out of range; expected pcm_f32le 24 kHz mono in [-1, 1]")
    return samples


def _request_context_segments(req, speaker: int) -> list:
    """Validate and decode req.context into Segments with the request's speaker."""
    items = getattr(req, "context", None) or []
    if not items:
        return []
    if len(items) > CONTEXT_MAX_SEGMENTS:
        raise ValueError(f"context allows at most {CONTEXT_MAX_SEGMENTS} segments; got {len(items)}")

    import torch
    from generator import Segment

    max_samples = int(CONTEXT_MAX_SECONDS * SAMPLE_RATE)
    total = 0
    segments = []
    for index, item in enumerate(items):
        text = (item.text or "").strip()
        if not text:
            raise ValueError(f"context[{index}].text is required")
        samples = _decode_context_audio(item.audio, index)
        total += samples.size
        if total > max_samples:
            raise ValueError(f"context audio totals more than {CONTEXT_MAX_SECONDS:g} s")
        segments.append(Segment(speaker=speaker, text=text, audio=torch.from_numpy(samples)))
    return segments


def _text_token_count(generator, text: str, speaker: int) -> int:
    """Prompt length of a text segment, as Generator._tokenize_text_segment builds it."""
    return len(generator._text_tokenizer.encode(f"[{speaker}] {text.lstrip()}"))


def _segment_prompt_len(segment, generator) -> int:
    # Text tokens, one frame per started 80 ms of audio, plus the audio EOS frame.
    frames = math.ceil(int(segment.audio.shape[-1]) / _frame_samples())
    return _text_token_count(generator, segment.text, segment.speaker) + frames + 1


def _fit_context(base: list, extra: list, text: str, speaker: int, max_audio_ms: float, generator) -> list:
    """Return base + as many of the newest ``extra`` segments as fit the prompt.

    The prompt must stay below max_seq_len minus the generation budget. ``base``
    (the clone reference) is never dropped; ``extra`` loses its oldest first."""
    # Generator.generate / generate_stream hard-code max_seq_len = 2048.
    max_seq_len = int(getattr(generator, "max_seq_len", 2048))
    limit = max_seq_len - _max_frames(max_audio_ms) - CONTEXT_TOKEN_MARGIN
    used = sum(_segment_prompt_len(s, generator) for s in base) + _text_token_count(generator, text, speaker)
    if used >= limit:
        # Same check Generator makes, raised here (with or without ``extra``) so
        # /synthesise/stream answers 422 instead of a 200 with an empty body.
        raise ValueError(f"voice reference and text are too long for the prompt ({used} >= {limit} tokens)")
    if not extra:
        return base
    lengths = [_segment_prompt_len(s, generator) for s in extra]
    kept = list(extra)
    while kept and used + sum(lengths) >= limit:
        kept.pop(0)
        lengths.pop(0)
    if len(kept) < len(extra):
        logger.warning(
            "miso_context_trimmed",
            dropped=len(extra) - len(kept),
            kept=len(kept),
            prompt_tokens=used + sum(lengths),
            limit=limit,
        )
    return base + kept


def _resolve_context(req):
    """Return (speaker_id, context, voice_meta) for a synthesis request.

    With a voice_clone_id we build a one-shot conditioning Segment from the
    stored reference (audio + transcript); otherwise context is empty and we use
    the requested/default integer speaker. voice_meta is the clone's meta.json
    (or {}) so callers can apply per-voice sampling defaults such as temperature.
    """
    from generator import Segment

    clone_id = getattr(req, "voice_clone_id", None)
    if clone_id:
        prompt = _load_clone(clone_id)
        import torch

        ref_audio = prompt["ref_audio"]
        if not isinstance(ref_audio, torch.Tensor):
            ref_audio = torch.tensor(ref_audio)
        ref_audio = ref_audio.to(torch.float32).flatten()
        speaker = int(prompt.get("speaker", DEFAULT_SPEAKER))
        ref_text = prompt.get("ref_text", "") or ""
        return speaker, [Segment(speaker=speaker, text=ref_text, audio=ref_audio)], _load_clone_meta(clone_id)

    speaker = DEFAULT_SPEAKER
    raw_spk = getattr(req, "speaker", None)
    if raw_spk:
        try:
            speaker = int(str(raw_spk).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"Miso speakers are integer ids; got {raw_spk!r}. Use a voice_clone_id for custom voices."
            )
    return speaker, [], {}


def _prepare_generation(req):
    """Resolve speaker, context, temperature/top-k and the length budget.

    Returns (speaker, context, temperature, topk, max_audio_ms). Raises
    ValueError (→ 422) for a bad clone, speaker or request context."""
    speaker, context, voice_meta = _resolve_context(req)
    temperature, topk = _sampling_params(req, voice_meta)
    max_audio_ms = _max_audio_ms_for(req.text)
    extra = _request_context_segments(req, speaker)
    context = _fit_context(context, extra, req.text, speaker, max_audio_ms, _mgr.generator)
    logger.info(
        "miso_sampling",
        temperature=temperature,
        topk=topk,
        voice_clone_id=getattr(req, "voice_clone_id", None),
        max_audio_s=round(max_audio_ms / 1000.0, 2),
        request_context=len(extra),
    )
    return speaker, context, temperature, topk, max_audio_ms


def _load_audio_24k_mono(path: str):
    """Load any audio file to a 1-D float32 torch tensor at 24 kHz mono."""
    import torch
    import torchaudio
    import torchaudio.functional as AF

    wav, sr = torchaudio.load(path)  # (channels, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = AF.resample(wav, sr, SAMPLE_RATE)
    return wav.squeeze(0).to(torch.float32).contiguous()


def _padded_speech_spans(timestamps: list[dict], pad: float, sr: int, total: int) -> list[tuple[int, int]]:
    """Speech timestamps (seconds) → sample spans widened by ``pad`` seconds,
    with overlapping spans merged. Silero allows gaps shorter than two pads,
    and slicing overlapping spans would repeat the audio between them."""
    spans: list[list[int]] = []
    for t in sorted(timestamps, key=lambda t: t["start"]):
        a = max(0, int((t["start"] - pad) * sr))
        b = min(total, int((t["end"] + pad) * sr))
        if b <= a:
            continue
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return [(a, b) for a, b in spans]


def _preprocess_reference(path: str) -> tuple["object", dict]:
    """Standardise a reference clip for cloning: mono → 24 kHz → DC removal →
    high-pass → VAD trim → length cap → peak/loudness guard. Returns
    (samples_tensor_24k, report). Degrades gracefully per stage."""
    import numpy as np
    import torch
    import torchaudio.functional as AF

    report: dict = {"applied": [], "warnings": []}
    samples = _load_audio_24k_mono(path)
    sr = SAMPLE_RATE
    report["original_seconds"] = round(samples.shape[-1] / sr, 2)
    report["applied"].append("mono+24k")

    samples = samples - samples.mean()
    samples = AF.highpass_biquad(samples.unsqueeze(0), sr, 85.0).squeeze(0)
    report["applied"].append("highpass85")

    if REF_PREPROCESS:
        try:
            from silero_vad import get_speech_timestamps

            model = _get_vad_model()
            vad_wav = AF.resample(samples.unsqueeze(0), sr, 16000).squeeze(0)
            ts = get_speech_timestamps(vad_wav, model, sampling_rate=16000, return_seconds=True)
            if ts:
                segs = [samples[a:b] for a, b in _padded_speech_spans(ts, 0.1, sr, samples.shape[-1])]
                if segs:
                    samples = torch.cat(segs)
                    report["applied"].append("vad_trim")
            else:
                report["warnings"].append("vad_no_speech")
        except Exception as exc:
            report["warnings"].append(f"vad_failed:{exc}")

    max_samps = int(REF_MAX_SECONDS * sr)
    if samples.shape[-1] > max_samps:
        samples = samples[:max_samps]
        report["applied"].append(f"capped{int(REF_MAX_SECONDS)}s")

    audio_np = samples.numpy().astype(np.float32)
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loud_in = meter.integrated_loudness(audio_np)
        if np.isfinite(loud_in):
            normed = pyln.normalize.loudness(audio_np, loud_in, -23.0)
            peak = float(np.max(np.abs(normed))) if normed.size else 0.0
            if peak > 0.99:
                normed = normed * (0.99 / peak)
            audio_np = normed.astype(np.float32)
            report["applied"].append("lufs_norm")
    except Exception as exc:
        report["warnings"].append(f"lufs_failed:{exc}")
        peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
        if peak > 0:
            audio_np = (audio_np * (0.97 / peak)).astype(np.float32)
            report["applied"].append("peak_norm")

    report["kept_seconds"] = round(len(audio_np) / sr, 2)
    return torch.from_numpy(audio_np), report


_vad_model = None


def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad

        _vad_model = load_silero_vad()
    return _vad_model


def _ref_wav_bytes(ref_audio) -> bytes:
    """The stored reference tensor as a WAV file, for the transcriber."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, _to_numpy_audio(ref_audio), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _transcribe_reference(wav_bytes: bytes) -> str:
    """Best-effort transcript for a clone reference via an external Whisper
    sidecar (MISO_WHISPER_URL). Returns "" on any failure — Miso can still use
    the audio as acoustic context, just with lower fidelity."""
    if not WHISPER_URL:
        return ""
    try:
        import httpx

        files = {"audio": ("ref.wav", wav_bytes, "audio/wav")}
        with httpx.Client(timeout=60) as client:
            r = client.post(f"{WHISPER_URL}/transcribe", files=files)
            r.raise_for_status()
            return (r.json().get("text") or "").strip()
    except Exception as exc:
        logger.warning("clone_transcribe_failed", error=str(exc))
        return ""


# ── App + request models ──────────────────────────────────────────────────────


def _is_stuck(busy: Optional[dict]) -> bool:
    return bool(busy) and busy["busy_s"] > STUCK_AFTER_S


async def _stuck_watchdog() -> None:
    """Exit when the GPU lock has been held past any possible generation.

    A thread stuck inside the model can't be stopped from Python, and Docker
    does not restart a container for being unhealthy, so leaving the process is
    the only way back: `restart: unless-stopped` starts a fresh one."""
    import sys

    while True:
        await asyncio.sleep(STUCK_CHECK_S)
        busy = _mgr.busy_info()
        if not _is_stuck(busy):
            continue
        logger.error("miso_stuck", **busy, stuck_after_s=STUCK_AFTER_S, exiting=STUCK_EXIT)
        if STUCK_EXIT:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(70)  # EX_SOFTWARE; skips shutdown, which would wait on the lock


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRELOAD:
        try:
            await _mgr.ensure_ready()
        except Exception:
            logger.warning("preload_failed", hint="the first request retries the load; /health shows the error", exc_info=True)
    watchdog = asyncio.create_task(_stuck_watchdog())
    yield
    watchdog.cancel()
    await _mgr.unload()


app = FastAPI(title="Voice Stack — Miso TTS 8B", version="1.0.0", lifespan=lifespan)


class _BodyOverCap(Exception):
    pass


class _UploadBodyCap:
    """Pure-ASGI guard for the upload routes (smk762/voluptas#30).

    A Content-Length over the route's cap is 413 before the body is read, so
    the multipart parser never runs and nothing is spooled. A chunked body is
    counted as it streams; once over the cap the parser's next read fails, so
    at most the cap reaches disk, and whatever the app would answer is
    replaced by the 413. A malformed Content-Length is 400."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        cap_for = _UPLOAD_BODY_CAPS.get(scope.get("path", "")) if scope["type"] == "http" else None
        if cap_for is None or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        cap = cap_for()
        raw = dict(scope["headers"]).get(b"content-length")
        if raw is not None:
            raw = raw.strip()
            if not raw.isdigit():
                return await JSONResponse({"error": "Content-Length is not a number"}, 400)(scope, receive, send)
            if int(raw) > cap:
                return await self._refuse(scope, receive, send, cap)
        seen = 0
        over = False

        async def counted():
            nonlocal seen, over
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > cap:
                    over = True
                    raise _BodyOverCap()
            return message

        async def guarded(message):
            if not over:
                await send(message)

        try:
            await self.app(scope, counted, guarded)
        except _BodyOverCap:
            pass
        if over:
            await self._refuse(scope, receive, send, cap)

    @staticmethod
    async def _refuse(scope, receive, send, cap):
        logger.warning("upload_body_too_large", path=scope.get("path"), cap_bytes=cap)
        body = {"error": f"upload is larger than {cap} bytes"}
        await JSONResponse(body, 413, headers={"Connection": "close"})(scope, receive, send)


app.add_middleware(_UploadBodyCap)


class ContextSegment(BaseModel):
    text: str
    # base64 pcm_f32le, 24 kHz mono: the bytes /synthesise/stream returned.
    audio: str


class SynthesiseRequest(BaseModel):
    text: str
    voice_clone_id: Optional[str] = None
    speaker: Optional[str] = None
    voice_description: Optional[str] = None  # unused by Miso; accepted for compat
    instruction: Optional[str] = None        # unused by Miso; accepted for compat
    emotion: str = "neutral"                  # unused by Miso; accepted for compat
    language: str = Field(default="en")
    speed: Optional[float] = None             # unused by Miso; accepted for compat
    # None → clone meta.json (when present) → MISO_TEMPERATURE / MISO_TOPK.
    temperature: Optional[float] = Field(default=None, allow_inf_nan=False)
    topk: Optional[int] = None
    # Previous utterance(s), oldest first, placed after the clone's reference.
    # Omitted → behaviour is unchanged.
    context: Optional[list[ContextSegment]] = None


def _to_numpy_audio(audio):
    """Generator output tensor → 1-D float32 numpy array."""
    import numpy as np

    arr = audio.detach().cpu()
    return np.asarray(arr.to_dense() if hasattr(arr, "to_dense") else arr, dtype=np.float32).flatten()


def _generate_audio(req: SynthesiseRequest, prepared=None):
    """Blocking generation → 1-D float32 numpy array @ SAMPLE_RATE, faded in
    and out. Caller holds the generation lock. ``prepared`` is a precomputed
    _prepare_generation()."""
    speaker, context, temperature, topk, max_audio_ms = prepared or _prepare_generation(req)
    audio = _mgr.generator.generate(
        text=req.text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=max_audio_ms,
        temperature=temperature,
        topk=topk,
    )
    data = _to_numpy_audio(audio)
    _log_if_capped(req.text, max_audio_ms, data.size, "synthesise")
    return _apply_fades(data)


def _generate_stream_chunks(req: SynthesiseRequest, stop, prepared=None):
    """Blocking generator → yields 1-D float32 numpy arrays @ SAMPLE_RATE as the
    model produces them (incremental Mimi decode). Caller holds the generation
    lock for the entire iteration. ``stop`` is a threading.Event used to abort
    early (e.g. client disconnect)."""
    speaker, context, temperature, topk, max_audio_ms = prepared or _prepare_generation(req)
    fader = _EdgeFader()
    raw_samples = 0
    for audio in _mgr.generator.generate_stream(
        text=req.text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=max_audio_ms,
        temperature=temperature,
        topk=topk,
        first_chunk_frames=STREAM_FIRST_FRAMES,
        decode_chunk_frames=STREAM_DECODE_FRAMES,
    ):
        data = _to_numpy_audio(audio)
        raw_samples += data.size
        out = fader.push(data)
        if out.size:
            yield out
        if stop.is_set():
            return
    _log_if_capped(req.text, max_audio_ms, raw_samples, "stream")
    tail = fader.finish()
    if tail.size:
        yield tail


# ── POST /synthesise ──────────────────────────────────────────────────────────


def _busy_response() -> JSONResponse:
    if _mgr.busy_info() is None and not _mgr._ready_for_generation():
        # acquire_gen() refused because the model was unloaded/demoted after
        # ensure_ready(); a retry reloads it.
        return JSONResponse(
            status_code=503,
            content={"error": "Model was unloaded while the request waited; retry", "model_state": _mgr.state.value},
            headers={"Retry-After": "1"},
        )
    return JSONResponse(
        status_code=503,
        content={"error": "TTS busy: another generation holds the GPU", "holder": _mgr.busy_info()},
        headers={"Retry-After": "5"},
    )


def _oom_response(what: str, exc: BaseException) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error": f"GPU OOM while {what}", "detail": str(exc), "gpu": _gpu_memory_info()},
        headers={"Retry-After": str(OOM_RETRY_AFTER_S)},
    )


def _load_failed_response(exc: BaseException) -> JSONResponse:
    """503 for ensure_ready() failing; an OOM gets Retry-After (nothing stays loaded)."""
    if getattr(exc, "oom", False) or _is_oom(exc):
        return _oom_response("loading the model", exc)
    return JSONResponse(status_code=503, content={"error": f"Model load failed: {exc}"})


@app.post("/synthesise")
async def synthesise(req: SynthesiseRequest):
    """Generate speech, store the WAV, return its URL (XTTS-compatible shape)."""
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})
    if not (req.text or "").strip():
        return JSONResponse(status_code=422, content={"error": "text is required"})

    # In flight from here until run_locked() owns the GPU lock: the idle
    # evict must not unload the model under _prepare_generation().
    async with _mgr.hold():
        try:
            await _mgr.ensure_ready()
        except Exception as exc:
            return _load_failed_response(exc)

        job_id = str(uuid.uuid4())
        t0 = time.monotonic()
        try:
            # Clone/meta storage I/O happens before taking the single-GPU lock.
            prepared = await asyncio.to_thread(_prepare_generation, req)
            if not await _mgr.acquire_gen(job_id, "synthesise", voice_clone_id=req.voice_clone_id):
                return _busy_response()
            fut = _mgr.run_locked(_generate_audio, req, prepared)
            limit_s = _generation_deadline_s(prepared[4])
            done, _ = await asyncio.wait({fut}, timeout=limit_s)
            if not done:
                # generate() can't be interrupted; the lock stays with the thread
                # until it returns, and /health reports it if it never does.
                logger.error("miso_generation_timeout", request_id=job_id, endpoint="synthesise", limit_s=round(limit_s, 1))
                return JSONResponse(status_code=504, content={"error": f"TTS generation timed out after {limit_s:.0f} s"})
            audio_data = fut.result()
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})
        except RuntimeError as exc:
            if _is_oom(exc):
                return _oom_response("generating", exc)
            if "cuda" in str(exc).lower():
                logger.error("miso_generation_cuda_error", error=str(exc), exc_info=True)
                return JSONResponse(status_code=503, content={"error": "GPU error", "detail": str(exc), "gpu": _gpu_memory_info()})
            logger.error("miso_generation_failed", error=str(exc), exc_info=True)
            return JSONResponse(status_code=500, content={"error": f"TTS generation failed: {exc}"})
        except Exception as exc:
            logger.error("miso_generation_failed", error=str(exc), exc_info=True)
            return JSONResponse(status_code=500, content={"error": f"TTS generation failed: {exc}"})

    import soundfile as sf

    duration = len(audio_data) / SAMPLE_RATE
    buf = io.BytesIO()
    await asyncio.to_thread(sf.write, buf, audio_data, SAMPLE_RATE, format="WAV")
    wav_bytes = buf.getvalue()

    s3_key = f"generated/voice/{job_id}.wav"
    audio_url = await asyncio.to_thread(_storage.put, s3_key, wav_bytes, "audio/wav")

    logger.info(
        "miso_synthesised", job_id=job_id, duration=round(duration, 2),
        total_s=round(time.monotonic() - t0, 2),
        rtf=round((time.monotonic() - t0) / duration, 2) if duration else None,
    )
    return {
        "audio_url": audio_url,
        "duration": duration,
        "duration_seconds": round(duration, 3),
        "format": "wav",
    }


# ── POST /synthesise/stream ───────────────────────────────────────────────────


@app.post("/synthesise/stream")
async def synthesise_stream(req: SynthesiseRequest):
    """Stream raw 24 kHz mono float32 PCM (pcm_f32le) with low first-byte latency.

    Instead of decoding the whole clip up front, Mimi frames are decoded
    incrementally (see ``Generator.generate_stream``) and emitted as soon as they
    are produced, so first-byte latency is ~``MISO_STREAM_FIRST_FRAMES`` × 80 ms
    of generation rather than the full clip. Byte layout (pcm_f32le, mono, 24 kHz)
    is identical to the XTTS sidecar, so the Wyoming bridge and browser live
    playback path are unchanged.

    The utterance is faded in and out. The last MISO_FADE_OUT_MS of each chunk
    is held back until the next chunk arrives or the stream ends, so the
    concatenated stream equals the faded /synthesise audio.

    ``context`` (optional) is a list of previous utterances: their text plus
    the base64 of the bytes this endpoint returned for them.
    """
    import threading

    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})
    if not (req.text or "").strip():
        return JSONResponse(status_code=422, content={"error": "text is required"})

    # In flight until run_locked() owns the GPU lock (see /synthesise).
    async with _mgr.hold():
        try:
            await _mgr.ensure_ready()
        except Exception as exc:
            return _load_failed_response(exc)

        # Resolve clone/speaker/sampling up front so a bad voice_clone_id or speaker
        # is a 422, not a 200 with an empty body.
        try:
            prepared = await asyncio.to_thread(_prepare_generation, req)
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})
        except Exception as exc:  # clone storage I/O, model unloaded mid-request
            logger.error("miso_stream_prepare_failed", error=str(exc), exc_info=True)
            return JSONResponse(status_code=500, content={"error": f"TTS preparation failed: {exc}"})

        # The GPU lock belongs to the producer thread: run_locked() releases it when
        # the producer returns. It used to be released in body_iter's finally, but a
        # client disconnect cancels body_iter inside an anyio cancel scope that
        # re-cancels every await, so that cleanup could be cut short and leak the
        # lock for good (#9). The producer always returns: it checks `stop`, a
        # deadline, and gives up on a client that stops reading.
        request_id = str(uuid.uuid4())
        if not await _mgr.acquire_gen(request_id, "stream", voice_clone_id=req.voice_clone_id):
            return _busy_response()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        stop = threading.Event()
        SENTINEL = object()
        limit_s = _generation_deadline_s(prepared[4])
        deadline = time.monotonic() + limit_s

        def _offer(item) -> bool:
            """Hand an item to the consumer. False → stop generating.

            Blocks while the queue is full (backpressure), but never for ever: it
            gives up once the stream is stopped or the client takes nothing for
            MISO_STREAM_STALL_S."""
            put = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            since = time.monotonic()
            while True:
                try:
                    put.result(timeout=0.25)
                    return True
                except concurrent.futures.TimeoutError:
                    if stop.is_set():
                        put.cancel()
                        return False
                    if time.monotonic() - since >= STREAM_STALL_S:
                        put.cancel()
                        logger.warning("miso_stream_stalled", request_id=request_id, stalled_s=STREAM_STALL_S)
                        return False

        def _producer():
            try:
                with contextlib.closing(_generate_stream_chunks(req, stop, prepared)) as chunks:
                    for chunk in chunks:
                        if stop.is_set():
                            return
                        if time.monotonic() > deadline:
                            logger.error(
                                "miso_generation_timeout", request_id=request_id,
                                endpoint="stream", limit_s=round(limit_s, 1),
                            )
                            _offer(SENTINEL)
                            return
                        if not _offer(chunk.astype("<f4", copy=False).tobytes()):
                            return
                _offer(SENTINEL)
            except Exception as exc:  # surfaced to the consumer, then logged
                _offer(exc)

        _mgr.run_locked(_producer)  # no await since acquire_gen: the lock can't leak

    async def body_iter():
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), STREAM_CHUNK_TIMEOUT_S)
                except asyncio.TimeoutError:
                    logger.error(
                        "miso_generation_timeout", request_id=request_id, endpoint="stream",
                        reason="no_chunk", waited_s=STREAM_CHUNK_TIMEOUT_S,
                    )
                    break
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    logger.error("miso_stream_failed", request_id=request_id, error=str(item), exc_info=item)
                    break
                yield item
        finally:
            # No awaits here: under a disconnect they would be cancelled again.
            stop.set()

    return StreamingResponse(
        body_iter(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Format": "pcm_f32le",
            "X-Channels": "1",
            "Cache-Control": "no-store",
        },
    )


# ── POST /analyze-clips ───────────────────────────────────────────────────────
# Miso is one-shot, so unlike XTTS we cannot score embedding *consistency*
# (no get_conditioning_latents). We return the same response schema the UI reads
# (clips[].score/rank/recommended, order, threshold) using quality-only metrics:
# clipping, usable speech seconds, and VAD-derived SNR. `consistency` is null.


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _quick_clip_metrics(path: str) -> dict:
    import torch
    import torchaudio
    import torchaudio.functional as AF

    m: dict = {"flags": []}
    wav, sr = torchaudio.load(path)
    mag = wav.abs()
    m["raw_seconds"] = round(wav.shape[-1] / sr, 2)
    m["peak"] = round(float(mag.max().item()) if mag.numel() else 0.0, 3)
    m["clip_ratio"] = round(float((mag > 0.99).float().mean().item()) if mag.numel() else 0.0, 4)

    mono = wav.mean(dim=0, keepdim=True) if wav.shape[0] > 1 else wav
    s16 = mono.squeeze(0) if sr == 16000 else AF.resample(mono, sr, 16000).squeeze(0)
    m["speech_seconds"] = m["raw_seconds"]
    m["snr_db"] = None
    m["noise_floor_dbfs"] = None
    try:
        from silero_vad import get_speech_timestamps

        ts = get_speech_timestamps(s16, _get_vad_model(), sampling_rate=16000, return_seconds=True)
        if not ts:
            m["flags"].append("vad_no_speech")
            m["speech_seconds"] = 0.0  # known, unlike the vad_failed fallback above
            return m
        total = s16.shape[-1]
        inset = int(0.05 * 16000)
        speech_segs, noise_segs = [], []
        prev_end = 0
        for t in ts:
            a, b = max(0, int(t["start"] * 16000)), min(total, int(t["end"] * 16000))
            if b > a:
                speech_segs.append(s16[a:b])
            g0, g1 = prev_end + inset, int(t["start"] * 16000) - inset
            if g1 - g0 > inset:
                noise_segs.append(s16[g0:g1])
            prev_end = int(t["end"] * 16000)
        tail = prev_end + inset
        if total - tail > inset:
            noise_segs.append(s16[tail:total])
        if speech_segs:
            speech = torch.cat(speech_segs)
            m["speech_seconds"] = round(speech.shape[-1] / 16000, 2)
            if noise_segs:
                noise = torch.cat(noise_segs)
                if noise.numel() > int(0.2 * 16000):
                    nrms = float(torch.sqrt(torch.mean(noise ** 2)).item())
                    srms = float(torch.sqrt(torch.mean(speech ** 2)).item())
                    m["noise_floor_dbfs"] = round(20.0 * math.log10(nrms + 1e-9), 1)
                    if srms > 0:
                        m["snr_db"] = round(20.0 * math.log10((srms + 1e-9) / (nrms + 1e-9)), 1)
    except Exception as exc:
        m["flags"].append(f"vad_failed:{exc}")
    return m


def _analyze_clips(raw_paths: list[str], names: list[str]) -> dict:
    items: list[dict] = []
    for path, name in zip(raw_paths, names):
        item: dict = {"name": name, "flags": [], "consistency": None}
        try:
            item.update(_quick_clip_metrics(path))
        except Exception as exc:
            item["flags"].append(f"unreadable:{exc}")
            item["score"] = 0
            item["recommended"] = False
            items.append(item)
            continue

        snr = item.get("snr_db")
        q_snr = _clamp01((snr - 10.0) / 25.0) if snr is not None else 0.7
        speech = item.get("speech_seconds") or 0.0
        q_dur = _clamp01(speech / 6.0)
        q_clip = 1.0 - _clamp01((item.get("clip_ratio") or 0.0) / 0.02)
        quality = 0.4 * q_snr + 0.35 * q_dur + 0.25 * q_clip
        item["score"] = round(quality * 100)
        item["_anchor"] = quality + 0.25 * _clamp01(speech / REF_MAX_SECONDS)

        if (item.get("clip_ratio") or 0.0) > 0.01:
            item["flags"].append("clipping")
        if snr is not None and snr < 15:
            item["flags"].append("noisy")
        if speech and speech < 2.0:
            item["flags"].append("very_short")
        item["recommended"] = item["score"] >= ANALYZE_SCORE_THRESHOLD
        item["flags"] = sorted(set(item["flags"]))
        items.append(item)

    order = sorted(range(len(items)), key=lambda i: items[i].get("_anchor", 0.0), reverse=True)
    for rank, idx in enumerate(order):
        items[idx]["rank"] = rank
    for item in items:
        item.pop("_anchor", None)

    return {
        "clips": items,
        "order": order,
        "threshold": ANALYZE_SCORE_THRESHOLD,
        "gpt_cond_len": REF_MAX_SECONDS,  # XTTS parity key; Miso ref cap (s)
        "engine": "miso",
    }


def _upload_refused(exc: ValueError) -> JSONResponse:
    """An upload over its cap is 413; a malformed one is 400."""
    return JSONResponse(status_code=413 if isinstance(exc, TooLarge) else 400, content={"error": str(exc)})


@app.post("/analyze-clips")
async def analyze_clips(request: Request):
    """Score candidate clips and recommend a subset + ordering (no clone made).

    Reads the ``audio`` multipart parts directly (robust to the lone-file-list
    body-parsing quirk in some Starlette versions)."""
    form = await request.form()
    clips = [u for u in form.getlist("audio") if hasattr(u, "read")]
    if not clips:
        return JSONResponse(status_code=422, content={"error": "No clips provided"})
    if len(clips) > MAX_CLIPS:
        return JSONResponse(status_code=422, content={"error": f"At most {MAX_CLIPS} clips (MISO_MAX_CLIPS)"})

    raw_paths: list[str] = []
    names: list[str] = []
    try:
        for clip in clips:
            try:
                data = await read_upload(clip, MAX_CLIP_BYTES, what="audio clip")
            except (TooLarge, BadInput) as exc:
                return _upload_refused(exc)
            suffix = os.path.splitext(clip.filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(data)
                raw_paths.append(f.name)
            names.append(clip.filename or f"clip-{len(names) + 1}")
        result = await asyncio.to_thread(_analyze_clips, raw_paths, names)
    finally:
        for path in raw_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
    return result


# ── POST /clone-voice ─────────────────────────────────────────────────────────


@app.post("/clone-voice")
async def clone_voice(
    audio: list[UploadFile] = File(...),
    reference_text: str = Form(""),
    companion_id: str = Form(...),
    voice_name: str = Form(""),
):
    """Create a one-shot Miso clone from one or more reference clips.

    Clips are standardised (mono → 24 kHz → high-pass → VAD trim → loudness),
    concatenated (capped at MISO_REF_MAX_SECONDS) and stored with their
    transcript + speaker as `voice_clones/{companion_id}/prompt.pt`, plus a
    `meta.json` sidecar — the same layout the XTTS sidecar uses, so /clones,
    download and import are unchanged. `reference_text` is the transcript of the
    reference audio; if empty and MISO_WHISPER_URL is set we transcribe it.
    """
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})

    companion_id = (companion_id or "").strip()
    if not _is_valid_companion_id(companion_id):
        return JSONResponse(
            status_code=422,
            content={"error": "companion_id must match ^[A-Za-z0-9_-]{1,64}$"},
        )
    clips = audio if isinstance(audio, list) else [audio]
    if not clips:
        return JSONResponse(status_code=422, content={"error": "At least one reference clip is required"})
    if len(clips) > MAX_CLIPS:
        return JSONResponse(status_code=422, content={"error": f"At most {MAX_CLIPS} clips (MISO_MAX_CLIPS)"})

    import torch

    raw_paths: list[str] = []
    reports: list[dict] = []
    try:
        segments = []
        for clip in clips:
            try:
                clip_bytes = await read_upload(clip, MAX_CLIP_BYTES, what="audio clip")
            except (TooLarge, BadInput) as exc:
                return _upload_refused(exc)
            suffix = os.path.splitext(clip.filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(clip_bytes)
                raw_paths.append(f.name)
            try:
                samples, report = await asyncio.to_thread(_preprocess_reference, raw_paths[-1])
                report["name"] = clip.filename
                segments.append(samples)
                reports.append(report)
            except Exception as exc:
                logger.warning("clip_preprocess_failed", name=clip.filename, error=str(exc))
                reports.append({"name": clip.filename, "warnings": [f"preprocess_failed:{exc}"], "applied": []})

        if not segments:
            return JSONResponse(status_code=422, content={"error": "No usable reference audio after preprocessing"})

        ref_audio = torch.cat(segments)
        max_samps = int(REF_MAX_SECONDS * SAMPLE_RATE)
        if ref_audio.shape[-1] > max_samps:
            ref_audio = ref_audio[:max_samps]
        total_seconds = round(ref_audio.shape[-1] / SAMPLE_RATE, 2)
        if total_seconds < 2.0:
            return JSONResponse(
                status_code=422,
                content={"error": "Usable speech under 2 seconds after preprocessing (6+ recommended)"},
            )
    finally:
        for path in raw_paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    ref_text = (reference_text or "").strip()
    if not ref_text and WHISPER_URL:
        # Transcribe what is stored (the clips that survived preprocessing, trimmed
        # and capped), not an upload: Miso conditions on the (audio, text) pair.
        wav_bytes = await asyncio.to_thread(_ref_wav_bytes, ref_audio)
        ref_text = await asyncio.to_thread(_transcribe_reference, wav_bytes)

    speaker = DEFAULT_SPEAKER
    prompt = {
        "ref_audio": ref_audio.detach().cpu().to(torch.float32),
        "ref_text": ref_text,
        "speaker": speaker,
        "sample_rate": SAMPLE_RATE,
        "engine": "miso",
    }
    buf = io.BytesIO()
    torch.save(prompt, buf)
    s3_key = f"voice_clones/{companion_id}/prompt.pt"
    # Re-cloning overwrites meta.json; keep any pinned temperature/top-k.
    previous_meta = await asyncio.to_thread(_load_clone_meta, s3_key)
    await asyncio.to_thread(_storage.put, s3_key, buf.getvalue(), "application/octet-stream")

    meta = {
        "voice_clone_id": s3_key,
        "companion_id": companion_id,
        "name": voice_name or companion_id,
        "num_clips": len(segments),  # the clips used; one that failed preprocessing is not in the clone
        "total_seconds": total_seconds,
        "has_transcript": bool(ref_text),
        "engine": "miso",
        "created_at": time.time(),
        **_sampling_meta_fields(previous_meta),
    }
    await asyncio.to_thread(
        _storage.put, f"voice_clones/{companion_id}/meta.json",
        json.dumps(meta).encode("utf-8"), "application/json",
    )

    _clone_cache[s3_key] = prompt
    if len(_clone_cache) > _CLONE_CACHE_MAX:
        _clone_cache.popitem(last=False)

    logger.info("voice_cloned", companion_id=companion_id, clips=len(segments), uploaded=len(clips), duration=total_seconds, s3_key=s3_key)
    return {**meta, "duration": total_seconds, "clips": reports}


# ── GET /clones, download, import ─────────────────────────────────────────────


def _list_clones() -> list[dict]:
    clones: list[dict] = []
    for key in _storage.list_keys("voice_clones/"):
        if not key.endswith("/meta.json"):
            continue
        try:
            meta = json.loads(_storage.get(key).decode("utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("meta.json is not an object")
            clones.append(meta)
        except Exception as exc:
            logger.warning("clone_meta_unreadable", key=key, error=str(exc))

    def _created(c: dict) -> float:
        value = c.get("created_at")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0

    clones.sort(key=_created, reverse=True)
    return clones


@app.get("/clones")
async def list_clones():
    try:
        clones = await asyncio.to_thread(_list_clones)
    except Exception as exc:
        logger.error("clone_list_error", error=str(exc))
        return JSONResponse(status_code=500, content={"error": f"Failed to list clones: {exc}"})
    return {"clones": clones}


def _resolve_clone_companion_id(companion_id: str) -> Optional[str]:
    """Map a delete id to a stored companion_id (exact key or slugified name)."""
    raw = (companion_id or "").strip()
    if not raw:
        return None
    def _stored(cid: str) -> bool:
        return _is_valid_companion_id(cid) and _storage.exists(f"voice_clones/{cid}/prompt.pt")

    if raw.startswith("voice_clones/"):
        # Accept only `voice_clones/{id}` or `voice_clones/{id}/prompt.pt` for an
        # existing clone; never '', '.', '..' or nested paths.
        parts = raw.split("/")
        if len(parts) == 2 or (len(parts) == 3 and parts[2] == "prompt.pt"):
            if _stored(parts[1]):
                return parts[1]
        return None

    if _stored(raw):
        return raw

    # Strict slug (no `voice-<ts>` fallback): an input with no slug characters,
    # like '..', must not match every clone whose name also slugifies to ''.
    def _strict_slug(value: str) -> str:
        return _SLUG_RE.sub("-", (value or "").strip().lower()).strip("-")

    slug = _strict_slug(raw)
    if slug and slug != raw and _stored(slug):
        return slug

    needle = raw.lower()
    for clone in _list_clones():
        cid = str(clone.get("companion_id", ""))
        name = str(clone.get("name", ""))
        if not _is_valid_companion_id(cid):
            continue
        if raw == cid or raw == name or needle == cid.lower() or needle == name.lower():
            return cid
        if slug and (slug == cid or slug == _strict_slug(name)):
            return cid
    return None


def _delete_clone(companion_id: str) -> bool:
    cid = _resolve_clone_companion_id(companion_id)
    if not cid or not _is_valid_companion_id(cid):
        return False

    prefix = f"voice_clones/{cid}/"
    keys = [k for k in _storage.list_keys(prefix) if k.startswith(prefix)]
    if not keys:
        return False

    removed = 0
    for key in keys:
        if _storage.delete(key):
            removed += 1
        _clone_cache.pop(key, None)

    _clone_cache.pop(_clone_prompt_key(cid), None)
    logger.info("voice_deleted", companion_id=cid, keys=keys)
    return removed > 0


@app.delete("/clones/{companion_id}")
async def delete_clone(companion_id: str):
    try:
        removed = await asyncio.to_thread(_delete_clone, companion_id)
    except Exception as exc:
        logger.error("clone_delete_error", companion_id=companion_id, error=str(exc))
        return JSONResponse(status_code=500, content={"error": f"Failed to delete clone: {exc}"})
    if not removed:
        return JSONResponse(status_code=404, content={"error": f"Voice clone not found: {companion_id}"})
    return {"status": "deleted", "companion_id": companion_id}


# Clone archive format (download and import). The prompt travels as
# prompt.safetensors: every tensor of the stored prompt (ref_audio) as a tensor,
# and its scalar fields (speaker, sample_rate, ref_text, engine) as strings in the
# safetensors `__metadata__` header. A safetensors file is a JSON header plus raw
# tensor bytes, so parsing one never runs code. A legacy prompt.pt (a torch pickle)
# in an uploaded archive is refused: torch.load(weights_only=True) on untrusted
# bytes is CVE-2025-32434 on torch 2.4, and later torch releases keep finding
# unpickler bypasses. Stored clones stay prompt.pt: this server wrote them.
_ARCHIVE_PROMPT = "prompt.safetensors"
_LEGACY_ARCHIVE_PROMPT = "prompt.pt"
_ARCHIVE_FORMAT = "miso-clone"
_ARCHIVE_FORMAT_VERSION = "1"
_ARCHIVE_INT_RE = re.compile(r"-?[0-9]{1,9}")


def _prompt_to_safetensors(prompt: dict) -> bytes:
    """Serialise a validated clone prompt as the archive's prompt.safetensors."""
    import torch
    from safetensors.torch import save

    tensors = {k: v.detach().cpu().contiguous() for k, v in prompt.items() if torch.is_tensor(v)}
    metadata = {
        "format": _ARCHIVE_FORMAT,
        "format_version": _ARCHIVE_FORMAT_VERSION,
        "engine": "miso",
        "speaker": str(int(prompt.get("speaker", DEFAULT_SPEAKER))),
        "sample_rate": str(int(prompt.get("sample_rate", SAMPLE_RATE))),
        "ref_text": str(prompt.get("ref_text", "") or ""),
    }
    return save(tensors, metadata=metadata)


def _safetensors_metadata(data: bytes) -> dict:
    """Read the `__metadata__` string map from a safetensors header (no tensors)."""
    if len(data) < 8:
        raise ValueError("prompt.safetensors is truncated")
    header_len = int.from_bytes(data[:8], "little")
    if header_len > len(data) - 8:
        raise ValueError("prompt.safetensors has a malformed header")
    try:
        header = json.loads(data[8 : 8 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"prompt.safetensors has a malformed header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("prompt.safetensors has a malformed header")
    metadata = header.get("__metadata__") or {}
    if not (isinstance(metadata, dict) and all(isinstance(v, str) for v in metadata.values())):
        raise ValueError("prompt.safetensors metadata must map names to strings")
    return metadata


def _archive_int(metadata: dict, field: str, default: int) -> int:
    value = metadata.get(field)
    if value is None:
        return default
    if not _ARCHIVE_INT_RE.fullmatch(value.strip()):
        raise ValueError(f"prompt.safetensors has a malformed {field}: {value[:32]!r}")
    return int(value)


def _prompt_from_safetensors(data: bytes, fallback_ref_text="") -> dict:
    """Parse and validate an uploaded prompt.safetensors into a clone prompt.

    Only safetensors' own parser touches the bytes: no pickle, no torch.load."""
    import torch
    from safetensors.torch import load

    metadata = _safetensors_metadata(data)
    try:
        tensors = load(data)
    except Exception as exc:
        raise ValueError(f"Invalid prompt.safetensors: {exc}") from exc
    ref_audio = tensors.get("ref_audio")
    if ref_audio is None:
        raise ValueError("prompt.safetensors must contain a Miso clone (ref_audio)")
    if ref_audio.ndim < 1 or ref_audio.numel() == 0:
        raise ValueError("ref_audio is empty")
    if ref_audio.is_complex():
        raise ValueError("ref_audio must be real-valued")
    speaker = _archive_int(metadata, "speaker", DEFAULT_SPEAKER)
    sample_rate = _archive_int(metadata, "sample_rate", SAMPLE_RATE)
    if sample_rate <= 0:
        raise ValueError("prompt.safetensors has a malformed sample_rate")
    ref_text = metadata.get("ref_text")
    if ref_text is None:
        ref_text = fallback_ref_text if isinstance(fallback_ref_text, str) else ""
    return {
        "ref_audio": ref_audio.detach().cpu().to(torch.float32),
        "ref_text": ref_text,
        "speaker": speaker,
        "sample_rate": sample_rate,
        "engine": "miso",
    }


def _build_clone_zip(companion_id: str) -> bytes:
    """Bundle a stored clone as prompt.safetensors (+ meta.json).

    The stored prompt.pt was written by this server and loads through
    _load_clone; legacy and new clones alike leave as safetensors."""
    if not _is_valid_companion_id(companion_id):
        raise ValueError(f"Invalid voice clone id: {companion_id!r}")
    prompt_key = f"voice_clones/{companion_id}/prompt.pt"
    meta_key = f"voice_clones/{companion_id}/meta.json"
    if not _storage.exists(prompt_key):
        raise FileNotFoundError(f"Voice clone not found: {companion_id}")
    prompt_bytes = _prompt_to_safetensors(_load_clone(companion_id))
    meta_bytes = _storage.get(meta_key) if _storage.exists(meta_key) else None

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_ARCHIVE_PROMPT, prompt_bytes)
        if meta_bytes is not None:
            zf.writestr("meta.json", meta_bytes)
    return zip_buf.getvalue()


@app.get("/clones/{companion_id}/download")
async def download_clone(companion_id: str):
    try:
        data = await asyncio.to_thread(_build_clone_zip, companion_id)
    except FileNotFoundError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    except Exception as exc:
        logger.error("clone_download_error", companion_id=companion_id, error=str(exc))
        return JSONResponse(status_code=500, content={"error": f"Failed to bundle clone: {exc}"})
    return Response(
        content=data, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{companion_id}.zip"'},
    )


_MAX_PROMPT_BYTES = 64 * 1024 * 1024
_MAX_META_BYTES = 1024 * 1024


class CloneExistsError(Exception):
    """The import's target companion_id already has a clone (and replace was not set)."""


def _import_clone(zip_bytes: bytes, companion_override: str, name_override: str, replace: bool = False) -> dict:
    """Validate and store a clone archive (prompt.safetensors [+ meta.json]).

    The target id comes from the override, else the archive's own metadata, so
    an existing clone under it is never replaced unless ``replace`` is true
    (``CloneExistsError`` otherwise), as on the XTTS engine. An archive holding
    only a legacy prompt.pt is refused unread: uploaded bytes never reach
    torch.load or any other unpickler."""
    import torch

    try:
        z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a valid zip archive: {exc}") from exc

    names = z.namelist()

    def _member(basename: str):
        return next((n for n in names if n.rsplit("/", 1)[-1] == basename), None)

    prompt_name = _member(_ARCHIVE_PROMPT)
    meta_name = _member("meta.json")
    # zipfile stops reading a member at its declared size, so checking that
    # size bounds what z.read() inflates.
    if prompt_name and z.getinfo(prompt_name).file_size > _MAX_PROMPT_BYTES:
        raise ValueError(f"{_ARCHIVE_PROMPT} is unexpectedly large")
    if meta_name and z.getinfo(meta_name).file_size > _MAX_META_BYTES:
        raise ValueError("meta.json is unexpectedly large")
    if not prompt_name:
        if _member(_LEGACY_ARCHIVE_PROMPT):
            raise ValueError(
                "This archive holds a legacy prompt.pt (a pickle), which is no longer imported because "
                "unpickling an uploaded file can run code. Re-export the clone from a Miso engine running "
                "this version: its download now produces prompt.safetensors."
            )
        raise ValueError(f"Archive does not contain {_ARCHIVE_PROMPT}")

    src_meta: dict = {}
    if meta_name:
        try:
            src_meta = json.loads(z.read(meta_name).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            src_meta = {}
        if not isinstance(src_meta, dict):
            src_meta = {}  # valid JSON, but not an object: as good as no sidecar

    prompt = _prompt_from_safetensors(z.read(prompt_name), src_meta.get("ref_text", ""))
    sample_rate = prompt["sample_rate"]

    def _meta_str(field: str) -> str:
        value = src_meta.get(field)
        return value.strip() if isinstance(value, str) else ""

    companion_id = (companion_override or "").strip() or _meta_str("companion_id")
    companion_id = _slugify(companion_id or name_override or _meta_str("name") or "imported-voice")
    companion_id = companion_id[:64].strip("-") or f"voice-{int(time.time())}"
    if not _is_valid_companion_id(companion_id):
        raise ValueError(f"Invalid companion_id: {companion_id!r}")
    if not replace and _storage.exists(f"voice_clones/{companion_id}/prompt.pt"):
        raise CloneExistsError(f"Voice clone {companion_id!r} already exists; send replace=true to overwrite")

    # Stored clones stay prompt.pt, re-serialised here from the validated tensors.
    buf = io.BytesIO()
    torch.save(prompt, buf)
    s3_key = f"voice_clones/{companion_id}/prompt.pt"
    _storage.put(s3_key, buf.getvalue(), "application/octet-stream")

    meta = {
        "voice_clone_id": s3_key,
        "companion_id": companion_id,
        "name": name_override or _meta_str("name") or companion_id,
        "num_clips": src_meta.get("num_clips", 0),
        # The clone's own rate: an imported prompt need not be at this server's SAMPLE_RATE.
        "total_seconds": src_meta.get("total_seconds", round(prompt["ref_audio"].shape[-1] / sample_rate, 2)),
        "imported": True,
        "engine": "miso",
        "created_at": time.time(),
        **_sampling_meta_fields(src_meta),
    }
    _storage.put(f"voice_clones/{companion_id}/meta.json", json.dumps(meta).encode("utf-8"), "application/json")

    _clone_cache[s3_key] = prompt
    if len(_clone_cache) > _CLONE_CACHE_MAX:
        _clone_cache.popitem(last=False)
    logger.info("voice_imported", companion_id=companion_id, s3_key=s3_key)
    return {**meta, "duration": meta["total_seconds"]}


@app.post("/clones/import")
async def import_clone(
    archive: UploadFile = File(...),
    voice_name: str = Form(""),
    companion_id: str = Form(""),
    replace: bool = Form(False),
):
    """Import a downloaded voice archive. 409 when the target id already has a
    clone, unless ``replace`` is true."""
    try:
        data = await read_upload(archive, MAX_ARCHIVE_BYTES, what="clone archive")
    except (TooLarge, BadInput) as exc:
        return _upload_refused(exc)
    try:
        meta = await asyncio.to_thread(_import_clone, data, companion_id, voice_name, replace)
    except CloneExistsError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    except Exception as exc:
        logger.error("clone_import_error", error=str(exc))
        return JSONResponse(status_code=500, content={"error": f"Failed to import voice: {exc}"})
    return meta


# ── GET /speakers ─────────────────────────────────────────────────────────────


@app.get("/speakers")
async def list_speakers():
    """Miso is clone-first and has no named studio speakers; it uses integer
    speaker ids for multi-speaker dialogue. We advertise a couple so the UI's
    dropdown has sane built-ins; custom voices come from /clones."""
    n = int(os.getenv("MISO_NUM_SPEAKERS", "2"))
    speakers = [
        {"id": str(i), "name": f"Speaker {i}", "language": "en", "languages": list(SUPPORTED_LANGS)}
        for i in range(max(0, n))
    ]
    return {"speakers": speakers}


# ── GET /health ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"status": "loading"})
    busy = _mgr.busy_info()
    body = {
        "status": "ok",
        # A bool, as argus-cortex's ResidentHealth and argus-iris expect;
        # who holds the GPU lock is in `active` (smk762/voluptas#26).
        "busy": busy is not None,
        "active": busy,
        "engine": "miso",
        "capabilities": ["tts"],  # the echo contracts this engine serves (docs/api.md)
        "model": MODEL_SOURCE,
        "device": DEVICE,
        "model_state": _mgr.state.value,
        "sample_rate": SAMPLE_RATE,
        "storage": STORAGE_BACKEND,
        "keep_alive_gpu": KEEP_ALIVE_GPU,
        "keep_alive_cpu": KEEP_ALIVE_CPU,
        "idle_s": round(_mgr._idle_s(), 1) if _mgr.state in (TTSState.GPU, TTSState.CPU) else None,
        "error": _mgr.error,
        "rss_mb": _rss_mb(),
        "gpu": _gpu_memory_info(),
    }
    if _is_stuck(busy):
        # Held past any possible generation: fail the container healthcheck.
        body["status"] = "stuck"
        body["stuck_after_s"] = STUCK_AFTER_S
        return JSONResponse(status_code=503, content=body)
    return body


# ── GET /media/{key} ──────────────────────────────────────────────────────────
# Serves generated audio / clone artifacts for the local storage backend so
# `audio_url` is directly fetchable. (With the S3 backend, URLs point at MinIO
# and this route is unused.)

_MEDIA_CTYPES = {".wav": "audio/wav", ".json": "application/json", ".pt": "application/octet-stream", ".zip": "application/zip"}


@app.get("/media/{key:path}")
async def serve_media(key: str):
    if not isinstance(_storage, LocalStorage):
        return JSONResponse(status_code=404, content={"error": "media route only serves the local backend"})
    try:
        found = _storage.exists(key)
    except ValueError:
        found = False  # a key that would leave the storage root
    if not found:
        return JSONResponse(status_code=404, content={"error": "not found"})
    data = await asyncio.to_thread(_storage.get, key)
    ext = os.path.splitext(key)[1].lower()
    return Response(content=data, media_type=_MEDIA_CTYPES.get(ext, "application/octet-stream"))


# ── Admin ─────────────────────────────────────────────────────────────────────

# /admin/* needs `Authorization: Bearer <ARGUS_ADMIN_TOKEN>` (argus-cortex's bearer). With the
# token unset the routes are closed (503), unless ARGUS_ADMIN_INSECURE=1 opens them for local
# development. The same rule holds on every argus-echo engine. Read per request, so rotating
# the token needs no restart.
def _admin_insecure() -> bool:
    return os.environ.get("ARGUS_ADMIN_INSECURE", "").strip().lower() in ("1", "true", "yes")


_admin_closed = bearer("ARGUS_ADMIN_TOKEN", open_when_unset=False)
_admin_open = bearer("ARGUS_ADMIN_TOKEN", open_when_unset=True)


async def _require_admin(conn: HTTPConnection) -> None:
    await (_admin_open if _admin_insecure() else _admin_closed)(conn)


@app.post("/admin/evict", dependencies=[Depends(_require_admin)])
async def admin_evict():
    await _mgr.demote_to_cpu()
    return {"status": "evicted", "model_state": _mgr.state.value}


@app.post("/admin/unload", dependencies=[Depends(_require_admin)])
async def admin_unload():
    await _mgr.unload()
    return {"status": "unloaded", "model_state": _mgr.state.value}


@app.post("/admin/wake", dependencies=[Depends(_require_admin)])
async def admin_wake():
    try:
        await _mgr.ensure_ready()
    except Exception as exc:
        return _load_failed_response(exc)
    return {"status": "ready", "model_state": _mgr.state.value}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("miso_server:app", host="0.0.0.0", port=PORT, log_level="info")

