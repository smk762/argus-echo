"""
argus-echo's XTTS-v2 engine — FastAPI, port 8031.

Coqui XTTS-v2 (via the maintained `coqui-tts` fork) for zero-shot voice cloning
with streaming output. Speaks the same HTTP contract as the Miso and Qwen3-TTS
engines, so a client can switch engines by changing its URL:

  POST /synthesise         — text → audio, uploaded to S3, returns {audio_url,...}
  POST /synthesise/stream  — text → chunked 24 kHz PCM (real within-sentence
                             streaming via XTTS inference_stream)
  POST /analyze-clips      — score candidate clips, recommend subset + ordering
  POST /clone-voice        — one or more reference clips → conditioning latents in S3
  GET  /clones             — list stored voice clones (from meta.json sidecars)
  GET  /clones/{id}/download — download a clone's prompt.safetensors + meta.json as a zip
  DELETE /clones/{id}      — permanently delete a stored clone from S3
  POST /clones/import      — import a downloaded voice archive (zip) back into S3
  GET  /speakers           — built-in XTTS studio speakers
  GET  /health             — status + GPU memory
  POST /admin/{evict,unload,wake}
  POST /admin/load         — load a fine-tuned checkpoint from XTTS_CHECKPOINT_ROOT

GPU memory lifecycle (three-state, like the Qwen sidecar / ModernBERT):
  unloaded → cpu → gpu

  On request the model is promoted to GPU; after XTTS_KEEP_ALIVE_GPU idle
  seconds it demotes to CPU, after a further XTTS_KEEP_ALIVE_CPU it fully
  unloads. XTTS is a single nn.Module so device moves are a plain .to().
  Work that touches the model holds a lease (XTTSModelManager.lease) for its
  whole duration, including a stream's full lifetime; demotion/unload are
  deferred while any lease is held and the idle timer restarts on release.
  A negative keep-alive pins that stage (GPU < 0 never demotes or unloads).

A "voice clone" here is the pair (gpt_cond_latent, speaker_embedding) computed
from the reference clip by get_conditioning_latents(); we torch.save it to S3
under the same `voice_clones/{companion_id}/prompt.pt` key the Qwen path used.

Env vars:
  XTTS_MODEL           — Coqui model id (default tts_models/multilingual/multi-dataset/xtts_v2)
  XTTS_DEVICE          — "cuda:0" (default) or "cpu"
  XTTS_PRELOAD         — "1" to load to GPU at startup, "0" (default) lazy
  XTTS_USE_DEEPSPEED   — "1" to enable DeepSpeed inference (default "0")
  XTTS_KEEP_ALIVE_GPU  — idle seconds on GPU before demoting to CPU (default 120)
  XTTS_KEEP_ALIVE_CPU  — idle seconds on CPU before full unload (default 600)
  XTTS_TEMPERATURE     — sampling temperature (default 0.7)
  XTTS_STREAM_CHUNK_SIZE — GPT tokens per streamed audio chunk (default 20)
  XTTS_CPU_FALLBACK    — "1" (default): when the model can't be moved onto the GPU
                         (CUDA OOM), synthesise on the CPU copy instead; "0": 503
  XTTS_OOM_RETRY_AFTER_S — Retry-After on that 503 (default 30). A promotion that ran out of memory is not
                         tried again for this long: requests run on the CPU (or get the 503) without
                         moving the weights, and only /admin/wake retries sooner
  XTTS_RNG_WAIT_S      — seconds a generation waits for the sampler while a seeded one holds it before it is
                         answered 503 (default 60; 0 waits for ever). A seeded stream whose client takes
                         nothing for this long is ended, so it can't hold the sampler longer
  XTTS_GENERATION_THREADS — threads that run generations (default 8). Generations wait for the sampler
                         there, never in the pool every other to_thread call shares
  XTTS_CHECKPOINT_ROOT — directory of fine-tuned checkpoints POST /admin/load may load
                         (unset: the route answers 503). See docs/xtts.md.
  ECHO_VERSION / ECHO_REVISION — the image's release and git revision (build args),
                         reported in /health as ``image``

  A synth request may send ``device: "cpu"`` (a client does when argus-iris refuses
  its GPU lease): the model is used where it is (loaded to CPU if it isn't) and is
  not promoted. While CPU work holds a lease the model is not moved, so other
  requests join it on the CPU. A GPU promotion that fails moves the weights back to
  the CPU and frees the cache, so the model stays usable (state ``cpu``).
  S3_ENDPOINT_URL / S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY /
  S3_BUCKET_NAME / S3_PUBLIC_URL
"""

import asyncio
import contextvars
import functools
import gc
import io
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import random
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import quote, urlsplit

import structlog
from argus_cortex.auth import bearer
from argus_cortex.fs import PathRefused, safe_join
from starlette.requests import HTTPConnection, Request
from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

XTTS_MODEL = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
DEVICE = os.getenv("XTTS_DEVICE", "cuda:0")
PRELOAD = os.getenv("XTTS_PRELOAD", "0") == "1"
USE_DEEPSPEED = os.getenv("XTTS_USE_DEEPSPEED", "0") == "1"
KEEP_ALIVE_GPU: float = float(os.getenv("XTTS_KEEP_ALIVE_GPU", "120"))
KEEP_ALIVE_CPU: float = float(os.getenv("XTTS_KEEP_ALIVE_CPU", "600"))
TEMPERATURE: float = float(os.getenv("XTTS_TEMPERATURE", "0.7"))
STREAM_CHUNK_SIZE: int = int(os.getenv("XTTS_STREAM_CHUNK_SIZE", "20"))
CPU_FALLBACK: bool = os.getenv("XTTS_CPU_FALLBACK", "1").strip().lower() not in ("0", "false", "no", "off")
OOM_RETRY_AFTER_S: int = int(os.getenv("XTTS_OOM_RETRY_AFTER_S", "30"))
RNG_WAIT_S: float = float(os.getenv("XTTS_RNG_WAIT_S", "60"))
GENERATION_THREADS: int = max(1, int(os.getenv("XTTS_GENERATION_THREADS", "8")))
# Fine-tuned checkpoints POST /admin/load may load, one directory each (see _resolve_checkpoint).
CHECKPOINT_ROOT: str = os.getenv("XTTS_CHECKPOINT_ROOT", "").strip()
# The image's identity, baked in at build time (release.yml); None for a local build.
ECHO_VERSION: Optional[str] = os.getenv("ECHO_VERSION", "").strip() or None
ECHO_REVISION: Optional[str] = os.getenv("ECHO_REVISION", "").strip() or None

# Reference-clip preprocessing (applied at clone time, before latents).
PREPROCESS: bool = os.getenv("XTTS_PREPROCESS", "1") == "1"
PREPROCESS_VAD: bool = os.getenv("XTTS_PREPROCESS_VAD", "1") == "1"
PREPROCESS_SR: int = int(os.getenv("XTTS_PREPROCESS_SR", "24000"))
PREPROCESS_TARGET_LUFS: float = float(os.getenv("XTTS_PREPROCESS_LUFS", "-23.0"))
PREPROCESS_HIGHPASS_HZ: float = float(os.getenv("XTTS_PREPROCESS_HIGHPASS_HZ", "85"))
# Cascade N/2 biquads → ~12 dB/oct per pass; 4 ⇒ 24 dB/oct to knock down HVAC rumble.
PREPROCESS_HIGHPASS_ORDER: int = int(os.getenv("XTTS_PREPROCESS_HIGHPASS_ORDER", "4"))
PREPROCESS_MAX_SECONDS: float = float(os.getenv("XTTS_PREPROCESS_MAX_SECONDS", "30"))
PREPROCESS_VAD_PAD: float = float(os.getenv("XTTS_PREPROCESS_VAD_PAD", "0.1"))
# Stationary denoise (e.g. HVAC hum) seeded from VAD gaps; gated on estimated SNR.
PREPROCESS_DENOISE: bool = os.getenv("XTTS_PREPROCESS_DENOISE", "1") == "1"
PREPROCESS_DENOISE_PROP: float = float(os.getenv("XTTS_PREPROCESS_DENOISE_PROP", "0.75"))
PREPROCESS_DENOISE_SNR_DB: float = float(os.getenv("XTTS_PREPROCESS_DENOISE_SNR_DB", "30"))

# Conditioning-latent extraction. The speaker embedding is averaged over ALL
# clips, but the GPT conditioning latent only draws on the first `gpt_cond_len`
# seconds of the concatenation — XTTS's own default is 6, which under-uses a
# multi-clip reference set, so we default to 30 (matching XTTS high-level
# inference) and order clips best-first so that window is the strongest audio.
GPT_COND_LEN: int = int(os.getenv("XTTS_GPT_COND_LEN", "30"))
GPT_COND_CHUNK_LEN: int = int(os.getenv("XTTS_GPT_COND_CHUNK_LEN", "4"))
MAX_REF_LEN: int = int(os.getenv("XTTS_MAX_REF_LEN", "30"))
# Score (0–100) below which a clip is excluded from the default selection.
ANALYZE_SCORE_THRESHOLD: float = float(os.getenv("XTTS_ANALYZE_THRESHOLD", "60"))
# Upload bounds for /clone-voice and /analyze-clips (413 past either).
MAX_CLIPS: int = int(os.getenv("XTTS_MAX_CLIPS", "16"))
CLIP_MAX_BYTES: int = int(os.getenv("XTTS_CLIP_MAX_BYTES", str(16 * 1024 * 1024)))

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
# No default credentials: a deployment that forgot them must not run on MinIO's
# well-known root account. See _require_s3_credentials().
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "").strip()
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "voice")
S3_PUBLIC_URL = os.getenv("S3_PUBLIC_URL", "http://localhost:9000/voice")

SAMPLE_RATE = 24000  # XTTS-v2 output sample rate

# XTTS-v2 supported languages → our public ISO codes accepted on the wire.
# Qwen used "zh"; XTTS expects "zh-cn", so we normalise.
ISO_TO_XTTS: dict[str, str] = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it",
    "pt": "pt", "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl",
    "cs": "cs", "ar": "ar", "ja": "ja", "hu": "hu", "ko": "ko",
    "hi": "hi", "zh": "zh-cn", "zh-cn": "zh-cn",
    # tolerate full names from older callers
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "russian": "ru", "japanese": "ja",
    "korean": "ko", "chinese": "zh-cn",
}

_CLONE_CACHE_MAX = 50
_STREAM_QUEUE_MAX = 64      # buffered PCM chunks per stream (backpressure bound)
_STREAM_PUT_POLL_S = 0.5    # producer/consumer poll slice for disconnect checks
_STREAM_STALL_TIMEOUT_S = 300.0  # give up on a consumer that takes nothing this long

# ── GPU memory lifecycle ─────────────────────────────────────────────────────


class TTSState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    CPU = "cpu"
    GPU = "gpu"
    ERROR = "error"


class GpuUnavailable(RuntimeError):
    """The model could not be moved onto the GPU (CUDA out of memory). It is back on
    the CPU, intact; the request runs there (``XTTS_CPU_FALLBACK``) or is answered 503
    with this reason."""


def _drop_cuda_attributes(model) -> None:
    """Drop CUDA tensors held as plain attributes, which ``Module.cpu()`` does not move (#22).

    coqui-tts's GPT2InferenceModel keeps the last conditioning prefix as ``cached_prefix_emb``; every generation
    sets it again, so dropping it after a demote costs nothing.
    """
    for module in model.modules():
        cached = getattr(module, "cached_prefix_emb", None)
        if cached is not None and getattr(cached, "is_cuda", False):
            module.cached_prefix_emb = None


def _release_cuda_cache() -> None:
    """Return the allocator's free blocks to the driver once the model is off the GPU (#22).

    cuBLAS keeps a workspace per stream inside the caching allocator, and while it lives ``empty_cache()`` cannot
    release the segment it sits in. Clearing the workspaces first lets the whole cache go; cuBLAS allocates a new
    one on the next matmul. What stays is the CUDA context itself, which only ending the process frees.
    """
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return
    gc.collect()
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is not None:
        clear()
    torch.cuda.empty_cache()


# ── Weights: checkpoints and their hashes ────────────────────────────────────

# What a checkpoint directory must hold: what Xtts.load_checkpoint reads. speakers_xtts.pth is optional (without it
# the model has no built-in speakers, so a request needs a voice_clone_id).
CHECKPOINT_REQUIRED = ("config.json", "model.pth", "vocab.json")
CHECKPOINT_OPTIONAL = ("speakers_xtts.pth",)


class CheckpointRefused(Exception):
    """``POST /admin/load`` can't use the named checkpoint; ``status`` is the HTTP answer."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _resolve_checkpoint(name: str) -> Path:
    """The directory of checkpoint *name* under ``XTTS_CHECKPOINT_ROOT``, checked before anything is unloaded.

    503 when no root is configured, 400 for a name that escapes the root (argus-cortex ``safe_join``: ``..``, an
    absolute path, a symlink out), 404 when there is no such directory, 422 when it isn't an XTTS checkpoint."""
    if not CHECKPOINT_ROOT:
        raise CheckpointRefused(503, "XTTS_CHECKPOINT_ROOT is not set: this engine loads no checkpoints")
    root = Path(CHECKPOINT_ROOT)
    if not root.is_dir():
        raise CheckpointRefused(503, f"XTTS_CHECKPOINT_ROOT ({CHECKPOINT_ROOT}) is not a directory")
    try:
        path = safe_join(root, name, label="checkpoint")
    except PathRefused as exc:
        raise CheckpointRefused(400, str(exc)) from None
    if not path.exists():
        raise CheckpointRefused(404, f"no checkpoint {name!r} under XTTS_CHECKPOINT_ROOT")
    if not path.is_dir():
        raise CheckpointRefused(422, f"checkpoint {name!r} is not a directory")
    base = root.resolve()
    missing = [f for f in CHECKPOINT_REQUIRED if not (path / f).is_file()]
    if missing:
        raise CheckpointRefused(
            422, f"checkpoint {name!r} is not an XTTS checkpoint: missing {', '.join(missing)}"
        )
    for f in CHECKPOINT_REQUIRED + CHECKPOINT_OPTIONAL:
        if (path / f).exists() and not (path / f).resolve().is_relative_to(base):
            raise CheckpointRefused(400, f"checkpoint {name!r}: {f} links outside XTTS_CHECKPOINT_ROOT")
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointRefused(422, f"checkpoint {name!r}: config.json is not JSON ({exc})") from None
    if not isinstance(config, dict) or config.get("model") != "xtts":
        raise CheckpointRefused(422, f"checkpoint {name!r}: config.json is not an XTTS config (model != \"xtts\")")
    return path


# sha256 by (resolved path, size, mtime): a reload of unchanged files (after an idle unload) is not hashed again.
_hash_cache: dict[tuple[str, int, int], str] = {}
_hash_cache_lock = threading.Lock()


def _sha256(path: Path) -> str:
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    with _hash_cache_lock:
        if key in _hash_cache:
            return _hash_cache[key]
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    value = digest.hexdigest()
    with _hash_cache_lock:
        _hash_cache[key] = value
    return value


def _weight_files(paths: dict[str, Path]) -> list[dict]:
    """``[{name, sha256, bytes}]`` for the files a load read. Blocking: called in the load's worker thread."""
    return [
        {"name": name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        for name, path in sorted(paths.items())
    ]


def _fingerprint(files: Optional[list[dict]]) -> Optional[str]:
    """One sha256 for a loaded weight set, recomputable from ``/health``'s ``weights.files`` (docs/api.md, "Weights
    fingerprint"): one line ``<name>:<sha256>`` per file, the lines sorted byte-wise as UTF-8, each ending in ``\\n``
    (the last one too), sha256 of those bytes, lowercase hex. Identical to the Qwen engine's."""
    if files is None:
        return None
    lines = sorted((f"{f['name']}:{f['sha256']}\n".encode() for f in files))
    return hashlib.sha256(b"".join(lines)).hexdigest()


# ── Seeded generation ────────────────────────────────────────────────────────

SEED_MAX = 2**32 - 1  # numpy's legacy seed range, and every engine's contract bound


class RngBusy(Exception):
    """A generation waited ``XTTS_RNG_WAIT_S`` for the sampler while a seeded one held it: answered 503."""


class _RngGate:
    """Who may draw from torch's process-wide RNG while a generation samples.

    Unseeded generations share it (they run side by side, as before seeds existed). A seeded one runs alone: it
    waits for those in flight, new ones wait for it, and it seeds at the start, so nothing else draws between its
    seed and its last sample. Blocking (threading): enter it from a generation thread (_GENERATION_POOL), never the
    event loop or the shared to_thread pool. Waiting seeded generations go first, so a stream of unseeded ones
    can't starve them. A wait longer than *timeout* raises RngBusy. Entering and leaving needn't happen on the same
    thread (a stream enters before its response starts and its producer thread leaves)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._shared = 0
        self._exclusive = False
        self._waiting = 0

    def enter_shared(self, timeout: Optional[float] = None) -> None:
        with self._cond:
            if not self._cond.wait_for(lambda: not self._exclusive and not self._waiting, timeout):
                raise RngBusy(f"waited {timeout:g}s for a seeded generation to finish")
            self._shared += 1

    def leave_shared(self) -> None:
        with self._cond:
            self._shared -= 1
            self._cond.notify_all()

    def enter_exclusive(self, timeout: Optional[float] = None) -> None:
        with self._cond:
            self._waiting += 1
            try:
                entered = self._cond.wait_for(lambda: not self._exclusive and not self._shared, timeout)
            finally:
                self._waiting -= 1
            if not entered:
                self._cond.notify_all()  # unseeded generations held back by this one may go
                raise RngBusy(f"waited {timeout:g}s for the generations in flight to finish")
            self._exclusive = True

    def leave_exclusive(self) -> None:
        with self._cond:
            self._exclusive = False
            self._cond.notify_all()

    @contextlib.contextmanager
    def shared(self, timeout: Optional[float] = None):
        self.enter_shared(timeout)
        try:
            yield
        finally:
            self.leave_shared()

    @contextlib.contextmanager
    def exclusive(self, timeout: Optional[float] = None):
        self.enter_exclusive(timeout)
        try:
            yield
        finally:
            self.leave_exclusive()


_rng_gate = _RngGate()
# Generations run here, not in the default executor: one waiting for the gate holds a thread, and a full default
# pool would stall every other to_thread call (latents, clone work, WAV encoding, S3 uploads).
_GENERATION_POOL = ThreadPoolExecutor(max_workers=GENERATION_THREADS, thread_name_prefix="xtts-gen")


def _rng_timeout() -> Optional[float]:
    return RNG_WAIT_S if RNG_WAIT_S > 0 else None


def _seed_rngs(seed: int) -> None:
    """Seed every RNG generation can draw from: python's, numpy's and torch's (``manual_seed`` seeds the CPU
    generator and every CUDA device's)."""
    random.seed(seed)
    with contextlib.suppress(ImportError):
        import numpy as np

        np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def _enter_sampling(seed: Optional[int]):
    """Enter the gate for one generation (in a generation thread): seeded and alone, or unseeded and shared.
    Returns the call that leaves it. RngBusy after ``XTTS_RNG_WAIT_S``. See _RngGate."""
    if seed is None:
        _rng_gate.enter_shared(_rng_timeout())
        return _rng_gate.leave_shared
    _rng_gate.enter_exclusive(_rng_timeout())
    try:
        _seed_rngs(seed)
    except BaseException:
        _rng_gate.leave_exclusive()
        raise
    return _rng_gate.leave_exclusive


@contextlib.contextmanager
def _sampling(seed: Optional[int]):
    """Around one generation (in its generation thread). See _enter_sampling."""
    leave = _enter_sampling(seed)
    try:
        yield
    finally:
        leave()


async def _in_generation_thread(fn, *args):
    """``fn(*args)`` on a generation thread (as asyncio.to_thread, context included, but not the shared pool)."""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(_GENERATION_POOL, functools.partial(ctx.run, fn, *args))


async def _enter_sampling_async(seed: Optional[int]):
    """_enter_sampling from the event loop, waiting on a generation thread. A caller cancelled while it waits
    leaves the gate as soon as the wait ends, so a gate entered for nobody is never held."""
    fut = asyncio.get_running_loop().run_in_executor(_GENERATION_POOL, _enter_sampling, seed)
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:

        def _leave(f) -> None:
            if not f.cancelled() and f.exception() is None:
                f.result()()

        fut.add_done_callback(_leave)
        raise


def _rng_busy_response(exc: RngBusy) -> JSONResponse:
    logger.warning("xtts_rng_busy", reason=str(exc))
    return JSONResponse(
        status_code=503, content={"error": f"busy: {exc}"}, headers={"Retry-After": "5"}
    )


class XTTSModelManager:
    """Three-state lifecycle: unloaded → cpu → gpu (mirrors the Qwen sidecar)."""

    def __init__(self) -> None:
        self.keep_alive_gpu = KEEP_ALIVE_GPU
        self.keep_alive_cpu = KEEP_ALIVE_CPU
        self._state = TTSState.UNLOADED
        self._error: Optional[str] = None
        self._model = None
        self._config = None
        self._last_used: float = 0.0
        self._active = 0  # leases held by in-flight work (see lease())
        self._lock = asyncio.Lock()
        self._evict_task: Optional[asyncio.Task] = None
        # A load or GPU<->CPU move runs as its own task that sets the resulting
        # state itself, awaited through asyncio.shield: a cancelled caller (client
        # gone, timeout) can't strand the state in LOADING or naming the old device.
        self._transition: Optional[asyncio.Task] = None
        # The weights a load reads: None for the stock model, else (name, directory) of a checkpoint that
        # /admin/load selected. /admin/load and /admin/unload change it, and so does a load of it that fails (the
        # stock weights are selected again, so a broken checkpoint isn't retried by every request); an idle unload
        # keeps it, so the next load reads the same weights.
        self._checkpoint: Optional[tuple[str, Path]] = None
        # After a promotion runs out of memory, no request tries another until this (time.monotonic()) passes: each
        # would move ~2 GB toward the GPU and back under _lock. /admin/wake (ensure_gpu) still tries.
        self._gpu_refused_until: float = 0.0
        self._gpu_refused_reason: str = ""
        # [{name, sha256, bytes}] of the files the last load of the selected weights read; None until one has.
        # Their fingerprint (_fingerprint), computed with them. Evicting to the CPU and promoting again moves the same
        # weights, so it keeps both; only a load (re)sets them.
        self._weight_files: Optional[list[dict]] = None
        self._weights_fingerprint: Optional[str] = None

    def _set_weight_files(self, files: Optional[list[dict]]) -> None:
        self._weight_files = files
        self._weights_fingerprint = _fingerprint(files)

    @property
    def state(self) -> TTSState:
        return self._state

    @property
    def weights(self) -> dict:
        """``/health``'s ``weights``: the selected checkpoint (None = stock), the files its last load hashed and their
        fingerprint."""
        return {
            "checkpoint": self._checkpoint[0] if self._checkpoint else None,
            "fingerprint": self._weights_fingerprint,
            "files": self._weight_files,
        }

    def identity(self) -> dict:
        """Which weights a request generates with: call it while holding a lease, so it names the weights that lease
        holds (``/admin/load`` is refused while any lease is held, so they can't change before the request ends)."""
        return {
            "checkpoint": self._checkpoint[0] if self._checkpoint else None,
            "weights_fingerprint": self._weights_fingerprint,
        }

    @property
    def model(self):
        return self._model

    @property
    def is_ready(self) -> bool:
        return self._state in (TTSState.CPU, TTSState.GPU)

    def _stock_paths(self) -> tuple[str, Optional[str], str]:
        """``(checkpoint_dir, checkpoint_path, config_path)`` of the stock model, downloaded if it isn't cached."""
        from TTS.utils.manage import ModelManager

        # Download (no-op if cached). coqui-tts may return the checkpoint
        # file itself; load_checkpoint rejects a file passed as checkpoint_dir.
        model_path, config_path, _ = ModelManager().download_model(XTTS_MODEL)
        model_path = os.fspath(model_path)
        if os.path.isfile(model_path):
            checkpoint_path = model_path
            checkpoint_dir = os.path.dirname(model_path)
        else:
            checkpoint_path = None
            checkpoint_dir = model_path
        if config_path is None:
            config_path = os.path.join(checkpoint_dir, "config.json")
        return checkpoint_dir, checkpoint_path, os.fspath(config_path)

    def _load_to_cpu(self) -> None:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        checkpoint = self._checkpoint
        self._set_weight_files(None)  # until this load hashes what it read
        if checkpoint is None:
            logger.info("loading_xtts", model=XTTS_MODEL, target="cpu")
            checkpoint_dir, checkpoint_path, config_path = self._stock_paths()
        else:
            logger.info("loading_xtts_checkpoint", checkpoint=checkpoint[0], target="cpu")
            checkpoint_dir = os.fspath(checkpoint[1])
            checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
            config_path = os.path.join(checkpoint_dir, "config.json")

        config = XttsConfig()
        config.load_json(str(config_path))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=checkpoint_path,
            use_deepspeed=USE_DEEPSPEED,
        )
        model.eval()
        # The files load_checkpoint read, hashed here (the load's worker thread) once per load, never per request.
        # A hash that fails fails the load: nothing is assigned, so no model runs unreported.
        read = {
            "config.json": Path(config_path),
            "model.pth": Path(checkpoint_path or os.path.join(checkpoint_dir, "model.pth")),
        }
        for name in ("vocab.json",) + CHECKPOINT_OPTIONAL:
            if os.path.isfile(os.path.join(checkpoint_dir, name)):
                read[name] = Path(checkpoint_dir, name)
        files = _weight_files(read)
        self._config = config
        self._model = model
        self._set_weight_files(files)
        logger.info("xtts_loaded_cpu", checkpoint=checkpoint[0] if checkpoint else None, files=len(files))

    def _move_to_device(self, device: str) -> None:
        import torch

        if self._model is None:
            return
        if device == "cpu":
            self._model.cpu()
            _drop_cuda_attributes(self._model)
            _release_cuda_cache()
            return
        try:
            self._model.to(torch.device(device))
        except BaseException as exc:
            # A failed .to() leaves some weights on the GPU: put them all back, or every
            # later request finds a model split across devices while the state says cpu.
            self._model.cpu()
            _drop_cuda_attributes(self._model)
            _release_cuda_cache()
            if isinstance(exc, Exception) and _is_oom(exc):
                free = _free_vram_mb()
                raise GpuUnavailable(
                    f"CUDA out of memory moving XTTS onto {device}"
                    + (f" ({free} MiB free)" if free is not None else "")
                ) from exc
            raise

    async def _do_load(self) -> None:
        """Load the selected weights to the CPU. A checkpoint that fails to load (from /admin/load, or a lazy reload
        after an idle unload) is dropped: the stock weights are selected, so the next request loads them rather than
        retrying the broken directory. /health and every response then name stock (checkpoint null)."""
        try:
            await asyncio.to_thread(self._load_to_cpu)
        except BaseException as exc:
            self._state = TTSState.ERROR
            self._error = str(exc) or type(exc).__name__
            if self._checkpoint is not None:
                logger.error(
                    "xtts_checkpoint_dropped", checkpoint=self._checkpoint[0], reason="load failed", error=self._error
                )
                self._error = f"checkpoint {self._checkpoint[0]}: {self._error}"
                self._checkpoint = None
                self._set_weight_files(None)
            logger.error("xtts_load_failed", error=self._error)
            raise
        self._state = TTSState.CPU
        self._error = None

    async def _do_move(self, device: str, state: TTSState) -> None:
        await asyncio.to_thread(self._move_to_device, device)
        self._state = state

    async def _run_transition(self, coro) -> None:
        """Run a load/move as a task and await it shielded. Cancelling the caller
        leaves it to finish and set the state; with nobody left holding the lock
        or a lease, it arms the idle timer so the weights don't stay resident."""

        async def run() -> None:
            await coro
            if self._active == 0 and not self._lock.locked():
                self._last_used = time.time()
                self._arm_evict()

        task = asyncio.create_task(run())
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        self._transition = task
        await asyncio.shield(task)

    async def _await_transition(self) -> None:
        """Let an orphaned load/move finish before the state is read or the model
        freed. Its failure is not ours: the state it left (ERROR) says so."""
        task = self._transition
        if task is not None and not task.done():
            with contextlib.suppress(Exception):  # only waiting for the orphaned task; its own handler records any failure
                await asyncio.shield(task)

    async def _ensure_gpu_locked(self, retry_now: bool = False) -> None:
        """Promote to GPU (loading first if needed). Caller holds ``_lock``.
        Raises unless the model is then resident, so a lease never gets None. Within ``XTTS_OOM_RETRY_AFTER_S`` of
        a promotion that ran out of memory it raises GpuUnavailable without trying, unless *retry_now*."""
        await self._await_transition()
        if self._state == TTSState.GPU and self._model is not None:
            return

        if self._state in (TTSState.UNLOADED, TTSState.ERROR):
            self._state = TTSState.LOADING
            await self._run_transition(self._do_load())

        if self._state == TTSState.CPU:
            if DEVICE.startswith("cuda"):
                if not retry_now and time.monotonic() < self._gpu_refused_until:
                    raise GpuUnavailable(f"{self._gpu_refused_reason} (not retried for {OOM_RETRY_AFTER_S}s)")
                logger.info("xtts_promoting_to_gpu", device=DEVICE)
                try:
                    await self._run_transition(self._do_move(DEVICE, TTSState.GPU))
                except GpuUnavailable as exc:
                    self._gpu_refused_until = time.monotonic() + OOM_RETRY_AFTER_S
                    self._gpu_refused_reason = str(exc)
                    raise
                self._gpu_refused_until = 0.0
                logger.info("xtts_on_gpu")
            else:
                self._state = TTSState.GPU  # CPU-only: treat as ready

        if self._state != TTSState.GPU or self._model is None:
            raise RuntimeError(f"xtts model not ready (state {self._state.value})")

    async def _ensure_loaded_locked(self) -> None:
        """Load to CPU if nothing is resident; a model already on the GPU stays there.
        Caller holds ``_lock``."""
        await self._await_transition()
        if self._state in (TTSState.UNLOADED, TTSState.ERROR):
            self._state = TTSState.LOADING
            await self._run_transition(self._do_load())
        if self._state not in (TTSState.CPU, TTSState.GPU) or self._model is None:
            raise RuntimeError(f"xtts model not ready (state {self._state.value})")

    def _cpu_work_in_flight(self) -> bool:
        """Leases are held on a model on the CPU of a GPU deployment: moving it would break that work."""
        return DEVICE.startswith("cuda") and self._state == TTSState.CPU and self._active > 0

    def _arm_evict_if_unleased(self) -> None:
        """After a failed acquire or promotion: a model left loaded with nothing in flight still gets its idle
        timer, so a refused request doesn't keep it in RAM for good."""
        if self._active == 0 and self.is_ready:
            self._last_used = time.time()
            self._arm_evict()

    async def ensure_gpu(self) -> None:
        """Guarantee the model is on GPU and reset the idle timer.

        This takes no lease, so it only suits callers that don't then use the
        model (preload, /admin/wake). Inference must go through ``lease()``.
        While CPU work holds a lease the model stays on the CPU (it is loaded, not
        moved); the next GPU request after that work promotes it. Raises
        :class:`GpuUnavailable` when the promotion runs out of memory."""
        async with self._lock:
            try:
                if self._cpu_work_in_flight():
                    logger.info("xtts_gpu_deferred_cpu_work_in_flight", active=self._active)
                    await self._ensure_loaded_locked()
                else:
                    await self._ensure_gpu_locked(retry_now=True)  # an explicit wake tries even in the cooldown
            except BaseException:
                self._arm_evict_if_unleased()
                raise
            self._last_used = time.time()
            if self._active == 0:
                self._arm_evict()

    @property
    def active(self) -> int:
        return self._active

    async def acquire(self, device: Optional[str] = None) -> str:
        """Take a lease: until the matching ``release()`` the model is neither
        moved nor unloaded. Returns where it runs (``cpu`` or ``XTTS_DEVICE``).

        By default the model is promoted to the GPU. ``device="cpu"`` uses it where
        it is, loading to CPU if needed (a model already on the GPU runs there: it
        holds that VRAM either way). A promotion that runs out of memory runs on the
        CPU (``XTTS_CPU_FALLBACK``) or raises :class:`GpuUnavailable`. While CPU work
        holds a lease the model can't move, so other requests join it there."""
        async with self._lock:
            try:
                if not DEVICE.startswith("cuda"):
                    await self._ensure_gpu_locked()  # CPU-only deployment: "gpu" is the CPU
                elif device == "cpu":
                    await self._ensure_loaded_locked()
                elif self._cpu_work_in_flight():
                    logger.info("xtts_gpu_deferred_cpu_work_in_flight", active=self._active)
                    await self._ensure_loaded_locked()
                else:
                    try:
                        await self._ensure_gpu_locked()
                    except GpuUnavailable as exc:
                        if not CPU_FALLBACK:
                            raise
                        logger.warning("xtts_gpu_unavailable_running_on_cpu", reason=str(exc))
                        await self._ensure_loaded_locked()
            except BaseException:
                self._arm_evict_if_unleased()
                raise
            self._active += 1
            self._last_used = time.time()
            self._cancel_evict()
            return "cpu" if DEVICE.startswith("cuda") and self._state == TTSState.CPU else DEVICE

    def release(self) -> None:
        """Drop a lease (event-loop thread only). The idle timer starts from
        the end of the work, and eviction is re-armed once nothing is in flight."""
        self._active = max(0, self._active - 1)
        self._last_used = time.time()
        if self._active == 0:
            self._arm_evict()

    @asynccontextmanager
    async def lease(self, device: Optional[str] = None):
        """``acquire()``/``release()`` around the body; yields where the model runs."""
        ran_on = await self.acquire(device)
        try:
            yield ran_on
        finally:
            self.release()

    async def demote_to_cpu(self) -> bool:
        """Move the model to CPU. Skipped (False) while a lease is held."""
        async with self._lock:
            if self._active > 0:
                logger.info("xtts_demote_deferred", active=self._active)
                return False
            await self._await_transition()
            if self._state != TTSState.GPU:
                return False
            if DEVICE == "cpu":
                return False
            logger.info("xtts_demoting_to_cpu")
            await self._run_transition(self._do_move("cpu", TTSState.CPU))
            logger.info("xtts_on_cpu")
            return True

    async def evict(self) -> bool:
        """``/admin/evict``: demote to CPU, then start the CPU-stage timer, so
        ``keep_alive_cpu`` decides when the model leaves RAM whatever
        ``keep_alive_gpu`` is (#52). False when nothing was demoted."""
        if not await self.demote_to_cpu():
            return False
        if self._active == 0:
            self._arm_evict()
        return True

    async def unload(self, force: bool = False, to_stock: bool = False) -> bool:
        """Drop the model entirely. Skipped (False) while a lease is held,
        unless ``force`` (app shutdown). ``to_stock`` (``/admin/unload``) also
        drops a loaded checkpoint, so the next load reads the stock weights;
        the idle unload keeps it."""
        async with self._lock:
            if self._active > 0 and not force:
                logger.info("xtts_unload_deferred", active=self._active)
                return False
            # An orphaned load's thread would otherwise assign the model after this frees it.
            await self._await_transition()
            self._cancel_evict()
            self._free_locked()
            if to_stock and self._checkpoint is not None:
                logger.info("xtts_checkpoint_dropped", checkpoint=self._checkpoint[0])
                self._checkpoint = None
                self._set_weight_files(None)
            self._state = TTSState.UNLOADED
            logger.info("xtts_fully_unloaded")
            return True

    def _free_locked(self) -> None:
        """Drop the model and return its memory. Caller holds ``_lock``; does not set the state."""
        if self._model is not None:
            del self._model
            self._model = None
        self._config = None
        _release_cuda_cache()

    async def load_weights(self, checkpoint: Optional[tuple[str, Path]]) -> bool:
        """``/admin/load``: replace whatever is loaded by *checkpoint* (``(name, directory)``; None = stock), loaded
        to the CPU now and hashed. False while a lease is held (busy): nothing changes.

        It waits for a load or move in progress, frees the old model, then loads the new one as a shielded
        transition, so a cancelled caller can't strand it half-way. A load that fails leaves nothing loaded and the
        stock weights selected (_do_load; the state says ``error``; the next request loads stock), and raises."""
        async with self._lock:
            if self._active > 0:
                logger.info("xtts_load_refused_busy", active=self._active)
                return False
            await self._await_transition()
            self._cancel_evict()
            self._free_locked()
            self._checkpoint = checkpoint
            self._set_weight_files(None)
            self._state = TTSState.LOADING
            await self._run_transition(self._do_load())
            self._last_used = time.time()
            if self._active == 0:
                self._arm_evict()
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
            # On a CPU-only device "GPU" is the resident state: it gets this
            # keep-alive, then unloads from it.
            if self._state == TTSState.GPU:
                # A negative keep-alive pins the resident model. It skips only
                # this step: a model evicted to the CPU still gets the CPU stage (#52).
                if self.keep_alive_gpu < 0:
                    return
                if self.keep_alive_gpu > 0:
                    await asyncio.sleep(self.keep_alive_gpu)
                    if time.time() - self._last_used < self.keep_alive_gpu:
                        return
                if self._state == TTSState.GPU and DEVICE != "cpu":
                    # A held lease defers this; release() re-arms the timer.
                    if not await self.demote_to_cpu():
                        return
            elif self._state != TTSState.CPU:
                return
            if self._active > 0 or self._state not in (TTSState.CPU, TTSState.GPU):
                return

            if self.keep_alive_cpu < 0:
                return
            if self.keep_alive_cpu > 0:
                cpu_since = time.time()
                await asyncio.sleep(self.keep_alive_cpu)
                if self._last_used > cpu_since:  # used during the CPU stage
                    return
            await self.unload()
        except asyncio.CancelledError:
            pass


# ── Global state ─────────────────────────────────────────────────────────────

_mgr = XTTSModelManager()
_s3_client = None
_clone_cache: OrderedDict = OrderedDict()
_clone_cache_lock = threading.Lock()
_vad_model = None
_vad_lock = threading.Lock()


def _get_vad_model():
    """Lazy-load the bundled Silero VAD model (no network/torch.hub fetch)."""
    global _vad_model
    with _vad_lock:
        if _vad_model is None:
            from silero_vad import load_silero_vad

            _vad_model = load_silero_vad()
        return _vad_model


def _require_s3_credentials() -> None:
    """Raise unless S3 is usable: both credentials, an http(s) endpoint and a bucket. Called at startup, and again
    before the first client is built for anything that skips the lifespan. An empty S3_ENDPOINT_URL used to pass
    here and fail every /synthesise after the audio was rendered (``ValueError: Invalid endpoint``, #22)."""
    if not (S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY):
        raise RuntimeError("S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be set; there are no defaults")
    endpoint = urlsplit(S3_ENDPOINT_URL.strip())
    if endpoint.scheme not in ("http", "https") or not endpoint.netloc:
        raise RuntimeError(f"S3_ENDPOINT_URL must be an http(s) URL, got {S3_ENDPOINT_URL!r}")
    if not S3_BUCKET_NAME.strip():
        raise RuntimeError("S3_BUCKET_NAME must be set")


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _require_s3_credentials()
        import boto3

        _s3_client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        )
    return _s3_client


def _upload_to_s3(data: bytes, key: str, content_type: str = "audio/wav") -> str:
    client = _get_s3_client()
    client.put_object(Bucket=S3_BUCKET_NAME, Key=key, Body=data, ContentType=content_type)
    return f"{S3_PUBLIC_URL}/{key}"


def _s3_object_exists(key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        _get_s3_client().head_object(Bucket=S3_BUCKET_NAME, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    return True


def _device_tensor(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _load_prompt(data) -> dict:
    """Deserialize and validate a stored clone (``prompt.pt``).

    Clone blobs come from S3 or an uploaded archive, so they're untrusted:
    ``weights_only=True`` restricts unpickling to tensors and plain containers
    (no arbitrary objects, so no code execution). Clones this server writes
    are ``{"gpt_cond_latent": Tensor, "speaker_embedding": Tensor}``, which
    loads as-is without any extra safe globals. Raises ``ValueError``."""
    import torch

    buf = data if hasattr(data, "read") else io.BytesIO(data)
    try:
        loaded = torch.load(buf, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Invalid or untrusted prompt.pt (not a tensor archive)") from exc

    if not (isinstance(loaded, dict) and "gpt_cond_latent" in loaded and "speaker_embedding" in loaded):
        raise ValueError("prompt.pt must contain gpt_cond_latent and speaker_embedding")
    gpt, spk = loaded["gpt_cond_latent"], loaded["speaker_embedding"]
    for t in (gpt, spk):
        if not (torch.is_tensor(t) and t.is_floating_point() and t.numel() > 0):
            raise ValueError("prompt.pt latents must be non-empty float tensors")
    return {"gpt_cond_latent": gpt.detach().cpu(), "speaker_embedding": spk.detach().cpu()}


def _get_clone_latents(clone_id: str):
    """Load (gpt_cond_latent, speaker_embedding) for a stored clone, cached."""
    with _clone_cache_lock:
        if clone_id in _clone_cache:
            _clone_cache.move_to_end(clone_id)
            return _clone_cache[clone_id]

    from botocore.exceptions import ClientError

    client = _get_s3_client()
    buf = io.BytesIO()
    try:
        client.download_fileobj(S3_BUCKET_NAME, clone_id, buf)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        logger.error("clone_latent_s3_error", clone_id=clone_id, error_code=error_code)
        raise ValueError(f"Voice clone not found: {clone_id}") from exc
    buf.seek(0)
    try:
        prompt = _load_prompt(buf)
    except ValueError as exc:
        logger.error("clone_latent_invalid", clone_id=clone_id, error=str(exc))
        raise ValueError(f"Voice clone is not a valid XTTS clone: {clone_id}") from exc

    with _clone_cache_lock:
        _clone_cache[clone_id] = prompt
        if len(_clone_cache) > _CLONE_CACHE_MAX:
            _clone_cache.popitem(last=False)

    return prompt


def _resolve_latents(req) -> tuple:
    """Return (gpt_cond_latent, speaker_embedding) on the model's device for a
    request, from a stored clone, a built-in speaker, or a default fallback."""
    model = _mgr.model
    device = next(model.parameters()).device

    if getattr(req, "voice_clone_id", None):
        prompt = _get_clone_latents(req.voice_clone_id)
        gpt = _device_tensor(prompt["gpt_cond_latent"], device)
        spk = _device_tensor(prompt["speaker_embedding"], device)
        return gpt, spk

    # Built-in studio speaker (or the first one as a sane default).
    speaker_name = getattr(req, "speaker", None)
    sm = getattr(model, "speaker_manager", None)
    if sm is not None and getattr(sm, "speakers", None):
        if not speaker_name:
            speaker_name = sorted(sm.speakers.keys())[0]
        if speaker_name not in sm.speakers:
            raise ValueError(f"Unknown speaker: {speaker_name!r}")
        entry = sm.speakers[speaker_name]
        gpt = _device_tensor(entry["gpt_cond_latent"], device)
        spk = _device_tensor(entry["speaker_embedding"], device)
        return gpt, spk

    raise ValueError("No voice_clone_id and no built-in speakers available")


def _audio_duration(path: str) -> float | None:
    """Best-effort clip duration in seconds (soundfile → torchaudio fallback).

    Returns None if neither backend can read the container, so the caller can
    defer to XTTS's own loader rather than rejecting a usable upload.
    """
    with contextlib.suppress(Exception):  # fall through to the torchaudio probe
        import soundfile as sf

        return float(sf.info(path).duration)
    with contextlib.suppress(Exception):  # fall through to the next duration estimate
        import torchaudio

        info = torchaudio.info(path)
        # Some backends report num_frames == 0 for a compressed or streamed
        # container whose length they can't tell up front: unknown, not 0 s.
        if info.sample_rate > 0 and info.num_frames > 0:
            return info.num_frames / info.sample_rate
    return None


# Set by a CUDA error that is not an out-of-memory (device-side assert, illegal
# memory access…). Those leave the process's CUDA context unusable, so every
# later request fails too: /health answers 503 from then on, until a restart.
_gpu_fault: Optional[str] = None


def _is_oom(exc: BaseException) -> bool:
    torch = sys.modules.get("torch")  # imported with the model; nothing raises its errors before that
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if isinstance(oom_type, type) and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()


def _gpu_error_response(exc: RuntimeError, event: str) -> Optional[JSONResponse]:
    """The response for a RuntimeError out of the model, or None if it is not a GPU error.

    Only a real out-of-memory is the retryable 503 "GPU OOM". Any other CUDA
    error is a 500 without the exception text, and marks the worker unhealthy."""
    global _gpu_fault
    if _is_oom(exc):
        gpu_info = _gpu_memory_info()
        logger.error(event, error=str(exc), gpu=gpu_info)
        return JSONResponse(status_code=503, content={"error": "GPU OOM", "detail": str(exc), "gpu": gpu_info})
    if "cuda" in str(exc).lower():
        _gpu_fault = str(exc)
        logger.error("xtts_gpu_fault", error=str(exc), exc_info=True)
        return JSONResponse(status_code=500, content={"error": "GPU error; this worker needs a restart"})
    return None


def _normalise_language(raw: str) -> str:
    lang = ISO_TO_XTTS.get((raw or "en").lower())
    if lang is None:
        raise ValueError(
            f"Unsupported language: {raw!r}. Supported: {sorted(set(ISO_TO_XTTS.values()))}"
        )
    return lang


def _gpu_memory_info() -> dict | None:
    with contextlib.suppress(Exception):  # /health reports memory when it can; a failed probe omits it
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            return {
                "allocated_mb": round(allocated, 1),
                "reserved_mb": round(reserved, 1),
                "total_mb": round(total, 1),
            }
    return None


def _list_speakers() -> list[str]:
    model = _mgr.model
    sm = getattr(model, "speaker_manager", None) if model is not None else None
    if sm is not None and getattr(sm, "speakers", None):
        return sorted(sm.speakers.keys())
    return []


# ── App lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    _require_s3_credentials()  # fail at startup, not on the first upload
    if PRELOAD:
        try:
            await _mgr.ensure_gpu()
        except GpuUnavailable as exc:
            # No room on the GPU now is no reason to crash-loop: the model is on the CPU and the
            # first request promotes it (or runs there, per XTTS_CPU_FALLBACK).
            logger.warning("xtts_preload_gpu_unavailable", reason=str(exc))
    yield
    await _mgr.unload(force=True)


app = FastAPI(title="Voice Stack — XTTS-v2", version="1.0.0", lifespan=lifespan)


@app.exception_handler(GpuUnavailable)
async def _gpu_unavailable(_request, exc: GpuUnavailable) -> JSONResponse:
    """No room on the GPU and no CPU fallback: busy, with the reason, never a bare 500. The promotion runs inside
    the lease, before any handler's try, so this handler answers it on every route."""
    logger.warning("xtts_gpu_unavailable", reason=str(exc))
    return JSONResponse(
        status_code=503,
        content={"error": str(exc), "detail": str(exc), "gpu": _gpu_memory_info()},
        headers={"Retry-After": str(OOM_RETRY_AFTER_S)},
    )


# ── Request models ────────────────────────────────────────────────────────────


class SynthesiseRequest(BaseModel):
    text: str
    voice_clone_id: Optional[str] = None
    speaker: Optional[str] = None
    voice_description: Optional[str] = None  # unused by XTTS; accepted for compat
    instruction: Optional[str] = None        # unused by XTTS; accepted for compat
    emotion: str = "neutral"                  # unused by XTTS; accepted for compat
    language: str = Field(default="en")
    speed: Optional[float] = None
    # "cpu": don't promote to the GPU (a client sends it when argus-iris refuses its GPU lease).
    device: Optional[Literal["cpu"]] = None
    # Seeds the sampler for this request (python, numpy, torch CPU and CUDA). Same seed, text, voice, device and
    # weights: the same audio, within the limits docs/api.md states.
    seed: Optional[int] = Field(default=None, ge=0, le=SEED_MAX)


# ── Synthesis helpers ──────────────────────────────────────────────────────────


def _run_inference(text: str, language: str, gpt, spk, speed: float, seed: Optional[int] = None):
    with _sampling(seed):
        out = _mgr.model.inference(
            text,
            language,
            gpt,
            spk,
            temperature=TEMPERATURE,
            speed=speed,
            enable_text_splitting=True,
        )
    return out["wav"]


# ── POST /synthesise ────────────────────────────────────────────────────────


@app.post("/synthesise")
async def synthesise(req: SynthesiseRequest):
    """Generate speech, upload WAV to S3, return the public URL."""
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})

    t_req = time.monotonic()
    # The lease keeps the model where it is (GPU, or CPU for device "cpu" or a
    # fallback) from latent resolution to the end of inference; the idle eviction
    # can't move it mid-synthesis. A promotion with no room raises GpuUnavailable
    # (the app's 503 handler) unless XTTS_CPU_FALLBACK runs this on the CPU.
    async with _mgr.lease(req.device) as ran_on:
        t_ready = time.monotonic()
        identity = _mgr.identity()  # the weights this lease holds; /admin/load is 409 until it ends

        try:
            language = _normalise_language(req.language)
            gpt, spk = await asyncio.to_thread(_resolve_latents, req)
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})

        job_id = str(uuid.uuid4())
        speed = req.speed if req.speed and req.speed > 0 else 1.0

        t_gen0 = time.monotonic()
        try:
            wav = await _in_generation_thread(_run_inference, req.text, language, gpt, spk, speed, req.seed)
        except RngBusy as exc:
            return _rng_busy_response(exc)
        except RuntimeError as exc:
            refused = _gpu_error_response(exc, "xtts_gpu_oom")
            if refused is not None:
                return refused
            raise
        except Exception as exc:
            logger.error("xtts_generation_failed", error=str(exc), exc_info=True)
            return JSONResponse(status_code=500, content={"error": f"TTS generation failed: {exc}"})
        t_gen1 = time.monotonic()

    import numpy as np

    audio_data = np.asarray(wav, dtype=np.float32)
    duration = len(audio_data) / SAMPLE_RATE

    import soundfile as sf

    buf = io.BytesIO()
    await asyncio.to_thread(sf.write, buf, audio_data, SAMPLE_RATE, format="WAV")
    wav_bytes = buf.getvalue()
    t_encoded = time.monotonic()

    s3_key = f"generated/voice/{job_id}.wav"
    audio_url = await asyncio.to_thread(_upload_to_s3, wav_bytes, s3_key)
    t_uploaded = time.monotonic()

    logger.info(
        "xtts_synthesised",
        job_id=job_id,
        duration=round(duration, 2),
        language=language,
        ready_s=round(t_ready - t_req, 2),
        generate_s=round(t_gen1 - t_gen0, 2),
        encode_s=round(t_encoded - t_gen1, 2),
        upload_s=round(t_uploaded - t_encoded, 2),
        total_s=round(t_uploaded - t_req, 2),
        rtf=round((t_gen1 - t_gen0) / duration, 2) if duration else None,
        device=ran_on,
    )

    return {
        "audio_url": audio_url,
        "duration": duration,
        "duration_seconds": round(duration, 3),
        "format": "wav",
        "device": ran_on,
        "seed": req.seed,
        "checkpoint": identity["checkpoint"],
        "weights_fingerprint": identity["weights_fingerprint"],
    }


# ── POST /synthesise/stream ───────────────────────────────────────────────────


@app.post("/synthesise/stream")
async def synthesise_stream(req: SynthesiseRequest, request: Request = None):
    """Stream raw 24 kHz mono float32 PCM as it is generated (XTTS
    inference_stream). The client concatenates chunks; the trailing
    `X-Sample-Rate`/`X-Audio-Format` headers describe the byte layout."""
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})

    speed = req.speed if req.speed and req.speed > 0 else 1.0
    loop = asyncio.get_running_loop()

    # One lease spans validation and the whole generation: the model can't be
    # demoted or unloaded between the 422 check and the stream starting (which
    # would mean a reload, or a failure after the 200 headers went out). It is
    # handed to the producer thread, which releases it when it is done.
    ran_on = await _mgr.acquire(req.device)
    handed_off = False
    leave_gate = None
    try:
        identity = _mgr.identity()  # the weights this lease holds; /admin/load is 409 until the stream ends
        # Resolve (and validate) the voice up front so a bad clone id is a 422,
        # not a truncated 200 stream.
        try:
            language = _normalise_language(req.language)
            gpt, spk = await asyncio.to_thread(_resolve_latents, req)
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})
        device = next(_mgr.model.parameters()).device
        gpt, spk = _device_tensor(gpt, device), _device_tensor(spk, device)

        # inference_stream samples lazily, chunk by chunk: a seeded stream keeps the sampler to itself until its last
        # chunk (see _RngGate). Entered here, on a generation thread, before the 200: a wait past XTTS_RNG_WAIT_S is
        # a 503, not an empty stream. The producer thread leaves it.
        try:
            leave_gate = await _enter_sampling_async(req.seed)
        except RngBusy as exc:
            return _rng_busy_response(exc)
        if request is not None and await request.is_disconnected():
            # Gone while it waited: don't take the sampler (and the GPU) for nobody.
            logger.info("xtts_stream_client_gone_before_start")
            return Response(status_code=499)
        # A seeded stream holds the sampler: a client that takes nothing for XTTS_RNG_WAIT_S ends it, so it can't
        # keep every other generation waiting for the full stall timeout.
        stall_timeout = _STREAM_STALL_TIMEOUT_S
        if req.seed is not None and _rng_timeout() is not None:
            stall_timeout = min(stall_timeout, RNG_WAIT_S)

        # Chunks travel on an asyncio.Queue (filled via call_soon_threadsafe), so
        # the consumer awaits without holding an executor thread; ``slots`` bounds
        # the in-flight chunks so a slow client applies backpressure.
        chunks: asyncio.Queue = asyncio.Queue()
        slots = threading.BoundedSemaphore(_STREAM_QUEUE_MAX)
        stop = threading.Event()

        def _deliver(item) -> bool:
            try:
                loop.call_soon_threadsafe(chunks.put_nowait, item)
                return True
            except RuntimeError:  # loop closed (shutdown)
                stop.set()
                return False

        def _put(item: bytes) -> bool:
            deadline = time.monotonic() + stall_timeout
            while not stop.is_set():
                if slots.acquire(timeout=_STREAM_PUT_POLL_S):
                    return _deliver(item)
                if time.monotonic() > deadline:
                    # Consumer stalled (or never started): give up rather than
                    # hold the lease forever.
                    logger.warning("xtts_stream_consumer_stalled")
                    stop.set()
                    return False
            return False

        def _produce():
            """Producer thread: owns the stream's model lease until it returns.

            It runs on its own thread, not the default executor, so concurrent
            streams can't starve the pool that every ``to_thread`` call shares."""
            import torch

            stream = None
            try:
                if stop.is_set():
                    return  # the client left before generation started
                stream = _mgr.model.inference_stream(
                    req.text,
                    language,
                    gpt,
                    spk,
                    temperature=TEMPERATURE,
                    speed=speed,
                    stream_chunk_size=STREAM_CHUNK_SIZE,
                    enable_text_splitting=True,
                )
                for chunk in stream:
                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.detach().cpu().to(torch.float32).numpy()
                    if not _put(chunk.tobytes()):
                        break  # client went away: stop generating
            except Exception as exc:  # surface as stream termination
                logger.error("xtts_stream_failed", error=str(exc), exc_info=True)
                if isinstance(exc, RuntimeError) and not _is_oom(exc) and "cuda" in str(exc).lower():
                    global _gpu_fault
                    _gpu_fault = str(exc)  # /health reports it; see _gpu_error_response
            finally:
                close = getattr(stream, "close", None)
                if close is not None:
                    with contextlib.suppress(Exception):  # closing an abandoned stream generator is best-effort
                        close()
                leave_gate()
                # The end-of-stream sentinel bypasses ``slots``: it is always
                # delivered, so a consumer can never wait forever.
                _deliver(None)
                try:
                    loop.call_soon_threadsafe(_mgr.release)
                except RuntimeError:  # loop already closed (shutdown)
                    pass

        threading.Thread(target=_produce, name="xtts-stream", daemon=True).start()
        handed_off = True
    finally:
        if not handed_off:
            if leave_gate is not None:
                leave_gate()
            _mgr.release()

    async def event_stream():
        try:
            while True:
                item = await chunks.get()
                if item is None:
                    break
                slots.release()
                yield item
        finally:
            stop.set()

    return StreamingResponse(
        event_stream(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Format": "pcm_f32le",
            "X-Channels": "1",
            "X-XTTS-Device": ran_on,
            "X-Echo-Checkpoint": quote(identity["checkpoint"] or ""),  # percent-encoded; empty for stock
            "X-Echo-Weights": identity["weights_fingerprint"] or "",
        },
    )


# ── Reference-clip preprocessing ──────────────────────────────────────────────


def _padded_spans(ts: list[dict], pad: float, sr: int, total_len: int) -> list[tuple[int, int]]:
    """Sample ranges of the VAD speech timestamps (seconds), each widened by
    ``pad`` on both sides. Padded ranges that would overlap are clamped to
    start where the previous one ended, so no sample is emitted twice."""
    spans: list[tuple[int, int]] = []
    last_b = 0
    for t in ts:
        a = max(0, int((t["start"] - pad) * sr), last_b)
        b = min(total_len, int((t["end"] + pad) * sr))
        if b > a:
            spans.append((a, b))
            last_b = b
    return spans


def _normalise_loudness(audio_np, sr: int, report: dict):
    """LUFS-normalise ``audio_np`` to PREPROCESS_TARGET_LUFS with a peak guard,
    falling back to peak normalisation. Records in ``report`` what the returned
    samples actually are: ``lufs_out`` and ``gain_db`` include the guard."""
    import numpy as np

    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loud_in = meter.integrated_loudness(audio_np)
        if np.isfinite(loud_in):
            report["lufs_in"] = round(float(loud_in), 1)
            normed = pyln.normalize.loudness(audio_np, loud_in, PREPROCESS_TARGET_LUFS)
            peak = float(np.max(np.abs(normed))) if normed.size else 0.0
            limited_db = 0.0
            if peak > 0.99:
                # The peak guard scales the clip back down, so it lands below the
                # target by exactly this much: report what was written.
                normed = normed * (0.99 / peak)
                limited_db = 20.0 * math.log10(0.99 / peak)
                report["warnings"].append("peak_limited")
            audio_np = normed.astype(np.float32)
            report["lufs_out"] = round(float(PREPROCESS_TARGET_LUFS + limited_db), 1)
            report["gain_db"] = round(float(PREPROCESS_TARGET_LUFS - loud_in + limited_db), 1)
            report["applied"].append("lufs_norm")
        else:
            report["warnings"].append("lufs_unmeasurable")
    except Exception as exc:
        report["warnings"].append(f"lufs_failed:{exc}")
        peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
        if peak > 0:
            audio_np = (audio_np * (0.97 / peak)).astype(np.float32)
            report["applied"].append("peak_norm")
    return audio_np


def _preprocess_clip(in_path: str) -> tuple[str, dict]:
    """Standardise a reference clip for cloning and report what changed.

    Chain: mono → resample → DC removal → cascaded high-pass (HVAC rumble) →
    Silero VAD split into speech + gap (noise) regions → SNR estimate → gated
    stationary denoise seeded from the gaps (HVAC hum under speech) → length cap
    → LUFS loudness normalisation (peak guarded). Writes a 16-bit WAV temp file.
    Each stage degrades gracefully and is recorded in the returned report so the
    UI can show per-clip results.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import torchaudio.functional as AF

    report: dict = {"applied": [], "warnings": []}

    wav, sr = torchaudio.load(in_path)  # (channels, samples) float32
    report["original_seconds"] = round(wav.shape[-1] / sr, 2)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        report["applied"].append("mono")

    if sr != PREPROCESS_SR:
        wav = AF.resample(wav, sr, PREPROCESS_SR)
        sr = PREPROCESS_SR

    wav = wav - wav.mean()  # DC offset removal

    if PREPROCESS_HIGHPASS_HZ > 0:
        passes = max(1, round(PREPROCESS_HIGHPASS_ORDER / 2))
        for _ in range(passes):
            wav = AF.highpass_biquad(wav, sr, PREPROCESS_HIGHPASS_HZ)
        report["applied"].append(f"highpass{int(PREPROCESS_HIGHPASS_HZ)}Hz/{passes * 12}dBoct")

    samples = wav.squeeze(0).contiguous()
    total_len = samples.shape[-1]

    # Split into speech vs non-speech (gap) regions. Speech feeds the clone; the
    # gaps are a clean, stationary noise profile — ideal for estimating HVAC hum.
    speech = samples
    noise = None
    if PREPROCESS_VAD:
        try:
            from silero_vad import get_speech_timestamps

            model = _get_vad_model()
            vad_wav = samples if sr == 16000 else AF.resample(samples.unsqueeze(0), sr, 16000).squeeze(0)
            ts = get_speech_timestamps(vad_wav, model, sampling_rate=16000, return_seconds=True)
            if ts:
                pad = PREPROCESS_VAD_PAD
                inset = int(0.05 * sr)  # shrink gaps so onsets/breaths don't pollute the noise profile
                min_gap = int(0.05 * sr)
                speech_segs, noise_segs = [], []
                prev_end = 0
                for a, b in _padded_spans(ts, pad, sr, total_len):
                    speech_segs.append(samples[a:b])
                for t in ts:
                    g0, g1 = prev_end + inset, int(t["start"] * sr) - inset
                    if g1 - g0 > min_gap:
                        noise_segs.append(samples[g0:g1])
                    prev_end = int(t["end"] * sr)
                tail = prev_end + inset
                if total_len - tail > min_gap:
                    noise_segs.append(samples[tail:total_len])

                if speech_segs:
                    speech = torch.cat(speech_segs)
                    report["applied"].append("vad_trim")
                if noise_segs:
                    noise = torch.cat(noise_segs)
            else:
                report["warnings"].append("vad_no_speech")
        except Exception as exc:
            report["warnings"].append(f"vad_failed:{exc}")

    # SNR / noise-floor estimate (post high-pass, so it reflects residual hum).
    eps = 1e-9
    snr_db = None
    if noise is not None and noise.numel() > int(0.2 * sr):
        speech_rms = float(torch.sqrt(torch.mean(speech ** 2)).item())
        noise_rms = float(torch.sqrt(torch.mean(noise ** 2)).item())
        report["noise_floor_dbfs"] = round(20.0 * math.log10(noise_rms + eps), 1)
        if speech_rms > 0:
            snr_db = 20.0 * math.log10((speech_rms + eps) / (noise_rms + eps))
            report["snr_db"] = round(snr_db, 1)

    # Targeted stationary denoise (HVAC hum). Gated on SNR so clean clips skip it.
    if PREPROCESS_DENOISE and (snr_db is None or snr_db < PREPROCESS_DENOISE_SNR_DB):
        try:
            import noisereduce as nr

            sp = speech.numpy().astype(np.float32)
            if noise is not None and noise.numel() > int(0.2 * sr):
                den = nr.reduce_noise(
                    y=sp, sr=sr, y_noise=noise.numpy().astype(np.float32),
                    stationary=True, prop_decrease=PREPROCESS_DENOISE_PROP,
                )
                report["applied"].append(f"denoise(seeded,{PREPROCESS_DENOISE_PROP})")
            else:
                # No clean gap to seed from: self-estimate, but be more conservative.
                den = nr.reduce_noise(
                    y=sp, sr=sr, stationary=True,
                    prop_decrease=min(PREPROCESS_DENOISE_PROP, 0.6),
                )
                report["applied"].append("denoise(self,0.6)")
            speech = torch.from_numpy(np.ascontiguousarray(den, dtype=np.float32))
        except Exception as exc:
            report["warnings"].append(f"denoise_failed:{exc}")

    samples = speech

    max_samps = int(PREPROCESS_MAX_SECONDS * sr)
    if samples.shape[-1] > max_samps:
        samples = samples[:max_samps]
        report["applied"].append(f"capped{int(PREPROCESS_MAX_SECONDS)}s")

    report["kept_seconds"] = round(samples.shape[-1] / sr, 2)
    audio_np = samples.numpy().astype(np.float32)

    audio_np = _normalise_loudness(audio_np, sr, report)

    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    sf.write(out.name, audio_np, sr, subtype="PCM_16")
    report["sample_rate"] = sr
    return out.name, report


def _conditioning_latents(paths: list[str]):
    """Compute (gpt_cond_latent, speaker_embedding) with our tuned windows."""
    return _mgr.model.get_conditioning_latents(
        paths,
        gpt_cond_len=GPT_COND_LEN,
        gpt_cond_chunk_len=GPT_COND_CHUNK_LEN,
        max_ref_length=MAX_REF_LEN,
    )


# ── Clip scoring / ranking ────────────────────────────────────────────────────


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _quick_clip_metrics(path: str) -> dict:
    """Lightweight per-clip metrics for scoring — no denoise/LUFS/file writes.

    Measures clipping/peak on the raw audio and uses Silero VAD (at 16 kHz) to
    derive usable speech seconds + an SNR/noise-floor estimate from the gaps.
    SNR is intentionally measured on the *raw* signal (pre-denoise) so the score
    reflects how noisy the source actually is.
    """
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
            # VAD ran and found no speech: none of the clip is usable. (The
            # raw_seconds fallback above stands only when VAD fails to run.)
            m["speech_seconds"] = 0.0
            m["flags"].append("vad_no_speech")
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
    """Score each candidate clip for cloning suitability and recommend a subset
    + an optimal ordering (best anchor first, for the GPT conditioning window).

    Per clip we measure lightweight quality metrics (clipping, SNR, usable
    speech seconds) and a speaker embedding. Clips are then scored on quality and
    on *consistency* — cosine similarity to the embedding centroid — so
    recordings from a different room/mic/voice surface as low-scoring outliers.
    The heavy denoise/LUFS pipeline is skipped here; it runs at clone time.
    """
    import torch

    model = _mgr.model
    items: list[dict] = []
    embeddings: list[Optional["torch.Tensor"]] = []

    for path, name in zip(raw_paths, names):
        item: dict = {"name": name, "flags": []}
        try:
            item.update(_quick_clip_metrics(path))
        except Exception as exc:
            item["flags"].append(f"unreadable:{exc}")
            items.append(item)
            embeddings.append(None)
            continue

        emb = None
        try:
            _gpt, spk = model.get_conditioning_latents([path])
            emb = spk.detach().cpu().flatten().float()
            emb = emb / (emb.norm() + 1e-9)
        except Exception as exc:
            item["flags"].append(f"embed_failed:{exc}")

        embeddings.append(emb)
        items.append(item)

    # Consistency: cosine similarity of each embedding to the centroid.
    valid = [e for e in embeddings if e is not None]
    centroid = None
    if valid:
        centroid = torch.stack(valid).mean(dim=0)
        centroid = centroid / (centroid.norm() + 1e-9)

    for item, emb in zip(items, embeddings):
        item["consistency"] = (
            round(float(torch.dot(emb, centroid).item()), 3)
            if (emb is not None and centroid is not None) else None
        )

    _score_clips(items)

    # Optimal sequence: best anchor first (drives the GPT conditioning window).
    # Unreadable clips (anchor -1) sort last.
    order = sorted(range(len(items)), key=lambda i: items[i]["_anchor"], reverse=True)
    for rank, idx in enumerate(order):
        items[idx]["rank"] = rank
    for item in items:
        item.pop("_anchor", None)

    return {
        "clips": items,
        "order": order,
        "threshold": ANALYZE_SCORE_THRESHOLD,
        "gpt_cond_len": GPT_COND_LEN,
    }


def _score_clips(items: list[dict]) -> None:
    """Score each clip (0–100), set ``recommended``, flags and the ``_anchor`` sort key.

    A clip that couldn't be read scores 0, is never recommended and sorts last.
    A clip whose speaker embedding failed has no consistency evidence, so it
    gets none (0) rather than a neutral half."""
    for item in items:
        if any(f.startswith("unreadable") for f in item["flags"]):
            item["consistency"] = None
            item["score"] = 0
            item["_anchor"] = -1.0
            item["recommended"] = False
            item["flags"] = sorted(set(item["flags"]))
            continue
        snr = item.get("snr_db")
        q_snr = _clamp01((snr - 10.0) / 25.0) if snr is not None else 0.7
        speech = item.get("speech_seconds") or 0.0
        q_dur = _clamp01(speech / 6.0)
        q_clip = 1.0 - _clamp01((item.get("clip_ratio") or 0.0) / 0.02)
        quality = 0.4 * q_snr + 0.35 * q_dur + 0.25 * q_clip

        cons = item.get("consistency")
        cons_norm = _clamp01((cons - 0.5) / 0.45) if cons is not None else 0.0

        score = round((0.65 * quality + 0.35 * cons_norm) * 100)
        item["score"] = score
        item["_anchor"] = 0.45 * quality + 0.3 * cons_norm + 0.25 * _clamp01(speech / float(GPT_COND_LEN))

        if (item.get("clip_ratio") or 0.0) > 0.01:
            item["flags"].append("clipping")
        if snr is not None and snr < 15:
            item["flags"].append("noisy")
        if speech and speech < 2.0:
            item["flags"].append("very_short")
        if cons_norm < 0.3:
            item["flags"].append("outlier")
        item["recommended"] = score >= ANALYZE_SCORE_THRESHOLD
        item["flags"] = sorted(set(item["flags"]))


class _ClipTooLarge(Exception):
    pass


async def _save_clips(clips: list[UploadFile], raw_paths: list[str]) -> None:
    """Copy each upload to a temp file in 1 MiB chunks, refusing one past
    CLIP_MAX_BYTES before it is all read. Paths are appended to ``raw_paths``
    as they are created, so the caller's cleanup also covers a refused clip."""
    for i, clip in enumerate(clips):
        label = clip.filename or f"clip {i + 1}"
        if clip.size is not None and clip.size > CLIP_MAX_BYTES:
            raise _ClipTooLarge(f"{label} is larger than {CLIP_MAX_BYTES} bytes")
        suffix = os.path.splitext(clip.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            raw_paths.append(f.name)
            total = 0
            while chunk := await clip.read(1024 * 1024):
                total += len(chunk)
                if total > CLIP_MAX_BYTES:
                    raise _ClipTooLarge(f"{label} is larger than {CLIP_MAX_BYTES} bytes")
                f.write(chunk)


@app.post("/analyze-clips")
async def analyze_clips(audio: list[UploadFile] = File(...)):
    """Score candidate clips and recommend a subset + ordering (no clone made)."""
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})

    clips = audio if isinstance(audio, list) else [audio]
    if not clips:
        return JSONResponse(status_code=422, content={"error": "No clips provided"})
    if len(clips) > MAX_CLIPS:
        return JSONResponse(status_code=413, content={"error": f"At most {MAX_CLIPS} clips per request"})

    raw_paths: list[str] = []
    names: list[str] = []
    try:
        try:
            await _save_clips(clips, raw_paths)
        except _ClipTooLarge as exc:
            return JSONResponse(status_code=413, content={"error": str(exc)})
        names = [clip.filename or f"clip-{i + 1}" for i, clip in enumerate(clips)]
        async with _mgr.lease():
            result = await asyncio.to_thread(_analyze_clips, raw_paths, names)
    except GpuUnavailable:
        raise  # the app's handler: 503 with Retry-After
    except RuntimeError as exc:
        refused = _gpu_error_response(exc, "xtts_analyze_gpu_oom")
        if refused is not None:
            return refused
        raise
    finally:
        for path in raw_paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    return result


# ── POST /clone-voice ─────────────────────────────────────────────────────────


# A clone's id is its S3 prefix (voice_clones/<id>/): one plain path segment,
# the same rule as the Qwen engine.
_COMPANION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_COMPANION_ID_ERROR = "companion_id must be 1-64 characters of A-Z, a-z, 0-9, '_' or '-'"


@app.post("/clone-voice")
async def clone_voice(
    audio: list[UploadFile] = File(...),
    reference_text: str = Form(""),
    companion_id: str = Form(...),
    voice_name: str = Form(""),
):
    """Compute XTTS conditioning latents from one or more reference clips.

    XTTS's ``get_conditioning_latents`` accepts a *list* of audio files: it
    concatenates them for the GPT conditioning and averages the per-clip
    speaker embeddings. A single 6 s clip works, but several clips (varied
    prosody, clean takes) yield a more robust, less noise-sensitive clone — so
    this endpoint accepts the `audio` form field repeated N times.

    When `XTTS_PREPROCESS` is on (default), each clip is standardised first
    (mono → resample → high-pass → Silero VAD trim → LUFS normalise); the
    response includes a per-clip `clips` report of what changed.

    `reference_text` is accepted for API parity with the Qwen sidecar but XTTS
    does not require a transcript to clone.

    ``companion_id`` must match ``^[A-Za-z0-9_-]{1,64}$`` (422 otherwise): it is
    the clone's S3 prefix. At most XTTS_MAX_CLIPS clips of XTTS_CLIP_MAX_BYTES each (413).
    """
    if not _COMPANION_ID_RE.fullmatch(companion_id):
        return JSONResponse(status_code=422, content={"error": _COMPANION_ID_ERROR})
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})

    clips = audio if isinstance(audio, list) else [audio]
    if not clips:
        return JSONResponse(status_code=422, content={"error": "At least one reference clip is required"})
    if len(clips) > MAX_CLIPS:
        return JSONResponse(status_code=413, content={"error": f"At most {MAX_CLIPS} clips per request"})

    raw_paths: list[str] = []
    latent_paths: list[str] = []  # what we feed to XTTS (preprocessed when enabled)
    reports: list[dict] = []
    try:
        try:
            await _save_clips(clips, raw_paths)
        except _ClipTooLarge as exc:
            return JSONResponse(status_code=413, content={"error": str(exc)})
        total_duration = 0.0
        measured = False  # some clip's duration is known (even if it is 0 s of speech)
        for clip, raw_path in zip(clips, raw_paths):
            if PREPROCESS:
                try:
                    proc_path, report = await asyncio.to_thread(_preprocess_clip, raw_path)
                    report["name"] = clip.filename
                    latent_paths.append(proc_path)
                    reports.append(report)
                    total_duration += report.get("kept_seconds") or 0.0
                    measured = measured or report.get("kept_seconds") is not None
                    continue
                except Exception as exc:
                    logger.warning("clip_preprocess_failed", name=clip.filename, error=str(exc))
                    reports.append({"name": clip.filename, "warnings": [f"preprocess_failed:{exc}"], "applied": []})
                    # fall through to use the raw clip

            latent_paths.append(raw_path)
            dur = await asyncio.to_thread(_audio_duration, raw_path)
            total_duration += dur or 0.0
            measured = measured or dur is not None

        # Nothing measured means no clip could be probed (e.g. exotic container);
        # let get_conditioning_latents be the real arbiter then. A measured total
        # of 0 is different: preprocessing found no speech in any clip.
        if measured and total_duration < 3.0:
            return JSONResponse(
                status_code=422,
                content={"error": "Usable speech totals under 3 seconds after preprocessing (6+ recommended)"},
            )

        try:
            async with _mgr.lease():
                gpt_cond_latent, speaker_embedding = await asyncio.to_thread(
                    _conditioning_latents, latent_paths
                )
        except GpuUnavailable:
            raise  # the app's handler: 503 with Retry-After
        except RuntimeError as exc:
            refused = _gpu_error_response(exc, "xtts_clone_gpu_oom")
            if refused is not None:
                return refused
            raise
    finally:
        for path in set(raw_paths + latent_paths):
            try:
                os.unlink(path)
            except OSError:
                pass

    import torch

    prompt = {
        "gpt_cond_latent": gpt_cond_latent.detach().cpu(),
        "speaker_embedding": speaker_embedding.detach().cpu(),
    }
    buf = io.BytesIO()
    torch.save(prompt, buf)
    prompt_bytes = buf.getvalue()

    s3_key = f"voice_clones/{companion_id}/prompt.pt"
    await asyncio.to_thread(_upload_to_s3, prompt_bytes, s3_key, "application/octet-stream")

    meta = {
        "voice_clone_id": s3_key,
        "companion_id": companion_id,
        "name": voice_name or companion_id,
        "num_clips": len(clips),
        "total_seconds": round(total_duration, 2),
        "preprocessed": PREPROCESS,
        "created_at": time.time(),
    }
    meta_key = f"voice_clones/{companion_id}/meta.json"
    await asyncio.to_thread(
        _upload_to_s3, json.dumps(meta).encode("utf-8"), meta_key, "application/json"
    )

    with _clone_cache_lock:
        _clone_cache[s3_key] = prompt
        if len(_clone_cache) > _CLONE_CACHE_MAX:
            _clone_cache.popitem(last=False)

    logger.info(
        "voice_cloned",
        companion_id=companion_id,
        clips=len(clips),
        duration=round(total_duration, 2),
        s3_key=s3_key,
    )
    return {**meta, "duration": total_duration, "clips": reports}


# ── GET /clones ───────────────────────────────────────────────────────────────


def _list_clones() -> list[dict]:
    """Enumerate stored clones from S3 by reading their meta.json sidecars."""
    from botocore.exceptions import ClientError

    client = _get_s3_client()
    clones: list[dict] = []
    paginator = client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix="voice_clones/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/meta.json"):
                    continue
                buf = io.BytesIO()
                try:
                    client.download_fileobj(S3_BUCKET_NAME, key, buf)
                    meta = json.loads(buf.getvalue().decode("utf-8"))
                    if not isinstance(meta, dict):
                        raise ValueError("meta.json is not an object")
                    clones.append(meta)
                except (ClientError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning("clone_meta_unreadable", key=key, error=str(exc))
    except ClientError as exc:
        logger.error("clone_list_failed", error=str(exc))
        raise

    def _created(c: dict) -> float:
        # One type for the sort key: a sidecar with a string or missing created_at sorts last.
        value = c.get("created_at")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0

    clones.sort(key=_created, reverse=True)
    return clones


@app.get("/clones")
async def list_clones():
    """List stored voice clones (id, name, clip count, duration)."""
    try:
        clones = await asyncio.to_thread(_list_clones)
    except Exception as exc:
        logger.error("clone_list_error", error=str(exc))
        return JSONResponse(status_code=500, content={"error": f"Failed to list clones: {exc}"})
    return {"clones": clones}


# Clone archive format (download and import), as on Miso (#20). The latents travel as prompt.safetensors:
# gpt_cond_latent and speaker_embedding as tensors, and format/engine strings in the safetensors `__metadata__`
# header. A safetensors file is a JSON header plus raw tensor bytes, so parsing one never runs code. A legacy
# prompt.pt (a torch pickle) in an uploaded archive is refused unread: torch.load(weights_only=True) on untrusted
# bytes is GHSA-63cw-57p8-fm3p on the torch this image installs (cu124 stops at 2.6.0; fixed in 2.10), and
# unpickler bypasses keep turning up. Stored clones stay prompt.pt: this server wrote them.
_ARCHIVE_PROMPT = "prompt.safetensors"
_LEGACY_ARCHIVE_PROMPT = "prompt.pt"
_ARCHIVE_FORMAT = "xtts-clone"
_ARCHIVE_FORMAT_VERSION = "1"
_LATENT_KEYS = ("gpt_cond_latent", "speaker_embedding")


def _to_array(tensor):
    """A latent tensor as a float32 numpy array (the archive's form)."""
    import torch

    return tensor.detach().cpu().to(torch.float32).contiguous().numpy()


def _from_array(array):
    """A validated archive array back as a CPU tensor."""
    import torch

    return torch.from_numpy(array.copy())


def _prompt_to_safetensors(prompt: dict) -> bytes:
    """Serialise validated latents as the archive's prompt.safetensors."""
    from safetensors.numpy import save

    arrays = {k: _to_array(prompt[k]) for k in _LATENT_KEYS}
    metadata = {"format": _ARCHIVE_FORMAT, "format_version": _ARCHIVE_FORMAT_VERSION, "engine": "xtts"}
    return save(arrays, metadata=metadata)


def _safetensors_metadata(data: bytes) -> dict:
    """The ``__metadata__`` string map from a safetensors header, read as JSON (no tensors)."""
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


def _prompt_from_safetensors(data: bytes) -> dict:
    """Parse and validate an uploaded prompt.safetensors into latents. Only safetensors' own parser touches the
    bytes (into numpy arrays): no pickle, no torch.load. Raises ``ValueError``."""
    import numpy as np
    from safetensors.numpy import load

    metadata = _safetensors_metadata(data)
    engine = metadata.get("engine")
    if engine is not None and engine.strip().lower() != "xtts":
        raise ValueError(f"prompt.safetensors is a {engine[:32]!r} clone, not an XTTS one")
    try:
        arrays = load(data)
    except Exception as exc:
        raise ValueError(f"Invalid prompt.safetensors: {exc}") from exc
    if not all(k in arrays for k in _LATENT_KEYS):
        raise ValueError("prompt.safetensors must contain gpt_cond_latent and speaker_embedding")
    for k in _LATENT_KEYS:
        a = arrays[k]
        if not (a.dtype.kind == "f" and a.size > 0 and np.isfinite(a).all()):
            raise ValueError("prompt.safetensors latents must be non-empty, finite float tensors")
    return {k: _from_array(arrays[k].astype(np.float32)) for k in _LATENT_KEYS}


def _build_clone_zip(companion_id: str) -> bytes:
    """Bundle a stored clone as prompt.safetensors (+ meta.json if present). The stored prompt.pt was written by
    this server and loads through _load_prompt; old and new clones alike leave as safetensors."""
    import zipfile
    from botocore.exceptions import ClientError

    client = _get_s3_client()
    prompt_key = f"voice_clones/{companion_id}/prompt.pt"
    meta_key = f"voice_clones/{companion_id}/meta.json"

    prompt_buf = io.BytesIO()
    try:
        client.download_fileobj(S3_BUCKET_NAME, prompt_key, prompt_buf)
    except ClientError as exc:
        raise FileNotFoundError(f"Voice clone not found: {companion_id}") from exc
    prompt_bytes = _prompt_to_safetensors(_load_prompt(prompt_buf.getvalue()))

    meta_buf = io.BytesIO()
    have_meta = True
    try:
        client.download_fileobj(S3_BUCKET_NAME, meta_key, meta_buf)
    except ClientError:
        have_meta = False

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_ARCHIVE_PROMPT, prompt_bytes)
        if have_meta:
            zf.writestr("meta.json", meta_buf.getvalue())
    return zip_buf.getvalue()


@app.get("/clones/{companion_id}/download")
async def download_clone(companion_id: str):
    """Download a clone's latents + metadata sidecar as a zip."""
    if not _COMPANION_ID_RE.fullmatch(companion_id):
        return JSONResponse(status_code=422, content={"error": _COMPANION_ID_ERROR})
    try:
        data = await asyncio.to_thread(_build_clone_zip, companion_id)
    except FileNotFoundError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    except Exception as exc:
        logger.error("clone_download_error", companion_id=companion_id, error=str(exc))
        return JSONResponse(status_code=500, content={"error": f"Failed to bundle clone: {exc}"})

    headers = {"Content-Disposition": f'attachment; filename="{companion_id}.zip"'}
    return Response(content=data, media_type="application/zip", headers=headers)


def _delete_clone(companion_id: str) -> int:
    """Delete every S3 object under ``voice_clones/{companion_id}/``.

    Returns the number of objects removed and evicts the clone's cached latents.
    Raises ``FileNotFoundError`` when nothing exists under the prefix.
    """
    from botocore.exceptions import ClientError  # noqa: F401  (parity w/ helpers)

    client = _get_s3_client()
    prefix = f"voice_clones/{companion_id}/"
    keys: list[dict] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append({"Key": obj["Key"]})

    if not keys:
        raise FileNotFoundError(f"Voice clone not found: {companion_id}")

    deleted = 0
    failed: list[str] = []
    try:
        for i in range(0, len(keys), 1000):  # S3 delete_objects caps at 1000/call
            batch = keys[i : i + 1000]
            # A per-key failure is not an exception: it comes back in "Errors".
            resp = client.delete_objects(Bucket=S3_BUCKET_NAME, Delete={"Objects": batch}) or {}
            deleted += len(resp.get("Deleted") or [])
            failed += [f"{e.get('Key')}: {e.get('Code')}" for e in resp.get("Errors") or []]
    finally:
        with _clone_cache_lock:
            _clone_cache.pop(f"{prefix}prompt.pt", None)
    if failed:
        logger.error("clone_delete_partial", companion_id=companion_id, deleted=deleted, failed=failed)
        raise RuntimeError(f"{len(failed)} of {len(keys)} objects were not deleted ({'; '.join(failed[:5])})")
    return deleted


@app.delete("/clones/{companion_id}")
async def delete_clone(companion_id: str):
    """Permanently delete a stored voice clone (prompt.pt + meta.json) from S3.
    A partial delete (some objects refused by S3) is a 500, never ``deleted``."""
    if not _COMPANION_ID_RE.fullmatch(companion_id):
        return JSONResponse(status_code=422, content={"error": _COMPANION_ID_ERROR})
    try:
        objects = await asyncio.to_thread(_delete_clone, companion_id)
    except FileNotFoundError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    except Exception as exc:
        logger.error("clone_delete_error", companion_id=companion_id, error=str(exc))
        return JSONResponse(status_code=500, content={"error": f"Failed to delete clone: {exc}"})
    logger.info("voice_deleted", companion_id=companion_id, objects=objects)
    return {"status": "deleted", "companion_id": companion_id, "objects": objects}


def _slug_id(value: str) -> str:
    """A valid companion id (``_COMPANION_ID_RE``) from free text."""
    s = (value or "").strip().lower()
    out = "".join(ch if ch.isascii() and ch.isalnum() else "-" for ch in s).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out[:64].strip("-") or f"imported-{int(time.time())}"


_MAX_PROMPT_BYTES = 64 * 1024 * 1024
_MAX_META_BYTES = 64 * 1024
_MAX_ARCHIVE_ENTRIES = 16
# The upload itself: a prompt.safetensors at its cap plus a little for meta.json and zip overhead.
_MAX_ARCHIVE_BYTES = _MAX_PROMPT_BYTES + 1024 * 1024


class CloneExistsError(Exception):
    """The import's target companion_id already has a clone (and replace was not set)."""


def _read_member(z, name: str, limit: int) -> bytes:
    """Read one archive member, refusing it past ``limit`` bytes by its declared
    size and by what decompression actually yields (a header can lie)."""
    if z.getinfo(name).file_size > limit:
        raise ValueError(f"{name} is unexpectedly large")
    with z.open(name) as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{name} is unexpectedly large")
    return data


def _import_clone(zip_bytes: bytes, companion_override: str, name_override: str, replace: bool = False) -> dict:
    """Validate and store a clone archive (prompt.safetensors [+ meta.json]).

    The uploaded prompt is parsed by safetensors only and re-serialised from the validated tensors, so the stored
    prompt.pt is clean whatever the archive contained. An archive holding only a legacy prompt.pt is refused
    unread: uploaded bytes never reach torch.load or any other unpickler. Members are read with size caps, so a
    compressed bomb can't exhaust memory.

    The target id comes from the override, else the archive's metadata, so an
    existing clone under it is never replaced unless ``replace`` is true
    (``CloneExistsError`` otherwise).
    """
    import zipfile

    import torch

    try:
        z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a valid zip archive: {exc}") from exc

    names = z.namelist()
    if len(names) > _MAX_ARCHIVE_ENTRIES:
        raise ValueError(f"Archive has more than {_MAX_ARCHIVE_ENTRIES} entries")
    def _member(basename: str):
        return next((n for n in names if n.rsplit("/", 1)[-1] == basename), None)

    prompt_name = _member(_ARCHIVE_PROMPT)
    if not prompt_name:
        if _member(_LEGACY_ARCHIVE_PROMPT):
            raise ValueError(
                "This archive holds a legacy prompt.pt (a pickle), which is no longer imported because "
                "unpickling an uploaded file can run code. Re-export the clone from an XTTS engine running "
                "this version: its download now produces prompt.safetensors."
            )
        raise ValueError(f"Archive does not contain {_ARCHIVE_PROMPT}")

    prompt = _prompt_from_safetensors(_read_member(z, prompt_name, _MAX_PROMPT_BYTES))

    src_meta: dict = {}
    meta_name = next((n for n in names if n.rsplit("/", 1)[-1] == "meta.json"), None)
    if meta_name:
        meta_raw = _read_member(z, meta_name, _MAX_META_BYTES)  # oversized: ValueError → 422
        try:
            src_meta = json.loads(meta_raw.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            src_meta = {}
        if not isinstance(src_meta, dict):
            src_meta = {}

    def _text(value) -> str:
        return value.strip() if isinstance(value, str) else ""

    companion_id = _text(companion_override) or _text(src_meta.get("companion_id"))
    companion_id = _slug_id(companion_id or _text(name_override) or _text(src_meta.get("name")) or "imported-voice")
    if not _COMPANION_ID_RE.fullmatch(companion_id):  # _slug_id guarantees it; belt and braces
        raise ValueError(_COMPANION_ID_ERROR)
    if not replace and _s3_object_exists(f"voice_clones/{companion_id}/prompt.pt"):
        raise CloneExistsError(f"Voice clone {companion_id!r} already exists; send replace=true to overwrite")

    buf = io.BytesIO()
    torch.save(prompt, buf)
    s3_key = f"voice_clones/{companion_id}/prompt.pt"
    _upload_to_s3(buf.getvalue(), s3_key, "application/octet-stream")

    meta = {
        "voice_clone_id": s3_key,
        "companion_id": companion_id,
        "name": name_override or src_meta.get("name") or companion_id,
        "num_clips": src_meta.get("num_clips", 0),
        "total_seconds": src_meta.get("total_seconds", 0),
        "imported": True,
        "created_at": time.time(),
    }
    _upload_to_s3(json.dumps(meta).encode("utf-8"), f"voice_clones/{companion_id}/meta.json", "application/json")

    with _clone_cache_lock:
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
    """Import a previously downloaded voice archive (prompt.safetensors + meta.json zip).

    409 when the target id already has a clone, unless ``replace`` is true; 413
    for an archive over ``_MAX_ARCHIVE_BYTES``."""
    if archive.size is not None and archive.size > _MAX_ARCHIVE_BYTES:
        return JSONResponse(status_code=413, content={"error": f"Archive is larger than {_MAX_ARCHIVE_BYTES} bytes"})
    data = await archive.read(_MAX_ARCHIVE_BYTES + 1)
    if len(data) > _MAX_ARCHIVE_BYTES:
        return JSONResponse(status_code=413, content={"error": f"Archive is larger than {_MAX_ARCHIVE_BYTES} bytes"})
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
    """List built-in XTTS studio speakers."""
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})

    async with _mgr.lease():
        names = _list_speakers()
    supported_langs = sorted(set(ISO_TO_XTTS.values()))
    speakers = [
        {"id": name, "name": name, "language": "en", "languages": supported_langs}
        for name in names
    ]
    return {"speakers": speakers}


# ── GET /health ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"status": "loading"})
    body = {
        "status": "ok",
        "engine": "xtts-v2",
        "capabilities": ["tts"],  # the echo contracts this engine serves (docs/api.md)
        "model": XTTS_MODEL,
        "device": DEVICE,
        "model_state": _mgr.state.value,
        "busy": _mgr.active > 0,
        "idle_s": _idle_s(),
        "keep_alive_gpu": KEEP_ALIVE_GPU,
        "keep_alive_cpu": KEEP_ALIVE_CPU,
        "gpu": _resident_gpu(),
        "weights": _mgr.weights,
        "image": await asyncio.to_thread(_image),
    }
    if _gpu_fault is not None:
        # A fatal CUDA error: fail the container healthcheck so the worker is recycled.
        body["status"] = "gpu_error"
        body["error"] = _gpu_fault
        return JSONResponse(status_code=503, content=body)
    return body


# ── Resident contract (argus-iris argus-http adapter) ─────────────────────────
# /health carries a ``gpu`` block (loaded, model_state, idle_s, free_vram_mb),
# POST /admin/evict demotes to CPU, /admin/unload frees the model, /admin/wake
# preloads it. The admin routes need ``Authorization: Bearer <ARGUS_ADMIN_TOKEN>``.
# They fail closed: with the token unset they answer 503, unless
# ARGUS_ADMIN_INSECURE=1 explicitly opens them (local dev only). Same rule as tts_server.


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


if not os.environ.get("ARGUS_ADMIN_TOKEN", "").strip():
    logger.warning(
        "admin_token_unset",
        admin_routes="open (ARGUS_ADMIN_INSECURE)" if _admin_insecure() else "refused with 503",
        hint="set ARGUS_ADMIN_TOKEN to the suite's shared value",
    )


def _free_vram_mb() -> Optional[int]:
    """Device-wide free VRAM, only when this process already holds a CUDA
    context: creating one just to answer /health would cost VRAM itself."""
    torch = sys.modules.get("torch")
    with contextlib.suppress(Exception):  # free VRAM is optional in /health
        if torch is not None and torch.cuda.is_initialized():
            return int(torch.cuda.mem_get_info()[0] / 1024**2)
    return None


def _resident_gpu() -> dict:
    """``gpu`` block for /health and the admin responses."""
    torch = sys.modules.get("torch")
    try:
        context = torch is not None and torch.cuda.is_initialized()
        mem = _gpu_memory_info() if context else None
    except Exception:
        context, mem = False, None
    return {
        **(mem or {}),
        # The process holds a CUDA context (a few hundred MB that torch can't see or free; only a restart returns it).
        # With the model off the GPU, reserved_mb is ~0 and the context is what iris measures as left (#22).
        "cuda_context": context,
        "loaded": _mgr.state in (TTSState.CPU, TTSState.GPU),
        "model_state": _mgr.state.value,
        "idle_s": _idle_s(),
        "free_vram_mb": _free_vram_mb(),
    }


def _idle_s() -> Optional[float]:
    """Seconds since work last finished; None while busy or never used."""
    last_used = _mgr._last_used
    if not last_used or _mgr.active:
        return None
    return round(time.time() - last_used, 1)


@app.post("/admin/evict", dependencies=[Depends(_require_admin)])
async def admin_evict():
    """Force-demote the model from GPU to CPU, freeing VRAM immediately. The
    CPU-stage timer (XTTS_KEEP_ALIVE_CPU) then decides when it leaves RAM."""
    if not await _mgr.evict() and _mgr.active:
        # In-flight work holds the model; its release re-arms idle eviction.
        return JSONResponse(
            status_code=409,
            content={"status": "busy", "active": _mgr.active, "model_state": _mgr.state.value},
        )
    return {"status": "evicted", "model_state": _mgr.state.value, "gpu": _resident_gpu()}


@app.post("/admin/unload", dependencies=[Depends(_require_admin)])
async def admin_unload():
    """Fully unload the model from memory (GPU + CPU). A loaded checkpoint is dropped too: the next load is stock."""
    was_loaded = _mgr.state in (TTSState.CPU, TTSState.GPU)
    if not await _mgr.unload(to_stock=True):
        return JSONResponse(
            status_code=409,
            content={"status": "busy", "active": _mgr.active, "model_state": _mgr.state.value},
        )
    return {
        "status": "unloaded",
        "unloaded": was_loaded,
        "model_state": _mgr.state.value,
        "gpu": _resident_gpu(),
        "weights": _mgr.weights,
    }


class LoadRequest(BaseModel):
    # A directory name under XTTS_CHECKPOINT_ROOT (may be nested: "forge/run-7"); null loads the stock weights.
    checkpoint: Optional[str] = Field(...)


def _image() -> dict:
    """``/health``'s ``image``: the release and git revision baked in at build time (null for a local build), and the
    runtime libraries (_runtime). A container can't see its own image digest; whoever deployed it (compose, iris)
    knows that. Blocking the first time: call it in a thread."""
    return {"version": ECHO_VERSION, "revision": ECHO_REVISION, "runtime": _runtime()}


# What shapes the audio besides the weights and the code: two builds of one revision can install different
# libraries (#32), so /health names them. Read once (importlib.metadata, and torch for CUDA and cuDNN).
RUNTIME_PACKAGES = ("coqui-tts", "transformers", "torchaudio", "numpy")


@functools.lru_cache(maxsize=1)
def _runtime() -> dict:
    """``/health``'s ``image.runtime``: ``{name: version or None}`` for python, torch, CUDA, cuDNN and
    RUNTIME_PACKAGES. Blocking the first time (it imports torch): call it in a thread."""
    import importlib.metadata
    import platform

    out: dict = {"python": platform.python_version(), "torch": None, "cuda": None, "cudnn": None}
    torch = None
    with contextlib.suppress(ImportError):  # no torch installed (the engines' tests): torch, cuda, cudnn stay null
        import torch
    if torch is not None:
        version = getattr(torch, "__version__", None)
        out["torch"] = str(version) if version else None
        out["cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
        cudnn = None
        with contextlib.suppress(Exception):  # a CPU-only build, or a stand-in torch, has no cuDNN to name
            cudnn = torch.backends.cudnn.version()
        out["cudnn"] = str(cudnn) if cudnn else None
    for name in RUNTIME_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


@app.post("/admin/load", dependencies=[Depends(_require_admin)])
async def admin_load(req: LoadRequest):
    """Load a fine-tuned checkpoint from ``XTTS_CHECKPOINT_ROOT`` (or the stock weights, for ``null``) onto the CPU,
    replacing the loaded model, and hash its files. The next request promotes it to the GPU as usual.

    400 a name that escapes the root, 404 no such checkpoint, 422 not an XTTS checkpoint, 409 busy (a request holds
    the model; nothing changes), 503 no checkpoint root configured, 500 the load failed (nothing is loaded and the
    stock weights are selected)."""
    selected = None
    if req.checkpoint is not None:
        try:
            selected = (req.checkpoint, _resolve_checkpoint(req.checkpoint))
        except CheckpointRefused as exc:
            logger.warning("xtts_checkpoint_refused", checkpoint=req.checkpoint, status=exc.status, error=str(exc))
            return JSONResponse(status_code=exc.status, content={"error": str(exc)})
    try:
        loaded = await _mgr.load_weights(selected)
    except Exception as exc:
        logger.error("xtts_checkpoint_load_failed", checkpoint=req.checkpoint, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"loading {req.checkpoint or 'the stock weights'} failed: {exc}",
                "model_state": _mgr.state.value,
                "weights": _mgr.weights,
            },
        )
    if not loaded:
        return JSONResponse(
            status_code=409,
            content={"status": "busy", "active": _mgr.active, "model_state": _mgr.state.value},
        )
    logger.info("xtts_checkpoint_loaded", checkpoint=req.checkpoint)
    return {"status": "loaded", "model_state": _mgr.state.value, "weights": _mgr.weights, "gpu": _resident_gpu()}


@app.post("/admin/wake", dependencies=[Depends(_require_admin)])
async def admin_wake():
    """Load the model and promote to GPU."""
    await _mgr.ensure_gpu()
    return {"status": "ready", "model_state": _mgr.state.value, "gpu": _resident_gpu()}
