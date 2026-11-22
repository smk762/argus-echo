"""
Qwen3-TTS server — FastAPI, port 8031.

argus-echo's Qwen3-TTS engine. Speaks the same HTTP contract as the Miso and
XTTS engines, and adds voice design: built-in speakers and instruction-based
delivery (`instruction`, `voice_description`) from the CustomVoice model.

Loads two model variants:
  - Base: voice cloning from reference audio
  - CustomVoice: predefined speakers + instruction-based emotion

GPU memory lifecycle (three-state, like ModernBERT):
  unloaded → gpu → cpu

  The first request loads with device_map set to TTS_DEVICE. Loading onto
  CPU and then calling Module.to("cuda") leaves Accelerate dispatch hooks
  in place: weights report cuda but generation runs tens of times slower
  than real time. The CPU copy is only produced later, by .to("cpu") of a
  model that was loaded on CUDA, which stays on the fast path when promoted
  again. After TTS_KEEP_ALIVE_GPU seconds of idle the models demote to CPU,
  freeing VRAM for other services. After a further TTS_KEEP_ALIVE_CPU
  seconds on CPU they are fully unloaded.

Env vars:
  TTS_MODEL_SIZE       — "0.6b" (default) or "1.7b"
  TTS_DEVICE           — "cuda:0" (default) or "cpu"
  TTS_LOAD_MODELS      — "both" (default), "custom_voice", or "base"
  TTS_PRELOAD          — "0" (default) lazy-load on first request;
                          "1" to load into GPU at startup
  TTS_KEEP_ALIVE_GPU   — seconds on GPU after last request before demoting
                          to CPU.  -1 = forever, 0 = immediate.  Default 120.
  TTS_KEEP_ALIVE_CPU   — seconds on CPU after GPU eviction before full
                          unload.  -1 = forever, 0 = immediate.  Default 600.
  TTS_CHECKPOINT_ROOT  — directory of fine-tuned checkpoints POST /admin/load may
                          load (unset: the route answers 503). See docs/qwen.md.
  ECHO_VERSION / ECHO_REVISION — the image's release and git revision (build
                          args), reported in /health as ``image``
  S3_ENDPOINT_URL      — MinIO/S3 endpoint
  S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY / S3_BUCKET_NAME / S3_PUBLIC_URL
"""

import asyncio
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
import weakref
from collections import OrderedDict
import contextlib
import hashlib
import random
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional
from urllib.parse import urlsplit

import structlog
from argus_cortex.auth import bearer
from argus_cortex.fs import PathRefused, safe_join
from argus_cortex.limits import BadInput, TooLarge, read_upload
from starlette.requests import ClientDisconnect, HTTPConnection
from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────


def _normalise_model_size(raw_size: str) -> str:
    normalised = raw_size.strip().lower()
    if normalised not in {"1.7b", "0.6b"}:
        raise ValueError("TTS_MODEL_SIZE must be one of: 1.7b, 1.7B, 0.6b, 0.6B")
    return normalised


MODEL_SIZE = _normalise_model_size(os.getenv("TTS_MODEL_SIZE", "0.6b"))
DEVICE = os.getenv("TTS_DEVICE", "cuda:0")
LOAD_MODELS = os.getenv("TTS_LOAD_MODELS", "both")
# Used when a request names no speaker, clone, or voice description.
# A request `speaker` always wins.
DEFAULT_SPEAKER = os.getenv("TTS_DEFAULT_SPEAKER", "").strip()
PRELOAD = os.getenv("TTS_PRELOAD", "0") == "1"
KEEP_ALIVE_GPU: float = float(os.getenv("TTS_KEEP_ALIVE_GPU", "120"))
KEEP_ALIVE_CPU: float = float(os.getenv("TTS_KEEP_ALIVE_CPU", "600"))
# Fine-tuned checkpoints POST /admin/load may load, one directory each (see _resolve_checkpoint).
CHECKPOINT_ROOT: str = os.getenv("TTS_CHECKPOINT_ROOT", "").strip()
# The image's identity, baked in at build time (release.yml); None for a local build.
ECHO_VERSION: Optional[str] = os.getenv("ECHO_VERSION", "").strip() or None
ECHO_REVISION: Optional[str] = os.getenv("ECHO_REVISION", "").strip() or None
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
# No default credentials: a deployment that forgot them must not run on MinIO's
# well-known root account. See _require_s3_credentials().
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "").strip()
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "voice")
S3_PUBLIC_URL = os.getenv("S3_PUBLIC_URL", "http://localhost:9000/voice")

# Cap autoregressive codec-token generation. The talker default
# (max_new_tokens=4096 ≈ 340s @12Hz) with do_sample=True can occasionally fail
# to emit EOS and run away, producing minute-long synthesis for a short clip.
# 0 = auto-size from text length; >0 = fixed cap.
TTS_MAX_NEW_TOKENS = int(os.getenv("TTS_MAX_NEW_TOKENS", "0"))
TTS_MAX_NEW_TOKENS_CEILING = int(os.getenv("TTS_MAX_NEW_TOKENS_CEILING", "1024"))

# Upload bounds for /clone-voice and /analyze-clips (413 past either), read
# through argus-cortex's read_upload so an oversized clip costs no more than the cap.
MAX_CLIPS: int = int(os.getenv("TTS_MAX_CLIPS", "16"))
CLIP_MAX_BYTES: int = int(os.getenv("TTS_CLIP_MAX_BYTES", str(16 * 1024 * 1024)))
# /analyze-clips score (0-100) below which a clip is not recommended.
ANALYZE_SCORE_THRESHOLD: float = float(os.getenv("TTS_ANALYZE_THRESHOLD", "60"))

MODEL_IDS = {
    "1.7b": {
        "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    },
    "0.6b": {
        "base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "custom_voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    },
}

EMOTION_TO_INSTRUCTION: dict[str, str] = {
    "happy": "speak happily and enthusiastically",
    "sad": "speak softly with a melancholic tone",
    "excited": "speak with high energy and excitement",
    "flirty": "speak warmly and playfully",
    "calm": "speak calmly and soothingly",
    "neutral": "",
}

ISO_TO_LANGUAGE: dict[str, str] = {
    "auto": "auto",
    "zh": "chinese", "en": "english", "fr": "french", "de": "german",
    "it": "italian", "ja": "japanese", "ko": "korean", "pt": "portuguese",
    "ru": "russian", "es": "spanish",
    "chinese": "chinese", "english": "english", "french": "french",
    "german": "german", "italian": "italian", "japanese": "japanese",
    "korean": "korean", "portuguese": "portuguese", "russian": "russian",
    "spanish": "spanish",
}

# Reverse mapping: full language name → ISO 639-1 code
_LANGUAGE_TO_ISO: dict[str, str] = {
    "auto": "auto", "chinese": "zh", "english": "en", "french": "fr",
    "german": "de", "italian": "it", "japanese": "ja", "korean": "ko",
    "portuguese": "pt", "russian": "ru", "spanish": "es",
}

_CLONE_CACHE_MAX = 50

# Names this engine in /health and in every clone's meta.json.
_ENGINE = "qwen-tts"

# A clone's id is its S3 prefix (voice_clones/<id>/): one plain path segment,
# the same rule as the Miso and XTTS engines.
_COMPANION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_COMPANION_ID_ERROR = "companion_id must be 1-64 characters of A-Z, a-z, 0-9, '_' or '-'"

# A Qwen prompt.pt is tens of KB (a speaker embedding plus, in ICL mode,
# the reference's codec tokens); these caps leave ample room.
_MAX_PROMPT_BYTES = 16 * 1024 * 1024
_MAX_META_BYTES = 64 * 1024
_MAX_SPK_EMBEDDING = 8192
_MAX_REF_CODE_NUMEL = 4_000_000
_MAX_REF_TEXT_CHARS = 10_000

# ── GPU memory lifecycle ─────────────────────────────────────────────────────


class TTSState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    CPU = "cpu"
    GPU = "gpu"
    ERROR = "error"


def _release_cuda_cache() -> None:
    """Return the allocator's free blocks to the driver once the models are off the GPU (#22).

    cuBLAS keeps a workspace per stream inside the caching allocator, and while it lives ``empty_cache()`` cannot
    release the segment it sits in (a few hundred MB). Clearing the workspaces first lets the whole cache go; cuBLAS
    allocates a new one on the next matmul. What stays is the CUDA context itself, which only ending the process frees.
    """
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return
    gc.collect()
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is not None:
        clear()
    torch.cuda.empty_cache()


def _move_speech_tokenizer(model, target) -> None:
    """Move the codec qwen-tts attaches as ``model.speech_tokenizer`` (#22).

    It is a ``Qwen3TTSTokenizer``, a plain object wrapping its own nn.Module, loaded with the same device_map, so
    ``Module.to()`` on the TTS model never reaches it: a demote to the CPU left one codec per loaded model on the GPU.
    Its cached ``.device`` is synced too, as the tokenizer places its inputs there.
    """
    tokenizer = getattr(model, "speech_tokenizer", None)
    module = getattr(tokenizer, "model", None)
    if module is None or not callable(getattr(module, "to", None)):
        return
    module.to(target)
    tokenizer.device = getattr(module, "device", target)


# ── Weights: checkpoints and their hashes ────────────────────────────────────

# What a checkpoint directory must hold to load with Qwen3TTSModel.from_pretrained: the model config and weights,
# generation_config.json and speech_tokenizer/ (its from_pretrained reads both), and the text tokenizer the
# processor loads. The Qwen3-TTS fine-tuning scripts write such a directory: a copy of the model they started from
# with the trained weights and config.json written over it.
CHECKPOINT_REQUIRED = ("config.json", "generation_config.json", "tokenizer_config.json", "speech_tokenizer/config.json")
CHECKPOINT_WEIGHTS = ("model.safetensors", "model.safetensors.index.json")
# The model slots a checkpoint can fill, by its config.json ``tts_model_type``.
CHECKPOINT_SLOTS = ("base", "custom_voice")


class CheckpointRefused(Exception):
    """``POST /admin/load`` can't use the named checkpoint; ``status`` is the HTTP answer."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _model_files(directory: Path) -> dict[str, Path]:
    """The files a load reads from a model directory: its top level and ``speech_tokenizer/`` (dotfiles skipped),
    by path relative to it."""
    files: dict[str, Path] = {}
    for sub in ("", "speech_tokenizer"):
        folder = directory / sub if sub else directory
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.name.startswith(".") or not path.is_file():
                continue
            files[f"{sub}/{path.name}" if sub else path.name] = path
    return files


def _resolve_checkpoint(name: str) -> tuple[Path, str]:
    """``(directory, slot)`` of checkpoint *name* under ``TTS_CHECKPOINT_ROOT``, checked before anything unloads.

    503 when no root is configured, 400 for a name that escapes the root (argus-cortex ``safe_join``: ``..``, an
    absolute path, a symlink out), 404 when there is no such directory, 422 when it isn't a Qwen3-TTS model
    directory of a type this engine serves (``base`` or ``custom_voice``)."""
    if not CHECKPOINT_ROOT:
        raise CheckpointRefused(503, "TTS_CHECKPOINT_ROOT is not set: this engine loads no checkpoints")
    root = Path(CHECKPOINT_ROOT)
    if not root.is_dir():
        raise CheckpointRefused(503, f"TTS_CHECKPOINT_ROOT ({CHECKPOINT_ROOT}) is not a directory")
    try:
        path = safe_join(root, name, label="checkpoint")
    except PathRefused as exc:
        raise CheckpointRefused(400, str(exc)) from None
    if not path.exists():
        raise CheckpointRefused(404, f"no checkpoint {name!r} under TTS_CHECKPOINT_ROOT")
    if not path.is_dir():
        raise CheckpointRefused(422, f"checkpoint {name!r} is not a directory")
    missing = [f for f in CHECKPOINT_REQUIRED if not (path / f).is_file()]
    if not any((path / f).is_file() for f in CHECKPOINT_WEIGHTS):
        missing.append(" or ".join(CHECKPOINT_WEIGHTS))
    if missing:
        raise CheckpointRefused(
            422, f"checkpoint {name!r} is not a Qwen3-TTS model directory: missing {', '.join(missing)}"
        )
    base = root.resolve()
    for rel, file in _model_files(path).items():
        if not file.resolve().is_relative_to(base):
            raise CheckpointRefused(400, f"checkpoint {name!r}: {rel} links outside TTS_CHECKPOINT_ROOT")
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointRefused(422, f"checkpoint {name!r}: config.json is not JSON ({exc})") from None
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_tts":
        raise CheckpointRefused(422, f"checkpoint {name!r}: config.json is not a Qwen3-TTS config")
    slot = config.get("tts_model_type")
    if slot not in CHECKPOINT_SLOTS:
        raise CheckpointRefused(
            422, f"checkpoint {name!r}: tts_model_type {slot!r} is not one this engine serves (base, custom_voice)"
        )
    return path, slot


def _stock_dir(model_id: str) -> Path:
    """The local snapshot of a stock model (already downloaded by its load)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, local_files_only=True))


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


# ── Seeded generation ────────────────────────────────────────────────────────

SEED_MAX = 2**32 - 1  # numpy's legacy seed range, and every engine's contract bound


class _RngGate:
    """Who may draw from torch's process-wide RNG while a generation samples.

    Unseeded generations share it (they run side by side, as before seeds existed). A seeded one runs alone: it
    waits for those in flight, new ones wait for it, and it seeds at the start, so nothing else draws between its
    seed and its last sample. Blocking (threading): enter it from the generation's worker thread, never the event
    loop. Waiting seeded generations go first, so a stream of unseeded ones can't starve them."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._shared = 0
        self._exclusive = False
        self._waiting = 0

    @contextlib.contextmanager
    def shared(self):
        with self._cond:
            self._cond.wait_for(lambda: not self._exclusive and not self._waiting)
            self._shared += 1
        try:
            yield
        finally:
            with self._cond:
                self._shared -= 1
                self._cond.notify_all()

    @contextlib.contextmanager
    def exclusive(self):
        with self._cond:
            self._waiting += 1
            try:
                self._cond.wait_for(lambda: not self._exclusive and not self._shared)
            finally:
                self._waiting -= 1
            self._exclusive = True
        try:
            yield
        finally:
            with self._cond:
                self._exclusive = False
                self._cond.notify_all()


_rng_gate = _RngGate()


def _seed_rngs(seed: int) -> None:
    """Seed every RNG generation can draw from: python's, numpy's and torch's (``manual_seed`` seeds the CPU
    generator and every CUDA device's)."""
    random.seed(seed)
    with contextlib.suppress(ImportError):
        import numpy as np

        np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


@contextlib.contextmanager
def _sampling(seed: Optional[int]):
    """Around one generation (in its worker thread): seeded and alone, or unseeded and shared. See _RngGate."""
    if seed is None:
        with _rng_gate.shared():
            yield
        return
    with _rng_gate.exclusive():
        _seed_rngs(seed)
        yield


def _seeded(render, seed: Optional[int]):
    """``render(text)`` as ``run(text, index=0)``: each call sampled under the gate, a seeded one from
    ``seed + index``. ``/synthesise`` renders once (index 0); ``/synthesise/stream`` renders sentence *index* with
    its own seed, so it reseeds before every sentence and needn't keep the RNG between them."""

    def run(text: str, index: int = 0):
        with _sampling(None if seed is None else (seed + index) % (SEED_MAX + 1)):
            return render(text)

    return run


class TTSModelManager:
    """Three-state lifecycle: unloaded → gpu → cpu.

    The first load uses device_map=<TTS_DEVICE>. A later idle demotes with
    .to("cpu"); the next request moves that same module back. Do not reload
    with device_map="cpu" and then .to("cuda") — Accelerate hooks survive
    that move and generation becomes ~50× real time.
    After keep_alive_gpu seconds of idle the models demote to CPU.
    After keep_alive_cpu seconds on CPU the models fully unload. That CPU stage follows
    any demotion, idle or /admin/evict: keep_alive_gpu=-1 only stops the idle demotion.

    Generation must hold a lease (``async with _mgr.use(): ...``) for as long as
    it touches the models. Demotion and unload wait until no lease is held, and
    the idle timer only starts when the last lease is released.
    """

    def __init__(self) -> None:
        self.keep_alive_gpu = KEEP_ALIVE_GPU
        self.keep_alive_cpu = KEEP_ALIVE_CPU
        self._state = TTSState.UNLOADED
        self._error: Optional[str] = None
        self._base_model = None
        self._custom_model = None
        self._last_used: float = 0.0
        self._lock = asyncio.Lock()
        self._evict_task: Optional[asyncio.Task] = None
        # The first load runs as its own task so a cancelled caller (client gone,
        # timeout) can't strand the state in LOADING: the task sets GPU/ERROR itself.
        self._load_task: Optional[asyncio.Task] = None
        # Likewise a GPU<->CPU move: its task sets the new state once the weights
        # have moved, so a cancelled caller can't leave _state naming the old device.
        self._move_task: Optional[asyncio.Task] = None
        self._in_use = 0
        self._idle = asyncio.Event()
        self._idle.set()
        # The checkpoint /admin/load selected, as (name, directory, slot), or None for the stock models. It replaces
        # the model of its slot (base or custom_voice); the other slot loads stock per TTS_LOAD_MODELS. Only
        # /admin/load and /admin/unload change it; an idle unload keeps it, so the next load reads the same weights.
        self._checkpoint: Optional[tuple[str, Path, str]] = None
        # [{name, sha256, bytes}] of the files the last load of the selected weights read; None until one has.
        self._weight_files: Optional[list[dict]] = None

    @property
    def state(self) -> TTSState:
        return self._state

    @property
    def weights(self) -> dict:
        """``/health``'s ``weights``: the selected checkpoint (None = stock) and the files its last load hashed."""
        return {
            "checkpoint": self._checkpoint[0] if self._checkpoint else None,
            "files": self._weight_files,
        }

    @property
    def base_model(self):
        return self._base_model

    @property
    def custom_model(self):
        return self._custom_model

    @property
    def is_ready(self) -> bool:
        return self._state in (TTSState.CPU, TTSState.GPU)

    def _load(self, device: str) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        dtype = torch.bfloat16
        ids = MODEL_IDS[MODEL_SIZE]
        # Single-device map on the real target. See the class docstring.
        device_map = device if str(device).startswith("cuda") else "cpu"

        def _load_one(model_id: str):
            # Prefer torch's built-in SDPA attention over the slow "manual PyTorch"
            # path used when flash-attn is absent. Fall back if the model rejects
            # the kwarg (custom model may not honor HF's attn dispatch).
            try:
                return Qwen3TTSModel.from_pretrained(
                    model_id, device_map=device_map, dtype=dtype,
                    attn_implementation="sdpa",
                )
            except (TypeError, ValueError) as exc:
                logger.warning("tts_sdpa_unsupported", error=str(exc))
                return Qwen3TTSModel.from_pretrained(
                    model_id, device_map=device_map, dtype=dtype,
                )

        # Each slot loads the selected checkpoint if it is that slot's, else the stock model if TTS_LOAD_MODELS
        # asks for it.
        checkpoint = self._checkpoint
        self._weight_files = None  # until this load hashes what it read
        sources: dict[str, tuple[str, Optional[Path]]] = {}  # slot -> (what to load, its directory if local)
        for slot in CHECKPOINT_SLOTS:
            if checkpoint is not None and checkpoint[2] == slot:
                sources[slot] = (os.fspath(checkpoint[1]), checkpoint[1])
            elif LOAD_MODELS in ("both", slot):
                sources[slot] = (ids[slot], None)

        if "base" in sources:
            logger.info("loading_qwen3_tts_base", model=sources["base"][0], target=device_map)
            self._base_model = _load_one(sources["base"][0])
            logger.info("qwen3_tts_base_loaded", device=device_map)

        if "custom_voice" in sources:
            logger.info("loading_qwen3_tts_custom_voice", model=sources["custom_voice"][0], target=device_map)
            self._custom_model = _load_one(sources["custom_voice"][0])
            logger.info("qwen3_tts_custom_voice_loaded", device=device_map)

        # The files those loads read, hashed here (the load's worker thread) once per load, never per request. A
        # hash that fails fails the load (_run_load frees the models), so no model runs unreported.
        read: dict[str, Path] = {}
        for slot, (source, directory) in sources.items():
            for rel, path in _model_files(directory or _stock_dir(source)).items():
                read[f"{slot}/{rel}"] = path
        self._weight_files = _weight_files(read)

    def _move_to_device(self, device: str) -> None:
        import torch

        target = torch.device(device)
        # Qwen3TTSModel is a plain wrapper with no .to(); the real nn.Module lives
        # on `.model`. Move that, then sync the wrapper's cached `.device` so
        # generate() places inputs on the same device as the weights.
        for wrapper in (self._base_model, self._custom_model):
            if wrapper is None:
                continue
            inner = getattr(wrapper, "model", None)
            if inner is not None and hasattr(inner, "to"):
                inner.to(target)
                wrapper.device = getattr(inner, "device", target)
            elif hasattr(wrapper, "to") and callable(wrapper.to):
                wrapper.to(target)
            _move_speech_tokenizer(inner if inner is not None else wrapper, target)

        if device == "cpu":
            _release_cuda_cache()

    async def _run_load(self) -> None:
        try:
            await asyncio.to_thread(self._load, DEVICE)
        except Exception as exc:
            # _load assigns the base model before it loads custom_voice. Free what
            # it got as far as, or that copy stays on the GPU with no idle timer
            # and the retry allocates a second one beside it.
            self._free_models()
            self._state = TTSState.ERROR
            self._error = str(exc)
            logger.error("tts_load_failed", error=str(exc))
            raise
        self._state = TTSState.GPU
        self._error = None
        if self._in_use == 0 and not self._lock.locked():
            # Nobody is waiting on this load (its caller was cancelled): start
            # the idle timer so the models don't sit on the GPU forever.
            self._last_used = time.time()
            self._arm_evict()

    async def _await_load(self) -> None:
        """Wait for an in-flight load. Shielded: cancelling the waiter leaves the
        load running, and the load task still sets the final state."""
        task = self._load_task
        if task is not None and not task.done():
            await asyncio.shield(task)

    async def _move(self, device: str, state: TTSState) -> None:
        """Move the weights to ``device`` and only then set ``state``, in a task
        the caller awaits through asyncio.shield: cancelling the caller leaves
        the move (and its state update) to finish, so ``_state`` always names
        where the weights are."""

        async def run() -> None:
            try:
                await asyncio.to_thread(self._move_to_device, device)
            except Exception as exc:
                # Part of the weights may have moved: neither device's state is
                # true any more, and no idle timer is armed. Drop the models so
                # nothing stays resident; the next request loads them afresh.
                self._free_models()
                self._state = TTSState.ERROR
                self._error = str(exc)
                logger.error("tts_move_failed", target=device, error=str(exc))
                raise
            self._state = state
            if self._in_use == 0 and not self._lock.locked():
                # Its caller was cancelled and nobody else holds the lock: arm
                # the idle timer it would have armed, so the next stage still runs.
                self._last_used = time.time()
                self._arm_evict()

        self._move_task = asyncio.create_task(run())
        self._move_task.add_done_callback(lambda t: t.cancelled() or t.exception())
        await asyncio.shield(self._move_task)

    async def _await_move(self) -> None:
        """Let an orphaned move finish (and set its state) before reading the state."""
        task = self._move_task
        if task is not None and not task.done():
            with contextlib.suppress(Exception):  # only waiting for the orphaned task; its own handler records any failure
                await asyncio.shield(task)

    async def _make_ready(self) -> None:
        # Caller holds self._lock.
        await self._await_move()
        if self._state == TTSState.LOADING:
            await self._await_load()  # raises the load's error, if it failed

        if self._state == TTSState.GPU:
            return

        if self._state in (TTSState.UNLOADED, TTSState.ERROR):
            self._state = TTSState.LOADING
            self._load_task = asyncio.create_task(self._run_load())
            # Retrieve the exception even when no caller is left to await it.
            self._load_task.add_done_callback(lambda t: t.cancelled() or t.exception())
            await self._await_load()
            return

        if self._state == TTSState.CPU:
            if DEVICE.startswith("cuda"):
                logger.info("tts_promoting_to_gpu", device=DEVICE)
                await self._move(DEVICE, TTSState.GPU)
                logger.info("tts_on_gpu")
            else:
                self._state = TTSState.GPU  # CPU-only: treat as "ready"

    async def ensure_gpu(self) -> None:
        """Guarantee models are on GPU and reset the idle timer.

        Does not protect a following generation; use ``use()`` for that.
        """
        async with self._lock:
            await self._make_ready()
            self._last_used = time.time()
            if self._in_use == 0:
                self._arm_evict()

    @asynccontextmanager
    async def use(self) -> AsyncIterator["TTSModelManager"]:
        """Put the models on GPU and hold them there for the whole block.

        Don't call ensure_gpu/demote_to_cpu/unload inside the block: they take
        the lock a waiting demote/unload holds, which would deadlock.
        """
        await self.acquire()
        try:
            yield self
        finally:
            self.release()

    async def acquire(self) -> None:
        """Take a lease by hand (``use()`` is the usual way). Every successful
        ``acquire()`` needs exactly one ``release()``, on the event loop thread.
        ``/synthesise/stream`` uses the pair to hand its lease to the response."""
        async with self._lock:
            await self._make_ready()
            self._in_use += 1
            self._idle.clear()
            if self._evict_task and not self._evict_task.done():
                self._evict_task.cancel()

    def release(self) -> None:
        self._in_use -= 1
        self._last_used = time.time()
        if self._in_use == 0:
            self._idle.set()
            self._arm_evict()

    async def _demote_locked(self) -> None:
        await self._await_move()
        if self._state != TTSState.GPU:
            return
        if DEVICE == "cpu":
            self._state = TTSState.CPU  # nothing to move; lets the CPU stage run
            return
        logger.info("tts_demoting_to_cpu")
        await self._move("cpu", TTSState.CPU)
        logger.info("tts_on_cpu")

    async def demote_to_cpu(self) -> None:
        """Move GPU → CPU (waits for in-flight generations), then start the CPU-stage
        timer: ``keep_alive_cpu`` decides when the models leave RAM, whatever
        ``keep_alive_gpu`` is (#52)."""
        async with self._lock:
            await self._idle.wait()
            await self._await_load_quietly()
            if self._state != TTSState.GPU:
                return
            await self._demote_locked()
            if self._state == TTSState.CPU:
                self._arm_evict()

    def _free_models(self) -> None:
        """Drop both models and return their memory. Does not set the state."""
        for attr in ("_base_model", "_custom_model"):
            model = getattr(self, attr)
            if model is not None:
                del model
                setattr(self, attr, None)
        _release_cuda_cache()

    def _unload_locked(self) -> None:
        self._free_models()
        self._state = TTSState.UNLOADED
        logger.info("tts_fully_unloaded")

    async def unload(self, to_stock: bool = False) -> None:
        """Full unload (waits for in-flight generations). ``to_stock``
        (``/admin/unload``) also drops a loaded checkpoint, so the next load
        reads the stock models."""
        async with self._lock:
            await self._idle.wait()
            await self._await_load_quietly()
            if (self._evict_task and not self._evict_task.done()
                    and self._evict_task is not asyncio.current_task()):
                self._evict_task.cancel()
            self._unload_locked()
            if to_stock and self._checkpoint is not None:
                logger.info("tts_checkpoint_dropped", checkpoint=self._checkpoint[0])
                self._checkpoint = None
                self._weight_files = None

    async def load_weights(self, checkpoint: Optional[tuple[str, Path, str]]) -> bool:
        """``/admin/load``: replace the loaded models by *checkpoint* (``(name, directory, slot)``; None = stock),
        loaded onto ``TTS_DEVICE`` now and hashed. False while a lease is held (busy): nothing changes.

        It waits for a load or move in progress, frees the old models, then loads the new ones as a shielded task
        (a cancelled caller can't strand it half-way). A load that fails frees whatever it loaded and leaves the
        stock models selected (the state says ``error``; the next request loads stock), and raises."""
        async with self._lock:
            if self._in_use:
                logger.info("tts_load_refused_busy", in_use=self._in_use)
                return False
            await self._await_load_quietly()
            if self._evict_task and not self._evict_task.done():
                self._evict_task.cancel()
            self._unload_locked()
            self._checkpoint = checkpoint
            self._weight_files = None
            self._state = TTSState.LOADING

            async def load() -> None:
                try:
                    await self._run_load()
                except BaseException:
                    self._checkpoint = None
                    self._weight_files = None
                    raise

            self._load_task = asyncio.create_task(load())
            self._load_task.add_done_callback(lambda t: t.cancelled() or t.exception())
            await self._await_load()
            self._last_used = time.time()
            if self._in_use == 0:
                self._arm_evict()
            return True

    async def _await_load_quietly(self) -> None:
        """Let an in-flight load finish before demoting/unloading, so the worker
        thread can't assign models after they were freed. Its failure is not ours.
        An orphaned GPU<->CPU move is waited for too."""
        with contextlib.suppress(Exception):  # only waiting for the orphaned load; its own handler records any failure
            await self._await_load()
        await self._await_move()

    def _arm_evict(self) -> None:
        if self._evict_task and not self._evict_task.done():
            self._evict_task.cancel()
        self._evict_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        # Armed when the models go idle; any new lease cancels it. A stage that
        # finds a lease held just stops: the lease's release re-arms the timer.
        try:
            # GPU keep-alive. Negative pins the models on GPU; it skips only this
            # step, so models evicted to the CPU still get the CPU stage (#52).
            if self._state == TTSState.GPU:
                if self.keep_alive_gpu < 0:
                    return
                if self.keep_alive_gpu > 0:
                    await asyncio.sleep(self.keep_alive_gpu)
                async with self._lock:
                    if self._state != TTSState.GPU or self._in_use:
                        return
                    await self._demote_locked()
            elif self._state != TTSState.CPU:
                return

            # CPU keep-alive (only once the models are actually on CPU)
            if self.keep_alive_cpu < 0:
                return
            if self.keep_alive_cpu > 0:
                await asyncio.sleep(self.keep_alive_cpu)
            async with self._lock:
                await self._await_move()
                if self._state != TTSState.CPU or self._in_use:
                    return
                self._unload_locked()
        except asyncio.CancelledError:
            pass
        except Exception:
            # Nothing awaits this task. A failed move has already freed the models
            # and set ERROR (see _move); log instead of dying unretrieved.
            logger.error("tts_eviction_failed", state=self._state.value, exc_info=True)


# ── Global state ─────────────────────────────────────────────────────────────

_mgr = TTSModelManager()
_s3_client = None
_clone_cache: OrderedDict = OrderedDict()
_clone_cache_lock = threading.Lock()


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


def _prompt_item_cls():
    """qwen_tts's ``VoiceClonePromptItem``: what ``create_voice_clone_prompt``
    returns (as a one-item list) and what a stored ``prompt.pt`` pickles."""
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    return VoiceClonePromptItem


def _load_prompt(data) -> list:
    """Deserialize and validate a stored Qwen clone (``prompt.pt``).

    Clone blobs come from S3 or an uploaded archive, so they're untrusted.
    ``weights_only=True`` restricts unpickling to tensors, plain containers and
    the one class allowed here, ``VoiceClonePromptItem`` (a dataclass of tensors,
    bools and a string), so nothing can run code. Without that allowance a
    ``prompt.pt`` written by ``/clone-voice`` does not load at all under
    ``weights_only``. Returns ``[VoiceClonePromptItem]`` rebuilt from the
    validated fields; raises ``ValueError``."""
    import torch

    cls = _prompt_item_cls()
    buf = data if hasattr(data, "read") else io.BytesIO(data)
    try:
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is not None:
            with safe_globals([cls]):
                loaded = torch.load(buf, map_location="cpu", weights_only=True)
        else:  # torch < 2.5
            torch.serialization.add_safe_globals([cls])
            loaded = torch.load(buf, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("prompt.pt is not a Qwen clone (it does not load as one)") from exc
    return _validate_prompt(loaded, cls)


def _validate_prompt(loaded, cls) -> list:
    """``loaded`` must be one ``VoiceClonePromptItem`` in a list, as
    ``create_voice_clone_prompt`` makes for one reference clip. Another engine's
    ``prompt.pt`` (a dict of tensors) fails here."""
    import torch

    if not (isinstance(loaded, list) and len(loaded) == 1 and isinstance(loaded[0], cls)):
        raise ValueError("prompt.pt must hold one Qwen VoiceClonePromptItem (is it another engine's clone?)")
    item = loaded[0]

    spk = item.ref_spk_embedding
    if not (torch.is_tensor(spk) and spk.is_floating_point() and spk.dim() == 1
            and 0 < spk.numel() <= _MAX_SPK_EMBEDDING):
        raise ValueError("prompt.pt's ref_spk_embedding must be a 1-D float tensor")
    xvec, icl = item.x_vector_only_mode, item.icl_mode
    if not (isinstance(xvec, bool) and isinstance(icl, bool) and xvec != icl):
        raise ValueError("prompt.pt must set exactly one of x_vector_only_mode and icl_mode")
    text = item.ref_text
    if text is not None and not (isinstance(text, str) and len(text) <= _MAX_REF_TEXT_CHARS):
        raise ValueError("prompt.pt's ref_text must be a string")

    code = None  # x-vector mode uses only the embedding, as create_voice_clone_prompt stores it
    if icl:
        code = item.ref_code
        if not (torch.is_tensor(code) and not code.is_floating_point() and not code.is_complex()
                and code.dtype != torch.bool and code.dim() in (1, 2) and 0 < code.numel() <= _MAX_REF_CODE_NUMEL):
            raise ValueError("an ICL prompt.pt needs its ref_code as an integer tensor")
        if not text:
            raise ValueError("an ICL prompt.pt needs its ref_text")
        code = code.detach().cpu()
    return [cls(ref_code=code, ref_spk_embedding=spk.detach().cpu(), x_vector_only_mode=xvec, icl_mode=icl,
                ref_text=text)]


def _get_clone_prompt(clone_id: str):
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
        logger.error("clone_prompt_s3_error", clone_id=clone_id, error_code=error_code)
        raise ValueError(f"Voice clone prompt not found: {clone_id}") from exc
    buf.seek(0)
    try:
        prompt = _load_prompt(buf)
    except ValueError as exc:
        logger.error("clone_prompt_invalid", clone_id=clone_id, error=str(exc))
        raise ValueError(f"Voice clone is not a valid Qwen clone: {clone_id}") from exc

    with _clone_cache_lock:
        _clone_cache[clone_id] = prompt
        if len(_clone_cache) > _CLONE_CACHE_MAX:
            _clone_cache.popitem(last=False)

    return prompt


def _gpu_memory_info() -> dict | None:
    try:
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
    except Exception as exc:
        logger.warning("gpu_memory_info_failed", error=str(exc))
    return None


def _get_supported_languages() -> list[str]:
    """Return ISO codes for languages the model supports."""
    return sorted(set(_LANGUAGE_TO_ISO.values()) - {"auto"})


# ── App lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    _require_s3_credentials()  # fail at startup, not on the first upload
    if PRELOAD:
        await _mgr.ensure_gpu()
    yield
    await _mgr.unload()


app = FastAPI(title="Voice Stack — Qwen3-TTS", version="1.1.0", lifespan=lifespan)


# ── Request / Response models ─────────────────────────────────────────────────


class SynthesiseRequest(BaseModel):
    text: str
    voice_clone_id: Optional[str] = None
    speaker: Optional[str] = None
    voice_description: Optional[str] = None
    instruction: Optional[str] = None
    emotion: str = "neutral"
    language: str = Field(default="en")
    speed: Optional[float] = None  # accepted for the shared contract; Qwen ignores it (XTTS applies it)
    device: Optional[str] = None  # XTTS-only ("cpu": no GPU promotion); Qwen ignores it
    # Seeds the sampler (python, numpy, torch CPU and CUDA). /synthesise seeds once; /synthesise/stream seeds
    # sentence i (from 0) with seed + i. See docs/api.md for what that does and doesn't make reproducible.
    seed: Optional[int] = Field(default=None, ge=0, le=SEED_MAX)


# ── POST /synthesise ──────────────────────────────────────────────────────────


def _max_new_tokens_for(text: str) -> int:
    """Bound the talker's codec-token budget so a runaway sample can't generate
    minutes of audio for a short clip. The talker runs at ~12 Hz; English needs
    well under one frame per character, so 4x + a floor is generous headroom."""
    if TTS_MAX_NEW_TOKENS > 0:
        return TTS_MAX_NEW_TOKENS
    est = int(len(text or "") * 4) + 64
    return max(192, min(est, TTS_MAX_NEW_TOKENS_CEILING))


async def _generate_audio(req: SynthesiseRequest):
    """Render speech. Returns ``(float32 samples, sample_rate)`` or a JSONResponse error.

    Uses Base model if voice_clone_id is provided (voice cloning path).
    Uses CustomVoice model otherwise (predefined speaker + instruction path).
    """
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})

    async with _mgr.use():
        return await _generate_audio_leased(req)


async def _generate_audio_leased(req: SynthesiseRequest):
    """Body of ``_generate_audio``; the caller holds a ``_mgr.use()`` lease."""
    request_id = uuid.uuid4().hex[:12]  # ties a client-visible error to its log line
    render = await _plan_generation(req, request_id)
    if isinstance(render, JSONResponse):
        return render
    return await _render(_seeded(render, req.seed), req.text, request_id)


async def _render(render, text: str, request_id: str):
    """Run ``render(text)`` in a thread: ``(float32 samples, sample_rate)`` or a JSONResponse."""
    try:
        audio_arrays, sample_rate = await asyncio.to_thread(render, text)
    except Exception as exc:
        return _generation_error(exc, request_id)
    return _to_pcm(audio_arrays, sample_rate, request_id, text)


def _generation_error(exc: Exception, request_id: str) -> JSONResponse:
    """The client's answer to a failed generation (or clone-prompt load)."""
    if isinstance(exc, RuntimeError):
        err_str = str(exc).lower()
        if "out of memory" in err_str or "cuda" in err_str:
            gpu_info = _gpu_memory_info()
            logger.error("tts_gpu_oom", request_id=request_id, error=str(exc), gpu=gpu_info)
            return JSONResponse(
                status_code=503, content={"error": "GPU OOM", "request_id": request_id, "gpu": gpu_info}
            )
    elif isinstance(exc, ValueError):
        logger.warning("tts_validation_error", error=str(exc))
        return JSONResponse(status_code=422, content={"error": str(exc)})
    # The exception text can name paths, devices and other internals: it goes
    # to the log, and the client gets the id to find it by.
    logger.error("tts_generation_failed", request_id=request_id, error=str(exc), exc_info=True)
    return JSONResponse(status_code=500, content={"error": "TTS generation failed", "request_id": request_id})


def _to_pcm(audio_arrays, sample_rate, request_id: str, text: str):
    """One contiguous little-endian float32 array from the model's output, or a JSONResponse."""
    if audio_arrays is None or len(audio_arrays) == 0:
        logger.error("tts_empty_output", request_id=request_id, text_chars=len(text or ""))
        return JSONResponse(status_code=500, content={"error": "TTS produced no audio", "request_id": request_id})

    import numpy as np

    audio = np.concatenate(audio_arrays) if len(audio_arrays) > 1 else audio_arrays[0]
    audio = np.ascontiguousarray(np.asarray(audio), dtype="<f4").reshape(-1)
    return audio, int(sample_rate)


async def _plan_generation(req: SynthesiseRequest, request_id: str):
    """Validate ``req`` and choose its model, before any audio is rendered.

    Returns ``render(text) -> (audio_arrays, sample_rate)``, a blocking call that
    speaks ``text`` in the request's voice, or a JSONResponse for a request that
    can't be served. A clone's prompt is loaded here, so a bad ``voice_clone_id``
    is an error before ``/synthesise/stream`` sends its 200. The caller holds a lease.

    Uses the Base model for a ``voice_clone_id`` (voice cloning) and the
    CustomVoice model otherwise (built-in speaker + instruction).
    """
    instruction = req.instruction
    if instruction is None:
        instruction = EMOTION_TO_INSTRUCTION.get(req.emotion, "")

    language = ISO_TO_LANGUAGE.get(req.language.lower(), req.language.lower())
    if language not in ISO_TO_LANGUAGE.values():
        return JSONResponse(
            status_code=422,
            content={
                "error": f"Unsupported language: {req.language!r}. "
                f"Supported: {sorted(set(ISO_TO_LANGUAGE.values()))}"
            },
        )

    if req.voice_clone_id:
        base = _mgr.base_model
        if base is None:
            return JSONResponse(status_code=503, content={"error": "Base model not loaded (TTS_LOAD_MODELS=custom_voice)"})
        try:
            clone_prompt = await asyncio.to_thread(_get_clone_prompt, req.voice_clone_id)
        except Exception as exc:
            return _generation_error(exc, request_id)

        def render(text: str):
            return base.generate_voice_clone(
                text, voice_clone_prompt=clone_prompt, language=language,
                max_new_tokens=_max_new_tokens_for(text),
            )

        return render

    if req.voice_description:
        design_instruct = req.voice_description
        if instruction:
            design_instruct = f"{req.voice_description}. {instruction}"

        base = _mgr.base_model
        if base is not None:
            def render(text: str):
                return base.generate_voice_design(
                    text, instruct=design_instruct, language=language,
                    max_new_tokens=_max_new_tokens_for(text),
                )

            return render

        custom = _mgr.custom_model
        if custom is None:
            return JSONResponse(status_code=503, content={"error": "No model loaded"})
        fallback_speaker = (req.speaker or "").strip() or DEFAULT_SPEAKER
        if not fallback_speaker:
            try:
                available = sorted(custom.get_supported_speakers() or [])
            except Exception as exc:
                return _generation_error(exc, request_id)
            if not available:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Voice design needs the Base model (TTS_LOAD_MODELS=custom_voice) "
                        "and the CustomVoice model lists no speaker to fall back to."
                    },
                )
            fallback_speaker = available[0]
        logger.info("voice_design_fallback_to_speaker", speaker=fallback_speaker)

        def render(text: str):
            return custom.generate_custom_voice(
                text=text, speaker=fallback_speaker,
                instruct=design_instruct, language=language,
                max_new_tokens=_max_new_tokens_for(text),
            )

        return render

    custom = _mgr.custom_model
    if custom is None:
        return JSONResponse(status_code=503, content={"error": "CustomVoice model not loaded (TTS_LOAD_MODELS=base)"})
    speaker = (req.speaker or "").strip() or DEFAULT_SPEAKER
    if not speaker:
        try:
            available = custom.get_supported_speakers()
        except Exception as exc:
            return _generation_error(exc, request_id)
        return JSONResponse(
            status_code=422,
            content={
                "error": "Either 'speaker', 'voice_description', or 'voice_clone_id' is required.",
                "available_speakers": list(available),
            },
        )

    def render(text: str):
        generate_kwargs: dict = {
            "text": text,
            "speaker": speaker,
            "language": language,
            "max_new_tokens": _max_new_tokens_for(text),
        }
        if instruction:
            generate_kwargs["instruct"] = instruction
        return custom.generate_custom_voice(**generate_kwargs)

    return render


def _encode_wav(audio, sample_rate: int) -> tuple[bytes, float]:
    """``(wav_bytes, duration_s)`` for ``audio``. Blocking file I/O: run it in a thread."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        sf.write(tmp_path, audio, sample_rate)
        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
        return wav_bytes, sf.info(tmp_path).duration
    finally:
        os.unlink(tmp_path)


def _write_temp_wav(data: bytes) -> str:
    """Write ``data`` to a new temp ``.wav`` and return its path. Blocking: run it in a thread."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        return f.name


@app.post("/synthesise")
async def synthesise(req: SynthesiseRequest):
    """Generate speech from text and store a WAV. Returns ``{audio_url, duration, ...}``."""
    t_req = time.monotonic()
    rendered = await _generate_audio(req)
    if isinstance(rendered, JSONResponse):
        return rendered
    audio, sample_rate = rendered

    wav_bytes, duration = await asyncio.to_thread(_encode_wav, audio, sample_rate)

    job_id = str(uuid.uuid4())
    s3_key = f"generated/voice/{job_id}.wav"
    audio_url = await asyncio.to_thread(_upload_to_s3, wav_bytes, s3_key)

    logger.info(
        "tts_synthesised",
        job_id=job_id,
        duration=round(duration, 2),
        language=req.language,
        total_s=round(time.monotonic() - t_req, 2),
        rtf=round((time.monotonic() - t_req) / duration, 1) if duration else None,
    )

    return {
        "audio_url": audio_url,
        "duration": duration,
        "duration_seconds": round(duration, 3),
        "format": "wav",
        "seed": req.seed,
    }


# A sentence ends at . ! ? or … (with any closing quotes or brackets) followed by
# whitespace, at CJK full-width end punctuation, or at a blank line.
_SENTENCE_BREAK = re.compile(r"([.!?…]+[\"'”’»)\]]*\s+|[。！？]+[\"”’」』）]*\s*|\n\s*\n)")
_CJK_END = ("。", "！", "？", "」", "』", "）")
# Fragments shorter than this join the next sentence: "Dr." or "Hi!" alone would
# be rendered with odd prosody, and a model call per word costs more than it saves.
_STREAM_MIN_CHARS = 24


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences for ``/synthesise/stream``, merging short ones."""
    parts = _SENTENCE_BREAK.split(text or "")
    pieces = []
    for i in range(0, len(parts), 2):
        piece = (parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")).strip()
        if piece:
            pieces.append(piece)

    def join(a: str, b: str) -> str:
        return a + b if a.endswith(_CJK_END) else f"{a} {b}"

    sentences: list[str] = []
    for piece in pieces:
        if sentences and len(sentences[-1]) < _STREAM_MIN_CHARS:
            sentences[-1] = join(sentences[-1], piece)
        else:
            sentences.append(piece)
    if len(sentences) > 1 and len(sentences[-1]) < _STREAM_MIN_CHARS:
        tail = sentences.pop()
        sentences[-1] = join(sentences[-1], tail)
    return sentences or [text]


class _SentenceStream:
    """The body of one ``/synthesise/stream`` response: the first sentence's PCM
    (already rendered), then each following sentence as soon as it is rendered.

    Owns the request's model lease from the moment the response is built, and
    gives it back exactly once in ``close()``. A sentence still rendering in its
    worker thread keeps the lease until it finishes, so the idle timer can't move
    the weights under it. After ``client_gone()`` or ``close()`` no new sentence starts.
    """

    def __init__(self, render, sentences: list[str], first, sample_rate: int, request_id: str):
        self._render = render
        self._rest = sentences[1:]
        self._first = first
        self.sample_rate = sample_rate
        self._request_id = request_id
        self._inflight: Optional[asyncio.Future] = None
        self._stop = False
        self._closed = False

    def client_gone(self) -> None:
        if not self._stop:
            logger.info("tts_stream_client_gone", request_id=self._request_id)
        self._stop = True

    def close(self) -> None:
        """Give the lease back (once). Called however the response ends."""
        self._stop = True
        if self._closed:
            return
        self._closed = True
        fut = self._inflight
        if fut is not None and not fut.done():

            def _release(f: asyncio.Future) -> None:
                if not f.cancelled():
                    f.exception()  # retrieved: the stream has already ended
                _mgr.release()

            fut.add_done_callback(_release)
        else:
            _mgr.release()

    async def chunks(self):
        try:
            first, self._first = self._first, None
            yield first.tobytes()
            for i, text in enumerate(self._rest, start=2):
                if self._stop:
                    return
                # Sentence i (from 0) is index i - 1: _seeded reseeds it with seed + index.
                self._inflight = asyncio.ensure_future(asyncio.to_thread(self._render, text, i - 1))
                try:
                    # Shielded: a cancelled response leaves the render to finish,
                    # and close() releases the lease when it does.
                    audio_arrays, sample_rate = await asyncio.shield(self._inflight)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # The 200 is already out: all we can do is end the stream early.
                    logger.error(
                        "tts_stream_failed", request_id=self._request_id, sentence=i, error=str(exc), exc_info=True
                    )
                    return
                self._inflight = None
                pcm = _to_pcm(audio_arrays, sample_rate, self._request_id, text)
                if isinstance(pcm, JSONResponse):
                    return
                audio, rate = pcm
                if rate != self.sample_rate:
                    logger.error(
                        "tts_stream_rate_changed", request_id=self._request_id, rate=rate, expected=self.sample_rate
                    )
                    return
                yield audio.tobytes()
        finally:
            self.close()


class _PCMStreamResponse(StreamingResponse):
    """Streams a ``_SentenceStream`` and releases its lease however the response
    ends (finished, failed or cancelled).

    Races the stream against the client's ``http.disconnect`` whatever the ASGI
    spec version (Starlette only listens for it below 2.4; at 2.4 it notices on
    the next send, after one more sentence has been rendered). On disconnect
    the response ends at once and no further sentence starts; one that is
    already rendering finishes in its thread and then gives the lease back."""

    def __init__(self, stream: _SentenceStream, **kwargs):
        super().__init__(stream.chunks(), **kwargs)
        self._stream = stream

    async def __call__(self, scope, receive, send) -> None:
        async def watch() -> None:
            while (await receive()).get("type") != "http.disconnect":
                pass
            self._stream.client_gone()

        streamer = asyncio.ensure_future(self.stream_response(send))
        watcher = asyncio.ensure_future(watch())
        try:
            await asyncio.wait({streamer, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if streamer.done():
                try:
                    streamer.result()
                except OSError as exc:
                    raise ClientDisconnect() from exc
        finally:
            for task in (streamer, watcher):
                task.cancel()
            with contextlib.suppress(BaseException):  # only reaping the cancelled tasks
                await asyncio.gather(streamer, watcher, return_exceptions=True)
            self._stream.close()


@app.post("/synthesise/stream")
async def synthesise_stream(req: SynthesiseRequest):
    """Stream the utterance as raw mono float32 PCM, one sentence at a time.

    The text is split into sentences, and each sentence's PCM is sent as soon as
    it is rendered. The first sentence is rendered before the ``200``, so every
    error up to and including it (validation, loading, out of memory) is a JSON
    error; a later failure can only end the stream early. Argus-mask ``POST
    /speak`` reads ``X-Sample-Rate``, ``X-Audio-Format`` and ``X-Channels`` and
    then the body. One model lease covers the whole stream.
    """
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})

    request_id = uuid.uuid4().hex[:12]
    await _mgr.acquire()
    handed_off = False
    try:
        render = await _plan_generation(req, request_id)
        if isinstance(render, JSONResponse):
            return render
        render = _seeded(render, req.seed)
        sentences = _split_sentences(req.text)
        first = await _render(render, sentences[0], request_id)
        if isinstance(first, JSONResponse):
            return first
        audio, sample_rate = first
        stream = _SentenceStream(render, sentences, audio, sample_rate, request_id)
        response = _PCMStreamResponse(
            stream,
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(sample_rate),
                "X-Audio-Format": "pcm_f32le",
                "X-Channels": "1",
                "Connection": "close",
            },
        )
        handed_off = True
        logger.info("tts_stream_started", request_id=request_id, sentences=len(sentences), language=req.language)
        return response
    finally:
        if not handed_off:
            _mgr.release()


# ── POST /clone-voice ─────────────────────────────────────────────────────────


def _clone_key(companion_id: str) -> str:
    """S3 key of a companion's clone prompt; ``companion_id`` must be validated."""
    return f"voice_clones/{companion_id}/prompt.pt"


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


# One lock per companion_id spans the existence check through the upload, so two
# concurrent replace=false clones of the same id can't both pass the check. The
# sidecar runs a single uvicorn worker, so an in-process lock is enough.
_clone_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()


def _clone_lock(companion_id: str) -> asyncio.Lock:
    lock = _clone_locks.get(companion_id)
    if lock is None:
        lock = asyncio.Lock()
        _clone_locks[companion_id] = lock
    return lock


@app.post("/clone-voice")
async def clone_voice(
    audio: UploadFile = File(...),
    reference_text: str = Form(""),
    companion_id: str = Form(...),
    replace: bool = Form(False),
    voice_name: str = Form(""),
):
    """Extract voice clone prompt from reference audio and store in S3.

    ``companion_id`` must match ``^[A-Za-z0-9_-]{1,64}$`` (422 otherwise). An
    existing clone is only overwritten when ``replace`` is true (409 otherwise).
    The clip is read through argus-cortex's ``read_upload``: over
    ``TTS_CLIP_MAX_BYTES`` is a 413. Writes ``prompt.pt`` and a ``meta.json``.
    This sidecar has no caller identity; ownership is enforced by the gateway.
    """
    if not _COMPANION_ID_RE.fullmatch(companion_id):
        return JSONResponse(status_code=422, content={"error": _COMPANION_ID_ERROR})
    s3_key = _clone_key(companion_id)

    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})

    try:
        audio_bytes = await read_upload(audio, CLIP_MAX_BYTES, what=audio.filename or "audio clip")
    except (TooLarge, BadInput) as exc:
        return _upload_refused(exc)

    lock = _clone_lock(companion_id)
    async with lock:
        if not replace and await asyncio.to_thread(_s3_object_exists, s3_key):
            return JSONResponse(
                status_code=409,
                content={"error": f"Voice clone {companion_id!r} already exists; send replace=true to overwrite"},
            )

        async with _mgr.use():
            return await _clone_voice_leased(audio_bytes, reference_text, companion_id, s3_key, voice_name=voice_name)


def _upload_refused(exc: ValueError) -> JSONResponse:
    """413 for an upload over its cap, 400 for a malformed one (argus-cortex limits)."""
    return JSONResponse(status_code=413 if isinstance(exc, TooLarge) else 400, content={"error": str(exc)})


async def _clone_voice_leased(
    audio_bytes: bytes, reference_text: str, companion_id: str, s3_key: str, voice_name: str = ""
):
    """Body of ``clone_voice``; the caller holds a ``_mgr.use()`` lease."""
    if _mgr.base_model is None:
        return JSONResponse(status_code=503, content={"error": "Base model not loaded (TTS_LOAD_MODELS=custom_voice)"})

    import soundfile as sf

    tmp_path = await asyncio.to_thread(_write_temp_wav, audio_bytes)

    try:
        info = await asyncio.to_thread(sf.info, tmp_path)
        duration = info.duration

        if duration < 3.0:
            return JSONResponse(status_code=422, content={"error": "Reference audio must be at least 3 seconds long"})

        use_xvector_only = not bool(reference_text)
        clone_prompt = await asyncio.to_thread(
            _mgr.base_model.create_voice_clone_prompt,
            tmp_path,
            ref_text=reference_text if reference_text else None,
            x_vector_only_mode=use_xvector_only,
        )
    except RuntimeError as exc:
        err_str = str(exc).lower()
        if "out of memory" in err_str or "cuda" in err_str:
            gpu_info = _gpu_memory_info()
            logger.error("tts_clone_gpu_oom", error=str(exc), gpu=gpu_info)
            return JSONResponse(status_code=503, content={"error": "GPU OOM", "gpu": gpu_info})
        raise
    finally:
        os.unlink(tmp_path)

    import torch

    buf = io.BytesIO()
    torch.save(clone_prompt, buf)
    prompt_bytes = buf.getvalue()

    await asyncio.to_thread(_upload_to_s3, prompt_bytes, s3_key, "application/octet-stream")

    meta = {
        "voice_clone_id": s3_key,
        "companion_id": companion_id,
        "name": voice_name or companion_id,
        "num_clips": 1,
        "total_seconds": round(duration, 2),
        "has_transcript": bool(reference_text),
        "engine": _ENGINE,
        "created_at": time.time(),
    }
    await asyncio.to_thread(_put_meta, companion_id, meta)

    with _clone_cache_lock:
        _clone_cache[s3_key] = clone_prompt
        if len(_clone_cache) > _CLONE_CACHE_MAX:
            _clone_cache.popitem(last=False)

    logger.info("voice_cloned", companion_id=companion_id, duration=duration, s3_key=s3_key)
    return {**meta, "duration": duration}


def _meta_key(companion_id: str) -> str:
    return f"voice_clones/{companion_id}/meta.json"


def _put_meta(companion_id: str, meta: dict) -> None:
    _upload_to_s3(json.dumps(meta).encode("utf-8"), _meta_key(companion_id), "application/json")


def _default_meta(companion_id: str, created_at: Optional[float] = None) -> dict:
    """What ``GET /clones`` reports for a clone stored without a ``meta.json``
    (every clone made before Qwen wrote one)."""
    return {
        "voice_clone_id": _clone_key(companion_id),
        "companion_id": companion_id,
        "name": companion_id,
        "num_clips": 1,
        "total_seconds": None,
        "engine": _ENGINE,
        "created_at": created_at,
    }


def _read_meta(client, companion_id: str) -> Optional[dict]:
    """A clone's ``meta.json`` as a dict, or None if it is missing or unreadable."""
    from botocore.exceptions import ClientError

    key = _meta_key(companion_id)
    buf = io.BytesIO()
    try:
        client.download_fileobj(S3_BUCKET_NAME, key, buf)
        if buf.tell() > _MAX_META_BYTES:
            raise ValueError("meta.json is unexpectedly large")
        meta = json.loads(buf.getvalue().decode("utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("meta.json is not an object")
    except (ClientError, ValueError) as exc:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        logger.warning("clone_meta_unreadable", key=key, error=str(exc))
        return None
    return meta


# ── GET /clones ───────────────────────────────────────────────────────────────


def _list_clones() -> list[dict]:
    """Every stored clone, newest first.

    A clone is a ``voice_clones/<id>/prompt.pt``; its ``meta.json`` supplies the
    listing entry when there is one. Clones made before Qwen wrote a
    ``meta.json`` are listed from the prompt alone, with its upload time."""
    client = _get_s3_client()
    prompts: dict[str, Optional[float]] = {}
    with_meta: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix="voice_clones/"):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) != 3 or not _COMPANION_ID_RE.fullmatch(parts[1]):
                continue
            if parts[2] == "prompt.pt":
                modified = obj.get("LastModified")
                prompts[parts[1]] = modified.timestamp() if hasattr(modified, "timestamp") else None
            elif parts[2] == "meta.json":
                with_meta.add(parts[1])

    clones: list[dict] = []
    for companion_id, modified in prompts.items():
        meta = _read_meta(client, companion_id) if companion_id in with_meta else None
        if meta is None:
            meta = _default_meta(companion_id, modified)
        else:
            meta.setdefault("voice_clone_id", _clone_key(companion_id))
            meta.setdefault("companion_id", companion_id)
            meta.setdefault("engine", _ENGINE)
        clones.append(meta)

    def _created(c: dict) -> float:
        # One type for the sort key: a string or missing created_at sorts last.
        value = c.get("created_at")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0

    clones.sort(key=_created, reverse=True)
    return clones


@app.get("/clones")
async def list_clones():
    """List stored voice clones (id, name, duration), newest first."""
    try:
        clones = await asyncio.to_thread(_list_clones)
    except Exception as exc:
        request_id = uuid.uuid4().hex[:12]
        logger.error("clone_list_error", request_id=request_id, error=str(exc))
        return JSONResponse(status_code=500, content={"error": "Failed to list clones", "request_id": request_id})
    return {"clones": clones}


# ── GET /clones/{companion_id}/download ───────────────────────────────────────


def _build_clone_zip(companion_id: str) -> bytes:
    """Bundle a clone's ``prompt.pt`` and ``meta.json`` into a zip. A clone stored
    without a ``meta.json`` gets a generated one, so the archive names its engine."""
    import zipfile

    from botocore.exceptions import ClientError

    client = _get_s3_client()
    prompt_buf = io.BytesIO()
    try:
        client.download_fileobj(S3_BUCKET_NAME, _clone_key(companion_id), prompt_buf)
    except ClientError as exc:
        raise FileNotFoundError(f"Voice clone not found: {companion_id}") from exc

    meta = _read_meta(client, companion_id) or _default_meta(companion_id)
    meta.setdefault("engine", _ENGINE)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prompt.pt", prompt_buf.getvalue())
        zf.writestr("meta.json", json.dumps(meta))
    return zip_buf.getvalue()


@app.get("/clones/{companion_id}/download")
async def download_clone(companion_id: str):
    """Download a clone's prompt and metadata as ``<companion_id>.zip``."""
    if not _COMPANION_ID_RE.fullmatch(companion_id):
        return JSONResponse(status_code=422, content={"error": _COMPANION_ID_ERROR})
    try:
        data = await asyncio.to_thread(_build_clone_zip, companion_id)
    except FileNotFoundError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    except Exception as exc:
        request_id = uuid.uuid4().hex[:12]
        logger.error("clone_download_error", request_id=request_id, companion_id=companion_id, error=str(exc))
        return JSONResponse(status_code=500, content={"error": "Failed to bundle clone", "request_id": request_id})

    headers = {"Content-Disposition": f'attachment; filename="{companion_id}.zip"'}
    return Response(content=data, media_type="application/zip", headers=headers)


# ── DELETE /clones/{companion_id} ─────────────────────────────────────────────


def _delete_clone(companion_id: str) -> int:
    """Delete every S3 object under ``voice_clones/{companion_id}/``.

    Returns the number of objects removed and evicts the clone's cached prompt.
    Raises ``FileNotFoundError`` when nothing exists under the prefix, and
    ``RuntimeError`` when S3 refuses some of the objects.
    """
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
            _clone_cache.pop(_clone_key(companion_id), None)
    if failed:
        logger.error("clone_delete_partial", companion_id=companion_id, deleted=deleted, failed=failed)
        raise RuntimeError(f"{len(failed)} of {len(keys)} objects were not deleted ({'; '.join(failed[:5])})")
    return deleted


@app.delete("/clones/{companion_id}")
async def delete_clone(companion_id: str):
    """Permanently delete a stored voice clone (``prompt.pt`` + ``meta.json``).
    A partial delete (some objects refused by S3) is a 500, never ``deleted``."""
    if not _COMPANION_ID_RE.fullmatch(companion_id):
        return JSONResponse(status_code=422, content={"error": _COMPANION_ID_ERROR})
    try:
        async with _clone_lock(companion_id):
            objects = await asyncio.to_thread(_delete_clone, companion_id)
    except FileNotFoundError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    except Exception as exc:
        request_id = uuid.uuid4().hex[:12]
        logger.error("clone_delete_error", request_id=request_id, companion_id=companion_id, error=str(exc))
        return JSONResponse(status_code=500, content={"error": "Failed to delete clone", "request_id": request_id})
    logger.info("voice_deleted", companion_id=companion_id, objects=objects)
    return {"status": "deleted", "companion_id": companion_id, "objects": objects}


# ── POST /clones/import ───────────────────────────────────────────────────────


def _slug_id(value: str) -> str:
    """A valid companion id (``_COMPANION_ID_RE``) from free text."""
    s = (value or "").strip().lower()
    out = "".join(ch if ch.isascii() and ch.isalnum() else "-" for ch in s).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out[:64].strip("-") or f"imported-{int(time.time())}"


_MAX_ARCHIVE_ENTRIES = 16
# The upload itself: a prompt.pt at its cap plus a little for meta.json and zip overhead.
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


def _parse_clone_archive(zip_bytes: bytes, companion_override: str, name_override: str) -> tuple[str, list, dict]:
    """Validate a clone archive: ``(companion_id, prompt items, source meta)``.

    Only a Qwen clone is accepted. A ``meta.json`` that names another engine is
    refused, and ``prompt.pt`` must load (``weights_only``, tensors and Qwen's
    ``VoiceClonePromptItem`` only) as one valid prompt item: a Miso or XTTS
    ``prompt.pt`` is a dict of other tensors and fails that check. Raises ``ValueError``.
    """
    import zipfile

    try:
        z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a valid zip archive: {exc}") from exc

    names = z.namelist()
    if len(names) > _MAX_ARCHIVE_ENTRIES:
        raise ValueError(f"Archive has more than {_MAX_ARCHIVE_ENTRIES} entries")

    src_meta: dict = {}
    meta_name = next((n for n in names if n.rsplit("/", 1)[-1] == "meta.json"), None)
    if meta_name:
        meta_raw = _read_member(z, meta_name, _MAX_META_BYTES)  # oversized: ValueError → 422
        try:
            src_meta = json.loads(meta_raw.decode("utf-8"))
        except ValueError:
            src_meta = {}
        if not isinstance(src_meta, dict):
            src_meta = {}
    engine = src_meta.get("engine")
    if isinstance(engine, str) and engine.strip().lower() not in ("", _ENGINE, "qwen"):
        raise ValueError(f"Archive is a {engine!r} clone; this engine only imports Qwen clones")

    prompt_name = next((n for n in names if n.rsplit("/", 1)[-1] == "prompt.pt"), None)
    if not prompt_name:
        raise ValueError("Archive does not contain prompt.pt")
    items = _load_prompt(_read_member(z, prompt_name, _MAX_PROMPT_BYTES))

    def _text(value) -> str:
        return value.strip() if isinstance(value, str) else ""

    companion_id = _text(companion_override) or _text(src_meta.get("companion_id"))
    companion_id = _slug_id(companion_id or _text(name_override) or _text(src_meta.get("name")) or "imported-voice")
    if not _COMPANION_ID_RE.fullmatch(companion_id):  # _slug_id guarantees it; belt and braces
        raise ValueError(_COMPANION_ID_ERROR)
    return companion_id, items, src_meta


def _store_imported_clone(companion_id: str, items: list, src_meta: dict, name_override: str, replace: bool) -> dict:
    """Store a validated import. ``CloneExistsError`` if the id is taken and not ``replace``.
    The caller holds the id's clone lock."""
    import torch

    s3_key = _clone_key(companion_id)
    if not replace and _s3_object_exists(s3_key):
        raise CloneExistsError(f"Voice clone {companion_id!r} already exists; send replace=true to overwrite")

    # Re-serialised from the validated items, so whatever else the uploaded
    # pickle held is not what gets stored.
    buf = io.BytesIO()
    torch.save(items, buf)
    _upload_to_s3(buf.getvalue(), s3_key, "application/octet-stream")

    def _number(value, default):
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else default

    src_name = src_meta.get("name")
    name = (name_override or "").strip() or (src_name.strip()[:200] if isinstance(src_name, str) else "") or companion_id
    meta = {
        "voice_clone_id": s3_key,
        "companion_id": companion_id,
        "name": name,
        "num_clips": _number(src_meta.get("num_clips"), 1),
        "total_seconds": _number(src_meta.get("total_seconds"), 0),
        "has_transcript": bool(items[0].icl_mode),
        "engine": _ENGINE,
        "imported": True,
        "created_at": time.time(),
    }
    _put_meta(companion_id, meta)

    with _clone_cache_lock:
        _clone_cache[s3_key] = items
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
    """Import a Qwen clone archive (the ``/clones/{id}/download`` zip).

    422 for a bad zip or anything but a valid Qwen ``prompt.pt`` (another
    engine's clone included); 409 when the target id already has a clone,
    unless ``replace`` is true; 413 for an archive over ``_MAX_ARCHIVE_BYTES``."""
    try:
        data = await read_upload(archive, _MAX_ARCHIVE_BYTES, what="clone archive")
    except (TooLarge, BadInput) as exc:
        return _upload_refused(exc)
    try:
        target, items, src_meta = await asyncio.to_thread(_parse_clone_archive, data, companion_id, voice_name)
        async with _clone_lock(target):
            return await asyncio.to_thread(_store_imported_clone, target, items, src_meta, voice_name, replace)
    except CloneExistsError as exc:
        return JSONResponse(status_code=409, content={"error": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    except Exception as exc:
        request_id = uuid.uuid4().hex[:12]
        logger.error("clone_import_error", request_id=request_id, error=str(exc))
        return JSONResponse(status_code=500, content={"error": "Failed to import voice", "request_id": request_id})


# ── POST /analyze-clips ───────────────────────────────────────────────────────
# Scores candidate reference clips so a client can pick the one to clone from
# (Qwen's /clone-voice takes one clip). The quality half is engine-independent:
# clipping, usable speech length and SNR from an energy gate (numpy, no VAD
# model). The consistency half compares Qwen's own speaker embeddings (the
# x-vector create_voice_clone_prompt computes), so a clip of another voice, room
# or microphone stands out.

# Speech this long or longer counts fully towards a clip's anchor rank. Reported
# as ``gpt_cond_len`` for clients written against the XTTS response.
_ANALYZE_REF_SECONDS = 15.0
# Only this much of each clip goes to the speaker encoder.
_ANALYZE_EMBED_SECONDS = 30.0
_FRAME_S = 0.03


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _decode_clip(data: bytes):
    """``(mono float32 samples, sample_rate)`` of an uploaded clip (anything libsndfile reads)."""
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return np.ascontiguousarray(wav.mean(axis=1), dtype=np.float32), int(sr)


def _quick_clip_metrics(wav, sr: int) -> dict:
    """Clipping, peak, usable speech seconds and SNR of one clip.

    Speech is found with an energy gate on 30 ms frames: a frame is speech when
    it is 10 dB over the clip's noise floor (its 10th-percentile frame) and
    within 35 dB of its loud frames (95th percentile). SNR compares the speech
    frames with the frames near the floor, when there is 0.2 s of those. A clip
    with no clear floor (no pauses) counts as all speech, with no SNR."""
    import numpy as np

    m: dict = {"flags": []}
    n = int(wav.shape[0])
    mag = np.abs(wav)
    m["raw_seconds"] = round(n / sr, 2) if sr else 0.0
    m["peak"] = round(float(mag.max()) if n else 0.0, 3)
    m["clip_ratio"] = round(float((mag > 0.99).mean()) if n else 0.0, 4)
    m["speech_seconds"] = 0.0
    m["snr_db"] = None
    m["noise_floor_dbfs"] = None

    frame = max(1, int(sr * _FRAME_S))
    if n < frame:
        m["flags"].append("vad_no_speech")
        return m
    frames = wav[: (n // frame) * frame].astype(np.float64).reshape(-1, frame)
    power = np.mean(frames**2, axis=1)
    db = 10.0 * np.log10(power + 1e-12)
    floor_db, loud_db = float(np.percentile(db, 10)), float(np.percentile(db, 95))

    if loud_db < -55.0:  # nothing above a whisper of noise
        m["flags"].append("vad_no_speech")
        return m
    if loud_db - floor_db < 10.0:  # no pauses to measure a floor from
        m["speech_seconds"] = m["raw_seconds"]
        return m

    speech = db > max(floor_db + 10.0, loud_db - 35.0)
    noise = db <= floor_db + 3.0
    m["speech_seconds"] = round(float(speech.sum()) * frame / sr, 2)
    noise_power = float(power[noise].mean()) if noise.sum() * frame >= 0.2 * sr else None
    if noise_power is not None:
        m["noise_floor_dbfs"] = round(10.0 * math.log10(noise_power + 1e-12), 1)
        m["snr_db"] = round(10.0 * math.log10((float(power[speech].mean()) + 1e-12) / (noise_power + 1e-12)), 1)
    return m


def _unit_vector(embedding):
    """A speaker embedding (torch tensor or array) as a unit-length float64 numpy vector."""
    import numpy as np

    if hasattr(embedding, "detach"):
        embedding = embedding.detach().float().cpu().numpy()
    v = np.asarray(embedding, dtype=np.float64).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


def _score_clips(items: list[dict]) -> None:
    """Score each clip (0–100), set ``recommended``, flags and the ``_anchor`` sort key.

    The same scoring as the XTTS engine. A clip that couldn't be read scores 0, is
    never recommended and sorts last. A clip without a speaker embedding has no
    consistency evidence, so it gets none (0) rather than a neutral half."""
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
        item["_anchor"] = 0.45 * quality + 0.3 * cons_norm + 0.25 * _clamp01(speech / _ANALYZE_REF_SECONDS)

        if (item.get("clip_ratio") or 0.0) > 0.01:
            item["flags"].append("clipping")
        if snr is not None and snr < 15:
            item["flags"].append("noisy")
        if speech and speech < 3.0:  # /clone-voice refuses under 3 s
            item["flags"].append("very_short")
        if cons_norm < 0.3:
            item["flags"].append("outlier")
        item["recommended"] = score >= ANALYZE_SCORE_THRESHOLD
        item["flags"] = sorted(set(item["flags"]))


def _analyze_clips(blobs: list[bytes], names: list[str], base) -> dict:
    """Score each candidate clip and rank them, best first. Blocking; the
    caller holds a model lease (the speaker embeddings run on the Base model)."""
    import numpy as np

    items: list[dict] = []
    embeddings: list = []
    for data, name in zip(blobs, names):
        item: dict = {"name": name, "flags": []}
        try:
            wav, sr = _decode_clip(data)
            item.update(_quick_clip_metrics(wav, sr))
        except Exception as exc:
            item["flags"].append(f"unreadable:{str(exc)[:200]}")
            items.append(item)
            embeddings.append(None)
            continue

        emb = None
        if item["speech_seconds"] > 0:
            try:
                prompt = base.create_voice_clone_prompt(
                    (wav[: int(_ANALYZE_EMBED_SECONDS * sr)], sr), x_vector_only_mode=True
                )
                emb = _unit_vector(prompt[0].ref_spk_embedding)
            except RuntimeError as exc:
                err = str(exc).lower()
                if "out of memory" in err or "cuda" in err:
                    raise  # the route answers 503
                item["flags"].append(f"embed_failed:{str(exc)[:200]}")
            except Exception as exc:
                item["flags"].append(f"embed_failed:{str(exc)[:200]}")
        embeddings.append(emb)
        items.append(item)

    # Consistency: cosine similarity of each embedding to the centroid.
    valid = [e for e in embeddings if e is not None]
    centroid = None
    if valid:
        centroid = np.mean(valid, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    for item, emb in zip(items, embeddings):
        item["consistency"] = (
            round(float(np.dot(emb, centroid)), 3) if (emb is not None and centroid is not None) else None
        )

    _score_clips(items)

    order = sorted(range(len(items)), key=lambda i: items[i]["_anchor"], reverse=True)
    for rank, idx in enumerate(order):
        items[idx]["rank"] = rank
    for item in items:
        item.pop("_anchor", None)

    return {
        "clips": items,
        "order": order,
        "threshold": ANALYZE_SCORE_THRESHOLD,
        "gpt_cond_len": _ANALYZE_REF_SECONDS,
        "engine": _ENGINE,
    }


@app.post("/analyze-clips")
async def analyze_clips(audio: list[UploadFile] = File(...)):
    """Score candidate reference clips and rank them, best first (no clone made).

    At most ``TTS_MAX_CLIPS`` clips of ``TTS_CLIP_MAX_BYTES`` each (413), read
    through argus-cortex's ``read_upload``. Needs the Base model (503 without it)."""
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})

    clips = audio if isinstance(audio, list) else [audio]
    if not clips:
        return JSONResponse(status_code=422, content={"error": "No clips provided"})
    if len(clips) > MAX_CLIPS:
        return JSONResponse(status_code=413, content={"error": f"At most {MAX_CLIPS} clips per request"})

    blobs: list[bytes] = []
    for i, clip in enumerate(clips):
        try:
            blobs.append(await read_upload(clip, CLIP_MAX_BYTES, what=clip.filename or f"clip {i + 1}"))
        except (TooLarge, BadInput) as exc:
            return _upload_refused(exc)
    names = [clip.filename or f"clip-{i + 1}" for i, clip in enumerate(clips)]

    async with _mgr.use():
        base = _mgr.base_model
        if base is None:
            return JSONResponse(status_code=503, content={"error": "Base model not loaded (TTS_LOAD_MODELS=custom_voice)"})
        try:
            return await asyncio.to_thread(_analyze_clips, blobs, names, base)
        except RuntimeError as exc:
            err = str(exc).lower()
            if "out of memory" in err or "cuda" in err:
                gpu_info = _gpu_memory_info()
                logger.error("tts_analyze_gpu_oom", error=str(exc), gpu=gpu_info)
                return JSONResponse(status_code=503, content={"error": "GPU OOM", "gpu": gpu_info})
            raise


# ── GET /speakers ─────────────────────────────────────────────────────────────


@app.get("/speakers")
async def list_speakers():
    """List available predefined CustomVoice speakers.

    Returns structured objects per speaker with id, name, and language.
    Clients use it to validate a speaker name or to fill a voice picker.
    """
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})

    async with _mgr.use():
        if _mgr.custom_model is None:
            return JSONResponse(status_code=503, content={"error": "CustomVoice model not loaded"})

        raw_speakers = await asyncio.to_thread(_mgr.custom_model.get_supported_speakers)
    supported_langs = _get_supported_languages()

    speakers = [
        {
            "id": name,
            "name": name,
            "language": "en",
            "languages": supported_langs,
        }
        for name in sorted(raw_speakers)
    ]

    return {"speakers": speakers}


# ── GET /health ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"status": "loading"})
    gpu_info = _resident_gpu()
    return {
        "status": "ok",
        "engine": _ENGINE,
        "capabilities": ["tts"],  # the echo contracts this engine serves (docs/api.md)
        "model": MODEL_SIZE,
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "load_models": LOAD_MODELS,
        "default_speaker": DEFAULT_SPEAKER or None,
        "model_state": _mgr.state.value,
        "busy": _mgr._in_use > 0,
        "idle_s": _idle_s(),
        "keep_alive_gpu": KEEP_ALIVE_GPU,
        "keep_alive_cpu": KEEP_ALIVE_CPU,
        "gpu": gpu_info,
        "weights": _mgr.weights,
        "image": _image(),
    }


# ── Resident contract (argus-iris argus-http adapter) ─────────────────────────
# /health carries a ``gpu`` block (loaded, model_state, idle_s, free_vram_mb),
# POST /admin/evict demotes to CPU, /admin/unload frees the model, /admin/wake
# preloads it. The admin routes need ``Authorization: Bearer <ARGUS_ADMIN_TOKEN>``.
# They fail closed: with the token unset they answer 503, unless
# ARGUS_ADMIN_INSECURE=1 explicitly opens them (local dev only).


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
    if not last_used or _mgr._in_use:
        return None
    return round(time.time() - last_used, 1)


@app.post("/admin/evict", dependencies=[Depends(_require_admin)])
async def admin_evict():
    """Force-demote models from GPU to CPU, freeing VRAM immediately."""
    await _mgr.demote_to_cpu()
    return {"status": "evicted", "model_state": _mgr.state.value, "gpu": _resident_gpu()}


@app.post("/admin/unload", dependencies=[Depends(_require_admin)])
async def admin_unload():
    """Fully unload models from memory (GPU + CPU). A loaded checkpoint is dropped too: the next load is stock."""
    was_loaded = _mgr.state in (TTSState.CPU, TTSState.GPU)
    await _mgr.unload(to_stock=True)
    return {
        "status": "unloaded",
        "unloaded": was_loaded,
        "model_state": _mgr.state.value,
        "gpu": _resident_gpu(),
        "weights": _mgr.weights,
    }


class LoadRequest(BaseModel):
    # A directory name under TTS_CHECKPOINT_ROOT (may be nested: "forge/run-7"); null loads the stock models.
    checkpoint: Optional[str] = Field(...)


def _image() -> dict:
    """``/health``'s ``image``: the release and git revision baked in at build time (null for a local build). A
    container can't see its own image digest; whoever deployed it (compose, iris) knows that."""
    return {"version": ECHO_VERSION, "revision": ECHO_REVISION}


@app.post("/admin/load", dependencies=[Depends(_require_admin)])
async def admin_load(req: LoadRequest):
    """Load a fine-tuned checkpoint from ``TTS_CHECKPOINT_ROOT`` into the slot its ``tts_model_type`` names (or the
    stock models, for ``null``) onto ``TTS_DEVICE``, replacing what is loaded, and hash its files.

    400 a name that escapes the root, 404 no such checkpoint, 422 not a Qwen3-TTS base/custom_voice model
    directory, 409 busy (a request holds the models; nothing changes), 503 no checkpoint root configured, 500 the
    load failed (nothing is loaded and the stock models are selected)."""
    selected = None
    if req.checkpoint is not None:
        try:
            path, slot = _resolve_checkpoint(req.checkpoint)
        except CheckpointRefused as exc:
            logger.warning("tts_checkpoint_refused", checkpoint=req.checkpoint, status=exc.status, error=str(exc))
            return JSONResponse(status_code=exc.status, content={"error": str(exc)})
        selected = (req.checkpoint, path, slot)
    try:
        loaded = await _mgr.load_weights(selected)
    except Exception as exc:
        logger.error("tts_checkpoint_load_failed", checkpoint=req.checkpoint, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"loading {req.checkpoint or 'the stock models'} failed: {exc}",
                "model_state": _mgr.state.value,
                "weights": _mgr.weights,
            },
        )
    if not loaded:
        return JSONResponse(
            status_code=409,
            content={"status": "busy", "active": _mgr._in_use, "model_state": _mgr.state.value},
        )
    logger.info("tts_checkpoint_loaded", checkpoint=req.checkpoint, slot=selected[2] if selected else None)
    return {
        "status": "loaded",
        "slot": selected[2] if selected else None,
        "model_state": _mgr.state.value,
        "weights": _mgr.weights,
        "gpu": _resident_gpu(),
    }


@app.post("/admin/wake", dependencies=[Depends(_require_admin)])
async def admin_wake():
    """Load models and promote to GPU."""
    await _mgr.ensure_gpu()
    return {"status": "ready", "model_state": _mgr.state.value, "gpu": _resident_gpu()}
