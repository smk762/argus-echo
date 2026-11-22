# Changelog

## [Unreleased]

### Added
- `gpu-tenant.yaml` gives Qwen a `vram_mb` (4800): Qwen3-TTS 1.7b with CustomVoice only, measured on an 8 GB
  RTX 4060 (dragon), CUDA context included. Generating stays inside the loaded footprint; evict and unload leave
  only the ~150 MB CUDA context (#6). Base and 0.6b are still to measure.
- Compose passes `QWEN_LOAD_MODELS` (`TTS_LOAD_MODELS`, default `both`) and `QWEN_DEFAULT_SPEAKER`
  (`TTS_DEFAULT_SPEAKER`) through to the Qwen engine; `custom_voice` roughly halves its VRAM when cloning isn't
  needed.

## [0.1.0] - 2026-11-22

### Added
- Miso `/clone-voice` fills an empty `reference_text` from the STT engine named by `MISO_STT_URL` (#13): each
  stored clip goes to `POST /transcribe` (with the new `language` form field) within `MISO_STT_TIMEOUT_S`
  (default 60), and `meta.json` records `reference_text_source` (`request`, `stt` or null) and
  `reference_text_stt`. An STT failure refuses the clone (502/503/504) instead of storing it without a transcript.
  `MISO_WHISPER_URL` is read as the old name.
- Speech to text (#4): the `whisper` engine (faster-whisper, Whisper large-v3-turbo, weights pinned by commit) on
  port 8144 with `POST /transcribe` and a VAD-endpointed `WS /transcribe/stream`, its Compose profile, image and
  `gpu-tenant.yaml` resident (3000 MB). The `stt` contract is `argus_echo.contract.stt`; docs/api.md and
  docs/whisper.md (model comparison with Parakeet).
- `argus_echo.contract` is a package with one module per capability (`tts`, `stt`); `from argus_echo.contract
  import ...` keeps working. `argus-echo check` runs only the suites an engine's `/health` `capabilities` name
  (none means `tts`), and `--transcribe` posts a bundled public-domain clip to an `stt` engine.
- Engines salvaged into one repo: Miso TTS 8B (from smk762/voluptas `MisoTTS/`), XTTS-v2 and Qwen3-TTS (from
  smk762/agent-composer `voice-stack/`), and the Wyoming TTS bridge.
- `compose.yaml` with a profile per engine, published on `ECHO_BIND` (loopback by default) at 8140–8142 and
  10240–10241.
- One S3 bucket per engine by default (`voice-miso`, `voice-xtts`, `voice-qwen`): the engines store clones under
  the same keys in different formats.
- `argus_echo.contract` (shared request fields, PCM stream headers) and `argus-echo check <url>`.
- `gpu-tenant.yaml` for argus-iris; docs for the shared contract, each engine, and voice cloning.

### Changed
- Evicting Qwen to the CPU left its speech tokenizer (the codec, one per loaded model) on the GPU: qwen-tts attaches it
  as a plain attribute, which `Module.to()` never reaches. It now moves with the model, and Qwen and XTTS clear
  cuBLAS's workspaces before `empty_cache()`, so an evicted engine holds only its CUDA context. Measured on an RTX
  3090: Qwen 0.6B CustomVoice evicted 776 → 322 MB, XTTS 336 → 314 MB. `/health`'s `gpu.cuda_context` says whether
  a context is held (#22).
- XTTS and Qwen refuse to start when `S3_ENDPOINT_URL` is not an http(s) URL or `S3_BUCKET_NAME` is empty, instead
  of failing each `/synthesise` after rendering (#22).
- Miso's `/synthesise` and `/synthesise/stream` answer `422` for a `language` other than `en` (or a region of it,
  `en-US`, `en_GB`) instead of mispronouncing it (#6).
- Miso clone archives (`GET /clones/{id}/download`, `POST /clones/import`) carry the prompt as
  `prompt.safetensors` (tensors, with speaker/sample_rate/ref_text in its metadata header) instead of a pickled
  `prompt.pt`. Import never unpickles an upload: `torch.load(weights_only=True)` on the unauthenticated import
  route was remote code execution on Miso's torch 2.4 (CVE-2025-32434). An archive holding only `prompt.pt` is
  refused with `422`; re-export it from a current engine. Stored clones keep loading and download as safetensors.
- Every engine's `/admin/*` uses argus-cortex's `bearer`: closed (503) while `ARGUS_ADMIN_TOKEN` is unset, unless
  `ARGUS_ADMIN_INSECURE=1`. Miso was open when unset in voluptas.
