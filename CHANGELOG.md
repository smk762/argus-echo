# Changelog

## [Unreleased]

### Added
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
- Miso clone archives (`GET /clones/{id}/download`, `POST /clones/import`) carry the prompt as
  `prompt.safetensors` (tensors, with speaker/sample_rate/ref_text in its metadata header) instead of a pickled
  `prompt.pt`. Import never unpickles an upload: `torch.load(weights_only=True)` on the unauthenticated import
  route was remote code execution on Miso's torch 2.4 (CVE-2025-32434). An archive holding only `prompt.pt` is
  refused with `422`; re-export it from a current engine. Stored clones keep loading and download as safetensors.
- Every engine's `/admin/*` uses argus-cortex's `bearer`: closed (503) while `ARGUS_ADMIN_TOKEN` is unset, unless
  `ARGUS_ADMIN_INSECURE=1`. Miso was open when unset in voluptas.
