# Changelog

## [Unreleased]

### Added
- Engines salvaged into one repo: Miso TTS 8B (from smk762/voluptas `MisoTTS/`), XTTS-v2 and Qwen3-TTS (from
  smk762/agent-composer `voice-stack/`), and the Wyoming TTS bridge.
- `compose.yaml` with a profile per engine, published on `ECHO_BIND` (loopback by default) at 8140–8142 and
  10240–10241.
- One S3 bucket per engine by default (`voice-miso`, `voice-xtts`, `voice-qwen`): the engines store clones under
  the same keys in different formats.
- `argus_echo.contract` (shared request fields, PCM stream headers) and `argus-echo check <url>`.
- `gpu-tenant.yaml` for argus-iris; docs for the shared contract, each engine, and voice cloning.

### Changed
- Every engine's `/admin/*` uses argus-cortex's `bearer`: closed (503) while `ARGUS_ADMIN_TOKEN` is unset, unless
  `ARGUS_ADMIN_INSECURE=1`. Miso was open when unset in voluptas.
