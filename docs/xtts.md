# XTTS-v2 engine

[Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) through the maintained
[`coqui-tts`](https://github.com/idiap/coqui-ai-TTS) fork: zero-shot cloning from several reference clips, 17
languages, and streaming within a sentence. About 2 GB of VRAM, so it fits an 8 GB card.

Source: [engines/xtts/xtts_server.py](../engines/xtts/xtts_server.py). Contract: [api.md](api.md). Published on
`8141` by `compose.yaml` (`--profile xtts`).

**Licence:** the XTTS-v2 weights are under the [Coqui Public Model Licence](https://coqui.ai/cpml), which allows
non-commercial use only. The image bakes them in and accepts that licence at build time (`COQUI_TOS_AGREED=1`).

## Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `XTTS_MODEL` | `tts_models/multilingual/multi-dataset/xtts_v2` | Coqui model name |
| `XTTS_DEVICE` | `cuda:0` | `cuda:N` or `cpu` |
| `XTTS_PRELOAD` | `0` | `1` loads the model at startup |
| `XTTS_USE_DEEPSPEED` | `0` | `1` uses DeepSpeed inference (needs it installed) |
| `XTTS_KEEP_ALIVE_GPU` | `120` | Idle seconds on the GPU before moving to CPU. Negative = never |
| `XTTS_KEEP_ALIVE_CPU` | `600` | Idle seconds on CPU before unloading. Negative = never |
| `XTTS_TEMPERATURE` | `0.7` | Default sampling temperature |
| `XTTS_STREAM_CHUNK_SIZE` | `20` | Tokens per streamed chunk; lower means an earlier first byte |
| `XTTS_GPT_COND_LEN` | `30` | Seconds of reference audio the conditioning latent draws on (clips are ordered best first) |
| `XTTS_GPT_COND_CHUNK_LEN` | `4` | Chunk length for that conditioning |
| `XTTS_MAX_REF_LEN` | `30` | Longest reference used, in seconds |
| `XTTS_MAX_CLIPS` | `16` | Most clips per `/clone-voice` or `/analyze-clips`; more is `413` |
| `XTTS_CLIP_MAX_BYTES` | `16777216` | Largest clip (16 MiB); larger is `413` |
| `XTTS_ANALYZE_THRESHOLD` | `60` | `/analyze-clips` score below which a clip is left out by default |
| `ARGUS_ADMIN_TOKEN`, `ARGUS_ADMIN_INSECURE` | empty | See [api.md#security](api.md#security) |

Reference-clip preprocessing, applied to each clip before the latents are computed:

| Variable | Default | Effect |
| --- | --- | --- |
| `XTTS_PREPROCESS` | `1` | `0` skips all preprocessing |
| `XTTS_PREPROCESS_SR` | `24000` | Resample to this rate |
| `XTTS_PREPROCESS_HIGHPASS_HZ`, `XTTS_PREPROCESS_HIGHPASS_ORDER` | `85`, `4` | High-pass filter |
| `XTTS_PREPROCESS_VAD`, `XTTS_PREPROCESS_VAD_PAD` | `1`, `0.1` | Trim non-speech with Silero VAD, keeping this much padding (s) |
| `XTTS_PREPROCESS_DENOISE`, `XTTS_PREPROCESS_DENOISE_PROP`, `XTTS_PREPROCESS_DENOISE_SNR_DB` | `1`, `0.75`, `30` | Stationary denoise seeded from the VAD gaps, applied only below this SNR |
| `XTTS_PREPROCESS_LUFS` | `-23.0` | Loudness target |
| `XTTS_PREPROCESS_MAX_SECONDS` | `30` | Longest clip kept |

Storage is S3 only: `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME` (compose:
`XTTS_S3_BUCKET`, default `voice-xtts`) and `S3_PUBLIC_URL` (compose: `XTTS_S3_PUBLIC_URL`). The bucket must exist.
`S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY` have no default: the server refuses to start without both.
