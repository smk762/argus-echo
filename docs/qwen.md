# Qwen3-TTS engine

[Qwen3-TTS](https://huggingface.co/Qwen) (12 Hz, 0.6B or 1.7B) loads two models: **Base**, which clones a voice
from one reference clip, and **CustomVoice**, which has built-in speakers that take a delivery `instruction` and
can design a voice from a `voice_description`. 10 languages. The 0.6B pair fits an 8 GB card.

Source: [engines/qwen/tts_server.py](../engines/qwen/tts_server.py). Contract: [api.md](api.md). Published on
`8142` by `compose.yaml` (`--profile qwen`). The weights are Apache-2.0 and are baked into the image at build
time (`QWEN_MODEL_SIZE`).

Differences from the other engines:

- `/synthesise/stream` renders the whole clip, then sends it in one piece. The format headers are the same, but
  there is no early first byte.
- `/clone-voice` takes one `audio` clip. There is no `/analyze-clips`, and no `/clones` listing, download, delete
  or import: a clone is replaced by cloning again with `replace=true`.
- `voice_description`, `instruction` and `emotion` are used (CustomVoice); the other engines ignore them.

## Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `TTS_MODEL_SIZE` | `0.6b` | `0.6b` or `1.7b` (compose: `QWEN_MODEL_SIZE`, also the image build arg) |
| `TTS_DEVICE` | `cuda:0` | `cuda:N` or `cpu` |
| `TTS_LOAD_MODELS` | `both` | `both`, `custom_voice` or `base` |
| `TTS_DEFAULT_SPEAKER` | empty | Speaker for requests that name none |
| `TTS_PRELOAD` | `0` | `1` loads at startup |
| `TTS_KEEP_ALIVE_GPU` | `120` | Idle seconds on the GPU before moving to CPU. `-1` = never, `0` = at once |
| `TTS_KEEP_ALIVE_CPU` | `600` | Idle seconds on CPU before unloading. `-1` = never, `0` = at once |
| `TTS_MAX_NEW_TOKENS` | `0` | Codec-token cap per utterance; `0` sizes it from the text (4 per character + 64), so a sample that misses its end token cannot run on for minutes |
| `TTS_MAX_NEW_TOKENS_CEILING` | `1024` | Upper bound on that automatic cap (about 85 s at 12 Hz) |
| `ARGUS_ADMIN_TOKEN`, `ARGUS_ADMIN_INSECURE` | empty | See [api.md#security](api.md#security) |

Storage is S3 only: `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME` (compose:
`QWEN_S3_BUCKET`, default `voice-qwen`) and `S3_PUBLIC_URL` (compose: `QWEN_S3_PUBLIC_URL`). The bucket must exist.
`S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY` have no default: the server refuses to start without both.
