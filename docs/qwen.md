# Qwen3-TTS engine

[Qwen3-TTS](https://huggingface.co/Qwen) (12 Hz, 0.6B or 1.7B) loads two models: **Base**, which clones a voice
from one reference clip, and **CustomVoice**, which has built-in speakers that take a delivery `instruction` and
can design a voice from a `voice_description`. 10 languages. The 0.6B pair fits an 8 GB card.

Source: [engines/qwen/tts_server.py](../engines/qwen/tts_server.py). Contract: [api.md](api.md). Published on
`8142` by `compose.yaml` (`--profile qwen`). The weights are Apache-2.0 and are baked into the image at build
time (`QWEN_MODEL_SIZE`).

Differences from the other engines:

- `/synthesise/stream` streams one sentence at a time: the text is split at `.`, `!`, `?`, `…`, CJK full stops
  and blank lines (fragments under 24 characters join a neighbour), and each sentence's PCM is sent as soon as it
  is rendered. The first byte comes after the first sentence, not within it as on Miso and XTTS. The first
  sentence is rendered before the `200`, so its errors are JSON errors; a later failure ends the stream early.
  One model lease covers the whole stream. When the client disconnects no further sentence starts; the one in
  progress finishes (generation can't be interrupted) and then releases the lease. Sentences are rendered
  independently, so prosody does not carry across a sentence boundary.
- `/clone-voice` takes one `audio` clip (3 s or more). With `reference_text` the clone uses in-context learning
  (the clip's codec tokens plus its transcript); without it, only the speaker embedding (x-vector). It refuses
  to overwrite an existing clone unless `replace=true` (`409`), and writes a `meta.json` (`voice_name` sets its
  `name`).
- `voice_description`, `instruction` and `emotion` are used (CustomVoice); the other engines ignore them.
- `speed` is accepted and ignored.

## Clones

`GET /clones`, `GET /clones/{companion_id}/download`, `DELETE /clones/{companion_id}` and `POST /clones/import`
behave as on XTTS (see [miso-api.md](miso-api.md#get-clones) for the shapes), with these Qwen specifics:

- A Qwen `prompt.pt` is qwen-tts's `create_voice_clone_prompt` result: a one-item list of
  `VoiceClonePromptItem` (speaker embedding, and for ICL clones the reference codec tokens and transcript). It
  is loaded with `torch.load(weights_only=True)` with only that class allowed, so a clone can't run code.
- `GET /clones` lists every `voice_clones/<id>/prompt.pt`. Clones made before Qwen wrote `meta.json` are listed
  from the prompt alone (`name` = id, `created_at` = upload time), and their download carries a generated
  `meta.json`. Every `meta.json` Qwen writes has `"engine": "qwen-tts"`.
- `POST /clones/import` accepts only a Qwen clone: `422` if the archive's `meta.json` names another engine, or if
  its `prompt.pt` is not exactly one valid `VoiceClonePromptItem` (a Miso or XTTS `prompt.pt` is a dict and
  fails). The prompt is re-serialised from the validated fields. `409` if the id is taken and `replace` is not
  `true`; `413` over 17 MiB.

## POST /analyze-clips

Ranks candidate reference clips so a client can pick the one to clone from (Qwen clones from a single clip, so
use the first index in `order`). The response has the XTTS shape (`clips`, `order`, `threshold`,
`gpt_cond_len`) plus `"engine": "qwen-tts"`. How it differs from XTTS:

- **Quality** is engine-independent: clipping, peak, and speech seconds and SNR from an energy gate on 30 ms
  frames (numpy, no Silero VAD). A frame is speech when it is 10 dB over the clip's noise floor and within 35 dB
  of its loud frames; a clip with no pauses counts as all speech, with `snr_db: null`. Clips are decoded with
  libsndfile (WAV, FLAC, OGG, and MP3 where libsndfile supports it); anything else is flagged `unreadable:…`.
- **Consistency** compares Qwen's own speaker embeddings (the x-vector `create_voice_clone_prompt` computes, from
  at most the first 30 s of each clip), so a clip of another voice, room or microphone is flagged `outlier`.
  It needs the Base model: `503` without it.
- `very_short` flags under 3 s of speech (the `/clone-voice` minimum). `gpt_cond_len` is reported as `15`, the
  speech length that counts fully towards a clip's rank; Qwen has no conditioning window.

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
| `TTS_MAX_CLIPS` | `16` | Most clips per `/analyze-clips`; more is `413` |
| `TTS_CLIP_MAX_BYTES` | `16777216` | Largest clip for `/clone-voice` and `/analyze-clips` (16 MiB); larger is `413` |
| `TTS_ANALYZE_THRESHOLD` | `60` | `/analyze-clips` score below which a clip is not recommended |
| `ARGUS_ADMIN_TOKEN`, `ARGUS_ADMIN_INSECURE` | empty | See [api.md#security](api.md#security) |

Storage is S3 only: `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME` (compose:
`QWEN_S3_BUCKET`, default `voice-qwen`) and `S3_PUBLIC_URL` (compose: `QWEN_S3_PUBLIC_URL`). The bucket must exist.
`S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY` have no default: the server refuses to start without both.
