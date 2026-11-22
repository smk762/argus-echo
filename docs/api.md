# The echo contracts

Echo has one contract per capability: **text to speech** (`tts`, below) and **speech to text** (`stt`,
[at the end](#speech-to-text)). Every engine of a capability is its own HTTP server with the same routes and
request fields, so a client can switch engines by changing a URL. The Python statement is
[src/argus_echo/contract/](../src/argus_echo/contract/) (`tts.py`, `stt.py`), and `argus-echo check <url>` tests a
running engine against the capabilities its `/health` names.

Per-engine detail: [Miso](miso-api.md) (the most complete reference; most of it holds for every engine),
[XTTS-v2](xtts.md), [Qwen3-TTS](qwen.md).

## Text to speech

The `tts` contract, served by **Miso**, **XTTS-v2** and **Qwen3-TTS** (each advertises `capabilities: ["tts"]`).
Its Python statement is [src/argus_echo/contract/tts.py](../src/argus_echo/contract/tts.py).

### Routes

| Method | Path | Miso | XTTS-v2 | Qwen3-TTS |
| --- | --- | --- | --- | --- |
| POST | `/synthesise` | yes | yes | yes |
| POST | `/synthesise/stream` | yes | yes | yes, one sentence at a time |
| POST | `/clone-voice` | yes | yes | yes |
| POST | `/analyze-clips` | yes | yes | yes |
| GET | `/clones` | yes | yes | yes |
| GET | `/clones/{companion_id}/download` | yes | yes | yes |
| DELETE | `/clones/{companion_id}` | yes | yes | yes |
| POST | `/clones/import` | yes | yes | yes |
| GET | `/speakers` | yes | yes | yes |
| GET | `/health` | yes | yes | yes |
| POST | `/admin/{evict,unload,wake}` | yes | yes | yes |
| GET | `/media/{key}` | local backend only | no | no |

Every engine listens on `8031` inside its container. [compose.yaml](../compose.yaml) publishes Miso on `8140`,
XTTS on `8141`, Qwen on `8142` and Whisper (speech to text) on `8144`.

### Synthesis

`POST /synthesise` and `POST /synthesise/stream` take the same JSON body:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `text` | string | required | |
| `voice_clone_id` | string | `null` | A stored clone. Wins over `speaker` |
| `speaker` | string | `null` | A built-in speaker from `GET /speakers` |
| `language` | string | `"en"` | ISO code. Miso: English only; any other language is `422` (`en-US`, `en_GB` are accepted). XTTS: 17 languages. Qwen: 10 (`en`, `zh`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`) |
| `voice_description` | string | `null` | Qwen only: describe a voice to design. Ignored elsewhere |
| `instruction` | string | `null` | Qwen only: delivery instruction for a built-in speaker. Ignored elsewhere |
| `emotion` | string | `"neutral"` | Qwen only. Ignored elsewhere |
| `speed` | number | `null` | XTTS only: speaking-rate multiplier passed to inference (`1.0` when unset or not positive). Miso and Qwen accept it and ignore it |
| `temperature`, `topk`, `context` | | | Miso only; see [miso-api.md](miso-api.md) |

An engine ignores fields it has no use for instead of refusing them, so one request body works on all three. The
exception is a `language` the engine cannot speak: Miso refuses it rather than mispronounce it.

`/synthesise` stores a WAV and returns `{"audio_url", "duration", "duration_seconds", "format"}`.

`/synthesise/stream` answers `200` with `Content-Type: application/octet-stream` and the PCM layout in headers:

| Header | Value on every engine today |
| --- | --- |
| `X-Sample-Rate` | `24000` |
| `X-Audio-Format` | `pcm_f32le` (little-endian float32 in [-1, 1]) |
| `X-Channels` | `1` |

Read the format from the headers rather than assuming it. Errors found before audio starts (validation, model
loading, out of memory, busy) are normal JSON errors; once the `200` is sent, a failure can only end the stream
early. How soon audio starts differs: Miso and XTTS stream within a sentence, while Qwen splits the text into
sentences and sends each one as soon as it is rendered (the first byte comes after the first sentence). Qwen
renders that first sentence before answering `200`, so an error in it is still a JSON error. When the client
disconnects, XTTS stops at the next chunk and Qwen starts no further sentence (the one in progress finishes).

Busy, loading and out-of-memory answers are `503`, with `Retry-After` where the engine knows a good wait.

### Voices

Built-in speakers (`GET /speakers` → `{"speakers": [{"id", "name", "language", "languages"}]}`):

- **Miso** has no real built-in voices: `0` and `1` are conversation turns, and without a clone the voice is
  different every time. Clone a voice for anything you keep.
- **XTTS-v2** has its studio speakers (`Claribel Dervla`, …).
- **Qwen3-TTS** has the CustomVoice model's speakers, and can design a voice from `voice_description`.

On XTTS and Qwen, `GET /speakers` loads the model if it is not loaded.

`POST /clone-voice` is `multipart/form-data` with one or more `audio` files, `reference_text` (the transcript;
Miso needs it) and `companion_id` (`^[A-Za-z0-9_-]{1,64}$`). Miso fills an empty `reference_text` from the
speech-to-text engine when `MISO_STT_URL` names one (see [miso-api.md](miso-api.md#post-clone-voice)). A clone is stored at
`voice_clones/<companion_id>/prompt.pt` with a `meta.json`, and used by passing its id as `voice_clone_id`.
**Clones are engine-specific**: each engine's `prompt.pt` is its own format, which is why each engine gets its
own bucket. Download archives differ too: Miso's holds `prompt.safetensors` + `meta.json` and its import refuses
an archive with only a pickled `prompt.pt` (an older Miso download; re-export it), while XTTS's and Qwen's hold
`prompt.pt`. `POST /clones/import` answers `409` when the target id already has a clone, unless the form field
`replace=true` is sent, and `422` for an archive that is not this engine's format. Qwen checks both the archive's
`meta.json` `engine` and the `prompt.pt` itself; see [qwen.md](qwen.md#clones). See
[voice-cloning.md](voice-cloning.md) for what makes a good reference.

## Health and lifecycle

`GET /health` is argus-cortex's `ResidentHealth`, which argus-iris reads: top-level `model_state` (`unloaded`,
`loading`, `cpu`, `gpu`, `error`), `busy`, `idle_s`, and a `gpu` memory block, plus `engine`, `model`,
`device` and `capabilities` (the contracts the engine serves: `["tts"]` or `["stt"]`; `argus-echo check` runs
only those suites, and treats an engine that names none as `tts`). It answers `503 {"status": "loading"}` while a model loads. XTTS also answers `503` with
`"status": "gpu_error"` after a CUDA error that is not an out-of-memory: such an error breaks the process's GPU
context, so the worker needs a restart. (An out-of-memory is a `503 GPU OOM` on the request and can be retried.)

Each engine loads its model on first use and leaves the GPU after an idle period (`MISO_KEEP_ALIVE_GPU`,
`XTTS_KEEP_ALIVE_GPU`, `TTS_KEEP_ALIVE_GPU`). The admin routes move it on demand:

| Route | Effect |
| --- | --- |
| `POST /admin/evict` | GPU to CPU RAM (stays warm) |
| `POST /admin/unload` | Free GPU and RAM |
| `POST /admin/wake` | Load onto the GPU now |

## Security

- `/admin/*` needs `Authorization: Bearer <ARGUS_ADMIN_TOKEN>` (argus-cortex's `bearer`). With the token unset
  the routes answer `503`; `ARGUS_ADMIN_INSECURE=1` opens them for local development only.
- **No other route is authenticated.** Anyone who can reach an engine can synthesise, and can create, replace,
  download and delete clones. The engines have no idea who is calling, so nothing stops one caller from using or
  deleting another's clone. Keep the ports on loopback or a trusted network (`ECHO_BIND`), or put a gateway in
  front that authenticates callers and owns the `companion_id` namespace.
- A clone of a real person's voice needs their consent. argus-mask's Voices page records it
  (`argus_cortex.consent`); calling `/clone-voice` directly records nothing.

## Speech to text

The `stt` contract, served by the **whisper** engine (published on `8144`). Its Python statement is
[src/argus_echo/contract/stt.py](../src/argus_echo/contract/stt.py). An engine names what it serves in
`GET /health` as `capabilities` (`["stt"]` here); an engine that names nothing is a TTS engine, and
`argus-echo check` runs only the matching checks. Model choice and settings: [whisper.md](whisper.md). A Wyoming
client (Home Assistant) reaches it through the `wyoming-whisper` bridge, which posts each utterance to
`/transcribe` ([voice-cloning.md#home-assistant](voice-cloning.md#home-assistant)).

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/transcribe` | One clip in, one transcript out |
| WS | `/transcribe/stream` | Live PCM in, partial and final transcripts out |
| GET | `/health` | `ResidentHealth` plus `engine`, `model`, `revision`, `device`, `capabilities`, `streams` |
| POST | `/admin/{evict,unload,wake}` | As above; evict moves the weights to host RAM |

### `POST /transcribe`

`multipart/form-data`:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `audio` | file | required | WAV or FLAC (sniffed from the magic bytes; nothing else is decoded), 1 or 2 channels (averaged), 8 to 96 kHz |
| `language` | string | detected | ISO 639-1 (`en`). A region suffix is dropped (`en-US` → `en`). An unknown code is `422` |
| `word_timestamps` | bool | `false` | Adds `words` to every segment |

The upload is read through argus-cortex's `read_upload` and `open_audio`, so the byte cap (`STT_MAX_UPLOAD_MB`,
default 50), the format, channels, rate and duration (`STT_MAX_SECONDS`, default 600) are checked before a sample
is decoded. Nothing is stored.

```json
{
  "text": "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
  "language": "en",
  "language_probability": 0.96,
  "duration": 11.0,
  "segments": [
    {"start": 0.0, "end": 10.38, "text": "And so, my fellow Americans, ...",
     "words": [{"start": 0.0, "end": 0.24, "word": " And", "probability": 0.9}]}
  ]
}
```

`language_probability` is null when the language was given. `words` is present only with `word_timestamps`. `text`
is empty for silence. Times are seconds from the start of the clip.

| Status | When |
| --- | --- |
| `400` | Not WAV or FLAC, unreadable, too many channels, a rate outside the list |
| `413` | Over the byte cap or the duration cap |
| `422` | Unknown `language` |
| `503` | Loading, out of GPU memory, or every slot busy for `STT_QUEUE_TIMEOUT_S` (default 30); with `Retry-After` |

Transcriptions run one at a time (`STT_CONCURRENCY`); later requests queue for up to `STT_QUEUE_TIMEOUT_S`.
Whisper writes numbers as digits (`7.30`, `12%`): normalise both sides before scoring a transcript against a script.

### `WS /transcribe/stream`

For live speech. The server cuts the stream into utterances with voice activity detection (Silero VAD, the
model faster-whisper ships, on the CPU) and transcribes each one.

1. The server sends `{"type": "ready", "sample_rate": 16000, "audio_format": "pcm_s16le", "channels": 1,
   "min_silence_ms": 600, "max_utterance_s": 30}`.
2. The client may send `{"type": "start", "language": "en", "word_timestamps": false}` before any audio.
3. The client sends audio as binary frames: 16 kHz mono little-endian int16, at most 1 MiB a frame, split
   anywhere (an odd byte is carried to the next frame).
4. While an utterance is open the server sends `{"type": "partial", "start", "text"}` every
   `STT_STREAM_PARTIAL_S` (default 1.0) seconds of speech; each replaces the last.
5. After `STT_STREAM_MIN_SILENCE_MS` of silence (default 600), or `STT_STREAM_MAX_UTTERANCE_S` of speech (default
   30), it sends `{"type": "final", "start", "end", ...}` with the `POST /transcribe` fields. Every time is seconds
   from the start of the stream; an utterance starts 200 ms before the detected onset, and `end` includes the
   silence that closed it.
6. `{"type": "end"}` flushes the open utterance (its final) and closes the socket.

A bad control message, a bad `language` or an oversized frame gets `{"type": "error", "error"}` and a close.
At most `STT_STREAM_MAX_SESSIONS` (default 4) streams are open at once; the next gets an error and close code
`1013`. The model lease is held per transcription, not per stream, so an idle stream does not keep the model on
the GPU. Silence alone loads nothing.

`/transcribe` and `/transcribe/stream` are unauthenticated, like the TTS routes: keep the port on a trusted
network.
