# The echo contract

Every engine is its own HTTP server with the same routes and request fields, so a client can switch engines by
changing a URL. The Python statement of the shared parts is
[src/argus_echo/contract.py](../src/argus_echo/contract.py), and `argus-echo check <url>` tests a running engine
against it.

Per-engine detail: [Miso](miso-api.md) (the most complete reference; most of it holds for every engine),
[XTTS-v2](xtts.md), [Qwen3-TTS](qwen.md).

## Routes

| Method | Path | Miso | XTTS-v2 | Qwen3-TTS |
| --- | --- | --- | --- | --- |
| POST | `/synthesise` | yes | yes | yes |
| POST | `/synthesise/stream` | yes | yes | yes, as one chunk at the end |
| POST | `/clone-voice` | yes | yes | yes |
| POST | `/analyze-clips` | yes | yes | no |
| GET | `/clones` | yes | yes | no |
| GET | `/clones/{companion_id}/download` | yes | yes | no |
| DELETE | `/clones/{companion_id}` | yes | yes | no |
| POST | `/clones/import` | yes | yes | no |
| GET | `/speakers` | yes | yes | yes |
| GET | `/health` | yes | yes | yes |
| POST | `/admin/{evict,unload,wake}` | yes | yes | yes |
| GET | `/media/{key}` | local backend only | no | no |

Every engine listens on `8031` inside its container. [compose.yaml](../compose.yaml) publishes Miso on `8140`,
XTTS on `8141` and Qwen on `8142`.

## Synthesis

`POST /synthesise` and `POST /synthesise/stream` take the same JSON body:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `text` | string | required | |
| `voice_clone_id` | string | `null` | A stored clone. Wins over `speaker` |
| `speaker` | string | `null` | A built-in speaker from `GET /speakers` |
| `language` | string | `"en"` | ISO code. Miso: English only. XTTS: 17 languages. Qwen: 10 (`en`, `zh`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`) |
| `voice_description` | string | `null` | Qwen only: describe a voice to design. Ignored elsewhere |
| `instruction` | string | `null` | Qwen only: delivery instruction for a built-in speaker. Ignored elsewhere |
| `emotion` | string | `"neutral"` | Qwen only. Ignored elsewhere |
| `speed` | number | `null` | Accepted, applied by no engine yet |
| `temperature`, `topk`, `context` | | | Miso only; see [miso-api.md](miso-api.md) |

An engine ignores fields it has no use for instead of refusing them, so one request body works on all three.

`/synthesise` stores a WAV and returns `{"audio_url", "duration", "duration_seconds", "format"}`.

`/synthesise/stream` answers `200` with `Content-Type: application/octet-stream` and the PCM layout in headers:

| Header | Value on every engine today |
| --- | --- |
| `X-Sample-Rate` | `24000` |
| `X-Audio-Format` | `pcm_f32le` (little-endian float32 in [-1, 1]) |
| `X-Channels` | `1` |

Read the format from the headers rather than assuming it. Errors found before audio starts (validation, model
loading, out of memory, busy) are normal JSON errors; once the `200` is sent, a failure can only end the stream
early. How soon audio starts differs: Miso and XTTS stream within a sentence, while Qwen renders the whole clip and
sends it at the end.

Busy, loading and out-of-memory answers are `503`, with `Retry-After` where the engine knows a good wait.

## Voices

Built-in speakers (`GET /speakers` → `{"speakers": [{"id", "name", "language", "languages"}]}`):

- **Miso** has no real built-in voices: `0` and `1` are conversation turns, and without a clone the voice is
  different every time. Clone a voice for anything you keep.
- **XTTS-v2** has its studio speakers (`Claribel Dervla`, …).
- **Qwen3-TTS** has the CustomVoice model's speakers, and can design a voice from `voice_description`.

On XTTS and Qwen, `GET /speakers` loads the model if it is not loaded.

`POST /clone-voice` is `multipart/form-data` with one or more `audio` files, `reference_text` (the transcript;
Miso needs it) and `companion_id` (`^[A-Za-z0-9_-]{1,64}$`). A clone is stored at
`voice_clones/<companion_id>/prompt.pt` with a `meta.json`, and used by passing its id as `voice_clone_id`.
**Clones are engine-specific**: each engine's `prompt.pt` is its own format, which is why each engine gets its
own bucket. See [voice-cloning.md](voice-cloning.md) for what makes a good reference.

## Health and lifecycle

`GET /health` is argus-cortex's `ResidentHealth`, which argus-iris reads: top-level `model_state` (`unloaded`,
`loading`, `cpu`, `gpu`, `error`), `busy`, `idle_s`, and a `gpu` memory block, plus `engine`, `model` and
`device`. It answers `503 {"status": "loading"}` while a model loads.

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
