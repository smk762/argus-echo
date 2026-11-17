# argus-echo

The Argus suite's speech servers. Three text-to-speech engines behind one HTTP contract, so a client switches
engine by changing a URL:

| Engine | Model | VRAM | Strengths |
| --- | --- | --- | --- |
| **miso** | [Miso TTS 8B](https://huggingface.co/MisoLabs/MisoTTS) | ~17 GB | Most natural and expressive; one-shot cloning; English |
| **xtts** | [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) | ~2 GB | Multi-clip cloning, 17 languages, low-latency streaming |
| **qwen** | [Qwen3-TTS](https://huggingface.co/Qwen) 0.6B / 1.7B | small | Built-in speakers, voice design from a description |

Plus a **Wyoming bridge** that puts any engine in front of Home Assistant or anything else that speaks
[Wyoming](https://github.com/OHF-Voice/wyoming).

Every engine loads its model on first use, unloads it when idle, and serves argus-cortex's resident contract
(`/health`, `/admin/{unload,evict,wake}`), so [argus-iris](https://github.com/smk762/argus-iris) can share a GPU
between them and other services. [argus-mask](https://github.com/smk762/argus-mask) uses them to give its faces a
voice.

Part of the [Argus suite](https://github.com/smk762/argus-cortex/blob/dev/SUITE.md).

## Run

Linux, Docker with Compose, an NVIDIA GPU and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Each engine is a Compose profile; run the ones your GPU can hold:

```sh
cp .env.example .env                                         # bind address, token, storage
docker compose --profile miso up -d --build                  # 24 GB card
docker compose --profile xtts --profile qwen up -d --build   # 8 GB card
```

Ports: Miso `8140`, XTTS `8141`, Qwen `8142`, Wyoming bridges `10240` (XTTS) and `10241` (Miso). They are
published on `ECHO_BIND`, which defaults to `127.0.0.1`; set it to `0.0.0.0` or a LAN address to serve other
machines. Miso downloads its ~33 GB checkpoint into `HF_HOME` on first use; the XTTS and Qwen images bake their
weights in at build time.

Check an engine (read-only; `--speak` also streams a sentence, which loads the model):

```sh
uvx argus-echo check http://localhost:8140
uvx argus-echo check http://localhost:8141 --speak "Hello there." --speaker "Claribel Dervla"
```

Or by hand:

```sh
curl -s localhost:8140/synthesise/stream -H 'content-type: application/json' \
     -d '{"text": "Hello there."}' --output hello.f32       # 24 kHz mono float32
```

## Storage

Generated WAVs and clones go to S3 (MinIO or any S3-compatible store): set `S3_ENDPOINT_URL` and the credentials
in `.env`. Each engine has **its own bucket** (`voice-miso`, `voice-xtts`, `voice-qwen` by default), because all
three store clones under the same keys in different formats. Create the buckets before first use. Miso can store
on local disk instead (`MISO_STORAGE_BACKEND=local`, the default) and serves those files itself.

## Security

`/admin/*` is closed until `ARGUS_ADMIN_TOKEN` is set (`openssl rand -hex 32`). **Every other route is open**,
clone creation and deletion included: keep the ports on a trusted network or behind an authenticating gateway.
Clone a real person's voice only with their consent. Details: [docs/api.md#security](docs/api.md#security).

## Documentation

| Doc | Contents |
| --- | --- |
| [docs/api.md](docs/api.md) | The shared contract, how the engines differ, health, admin, security |
| [docs/miso-api.md](docs/miso-api.md), [docs/miso-config.md](docs/miso-config.md) | Miso's full API and every `MISO_*` setting |
| [docs/xtts.md](docs/xtts.md) | XTTS-v2 settings and licence |
| [docs/qwen.md](docs/qwen.md) | Qwen3-TTS settings and differences |
| [docs/voice-cloning.md](docs/voice-cloning.md) | What makes a good reference on each engine; Home Assistant |

## Layout

```
compose.yaml          one profile per engine
gpu-tenant.yaml       the residents argus-iris manages
engines/<engine>/     each engine's server, Dockerfile and tests (its own Python environment)
src/argus_echo/       the contract in Python, and `argus-echo check`
```

The engines don't import `argus_echo`: their dependencies conflict (Miso pins torch 2.4), so they share the
contract, not code.

## Develop

```sh
make install && make test && make lint      # the argus_echo package
```

Each engine's tests run in its own environment, with the models stubbed (no GPU or weights). The commands are in
[.github/workflows/engines.yml](.github/workflows/engines.yml), for example:

```sh
cd engines/xtts && uv run --no-project --with "argus-cortex[server]>=0.8,<0.10" --with fastapi --with httpx \
  --with structlog --with boto3 --with numpy --with soundfile --with python-multipart --with pytest \
  python -m pytest -q tests
```

## Licence

MIT, see [LICENSE](LICENSE), except [engines/miso/](engines/miso/), which carries the upstream MisoTTS inference
code under Miso Labs' modified MIT licence ([engines/miso/LICENSE](engines/miso/LICENSE)). Model weights carry
their own licences: Miso TTS (modified MIT), XTTS-v2 (Coqui Public Model Licence, **non-commercial**), Qwen3-TTS
(Apache-2.0). Audio from Miso's `/synthesise` carries an imperceptible AI-generated watermark; streamed audio does
not.
