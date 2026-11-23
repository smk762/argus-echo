"""With ARGUS_GPU_BROKER_URL set, the lifespan registers gpu-tenant.yaml with argus-iris and signs off on the way
out; unset, it sends nothing. No model loads (STT_PRELOAD is off)."""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx  # noqa: E402
import stt_server as SERVER  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RESIDENT = "whisper"


def _lifespan_requests(monkeypatch, **env):
    """Run the app's lifespan with *env* and a fake broker; return the (method, path, headers, body) it received."""
    for key in ("ARGUS_GPU_COORDINATOR", "ARGUS_GPU_TENANT", "ARGUS_GPU_TENANT_MANIFEST", "ARGUS_GPU_BROKER_URL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(REPO)  # where the manifest is found, as /app/gpu-tenant.yaml is in the image
    seen = []

    def broker(request):
        seen.append((request.method, request.url.path, request.headers, request.content))
        return httpx.Response(200, json={})

    real = SERVER.coordinator_from_env

    def with_fake_broker():
        coordinator = real()
        coordinator._async_transport = httpx.MockTransport(broker)  # the only way in that coordinator_from_env leaves
        return coordinator

    monkeypatch.setattr(SERVER, "coordinator_from_env", with_fake_broker)

    async def run():
        async with SERVER.lifespan(SERVER.app):
            for _ in range(200):
                if seen:
                    break
                await asyncio.sleep(0.01)

    asyncio.run(run())
    return seen


def test_the_lifespan_registers_the_manifest_with_the_broker_and_signs_off(monkeypatch):
    seen = _lifespan_requests(
        monkeypatch, ARGUS_GPU_BROKER_URL="http://iris.test:8120", ARGUS_GPU_BROKER_TOKEN="bt", ARGUS_ADMIN_TOKEN="at"
    )
    assert [(method, path) for method, path, *_ in seen] == [("POST", "/tenants"), ("DELETE", "/tenants/argus-echo")]
    _, _, headers, content = seen[0]
    manifest = json.loads(content)
    assert manifest["tenant"] == "argus-echo"
    assert RESIDENT in {r["name"] for r in manifest["residents"]}
    assert headers["authorization"] == "Bearer bt"
    assert headers["x-argus-admin-token"] == "at"  # iris calls /admin/* with it


def test_the_lifespan_registers_nothing_without_a_broker_url(monkeypatch):
    assert _lifespan_requests(monkeypatch, ARGUS_ADMIN_TOKEN="at") == []
