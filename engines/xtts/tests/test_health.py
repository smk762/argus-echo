"""GET /health is argus-cortex's ResidentHealth (what argus-iris reads) and names the echo capabilities served."""

import pytest
import xtts_server as xs
from argus_cortex.gpu_wire import ResidentHealth
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(xs, "_gpu_fault", None, raising=False)
    return TestClient(xs.app, raise_server_exceptions=False)


def test_health_is_resident_health_and_advertises_tts(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    ResidentHealth.model_validate(body)
    assert body["capabilities"] == ["tts"]
    assert body["engine"] == "xtts-v2"  # clients key on it (docs/api.md)
