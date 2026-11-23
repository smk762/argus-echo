"""GET /health is argus-cortex's ServiceHealth and ResidentHealth (what argus-iris reads) in one body, names the echo
capabilities served, and keeps every field it had before ServiceHealth. The 503s are XTTS's own bodies."""

import pytest
import xtts_server as xs
from argus_cortex.gpu_wire import ResidentHealth
from argus_cortex.service import ServiceHealth
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


# What /health returned before it was a ServiceHealth: clients (argus-iris, argus-proof, `argus-echo check`) read these.
FIELDS_BEFORE_SERVICE_HEALTH = {
    "status", "engine", "capabilities", "model", "device", "model_state", "busy", "idle_s", "keep_alive_gpu",
    "keep_alive_cpu", "gpu", "weights", "image",
}  # fmt: skip


def test_health_is_service_health_and_keeps_every_earlier_field(client):
    body = client.get("/health").json()
    health = ServiceHealth.model_validate(body)
    assert (health.service, health.status, health.capability_names) == ("argus-echo", "ok", ("tts",))
    assert body["checks"]["model"]["ok"] is True
    assert FIELDS_BEFORE_SERVICE_HEALTH <= set(body)
    assert body["model_state"] == "unloaded" and body["busy"] is False  # ResidentHealth's fields stay top-level


def test_loading_is_still_503_with_the_body_it_always_had(client, monkeypatch):
    monkeypatch.setattr(xs._mgr, "_state", xs.TTSState.LOADING)
    r = client.get("/health")
    assert (r.status_code, r.json()) == (503, {"status": "loading"})


def test_a_gpu_fault_is_still_503_gpu_error_with_every_field(client, monkeypatch):
    monkeypatch.setattr(xs, "_gpu_fault", "CUDA error: device-side assert triggered")
    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert (body["status"], body["error"]) == ("gpu_error", "CUDA error: device-side assert triggered")  # argus-proof
    assert FIELDS_BEFORE_SERVICE_HEALTH <= set(body)


def test_a_failed_load_is_degraded_and_still_200(client, monkeypatch):
    monkeypatch.setattr(xs._mgr, "_state", xs.TTSState.ERROR)
    r = client.get("/health")
    assert r.status_code == 200  # argus-iris raises on anything else
    body = r.json()
    assert (body["status"], body["model_state"]) == ("degraded", "error")
    assert body["checks"]["model"]["ok"] is False
