"""GET /health is argus-cortex's ServiceHealth and ResidentHealth (what argus-iris reads) in one body, names the echo
capabilities served, and keeps every field it had before ServiceHealth. The loading 503 is Qwen's own body.

Torch is stubbed by test_tts_server's pattern; no model loads.
"""

import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.device = lambda d: d
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules["torch"] = _torch

import tts_server  # noqa: E402
from argus_cortex.gpu_wire import ResidentHealth  # noqa: E402
from argus_cortex.service import ServiceHealth  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# What /health returned before it was a ServiceHealth: clients (argus-iris, argus-proof, `argus-echo check`) read these.
FIELDS_BEFORE_SERVICE_HEALTH = {
    "status", "engine", "capabilities", "model", "model_size", "device", "load_models", "default_speaker",
    "model_state", "busy", "idle_s", "keep_alive_gpu", "keep_alive_cpu", "gpu", "weights", "image",
}  # fmt: skip


class HealthTest(unittest.TestCase):
    def test_health_is_resident_health_and_advertises_tts(self):
        r = TestClient(tts_server.app).get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        ResidentHealth.model_validate(body)
        self.assertEqual(body["capabilities"], ["tts"])
        self.assertEqual(body["engine"], "qwen-tts")  # clients key on it (docs/api.md)

    def test_health_is_service_health_and_keeps_every_earlier_field(self):
        body = TestClient(tts_server.app).get("/health").json()
        health = ServiceHealth.model_validate(body)
        self.assertEqual((health.service, health.status, health.capability_names), ("argus-echo", "ok", ("tts",)))
        self.assertIs(body["checks"]["model"]["ok"], True)
        self.assertLessEqual(FIELDS_BEFORE_SERVICE_HEALTH, set(body))
        self.assertEqual(body["model_state"], "unloaded")  # ResidentHealth's fields stay top-level
        self.assertIs(body["busy"], False)

    def test_loading_is_still_503_with_the_body_it_always_had(self):
        with mock.patch.object(tts_server._mgr, "_state", tts_server.TTSState.LOADING):
            r = TestClient(tts_server.app).get("/health")
        self.assertEqual((r.status_code, r.json()), (503, {"status": "loading"}))

    def test_a_failed_load_is_degraded_and_still_200(self):
        with mock.patch.object(tts_server._mgr, "_state", tts_server.TTSState.ERROR):
            r = TestClient(tts_server.app).get("/health")
        self.assertEqual(r.status_code, 200)  # argus-iris raises on anything else
        body = r.json()
        self.assertEqual((body["status"], body["model_state"]), ("degraded", "error"))
        self.assertIs(body["checks"]["model"]["ok"], False)


if __name__ == "__main__":
    unittest.main()
