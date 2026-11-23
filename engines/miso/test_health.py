"""GET /health is argus-cortex's ServiceHealth and ResidentHealth (what argus-iris reads) in one body, names the
echo capabilities served, and keeps every field it had before ServiceHealth. The 503s are Miso's own bodies.
No model loads: the lifespan does not run without `with TestClient(...)`."""

import unittest
from unittest import mock

from argus_cortex.gpu_wire import ResidentHealth
from argus_cortex.service import ServiceHealth
from fastapi.testclient import TestClient

import miso_server as srv

# What /health returned before it was a ServiceHealth: clients (argus-iris, argus-proof, `argus-echo check`) read these.
FIELDS_BEFORE_SERVICE_HEALTH = {
    "status", "busy", "active", "engine", "capabilities", "model", "device", "model_state", "sample_rate",
    "storage", "keep_alive_gpu", "keep_alive_cpu", "idle_s", "error", "rss_mb", "gpu", "image",
}  # fmt: skip


class HealthTest(unittest.TestCase):
    def test_health_is_resident_health_and_advertises_tts(self):
        r = TestClient(srv.app).get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        ResidentHealth.model_validate(body)
        self.assertEqual(body["capabilities"], ["tts"])
        self.assertEqual(body["engine"], "miso")  # clients key on it (docs/api.md)
        # image also carries runtime, checked below; a local build has no release identity.
        self.assertEqual((body["image"]["version"], body["image"]["revision"]), (None, None))

    def test_health_is_service_health_and_keeps_every_earlier_field(self):
        body = TestClient(srv.app).get("/health").json()
        health = ServiceHealth.model_validate(body)
        self.assertEqual((health.service, health.status, health.capability_names), ("argus-echo", "ok", ("tts",)))
        self.assertEqual(body["checks"]["model"]["ok"], True)
        self.assertLessEqual(FIELDS_BEFORE_SERVICE_HEALTH, set(body))
        self.assertEqual(body["model_state"], "unloaded")  # ResidentHealth's fields stay top-level
        self.assertIs(body["busy"], False)

    def test_loading_is_still_503_with_the_body_it_always_had(self):
        with mock.patch.object(srv._mgr, "_state", srv.TTSState.LOADING):
            r = TestClient(srv.app).get("/health")
        self.assertEqual((r.status_code, r.json()), (503, {"status": "loading"}))

    def test_a_stuck_generation_is_still_503_stuck(self):
        busy = {"endpoint": "synthesise", "busy_s": 999.0}
        with mock.patch.object(srv._mgr, "busy_info", return_value=busy), mock.patch.object(srv, "STUCK_AFTER_S", 1.0):
            r = TestClient(srv.app).get("/health")
        self.assertEqual(r.status_code, 503)
        body = r.json()
        self.assertEqual((body["status"], body["stuck_after_s"], body["active"]), ("stuck", 1.0, busy))
        self.assertLessEqual(FIELDS_BEFORE_SERVICE_HEALTH, set(body))

    def test_a_failed_load_is_degraded_and_still_200(self):
        with mock.patch.object(srv._mgr, "_state", srv.TTSState.ERROR):
            r = TestClient(srv.app).get("/health")
        self.assertEqual(r.status_code, 200)  # argus-iris raises on anything else
        body = r.json()
        self.assertEqual((body["status"], body["model_state"]), ("degraded", "error"))
        self.assertEqual(body["checks"]["model"]["ok"], False)

    def test_health_reports_the_image_identity(self):
        with mock.patch.object(srv, "ECHO_VERSION", "0.2.0"), mock.patch.object(srv, "ECHO_REVISION", "abc123"):
            body = TestClient(srv.app).get("/health").json()
        self.assertEqual((body["image"]["version"], body["image"]["revision"]), ("0.2.0", "abc123"))

    def test_health_reports_the_runtime_libraries(self):
        """#32: the library versions that shape the audio (null for one that isn't installed)."""
        runtime = TestClient(srv.app).get("/health").json()["image"]["runtime"]
        expected = {"python", "torch", "cuda", "cudnn", "moshi", "transformers", "torchaudio", "numpy"}
        self.assertEqual(set(runtime), expected)
        self.assertTrue(runtime["python"])


if __name__ == "__main__":
    unittest.main()
