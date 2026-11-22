"""GET /health is argus-cortex's ResidentHealth (what argus-iris reads) and names the echo capabilities served.
No model loads: the lifespan does not run without `with TestClient(...)`."""

import unittest
from unittest import mock

from argus_cortex.gpu_wire import ResidentHealth
from fastapi.testclient import TestClient

import miso_server as srv


class HealthTest(unittest.TestCase):
    def test_health_is_resident_health_and_advertises_tts(self):
        r = TestClient(srv.app).get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        ResidentHealth.model_validate(body)
        self.assertEqual(body["capabilities"], ["tts"])
        self.assertEqual(body["image"], {"version": None, "revision": None})

    def test_health_reports_the_image_identity(self):
        with mock.patch.object(srv, "ECHO_VERSION", "0.2.0"), mock.patch.object(srv, "ECHO_REVISION", "abc123"):
            body = TestClient(srv.app).get("/health").json()
        self.assertEqual(body["image"], {"version": "0.2.0", "revision": "abc123"})


if __name__ == "__main__":
    unittest.main()
