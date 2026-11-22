"""GET /health is argus-cortex's ResidentHealth (what argus-iris reads) and names the echo capabilities served.
No model loads: the lifespan does not run without `with TestClient(...)`."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
