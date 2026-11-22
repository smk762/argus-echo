"""GET /health is argus-cortex's ResidentHealth (what argus-iris reads) and names the echo capabilities served.

Torch is stubbed by test_tts_server's pattern; no model loads.
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.device = lambda d: d
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules["torch"] = _torch

import tts_server  # noqa: E402
from argus_cortex.gpu_wire import ResidentHealth  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class HealthTest(unittest.TestCase):
    def test_health_is_resident_health_and_advertises_tts(self):
        r = TestClient(tts_server.app).get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        ResidentHealth.model_validate(body)
        self.assertEqual(body["capabilities"], ["tts"])


if __name__ == "__main__":
    unittest.main()
