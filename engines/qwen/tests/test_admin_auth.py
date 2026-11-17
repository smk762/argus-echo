"""/admin/* auth: argus-cortex's bearer, closed while ARGUS_ADMIN_TOKEN is unset unless
ARGUS_ADMIN_INSECURE=1. The same rule as every argus-echo engine. torch is stubbed and the
model manager mocked, so nothing loads."""

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

from fastapi.testclient import TestClient  # noqa: E402

import tts_server  # noqa: E402

ROUTES = ("/admin/evict", "/admin/unload", "/admin/wake")


class AdminAuthTest(unittest.TestCase):
    def setUp(self):
        for name in ("demote_to_cpu", "unload", "ensure_gpu"):
            p = mock.patch.object(tts_server._mgr, name, mock.AsyncMock())
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(tts_server.app)

    def test_closed_when_token_unset(self):
        with mock.patch.dict("os.environ", {"ARGUS_ADMIN_TOKEN": "", "ARGUS_ADMIN_INSECURE": ""}):
            for route in ROUTES:
                self.assertEqual(self.client.post(route).status_code, 503, route)

    def test_insecure_opens_when_token_unset(self):
        with mock.patch.dict("os.environ", {"ARGUS_ADMIN_TOKEN": "", "ARGUS_ADMIN_INSECURE": "1"}):
            for route in ROUTES:
                self.assertEqual(self.client.post(route).status_code, 200, route)

    def test_token_required_when_set(self):
        with mock.patch.dict("os.environ", {"ARGUS_ADMIN_TOKEN": "s3cret", "ARGUS_ADMIN_INSECURE": "1"}):
            for route in ROUTES:
                self.assertEqual(self.client.post(route).status_code, 401, route)
                bad = {"Authorization": "Bearer wrong"}
                self.assertEqual(self.client.post(route, headers=bad).status_code, 401, route)
                ok = {"Authorization": "Bearer s3cret"}
                self.assertEqual(self.client.post(route, headers=ok).status_code, 200, route)


if __name__ == "__main__":
    unittest.main()
