"""LocalStorage: keys stay under the root, a failed write leaves no temp file,
a write survives a concurrent delete's pruning, and S3 has no default credentials."""

import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import miso_server as srv


class LocalStorageTest(unittest.TestCase):
    def setUp(self):
        self.outer = tempfile.TemporaryDirectory()
        self.addCleanup(self.outer.cleanup)
        self.root = os.path.join(self.outer.name, "data")
        self.storage = srv.LocalStorage(self.root, "http://x")
        with open(os.path.join(self.outer.name, "secret.txt"), "w") as f:
            f.write("secret")
        patcher = mock.patch.object(srv, "_storage", self.storage)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(srv.app)

    def _all_files(self):
        return [name for _d, _dirs, files in os.walk(self.root) for name in files]

    def test_keys_cannot_leave_the_root(self):
        for bad in ("../secret.txt", "a/../../secret.txt", "..", "a\x00b", "a\\..\\b"):
            with self.assertRaises(ValueError, msg=bad):
                self.storage._path(bad)
        self.assertEqual(self.storage._path("/etc/passwd"), os.path.join(self.root, "etc/passwd"))
        self.assertEqual(self.storage._path("a/../b.wav"), os.path.join(self.root, "b.wav"))

    def test_an_unnormalised_root_still_stores_and_confines(self):
        for root in (self.root + "/", os.path.join(self.root, ".", "")):
            storage = srv.LocalStorage(root, "http://x")
            storage.put("generated/a.wav", b"RIFF")
            self.assertEqual(storage.get("generated/a.wav"), b"RIFF")
            self.assertEqual(storage.list_keys("generated/"), ["generated/a.wav"])
            with self.assertRaises(ValueError):
                storage._path("../secret.txt")
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.outer.name)
        relative = srv.LocalStorage("./data/", "http://x")
        self.assertEqual(relative.get("generated/a.wav"), b"RIFF")

    def test_media_route_answers_404_for_traversal(self):
        self.storage.put("generated/a.wav", b"RIFF")
        self.assertEqual(self.client.get("/media/generated/a.wav").content, b"RIFF")
        for key in ("..%2Fsecret.txt", "%2E%2E%2Fsecret.txt", "generated%2F..%2F..%2Fsecret.txt", "%2Fetc%2Fpasswd"):
            r = self.client.get(f"/media/{key}")
            self.assertEqual(r.status_code, 404, key)
            self.assertNotIn(b"secret", r.content)

    def test_failed_replace_leaves_no_temp_file(self):
        os.makedirs(os.path.join(self.root, "voice_clones", "alice", "prompt.pt"))  # a directory in the way
        with self.assertRaises(OSError):
            self.storage.put("voice_clones/alice/prompt.pt", b"data")
        self.assertEqual(self._all_files(), [])

    def test_failed_write_leaves_no_temp_file(self):
        with mock.patch("os.replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.storage.put("generated/a.wav", b"data")
        self.assertEqual(self._all_files(), [])

    def test_put_survives_a_delete_pruning_its_directory(self):
        real_makedirs = os.makedirs
        pruned = []

        def makedirs_then_prune(path, *args, **kwargs):
            real_makedirs(path, *args, **kwargs)
            # os.makedirs calls itself for the parents; prune only the leaf, once,
            # as a concurrent delete() removes the still-empty directory.
            if not pruned and path.endswith("alice"):
                pruned.append(path)
                os.rmdir(path)

        with mock.patch("os.makedirs", side_effect=makedirs_then_prune):
            self.storage.put("voice_clones/alice/prompt.pt", b"data")
        self.assertEqual(len(pruned), 1)
        self.assertEqual(self.storage.get("voice_clones/alice/prompt.pt"), b"data")


class S3CredentialsTest(unittest.TestCase):
    def test_no_default_credentials(self):
        for env in ({}, {"S3_ACCESS_KEY_ID": "key"}, {"S3_ACCESS_KEY_ID": "", "S3_SECRET_ACCESS_KEY": "secret"}):
            with mock.patch.dict("os.environ", env):
                for name in ("S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"):
                    if name not in env:
                        os.environ.pop(name, None)
                with self.assertRaisesRegex(RuntimeError, "S3_ACCESS_KEY_ID"):
                    srv.S3Storage()

    def test_configured_credentials_are_used(self):
        with mock.patch.dict("os.environ", {"S3_ACCESS_KEY_ID": "key", "S3_SECRET_ACCESS_KEY": "secret"}):
            store = srv.S3Storage()
        self.assertEqual((store.access_key, store.secret_key), ("key", "secret"))


if __name__ == "__main__":
    unittest.main()
