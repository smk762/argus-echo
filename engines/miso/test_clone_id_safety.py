"""Clone storage safety: companion ids can't escape `voice_clones/{id}/`, and
stored prompt.pt blobs are never unpickled with code execution."""

import asyncio
import io
import os
import pickle
import tempfile
import unittest
from unittest import mock

import torch

import miso_server as srv


class _Boom:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def _prompt_bytes(**extra) -> bytes:
    buf = io.BytesIO()
    torch.save({"ref_audio": torch.zeros(10), "ref_text": "hi", "speaker": 0, **extra}, buf)
    return buf.getvalue()


class CloneIdSafetyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.storage = srv.LocalStorage(self.tmp.name, "http://x")
        for cid in ("alice", "bob"):
            self.storage.put(f"voice_clones/{cid}/prompt.pt", _prompt_bytes())
            self.storage.put(f"voice_clones/{cid}/meta.json", b'{"companion_id": "%s"}' % cid.encode())
        self.storage.put("media/keep.wav", b"x")
        patcher = mock.patch.object(srv, "_storage", self.storage)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        srv._clone_cache.clear()

    def test_companion_id_allowlist(self):
        for ok in ("alice", "A_b-9", "x" * 64):
            self.assertTrue(srv._is_valid_companion_id(ok), ok)
        for bad in ("", ".", "..", "a/b", "a\\b", "../x", "a.b", "x" * 65, None):
            self.assertFalse(srv._is_valid_companion_id(bad), bad)

    def test_clone_prompt_key_rejects_arbitrary_keys(self):
        self.assertEqual(srv._clone_prompt_key("alice"), "voice_clones/alice/prompt.pt")
        self.assertEqual(srv._clone_prompt_key("voice_clones/alice/prompt.pt"), "voice_clones/alice/prompt.pt")
        for bad in ("media/keep.wav", "voice_clones/../media/keep.wav", "voice_clones//prompt.pt", "..", "a b"):
            with self.assertRaises(ValueError, msg=bad):
                srv._clone_prompt_key(bad)

    def test_resolve_rejects_traversal_and_empty_segments(self):
        for bad in ("voice_clones/../x", "voice_clones//x", "voice_clones/", "voice_clones/..",
                    "voice_clones/nope", "voice_clones/alice/../bob"):
            self.assertIsNone(srv._resolve_clone_companion_id(bad), bad)
        self.assertEqual(srv._resolve_clone_companion_id("voice_clones/alice"), "alice")
        self.assertEqual(srv._resolve_clone_companion_id("voice_clones/alice/prompt.pt"), "alice")

    def test_delete_traversal_removes_nothing(self):
        for bad in ("voice_clones/../x", "voice_clones//x", "..", "voice_clones/.."):
            self.assertFalse(srv._delete_clone(bad), bad)
        self.assertTrue(self.storage.exists("voice_clones/alice/prompt.pt"))
        self.assertTrue(self.storage.exists("voice_clones/bob/prompt.pt"))
        self.assertTrue(self.storage.exists("media/keep.wav"))
        self.assertTrue(srv._delete_clone("alice"))
        self.assertFalse(self.storage.exists("voice_clones/alice/prompt.pt"))
        self.assertTrue(self.storage.exists("voice_clones/bob/prompt.pt"))

    def test_download_rejects_invalid_id(self):
        with self.assertRaises(ValueError):
            srv._build_clone_zip("../media")
        resp = asyncio.run(srv.download_clone("a.b"))
        self.assertEqual(resp.status_code, 422)

    def test_clone_voice_rejects_invalid_companion_id(self):
        for bad in ("../../media/x", "other/../victim", "", "a.b"):
            resp = asyncio.run(srv.clone_voice(audio=[mock.Mock()], reference_text="", companion_id=bad, voice_name=""))
            self.assertEqual(resp.status_code, 422, bad)

    def test_load_clone_uses_weights_only(self):
        self.storage.put("voice_clones/evil/prompt.pt", pickle.dumps({"ref_audio": _Boom()}))
        with mock.patch("os.system") as system:
            with self.assertRaises(ValueError):
                srv._load_clone("evil")
            system.assert_not_called()
        self.assertNotIn("voice_clones/evil/prompt.pt", srv._clone_cache)

    def test_load_clone_validates_structure(self):
        buf = io.BytesIO()
        torch.save({"ref_text": "no audio"}, buf)
        self.storage.put("voice_clones/bad/prompt.pt", buf.getvalue())
        with self.assertRaises(ValueError):
            srv._load_clone("bad")
        prompt = srv._load_clone("alice")
        self.assertTrue(torch.is_tensor(prompt["ref_audio"]))


if __name__ == "__main__":
    unittest.main()
