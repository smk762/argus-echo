"""tts_bridge: which voice ids are forwarded as clone keys, and the clone and
speaker caches. The sidecar is a fake httpx client; nothing listens.

Run from engines/wyoming/: python -m pytest tests -q
"""

import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tts_bridge  # noqa: E402
from wyoming.tts import Synthesize, SynthesizeVoice  # noqa: E402


class _Response:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _sidecar(responses: dict, calls: list, delay: float = 0.0):
    """An httpx.AsyncClient stand-in answering GET by path suffix."""

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url):
            calls.append(url)
            await asyncio.sleep(delay)
            body = responses[url.rsplit("/", 1)[-1]]
            if isinstance(body, Exception):
                raise body
            return _Response(body)

    return mock.patch.object(tts_bridge.httpx, "AsyncClient", Client)


class _CacheCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tts_bridge._CLONE_KEYS.clear()
        tts_bridge._SPEAKER_CACHE.clear()
        self.addCleanup(tts_bridge._CLONE_KEYS.clear)
        self.addCleanup(tts_bridge._SPEAKER_CACHE.clear)


class CloneKeyTests(_CacheCase):
    def test_bare_id_uses_the_well_known_layout(self):
        self.assertEqual(tts_bridge._clone_key(" alice "), "voice_clones/alice/prompt.pt")
        self.assertEqual(tts_bridge._clone_key("A_b-9"), "voice_clones/A_b-9/prompt.pt")

    def test_full_key_in_the_clone_layout_is_accepted(self):
        self.assertEqual(tts_bridge._clone_key("voice_clones/alice/prompt.pt"), "voice_clones/alice/prompt.pt")

    def test_arbitrary_keys_are_refused(self):
        for bad in (
            "generated/voice/x.wav",
            "voice_clones/../secrets/prompt.pt",
            "voice_clones/alice/../bob/prompt.pt",
            "voice_clones/alice/meta.json",
            "voice_clones/alice/extra/prompt.pt",
            "/etc/passwd",
            "a b",
            "..",
            "",
            "x" * 65,
        ):
            with self.assertRaises(ValueError, msg=bad):
                tts_bridge._clone_key(bad)

    def test_keys_the_sidecar_listed_are_accepted_as_given(self):
        tts_bridge._CLONE_KEYS["old voice"] = "legacy/old voice/prompt.pt"
        self.assertEqual(tts_bridge._clone_key("old voice"), "legacy/old voice/prompt.pt")
        self.assertEqual(tts_bridge._clone_key("legacy/old voice/prompt.pt"), "legacy/old voice/prompt.pt")

    async def test_refused_voice_gets_an_empty_clip_and_no_upstream_request(self):
        handler = tts_bridge.XttsEventHandler.__new__(tts_bridge.XttsEventHandler)
        handler.write_event = mock.AsyncMock()
        synth = Synthesize(text="hello", voice=SynthesizeVoice(name="clone:../../secrets/key.pt"))
        with mock.patch.object(tts_bridge.httpx, "AsyncClient") as upstream:
            await handler._synthesize(synth)
        upstream.assert_not_called()
        self.assertEqual([c.args[0].type for c in handler.write_event.call_args_list], ["audio-start", "audio-stop"])


class CloneCacheTests(_CacheCase):
    async def test_deleted_clone_leaves_the_key_cache(self):
        calls = []
        both = {"clones": [
            {"companion_id": "alice", "voice_clone_id": "voice_clones/alice/prompt.pt"},
            {"companion_id": "bob", "voice_clone_id": "voice_clones/bob/prompt.pt"},
        ]}
        with _sidecar({"clones": both}, calls):
            await tts_bridge._fetch_clones()
        self.assertEqual(set(tts_bridge._CLONE_KEYS), {"alice", "bob"})

        with _sidecar({"clones": {"clones": both["clones"][:1]}}, calls):
            listed = await tts_bridge._fetch_clones()
        self.assertEqual([c["companion_id"] for c in listed], ["alice"])
        self.assertEqual(tts_bridge._CLONE_KEYS, {"alice": "voice_clones/alice/prompt.pt"})

    async def test_failed_fetch_keeps_the_last_known_keys(self):
        tts_bridge._CLONE_KEYS["alice"] = "voice_clones/alice/prompt.pt"
        with _sidecar({"clones": RuntimeError("sidecar down")}, []):
            self.assertEqual(await tts_bridge._fetch_clones(), [])
        self.assertEqual(tts_bridge._CLONE_KEYS, {"alice": "voice_clones/alice/prompt.pt"})

    async def test_malformed_entries_are_skipped(self):
        body = {"clones": ["junk", None, {"companion_id": 7, "voice_clone_id": None}, {"companion_id": "a", "voice_clone_id": "k"}]}
        with _sidecar({"clones": body}, []):
            listed = await tts_bridge._fetch_clones()
        self.assertEqual(len(listed), 2)
        self.assertEqual(tts_bridge._CLONE_KEYS, {"a": "k"})


class SpeakerCacheTests(_CacheCase):
    async def test_concurrent_fills_fetch_once_and_do_not_duplicate(self):
        calls = []
        body = {"speakers": [{"name": "Ana"}, {"name": "Ben"}, {"id": "no-name"}]}
        with _sidecar({"speakers": body}, calls, delay=0.05):
            results = await asyncio.gather(*(tts_bridge._fetch_speakers() for _ in range(5)))
        self.assertEqual(tts_bridge._SPEAKER_CACHE, ["Ana", "Ben"])
        self.assertEqual([list(r) for r in results], [["Ana", "Ben"]] * 5)
        self.assertEqual(len(calls), 1)  # the model was woken once

    async def test_failed_fetch_is_not_cached(self):
        with _sidecar({"speakers": RuntimeError("loading")}, []):
            self.assertEqual(await tts_bridge._fetch_speakers(), [])
        self.assertEqual(tts_bridge._SPEAKER_CACHE, [])


if __name__ == "__main__":
    unittest.main()
