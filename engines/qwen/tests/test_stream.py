"""/synthesise/stream: one sentence at a time, errors before the 200, one lease per stream.

torch is stubbed and the models are fakes, so nothing loads. The app is driven over raw ASGI
(httpx's ASGITransport buffers the whole body), so a test can read the first sentence's PCM
while the second is still rendering, and can hang up mid-stream.
"""

import asyncio
import json
import os
import sys
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.device = lambda d: d
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules["torch"] = _torch

import numpy as np  # noqa: E402

import tts_server  # noqa: E402
from tts_server import TTSModelManager, TTSState  # noqa: E402

S1 = "The first sentence is long enough."
S2 = "Here is the second one, also long."
S3 = "And a third sentence ends the text."
TEXT = f"{S1} {S2} {S3}"


class SplitSentencesTests(unittest.TestCase):
    def test_splits_on_sentence_ends(self):
        self.assertEqual(tts_server._split_sentences(TEXT), [S1, S2, S3])

    def test_keeps_closing_quotes_and_brackets(self):
        text = 'He said "this is the first sentence." (And then the second one.) Then a third one follows here.'
        self.assertEqual(
            tts_server._split_sentences(text),
            ['He said "this is the first sentence."', "(And then the second one.)", "Then a third one follows here."],
        )

    def test_short_fragments_join_their_neighbours(self):
        self.assertEqual(
            tts_server._split_sentences("Hi! Dr. Smith will see you now, please come in."),
            ["Hi! Dr. Smith will see you now, please come in."],
        )
        # A short tail joins the sentence before it.
        self.assertEqual(tts_server._split_sentences(f"{S1} Bye."), [f"{S1} Bye."])

    def test_cjk_punctuation_and_blank_lines(self):
        a, b = "今天天气很好，我们一起去公园散步吧，好不好呀朋友们。", "明天可能会下雨，所以记得带上雨伞出门，别忘了哦朋友们！"
        self.assertEqual(tts_server._split_sentences(a + b), [a, b])
        self.assertEqual(tts_server._split_sentences(f"{S1[:-1]}\n\n{S2}"), [S1[:-1], S2])

    def test_no_terminator_is_one_sentence(self):
        self.assertEqual(tts_server._split_sentences("just some words"), ["just some words"])
        self.assertEqual(tts_server._split_sentences(""), [""])


class FakeCustom:
    """A CustomVoice stand-in: sentence n renders as 10 samples of value n."""

    def __init__(self, fail_on=None, gates=None):
        self.texts = []
        self.fail_on = fail_on or {}
        self.gates = gates or {}  # sentence index -> threading.Event to wait for

    def get_supported_speakers(self):
        return ["ryan"]

    def generate_custom_voice(self, **kw):
        self.texts.append(kw["text"])
        n = len(self.texts)
        gate = self.gates.get(n)
        if gate is not None:
            gate.wait(5)
        if n in self.fail_on:
            raise self.fail_on[n]
        return [np.full(10, n, dtype="<f4")], 24000


def _manager(custom) -> TTSModelManager:
    mgr = TTSModelManager()
    mgr.keep_alive_gpu = mgr.keep_alive_cpu = -1
    mgr._custom_model = custom
    mgr._state = TTSState.GPU
    return mgr


async def _post(body: dict, on_message, disconnect: asyncio.Event, spec_version: str = "2.4"):
    raw = json.dumps(body).encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": spec_version}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/synthesise/stream", "raw_path": b"/synthesise/stream",
        "query_string": b"", "root_path": "", "client": ("t", 1), "server": ("t", 80),
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())],
    }
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    await tts_server.app(scope, receive, on_message)


class Collector:
    def __init__(self):
        self.status = None
        self.headers = {}
        self.chunks = []
        self.ended = False

    async def __call__(self, message):
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.headers = {k.decode(): v.decode() for k, v in message["headers"]}
        elif message["type"] == "http.response.body":
            if message.get("body"):
                self.chunks.append(message["body"])
            if not message.get("more_body"):
                self.ended = True

    def json(self):
        return json.loads(b"".join(self.chunks))

    def samples(self):
        return [np.frombuffer(c, dtype="<f4") for c in self.chunks]


async def _until(cond, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not cond():
        if loop.time() > deadline:
            raise AssertionError("condition not met")
        await asyncio.sleep(0.005)


class StreamTests(unittest.IsolatedAsyncioTestCase):
    def _use(self, custom):
        mgr = _manager(custom)
        for name, value in (("_mgr", mgr), ("DEFAULT_SPEAKER", "")):
            p = mock.patch.object(tts_server, name, value)
            p.start()
            self.addCleanup(p.stop)
        return mgr

    async def test_each_sentence_is_its_own_chunk(self):
        custom = FakeCustom()
        mgr = self._use(custom)
        out = Collector()
        await _post({"text": TEXT, "speaker": "ryan"}, out, asyncio.Event())
        self.assertEqual(out.status, 200)
        self.assertEqual(
            (out.headers["x-sample-rate"], out.headers["x-audio-format"], out.headers["x-channels"]),
            ("24000", "pcm_f32le", "1"),
        )
        self.assertEqual(custom.texts, [S1, S2, S3])
        self.assertEqual([list(set(s.tolist())) for s in out.samples()], [[1.0], [2.0], [3.0]])
        self.assertTrue(out.ended)
        self.assertEqual(mgr._in_use, 0)

    async def test_first_sentence_is_sent_before_the_second_is_rendered(self):
        gate = threading.Event()
        custom = FakeCustom(gates={2: gate})
        mgr = self._use(custom)
        out = Collector()
        task = asyncio.ensure_future(_post({"text": TEXT, "speaker": "ryan"}, out, asyncio.Event()))
        try:
            await _until(lambda: len(out.chunks) == 1 and len(custom.texts) == 2)
            self.assertEqual(out.samples()[0][0], 1.0)  # sentence 1 is out; sentence 2 is still rendering
            self.assertEqual(mgr._in_use, 1)  # the stream holds the lease meanwhile
        finally:
            gate.set()
        await asyncio.wait_for(task, 2)
        self.assertEqual(len(out.chunks), 3)
        self.assertEqual(mgr._in_use, 0)

    async def test_errors_before_audio_are_json_and_release_the_lease(self):
        cases = (
            ({"text": TEXT, "speaker": "ryan", "language": "xx"}, FakeCustom(), 422),
            ({"text": TEXT}, FakeCustom(), 422),  # no speaker, clone or description
            ({"text": TEXT, "speaker": "ryan"}, FakeCustom(fail_on={1: RuntimeError("CUDA out of memory")}), 503),
            ({"text": TEXT, "speaker": "ryan"}, FakeCustom(fail_on={1: OSError("disk")}), 500),
        )
        for body, custom, status in cases:
            mgr = _manager(custom)
            with mock.patch.object(tts_server, "_mgr", mgr), mock.patch.object(tts_server, "DEFAULT_SPEAKER", ""), \
                    mock.patch.object(tts_server, "_gpu_memory_info", return_value=None):
                out = Collector()
                await _post(body, out, asyncio.Event())
            self.assertEqual(out.status, status, body)
            self.assertIn("error", out.json())
            self.assertEqual(mgr._in_use, 0, body)

    async def test_unknown_clone_is_422_before_the_stream(self):
        base = mock.Mock()
        mgr = _manager(FakeCustom())
        mgr._base_model = base
        with mock.patch.object(tts_server, "_mgr", mgr), \
                mock.patch.object(tts_server, "_get_clone_prompt", side_effect=ValueError("Voice clone not found")):
            out = Collector()
            await _post({"text": TEXT, "voice_clone_id": "voice_clones/nobody/prompt.pt"}, out, asyncio.Event())
        self.assertEqual(out.status, 422)
        base.generate_voice_clone.assert_not_called()
        self.assertEqual(mgr._in_use, 0)

    async def test_clone_stream_renders_each_sentence_with_the_prompt(self):
        prompt = object()
        base = mock.Mock()
        base.generate_voice_clone.side_effect = lambda text, **kw: ([np.zeros(4, dtype="<f4")], 24000)
        mgr = _manager(None)
        mgr._base_model = base
        with mock.patch.object(tts_server, "_mgr", mgr), \
                mock.patch.object(tts_server, "_get_clone_prompt", return_value=prompt) as load:
            out = Collector()
            await _post({"text": TEXT, "voice_clone_id": "voice_clones/a/prompt.pt"}, out, asyncio.Event())
        self.assertEqual(out.status, 200)
        load.assert_called_once_with("voice_clones/a/prompt.pt")  # once per stream, not per sentence
        self.assertEqual([c.args[0] for c in base.generate_voice_clone.call_args_list], [S1, S2, S3])
        self.assertTrue(all(c.kwargs["voice_clone_prompt"] is prompt for c in base.generate_voice_clone.call_args_list))
        self.assertEqual(len(out.chunks), 3)

    async def test_a_later_failure_ends_the_stream_early(self):
        custom = FakeCustom(fail_on={2: RuntimeError("boom")})
        mgr = self._use(custom)
        out = Collector()
        with mock.patch.object(tts_server, "logger") as log:
            await _post({"text": TEXT, "speaker": "ryan"}, out, asyncio.Event())
        self.assertEqual(out.status, 200)
        self.assertEqual(len(out.chunks), 1)
        self.assertEqual(custom.texts, [S1, S2])  # sentence 3 never started
        self.assertIn("tts_stream_failed", [c.args[0] for c in log.error.call_args_list])
        self.assertEqual(mgr._in_use, 0)

    async def test_disconnect_stops_rendering_and_keeps_the_lease_until_the_sentence_finishes(self):
        for spec_version in ("2.4", "2.3"):  # 2.3: Starlette's own listener cancels the response
            with self.subTest(spec_version=spec_version):
                gate = threading.Event()
                custom = FakeCustom(gates={2: gate})
                mgr = _manager(custom)
                with mock.patch.object(tts_server, "_mgr", mgr), mock.patch.object(tts_server, "DEFAULT_SPEAKER", ""):
                    out = Collector()
                    hangup = asyncio.Event()
                    task = asyncio.ensure_future(
                        _post({"text": TEXT, "speaker": "ryan"}, out, hangup, spec_version=spec_version)
                    )
                    await _until(lambda: len(custom.texts) == 2)
                    hangup.set()  # the client goes away while sentence 2 renders
                    await asyncio.wait_for(task, 2)
                    self.assertEqual(mgr._in_use, 1)  # sentence 2's thread still runs: lease kept
                    gate.set()
                    await _until(lambda: mgr._in_use == 0)
                    await asyncio.sleep(0.05)
                self.assertEqual(custom.texts, [S1, S2])  # sentence 3 was never rendered
                self.assertEqual(len(out.chunks), 1)

    async def test_a_send_failure_releases_the_lease(self):
        custom = FakeCustom()
        mgr = self._use(custom)
        out = Collector()

        async def send(message):
            await out(message)
            if len(out.chunks) == 1:
                raise OSError("connection reset")

        from starlette.requests import ClientDisconnect

        with self.assertRaises((OSError, ClientDisconnect)):
            await _post({"text": TEXT, "speaker": "ryan"}, send, asyncio.Event())
        self.assertEqual(custom.texts, [S1])
        self.assertEqual(mgr._in_use, 0)


if __name__ == "__main__":
    unittest.main()
