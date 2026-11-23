"""GPU-lock hardening (#9): the lock is always released, waits are bounded, and
/health reports a lock held past any possible generation."""

import asyncio
import json
import threading
import time
import unittest
from unittest import mock

import numpy as np
import torch

import miso_server as srv

PREPARED = (0, [], 0.9, 50, 30000.0)  # speaker, context, temperature, topk, max_audio_ms
CHUNK = np.full(2400, 0.1, dtype=np.float32)


class _SlowGen:
    """generate_stream yields `n` chunks, `delay` s apart; generate sleeps `delay`."""

    def __init__(self, n=40, delay=0.05, release: threading.Event | None = None):
        self.n, self.delay, self.release = n, delay, release
        self.yielded = 0

    def _wait(self):
        if self.release is not None:
            self.release.wait(10)
        else:
            time.sleep(self.delay)

    def generate_stream(self, **kwargs):
        for _ in range(self.n):
            self._wait()
            self.yielded += 1
            yield torch.from_numpy(CHUNK)

    def generate(self, **kwargs):
        self._wait()
        return torch.from_numpy(CHUNK)


class _Exited(Exception):
    """Stands in for os._exit (SystemExit would escape the test's event loop)."""


def _scope(path):
    return {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "POST", "path": path, "raw_path": path.encode(), "query_string": b"",
        "headers": [(b"content-type", b"application/json")], "client": ("test", 1),
        "server": ("test", 80), "scheme": "http", "root_path": "",
    }


class GenLockTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mgr = srv.MisoModelManager()
        self.mgr.ensure_ready = mock.AsyncMock()
        patches = [
            mock.patch.object(srv, "_mgr", self.mgr),
            mock.patch.object(srv, "_prepare_generation", lambda req: PREPARED),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    async def _until_released(self, timeout=5.0):
        end = time.monotonic() + timeout
        while self.mgr.gen_lock.locked() and time.monotonic() < end:
            await asyncio.sleep(0.02)
        return not self.mgr.gen_lock.locked()

    async def test_client_disconnect_mid_stream_releases_lock(self):
        """The incident: a disconnect cancels the stream inside an anyio cancel scope."""
        gen = _SlowGen()
        self.mgr._generator = gen
        got_chunk = asyncio.Event()
        msgs = [{"type": "http.request", "body": b'{"text": "hello"}', "more_body": False}]

        async def receive():
            if msgs:
                return msgs.pop(0)
            await got_chunk.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                got_chunk.set()

        await asyncio.wait_for(srv.app(_scope("/synthesise/stream"), receive, send), 10)
        self.assertTrue(await self._until_released())
        self.assertIsNone(self.mgr.busy_info())
        self.assertLess(gen.yielded, gen.n)  # generation stopped early, not run to the end

    async def test_stream_never_read_releases_lock_after_stall(self):
        self.mgr._generator = _SlowGen(n=100, delay=0.0)
        with mock.patch.object(srv, "STREAM_STALL_S", 0.3):
            resp = await srv.synthesise_stream(srv.SynthesiseRequest(text="hello"))
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(self.mgr.gen_lock.locked())
            with mock.patch.object(srv, "logger") as log:
                self.assertTrue(await self._until_released())
        self.assertIn("miso_stream_stalled", [c.args[0] for c in log.warning.call_args_list])

    async def test_stream_deadline_ends_the_stream_and_releases(self):
        self.mgr._generator = _SlowGen(n=1000, delay=0.02)
        with mock.patch.object(srv, "GEN_TIMEOUT_PAD_S", 0.2), mock.patch.object(
            srv, "GEN_TIMEOUT_FACTOR", 0.0
        ), mock.patch.object(srv, "logger") as log:
            resp = await srv.synthesise_stream(srv.SynthesiseRequest(text="hello"))
            chunks = [c async for c in resp.body_iterator]
            self.assertTrue(await self._until_released())
        self.assertGreater(len(chunks), 0)
        self.assertLess(len(chunks), 1000)
        self.assertIn("miso_generation_timeout", [c.args[0] for c in log.error.call_args_list])

    async def test_stream_with_no_chunks_times_out_for_the_client(self):
        release = threading.Event()
        self.mgr._generator = _SlowGen(n=1, release=release)
        with mock.patch.object(srv, "STREAM_CHUNK_TIMEOUT_S", 0.3):
            resp = await srv.synthesise_stream(srv.SynthesiseRequest(text="hello"))
            t0 = time.monotonic()
            chunks = [c async for c in resp.body_iterator]
        self.assertEqual(chunks, [])
        self.assertLess(time.monotonic() - t0, 3)
        release.set()  # the "stuck" model returns; the lock follows
        self.assertTrue(await self._until_released())

    async def test_busy_gpu_answers_503_and_health_reports_holder(self):
        release = threading.Event()
        self.mgr._generator = _SlowGen(n=1, release=release)
        first = asyncio.create_task(srv.synthesise(srv.SynthesiseRequest(text="first")))
        while not self.mgr.gen_lock.locked():
            await asyncio.sleep(0.01)

        with mock.patch.object(srv, "LOCK_WAIT_S", 0.2):
            for handler in (srv.synthesise, srv.synthesise_stream):
                resp = await handler(srv.SynthesiseRequest(text="second"))
                self.assertEqual(resp.status_code, 503)
                body = json.loads(resp.body)
                self.assertEqual(body["holder"]["endpoint"], "synthesise")
                self.assertEqual(resp.headers["retry-after"], "5")

        self.assertIsNone(await srv._health_unready())
        health = await srv._health_extra()
        self.assertIs(health["busy"], True)
        self.assertEqual(health["active"]["endpoint"], "synthesise")

        with mock.patch.object(srv, "STUCK_AFTER_S", 0.0):
            await asyncio.sleep(0.15)
            stuck = await srv._health_unready()  # what GET /health answers with 503 (test_health.py)
        self.assertEqual(stuck["status"], "stuck")
        self.assertEqual(stuck["active"]["endpoint"], "synthesise")

        release.set()
        with mock.patch.object(srv._storage, "put", return_value="http://x/a.wav"):
            self.assertIn("audio_url", await first)
        self.assertTrue(await self._until_released())
        idle = await srv._health_extra()
        self.assertIs(idle["busy"], False)
        self.assertIsNone(idle["active"])

    async def test_synthesise_timeout_is_504_and_lock_follows_the_thread(self):
        release = threading.Event()
        self.mgr._generator = _SlowGen(n=1, release=release)
        with mock.patch.object(srv, "GEN_TIMEOUT_PAD_S", 0.2), mock.patch.object(srv, "GEN_TIMEOUT_FACTOR", 0.0):
            resp = await srv.synthesise(srv.SynthesiseRequest(text="hello"))
        self.assertEqual(resp.status_code, 504)
        # The thread is still running, so the GPU is still in use.
        self.assertTrue(self.mgr.gen_lock.locked())
        release.set()
        self.assertTrue(await self._until_released())

    async def test_cancelling_run_locked_keeps_the_lock_until_the_thread_returns(self):
        """wait_for / a disconnect cancels the caller's future; the thread still owns the GPU."""
        release = threading.Event()
        self.addCleanup(release.set)  # don't leave the worker thread waiting if an assertion fails
        self.mgr._generator = _SlowGen(n=1, release=release)
        self.assertTrue(await self.mgr.acquire_gen("r1", "synthesise"))
        fut = self.mgr.run_locked(release.wait, 10)
        await asyncio.sleep(0.05)
        fut.cancel()
        await asyncio.sleep(0.05)
        self.assertTrue(self.mgr.gen_lock.locked())  # the thread is still running
        self.assertIsNotNone(self.mgr.busy_info())
        release.set()
        self.assertTrue(await self._until_released())

    async def test_cancelled_waiter_does_not_take_the_lock(self):
        await self.mgr.gen_lock.acquire()
        waiter = asyncio.create_task(self.mgr.acquire_gen("r2", "synthesise"))
        await asyncio.sleep(0.05)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.mgr.gen_lock.release()
        await asyncio.sleep(0.05)
        self.assertFalse(self.mgr.gen_lock.locked())

    async def test_eviction_skips_while_generating(self):
        await self.mgr.gen_lock.acquire()
        self.mgr.demote_to_cpu = mock.AsyncMock()
        with mock.patch.object(srv, "KEEP_ALIVE_GPU", 0.01), mock.patch.object(srv, "KEEP_ALIVE_CPU", 0.0):
            await self.mgr._evict_loop()
            self.mgr.demote_to_cpu.assert_not_called()
            if self.mgr._evict_task:
                self.mgr._evict_task.cancel()
        self.mgr.gen_lock.release()

    async def test_watchdog_exits_when_stuck(self):
        self.mgr._busy_since = time.monotonic() - 10
        self.mgr._busy_request, self.mgr._busy_endpoint = "r1", "stream"
        with mock.patch.object(srv, "STUCK_AFTER_S", 1.0), mock.patch.object(
            srv, "STUCK_CHECK_S", 0.01
        ), mock.patch.object(srv.os, "_exit", side_effect=_Exited) as exit_:
            with self.assertRaises(_Exited):
                await asyncio.wait_for(srv._stuck_watchdog(), 2)
        exit_.assert_called_once_with(70)


if __name__ == "__main__":
    unittest.main()
