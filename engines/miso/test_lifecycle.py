"""Idle lifecycle: gpu → cpu → unloaded, with 0 skipping the
CPU stage and -1 pinning a stage, no eviction under an in-flight request, and
no partial copy kept after a failed (OOM) load."""

import asyncio
import json
import time
import unittest
from unittest import mock

import miso_server as srv


class _FakeGen:
    sample_rate = 24000


class LifecycleTest(unittest.IsolatedAsyncioTestCase):
    def _configure(self, gpu, cpu, device="cuda", load=None):
        self.moves: list[str] = []
        self.loads = 0

        def _load(mgr):
            self.loads += 1
            if load is not None:
                load()
            mgr._generator = _FakeGen()

        def _move(mgr, dev):
            self.moves.append(dev)

        patches = [
            mock.patch.object(srv, "KEEP_ALIVE_GPU", gpu),
            mock.patch.object(srv, "KEEP_ALIVE_CPU", cpu),
            mock.patch.object(srv, "DEVICE", device),
            mock.patch.object(srv.MisoModelManager, "_load", _load),
            mock.patch.object(srv.MisoModelManager, "_move", _move),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.mgr = srv.MisoModelManager()
        self.addCleanup(self._cancel_evict)

    def _cancel_evict(self):
        if self.mgr._evict_task:
            self.mgr._evict_task.cancel()

    async def _until(self, pred, timeout=2.0):
        end = time.monotonic() + timeout
        while not pred() and time.monotonic() < end:
            await asyncio.sleep(0.01)
        return pred()

    async def test_failed_demotion_drops_the_model(self):
        """A move that fails half-way may leave weights on both devices: the
        state must not stay "gpu" with that generator."""
        self._configure(gpu=0.05, cpu=10)

        def _move(mgr, dev):
            self.moves.append(dev)
            raise RuntimeError("cannot allocate memory")

        with mock.patch.object(srv.MisoModelManager, "_move", _move), mock.patch.object(srv, "logger") as log:
            await self.mgr.ensure_ready()
            self.assertTrue(await self._until(lambda: self.mgr.state == srv.TTSState.UNLOADED))
            self.assertEqual(self.moves, ["cpu"])
            self.assertIsNone(self.mgr.generator)
            self.assertFalse(self.mgr._ready_for_generation())
            self.assertIn("cannot allocate memory", self.mgr.error)
            self.assertIn("miso_demote_failed", [c.args[0] for c in log.error.call_args_list])
        # The next request starts over with a fresh load.
        await self.mgr.ensure_ready()
        self.assertEqual(self.mgr.state, srv.TTSState.GPU)
        self.assertEqual(self.loads, 2)

    async def test_evict_loop_logs_a_failed_transition(self):
        self._configure(gpu=0.05, cpu=10)
        with mock.patch.object(
            srv.MisoModelManager, "demote_to_cpu", mock.AsyncMock(side_effect=RuntimeError("boom"))
        ), mock.patch.object(srv, "logger") as log:
            await self.mgr.ensure_ready()
            task = self.mgr._evict_task
            await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            self.assertIsNone(task.exception())
            self.assertIn("miso_evict_failed", [c.args[0] for c in log.error.call_args_list])

    async def test_cpu_zero_skips_cpu_stage_and_unloads(self):
        self._configure(gpu=0.05, cpu=0)
        await self.mgr.ensure_ready()
        self.assertEqual(self.mgr.state, srv.TTSState.GPU)
        self.assertTrue(await self._until(lambda: self.mgr.state == srv.TTSState.UNLOADED))
        self.assertIsNone(self.mgr.generator)
        self.assertNotIn("cpu", self.moves)  # never parked on CPU

    async def test_gpu_then_cpu_then_unload(self):
        self._configure(gpu=0.05, cpu=0.1)
        await self.mgr.ensure_ready()
        self.assertTrue(await self._until(lambda: self.mgr.state == srv.TTSState.CPU))
        self.assertEqual(self.moves, ["cpu"])
        self.assertTrue(await self._until(lambda: self.mgr.state == srv.TTSState.UNLOADED))
        self.assertIsNone(self.mgr.generator)

    async def test_cpu_minus_one_parks_on_cpu_for_ever(self):
        self._configure(gpu=0.05, cpu=-1)
        await self.mgr.ensure_ready()
        self.assertTrue(await self._until(lambda: self.mgr.state == srv.TTSState.CPU))
        await asyncio.sleep(0.2)
        self.assertEqual(self.mgr.state, srv.TTSState.CPU)
        self.assertIsNotNone(self.mgr.generator)

    async def test_gpu_minus_one_stays_on_gpu(self):
        self._configure(gpu=-1, cpu=0)
        await self.mgr.ensure_ready()
        await asyncio.sleep(0.1)
        self.assertEqual(self.mgr.state, srv.TTSState.GPU)
        self.assertIsNone(self.mgr._evict_task)

    async def test_promote_from_cpu_reloads_nothing(self):
        self._configure(gpu=0.05, cpu=-1)
        await self.mgr.ensure_ready()
        self.assertTrue(await self._until(lambda: self.mgr.state == srv.TTSState.CPU))
        await self.mgr.ensure_ready()
        self.assertEqual(self.mgr.state, srv.TTSState.GPU)
        self.assertEqual(self.moves, ["cpu", "cuda"])
        self.assertEqual(self.loads, 1)

    async def test_hold_blocks_eviction_until_released(self):
        """The window between ensure_ready() and run_locked() (pack daef21e7dd)."""
        self._configure(gpu=0, cpu=0)
        async with self.mgr.hold():
            await self.mgr.ensure_ready()
            await asyncio.sleep(0.1)  # with keep-alive 0 the evict loop already ran
            self.assertEqual(self.mgr.state, srv.TTSState.GPU)
            self.assertIsNotNone(self.mgr.generator)
        self.assertTrue(await self._until(lambda: self.mgr.state == srv.TTSState.UNLOADED))

    async def test_idle_transition_rechecks_under_load_lock(self):
        """A request that enters hold() after the evict loop's check still wins."""
        self._configure(gpu=-1, cpu=0)
        await self.mgr.ensure_ready()
        async with self.mgr.hold():
            await self.mgr.unload(if_idle=True)
            await self.mgr.demote_to_cpu(if_idle=True)
            self.assertEqual(self.mgr.state, srv.TTSState.GPU)
            self.assertIsNotNone(self.mgr.generator)

    async def test_gen_lock_blocks_eviction_and_release_rearms(self):
        self._configure(gpu=0, cpu=0)
        await self.mgr.ensure_ready()
        # ensure_ready armed an immediate evict; take the lock before it runs.
        self.assertTrue(await self.mgr.acquire_gen("r1", "test"))
        await asyncio.sleep(0.1)
        self.assertEqual(self.mgr.state, srv.TTSState.GPU)
        await self.mgr.run_locked(lambda: None)
        self.assertTrue(await self._until(lambda: self.mgr.state == srv.TTSState.UNLOADED))

    async def test_acquire_after_unload_refuses(self):
        """An /admin/unload that lands after ensure_ready() must not hand out a None generator."""
        self._configure(gpu=-1, cpu=0)
        await self.mgr.ensure_ready()
        await self.mgr.unload()
        self.assertFalse(await self.mgr.acquire_gen("r1", "test"))
        self.assertFalse(self.mgr.gen_lock.locked())

    async def test_acquire_after_demote_refuses_cpu_generation(self):
        self._configure(gpu=-1, cpu=-1)
        await self.mgr.ensure_ready()
        await self.mgr.demote_to_cpu()
        self.assertFalse(await self.mgr.acquire_gen("r1", "test"))
        self.assertFalse(self.mgr.gen_lock.locked())

    async def test_cpu_device_unloads_after_both_keep_alives(self):
        self._configure(gpu=0.05, cpu=0.05, device="cpu")
        await self.mgr.ensure_ready()
        self.assertEqual(self.mgr.state, srv.TTSState.CPU)
        self.assertTrue(await self._until(lambda: self.mgr.state == srv.TTSState.UNLOADED))
        self.assertEqual(self.moves, [])

    async def test_oom_on_load_keeps_nothing(self):
        import torch

        def _oom():
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

        self._configure(gpu=-1, cpu=0, load=_oom)
        with self.assertRaises(srv.MisoLoadError) as ctx:
            await self.mgr.ensure_ready()
        self.assertTrue(ctx.exception.oom)
        self.assertIsNone(ctx.exception.__context__)  # nothing pins the failed load's frames
        self.assertEqual(self.mgr.state, srv.TTSState.UNLOADED)
        self.assertIsNone(self.mgr.generator)

    async def test_non_oom_load_failure_is_error_state(self):
        def _boom():
            raise FileNotFoundError("no weights")

        self._configure(gpu=-1, cpu=0, load=_boom)
        with self.assertRaises(srv.MisoLoadError) as ctx:
            await self.mgr.ensure_ready()
        self.assertFalse(ctx.exception.oom)
        self.assertEqual(self.mgr.state, srv.TTSState.ERROR)

    async def test_oom_promotion_unloads_instead_of_splitting(self):
        import torch

        self._configure(gpu=0.05, cpu=-1)
        await self.mgr.ensure_ready()
        self.assertTrue(await self._until(lambda: self.mgr.state == srv.TTSState.CPU))

        def _move_oom(mgr, dev):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        with mock.patch.object(srv.MisoModelManager, "_move", _move_oom):
            with self.assertRaises(srv.MisoLoadError):
                await self.mgr.ensure_ready()
        self.assertEqual(self.mgr.state, srv.TTSState.UNLOADED)
        self.assertIsNone(self.mgr.generator)

    async def test_synthesise_oom_load_is_503_with_retry_after(self):
        mgr = srv.MisoModelManager()
        mgr.ensure_ready = mock.AsyncMock(side_effect=srv.MisoLoadError("CUDA out of memory", oom=True))
        with mock.patch.object(srv, "_mgr", mgr):
            resp = await srv.synthesise(srv.SynthesiseRequest(text="hello"))
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.headers["retry-after"], str(srv.OOM_RETRY_AFTER_S))
        self.assertIn("OOM", json.loads(resp.body)["error"])
        self.assertEqual(mgr._holds, 0)

    async def test_health_reports_lifecycle(self):
        self._configure(gpu=-1, cpu=0)
        with mock.patch.object(srv, "_mgr", self.mgr):
            body = await srv.health()
        self.assertEqual(body["model_state"], "unloaded")
        self.assertIn("rss_mb", body)
        self.assertEqual(body["keep_alive_gpu"], -1)


if __name__ == "__main__":
    unittest.main()


class CudaCacheReleaseTest(unittest.TestCase):
    """smk762/voluptas#26: cuBLAS's workspace pins a load-time segment, so unload must
    clear the workspaces before empty_cache() or ~2 GB stays reserved."""

    def test_clears_cublas_workspaces_before_empty_cache(self):
        import torch

        calls: list[str] = []
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "synchronize", lambda: calls.append("sync")), \
             mock.patch.object(torch._C, "_cuda_clearCublasWorkspaces", lambda: calls.append("clear"), create=True), \
             mock.patch.object(torch.cuda, "empty_cache", lambda: calls.append("empty")):
            srv._empty_cuda_cache()
        self.assertEqual(calls, ["sync", "clear", "empty"])

    def test_no_cuda_is_a_no_op(self):
        import torch

        with mock.patch.object(torch.cuda, "is_available", return_value=False), \
             mock.patch.object(torch.cuda, "empty_cache") as empty:
            srv._empty_cuda_cache()
        empty.assert_not_called()
