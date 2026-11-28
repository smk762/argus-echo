"""xtts_server behaviour that needs no model or torch: S3 credentials, the
duration probe, GPU error classification, the loudness report, the
minimum-speech guard and a clone listing with malformed sidecars."""

import contextlib
import json
import sys
import types

import numpy as np
import pytest
import xtts_server as xs
from fastapi.testclient import TestClient


class _Mgr:
    """A manager whose lease reports the GPU and never touches a model."""

    state = xs.TTSState.GPU
    weights = {"checkpoint": None, "files": None}
    active = 0
    _last_used = 0.0

    def identity(self):
        return {"checkpoint": None, "weights_fingerprint": None}

    @contextlib.asynccontextmanager
    async def lease(self, device=None):
        yield "cuda:0"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(xs, "_mgr", _Mgr())
    monkeypatch.setattr(xs, "_gpu_fault", None, raising=False)
    monkeypatch.setattr(xs, "_gpu_memory_info", lambda: None)
    return TestClient(xs.app, raise_server_exceptions=False)


# ── S3 credentials: no fallback to MinIO's stock root account ─────────────────


@pytest.mark.parametrize("key,secret", [("", ""), ("key", ""), ("", "secret")])
def test_missing_s3_credentials_are_refused(monkeypatch, key, secret):
    monkeypatch.setattr(xs, "S3_ACCESS_KEY_ID", key)
    monkeypatch.setattr(xs, "S3_SECRET_ACCESS_KEY", secret)
    monkeypatch.setattr(xs, "_s3_client", None)
    with pytest.raises(RuntimeError, match="S3_ACCESS_KEY_ID"):
        xs._get_s3_client()


def test_module_has_no_default_s3_credentials(monkeypatch):
    import importlib.util

    monkeypatch.delenv("S3_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("S3_SECRET_ACCESS_KEY", raising=False)
    spec = importlib.util.spec_from_file_location("xtts_server_noenv", xs.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (module.S3_ACCESS_KEY_ID, module.S3_SECRET_ACCESS_KEY) == ("", "")


# ── 72f6650856: an unknown frame count is None, not 0.0 s ─────────────────────


@pytest.mark.parametrize("frames,rate,expected", [(0, 44100, None), (48000, 0, None), (48000, 24000, 2.0)])
def test_torchaudio_duration_is_none_when_the_length_is_unknown(monkeypatch, tmp_path, frames, rate, expected):
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"not something soundfile reads")
    fake = types.ModuleType("torchaudio")
    fake.info = lambda path: types.SimpleNamespace(num_frames=frames, sample_rate=rate)
    monkeypatch.setitem(sys.modules, "torchaudio", fake)
    assert xs._audio_duration(str(clip)) == expected


# ── 086cd6e1af: only a real OOM is the retryable 503 ──────────────────────────


def _synthesise_raising(client, monkeypatch, exc):
    def boom(*args):
        raise exc

    monkeypatch.setattr(xs, "_resolve_latents", lambda req: (None, None))
    monkeypatch.setattr(xs, "_run_inference", boom)
    return client.post("/synthesise", json={"text": "hi", "speaker": "x"})


def test_fatal_cuda_error_is_500_and_marks_the_worker_unhealthy(client, monkeypatch):
    assert client.get("/health").status_code == 200
    r = _synthesise_raising(client, monkeypatch, RuntimeError("CUDA error: device-side assert triggered at 0x7f"))
    assert r.status_code == 500
    assert r.json()["error"] != "GPU OOM"
    assert "device-side" not in r.text and "detail" not in r.json()
    health = client.get("/health")
    assert health.status_code == 503
    assert health.json()["status"] == "gpu_error"
    assert health.json()["model_state"] == "gpu"  # still a ResidentHealth body


def test_out_of_memory_stays_a_retryable_503(client, monkeypatch):
    r = _synthesise_raising(client, monkeypatch, RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert r.status_code == 503
    assert r.json()["error"] == "GPU OOM"
    assert client.get("/health").status_code == 200


def test_oom_is_recognised_by_type(monkeypatch):
    class OutOfMemoryError(RuntimeError):
        pass

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(OutOfMemoryError=OutOfMemoryError)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert xs._is_oom(OutOfMemoryError("allocation failed"))
    assert not xs._is_oom(RuntimeError("CUDA error: illegal memory access"))


# ── ddc089bfe9: the report says what the peak guard left ──────────────────────


def _fake_pyloudnorm(monkeypatch, loud_in):
    pyln = types.ModuleType("pyloudnorm")
    pyln.Meter = lambda sr: types.SimpleNamespace(integrated_loudness=lambda audio: loud_in)
    pyln.normalize = types.SimpleNamespace(
        loudness=lambda audio, measured, target: audio * (10.0 ** ((target - measured) / 20.0))
    )
    monkeypatch.setitem(sys.modules, "pyloudnorm", pyln)


def test_peak_limited_clip_reports_the_level_it_was_written_at(monkeypatch):
    monkeypatch.setattr(xs, "PREPROCESS_TARGET_LUFS", -20.0)
    _fake_pyloudnorm(monkeypatch, loud_in=-40.0)  # +20 dB wanted
    audio = np.full(1000, 0.01, dtype=np.float32)
    audio[0] = 0.5  # high crest factor: +20 dB would put this peak at 5.0
    report = {"applied": [], "warnings": []}
    out = xs._normalise_loudness(audio, 24000, report)
    assert float(np.max(np.abs(out))) == pytest.approx(0.99, abs=1e-4)
    # 0.5 → 0.99 is +5.9 dB, not the +20 dB the target asked for.
    assert report["gain_db"] == pytest.approx(5.9, abs=0.05)
    assert report["lufs_out"] == pytest.approx(-34.1, abs=0.05)
    assert "peak_limited" in report["warnings"]


def test_unlimited_clip_reports_the_target(monkeypatch):
    monkeypatch.setattr(xs, "PREPROCESS_TARGET_LUFS", -20.0)
    _fake_pyloudnorm(monkeypatch, loud_in=-26.0)
    report = {"applied": [], "warnings": []}
    xs._normalise_loudness(np.full(1000, 0.1, dtype=np.float32), 24000, report)
    assert (report["lufs_out"], report["gain_db"]) == (-20.0, 6.0)
    assert report["warnings"] == []


# ── 3d8723b629: no speech in any clip is under the minimum, not "unknown" ─────


def _clone(client, monkeypatch, tmp_path, kept_seconds):
    reached_model = []

    def preprocess(raw_path):
        out = tmp_path / f"proc-{len(list(tmp_path.iterdir()))}.wav"
        out.write_bytes(b"")
        return str(out), {"applied": [], "warnings": [], "kept_seconds": kept_seconds}

    def latents(paths):
        reached_model.append(paths)
        raise RuntimeError("stop here")

    monkeypatch.setattr(xs, "PREPROCESS", True)
    monkeypatch.setattr(xs, "_preprocess_clip", preprocess)
    monkeypatch.setattr(xs, "_conditioning_latents", latents)
    files = [("audio", (f"c{i}.wav", b"x" * 10, "audio/wav")) for i in range(2)]
    return client.post("/clone-voice", files=files, data={"companion_id": "abc"}), reached_model


def test_clone_with_no_speech_in_any_clip_is_422(client, monkeypatch, tmp_path):
    r, reached_model = _clone(client, monkeypatch, tmp_path, kept_seconds=0.0)
    assert r.status_code == 422
    assert "under 3 seconds" in r.json()["error"]
    assert reached_model == []


def test_clone_with_enough_speech_reaches_the_model(client, monkeypatch, tmp_path):
    _r, reached_model = _clone(client, monkeypatch, tmp_path, kept_seconds=4.0)
    assert len(reached_model) == 1


class _Lease:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


def _clone_reports(client, monkeypatch, tmp_path, overrides):
    """Clone from one clip per override (None: preprocessing fails); returns the response, what reached the model
    and the meta stored."""
    work = tmp_path / "work"
    work.mkdir()
    queue = list(overrides)
    reached, stored = [], {}

    def preprocess(raw_path):
        override = queue.pop(0)
        if override is None:
            raise ValueError("cannot decode")
        out = work / f"proc-{len(list(work.iterdir()))}.wav"
        out.write_bytes(b"")
        return str(out), {"applied": [], "warnings": [], "kept_seconds": 4.0, **override}

    class _Tensor:
        def detach(self):
            return self

        cpu = detach

    def latents(paths):
        reached.extend(paths)
        return _Tensor(), _Tensor()

    torch = types.ModuleType("torch")
    torch.save = lambda obj, buf: buf.write(b"STORED")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(xs, "PREPROCESS", True)
    monkeypatch.setattr(xs, "_preprocess_clip", preprocess)
    monkeypatch.setattr(xs, "_audio_duration", lambda path: None)
    monkeypatch.setattr(xs, "_conditioning_latents", latents)
    monkeypatch.setattr(xs._mgr, "lease", lambda *a, **k: _Lease())
    monkeypatch.setattr(xs, "_upload_to_s3", lambda data, key, ct="": stored.__setitem__(key, data))
    monkeypatch.setattr(xs, "_clone_cache", xs.OrderedDict())
    files = [("audio", (f"c{i}.wav", b"x" * 10, "audio/wav")) for i in range(len(overrides))]
    r = client.post("/clone-voice", files=files, data={"companion_id": "abc"})
    assert list(work.iterdir()) == []  # a skipped clip's file is removed too
    return r, reached, stored


# The VAD keeps the whole clip when it hears no speech, so kept_seconds alone doesn't show it.
SILENT = {"warnings": ["vad_no_speech"], "kept_seconds": 20.0}


@pytest.mark.parametrize("bad", [SILENT, {"kept_seconds": 0.0}, {"kept_seconds": 0.2}])
def test_a_clip_with_no_speech_is_left_out_of_the_clone(client, monkeypatch, tmp_path, bad):
    r, reached, stored = _clone_reports(client, monkeypatch, tmp_path, [{}, dict(bad, warnings=list(bad.get("warnings", [])))])
    assert r.status_code == 200, r.text
    assert len(reached) == 1
    body = r.json()
    assert (body["num_clips"], body["duration"]) == (1, 4.0)
    assert "skipped_no_speech" not in body["clips"][0]["warnings"]
    assert "skipped_no_speech" in body["clips"][1]["warnings"]
    assert json.loads(stored["voice_clones/abc/meta.json"])["num_clips"] == 1


def test_only_silent_clips_is_422_however_long(client, monkeypatch, tmp_path):
    r, reached, stored = _clone_reports(client, monkeypatch, tmp_path, [dict(SILENT, warnings=["vad_no_speech"])])
    assert r.status_code == 422
    assert "skipped_no_speech" in r.json()["clips"][0]["warnings"]
    assert (reached, stored) == ([], {})


def test_an_unprobeable_clip_beside_a_silent_one_is_left_to_the_model(client, monkeypatch, tmp_path):
    r, reached, _ = _clone_reports(client, monkeypatch, tmp_path, [None, dict(SILENT, warnings=["vad_no_speech"])])
    assert r.status_code == 200, r.text
    assert len(reached) == 1


def test_a_short_measured_clip_beside_an_unprobeable_one_is_left_to_the_model(client, monkeypatch, tmp_path):
    # 1 s measured, the other clip's length unknown (preprocessing failed, the raw probe reads None): it may hold
    # plenty of speech, so the 3 s minimum is not applied on the measured second alone.
    r, reached, _ = _clone_reports(client, monkeypatch, tmp_path, [{"kept_seconds": 1.0}, None])
    assert r.status_code == 200, r.text
    assert len(reached) == 2


def test_short_clips_that_were_all_measured_are_still_422(client, monkeypatch, tmp_path):
    r, reached, stored = _clone_reports(client, monkeypatch, tmp_path, [{"kept_seconds": 1.0}, {"kept_seconds": 1.5}])
    assert r.status_code == 422 and "under 3 seconds" in r.json()["error"]
    assert (reached, stored) == ([], {})


def test_clone_with_unprobeable_clips_is_left_to_the_model(client, monkeypatch, tmp_path):
    reached_model = []

    def latents(paths):
        reached_model.append(paths)
        raise RuntimeError("stop here")

    monkeypatch.setattr(xs, "PREPROCESS", False)
    monkeypatch.setattr(xs, "_audio_duration", lambda path: None)
    monkeypatch.setattr(xs, "_conditioning_latents", latents)
    client.post("/clone-voice", files=[("audio", ("c.ogg", b"x" * 10, "audio/ogg"))], data={"companion_id": "abc"})
    assert len(reached_model) == 1


# ── e754de9668: one bad meta.json does not fail the listing ───────────────────


class _S3:
    def __init__(self, objects):
        self.objects = objects

    def get_paginator(self, name):
        keys = [{"Key": k} for k in self.objects]
        return types.SimpleNamespace(paginate=lambda **kw: [{"Contents": keys}])

    def download_fileobj(self, bucket, key, buf):
        buf.write(self.objects[key])


def test_listing_skips_malformed_sidecars(client, monkeypatch):
    objects = {
        "voice_clones/a/meta.json": json.dumps({"companion_id": "a", "created_at": 5}).encode(),
        "voice_clones/b/meta.json": b"[1, 2]",
        "voice_clones/c/meta.json": b"null",
        "voice_clones/d/meta.json": json.dumps({"companion_id": "d", "created_at": "2026-01-01T00:00:00"}).encode(),
        "voice_clones/e/meta.json": json.dumps({"companion_id": "e"}).encode(),
        "voice_clones/f/meta.json": json.dumps({"companion_id": "f", "created_at": 9.5}).encode(),
        "voice_clones/f/prompt.pt": b"",
    }
    monkeypatch.setattr(xs, "_get_s3_client", lambda: _S3(objects))
    assert [c["companion_id"] for c in xs._list_clones()] == ["f", "a", "d", "e"]
    r = client.get("/clones")
    assert r.status_code == 200
