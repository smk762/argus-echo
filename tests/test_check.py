from __future__ import annotations

import json
import struct

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from argus_echo.check import check
from argus_echo.cli import app
from argus_echo.contract import SEED_MAX, LoadCheckpointRequest, StreamFormat, SynthesiseRequest, Weights

HEALTH = {"status": "ok", "engine": "miso", "model": "MisoLabs/MisoTTS", "model_state": "unloaded", "busy": False}
PCM = struct.pack("<4f", 0.0, 0.1, -0.1, 0.0)
STREAM_HEADERS = {"X-Sample-Rate": "24000", "X-Audio-Format": "pcm_f32le", "X-Channels": "1"}


def engine(*, health=HEALTH, clones=200, stream_headers=STREAM_HEADERS, pcm=PCM, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(f"{request.method} {request.url.path}")
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json=health)
        if path == "/speakers":
            return httpx.Response(200, json={"speakers": [{"id": "0", "name": "0"}]})
        if path == "/clones":
            return httpx.Response(clones, json={"clones": []} if clones == 200 else {"detail": "Not Found"})
        if path == "/synthesise/stream":
            assert json.loads(request.content)["text"]
            return httpx.Response(200, headers=stream_headers, content=pcm)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_read_only_check_passes_and_does_not_load_the_model() -> None:
    seen: list[str] = []
    report = check("http://e", transport=engine(seen=seen))
    assert report.ok, report.findings
    assert "GET /speakers" not in seen  # listing speakers can load an unloaded model


def test_speakers_listed_when_model_is_loaded() -> None:
    seen: list[str] = []
    report = check("http://e", transport=engine(health={**HEALTH, "model_state": "gpu"}, seen=seen))
    assert report.ok
    assert "GET /speakers" in seen


def test_health_that_is_not_resident_health_fails() -> None:
    report = check("http://e", transport=engine(health={"status": "ok"}))
    assert not report.ok
    assert "ResidentHealth" in report.findings[0].detail


def test_engine_without_clone_list_is_not_a_failure() -> None:
    report = check("http://e", transport=engine(clones=404))
    assert report.ok
    assert "not served" in [f for f in report.findings if f.name == "GET /clones"][0].detail


def test_speak_measures_the_stream() -> None:
    report = check("http://e", speak="Hello.", transport=engine())
    assert report.ok, report.findings
    stream = [f for f in report.findings if f.name == "POST /synthesise/stream"][0]
    assert "pcm_f32le 24000 Hz" in stream.detail


@pytest.mark.parametrize(
    ("headers", "pcm", "why"),
    [
        ({"X-Sample-Rate": "24000"}, PCM, "bad stream headers"),
        (STREAM_HEADERS, PCM[:-1], "not a whole number"),
        (STREAM_HEADERS, b"", "no audio"),
    ],
)
def test_speak_rejects_a_bad_stream(headers, pcm, why) -> None:
    report = check("http://e", speak="Hello.", transport=engine(stream_headers=headers, pcm=pcm))
    assert not report.ok
    assert why in [f for f in report.findings if f.name == "POST /synthesise/stream"][0].detail


def test_stream_format_seconds() -> None:
    fmt = StreamFormat.from_headers({"x-sample-rate": "24000", "x-audio-format": "pcm_s16le", "x-channels": "1"})
    assert fmt.seconds(48000) == 1.0


def test_request_keeps_engine_specific_fields() -> None:
    body = SynthesiseRequest(text="hi", temperature=0.7).model_dump(exclude_none=True)
    assert body["temperature"] == 0.7


def test_cli_exit_code_follows_the_report(monkeypatch) -> None:
    import argus_echo.check as mod

    real = mod.check
    monkeypatch.setattr("argus_echo.cli.run_check", lambda url, **kw: real(url, transport=engine(clones=500)))
    result = CliRunner().invoke(app, ["check", "http://e"])
    assert result.exit_code == 1
    assert "FAIL GET /clones" in result.output


SHA = "a" * 64


def test_check_reports_the_loaded_checkpoint_and_its_hashes_without_loading_anything() -> None:
    seen: list[str] = []
    health = {
        **HEALTH,
        "engine": "xtts-v2",
        "image": {"version": "0.2.0", "revision": "0123456789abcdef"},
        "weights": {"checkpoint": "forge/run-7", "files": [{"name": "model.pth", "sha256": SHA, "bytes": 3}]},
    }
    report = check("http://e", transport=engine(health=health, seen=seen))
    assert report.ok, report.findings
    found = {f.name: f.detail for f in report.findings}
    assert found["weights"] == f"checkpoint=forge/run-7; 1 file(s): model.pth sha256:{SHA}"
    assert found["image"] == "version=0.2.0 revision=0123456789abcdef"
    assert not any(s.startswith("POST") for s in seen)  # read-only: never POST /admin/load


def test_check_reports_stock_weights_not_yet_hashed() -> None:
    health = {**HEALTH, "image": {"version": None, "revision": None}, "weights": {"checkpoint": None, "files": None}}
    report = check("http://e", transport=engine(health=health))
    assert report.ok
    found = {f.name: f.detail for f in report.findings}
    assert found["weights"].startswith("stock; not loaded")
    assert found["image"] == "version=unset revision=unset"


def test_check_fails_malformed_weights() -> None:
    health = {**HEALTH, "weights": {"checkpoint": None, "files": [{"name": "model.pth", "sha256": "nope"}]}}
    report = check("http://e", transport=engine(health=health))
    assert not report.ok
    assert [f for f in report.findings if f.name == "weights"][0].detail.startswith("malformed")


def test_engines_without_weights_report_nothing_extra() -> None:
    report = check("http://e", transport=engine())
    assert {f.name for f in report.findings}.isdisjoint({"weights", "image"})


def test_seed_is_optional_and_bounded() -> None:
    assert SynthesiseRequest(text="hi").seed is None
    assert SynthesiseRequest(text="hi", seed=SEED_MAX).seed == SEED_MAX
    for bad in (-1, SEED_MAX + 1):
        with pytest.raises(ValidationError):
            SynthesiseRequest(text="hi", seed=bad)


def test_load_and_weights_models() -> None:
    assert LoadCheckpointRequest(checkpoint=None).checkpoint is None
    with pytest.raises(ValidationError):
        LoadCheckpointRequest()
    w = Weights.model_validate({"checkpoint": "x", "files": [{"name": "model.pth", "sha256": SHA}]})
    assert w.files[0].bytes is None
