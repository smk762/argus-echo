from __future__ import annotations

import json
import struct

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from argus_echo.check import check
from argus_echo.cli import app
from argus_echo.contract import (
    SEED_MAX,
    LoadCheckpointRequest,
    StreamFormat,
    SynthesiseRequest,
    SynthesiseResponse,
    Weights,
    stream_identity,
    weights_fingerprint,
)

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


# What an engine on argus-cortex 0.16 adds to /health: ServiceHealth's fields beside the ones above.
SERVICE = {
    "service_version": "1.1",
    "service": "argus-echo",
    "version": "0.4.0",
    "status": "ok",
    "started_at": "2026-11-23T10:00:00Z",
    "wire": {},
    "capabilities": ["tts"],
    "checks": {"model": {"ok": True, "detail": None, "latency_ms": 0.1, "required": True}},
    "read_only": False,
    "build": "abc123",
    "peer": "miso",
}


def _finding(report, name: str):
    return next((f for f in report.findings if f.name == name), None)


def test_a_service_health_body_passes_and_names_the_service() -> None:
    report = check("http://e", transport=engine(health={**HEALTH, **SERVICE}))
    assert report.ok, report.findings
    assert _finding(report, "service").detail == "argus-echo 0.4.0 status=ok peer=miso"
    assert "capabilities=tts" in _finding(report, "GET /health").detail


def test_a_degraded_engine_passes_and_names_the_failed_check() -> None:
    failed = {"model": {"ok": False, "detail": "error", "required": True}}
    health = {**HEALTH, **SERVICE, "status": "degraded", "model_state": "error", "checks": failed}
    report = check("http://e", transport=engine(health=health))
    assert report.ok, report.findings  # the contract is served; whether the model loaded is not a contract check
    assert _finding(report, "service").detail.endswith("status=degraded peer=miso; failed checks: model")


def test_an_engine_from_before_service_health_is_not_failed_for_it() -> None:
    report = check("http://e", transport=engine())
    assert report.ok and _finding(report, "service") is None


def test_a_malformed_service_health_fails() -> None:
    report = check("http://e", transport=engine(health={**HEALTH, **SERVICE, "status": "loading"}))
    assert not report.ok
    assert _finding(report, "service").detail.startswith("not argus-cortex ServiceHealth: status")


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


def test_check_reports_the_runtime_libraries() -> None:
    runtime = {"python": "3.11.18", "torch": "2.6.0+cu124", "cuda": "12.4", "cudnn": "90100", "coqui-tts": None}
    health = {**HEALTH, "image": {"version": "0.3.0", "revision": "abc", "runtime": runtime}}
    report = check("http://e", transport=engine(health=health))
    assert report.ok, report.findings
    found = {f.name: f.detail for f in report.findings}
    assert found["image"] == (
        "version=0.3.0 revision=abc; python=3.11.18 torch=2.6.0+cu124 cuda=12.4 cudnn=90100 coqui-tts=none"
    )


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


# The worked examples in docs/api.md ("Weights fingerprint"); the engines' tests check the same digests.
QWEN_FILES = [
    {
        "name": "custom_voice/model.safetensors",
        "sha256": "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
    },
    {"name": "custom_voice/config.json", "sha256": "b79606fb3afea5bd1609ed40b622142f1c98125abcfe89a76a661b0e8e343910"},
    {"name": "base/model.safetensors", "sha256": "cae662172fd450bb0cd710a769079c05bfc5d8e35efa6576edc7d0377afdd4a2"},
]
CV_FP = "b955ea711653a31092cf4acad596e71ac31cb8f85c6feabe209f32ada86fecfb"
BASE_FP = "8763f2a5235402fc83d026fd46b563e7fa6c7b0c9d12a3cad770d4856148eebc"
ALL_FP = "e205d97735019470b091f339590c5f050f068b0f7817439400ed4337db73c89c"


def test_weights_fingerprint_matches_the_documented_examples() -> None:
    assert weights_fingerprint(QWEN_FILES, "custom_voice") == CV_FP
    assert weights_fingerprint(QWEN_FILES, "base") == BASE_FP
    assert weights_fingerprint(QWEN_FILES) == ALL_FP
    xtts = [
        {
            "name": f["name"].removeprefix("custom_voice/").replace("model.safetensors", "model.pth"),
            "sha256": f["sha256"],
        }
        for f in QWEN_FILES[:2]
    ]
    assert weights_fingerprint(xtts) == "311b8ed7abed4ae45274fedccda241ff6d0a129db13ea34b25b9fe0431de2649"


def _qwen_health(**models) -> dict:
    return {
        **HEALTH,
        "engine": "qwen-tts",
        "weights": {
            "checkpoint": "voice",
            "fingerprint": ALL_FP,
            "files": QWEN_FILES,
            "models": models
            or {
                "base": {"checkpoint": None, "fingerprint": BASE_FP},
                "custom_voice": {"checkpoint": "voice", "fingerprint": CV_FP},
            },
        },
    }


def test_check_reports_and_verifies_the_fingerprints() -> None:
    report = check("http://e", transport=engine(health=_qwen_health()))
    assert report.ok, report.findings
    detail = [f for f in report.findings if f.name == "weights"][0].detail
    assert f"fingerprint {ALL_FP}" in detail
    assert f"custom_voice: checkpoint=voice fingerprint {CV_FP}" in detail
    assert f"base: stock fingerprint {BASE_FP}" in detail


def test_check_fails_a_fingerprint_that_does_not_match_its_files() -> None:
    health = _qwen_health(base={"checkpoint": None, "fingerprint": CV_FP})
    report = check("http://e", transport=engine(health=health))
    assert not report.ok
    assert "does not match its base/ files" in [f for f in report.findings if f.name == "weights"][0].detail


def test_speak_reports_the_streams_weights() -> None:
    headers = {
        **STREAM_HEADERS,
        "X-Echo-Model": "custom_voice",
        "X-Echo-Checkpoint": "forge/run%207",
        "X-Echo-Weights": CV_FP,
    }
    report = check("http://e", speak="Hello.", transport=engine(stream_headers=headers))
    stream = [f for f in report.findings if f.name == "POST /synthesise/stream"][0]
    assert stream.detail.endswith(f"custom_voice forge/run 7 weights {CV_FP}")
    assert stream_identity({"x-echo-weights": CV_FP, "X-Echo-Checkpoint": ""}) == {
        "model": None,
        "checkpoint": None,
        "weights_fingerprint": CV_FP,
    }


def test_synthesise_response_model() -> None:
    body = {"audio_url": "u", "seed": 1, "model": "base", "checkpoint": None, "weights_fingerprint": BASE_FP, "x": 1}
    assert SynthesiseResponse.model_validate(body).weights_fingerprint == BASE_FP
