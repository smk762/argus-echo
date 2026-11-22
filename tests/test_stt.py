"""The ``stt`` contract and ``check``'s capability handling, against fake engines."""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from argus_echo.check import check
from argus_echo.cli import app
from argus_echo.contract import DEFAULT_CAPABILITIES, SynthesiseRequest
from argus_echo.contract.stt import (
    StreamFinal,
    StreamStart,
    TranscribeOptions,
    Transcription,
    TranscriptionsRequest,
    TranscriptionsResult,
)

HEALTH = {"status": "ok", "engine": "whisper", "model": "m", "model_state": "unloaded", "busy": False}
RESULT = {
    "text": "And so, my fellow Americans, ask not what your country can do for you.",
    "language": "en",
    "language_probability": 0.96,
    "duration": 11.0,
    "segments": [{"start": 0.0, "end": 10.4, "text": "And so, my fellow Americans, ask not what your country can do."}],
}


def stt_engine(*, health=None, result=RESULT, status=200, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if seen is not None:
            seen.append(f"{request.method} {path}")
        if path == "/health":
            return httpx.Response(200, json=health or {**HEALTH, "capabilities": ["stt"]})
        if path == "/transcribe":
            assert request.headers["content-type"].startswith("multipart/form-data")
            assert b'name="audio"' in request.content
            return httpx.Response(status, json=result)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_the_tts_import_path_still_works() -> None:
    from argus_echo.contract.tts import SynthesiseRequest as Direct

    assert SynthesiseRequest is Direct
    assert DEFAULT_CAPABILITIES == ("tts",)


def test_transcription_round_trips_with_and_without_words() -> None:
    t = Transcription.model_validate(RESULT)
    assert t.segments[0].words is None
    words = {
        **RESULT,
        "segments": [
            {**RESULT["segments"][0], "words": [{"start": 0, "end": 0.2, "word": " And", "probability": 0.9}]}
        ],
    }
    assert Transcription.model_validate(words).segments[0].words[0].word == " And"


def test_options_default_to_detection_and_refuse_unknown_fields() -> None:
    assert TranscribeOptions().language is None
    with pytest.raises(ValidationError):
        TranscribeOptions(task="translate")


def test_transcriptions_takes_exactly_one_source_and_answers_composers_shape() -> None:
    # A caller may send all three keys, nulls included; anything else is ignored, as composer did.
    req = TranscriptionsRequest.model_validate({"audio_url": "https://cdn.example/a.mp3", "audio_base64": None, "x": 1})
    assert req.language is None
    for body in ({}, {"audio_url": None, "audio_base64": ""}, {"audio_url": "https://a/b", "audio_base64": "AAAA"}):
        with pytest.raises(ValidationError):
            TranscriptionsRequest.model_validate(body)
    out = {"text": "hello", "language": "en", "duration_seconds": 1.5, "confidence": 0.98}
    assert TranscriptionsResult.model_validate(out).model_dump() == out


def test_stream_messages() -> None:
    assert StreamStart.model_validate({"type": "start"}).language is None
    final = StreamFinal.model_validate({**RESULT, "type": "final", "start": 1.0, "end": 12.0})
    assert final.start == 1.0


def test_an_stt_engine_gets_no_tts_checks_and_a_read_only_check_loads_nothing() -> None:
    seen: list[str] = []
    report = check("http://e", transport=stt_engine(seen=seen))
    assert report.ok, report.findings
    assert seen == ["GET /health"]
    assert "skipped" in report.findings[-1].detail


def test_transcribe_posts_the_bundled_clip_and_checks_the_text() -> None:
    report = check("http://e", transcribe=True, transport=stt_engine())
    assert report.ok, report.findings
    assert "11.0 s of en" in report.findings[-1].detail


@pytest.mark.parametrize(
    ("result", "status", "why"),
    [
        ({**RESULT, "text": " "}, 200, "empty transcript"),
        ({**RESULT, "text": "something else"}, 200, "does not match"),
        ({"text": "ask country"}, 200, "not an stt Transcription"),
        ({"error": "busy"}, 503, "HTTP 503"),
    ],
)
def test_transcribe_failures(result, status, why) -> None:
    report = check("http://e", transcribe=True, transport=stt_engine(result=result, status=status))
    assert not report.ok
    assert why in report.findings[-1].detail


def test_a_custom_clip_only_needs_non_empty_text(tmp_path) -> None:
    clip = tmp_path / "x.wav"
    clip.write_bytes(b"RIFF....WAVE")
    report = check("http://e", transcribe=clip, transport=stt_engine(result={**RESULT, "text": "hello"}))
    assert report.ok, report.findings
    missing = check("http://e", transcribe=tmp_path / "nope.wav", transport=stt_engine())
    assert "cannot read" in missing.findings[-1].detail


def test_an_engine_advertising_both_gets_both_suites() -> None:
    seen: list[str] = []
    health = {**HEALTH, "capabilities": ["tts", "stt"]}
    report = check("http://e", transport=stt_engine(health=health, seen=seen))
    assert "GET /clones" in seen
    assert any(f.name == "POST /transcribe" for f in report.findings), "the read-only stt check is missing"

    seen.clear()
    check("http://e", transcribe=True, transport=stt_engine(health=health, seen=seen))
    assert "GET /clones" in seen
    assert "POST /transcribe" in seen


def test_cli_transcribe_flag(monkeypatch) -> None:
    import argus_echo.check as mod

    got = {}
    real = mod.check

    def fake(url, **kw):
        got.update(kw)
        return real(url, transcribe=kw["transcribe"], transport=stt_engine())

    monkeypatch.setattr("argus_echo.cli.run_check", fake)
    result = CliRunner().invoke(app, ["check", "http://e", "--transcribe"])
    assert result.exit_code == 0, result.output
    assert got["transcribe"] is True
