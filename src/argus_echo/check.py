"""Check a running engine against the argus-echo contracts it advertises.

``/health`` names the engine's capabilities (``tts``, ``stt``); one that names none is a ``tts`` engine. Only the
matching checks run. Read-only by default: ``/health``, ``/speakers`` and ``/clones`` load nothing. ``speak`` also
streams one utterance and ``transcribe`` posts one clip, and either loads the model if it is not loaded (that can
take minutes for Miso 8B).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import httpx
from argus_cortex.gpu_wire import ResidentHealth
from pydantic import ValidationError

from argus_echo.contract import DEFAULT_CAPABILITIES, StreamFormat, SynthesiseRequest
from argus_echo.contract.stt import TRANSCRIBE_PATH, Transcription

#: The clip ``check --transcribe`` sends by default: President Kennedy's 1961 inaugural address (11 s, public domain
#: as a work of the US federal government), 16 kHz mono FLAC.
BUNDLED_CLIP = "jfk.flac"
BUNDLED_CLIP_WORDS = ("ask", "country")


@dataclass
class Finding:
    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    url: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(f.ok for f in self.findings)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.findings.append(Finding(name, ok, detail))


def _get_json(client: httpx.Client, report: Report, path: str) -> object | None:
    try:
        r = client.get(path)
    except httpx.HTTPError as exc:
        report.add(f"GET {path}", False, f"request failed: {exc}")
        return None
    if r.status_code != 200:
        report.add(f"GET {path}", False, f"HTTP {r.status_code}: {r.text[:200]}")
        return None
    try:
        return r.json()
    except ValueError:
        report.add(f"GET {path}", False, "body is not JSON")
        return None


def check(
    url: str,
    *,
    speak: str | None = None,
    speaker: str | None = None,
    timeout: float = 30.0,
    speak_timeout: float = 600.0,
    transcribe: bool | str | Path = False,
    transport: httpx.BaseTransport | None = None,
) -> Report:
    """Run the checks against the engine at *url* and return what each found.

    *transcribe* posts a clip to an ``stt`` engine: ``True`` for the bundled one, or a path to a WAV or FLAC file.
    """
    url = url.rstrip("/")
    report = Report(url)
    with httpx.Client(base_url=url, timeout=timeout, transport=transport) as client:
        loaded = False
        capabilities: tuple[str, ...] = DEFAULT_CAPABILITIES
        body = _get_json(client, report, "/health")
        if body is not None:
            try:
                health = ResidentHealth.model_validate(body)
            except ValidationError as exc:
                report.add("GET /health", False, f"not argus-cortex ResidentHealth: {exc.errors()[0]['msg']}")
            else:
                extra = body if isinstance(body, dict) else {}
                engine = extra.get("engine") or "?"
                loaded = health.model_state in ("gpu", "cpu")
                advertised = extra.get("capabilities")
                if isinstance(advertised, list) and advertised and all(isinstance(c, str) for c in advertised):
                    capabilities = tuple(advertised)
                report.add(
                    "GET /health",
                    True,
                    f"engine={engine} model={extra.get('model')} model_state={health.model_state} busy={health.busy} "
                    f"capabilities={','.join(capabilities)}",
                )

        if "tts" in capabilities:
            _check_tts(client, report, loaded=loaded, speak=speak, speaker=speaker, speak_timeout=speak_timeout)
        if "stt" in capabilities:
            if transcribe:
                clip = None if transcribe is True else Path(transcribe)
                _check_transcribe(client, report, clip, speak_timeout)
            else:
                report.add(f"POST {TRANSCRIBE_PATH}", True, "skipped: transcribing loads the model (--transcribe)")
    return report


def _check_tts(
    client: httpx.Client, report: Report, *, loaded: bool, speak: str | None, speaker: str | None, speak_timeout: float
) -> None:
    # On XTTS and Qwen, /speakers takes a model lease, so it would load an unloaded model.
    if loaded or speak:
        speakers = _get_json(client, report, "/speakers")
        if speakers is not None:
            items = speakers.get("speakers") if isinstance(speakers, dict) else None
            ok = isinstance(items, list)
            report.add("GET /speakers", ok, f"{len(items)} speaker(s)" if ok else "no `speakers` list")
    else:
        report.add("GET /speakers", True, "skipped: the model is not loaded, and listing speakers may load it")

    try:
        r = client.get("/clones")
    except httpx.HTTPError as exc:
        report.add("GET /clones", False, f"request failed: {exc}")
    else:
        if r.status_code == 404:
            report.add("GET /clones", True, "not served by this engine (it keeps no clone list)")
        elif r.status_code != 200:
            report.add("GET /clones", False, f"HTTP {r.status_code}: {r.text[:200]}")
        else:
            try:
                items = r.json().get("clones")
            except (ValueError, AttributeError):
                items = None
            ok = isinstance(items, list)
            report.add("GET /clones", ok, f"{len(items)} clone(s)" if ok else "no `clones` list")

    if speak:
        _check_stream(client, report, SynthesiseRequest(text=speak, speaker=speaker), speak_timeout)


def _check_transcribe(client: httpx.Client, report: Report, clip: Path | None, timeout: float) -> None:
    name = f"POST {TRANSCRIBE_PATH}"
    if clip is None:
        data, filename = files("argus_echo").joinpath("data", BUNDLED_CLIP).read_bytes(), BUNDLED_CLIP
    else:
        try:
            data, filename = clip.read_bytes(), clip.name
        except OSError as exc:
            report.add(name, False, f"cannot read {clip}: {exc.strerror}")
            return
    start = time.monotonic()
    try:
        r = client.post(TRANSCRIBE_PATH, files={"audio": (filename, data)}, timeout=timeout)
    except httpx.HTTPError as exc:
        report.add(name, False, f"request failed: {exc}")
        return
    total = time.monotonic() - start
    if r.status_code != 200:
        report.add(name, False, f"HTTP {r.status_code}: {r.text[:200]}")
        return
    try:
        result = Transcription.model_validate(r.json())
    except (ValueError, ValidationError) as exc:
        report.add(name, False, f"not an stt Transcription: {str(exc)[:200]}")
        return
    if not result.text.strip():
        report.add(name, False, "empty transcript")
        return
    if clip is None and not all(w in result.text.lower() for w in BUNDLED_CLIP_WORDS):
        report.add(name, False, f"transcript does not match the bundled clip: {result.text[:120]!r}")
        return
    report.add(
        name,
        True,
        f"{result.duration:.1f} s of {result.language} in {total:.2f} s ({len(result.segments)} segment(s)): "
        f"{result.text[:80]!r}",
    )


def _check_stream(client: httpx.Client, report: Report, req: SynthesiseRequest, timeout: float) -> None:
    name = "POST /synthesise/stream"
    payload = req.model_dump(exclude_none=True)
    start = time.monotonic()
    first: float | None = None
    n_bytes = 0
    try:
        with client.stream("POST", "/synthesise/stream", json=payload, timeout=timeout) as r:
            if r.status_code != 200:
                r.read()
                report.add(name, False, f"HTTP {r.status_code}: {r.text[:200]}")
                return
            try:
                fmt = StreamFormat.from_headers(r.headers)
                width = fmt.sample_width
            except (ValueError, KeyError) as exc:
                report.add(name, False, f"bad stream headers: {exc}")
                return
            for chunk in r.iter_bytes():
                if chunk and first is None:
                    first = time.monotonic() - start
                n_bytes += len(chunk)
    except httpx.HTTPError as exc:
        report.add(name, False, f"request failed: {exc}")
        return
    total = time.monotonic() - start
    if n_bytes == 0:
        report.add(name, False, "no audio")
        return
    if n_bytes % (width * fmt.channels):
        report.add(name, False, f"{n_bytes} bytes is not a whole number of {fmt.audio_format} frames")
        return
    audio_s = fmt.seconds(n_bytes)
    report.add(
        name,
        True,
        f"{audio_s:.2f} s of {fmt.audio_format} {fmt.sample_rate} Hz x{fmt.channels}; "
        f"first byte {first:.2f} s, total {total:.2f} s (RTF {total / audio_s:.2f})",
    )
