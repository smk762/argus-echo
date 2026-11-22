"""Rules from CLAUDE.md's "Must never happen" table that a file scan can enforce."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ROOT / "engines"


def _engine_sources() -> list[Path]:
    skip = {".venv", "__pycache__", "node_modules"}
    return [p for p in ENGINES.rglob("*.py") if not skip.intersection(p.relative_to(ENGINES).parts)]


def _imports_argus_echo(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "argus_echo" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and not node.level and (node.module or "").split(".")[0] == "argus_echo":
            return True
    return False


def test_the_scan_sees_every_engine() -> None:
    engines = {p.relative_to(ENGINES).parts[0] for p in _engine_sources()}
    assert {"miso", "xtts", "qwen", "whisper", "wyoming"} <= engines


def test_no_engine_imports_argus_echo() -> None:
    # An engine runs in its own environment, which does not have (and must not need) the argus-echo package.
    offenders = [str(p.relative_to(ROOT)) for p in _engine_sources() if _imports_argus_echo(p.read_text())]
    assert offenders == []


def test_the_import_check_catches_both_forms() -> None:
    assert _imports_argus_echo("import argus_echo.contract")
    assert _imports_argus_echo("from argus_echo.contract import tts")
    assert not _imports_argus_echo("from .argus_echo import x\nimport argus_echoes")


_BUCKET = re.compile(r"^\s*S3_BUCKET_NAME:\s*\$\{(?P<var>[A-Z0-9_]+):-(?P<default>[^}]+)\}\s*$")


def test_no_two_engines_share_an_s3_bucket() -> None:
    # Clone keys collide and clone formats differ between engines, so each engine gets its own bucket.
    found = [m.groupdict() for line in (ROOT / "compose.yaml").read_text().splitlines() if (m := _BUCKET.match(line))]
    assert len(found) >= 3, "every engine that stores clones sets S3_BUCKET_NAME: ${<ENGINE>_S3_BUCKET:-<default>}"
    defaults = [f["default"] for f in found]
    variables = [f["var"] for f in found]
    assert len(set(defaults)) == len(defaults), defaults
    assert len(set(variables)) == len(variables), variables
