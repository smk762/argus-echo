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


# Engine images whose every pip install must be constrained by an exact-version file (#32), and that file.
PINNED_IMAGES = {
    "xtts": ("pip install", "-c /tmp/constraints.txt", "constraints.txt"),
    "qwen": ("pip install", "-c /tmp/constraints.txt", "constraints.txt"),
    "miso": ("uv pip install", "-c server-constraints.txt", "server-constraints.txt"),
}


def _install_commands(dockerfile: str, command: str) -> list[str]:
    """Each RUN instruction's text (continuation lines joined) that runs *command*."""
    joined = re.sub(r"\s*\\\n\s*", " ", dockerfile)
    return [line for line in joined.splitlines() if line.lstrip().startswith("RUN") and command in line]


def test_every_image_install_is_pinned_to_exact_versions() -> None:
    # A revision must imply its library stack: an unconstrained install resolves whatever is newest at build time.
    for engine, (command, flag, constraints) in PINNED_IMAGES.items():
        dockerfile = (ENGINES / engine / "Dockerfile").read_text()
        installs = _install_commands(dockerfile, command)
        assert installs, f"{engine}: no `{command}` found"
        unpinned = [i for i in installs if flag not in i]
        assert unpinned == [], f"{engine}: install without {flag}: {unpinned}"
        pins = [
            line
            for line in (ENGINES / engine / constraints).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        loose = [p for p in pins if not re.fullmatch(r"[A-Za-z0-9._-]+==[A-Za-z0-9.+!-]+", p.strip())]
        assert pins and loose == [], f"{engine}: {constraints} has entries that aren't exact pins: {loose}"


def test_the_install_scan_joins_continuation_lines() -> None:
    dockerfile = "RUN --mount=type=cache \\\n    pip install -c /tmp/constraints.txt torch \\\n        torchaudio\n"
    assert _install_commands(dockerfile, "pip install") == [
        "RUN --mount=type=cache pip install -c /tmp/constraints.txt torch torchaudio"
    ]


# Each engine's compose service, the argus.peer name callers ask for, and (in gpu-tenant.yaml) its container.
ENGINE_PEERS = {"miso": "miso", "xtts": "xtts", "qwen": "tts", "whisper": "whisper"}


def _compose_service(name: str) -> str:
    """The text of one service's block in compose.yaml (two-space indented, up to the next service)."""
    match = re.search(
        rf"^  {re.escape(name)}:(?:\s+#.*)?\n(?P<body>(?:(?:    .*|\s*)\n)+)", (ROOT / "compose.yaml").read_text(), re.M
    )
    assert match, f"compose.yaml has no service {name}"
    return match["body"]


def test_every_engine_container_carries_the_discovery_labels() -> None:
    # argus-iris's agents list the suite's services from these labels (argus-cortex's service module); ARGUS_PEER
    # repeats the name in the environment, because a process cannot read its own labels, and /health reports it.
    for service, peer in ENGINE_PEERS.items():
        block = _compose_service(service)
        for line in (
            f"argus.peer: {peer}",
            "argus.service: argus-echo",
            "argus.health: /health",
            'argus.port: "8031"',
            f"ARGUS_PEER: {peer}",
        ):
            assert re.search(rf"^\s+{re.escape(line)}(\s+#.*)?$", block, re.M), f"{service}: no `{line}`"
        assert ':8031"' in block, f"{service}: argus.port must be the container port it publishes"


def test_a_wyoming_bridge_is_not_labelled_as_a_peer() -> None:
    # The bridges speak Wyoming and serve no /health, so a directory built from the labels must not list them.
    for service in ("wyoming-xtts", "wyoming-miso", "wyoming-qwen", "wyoming-whisper"):
        assert "argus.peer" not in _compose_service(service)
        # argus-ci's service-peer-label rule needs the not-a-peer mark on the service's own line.
        compose = (ROOT / "compose.yaml").read_text()
        assert re.search(rf"^  {service}:\s+# nosemgrep: service-peer-label\s+\S", compose, re.M), service


def test_the_suite_file_mount_is_opt_in() -> None:
    # compose cannot make a bind mount optional, so the mount lives in an override file a host opts into.
    assert "suite.env" not in "".join(
        line
        for line in (ROOT / "compose.yaml").read_text().splitlines(keepends=True)
        if not line.lstrip().startswith("#")
    )
    suite = (ROOT / "compose.suite.yaml").read_text()
    assert "target: /etc/argus/suite.env" in suite and "read_only: true" in suite
    for service in ENGINE_PEERS:
        assert re.search(rf"^  {service}:\n    volumes:\n      - \*suite-env$", suite, re.M), service
