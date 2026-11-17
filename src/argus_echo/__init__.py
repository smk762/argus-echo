"""argus-echo — The Argus suite's speech servers: Miso, XTTS-v2 and Qwen3-TTS behind one HTTP contract, plus a Wyoming bridge"""

from __future__ import annotations

try:
    # Written by hatch-vcs at build time (see pyproject [tool.hatch.build.hooks.vcs]).
    from argus_echo._version import __version__
except ImportError:  # running from a source checkout that hasn't been built
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("argus-echo")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
