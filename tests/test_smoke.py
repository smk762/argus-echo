from __future__ import annotations

import argus_echo


def test_version_is_exposed() -> None:
    assert isinstance(argus_echo.__version__, str)
    assert argus_echo.__version__
