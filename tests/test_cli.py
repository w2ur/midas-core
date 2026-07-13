"""The midas CLI exposes the engine entry points."""

import subprocess
import sys


def test_help_lists_commands():
    out = subprocess.run(
        [sys.executable, "-m", "engine.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0
    for cmd in [
        "run-session",
        "fill-day",
        "check-triggers",
        "backfill-baselines",
        "build-bundle",
        "fetch-ohlcv",
    ]:
        assert cmd in out.stdout
