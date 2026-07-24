"""The midas CLI exposes the engine entry points."""

import importlib.util
import subprocess
import sys

from engine.cli import _COMMANDS


def test_every_command_module_is_importable():
    """Every module a subcommand dispatches to must be importable.

    Regression guard for the exact bug class that shipped: `midas fetch-ohlcv`
    mapped to ``scripts.fetch_ohlcv``, but that script was not in the sync_core
    manifest, so it was absent from the public repo and the subcommand raised
    ModuleNotFoundError at runtime. ``find_spec`` catches such a dangling mapping
    without executing the module's ``__main__`` side effects.
    """
    missing = [
        (cmd, mod)
        for cmd, mod in _COMMANDS.items()
        if importlib.util.find_spec(mod) is None
    ]
    assert missing == [], f"CLI subcommands map to unimportable modules: {missing}"


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
