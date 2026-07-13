"""midas — console entry point.

Thin dispatch to the existing scripts and engine modules.  Each subcommand
delegates to the target module's ``__main__`` block via ``runpy.run_module``,
forwarding remaining argv so each module's own argument parser takes over.
"""

from __future__ import annotations

import argparse
import runpy
import sys

_COMMANDS: dict[str, str] = {
    "run-session": "scripts.daily_session",
    "fill-day": "engine.paper_broker",
    "check-triggers": "scripts.check_triggers",
    "backfill-baselines": "scripts.backfill_baselines",
    "build-bundle": "engine.output_bundle",
    "fetch-ohlcv": "scripts.fetch_ohlcv",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="midas",
        description="Midas fund-manager engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in _COMMANDS:
        sub.add_parser(name, add_help=False, help=f"run {name}")
    args, rest = parser.parse_known_args(argv)
    module = _COMMANDS[args.command]
    sys.argv = [module, *rest]
    runpy.run_module(module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
