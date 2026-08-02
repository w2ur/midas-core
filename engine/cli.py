"""midas — console entry point.

Thin dispatch to the existing scripts and engine modules.  Each subcommand
delegates to the target module's ``__main__`` block via ``runpy.run_module``,
forwarding remaining argv so each module's own argument parser takes over.

``init-demo`` is the one exception: it materialises packaged files rather than
running an engine module, so it is implemented here and branches before the
``runpy`` dispatch.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

_COMMANDS: dict[str, str] = {
    "run-session": "scripts.daily_session",
    "fill-day": "engine.paper_broker",
    "check-triggers": "scripts.check_triggers",
    "backfill-baselines": "scripts.backfill_baselines",
    "build-bundle": "engine.output_bundle",
    "fetch-ohlcv": "scripts.fetch_ohlcv",
}

# The demo desk ships in the wheel as a package (examples/demo-desk is mapped to
# this name in pyproject.toml), so it is reachable from site-packages without a
# checkout. Resolve it through importlib.resources only — path arithmetic on
# ``__file__`` would work in a checkout and break for an installed user.
_DEMO_PACKAGE = "midas_demo_desk"
# Packaging scaffolding, not desk content: __init__.py only exists so the desk
# resolves as a regular package (see its docstring), and it must not land in the
# desk the user ends up running.
_DEMO_SKIP_NAMES = frozenset({"__init__.py", "__pycache__"})


def _copy_demo_tree(source: Traversable, dest: Path) -> list[Path]:
    """Copy a packaged directory tree to disk, preserving the relative layout.

    Walks ``Traversable`` rather than the filesystem so it works whatever the
    package was installed from (directory, zip import, editable finder).
    """
    written: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir(), key=lambda item: item.name):
        if entry.name in _DEMO_SKIP_NAMES:
            continue
        target = dest / entry.name
        if entry.is_dir():
            written.extend(_copy_demo_tree(entry, target))
        else:
            target.write_bytes(entry.read_bytes())
            written.append(target)
    return written


def _init_demo(argv: list[str]) -> int:
    """Write the packaged demo desk into a directory the user names."""
    parser = argparse.ArgumentParser(
        prog="midas init-demo",
        description="Write the packaged demo desk into a directory you name.",
    )
    parser.add_argument("target", help="directory to write the demo desk into")
    parser.add_argument(
        "--force",
        action="store_true",
        help="write into an existing non-empty directory (same-named files are overwritten)",
    )
    args = parser.parse_args(argv)

    target = Path(args.target).expanduser().resolve()
    if target.exists() and not target.is_dir():
        print(
            f"midas init-demo: {target} exists and is not a directory", file=sys.stderr
        )
        return 1
    if target.is_dir() and any(target.iterdir()) and not args.force:
        print(
            f"midas init-demo: {target} is not empty — pass --force to write into it",
            file=sys.stderr,
        )
        return 1

    try:
        source = files(_DEMO_PACKAGE)
    except ModuleNotFoundError:
        print(
            "midas init-demo: the packaged demo desk is not installed — "
            "install midas-core (`pip install midas-core`, or `pip install -e .` "
            "from a checkout)",
            file=sys.stderr,
        )
        return 1

    written = _copy_demo_tree(source, target)
    for path in written:
        print(f"  {path.relative_to(target)}")
    print(f"Wrote {len(written)} file(s) to {target}")
    print("Next:")
    print(f"  export MIDAS_DATA_DIR={target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="midas",
        description="Midas fund-manager engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in _COMMANDS:
        sub.add_parser(name, add_help=False, help=f"run {name}")
    sub.add_parser(
        "init-demo",
        add_help=False,
        help="write the packaged demo desk into a directory you name",
    )
    args, rest = parser.parse_known_args(argv)
    if args.command == "init-demo":
        return _init_demo(rest)
    module = _COMMANDS[args.command]
    sys.argv = [module, *rest]
    runpy.run_module(module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
