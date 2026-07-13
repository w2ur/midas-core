"""Import-smoke: every engine module and script must import cleanly.

Guards the *deleted-symbol-import* regression class — a module that imports a
symbol removed during a refactor (e.g. the `fetch_ohlcv` import of the deleted
`engine.tickers.DEFAULT_PATH` caught during the config extraction). Ordinary
unit tests miss this when no test imports the offending module; a daily cron
session, which imports the whole chain, would crash at runtime instead.

Parametrised so each module reports independently. Runs at the default env
(no ``MIDAS_DATA_DIR``); imports never touch config (path/roster resolution is
lazy), so this stays a pure import check.
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _all_modules() -> list[str]:
    mods: list[str] = []
    for p in sorted((_ROOT / "engine").rglob("*.py")):
        if p.name == "__init__.py":
            continue
        mods.append(".".join(p.relative_to(_ROOT).with_suffix("").parts))
    for p in sorted((_ROOT / "scripts").glob("*.py")):
        if p.name == "__init__.py":
            continue
        mods.append("scripts." + p.stem)
    return mods


@pytest.mark.parametrize("module", _all_modules())
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
