"""Tests for engine.survivorship — survivorship-bias detection.

Regression: an sp500 backtest starting in 2024 must carry a SURVIVORSHIP_BIAS
warning, because the committed sp500.json holds *today's* constituents. An
early factor-research run this way inflated returns ~194%.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, time, timezone

import pytest

from engine.survivorship import (
    SURVIVORSHIP_PRONE_UNIVERSES,
    survivorship_warning,
    universe_last_refresh,
)


def _seed_universe(root, name: str, tickers: list[str], refreshed_on: date) -> None:
    """Write data/universes/<name>.json under the tmp root, mtime = refreshed_on."""
    path = root / "data" / "universes" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tickers), encoding="utf-8")
    epoch = datetime.combine(refreshed_on, time(12, 0), tzinfo=timezone.utc).timestamp()
    os.utime(path, (epoch, epoch))


def test_sp500_backtest_from_2024_carries_warning(midas_data_root) -> None:
    # Refresh stamp in 2026, well after the 2024 backtest start.
    _seed_universe(midas_data_root, "sp500", ["AAPL", "MSFT"], date(2026, 6, 1))
    warning = survivorship_warning("sp500", date(2024, 1, 1))
    assert warning is not None
    assert warning.startswith("SURVIVORSHIP_BIAS")
    assert "sp500" in warning
    assert "2024-01-01" in warning


def test_dow30_is_exempt() -> None:
    # dow30 is stable and intentionally not survivorship-prone.
    assert "dow30" not in SURVIVORSHIP_PRONE_UNIVERSES
    assert survivorship_warning("dow30", date(2024, 1, 1)) is None


def test_etf_broad_is_exempt() -> None:
    assert survivorship_warning("etf-broad", date(2024, 1, 1)) is None


def test_start_after_refresh_is_clean(midas_data_root) -> None:
    _seed_universe(midas_data_root, "sp500", ["AAPL"], date(2024, 1, 1))
    # A start on/after the refresh date carries no survivorship warning.
    assert survivorship_warning("sp500", date(2025, 1, 1)) is None


def test_missing_universe_file_is_clean(midas_data_root) -> None:
    # No committed file → no refresh date → no false warning.
    assert universe_last_refresh("nasdaq100") is None
    assert survivorship_warning("nasdaq100", date(2024, 1, 1)) is None


@pytest.mark.parametrize("universe_id", sorted(SURVIVORSHIP_PRONE_UNIVERSES))
def test_all_prone_universes_warn_for_old_start(midas_data_root, universe_id) -> None:
    stem = "stoxx600" if universe_id == "stoxx-600" else universe_id
    _seed_universe(midas_data_root, stem, ["AAA", "BBB"], date(2026, 6, 1))
    assert survivorship_warning(universe_id, date(2020, 1, 1)) is not None
