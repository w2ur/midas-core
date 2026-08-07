"""Unit tests for engine/fx.py — currency conversion via committed OHLCV store.

Uses a tmp_path-backed _OHLCV directory (monkeypatched) so the tests are
hermetic and don't depend on the live state of data/market/ohlcv/.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from engine import fx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def fake_ohlcv(midas_data_root):
    """Redirect the OHLCV store (via MIDAS_DATA_DIR) to a tmp dir; return it.

    engine.fx now reads get_config().ohlcv_dir at call time, so redirecting the
    data root relocates the FX store hermetically.
    """
    from engine.config import get_config

    ohlcv = get_config().ohlcv_dir
    ohlcv.mkdir(parents=True, exist_ok=True)
    return ohlcv


# ---------------------------------------------------------------------------
# _load_store_series
# ---------------------------------------------------------------------------


class TestLoadStoreSeries:
    def test_missing_file_returns_empty(self, fake_ohlcv):
        assert fx._load_store_series("DOES_NOT_EXIST=X") == {}

    def test_parses_jsonl_close(self, fake_ohlcv):
        _write_jsonl(
            fake_ohlcv / "EURUSD=X.jsonl",
            [
                {"date": "2025-01-02", "close": 1.10},
                {"date": "2025-01-03", "close": 1.12},
            ],
        )
        s = fx._load_store_series("EURUSD=X")
        assert s == {"2025-01-02": 1.10, "2025-01-03": 1.12}

    def test_reads_raw_close_and_ignores_adj_close(self, fake_ohlcv):
        """Raw `close`, never `adj_close` (2026-08-07 review §5.2).

        Asserted the opposite until 2026-08-07. An FX pair's two fields are
        equal in practice, so this fixture makes them differ on purpose —
        otherwise the test could not tell the two bases apart.
        """
        _write_jsonl(
            fake_ohlcv / "X.jsonl",
            [{"date": "2025-01-02", "close": 1.0, "adj_close": 1.05}],
        )
        s = fx._load_store_series("X")
        assert s == {"2025-01-02": 1.0}

    def test_skips_row_with_null_close(self, fake_ohlcv):
        """A row with no close is skipped, not served from `adj_close`."""
        _write_jsonl(
            fake_ohlcv / "X.jsonl",
            [
                {"date": "2025-01-02", "close": None, "adj_close": 1.05},
                {"date": "2025-01-03", "close": 1.12, "adj_close": 1.20},
            ],
        )
        s = fx._load_store_series("X")
        assert s == {"2025-01-03": 1.12}

    def test_skips_blank_lines(self, fake_ohlcv):
        path = fake_ohlcv / "X.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n{"date": "2025-01-02", "close": 1.0}\n\n')
        assert fx._load_store_series("X") == {"2025-01-02": 1.0}

    def test_skips_invalid_json(self, fake_ohlcv):
        path = fake_ohlcv / "X.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{not json}\n{"date": "2025-01-02", "close": 1.0}\n')
        assert fx._load_store_series("X") == {"2025-01-02": 1.0}

    def test_skips_rows_missing_date_or_close(self, fake_ohlcv):
        _write_jsonl(
            fake_ohlcv / "X.jsonl",
            [
                {"close": 1.0},  # no date
                {"date": "2025-01-02"},  # no close
                {"date": "2025-01-03", "close": 1.5},
            ],
        )
        assert fx._load_store_series("X") == {"2025-01-03": 1.5}


# ---------------------------------------------------------------------------
# _latest_on_or_before
# ---------------------------------------------------------------------------


class TestLatestOnOrBefore:
    def test_empty_series_returns_none(self):
        assert fx._latest_on_or_before({}, date(2025, 1, 1)) is None

    def test_no_eligible_dates_returns_none(self):
        series = {"2025-02-01": 1.0, "2025-02-02": 1.1}
        assert fx._latest_on_or_before(series, date(2025, 1, 31)) is None

    def test_returns_exact_match(self):
        series = {"2025-01-02": 1.10, "2025-01-03": 1.12}
        assert fx._latest_on_or_before(series, date(2025, 1, 3)) == 1.12

    def test_returns_latest_before_target(self):
        series = {"2025-01-02": 1.10, "2025-01-04": 1.15}
        # target between 02 and 04 → returns 02
        assert fx._latest_on_or_before(series, date(2025, 1, 3)) == 1.10

    def test_returns_latest_when_target_after_all(self):
        series = {"2025-01-02": 1.10, "2025-01-04": 1.15}
        assert fx._latest_on_or_before(series, date(2030, 1, 1)) == 1.15


# ---------------------------------------------------------------------------
# get_rate — direct pairs and edge cases
# ---------------------------------------------------------------------------


class TestGetRate:
    def test_same_currency_returns_one(self, fake_ohlcv):
        assert fx.get_rate("EUR", "EUR", date(2025, 1, 1)) == 1.0
        assert fx.get_rate("USD", "USD") == 1.0

    def test_eur_to_usd_direct_pair(self, fake_ohlcv):
        # Stored EURUSD=X = USD per EUR. EUR→USD is the direct read.
        _write_jsonl(
            fake_ohlcv / "EURUSD=X.jsonl",
            [{"date": "2025-01-02", "close": 1.10}],
        )
        rate = fx.get_rate("EUR", "USD", date(2025, 1, 2))
        assert rate == pytest.approx(1.10)

    def test_usd_to_eur_inverts_eur_usd_pair(self, fake_ohlcv):
        # USD→EUR is the inverted read of the same EURUSD=X pair.
        _write_jsonl(
            fake_ohlcv / "EURUSD=X.jsonl",
            [{"date": "2025-01-02", "close": 1.25}],
        )
        rate = fx.get_rate("USD", "EUR", date(2025, 1, 2))
        assert rate == pytest.approx(1 / 1.25)

    def test_missing_pair_returns_none(self, fake_ohlcv):
        # No file → no rate.
        assert fx.get_rate("EUR", "USD", date(2025, 1, 2)) is None

    def test_zero_rate_returns_none(self, fake_ohlcv):
        # Defensive: division by zero on inverted pairs.
        _write_jsonl(
            fake_ohlcv / "EURUSD=X.jsonl",
            [{"date": "2025-01-02", "close": 0.0}],
        )
        assert fx.get_rate("USD", "EUR", date(2025, 1, 2)) is None

    def test_uses_latest_on_or_before(self, fake_ohlcv):
        _write_jsonl(
            fake_ohlcv / "EURUSD=X.jsonl",
            [
                {"date": "2025-01-02", "close": 1.10},
                {"date": "2025-01-04", "close": 1.20},
            ],
        )
        # Sunday 01-05 → falls back to 01-04.
        assert fx.get_rate("EUR", "USD", date(2025, 1, 5)) == pytest.approx(1.20)
        # 01-03 between → falls back to 01-02.
        assert fx.get_rate("EUR", "USD", date(2025, 1, 3)) == pytest.approx(1.10)


# ---------------------------------------------------------------------------
# get_rate — indirect (via USD) and via fallback usd_pair_map
# ---------------------------------------------------------------------------


class TestGetRateIndirect:
    def test_chf_to_eur_routes_via_usd(self, fake_ohlcv):
        # CHF→USD via USDCHF=X (inverted). USD→EUR via EURUSD=X (inverted).
        _write_jsonl(
            fake_ohlcv / "USDCHF=X.jsonl",
            [{"date": "2025-01-02", "close": 0.90}],  # USD per CHF inverse
        )
        _write_jsonl(
            fake_ohlcv / "EURUSD=X.jsonl",
            [{"date": "2025-01-02", "close": 1.10}],
        )
        # Expected: (1/0.90) * (1/1.10)
        rate = fx.get_rate("CHF", "EUR", date(2025, 1, 2))
        assert rate == pytest.approx((1 / 0.90) * (1 / 1.10))

    def test_indirect_returns_none_when_one_leg_missing(self, fake_ohlcv):
        _write_jsonl(
            fake_ohlcv / "USDCHF=X.jsonl",
            [{"date": "2025-01-02", "close": 0.90}],
        )
        # Missing EURUSD=X → cannot complete CHF→EUR.
        assert fx.get_rate("CHF", "EUR", date(2025, 1, 2)) is None

    def test_usd_to_chf_via_fallback_pair(self, fake_ohlcv):
        # USDCHF=X stores CHF per USD directly.
        _write_jsonl(
            fake_ohlcv / "USDCHF=X.jsonl",
            [{"date": "2025-01-02", "close": 0.90}],
        )
        assert fx.get_rate("USD", "CHF", date(2025, 1, 2)) == pytest.approx(0.90)

    def test_aud_to_usd_via_fallback_pair_not_inverted(self, fake_ohlcv):
        # AUDUSD=X stores USD per AUD directly (not inverted).
        _write_jsonl(
            fake_ohlcv / "AUDUSD=X.jsonl",
            [{"date": "2025-01-02", "close": 0.65}],
        )
        assert fx.get_rate("AUD", "USD", date(2025, 1, 2)) == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# convert + to_eur
# ---------------------------------------------------------------------------


class TestConvert:
    def test_applies_rate_to_amount(self, fake_ohlcv):
        _write_jsonl(
            fake_ohlcv / "EURUSD=X.jsonl",
            [{"date": "2025-01-02", "close": 1.10}],
        )
        assert fx.convert(100, "EUR", "USD", date(2025, 1, 2)) == pytest.approx(110)

    def test_returns_none_when_rate_unavailable(self, fake_ohlcv):
        assert fx.convert(100, "EUR", "USD", date(2025, 1, 2)) is None

    def test_same_currency_is_identity(self, fake_ohlcv):
        assert fx.convert(42.5, "USD", "USD", date(2025, 1, 2)) == 42.5


class TestToEur:
    def test_routes_through_convert(self, fake_ohlcv):
        _write_jsonl(
            fake_ohlcv / "EURUSD=X.jsonl",
            [{"date": "2025-01-02", "close": 1.25}],
        )
        # USD→EUR: amount * (1/1.25)
        assert fx.to_eur(125, "USD", date(2025, 1, 2)) == pytest.approx(100.0)

    def test_eur_to_eur_identity(self, fake_ohlcv):
        assert fx.to_eur(100, "EUR", date(2025, 1, 2)) == 100.0

    def test_returns_none_when_rate_unavailable(self, fake_ohlcv):
        assert fx.to_eur(100, "USD", date(2025, 1, 2)) is None
