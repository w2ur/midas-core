"""Unit tests for engine/fx.py — currency conversion via committed OHLCV store.

Uses a tmp_path-backed _OHLCV directory (monkeypatched) so the tests are
hermetic and don't depend on the live state of data/market/ohlcv/.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

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
# get_rate — composed through USD, and the USD-quoted pairs
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


class TestEveryStoredPairIsRouted:
    """Regression: round-3 review, 2026-10-06. The store held GBPUSD=X and
    USDJPY=X, yet GBP->USD and JPY->USD answered None: the routes were two
    hand-written maps and the USD one omitted both pairs. Routes now come
    from one table, and the table must name every pair the store holds."""

    _DAY = date(2025, 1, 2)

    # Reads the live committed OHLCV store, which core does not ship.
    @pytest.mark.live_cast
    def test_the_table_lists_every_pair_in_the_committed_store(self):
        from engine.config import get_config

        stored = sorted(p.stem for p in get_config().ohlcv_dir.glob("*=X.jsonl"))
        assert stored, "the probe must see the committed store's pairs"
        assert sorted(fx.STORE_PAIRS) == stored

    @pytest.mark.parametrize("pair", fx.STORE_PAIRS)
    def test_both_directions_of_every_stored_pair_resolve(self, fake_ohlcv, pair):
        _write_jsonl(fake_ohlcv / f"{pair}.jsonl", [{"date": "2025-01-02", "close": 1.25}])
        base, quote = pair[:3], pair[3:6]
        assert fx.get_rate(base, quote, self._DAY) == pytest.approx(1.25)
        assert fx.get_rate(quote, base, self._DAY) == pytest.approx(0.8)

    @pytest.mark.parametrize(
        ("frm", "to", "expected"),
        [
            ("GBP", "USD", 1.27),
            ("USD", "GBP", 1 / 1.27),
            ("JPY", "USD", 1 / 150.0),
            ("USD", "JPY", 150.0),
        ],
    )
    def test_gbp_and_jpy_reach_usd(self, fake_ohlcv, frm, to, expected):
        _write_jsonl(fake_ohlcv / "GBPUSD=X.jsonl", [{"date": "2025-01-02", "close": 1.27}])
        _write_jsonl(fake_ohlcv / "USDJPY=X.jsonl", [{"date": "2025-01-02", "close": 150.0}])
        assert fx.get_rate(frm, to, self._DAY) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("to", "pair", "close", "expected"),
        [
            ("CHF", "USDCHF=X", 0.90, 1.10 * 0.90),
            ("CAD", "USDCAD=X", 1.35, 1.10 * 1.35),
            ("AUD", "AUDUSD=X", 0.65, 1.10 / 0.65),
            ("NZD", "NZDUSD=X", 0.60, 1.10 / 0.60),
        ],
    )
    def test_eur_still_composes_through_usd(self, fake_ohlcv, to, pair, close, expected):
        _write_jsonl(fake_ohlcv / "EURUSD=X.jsonl", [{"date": "2025-01-02", "close": 1.10}])
        _write_jsonl(fake_ohlcv / f"{pair}.jsonl", [{"date": "2025-01-02", "close": close}])
        assert fx.get_rate("EUR", to, self._DAY) == pytest.approx(expected)
        assert fx.get_rate(to, "EUR", self._DAY) == pytest.approx(1 / expected)

    def test_gbp_composes_to_chf_through_usd(self, fake_ohlcv):
        _write_jsonl(fake_ohlcv / "GBPUSD=X.jsonl", [{"date": "2025-01-02", "close": 1.27}])
        _write_jsonl(fake_ohlcv / "USDCHF=X.jsonl", [{"date": "2025-01-02", "close": 0.90}])
        assert fx.get_rate("GBP", "CHF", self._DAY) == pytest.approx(1.27 * 0.90)

    def test_an_unstored_currency_still_has_no_route(self, fake_ohlcv):
        for pair in fx.STORE_PAIRS:
            _write_jsonl(fake_ohlcv / f"{pair}.jsonl", [{"date": "2025-01-02", "close": 1.1}])
        assert fx.get_rate("SEK", "EUR", self._DAY) is None
        assert fx.get_rate("USD", "SEK", self._DAY) is None

    @pytest.mark.parametrize("bad", [-1.1, float("nan"), float("inf")])
    def test_a_close_that_is_not_a_positive_finite_number_is_no_rate(self, fake_ohlcv, bad):
        path = fake_ohlcv / "EURUSD=X.jsonl"
        path.write_text(json.dumps({"date": "2025-01-02", "close": bad}) + "\n")
        assert fx.get_rate("EUR", "USD", self._DAY) is None
        assert fx.get_rate("CHF", "EUR", self._DAY) is None


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


class TestRateTickers:
    """`rate_tickers` names the store row(s) a missing rate is waiting for
    (round-4 review, 2026-10-06: the coin flip's empty-draw concern names
    them). It must name exactly the pairs `get_rate` reads."""

    def test_a_stored_pair_in_either_direction(self):
        assert fx.rate_tickers("USD", "EUR") == ("EURUSD=X",)
        assert fx.rate_tickers("GBP", "USD") == ("GBPUSD=X",)

    def test_a_pair_composed_through_usd_names_both_legs(self):
        assert fx.rate_tickers("CHF", "EUR") == ("USDCHF=X", "EURUSD=X")

    def test_no_route_and_no_conversion_name_nothing(self):
        assert fx.rate_tickers("SEK", "EUR") == ()
        assert fx.rate_tickers("EUR", "EUR") == ()


# ---------------------------------------------------------------------------
# store_cache and the bisected lookup (round-4 review, 2026-10-06)
# ---------------------------------------------------------------------------


class TestStoreCache:
    def _count_loads(self, monkeypatch) -> list[str]:
        loads: list[str] = []
        real = fx._load_store_series

        def counting(ticker):
            loads.append(ticker)
            return real(ticker)

        monkeypatch.setattr(fx, "_load_store_series", counting)
        return loads

    def test_inside_a_block_each_pair_file_is_read_once(self, fake_ohlcv, monkeypatch):
        _write_jsonl(fake_ohlcv / "EURUSD=X.jsonl", [{"date": "2025-01-02", "close": 1.25}])
        loads = self._count_loads(monkeypatch)
        with fx.store_cache():
            for day in range(2, 30):
                assert fx.get_rate("USD", "EUR", date(2025, 1, day)) == pytest.approx(0.8)
        assert loads == ["EURUSD=X"]

    def test_outside_a_block_every_ask_reads_the_file_afresh(self, fake_ohlcv, monkeypatch):
        """The broker's behaviour is unchanged: a rewrite between two asks is seen."""
        path = fake_ohlcv / "EURUSD=X.jsonl"
        _write_jsonl(path, [{"date": "2025-01-02", "close": 1.25}])
        loads = self._count_loads(monkeypatch)
        assert fx.get_rate("EUR", "USD", date(2025, 1, 3)) == 1.25
        _write_jsonl(path, [{"date": "2025-01-02", "close": 1.5}])
        assert fx.get_rate("EUR", "USD", date(2025, 1, 3)) == 1.5
        assert loads == ["EURUSD=X", "EURUSD=X"]

    def test_the_cache_ends_with_the_outermost_block(self, fake_ohlcv):
        path = fake_ohlcv / "EURUSD=X.jsonl"
        _write_jsonl(path, [{"date": "2025-01-02", "close": 1.25}])
        with fx.store_cache():
            assert fx.get_rate("EUR", "USD", date(2025, 1, 3)) == 1.25
            with fx.store_cache():
                pass
            _write_jsonl(path, [{"date": "2025-01-02", "close": 1.5}])
            assert fx.get_rate("EUR", "USD", date(2025, 1, 3)) == 1.25, "still cached"
        assert fx._SERIES_CACHE is None
        assert fx.get_rate("EUR", "USD", date(2025, 1, 3)) == 1.5

    def test_a_coin_flip_advance_reads_its_rate_file_once(self, fake_ohlcv, monkeypatch):
        """The consumer: a multi-day converted advance used to re-read
        EURUSD=X for every day it valued."""
        from datetime import timedelta

        from engine.baselines import advance_coin_flip
        from engine.config import get_config

        days = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(8)]
        for t, c in (("U", 30.0), ("V", 40.0)):
            _write_jsonl(fake_ohlcv / f"{t}.jsonl", [{"date": d, "close": c} for d in days])
        _write_jsonl(fake_ohlcv / "EURUSD=X.jsonl", [{"date": d, "close": 1.25} for d in days])
        loads = self._count_loads(monkeypatch)
        advance_coin_flip(
            agent_id="probe", tickers=["U", "V"], currency="EUR", max_positions=2,
            series_path=get_config().baselines_dir / "probe" / "coinflip.json",
            from_date=date(2026, 1, 1), to_date=date(2026, 1, 8),
        )
        assert loads == ["EURUSD=X"]


_ISO_DAYS = st.dates(min_value=date(2024, 1, 1), max_value=date(2027, 12, 31))


@given(
    series=st.dictionaries(
        _ISO_DAYS.map(date.isoformat),
        st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
        max_size=40,
    ),
    target=_ISO_DAYS,
)
def test_the_bisected_lookup_equals_the_linear_scan(series, target):
    """The cached path bisects sorted dates (`_latest_in`), the uncached one
    scans (`_latest_on_or_before`); both give the answer the linear `max`
    over every date gave, for any series and any target."""
    eligible = [d for d in series if d <= target.isoformat()]
    expected = series[max(eligible)] if eligible else None
    assert fx._latest_on_or_before(series, target) == expected
    assert fx._latest_in(*fx._sorted(series), target) == expected


class TestUncachedPath:
    """Round-5 review, 2026-10-06. The bisection added for the cached path
    also ran outside a `store_cache()` block, where each ask reads the file
    afresh: it sorted the whole series for one lookup, slower than the old
    O(n) filter-and-max, and left `_latest_on_or_before` with no production
    caller."""

    def test_an_uncached_ask_never_sorts(self, fake_ohlcv, monkeypatch):
        _write_jsonl(
            fake_ohlcv / "EURUSD=X.jsonl",
            [{"date": "2025-01-03", "close": 1.5}, {"date": "2025-01-02", "close": 1.25}],
        )

        def no_sort(series):
            raise AssertionError("the uncached path sorted the series")

        monkeypatch.setattr(fx, "_sorted", no_sort)
        assert fx.get_rate("EUR", "USD", date(2025, 1, 2)) == 1.25
        assert fx.get_rate("EUR", "USD", date(2025, 1, 9)) == 1.5

    def test_an_uncached_ask_is_the_linear_scan(self, fake_ohlcv, monkeypatch):
        _write_jsonl(fake_ohlcv / "EURUSD=X.jsonl", [{"date": "2025-01-02", "close": 1.25}])
        calls: list[date] = []
        real = fx._latest_on_or_before

        def counting(series, target):
            calls.append(target)
            return real(series, target)

        monkeypatch.setattr(fx, "_latest_on_or_before", counting)
        assert fx.get_rate("EUR", "USD", date(2025, 1, 3)) == 1.25
        assert calls == [date(2025, 1, 3)]


def test_a_config_reset_inside_a_block_serves_the_new_stores_rates(
    fake_ohlcv, tmp_path, monkeypatch
):
    """Regression: round-5 review, 2026-10-06. `_SERIES_CACHE` was keyed on
    the ticker alone and survived `reset_config_cache()`, so a block open
    across a `MIDAS_DATA_DIR` switch kept serving the old store's rates."""
    import shutil

    from engine.config import _LEGACY_ROOT, get_config, reset_config_cache

    _write_jsonl(fake_ohlcv / "EURUSD=X.jsonl", [{"date": "2025-01-02", "close": 1.25}])
    other = (tmp_path / "other").resolve()
    other.mkdir()
    shutil.copy(_LEGACY_ROOT / "roster.yaml", other / "roster.yaml")
    with fx.store_cache():
        assert fx.get_rate("EUR", "USD", date(2025, 1, 3)) == 1.25
        monkeypatch.setenv("MIDAS_DATA_DIR", str(other))
        reset_config_cache()
        _write_jsonl(get_config().ohlcv_dir / "EURUSD=X.jsonl", [{"date": "2025-01-02", "close": 2.0}])
        assert fx.get_rate("EUR", "USD", date(2025, 1, 3)) == 2.0
        assert fx._SERIES_CACHE is not None, "the block is still open"
