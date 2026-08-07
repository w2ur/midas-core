"""The arithmetic behind the numbers the agents are graded against.

`engine/baselines.py` publishes the passive benchmark and the coin flip — the
two controls every agent's record is measured against — and had 22 tests for
421 lines, almost all of them at the "does it produce a series" level. On
2026-08-07 the raw-close basis change moved **757 coin-flip rows** through this
module, which is the most exercise its arithmetic had ever had, and nothing in
the suite would have caught a ratio computed the wrong way round.

These tests cover the value arithmetic itself, the degenerate inputs, and two
properties the module's own docstrings assert but nothing checked:

  1. the passive benchmark is **scale-invariant** — it is computed from price
     ratios, so multiplying the whole series by a constant cannot move it;
  2. the coin flip is **not**, because `bt.Backtest` defaults to
     `integer_positions=True` and the rounding residue depends on absolute
     price. That asymmetry is load-bearing: it is why the pence→pounds
     normalisation moved the coin flip by up to 3.79% on a day when no return
     had changed, and why that series had to be restated onto the new basis.

Both are asserted here, in that order, so the pair cannot silently converge.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from engine.baselines import (
    _daterange,
    _load_ohlcv,
    compute_coin_flip,
    compute_global_reference,
    compute_passive_benchmark,
)
from engine.config import BenchmarkSpec, get_config

_INITIAL = 10_000.0


@pytest.fixture
def ohlcv(midas_data_root) -> Path:
    store = get_config().ohlcv_dir
    store.mkdir(parents=True, exist_ok=True)
    return store


def _write(store: Path, ticker: str, rows: list[tuple[str, float]]) -> None:
    store.joinpath(f"{ticker}.jsonl").write_text(
        "".join(json.dumps({"date": d, "close": c}) + "\n" for d, c in rows),
        encoding="utf-8",
    )


def _spec(ticker: str = "T", currency: str = "EUR") -> BenchmarkSpec:
    return BenchmarkSpec(
        label=f"{ticker} buy-and-hold", ticker=ticker, currency=currency
    )


# ---------------------------------------------------------------------------
# Passive benchmark — the value arithmetic
# ---------------------------------------------------------------------------


class TestPassiveBenchmarkArithmetic:
    def test_value_is_initial_times_the_price_ratio_to_the_first_close(self, ohlcv):
        _write(ohlcv, "T", [("2026-01-01", 50.0), ("2026-01-02", 75.0)])
        series = compute_passive_benchmark(_spec(), date(2026, 1, 1), date(2026, 1, 2))
        # 75/50 = 1.5 — asserted as an exact number, not "greater than".
        assert series[0]["portfolio_value"] == pytest.approx(10_000.0)
        assert series[1]["portfolio_value"] == pytest.approx(15_000.0)

    def test_a_halving_halves_the_book(self, ohlcv):
        """Direction matters: an inverted ratio passes a `!= initial` test."""
        _write(ohlcv, "T", [("2026-01-01", 80.0), ("2026-01-02", 40.0)])
        series = compute_passive_benchmark(_spec(), date(2026, 1, 1), date(2026, 1, 2))
        assert series[1]["portfolio_value"] == pytest.approx(5_000.0)

    def test_positions_value_carries_the_whole_book_and_cash_is_zero(self, ohlcv):
        """A passive benchmark is fully invested by construction."""
        _write(ohlcv, "T", [("2026-01-01", 10.0), ("2026-01-02", 11.0)])
        for row in compute_passive_benchmark(
            _spec(), date(2026, 1, 1), date(2026, 1, 2)
        ):
            assert row["cash"] == 0.0
            assert row["positions_value"] == pytest.approx(row["portfolio_value"])

    def test_a_non_trading_day_carries_the_last_close_rather_than_interpolating(
        self, ohlcv
    ):
        _write(ohlcv, "T", [("2026-01-01", 100.0), ("2026-01-05", 110.0)])
        series = compute_passive_benchmark(_spec(), date(2026, 1, 1), date(2026, 1, 5))
        held = [r["portfolio_value"] for r in series[:4]]
        assert held == [pytest.approx(10_000.0)] * 4
        assert series[4]["portfolio_value"] == pytest.approx(11_000.0)

    def test_the_window_opens_at_the_first_close_not_the_first_calendar_day(
        self, ohlcv
    ):
        """Days before any price exist are omitted, not valued at initial."""
        _write(ohlcv, "T", [("2026-01-04", 20.0), ("2026-01-05", 30.0)])
        series = compute_passive_benchmark(_spec(), date(2026, 1, 1), date(2026, 1, 5))
        assert [r["date"] for r in series] == ["2026-01-04", "2026-01-05"]
        assert series[0]["portfolio_value"] == pytest.approx(10_000.0)

    def test_a_ticker_with_no_data_draws_no_line(self, ohlcv):
        assert (
            compute_passive_benchmark(
                _spec("ABSENT"), date(2026, 1, 1), date(2026, 1, 5)
            )
            == []
        )

    def test_a_window_entirely_before_the_first_close_draws_no_line(self, ohlcv):
        _write(ohlcv, "T", [("2026-06-01", 10.0)])
        assert (
            compute_passive_benchmark(_spec(), date(2026, 1, 1), date(2026, 1, 5)) == []
        )

    def test_currency_is_the_spec_currency_not_the_tickers(self, ohlcv):
        _write(ohlcv, "T", [("2026-01-01", 10.0)])
        series = compute_passive_benchmark(
            _spec(currency="USD"), date(2026, 1, 1), date(2026, 1, 1)
        )
        assert series[0]["currency"] == "USD"

    def test_cash_sentinel_is_flat_and_fully_in_cash(self, ohlcv):
        series = compute_passive_benchmark(
            _spec("EUR_CASH_FLAT"), date(2026, 1, 1), date(2026, 1, 4)
        )
        assert len(series) == 4
        assert {r["portfolio_value"] for r in series} == {_INITIAL}
        assert {r["positions_value"] for r in series} == {0.0}
        assert {r["cash"] for r in series} == {_INITIAL}

    def test_cash_sentinel_needs_no_store_at_all(self, ohlcv):
        """It must not depend on a file named EUR_CASH_FLAT.jsonl existing."""
        assert not (ohlcv / "EUR_CASH_FLAT.jsonl").exists()
        assert compute_passive_benchmark(
            _spec("EUR_CASH_FLAT"), date(2026, 1, 1), date(2026, 1, 1)
        )


class TestPassiveBenchmarkIsScaleInvariant:
    """The docstring's claim: ratios "cancel any constant factor"."""

    @given(
        factor=st.floats(
            min_value=1e-3,
            max_value=1e3,
            allow_nan=False,
            allow_infinity=False,
        ),
        prices=st.lists(
            st.floats(
                min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=12,
        ),
    )
    def test_rescaling_every_price_leaves_the_curve_identical(
        self, midas_data_root, factor, prices
    ):
        store = get_config().ohlcv_dir
        store.mkdir(parents=True, exist_ok=True)
        days = [
            (date(2026, 1, 1) + timedelta(days=i)).isoformat()
            for i in range(len(prices))
        ]
        start, end = date(2026, 1, 1), date(2026, 1, len(prices))

        _write(store, "S", list(zip(days, prices)))
        plain = compute_passive_benchmark(_spec("S"), start, end)
        _write(store, "S", [(d, p * factor) for d, p in zip(days, prices)])
        scaled = compute_passive_benchmark(_spec("S"), start, end)

        assert len(plain) == len(scaled)
        for a, b in zip(plain, scaled):
            assert a["portfolio_value"] == pytest.approx(b["portfolio_value"], rel=1e-6)


@pytest.mark.live_cast
class TestCoinFlipIsNotScaleInvariant:
    """The documented counterpart, and the reason the coin flip gets restated.

    `bt.Backtest` defaults to `integer_positions=True`, so share counts round
    down and the residue depends on absolute price. If this ever starts passing
    as "invariant", the coin flip's restatement rationale has silently changed
    and `compute_coin_flip`'s docstring is wrong.
    """

    def _series(self, store: Path, scale: float) -> list[dict]:
        days = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(20)]
        for ticker, base in (("AAA", 116.0), ("BBB", 240.0), ("CCC", 37.0)):
            _write(
                store,
                ticker,
                [(d, (base + i * 1.7) * scale) for i, d in enumerate(days)],
            )
        return compute_coin_flip(
            agent_id="scale-probe",
            tickers=["AAA", "BBB", "CCC"],
            currency="EUR",
            max_positions=2,
            from_date=date(2026, 1, 1),
            to_date=date(2026, 1, 20),
        )

    def test_dividing_the_store_by_one_hundred_moves_the_control(self, ohlcv):
        pence = self._series(ohlcv, 1.0)
        pounds = self._series(ohlcv, 0.01)
        assert pence and pounds
        # Normalised to the same starting capital, the curves must differ
        # somewhere — that is the integer-position residue.
        assert any(
            a["portfolio_value"] != pytest.approx(b["portfolio_value"], rel=1e-9)
            for a, b in zip(pence, pounds)
        ), (
            "coin flip appears scale-invariant; compute_coin_flip's docstring says otherwise"
        )


class TestCoinFlipDegenerateInputs:
    def test_an_empty_universe_draws_no_line(self, ohlcv):
        assert (
            compute_coin_flip(
                agent_id="a",
                tickers=[],
                currency="EUR",
                max_positions=3,
                from_date=date(2026, 1, 1),
                to_date=date(2026, 1, 5),
            )
            == []
        )

    def test_a_universe_of_absent_tickers_draws_no_line(self, ohlcv):
        assert (
            compute_coin_flip(
                agent_id="a",
                tickers=["NOPE", "ALSO-NOPE"],
                currency="EUR",
                max_positions=2,
                from_date=date(2026, 1, 1),
                to_date=date(2026, 1, 5),
            )
            == []
        )

    def test_currency_is_propagated_onto_every_row(self, ohlcv):
        _write(ohlcv, "AAA", [("2026-01-0%d" % i, 10.0 + i) for i in range(1, 6)])
        series = compute_coin_flip(
            agent_id="a",
            tickers=["AAA"],
            currency="USD",
            max_positions=1,
            from_date=date(2026, 1, 1),
            to_date=date(2026, 1, 5),
        )
        assert series
        assert {r["currency"] for r in series} == {"USD"}

    def test_max_positions_of_zero_does_not_divide_by_zero(self, ohlcv):
        """`max_weight = 1.0 / max(max_positions, 1)` guards this — pin it."""
        _write(ohlcv, "AAA", [("2026-01-0%d" % i, 10.0 + i) for i in range(1, 6)])
        compute_coin_flip(
            agent_id="a",
            tickers=["AAA"],
            currency="EUR",
            max_positions=0,
            from_date=date(2026, 1, 1),
            to_date=date(2026, 1, 5),
        )


class TestLoadOhlcvRobustness:
    def test_a_missing_file_is_an_empty_map_not_an_exception(self, ohlcv):
        assert _load_ohlcv("NOT-THERE") == {}

    def test_blank_lines_are_skipped(self, ohlcv):
        ohlcv.joinpath("T.jsonl").write_text(
            '\n{"date":"2026-01-01","close":5.0}\n\n', encoding="utf-8"
        )
        assert _load_ohlcv("T") == {"2026-01-01": 5.0}

    def test_the_last_row_for_a_date_wins(self, ohlcv):
        """A duplicated date must resolve deterministically, not arbitrarily."""
        ohlcv.joinpath("T.jsonl").write_text(
            '{"date":"2026-01-01","close":5.0}\n{"date":"2026-01-01","close":6.0}\n',
            encoding="utf-8",
        )
        assert _load_ohlcv("T") == {"2026-01-01": 6.0}


class TestGlobalReference:
    def test_it_is_the_configured_global_reference_as_a_passive_benchmark(self, ohlcv):
        spec = get_config().global_reference
        _write(ohlcv, spec.ticker, [("2026-01-01", 100.0), ("2026-01-02", 120.0)])
        assert compute_global_reference(
            date(2026, 1, 1), date(2026, 1, 2)
        ) == compute_passive_benchmark(spec, date(2026, 1, 1), date(2026, 1, 2))

    def test_it_moves_with_its_ticker(self, ohlcv):
        spec = get_config().global_reference
        _write(ohlcv, spec.ticker, [("2026-01-01", 100.0), ("2026-01-02", 120.0)])
        series = compute_global_reference(date(2026, 1, 1), date(2026, 1, 2))
        assert series[1]["portfolio_value"] == pytest.approx(12_000.0)


class TestDateRange:
    def test_both_ends_are_inclusive(self):
        days = list(_daterange(date(2026, 1, 1), date(2026, 1, 3)))
        assert days == [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]

    def test_a_single_day_window_yields_that_day(self):
        assert list(_daterange(date(2026, 1, 1), date(2026, 1, 1))) == [
            date(2026, 1, 1)
        ]

    def test_an_inverted_window_yields_nothing(self):
        assert list(_daterange(date(2026, 1, 5), date(2026, 1, 1))) == []
