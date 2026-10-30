"""Unit discipline in the yfinance fallback (2026-08-07 review, W7.1).

Hermetic — `yf.download` is monkeypatched, nothing here touches the network
(unlike `test_market_data.py`, which is an integration suite by design).

Two defects, both on the backtester's read path:

1. **The fallback served raw vendor units.** The store is ISO-denominated at
   ingest, so `LLOY.L` reads 1.166 out of it; yfinance serves the same bar as
   116.60. Nothing scaled the fallback, so whichever source answered decided
   the units — a 100x difference, silently.

2. **Store coverage was all-or-nothing.** One ticker missing or short sent the
   *whole* request to yfinance, including every ticker the store held
   perfectly well. So defect 1 was not confined to thin names: one thin name
   changed the units of every other column in the frame.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

import engine.market_data as md
from engine.config import get_config
from engine.market_data import MarketDataFetcher

START = date(2026, 8, 3)
END = date(2026, 8, 7)


@pytest.fixture
def store(midas_data_root: Path) -> Path:
    d = get_config().ohlcv_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _seed(store: Path, ticker: str, closes: dict[str, float]) -> None:
    (store / f"{ticker}.jsonl").write_text(
        "".join(
            f'{{"date": "{d}", "open": {c}, "high": {c}, "low": {c}, '
            f'"close": {c}, "adj_close": {c}, "volume": 100}}\n'
            for d, c in sorted(closes.items())
        ),
        encoding="utf-8",
    )


def _vendor_frame(prices: dict[str, float]) -> pd.DataFrame:
    """A yfinance-shaped result: MultiIndex columns, level 0 = price type."""
    idx = pd.to_datetime(["2026-08-03", "2026-08-04"])
    cols = pd.MultiIndex.from_product([["Close", "Open"], list(prices)])
    data = {(kind, t): [p, p] for kind in ("Close", "Open") for t, p in prices.items()}
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Capture what the vendor was asked for, and answer with vendor units."""
    calls: dict = {"tickers": None, "prices": {}}

    def _download(tickers, **_kwargs):
        calls["tickers"] = list(tickers) if not isinstance(tickers, str) else [tickers]
        return _vendor_frame(calls["prices"])

    monkeypatch.setattr(md.yf, "download", _download)
    return calls


class TestFallbackUnits:
    def test_pence_ticker_from_the_vendor_arrives_in_pounds(
        self, store: Path, tmp_path: Path, fake_download: dict
    ) -> None:
        """LLOY.L is not in the store, so the vendor answers — at 116.60 pence.
        The frame must carry 1.166, the same number the store would give."""
        fake_download["prices"] = {"LLOY.L": 116.60}

        df = MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(
            ["LLOY.L"], START, END
        )
        assert df["LLOY.L"].iloc[0] == pytest.approx(1.166)

    def test_a_non_pence_ticker_is_not_scaled(
        self, store: Path, tmp_path: Path, fake_download: dict
    ) -> None:
        """The control. A scale applied to everything would be just as wrong,
        and would pass the test above."""
        fake_download["prices"] = {"AAPL": 212.06}

        df = MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(
            ["AAPL"], START, END
        )
        assert df["AAPL"].iloc[0] == pytest.approx(212.06)

    def test_current_prices_fallback_is_normalised_too(
        self, store: Path, fake_download: dict
    ) -> None:
        fake_download["prices"] = {"LLOY.L": 116.60}

        out = MarketDataFetcher().fetch_current_prices(["LLOY.L"])
        assert out["LLOY.L"] == pytest.approx(1.166)


class TestPartialStoreCoverage:
    def test_only_the_uncovered_tickers_reach_the_vendor(
        self, store: Path, tmp_path: Path, fake_download: dict
    ) -> None:
        _seed(store, "AAPL", {"2026-08-03": 212.0, "2026-08-04": 213.0})
        _seed(store, "MSFT", {"2026-08-03": 500.0, "2026-08-04": 501.0})
        fake_download["prices"] = {"THIN.L": 250.0}

        df = MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(
            ["AAPL", "MSFT", "THIN.L"], START, END
        )

        assert fake_download["tickers"] == ["THIN.L"]
        assert list(df.columns) == ["AAPL", "MSFT", "THIN.L"]
        assert df["AAPL"].iloc[0] == pytest.approx(212.0)
        assert df["MSFT"].iloc[0] == pytest.approx(500.0)
        assert df["THIN.L"].iloc[0] == pytest.approx(2.50)

    def test_one_thin_ticker_does_not_change_the_others_units(
        self, store: Path, tmp_path: Path, fake_download: dict
    ) -> None:
        """The defect stated directly. `BP.L` sits in the store in pounds; a
        second, uncovered ticker used to send the whole request to the vendor,
        and `BP.L` would have come back at 402.0 instead of 4.02."""
        _seed(store, "BP.L", {"2026-08-03": 4.02, "2026-08-04": 4.05})
        fake_download["prices"] = {"NEWCO.L": 250.0}

        df = MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(
            ["BP.L", "NEWCO.L"], START, END
        )
        assert df["BP.L"].iloc[0] == pytest.approx(4.02)
        assert "BP.L" not in (fake_download["tickers"] or [])

    def test_full_store_coverage_never_calls_the_vendor(
        self, store: Path, tmp_path: Path, fake_download: dict
    ) -> None:
        """The sandbox contract: no outbound HTTP when the store can answer."""
        _seed(store, "AAPL", {"2026-08-03": 212.0, "2026-08-04": 213.0})

        MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(["AAPL"], START, END)
        assert fake_download["tickers"] is None

    def test_a_ticker_neither_source_serves_is_absent(
        self, store: Path, tmp_path: Path, fake_download: dict
    ) -> None:
        _seed(store, "AAPL", {"2026-08-03": 212.0, "2026-08-04": 213.0})
        fake_download["prices"] = {}

        df = MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(
            ["AAPL", "GHOST"], START, END
        )
        assert list(df.columns) == ["AAPL"]


# ---------------------------------------------------------------------------
# The two sources must agree on what `end` means (2026-10-30)
# ---------------------------------------------------------------------------


def _ranged_download(monkeypatch: pytest.MonkeyPatch, closes: dict[str, dict[str, float]]) -> dict:
    """A yfinance stand-in that honours `start`/`end` the way yfinance does:
    `start` inclusive, `end` EXCLUSIVE. The fixture above ignores the range,
    which is why the off-by-one between the two sources was invisible here."""
    calls: dict = {"end": None}

    def _download(tickers, start=None, end=None, **_kwargs):
        calls["end"] = end
        wanted = [tickers] if isinstance(tickers, str) else list(tickers)
        dates = sorted(
            {d for t in wanted for d in closes.get(t, {}) if start <= d < end}
        )
        cols = pd.MultiIndex.from_product([["Close"], wanted])
        data = {("Close", t): [closes.get(t, {}).get(d) for d in dates] for t in wanted}
        return pd.DataFrame(data, index=pd.to_datetime(dates), columns=cols).dropna(
            how="all"
        )

    monkeypatch.setattr(md.yf, "download", _download)
    return calls


class TestEndDateIsInclusiveForBothSources:
    """Regression: a36311565 — `run_backtest.py` raised "Position is open
    ... latest price is NaN" for a ticker the store held on the end date.

    `_store_series` serves `[start, end]`; `yf.download` serves `[start, end)`.
    A ticker whose store file starts after `start` (CSCO's 636 rows, most of
    sp500's two-year backfills) went to the vendor and came back one row short,
    so the frame's last row was NaN for every vendor-served column, on every
    end date. Seen on dow30 2026-10-28 (CSCO, SHW) and sp500 2025-12-31 (442
    of 503 columns).
    """

    def test_a_vendor_served_ticker_has_a_close_on_the_end_date(
        self, store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(store, "AAPL", {"2026-08-03": 212.0, "2026-08-06": 213.0, "2026-08-07": 214.0})
        _ranged_download(
            monkeypatch,
            {"CSCO": {"2026-08-03": 110.0, "2026-08-06": 111.0, "2026-08-07": 112.0}},
        )

        df = MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(
            ["AAPL", "CSCO"], START, END
        )

        assert df.index[-1] == pd.Timestamp(END)
        assert not df.iloc[-1].isna().any()
        assert df.loc[pd.Timestamp(END), "CSCO"] == pytest.approx(112.0)

    def test_the_vendor_is_never_asked_for_todays_forming_bar(
        self, store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Making `end` inclusive must not reach today: `--to` defaults to
        today, and Yahoo serves a crypto or intraday equity bar for a day still
        trading (the rule `scripts/fetch_ohlcv.py` keeps on the ingest side)."""
        today = datetime.now(UTC).date()
        calls = _ranged_download(monkeypatch, {"CSCO": {}})

        MarketDataFetcher().fetch_prices(["CSCO"], START, today)

        assert calls["end"] == str(today)  # exclusive: yesterday is the newest bar

    def test_a_trailing_date_only_one_source_holds_is_dropped(
        self, store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default `--to` (today) on a checkout whose store lags the
        vendor by a day: the vendor serves 08-07, the store stops at 08-06.
        Keeping 08-07 put a row in the frame on which every store-served
        column was NaN (2026-10-29: 498 of 503 sp500 columns, CRL held)."""
        _seed(store, "AAPL", {"2026-08-03": 212.0, "2026-08-06": 213.0})
        _ranged_download(
            monkeypatch,
            {"CSCO": {"2026-08-03": 110.0, "2026-08-06": 111.0, "2026-08-07": 112.0}},
        )

        df = MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(
            ["AAPL", "CSCO"], START, date(2026, 8, 8)
        )

        assert df.index[-1] == pd.Timestamp("2026-08-06")
        assert not df.iloc[-1].isna().any()

    def test_a_single_source_frame_is_not_trimmed(
        self, store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control for the trim: with one source there is nothing to
        reconcile, and a store column that stopped early (a stale or dead
        symbol) must not cut every other column's history short."""
        _seed(store, "AAPL", {"2026-08-03": 212.0, "2026-08-07": 214.0})
        _seed(store, "DEAD", {"2026-08-03": 5.0, "2026-08-04": 5.0})
        calls = _ranged_download(monkeypatch, {})

        df = MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(
            ["AAPL", "DEAD"], START, END
        )

        assert calls["end"] is None  # the store covered both
        assert df.index[-1] == pd.Timestamp(END)

    def test_a_stale_sole_vendor_column_does_not_cut_the_store_columns(
        self, store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: ae6c218c8 — the trim took min(store_last, vendor_last), so
        a sole vendor column that went dark (a delisted name) cut every store
        column back to its own last date. AAPL through 2026-10-28 plus OLD
        ending 2025-03-24 came back ending 2025-03-24."""
        _seed(store, "AAPL", {"2026-08-03": 212.0, "2026-08-14": 214.0})
        _ranged_download(monkeypatch, {"OLD": {"2026-08-03": 9.0, "2026-08-04": 9.5}})

        df = MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(
            ["AAPL", "OLD"], START, date(2026, 8, 14)
        )

        assert df.index[-1] == pd.Timestamp("2026-08-14")
        assert df.loc[pd.Timestamp("2026-08-14"), "AAPL"] == pytest.approx(214.0)

    def test_a_weekend_sized_lag_between_the_sources_is_still_trimmed(
        self, store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The store stops on a Friday, the vendor serves through the next
        Tuesday (four calendar days): still the day-late shape, still trimmed."""
        _seed(store, "AAPL", {"2026-08-03": 212.0, "2026-08-07": 214.0})
        _ranged_download(
            monkeypatch,
            {"CSCO": {"2026-08-03": 110.0, "2026-08-07": 112.0, "2026-08-11": 113.0}},
        )

        df = MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(
            ["AAPL", "CSCO"], START, date(2026, 8, 12)
        )

        assert df.index[-1] == pd.Timestamp("2026-08-07")

    def test_a_trim_names_the_dropped_dates_and_the_columns_to_stderr(
        self,
        store: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _seed(store, "AAPL", {"2026-08-03": 212.0, "2026-08-06": 213.0})
        _ranged_download(
            monkeypatch,
            {"CSCO": {"2026-08-03": 110.0, "2026-08-06": 111.0, "2026-08-07": 112.0}},
        )

        MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(
            ["AAPL", "CSCO"], START, date(2026, 8, 8)
        )

        err = capsys.readouterr().err
        assert "2026-08-07" in err
        assert "1 vendor-served column" in err

    def test_no_trim_prints_nothing(
        self,
        store: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _seed(store, "AAPL", {"2026-08-03": 212.0, "2026-08-07": 214.0})
        _ranged_download(monkeypatch, {"CSCO": {"2026-08-03": 110.0, "2026-08-07": 112.0}})

        MarketDataFetcher(cache_dir=tmp_path / "c").fetch_prices(["AAPL", "CSCO"], START, END)

        assert capsys.readouterr().err == ""
