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

from datetime import date
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
