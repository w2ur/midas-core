"""Tests for the NO_DATA sentinel pattern in engine/market_data.py.

DESIGN RATIONALE — minimal blast-radius approach
-------------------------------------------------
Three call-site patterns were audited before deciding where to raise:

1. scripts/fetch_market_data.py → latest_close_and_date_from_store(ticker)
   Pattern: iterates multiple fallback tickers, checks `if result is None: continue`.
   This is a legitimate skip-on-missing pattern. We MUST NOT change the return
   signature of latest_close_and_date_from_store — it would break this caller.

2. scripts/run_backtest.py + scripts/run_all_combos.py → fetcher.fetch_prices(...)
   Pattern: wrapped in `try/except Exception`, already handles failures.
   These could absorb a raise if needed, but fetch_prices is a bulk call (list of
   tickers), so raising on ANY missing ticker is too aggressive — it would break
   valid prefetch loops for large universes (sp500 etc.) that may have partial coverage.

3. engine/market_data.py fetch_current_prices() → internal _latest_close_from_store()
   Pattern: bulk loop, silently skips tickers absent from the store. Returns a partial
   dict. Agent callers must tolerate missing entries. DO NOT change this behaviour.

CHOSEN DESIGN:
- Add NoMarketDataError exception (typed, carries symbol).
- Add no_data_sentinel(symbol) helper — pure string, no side effects.
- Add get_latest_price(symbol) — NEW strict single-symbol lookup. Raises
  NoMarketDataError when the symbol is absent from the committed OHLCV store.
  This is the agent-context-builder entrypoint; it never silently returns None.
- All existing callers are unchanged — they use the existing None-returning helpers.
- The bulk fetch_current_prices() remains as-is (returns partial dict, skips missing).
"""

import json
import pytest
from pathlib import Path

from engine.market_data import (
    NoMarketDataError,
    no_data_sentinel,
    get_latest_price,
)


# ---------------------------------------------------------------------------
# NoMarketDataError — exception contract
# ---------------------------------------------------------------------------


class TestNoMarketDataError:
    def test_is_exception_subclass(self):
        assert issubclass(NoMarketDataError, Exception)

    def test_carries_symbol_attribute(self):
        err = NoMarketDataError("FAKE")
        assert err.symbol == "FAKE"

    def test_message_contains_symbol(self):
        err = NoMarketDataError("FAKE")
        assert "FAKE" in str(err)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(NoMarketDataError) as exc_info:
            raise NoMarketDataError("GHOST")
        assert exc_info.value.symbol == "GHOST"

    def test_can_be_caught_as_generic_exception(self):
        with pytest.raises(Exception):
            raise NoMarketDataError("GHOST")


# ---------------------------------------------------------------------------
# no_data_sentinel — exact-match string contract
# ---------------------------------------------------------------------------


class TestNoDataSentinel:
    def test_exact_string_format(self):
        result = no_data_sentinel("AAPL")
        assert (
            result
            == "NO_DATA_AVAILABLE: AAPL not in committed store — do not fabricate"
        )

    def test_symbol_interpolated_correctly(self):
        result = no_data_sentinel("BTC-USD")
        assert "BTC-USD" in result

    def test_different_symbols_produce_different_strings(self):
        assert no_data_sentinel("AAPL") != no_data_sentinel("MSFT")

    def test_returns_str(self):
        assert isinstance(no_data_sentinel("X"), str)


# ---------------------------------------------------------------------------
# get_latest_price — strict single-symbol lookup (raises on missing)
# ---------------------------------------------------------------------------


def _write_store(store_dir: Path, ticker: str, rows: list[dict]) -> None:
    """Write a minimal OHLCV JSONL file into a temp store directory."""
    path = store_dir / f"{ticker}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class TestGetLatestPrice:
    def test_garbage_ticker_raises_no_market_data_error(self, midas_data_root):
        """Unknown symbol must raise NoMarketDataError, never silently return None/0."""
        from engine.config import get_config

        ohlcv = get_config().ohlcv_dir
        ohlcv.mkdir(parents=True, exist_ok=True)
        with pytest.raises(NoMarketDataError) as exc_info:
            get_latest_price("DEFINITELY_NOT_A_REAL_TICKER_XYZXYZ")
        assert exc_info.value.symbol == "DEFINITELY_NOT_A_REAL_TICKER_XYZXYZ"

    def test_symbol_in_exception_message(self, midas_data_root):
        from engine.config import get_config

        ohlcv = get_config().ohlcv_dir
        ohlcv.mkdir(parents=True, exist_ok=True)
        with pytest.raises(NoMarketDataError) as exc_info:
            get_latest_price("GHOST")
        assert "GHOST" in str(exc_info.value)

    def test_known_ticker_returns_float(self, midas_data_root):
        from engine.config import get_config

        ohlcv = get_config().ohlcv_dir
        ohlcv.mkdir(parents=True, exist_ok=True)
        _write_store(
            ohlcv,
            "AAPL",
            [
                {"date": "2026-06-10", "close": 193.5, "adj_close": 193.5},
                {"date": "2026-06-11", "close": 195.0, "adj_close": 195.0},
            ],
        )
        price = get_latest_price("AAPL")
        assert isinstance(price, float)
        assert price == pytest.approx(195.0)

    def test_returns_latest_row_not_first(self, midas_data_root):
        from engine.config import get_config

        ohlcv = get_config().ohlcv_dir
        ohlcv.mkdir(parents=True, exist_ok=True)
        _write_store(
            ohlcv,
            "MSFT",
            [
                {"date": "2026-06-09", "close": 400.0, "adj_close": 400.0},
                {"date": "2026-06-11", "close": 420.0, "adj_close": 420.0},
                {"date": "2026-06-10", "close": 410.0, "adj_close": 410.0},
            ],
        )
        price = get_latest_price("MSFT")
        assert price == pytest.approx(420.0)

    def test_serves_row_carrying_only_close(self, midas_data_root):
        """A row with no `adj_close` at all is served from `close`.

        Trivially true since 2026-08-07 (every read path takes `close`), but
        kept: it was the fallback branch before the basis change and the
        shape still occurs in crypto rows.
        """
        from engine.config import get_config

        ohlcv = get_config().ohlcv_dir
        ohlcv.mkdir(parents=True, exist_ok=True)
        _write_store(
            ohlcv,
            "BTC-USD",
            [
                {"date": "2026-06-11", "close": 67000.0},
            ],
        )
        price = get_latest_price("BTC-USD")
        assert price == pytest.approx(67000.0)

    def test_raises_no_market_data_not_value_error(self, midas_data_root):
        """Must raise NoMarketDataError specifically, not a generic ValueError/None."""
        from engine.config import get_config

        ohlcv = get_config().ohlcv_dir
        ohlcv.mkdir(parents=True, exist_ok=True)
        with pytest.raises(NoMarketDataError):
            get_latest_price("NOTHERE")


# ---------------------------------------------------------------------------
# Bulk path preservation — fetch_current_prices still skips missing tickers
# ---------------------------------------------------------------------------


class TestBulkPathPreservation:
    """fetch_current_prices must still return a partial dict for bulk lookups.

    Agents and portfolio valuation code call this for lists of tickers and
    rely on missing tickers simply being absent from the returned dict (not
    raising). This test confirms the bulk path is unchanged.
    """

    def test_missing_ticker_absent_from_result_not_raises(self, midas_data_root):
        from engine.config import get_config

        ohlcv = get_config().ohlcv_dir
        ohlcv.mkdir(parents=True, exist_ok=True)
        _write_store(
            ohlcv,
            "AAPL",
            [
                {"date": "2026-06-11", "close": 193.0, "adj_close": 193.0},
            ],
        )
        from unittest.mock import patch, MagicMock
        import pandas as pd

        # Patch yfinance so no network call happens for the missing ticker
        mock_raw = MagicMock()
        mock_close = pd.DataFrame({"GHOST": pd.Series([], dtype=float)})
        mock_raw.__getitem__ = lambda self, key: mock_close
        with patch("engine.market_data.yf.download", return_value=mock_raw):
            from engine.market_data import MarketDataFetcher

            fetcher = MarketDataFetcher()
            result = fetcher.fetch_current_prices(["AAPL", "GHOST"])

        assert "AAPL" in result
        assert result["AAPL"] == pytest.approx(193.0)
        assert "GHOST" not in result  # silently absent, not raised

    def test_all_known_tickers_returned(self, midas_data_root):
        from engine.config import get_config

        ohlcv = get_config().ohlcv_dir
        ohlcv.mkdir(parents=True, exist_ok=True)
        _write_store(
            ohlcv,
            "AAPL",
            [
                {"date": "2026-06-11", "close": 193.0, "adj_close": 193.0},
            ],
        )
        _write_store(
            ohlcv,
            "MSFT",
            [
                {"date": "2026-06-11", "close": 410.0, "adj_close": 410.0},
            ],
        )
        from engine.market_data import MarketDataFetcher

        fetcher = MarketDataFetcher()
        result = fetcher.fetch_current_prices(["AAPL", "MSFT"])
        assert "AAPL" in result
        assert "MSFT" in result
