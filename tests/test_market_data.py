"""Integration tests for engine/market_data.py — hits Yahoo Finance."""

import pytest
import pandas as pd
from datetime import date
from pathlib import Path

from engine.market_data import MarketDataFetcher


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fetcher():
    return MarketDataFetcher()


@pytest.fixture
def fetcher_with_cache(tmp_path):
    return MarketDataFetcher(cache_dir=tmp_path)


START = date(2024, 1, 2)
END = date(2024, 1, 31)


# ---------------------------------------------------------------------------
# fetch_prices — single ticker
# ---------------------------------------------------------------------------

class TestFetchPricesSingle:
    def test_returns_dataframe(self, fetcher):
        df = fetcher.fetch_prices(["AAPL"], START, END)
        assert isinstance(df, pd.DataFrame)

    def test_has_ticker_column(self, fetcher):
        df = fetcher.fetch_prices(["AAPL"], START, END)
        assert "AAPL" in df.columns

    def test_index_is_datetime(self, fetcher):
        df = fetcher.fetch_prices(["AAPL"], START, END)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_values_are_positive(self, fetcher):
        df = fetcher.fetch_prices(["AAPL"], START, END)
        assert (df["AAPL"].dropna() > 0).all()


# ---------------------------------------------------------------------------
# fetch_prices — multiple tickers
# ---------------------------------------------------------------------------

class TestFetchPricesMultiple:
    def test_returns_dataframe_with_all_columns(self, fetcher):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        df = fetcher.fetch_prices(tickers, START, END)
        assert isinstance(df, pd.DataFrame)
        for ticker in tickers:
            assert ticker in df.columns

    def test_multiple_tickers_have_rows(self, fetcher):
        df = fetcher.fetch_prices(["AAPL", "MSFT"], START, END)
        assert len(df) > 0

    def test_only_requested_columns_present(self, fetcher):
        tickers = ["AAPL", "MSFT"]
        df = fetcher.fetch_prices(tickers, START, END)
        assert set(df.columns) == set(tickers)


# ---------------------------------------------------------------------------
# fetch_benchmarks
# ---------------------------------------------------------------------------

class TestFetchBenchmarks:
    def test_returns_all_four_columns(self, fetcher):
        df = fetcher.fetch_benchmarks(START, END)
        assert isinstance(df, pd.DataFrame)
        for col in ["sp500", "msci_world", "gold", "btc"]:
            assert col in df.columns

    def test_has_rows(self, fetcher):
        df = fetcher.fetch_benchmarks(START, END)
        assert len(df) > 0

    def test_values_are_positive(self, fetcher):
        df = fetcher.fetch_benchmarks(START, END)
        for col in ["sp500", "msci_world", "gold", "btc"]:
            assert (df[col].dropna() > 0).all()


# ---------------------------------------------------------------------------
# fetch_dividends
# ---------------------------------------------------------------------------

class TestFetchDividends:
    def test_returns_series(self, fetcher):
        result = fetcher.fetch_dividends("AAPL", START, END)
        assert isinstance(result, pd.Series)

    def test_aapl_has_dividends_in_2024(self, fetcher):
        # AAPL pays quarterly dividends — use a full year to guarantee a hit
        result = fetcher.fetch_dividends("AAPL", date(2024, 1, 1), date(2024, 12, 31))
        assert len(result) > 0

    def test_dividends_are_positive(self, fetcher):
        result = fetcher.fetch_dividends("AAPL", date(2024, 1, 1), date(2024, 12, 31))
        assert (result > 0).all()

    def test_dividends_within_date_range(self, fetcher):
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)
        result = fetcher.fetch_dividends("AAPL", start, end)
        if len(result) > 0:
            assert result.index.min().date() >= start
            assert result.index.max().date() <= end


# ---------------------------------------------------------------------------
# fetch_current_prices
# ---------------------------------------------------------------------------

class TestFetchCurrentPrices:
    def test_returns_dict(self, fetcher):
        result = fetcher.fetch_current_prices(["AAPL"])
        assert isinstance(result, dict)

    def test_has_all_tickers(self, fetcher):
        tickers = ["AAPL", "MSFT"]
        result = fetcher.fetch_current_prices(tickers)
        for ticker in tickers:
            assert ticker in result

    def test_prices_are_positive_floats(self, fetcher):
        result = fetcher.fetch_current_prices(["AAPL", "MSFT"])
        for ticker, price in result.items():
            assert isinstance(price, float)
            assert price > 0


# ---------------------------------------------------------------------------
# Disk caching
# ---------------------------------------------------------------------------

class TestDiskCaching:
    def test_second_call_returns_same_data(self, fetcher_with_cache):
        df1 = fetcher_with_cache.fetch_prices(["AAPL"], START, END)
        df2 = fetcher_with_cache.fetch_prices(["AAPL"], START, END)
        pd.testing.assert_frame_equal(df1, df2)

    def test_parquet_file_created(self, fetcher_with_cache, tmp_path):
        fetcher_with_cache.fetch_prices(["AAPL"], START, END)
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1

    def test_different_queries_produce_different_cache_files(self, fetcher_with_cache, tmp_path):
        fetcher_with_cache.fetch_prices(["AAPL"], START, END)
        fetcher_with_cache.fetch_prices(["MSFT"], START, END)
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 2

    def test_no_cache_dir_does_not_create_files(self, fetcher, tmp_path):
        fetcher.fetch_prices(["AAPL"], START, END)
        # No parquet files should appear in the current dir
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 0
