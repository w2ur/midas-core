"""Market data fetcher — yfinance wrapper with optional disk caching.

Preferred data source is the committed OHLCV store at data/market/ohlcv/,
populated by the fetch-ohlcv GitHub Action. In sandboxed environments where
yfinance is blocked or rate-limited, the store is the only way to get data.
yfinance is used as a fallback only for ranges the store doesn't cover.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from engine.config import get_config


# ---------------------------------------------------------------------------
# Benchmark ticker mapping
# ---------------------------------------------------------------------------

BENCHMARK_TICKERS: dict[str, str] = {
    "sp500": "^GSPC",
    "msci_world": "URTH",
    "gold": "GC=F",
    "btc": "BTC-USD",
}


# ---------------------------------------------------------------------------
# Committed OHLCV store (data/market/ohlcv/{SYMBOL}.jsonl)
# ---------------------------------------------------------------------------


def _read_store_file(ticker: str) -> list[dict] | None:
    """Load all rows for a ticker from the committed JSONL store, or None."""
    path = get_config().ohlcv_dir / f"{ticker}.jsonl"
    if not path.exists():
        return None
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows if rows else None


def _load_prices_from_store(
    tickers: list[str], start: date, end: date
) -> pd.DataFrame | None:
    """Return adjusted-close DataFrame for tickers over [start, end] from the store.

    Returns None if any ticker is missing from the store, or if the store
    doesn't reach back to `start`. Partial coverage falls back to yfinance.
    """
    columns: dict[str, pd.Series] = {}
    for ticker in tickers:
        rows = _read_store_file(ticker)
        if rows is None:
            return (
                None  # Missing entirely — fall back to yfinance for the whole request
            )
        earliest = min(r["date"] for r in rows)
        if earliest > start.isoformat():
            return None  # Store doesn't cover the requested range
        # Use adj_close when present; fall back to close for instruments without splits/divs.
        series_data = {
            r["date"]: (
                r.get("adj_close") if r.get("adj_close") is not None else r.get("close")
            )
            for r in rows
            if start.isoformat() <= r["date"] <= end.isoformat()
        }
        if not series_data:
            return None
        idx = pd.to_datetime(sorted(series_data.keys()))
        columns[ticker] = pd.Series(
            [series_data[d.strftime("%Y-%m-%d")] for d in idx], index=idx
        )

    df = pd.DataFrame(columns)
    df.index.name = "Date"
    return df


def _latest_close_from_store(ticker: str) -> float | None:
    """Return the latest adj_close (or close) for a ticker from the store."""
    rows = _read_store_file(ticker)
    if not rows:
        return None
    latest = max(rows, key=lambda r: r["date"])
    val = latest.get("adj_close")
    if val is None:
        val = latest.get("close")
    return float(val) if val is not None else None


def latest_close_and_date_from_store(ticker: str) -> tuple[float, str] | None:
    """Return (close, ISO-date) of the most recent row for a ticker, or None."""
    rows = _read_store_file(ticker)
    if not rows:
        return None
    latest = max(rows, key=lambda r: r["date"])
    val = (
        latest.get("adj_close")
        if latest.get("adj_close") is not None
        else latest.get("close")
    )
    if val is None:
        return None
    return (float(val), str(latest["date"]))


# ---------------------------------------------------------------------------
# NO_DATA sentinel — anti-fabrication primitives
# ---------------------------------------------------------------------------


class NoMarketDataError(Exception):
    """Raised when a symbol is not present in the committed OHLCV store.

    Carrying the symbol as a typed attribute lets callers log or surface it
    precisely without parsing the message string.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        super().__init__(
            f"{symbol} not found in committed OHLCV store — "
            "do not use fabricated or stale prices"
        )


def no_data_sentinel(symbol: str) -> str:
    """Return a prompt-facing sentinel string for an unknown symbol.

    Future agent-context builders embed this string so the LLM is explicitly
    told not to invent a price rather than silently receiving an empty field.
    The exact format is part of the public contract — do not change it.
    """
    return f"NO_DATA_AVAILABLE: {symbol} not in committed store — do not fabricate"


def get_latest_price(symbol: str) -> float:
    """Return the latest closing price for a single symbol from the committed store.

    This is the strict single-symbol entrypoint for agent context builders.
    Unlike the bulk helpers (fetch_current_prices, _latest_close_from_store),
    this function NEVER returns None or silently degrades — it raises
    NoMarketDataError when the symbol is absent. That loud failure is
    intentional: an unknown symbol must never silently price at 0/empty
    and feed fabricated context to an LLM.

    Raises
    ------
    NoMarketDataError
        If the symbol has no file in the committed OHLCV store, or if the
        store file exists but contains no valid price rows.
    """
    price = _latest_close_from_store(symbol)
    if price is None:
        raise NoMarketDataError(symbol)
    return price


class MarketDataFetcher:
    """Fetches prices, dividends, and benchmarks via yfinance with disk caching.

    Parameters
    ----------
    cache_dir:
        Optional directory for parquet-based query caching. When provided,
        identical queries are served from disk on subsequent calls.
    """

    def __init__(self, cache_dir: Optional[str | Path] = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices for the given tickers.

        Returns a DataFrame with dates as index and tickers as columns.
        Serves from the committed OHLCV store when it covers the range;
        otherwise falls back to yfinance.
        """
        cache_key = self._make_cache_key(
            "prices", tickers=sorted(tickers), start=str(start), end=str(end)
        )
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        store_df = _load_prices_from_store(tickers, start, end)
        if store_df is not None:
            store_df = self._normalize_index(store_df)
            self._save_cache(cache_key, store_df)
            return store_df

        raw = yf.download(
            tickers,
            start=str(start),
            end=str(end),
            auto_adjust=True,
            progress=False,
        )

        # yfinance always returns MultiIndex columns regardless of ticker count.
        # Level 0 is price type (Close, Open, …), level 1 is ticker symbol.
        df = raw["Close"]
        df = self._normalize_index(df)

        self._save_cache(cache_key, df)
        return df

    def fetch_benchmarks(self, start: date, end: date) -> pd.DataFrame:
        """Fetch all four benchmark assets and return with friendly column names.

        Columns: sp500, msci_world, gold, btc
        """
        tickers = list(BENCHMARK_TICKERS.values())
        cache_key = self._make_cache_key("benchmarks", start=str(start), end=str(end))
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        store_df = _load_prices_from_store(tickers, start, end)
        if store_df is not None:
            reverse_map = {v: k for k, v in BENCHMARK_TICKERS.items()}
            renamed = store_df.rename(columns=reverse_map)[
                list(BENCHMARK_TICKERS.keys())
            ]
            renamed = self._normalize_index(renamed)
            self._save_cache(cache_key, renamed)
            return renamed

        raw = yf.download(
            tickers,
            start=str(start),
            end=str(end),
            auto_adjust=True,
            progress=False,
        )

        # Multiple tickers always → MultiIndex
        close = raw["Close"]

        # Rename yfinance tickers to friendly names
        reverse_map = {v: k for k, v in BENCHMARK_TICKERS.items()}
        df = close.rename(columns=reverse_map)[list(BENCHMARK_TICKERS.keys())]
        df = self._normalize_index(df)

        self._save_cache(cache_key, df)
        return df

    def fetch_dividends(self, ticker: str, start: date, end: date) -> pd.Series:
        """Fetch dividend history for a single ticker filtered to the date range.

        Returns a pd.Series with timezone-naive DatetimeIndex.
        """
        raw_divs = yf.Ticker(ticker).dividends

        # yfinance may return a DataFrame with a "Dividends" column — extract it as a Series
        if isinstance(raw_divs, pd.DataFrame):
            divs: pd.Series = raw_divs["Dividends"]
        else:
            divs = raw_divs

        # Strip timezone info so comparisons work consistently
        if divs.index.tz is not None:
            divs.index = divs.index.tz_localize(None)

        mask = (divs.index.date >= start) & (divs.index.date <= end)
        return divs.loc[mask]

    def fetch_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch the most recent closing price for each ticker.

        Reads the latest row from the committed OHLCV store first; falls back
        to yfinance only for tickers missing from the store.
        Returns dict mapping ticker → float price.
        """
        result: dict[str, float] = {}
        missing: list[str] = []
        for ticker in tickers:
            latest = _latest_close_from_store(ticker)
            if latest is not None:
                result[ticker] = latest
            else:
                missing.append(ticker)

        if not missing:
            return result

        raw = yf.download(
            missing,
            period="5d",
            auto_adjust=True,
            progress=False,
        )

        close = raw["Close"]
        for ticker in missing:
            try:
                series = close[ticker].dropna()
                if len(series) > 0:
                    result[ticker] = float(series.iloc[-1])
            except (KeyError, IndexError):
                pass
        return result

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _normalize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DatetimeIndex to second resolution for consistent roundtrips."""
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = df.index.as_unit("s")
        return df

    def _make_cache_key(self, prefix: str, **kwargs) -> str:
        """Build a deterministic MD5 cache key from the query parameters."""
        payload = json.dumps({"prefix": prefix, **kwargs}, sort_keys=True)
        digest = hashlib.md5(payload.encode()).hexdigest()
        return digest

    def _cache_path(self, key: str) -> Path:
        assert self._cache_dir is not None
        return self._cache_dir / f"{key}.parquet"

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        if self._cache_dir is None:
            return None
        path = self._cache_path(key)
        if path.exists():
            df = pd.read_parquet(path)
            # Normalize DatetimeIndex resolution — parquet may store ms vs s
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.as_unit("s")
            return df
        return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        if self._cache_dir is None:
            return
        df.to_parquet(self._cache_path(key))
