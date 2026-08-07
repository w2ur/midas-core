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
from engine.quotes import vendor_unit_scale


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


def _store_series(ticker: str, start: date, end: date) -> pd.Series | None:
    """One ticker's raw-close series from the store, or None if uncovered.

    "Uncovered" means: absent from the store, or its earliest row postdates
    `start`, or it has no row inside the window.

    Raw `close`, never `adj_close` — see the `engine.ohlcv_store` module
    docstring.
    """
    rows = _read_store_file(ticker)
    if rows is None:
        return None
    if min(r["date"] for r in rows) > start.isoformat():
        return None
    series_data = {
        r["date"]: r.get("close")
        for r in rows
        if start.isoformat() <= r["date"] <= end.isoformat()
    }
    if not series_data:
        return None
    idx = pd.to_datetime(sorted(series_data.keys()))
    return pd.Series([series_data[d.strftime("%Y-%m-%d")] for d in idx], index=idx)


def _load_prices_from_store(
    tickers: list[str], start: date, end: date
) -> tuple[dict[str, pd.Series], list[str]]:
    """Split *tickers* into store-served series and the ones the store cannot cover.

    Per-ticker, not all-or-nothing (2026-08-07 review, W7.1). This used to
    return `None` for the **whole request** the moment one ticker was missing
    or short, which sent every other ticker — all of them present, all of them
    already ISO-normalised in the store — down the raw-vendor path instead.
    One thin or newly-listed name was enough to change the units of an entire
    backtest.
    """
    served: dict[str, pd.Series] = {}
    missing: list[str] = []
    for ticker in tickers:
        series = _store_series(ticker, start, end)
        if series is None:
            missing.append(ticker)
        else:
            served[ticker] = series
    return served, missing


def _normalise_vendor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Scale a yfinance close frame from vendor units into ISO currency.

    The store is ISO-denominated at ingest, so anything served from it is
    already in the ticker's ISO currency. A yfinance fallback is **not** —
    `LLOY.L` arrives at 116.60 meaning GBP 1.166. Mixing the two in one frame
    put a 100x step in the middle of a price series, and the backtester is
    the product that reads it.

    Mirrors `scripts.fetch_ohlcv._normalise_vendor_units`, which does the same
    job on the ingest path. Columns are tickers here (a close-only frame), so
    the scale is per column rather than per price-column.
    """
    if df.empty:
        return df
    out = df.copy()
    for ticker in out.columns:
        scale = vendor_unit_scale(str(ticker))
        if scale != 1.0:
            out[ticker] = out[ticker] * scale
    return out


def _vendor_close(raw: pd.DataFrame) -> pd.DataFrame | None:
    """The `Close` block of a yfinance result, or None if it served nothing.

    yfinance returns MultiIndex columns regardless of ticker count — level 0
    is the price type, level 1 the symbol. An empty response has no `Close`
    level at all, and `raw["Close"]` raises `KeyError` on it. A vendor miss is
    not an exception here: the store may have served every other column of the
    request, and raising would discard those too.

    Callers download with `auto_adjust=False`, so level 0 carries `Close` and
    `Adj Close` as separate blocks and this picks the raw one — the same basis
    the store serves. Under `auto_adjust=True` there is only a `Close` block
    and it is the *adjusted* series, which would put a store-served raw column
    and a vendor-served total-return column side by side in one frame. Same
    shape of defect as the pence/pounds mixing W7.1 fixed, one field over.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"] if "Close" in raw.columns.get_level_values(0) else None
    return raw[["Close"]] if "Close" in raw.columns else None


def _close_frame(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Close prices for *tickers* over [start, end], every column ISO-denominated.

    Store first, per ticker; yfinance only for the ones the store cannot
    cover, and normalised into ISO on the way in so the two sources cannot
    end up in different units inside one frame. A ticker neither source can
    serve is absent from the result rather than present and empty.

    The single composition point for `fetch_prices` and `fetch_benchmarks`,
    which previously each had their own copy of "try the store, else fetch
    everything" — and each carried the same two defects (W7.1).
    """
    served, missing = _load_prices_from_store(tickers, start, end)
    if missing:
        raw = yf.download(
            missing,
            start=str(start),
            end=str(end),
            auto_adjust=False,  # raw Close, matching the store's basis
            progress=False,
        )
        close = _vendor_close(raw)
        if close is not None:
            fetched = _normalise_vendor_frame(close)
            for ticker in fetched.columns:
                served[str(ticker)] = fetched[ticker]

    # Preserve the caller's column order.
    df = pd.DataFrame({t: served[t] for t in tickers if t in served})
    df.index.name = "Date"
    return df


def _latest_close_from_store(ticker: str) -> float | None:
    """Return the latest raw close for a ticker from the store.

    Raw `close`, never `adj_close` — see the `engine.ohlcv_store` module
    docstring.
    """
    rows = _read_store_file(ticker)
    if not rows:
        return None
    latest = max(rows, key=lambda r: r["date"])
    val = latest.get("close")
    return float(val) if val is not None else None


def latest_close_and_date_from_store(ticker: str) -> tuple[float, str] | None:
    """Return (raw close, ISO-date) of the most recent row for a ticker, or None."""
    rows = _read_store_file(ticker)
    if not rows:
        return None
    latest = max(rows, key=lambda r: r["date"])
    val = latest.get("close")
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

        df = self._normalize_index(_close_frame(tickers, start, end))
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

        close = _close_frame(tickers, start, end)
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
            auto_adjust=False,  # raw Close, matching the store's basis
            progress=False,
        )

        vendor_close = _vendor_close(raw)
        if vendor_close is None:
            return result
        close = _normalise_vendor_frame(vendor_close)
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
