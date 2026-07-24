"""Index universe resolvers.

US: S&P 500, Dow 30, Nasdaq 100.
EU: CAC 40, DAX, FTSE 100, STOXX Europe 600.

Universe lists live in `data/universes/{name}.json`, **committed** to the
repo. The cloud sandbox has no outbound HTTP, so resolvers must NEVER hit
Wikipedia at runtime. File presence is authoritative. Periodic refresh runs
out-of-band (manual `scripts/refresh_universes.py` or the GitHub Actions
weekly cron `refresh-universes.yml`) and commits the diff.

This module previously kept these files under `data/cache/universes/`
(gitignored) with a 24-hour TTL — that combination crashed every cloud
session whose cache was older than a day. Apr 29 incident.
"""

from __future__ import annotations

import io
import json
import logging
import urllib.request

import pandas as pd

from engine.config import get_config

logger = logging.getLogger(__name__)

_WIKI_USER_AGENT = "midas-fund/0.1 (https://github.com/w2ur/midas; research)"


def _fetch_html_tables(url: str) -> list[pd.DataFrame]:
    """Fetch an HTML page with a descriptive User-Agent and parse its tables.

    Wikipedia (and Slickcharts) reject pandas' default Python-urllib UA, so we
    fetch the HTML ourselves before handing it to pd.read_html. Used for both
    the Wikipedia index pages and the Slickcharts Nasdaq-100 source.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _WIKI_USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8")
    return pd.read_html(io.StringIO(html))


def _largest_table_with_column(
    tables: list[pd.DataFrame], column: str
) -> pd.DataFrame | None:
    """Return the largest table containing `column` in its columns.

    Robust against Wikipedia page layout changes: avoids picking small
    "examples" or "recent changes" tables that happen to share a column name.
    """
    candidates = [t for t in tables if column in [str(c) for c in t.columns]]
    if not candidates:
        return None
    return max(candidates, key=len)


def _read_data(name: str) -> list[str] | None:
    """Return committed tickers for `name`, or None if the file is missing.

    No TTL: the file is the source of truth. If you want to refresh from
    Wikipedia, call `refresh_<name>()` explicitly.
    """
    path = get_config().universes_dir / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_data(name: str, tickers: list[str]) -> None:
    """Persist tickers to `data/universes/{name}.json`."""
    data_dir = get_config().universes_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{name}.json").write_text(json.dumps(tickers), encoding="utf-8")


def _normalise(ticker: str) -> str:
    """Replace dots with hyphens for yfinance compatibility (BRK.B → BRK-B)."""
    return ticker.replace(".", "-").strip()


# ---------------------------------------------------------------------------
# S&P 500
# ---------------------------------------------------------------------------


def get_sp500_tickers() -> list[str]:
    """Return committed S&P 500 constituents.

    Reads `data/universes/sp500.json`. Falls back to a Wikipedia refresh ONLY
    when the file is missing — which should never happen in production since
    the file is committed.
    """
    cached = _read_data("sp500")
    if cached is not None:
        return cached
    return refresh_sp500()


def refresh_sp500() -> list[str]:
    """Re-fetch S&P 500 from Wikipedia and overwrite the committed file."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _fetch_html_tables(url)
    table = _largest_table_with_column(tables, "Symbol")
    if table is None:
        raise RuntimeError("S&P 500: no 'Symbol' column on Wikipedia page")
    tickers = sorted({_normalise(str(t)) for t in table["Symbol"].tolist()})
    if len(tickers) < 100:
        raise RuntimeError(
            f"S&P 500: {len(tickers)} tickers — Wikipedia layout may have changed"
        )
    _write_data("sp500", tickers)
    return tickers


# ---------------------------------------------------------------------------
# Dow 30
# ---------------------------------------------------------------------------


def get_dow30_tickers() -> list[str]:
    """Return committed Dow Jones Industrial Average constituents."""
    cached = _read_data("dow30")
    if cached is not None:
        return cached
    return refresh_dow30()


def refresh_dow30() -> list[str]:
    """Re-fetch Dow 30 from Wikipedia and overwrite the committed file."""
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    tables = _fetch_html_tables(url)
    table = _largest_table_with_column(tables, "Symbol")
    if table is None:
        raise RuntimeError("Dow 30: no 'Symbol' column on Wikipedia page")
    raw = [str(t) for t in table["Symbol"].dropna().tolist() if str(t) != "Symbol"]
    tickers = sorted({_normalise(t) for t in raw if t})
    if len(tickers) < 20:
        raise RuntimeError(
            f"Dow 30: {len(tickers)} tickers — Wikipedia layout may have changed"
        )
    _write_data("dow30", tickers)
    return tickers


# ---------------------------------------------------------------------------
# Nasdaq 100
# ---------------------------------------------------------------------------


def get_nasdaq100_tickers() -> list[str]:
    """Return committed Nasdaq-100 constituents."""
    cached = _read_data("nasdaq100")
    if cached is not None:
        return cached
    return refresh_nasdaq100()


# Candidate ticker-column header names on the Slickcharts Nasdaq-100 page,
# tried in order. Tolerant to a future header rename (Slickcharts has already
# forced one source change — see docstring below) without another crash.
_NASDAQ100_SYMBOL_COLUMNS: tuple[str, ...] = ("Symbol", "Ticker")


def refresh_nasdaq100() -> list[str]:
    """Re-fetch Nasdaq-100 from Slickcharts and overwrite the committed file.

    Source moved off Wikipedia on 2026-07-13: the en.wikipedia.org/wiki/Nasdaq-100
    article dropped its constituents table entirely (the "Components" section is
    now just an external link to nasdaq.com), so no column-name variant could
    recover it. Slickcharts publishes a clean weighted table with a "Symbol"
    column (~100 rows, dual-class shares like GOOGL/GOOG included). Column
    detection tries each of `_NASDAQ100_SYMBOL_COLUMNS` in turn so a future
    Slickcharts header rename (e.g. back to "Ticker") degrades gracefully
    instead of raising immediately.
    """
    url = "https://www.slickcharts.com/nasdaq100"
    tables = _fetch_html_tables(url)
    table = None
    symbol_col = None
    for candidate in _NASDAQ100_SYMBOL_COLUMNS:
        table = _largest_table_with_column(tables, candidate)
        if table is not None:
            symbol_col = candidate
            break
    if table is None:
        raise RuntimeError(
            "Nasdaq-100: no "
            f"{' or '.join(repr(c) for c in _NASDAQ100_SYMBOL_COLUMNS)} "
            "column on Slickcharts page"
        )
    raw = [str(t) for t in table[symbol_col].dropna().tolist() if str(t) != symbol_col]
    tickers = sorted({_normalise(t) for t in raw if t})
    if len(tickers) < 90:
        raise RuntimeError(
            f"Nasdaq-100: {len(tickers)} tickers — Slickcharts layout may have changed"
        )
    _write_data("nasdaq100", tickers)
    return tickers


# ---------------------------------------------------------------------------
# EU indices — CAC 40, DAX, FTSE 100, STOXX Europe 600
# ---------------------------------------------------------------------------

# Country → yfinance exchange suffix, used for STOXX 600 (tickers without suffix).
_STOXX_COUNTRY_SUFFIX: dict[str, str] = {
    "Austria": ".VI",
    "Belgium": ".BR",
    "Denmark": ".CO",
    "Finland": ".HE",
    "France": ".PA",
    "Germany": ".DE",
    "Greece": ".AT",
    "Ireland": ".IR",
    "Italy": ".MI",
    "Luxembourg": ".LU",
    "Netherlands": ".AS",
    "Norway": ".OL",
    "Poland": ".WA",
    "Portugal": ".LS",
    "Spain": ".MC",
    "Sweden": ".ST",
    "Switzerland": ".SW",
    "United Kingdom": ".L",
    # Jersey / Bermuda / Israel companies often list on LSE
    "Jersey": ".L",
    "Bermuda": ".L",
    "Israel": ".L",
}


def _clean_ticker(raw: object) -> str | None:
    s = str(raw).strip()
    if not s or s.lower() in ("ticker", "nan", "none", "—"):
        return None
    if "[" in s:
        s = s.split("[", 1)[0].strip()
    return s or None


def get_cac40_tickers() -> list[str]:
    """Return committed CAC 40 constituents."""
    cached = _read_data("cac40")
    if cached is not None:
        return cached
    return refresh_cac40()


def refresh_cac40() -> list[str]:
    url = "https://en.wikipedia.org/wiki/CAC_40"
    tables = _fetch_html_tables(url)
    table = _largest_table_with_column(tables, "Ticker")
    if table is None:
        raise RuntimeError("CAC 40: no 'Ticker' column on Wikipedia page")
    tickers = sorted({t for t in (_clean_ticker(v) for v in table["Ticker"]) if t})
    if len(tickers) < 30:
        raise RuntimeError(f"CAC 40: {len(tickers)} tickers — layout changed")
    _write_data("cac40", tickers)
    return tickers


def get_dax_tickers() -> list[str]:
    """Return committed DAX constituents."""
    cached = _read_data("dax")
    if cached is not None:
        return cached
    return refresh_dax()


def refresh_dax() -> list[str]:
    url = "https://en.wikipedia.org/wiki/DAX"
    tables = _fetch_html_tables(url)
    table = _largest_table_with_column(tables, "Ticker")
    if table is None:
        raise RuntimeError("DAX: no 'Ticker' column on Wikipedia page")
    tickers = sorted({t for t in (_clean_ticker(v) for v in table["Ticker"]) if t})
    if len(tickers) < 30:
        raise RuntimeError(f"DAX: {len(tickers)} tickers — layout changed")
    _write_data("dax", tickers)
    return tickers


def get_ftse100_tickers() -> list[str]:
    """Return committed FTSE 100 constituents (.L suffix appended for yfinance)."""
    cached = _read_data("ftse100")
    if cached is not None:
        return cached
    return refresh_ftse100()


def refresh_ftse100() -> list[str]:
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    tables = _fetch_html_tables(url)
    table = _largest_table_with_column(tables, "Ticker")
    if table is None:
        raise RuntimeError("FTSE 100: no 'Ticker' column on Wikipedia page")
    tickers: set[str] = set()
    for raw in table["Ticker"]:
        t = _clean_ticker(raw)
        if t is None:
            continue
        if not t.endswith(".L"):
            t = f"{t}.L"
        tickers.add(t)
    result = sorted(tickers)
    if len(result) < 80:
        raise RuntimeError(f"FTSE 100: {len(result)} tickers — layout changed")
    _write_data("ftse100", result)
    return result


def get_stoxx600_tickers() -> list[str]:
    """Return committed STOXX Europe 600 constituents (with country suffixes)."""
    cached = _read_data("stoxx600")
    if cached is not None:
        return cached
    return refresh_stoxx600()


def refresh_stoxx600() -> list[str]:
    url = "https://en.wikipedia.org/wiki/STOXX_Europe_600"
    tables = _fetch_html_tables(url)
    table = _largest_table_with_column(tables, "Ticker")
    if table is None:
        raise RuntimeError("STOXX 600: no 'Ticker' column on Wikipedia page")
    if "Country" not in [str(c) for c in table.columns]:
        raise RuntimeError("STOXX 600: no 'Country' column — cannot map suffixes")
    tickers: set[str] = set()
    for _, row in table.iterrows():
        t = _clean_ticker(row["Ticker"])
        country = str(row["Country"]).strip() if row["Country"] is not None else ""
        if t is None or not country:
            continue
        suffix = _STOXX_COUNTRY_SUFFIX.get(country)
        if suffix is None:
            continue
        if not t.endswith(suffix):
            t = f"{t}{suffix}"
        tickers.add(t)
    result = sorted(tickers)
    if len(result) < 400:
        raise RuntimeError(f"STOXX 600: {len(result)} tickers — layout changed")
    _write_data("stoxx600", result)
    return result


# ---------------------------------------------------------------------------
# Bulk refresh
# ---------------------------------------------------------------------------


# Canonical {name: refresher-function-name} mapping — the single source of
# truth for which indexes exist. `scripts/refresh_universes.py` derives its
# skip report from these keys, so adding an index here is the only change
# needed. Values are attribute names resolved at call time (late-bound) so
# tests can monkeypatch the individual refresh_* functions.
INDEX_REFRESHERS = {
    "sp500": "refresh_sp500",
    "dow30": "refresh_dow30",
    "nasdaq100": "refresh_nasdaq100",
    "cac40": "refresh_cac40",
    "dax": "refresh_dax",
    "ftse100": "refresh_ftse100",
    "stoxx600": "refresh_stoxx600",
}


def refresh_all_indexes() -> dict[str, int]:
    """Re-fetch every index universe from Wikipedia/Slickcharts and overwrite
    committed files.

    Used by `scripts/refresh_universes.py` and the weekly GitHub Actions cron.
    Each index refreshes independently: a scraper that raises (e.g. an
    upstream layout change like the 2026-07-13 Nasdaq-100/Wikipedia break)
    logs a warning and is skipped rather than aborting the whole run — one
    broken source must never take the other six indexes down with it. The
    committed file for a skipped index is left untouched at its last known
    -good value. Returns {name: ticker_count} for indexes that succeeded
    only; a failed index is simply absent from the result (callers alert on
    the gap — the weekly workflow exits non-zero so the failure still emails).
    """
    results: dict[str, int] = {}
    for name, fn_name in INDEX_REFRESHERS.items():
        refresher = globals()[fn_name]
        try:
            results[name] = len(refresher())
        except Exception as exc:
            logger.warning("refresh_all_indexes: skipping %s — %s", name, exc)
    return results
