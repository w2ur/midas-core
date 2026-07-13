"""OHLCV ingestion primitives — column normalization, row coercion, idempotent merge.

Pure logic extracted from ``scripts/fetch_ohlcv.py`` so it can be unit-tested with
synthetic yfinance-shaped frames without touching the network. ``scripts/fetch_ohlcv.py``
is a thin CLI over these functions.

On-disk format (``data/market/ohlcv/{SYMBOL}.jsonl``): one JSON object per line, keys
in the order ``date, open, high, low, close, adj_close, volume``, serialized with
``json.dumps()`` defaults. This byte layout is committed to git and read by the
sandboxed agent — preserving it exactly is a hard requirement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a yfinance MultiIndex column frame to its level-0 field names.

    yfinance returns a ``MultiIndex`` (field, symbol) for single-symbol downloads
    in some versions. Reading Open/High/Low/Close/Adj Close/Volume needs flat
    column labels. A frame that is already flat is returned unchanged.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def safe_float(v: object) -> float | None:
    """Coerce a DataFrame cell to float, defending against accidental Series values."""
    if v is None:
        return None
    if isinstance(v, pd.Series):
        if v.empty:
            return None
        v = v.iloc[0]
    if pd.isna(v):
        return None
    return float(v)


def safe_int(v: object) -> int | None:
    """Coerce a DataFrame cell to int, defending against accidental Series values."""
    if v is None:
        return None
    if isinstance(v, pd.Series):
        if v.empty:
            return None
        v = v.iloc[0]
    if pd.isna(v):
        return None
    return int(v)


def existing_dates(path: Path) -> set[str]:
    """Return the set of ISO dates already present in a ``{SYMBOL}.jsonl`` store file.

    Missing file → empty set. Blank and unparseable lines are skipped so a
    partially-written file never aborts a refresh.
    """
    if not path.exists():
        return set()
    dates: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = row.get("date")
            if d:
                dates.add(d)
    return dates


def row_to_record(row_date: str, row: object) -> dict:
    """Normalize one yfinance frame row into the committed record shape.

    Key order is load-bearing — it defines the on-disk JSON byte layout.
    """
    return {
        "date": row_date,
        "open": safe_float(row.get("Open")),
        "high": safe_float(row.get("High")),
        "low": safe_float(row.get("Low")),
        "close": safe_float(row.get("Close")),
        "adj_close": safe_float(row.get("Adj Close")),
        "volume": safe_int(row.get("Volume")),
    }


def build_new_rows(df: pd.DataFrame, existing: set[str]) -> list[tuple[str, str]]:
    """Merge a fetched frame against already-stored dates.

    Returns a date-sorted list of ``(iso_date, json_line)`` pairs for dates not
    already in ``existing``. Rows whose close is missing are dropped (a store row
    with no close is useless for valuation and fills). Idempotent: re-running with
    a frame whose dates are all present yields an empty list.
    """
    rows_to_append: list[tuple[str, str]] = []
    for ts, row in df.iterrows():
        d = ts.date().isoformat() if hasattr(ts, "date") else str(ts)
        if d in existing:
            continue
        record = row_to_record(d, row)
        if record["close"] is None:
            continue
        rows_to_append.append((d, json.dumps(record)))
    rows_to_append.sort(key=lambda pair: pair[0])
    return rows_to_append


def append_new_rows(path: Path, df: pd.DataFrame) -> int:
    """Append every new daily row in ``df`` to the ``{SYMBOL}.jsonl`` store at ``path``.

    Reads the existing dates from ``path``, keeps only unseen dates (idempotent
    re-write), and appends them in date order. Returns the number of rows written.
    """
    existing = existing_dates(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_append = build_new_rows(df, existing)
    if rows_to_append:
        with path.open("a", encoding="utf-8") as f:
            for _, line in rows_to_append:
                f.write(line + "\n")
    return len(rows_to_append)
