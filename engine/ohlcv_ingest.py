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
import os
from datetime import date, timedelta
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
    partially-written file never aborts a refresh. A line that is valid JSON but
    not an object (e.g. ``42``) counts as unparseable — ``.get`` on it raises
    ``AttributeError``, and an uncaught raise here would abort the whole
    ~1,150-symbol nightly run over one corrupt line.
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
                d = json.loads(line).get("date")
            except (json.JSONDecodeError, AttributeError):
                continue
            if d:
                dates.add(d)
    return dates


def fetch_window_start(
    last: date | None,
    end: date,
    history_days: int,
    *,
    revise_days: int = 0,
) -> date | None:
    """Return the inclusive start date for the next fetch, or None to skip.

    ``end`` is the last day we want (normally today). The OHLCV cron runs at
    22:30 UTC — after the US close — so a store holding ``end - 1`` must still
    fetch ``end``: the bar exists. Only a store that already holds ``end`` is
    skipped.

    ``revise_days`` re-requests that many already-stored trailing days so a bar
    that was still forming when it was first written can be corrected by its
    final value. It has no effect on an empty store, which fetches the full
    ``history_days`` window regardless.
    """
    if last is None:
        return end - timedelta(days=history_days)
    if last >= end:
        return None
    return last + timedelta(days=1 - revise_days)


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


def merge_rows(
    path: Path, df: pd.DataFrame, revise_from: str | None = None
) -> tuple[int, int]:
    """Merge ``df`` into the store at ``path``. Returns ``(appended, revised)``.

    With ``revise_from=None`` this is exactly ``append_new_rows`` — unseen dates
    are appended, stored dates are never touched.

    With ``revise_from`` set to an ISO date, an already-stored row on or after
    that date is *replaced* when the fetched value differs. This exists for bars
    that are still forming when they are first written — 24/7 crypto, commodity
    futures whose next session has already opened, FX after its 17:00 ET roll —
    which the store previously had no way to correct. A bar that was already
    final re-fetches identical, so nothing is replaced and the file is not
    rewritten at all.

    Revision rewrites the whole file in date order, so a store containing a line
    with no parseable date is left alone and degrades to append-only — reordering
    it would silently drop that line.
    """
    existing = existing_dates(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if revise_from is None:
        return append_new_rows(path, df), 0

    stored: dict[str, str] = {}
    unparseable = 0
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line).get("date")
                except (json.JSONDecodeError, AttributeError):
                    unparseable += 1
                    continue
                if d:
                    stored[d] = line
                else:
                    unparseable += 1
    if unparseable:
        return append_new_rows(path, df), 0

    appended = revised = 0
    for ts, row in df.iterrows():
        d = ts.date().isoformat() if hasattr(ts, "date") else str(ts)
        record = row_to_record(d, row)
        if record["close"] is None:
            continue
        line = json.dumps(record)
        if d not in existing:
            stored[d] = line
            appended += 1
        elif d >= revise_from and stored.get(d) != line:
            stored[d] = line
            revised += 1

    if appended or revised:
        tmp = path.with_name(path.name + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                for d in sorted(stored):
                    f.write(stored[d] + "\n")
                # os.replace is atomic w.r.t. the rename but says nothing about
                # durability: a crash after the rename can leave the store file
                # truncated. This is the source of truth for every valuation.
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)
        finally:
            # A crash between open and replace would otherwise leave an orphan
            # `{SYMBOL}.jsonl.tmp` that the fetch-ohlcv workflow's
            # `git add data/market/ohlcv/` would commit into the store.
            tmp.unlink(missing_ok=True)
    return appended, revised
