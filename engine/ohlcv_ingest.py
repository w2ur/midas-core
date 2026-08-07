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
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

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



# ---------------------------------------------------------------------------
# Ingest anomaly tripwire
# ---------------------------------------------------------------------------

#: A *revision* — a value replacing an already-stored close for the same date —
#: moving more than this fraction is quarantined rather than ingested.
#:
#: Calibrated against the measured legitimate revisions, not guessed. Commodity
#: futures drift up to +3.37% between the 22:30 UTC fetch and the final close;
#: FX up to -1.56% on a Friday. 20% clears both by better than 5x while sitting
#: two orders of magnitude below a units flip (100x) or a typical split ratio.
REVISION_LIMIT = 0.20

#: A *new* row whose close is this far from the previous stored close is
#: quarantined. Looser than the revision limit because a genuine gap in the
#: store (a suspended ticker, a long weekend in a thin name) legitimately
#: produces a large step, and because a single day's real move can be violent
#: — the FCIT.L bad tick that motivated this sat at 5,275 against a 320-330
#: range, a factor of 16.
NEW_ROW_LIMIT = 0.40


class QuarantinedRow(NamedTuple):
    """A row refused at ingest, with enough context to adjudicate it by hand."""

    symbol: str
    date: str
    kind: str  # "revision" | "new-row"
    stored_close: float
    incoming_close: float
    ratio: float

    def describe(self) -> str:
        return (
            f"{self.symbol} {self.date} {self.kind}: "
            f"{self.stored_close} -> {self.incoming_close} "
            f"(x{self.ratio:.4g})"
        )


class MergeResult(NamedTuple):
    appended: int
    revised: int
    quarantined: int = 0


def _close_of(line: str) -> float | None:
    try:
        value = json.loads(line).get("close")
    except (json.JSONDecodeError, AttributeError):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _out_of_band(stored: float, incoming: float, limit: float) -> float | None:
    """The ratio when the move exceeds `limit`, else None.

    Returns the ratio rather than a bool so the quarantine record can say how
    far out it was — "x100.3" is diagnosable, "anomaly" is not.
    """
    if stored is None or incoming is None or stored <= 0 or incoming <= 0:
        return None
    ratio = incoming / stored
    if abs(ratio - 1.0) > limit:
        return ratio
    return None


def write_quarantine(path: Path, rows: list[QuarantinedRow]) -> None:
    """Append refused rows to a sidecar so nothing is silently dropped.

    Quarantine lives OUTSIDE the store directory: every reader in the engine
    opens `{ohlcv_dir}/{TICKER}.jsonl` by name, but a stray file under that
    directory is exactly the kind of thing a future glob picks up. A refused
    row must never be one `glob("*.jsonl")` away from becoming a price.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row._asdict()) + "\n")


def merge_rows(
    path: Path,
    df: pd.DataFrame,
    revise_from: str | None = None,
    *,
    quarantine: Path | None = None,
) -> MergeResult:
    """Merge ``df`` into the store at ``path``. Returns ``(appended, revised)``.

    With ``revise_from=None`` this is exactly ``append_new_rows`` — unseen dates
    are appended, stored dates are never touched.

    With ``revise_from`` set to an ISO date, an already-stored row on or after
    that date is *replaced* when the fetched value differs. This exists for bars
    that are still forming when they are first written — 24/7 crypto, commodity
    futures whose next session has already opened, FX after its 17:00 ET roll —
    which the store previously had no way to correct. A bar that was already
    final re-fetches identical, so no stored value is replaced.

    **The store's existing line order is preserved.** A revision overwrites its
    row in place; new dates are appended at the end, in ascending date order
    among themselves. The file is NOT re-sorted — 529 of the 1,046 committed
    files are not in date order (a later long-history backfill was appended
    behind the original window), and since the universal revision window means
    the rewrite path runs nightly, sorting here would make a scheduled job emit
    a ~230 MB, 1.3-million-line reorder commit. Nightly diffs stay one or two
    lines per file. Readers are order-insensitive; canonicalising the store is a
    deliberate, separately-reviewed decision, not a cron side effect.

    Revision rewrites the whole file from a date-keyed map, so a line carrying no
    parseable date has no key to survive under. Such a store is left alone and
    degrades to append-only rather than losing that line.

    ``quarantine`` enables the anomaly tripwire: a revision moving a stored
    close by more than ``REVISION_LIMIT``, or a new row more than
    ``NEW_ROW_LIMIT`` from the previous stored close, is written to that
    sidecar path instead of into the store.

    This is the one place in the system that holds **both** the old value and
    the new one for the same date, which is what makes it the only place a
    vendor unit flip, a re-denomination or a bad tick can be caught *before*
    it becomes that evening's fill price. Every such defect so far has been
    found by a human, weeks to months later, reading a number that looked
    wrong.

    Deliberately opt-in rather than always-on: a resweep legitimately rewrites
    hundreds of rows when a real corporate action lands, and
    ``engine.corporate_actions.detect_split`` is the mechanism that adjudicates
    those. Enable it on the nightly one-day-window path, where a wholesale
    rewrite of history is never legitimate; leave it off on the resweep path,
    where it is the expected outcome.
    """
    existing = existing_dates(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if revise_from is None:
        return MergeResult(append_new_rows(path, df), 0)

    # Read top to bottom: dicts preserve insertion order, so `stored`'s order
    # IS the file's line order, and assigning to an existing key replaces that
    # row without moving it.
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
        return MergeResult(append_new_rows(path, df), 0)

    symbol = path.stem
    # The newest stored close, used as the reference for a brand-new row.
    # Taken by date rather than by file position: 529 of the committed files
    # are not in date order, so "the last line" is not "the latest bar".
    latest_stored_close: float | None = None
    if stored:
        latest_stored_close = _close_of(stored[max(stored)])

    new_rows: dict[str, str] = {}
    revised = 0
    refused: list[QuarantinedRow] = []
    for ts, row in df.iterrows():
        d = ts.date().isoformat() if hasattr(ts, "date") else str(ts)
        record = row_to_record(d, row)
        if record["close"] is None:
            continue
        line = json.dumps(record)
        incoming = record["close"]
        if d not in existing:
            if quarantine is not None and latest_stored_close is not None:
                ratio = _out_of_band(latest_stored_close, incoming, NEW_ROW_LIMIT)
                if ratio is not None:
                    refused.append(
                        QuarantinedRow(
                            symbol, d, "new-row", latest_stored_close, incoming, ratio
                        )
                    )
                    continue
            new_rows[d] = line
        elif d >= revise_from and stored.get(d) != line:
            if quarantine is not None:
                previous = _close_of(stored[d])
                ratio = _out_of_band(previous, incoming, REVISION_LIMIT)
                if ratio is not None:
                    refused.append(
                        QuarantinedRow(symbol, d, "revision", previous, incoming, ratio)
                    )
                    continue
            stored[d] = line  # in place — keeps this row's position in the file
            revised += 1

    # Only the NEW dates are sorted, so a multi-day catch-up lands
    # chronologically among itself; the pre-existing order is untouched. On an
    # empty store every row is new, so the file comes out ascending.
    for d in sorted(new_rows):
        stored[d] = new_rows[d]
    appended = len(new_rows)

    if appended or revised:
        tmp = path.with_name(path.name + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                for d in stored:
                    f.write(stored[d] + "\n")
                # os.replace makes the rename atomic, but the bytes behind it
                # are not durable until they reach disk — without this a crash
                # could leave the renamed file truncated. (The directory entry
                # is still un-fsynced, so the rename itself survives only at the
                # filesystem's discretion.) This is the source of truth for
                # every valuation.
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)
        finally:
            # A crash between open and replace would otherwise leave an orphan
            # `{SYMBOL}.jsonl.tmp` that the fetch-ohlcv workflow's
            # `git add data/market/ohlcv/` would commit into the store.
            tmp.unlink(missing_ok=True)

    if refused:
        write_quarantine(quarantine, refused)
        for row in refused:
            print(f"  [QUARANTINED] {row.describe()}", file=sys.stderr)
    return MergeResult(appended, revised, len(refused))
