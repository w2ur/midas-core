"""OHLCV store access helpers.

Single source of truth for reading the committed OHLCV JSONL store at
data/market/ohlcv/{TICKER}.jsonl. Used by valuation (for MTM) and the paper
broker (for fill prices).

**Read paths take raw ``close``, never ``adj_close``** (2026-08-07, review
§5.2). Both fields are stored — `adj_close` stays at rest as vendor data —
but every price a valuation, fill, benchmark or FX rate is derived from is
the raw close. Two reasons, and they are not stylistic:

- The paper broker never credits dividend cash. A book that holds a payer
  therefore has no dividend in its cash, so pricing its position on a
  dividend-*reinvested* series values a return the book did not receive.
  Price return on ``close`` is the internally consistent basis.
- Yahoo re-bases ``adj_close`` across a symbol's entire history after every
  payout. A value that the vendor rewrites retroactively cannot sit under
  ``add_snapshot``/``merge_baseline_series``' keep-what-is-published contract: the
  same date would price differently on two different days for no reason
  anyone recorded.

Splits are *not* the exception that argues for ``adj_close`` here: Yahoo
restates raw ``close`` itself for a split, and `engine.corporate_actions`
adjudicates that on the store side.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import NamedTuple

from engine.config import get_config


class DatedClose(NamedTuple):
    """A stored raw close together with the date of the row it came from.

    `as_of` is the row's own market date, never the date the caller asked
    about. The two differ whenever the store has no row for the requested
    day (a weekend, a holiday, or a vendor that has not served the bar yet),
    and until 2026-10-03 every read path dropped that difference on the
    floor: a CTVA order on 2026-10-02 would have filled at its 09-30 close
    with no way to see that the price was two sessions old (a what-if run
    against the store at 0f981dd99; no CTVA order was ever placed). Carrying the date is
    what lets a caller tell "today's close" from "the last close we have".
    """

    close: float
    as_of: date


def latest_close_on_or_before(
    ticker: str, on: date | None = None, store: Path | None = None
) -> DatedClose | None:
    """Return the most recent raw `close` for `ticker` with date <= `on`,
    with the date of the row it was read from.

    Deliberately ignores `adj_close` — see the module docstring. Every stored
    row carries a `close` (`ohlcv_ingest.build_new_rows` drops rows without
    one), so there is no fallback: a row with no close yields None rather
    than silently switching basis.

    Returns None if the ticker is not in the store, no row satisfies the date
    bound, or the newest such row carries no close.
    `store` defaults to ``get_config().ohlcv_dir`` (MIDAS_DATA_DIR-aware, resolved at
    call time); tests may pass a tmp path.
    """
    store = store if store is not None else get_config().ohlcv_dir
    path = store / f"{ticker}.jsonl"
    if not path.exists():
        return None
    target = on.isoformat() if on is not None else "9999-99-99"
    best_date: str | None = None
    best_price: float | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_date = row.get("date")
            if row_date is None or row_date > target:
                continue
            if best_date is None or row_date > best_date:
                best_date = row_date
                val = row.get("close")
                best_price = float(val) if val is not None else None
    if best_date is None or best_price is None:
        return None
    return DatedClose(best_price, date.fromisoformat(best_date))


def __getattr__(name: str) -> object:
    """Lazily expose ``OHLCV_STORE`` as the current config's OHLCV dir (PEP 562).

    Kept as a module attribute so existing readers
    (``from engine.ohlcv_store import OHLCV_STORE``) resolve through
    ``get_config()`` at access time — MIDAS_DATA_DIR-aware, never frozen at import.
    """
    if name == "OHLCV_STORE":
        return get_config().ohlcv_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
