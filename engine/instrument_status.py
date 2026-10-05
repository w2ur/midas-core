"""Instrument status registry: which symbols the store must not be trusted for.

``data/market/instrument_status.json`` (committed, beside the store) records
the symbols whose stored series is known to be frozen or broken. Absence means
nothing is known against the symbol. Two statuses exist:

- ``suspended``: the ingest tripwire refused a row for the symbol, so the store
  stopped short of what the vendor served and the newest stored close may
  belong to a different instrument (CTVA after its 2026-10-01 separation, MNST
  across its unrestated 2:1 split). Written automatically by
  ``scripts.fetch_ohlcv`` whenever ``merge_rows`` quarantines a row.
- ``delisted``: the symbol serves nothing while its bucket advances. The value
  is part of the schema now; the writer is Stage 3.3 of the 2026-10-03 plan.

A status leaves the registry only by adjudication: the calendar path in
``scripts.fetch_ohlcv._adjudicate`` once its re-merge has landed, or a human
``clear`` with a reason (``python -m engine.instrument_status clear``).

**An unreadable registry fails closed.** ``status_of`` then answers
``suspended`` for every symbol and logs an error: a registry that cannot say
which instruments are broken cannot vouch for any of them, and the cost of
that answer (no trade fills until a human repairs the file) is visible the
same night, where the opposite answer is silent. A *missing* file is an empty
registry, like the corporate-action ledger, because a fork or a fresh data
root has never refused a row, **unless the quarantine beside it holds any row**:
every tripwire refusal writes a quarantine row and a registry entry, and no
writer ever deletes the file, so a quarantine with no registry means the file
was lost, and ``status_of`` fails closed exactly as for an unreadable one. The
desk's own CI test (``tests/test_instrument_status_live.py``) is advisory; this
is what holds a deleted registry at runtime. Writers never overwrite a file
they could not read. **A lost registry is rebuilt, never restarted empty**:
before a writer touches a missing file beside a non-empty quarantine it
re-seeds from the unadjudicated rows (`seed_entries`, the same rule as the
``seed`` command), because reading it as empty and writing back one symbol
would turn the fail-closed state into fail-open for every other suspension
the lost file held (CTVA, at its frozen 09-30 close). A reconstruction that
cannot be computed raises ``RegistryUnreadable`` and writes nothing, so the
fail-closed state holds. On a fresh root the rebuild holds only the symbol
being refused, so the first refusal still creates the file.

The paper broker refuses BUY and SELL on any recorded status
(``INSTRUMENT_SUSPENDED``, `engine.paper_broker._instrument_suspended`). Engine
imports only, no vendor client: this module ships in the midas-core mirror.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

from engine.config import get_config

# How far a ledgered action's effective date may sit from a quarantined row it
# is taken to explain: the constant adjudication itself uses, imported rather
# than restated so the seed and the live path cannot disagree.
from engine.corporate_actions import RECENCY_DAYS

logger = logging.getLogger(__name__)

SUSPENDED = "suspended"
DELISTED = "delisted"
STATUSES = frozenset({SUSPENDED, DELISTED})

SCHEMA_VERSION = 1


class RegistryUnreadable(Exception):
    """The registry file exists but is not a valid registry."""


class Entry(NamedTuple):
    status: str
    since: str  # ISO date of the first refused row / the observation
    source: str  # "tripwire" | "seed" | "human" | ...
    reason: str

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "since": self.since,
            "source": self.source,
            "reason": self.reason,
        }


def registry_path() -> Path:
    """``data/market/instrument_status.json`` under the configured data root."""
    return get_config().ohlcv_dir.parent / "instrument_status.json"


def _parse(text: str) -> dict[str, Entry]:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RegistryUnreadable(f"not JSON ({exc})") from exc
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA_VERSION:
        raise RegistryUnreadable(f"schema is not {SCHEMA_VERSION}")
    instruments = doc.get("instruments")
    if not isinstance(instruments, dict):
        raise RegistryUnreadable("'instruments' is not an object")
    out: dict[str, Entry] = {}
    for symbol, raw in instruments.items():
        if not isinstance(raw, dict):
            raise RegistryUnreadable(f"{symbol}: entry is not an object")
        try:
            entry = Entry(
                status=raw["status"],
                since=raw["since"],
                source=raw["source"],
                reason=raw["reason"],
            )
        except KeyError as exc:
            raise RegistryUnreadable(f"{symbol}: missing {exc}") from exc
        if not isinstance(entry.status, str) or entry.status not in STATUSES:
            raise RegistryUnreadable(f"{symbol}: unknown status {entry.status!r}")
        try:
            date.fromisoformat(entry.since)
        except (TypeError, ValueError) as exc:
            raise RegistryUnreadable(f"{symbol}: bad 'since' {entry.since!r}") from exc
        if not isinstance(entry.reason, str) or not entry.reason.strip():
            raise RegistryUnreadable(f"{symbol}: empty reason")
        out[symbol] = entry
    return out


def render(entries: dict[str, Entry]) -> str:
    """Deterministic serialisation: sorted symbols, two-space indent, newline."""
    doc = {
        "schema": SCHEMA_VERSION,
        "instruments": {s: entries[s].to_json() for s in sorted(entries)},
    }
    return json.dumps(doc, indent=2, sort_keys=False) + "\n"


def load(path: Path | None = None) -> dict[str, Entry]:
    """Every entry in the registry. Raises ``RegistryUnreadable``; a missing
    file is an empty registry."""
    path = path if path is not None else registry_path()
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # UnicodeDecodeError is a ValueError, not an OSError: a file that is
        # not UTF-8 is as unreadable as one that cannot be opened.
        raise RegistryUnreadable(f"cannot read {path} ({exc})") from exc
    return _parse(text)


def _lost(path: Path) -> bool:
    """The registry is absent although the quarantine beside it is not."""
    if path.exists():
        return False
    quarantine = path.parent / "quarantine"
    return quarantine.is_dir() and any(quarantine.glob("*.jsonl"))


def status_of(symbol: str, path: Path | None = None) -> str | None:
    """The symbol's status, or ``None`` when nothing is recorded against it.

    Fails closed: an unreadable registry, or a missing one beside a non-empty
    quarantine (`_lost`), answers ``suspended`` for every symbol, and says why
    in the log.
    """
    path = path if path is not None else registry_path()
    if _lost(path):
        logger.error(
            "instrument status registry %s is missing although %s holds refused "
            "rows; treating %s as suspended (rebuild it with "
            "`python -m engine.instrument_status seed`)",
            path,
            path.parent / "quarantine",
            symbol,
        )
        return SUSPENDED
    try:
        entries = load(path)
    except RegistryUnreadable as exc:
        logger.error(
            "instrument status registry unreadable (%s); treating %s as suspended",
            exc,
            symbol,
        )
        return SUSPENDED
    entry = entries.get(symbol)
    return entry.status if entry is not None else None


def _save(entries: dict[str, Entry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(entries), encoding="utf-8")


def _load_for_write(path: Path) -> dict[str, Entry]:
    """What a writer starts from: the registry, or its reconstruction if lost.

    Regression (review of feat/stage1-asof-reads): writers read a lost file as
    empty, so the next tripwire refusal wrote a registry holding only the new
    symbol, and every suspension the lost file held (CTVA) silently cleared.
    """
    if not _lost(path):
        return load(path)
    market = path.parent
    try:
        entries = seed_entries(
            market / "quarantine", market / "corporate_actions.jsonl", market / "ohlcv"
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # json.JSONDecodeError and UnicodeDecodeError are ValueErrors.
        raise RegistryUnreadable(
            f"{path} is missing beside a non-empty quarantine and cannot be "
            f"rebuilt from it ({exc!r})"
        ) from exc
    logger.error(
        "instrument status registry %s was missing beside a non-empty quarantine; "
        "rebuilt %d entr(y/ies) from the unadjudicated rows before writing",
        path,
        len(entries),
    )
    return entries


def mark(
    symbol: str,
    status: str,
    *,
    since: str,
    source: str,
    reason: str,
    path: Path | None = None,
) -> Entry:
    """Record ``status`` against ``symbol``; returns the entry now held.

    Idempotent: a symbol already carrying the same status keeps its earliest
    ``since`` and its original reason, so a symbol refused night after night
    still says when it froze. Raises ``RegistryUnreadable`` rather than
    overwriting a file it could not read, and rebuilds a lost one rather than
    restarting it empty (`_load_for_write`).
    """
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if not reason.strip():
        raise ValueError("a status needs a reason")
    date.fromisoformat(since)
    path = path if path is not None else registry_path()
    rebuilt = _lost(path)
    entries = _load_for_write(path)
    if rebuilt:
        # The caller is the authority on its own symbol: the rebuild's entry
        # for it is a reconstruction, so only its earlier date survives.
        seeded = entries.pop(symbol, None)
        if seeded is not None and seeded.status == status and seeded.since < since:
            since = seeded.since
    held = entries.get(symbol)
    if held is not None and held.status == status:
        changed = since < held.since
        if changed:
            held = held._replace(since=since)
            entries[symbol] = held
        if changed or rebuilt:
            _save(entries, path)
        return held
    entry = Entry(status=status, since=since, source=source, reason=reason)
    entries[symbol] = entry
    _save(entries, path)
    return entry


def mark_suspended(
    symbol: str, *, since: str, source: str, reason: str, path: Path | None = None
) -> Entry:
    return mark(symbol, SUSPENDED, since=since, source=source, reason=reason, path=path)


def clear(symbol: str, *, reason: str, path: Path | None = None) -> Entry | None:
    """Remove ``symbol``'s entry; returns what was removed (``None`` if none).

    The reason is required even though the file keeps no history: the
    caller logs it, and the commit carrying the change is the audit trail.
    """
    if not reason.strip():
        raise ValueError("clearing a status needs a reason")
    path = path if path is not None else registry_path()
    rebuilt = _lost(path)
    entries = _load_for_write(path)
    removed = entries.pop(symbol, None)
    if removed is not None or rebuilt:
        _save(entries, path)
    if removed is not None:
        logger.info("instrument status: cleared %s (%s)", symbol, reason)
    return removed


# ---------------------------------------------------------------------------
# Seeding from the quarantine and the corporate-action ledger
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _store_closes(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        close = row.get("close")
        if isinstance(row.get("date"), str) and isinstance(close, (int, float)):
            out[row["date"]] = float(close)
    return out


def _explained_by_ledger(row: dict, effectives: Iterable[str]) -> bool:
    """A ledgered action for the symbol, effective near the refused row."""
    try:
        refused_on = date.fromisoformat(row["date"])
    except (KeyError, TypeError, ValueError):
        return False
    window = timedelta(days=RECENCY_DAYS)
    for effective in effectives:
        try:
            on = date.fromisoformat(effective)
        except (TypeError, ValueError):
            continue
        if abs(on - refused_on) <= window:
            return True
    return False


def _landed_in_store(row: dict, closes: dict[str, float]) -> bool:
    """The store now holds the refused row's date on the REFUSED side.

    The tripwire held the store at ``stored_close``; a human re-merge with the
    tripwire off (the documented path, MRNA 2026-08-21, ``8b6907458``) lands
    the vendor's value. So the store has taken the refusal's side when its
    close for that date is nearer the incoming value than the one it was
    refused against. Parameter-free on purpose: no tolerance to calibrate.
    """
    held = closes.get(row.get("date"))
    incoming = row.get("incoming_close")
    refused_against = row.get("stored_close")
    if held is None or not isinstance(incoming, (int, float)):
        return False
    if not isinstance(refused_against, (int, float)):
        return False
    return abs(held - incoming) < abs(held - refused_against)


def unadjudicated_rows(
    quarantine_dir: Path, ledger: Path, ohlcv_dir: Path
) -> dict[str, list[dict]]:
    """Quarantined rows nobody has adjudicated, per symbol.

    A refused row counts as adjudicated when EITHER
    - the corporate-action ledger holds an action for its symbol effective
      within ``RECENCY_DAYS`` of the row (the calendar path, and a human who
      wrote the ledger row: MNST, JMAT.L, BYND, AVB, APH); or
    - the store now holds the row's date on the refused side (a human re-merge
      of a real move that was no corporate action, so wrote no ledger row:
      MRNA 2026-08-21).

    The plan's review (SHOULD 4) gave only the first arm. Measured on
    2026-10-03 it would have suspended MRNA, a live S&P 500 name whose
    refused rows a human accepted on 2026-08-21 and which has traded normally
    in the store since. The second arm is what tells those apart.

    Raises on an unreadable ledger or quarantine file: a seed built on a
    ledger it could not read would clear what it cannot vouch for.
    """
    effectives: dict[str, list[str]] = {}
    if ledger.exists():
        for record in _read_jsonl(ledger):
            effectives.setdefault(record["symbol"], []).append(record["effective"])
    out: dict[str, list[dict]] = {}
    if not quarantine_dir.exists():
        return out
    for qfile in sorted(quarantine_dir.glob("*.jsonl")):
        symbol = qfile.name[: -len(".jsonl")]
        closes = _store_closes(ohlcv_dir / qfile.name)
        for row in _read_jsonl(qfile):
            if _explained_by_ledger(row, effectives.get(symbol, ())):
                continue
            if _landed_in_store(row, closes):
                continue
            out.setdefault(symbol, []).append(row)
    return out


def seed_entries(
    quarantine_dir: Path, ledger: Path, ohlcv_dir: Path
) -> dict[str, Entry]:
    """The registry the current quarantine and ledger imply."""
    entries: dict[str, Entry] = {}
    for symbol, rows in unadjudicated_rows(quarantine_dir, ledger, ohlcv_dir).items():
        dates = sorted({r["date"] for r in rows})
        entries[symbol] = Entry(
            status=SUSPENDED,
            since=dates[0],
            source="seed",
            reason=(
                f"{len(dates)} quarantined date(s) ({', '.join(dates)}) with no "
                "ledgered corporate action and no re-merge into the store"
            ),
        )
    return entries


def _default_dirs() -> tuple[Path, Path, Path]:
    market = get_config().ohlcv_dir.parent
    return market / "quarantine", market / "corporate_actions.jsonl", market / "ohlcv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m engine.instrument_status",
        description="Inspect, seed or clear the instrument status registry.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="print the registry")
    seed = sub.add_parser(
        "seed",
        help="rebuild the registry from unadjudicated quarantine rows "
        "(refuses to overwrite a non-empty registry without --force)",
    )
    seed.add_argument("--force", action="store_true")
    clr = sub.add_parser("clear", help="clear one symbol's status (human adjudication)")
    clr.add_argument("symbol")
    clr.add_argument("--reason", required=True)
    args = parser.parse_args(argv)

    path = registry_path()
    if args.command == "show":
        print(render(load(path)), end="")
        return 0
    if args.command == "seed":
        existing = load(path)
        if existing and not args.force:
            print(
                f"{path} already holds {len(existing)} entr(y/ies); "
                "seeding would drop tripwire and human history. Use --force.",
                file=sys.stderr,
            )
            return 2
        entries = seed_entries(*_default_dirs())
        _save(entries, path)
        print(render(entries), end="")
        return 0
    if not args.reason.strip():
        parser.error("--reason must not be empty")
    removed = clear(args.symbol, reason=args.reason, path=path)
    if removed is None:
        print(f"{args.symbol}: no status recorded; nothing to clear", file=sys.stderr)
        return 1
    print(
        f"{args.symbol}: cleared {removed.status} (since {removed.since}): {args.reason}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
