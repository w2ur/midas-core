"""One-shot migration: convert vendor sub-unit prices in the OHLCV store to ISO.

The London Stock Exchange quotes in pence, so yfinance serves `LLOY.L` at
116.60 meaning GBP 1.166. The store used to hold that raw pence value and
every *reader* was responsible for dividing by 100. That design failed twice:

1. Three pricing paths forgot the division outright (fixed 2026-08-07 by
   centralising it in `engine.quotes`).
2. Centralising it on the read side still left the **agents** — who read
   `data/market/ohlcv/*.jsonl` directly, with bash, not through the engine —
   seeing pence while their own books recorded pounds. An agent sizing an LSE
   buy off a 116.60 quote asks for a position it believes costs ~EUR 1,090 and
   receives one costing EUR 10.90, under-deploying by 100x, and its stated
   reasoning about position sizing is nonsense. No prose instruction fixes
   that reliably; making the units agree does.

So the conversion moves to **ingest** (`scripts.fetch_ohlcv._normalise_vendor_units`)
and the store becomes ISO-denominated. This script migrates the history that
was written under the old contract.

**What changes:** `open`/`high`/`low`/`close`/`adj_close` are multiplied by the
sub-unit scale (0.01 for GBp). `volume` is a share count and is never scaled.
`date` is untouched, and **line order is preserved exactly** — 529 of the
store's files are deliberately out of date order, and re-sorting them would
emit a 230 MB reorder commit.

**What must NOT change:** any published valuation. Every reader that consumed
these prices already divided by 100 at read time, so scaling the store and
removing that division is a no-op end to end. That invariant is the acceptance
test for this migration, not an aspiration — see the changelog entry.

**Idempotence is enforced with a marker**, not inferred from the data: a store
already in pounds is indistinguishable from one in pence by inspection (a 1.17
close is a plausible penny stock). `data/market/ohlcv/.unit-migrations.json`
records every symbol converted and when; a symbol listed there is refused.
Re-running this script after a partial failure resumes safely.

Dry-run is the default. Pass ``--apply`` to write.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from engine.config import get_config
from engine.quotes import ticker_currency, vendor_quote_unit, vendor_unit_scale

#: Record of symbols already migrated. Lives beside the store it describes so a
#: fork that copies data/market/ohlcv carries its own provenance with it.
MARKER_NAME = ".unit-migrations.json"

#: Price fields to scale. `volume` is a share count; `date` is a label.
PRICE_FIELDS = ("open", "high", "low", "close", "adj_close")


def load_marker(store_dir: Path) -> dict:
    path = store_dir / MARKER_NAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # A corrupt marker must fail loudly: silently treating it as empty
        # would re-migrate every symbol and divide the store by 100 twice.
        raise SystemExit(f"FATAL: {path} is unreadable — refusing to migrate blind.")


def migrate_symbol(path: Path, scale: float) -> tuple[int, int]:
    """Rewrite one store file in place, scaling price fields.

    Returns ``(rows_scaled, rows_skipped)``. A line that will not parse is
    passed through verbatim rather than dropped — the same degrade-don't-destroy
    rule `engine.ohlcv_ingest.merge_rows` follows.
    """
    scaled = skipped = 0
    out: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            out.append(stripped)
            skipped += 1
            continue
        for field in PRICE_FIELDS:
            value = row.get(field)
            if isinstance(value, (int, float)):
                row[field] = value * scale
        out.append(json.dumps(row))
        scaled += 1
    path.write_text("\n".join(out) + "\n")
    return scaled, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--apply", action="store_true", help="Write the converted store."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print without writing (default behavior).",
    )
    args = parser.parse_args()

    store_dir = get_config().ohlcv_dir
    marker = load_marker(store_dir)
    already = set(marker.get("migrated", {}))

    candidates: list[tuple[Path, str, float]] = []
    for path in sorted(store_dir.glob("*.jsonl")):
        symbol = path.stem
        scale = vendor_unit_scale(symbol)
        if scale == 1.0:
            continue
        if symbol in already:
            print(
                f"  = {symbol}: already migrated ({marker['migrated'][symbol]}) — skipped"
            )
            continue
        candidates.append((path, symbol, scale))

    print(f"\nsymbols to migrate: {len(candidates)}")
    total_rows = 0
    for path, symbol, scale in candidates:
        rows = sum(1 for line in path.read_text().splitlines() if line.strip())
        total_rows += rows
        unit = vendor_quote_unit(symbol)
        print(
            f"  {symbol:<14} {unit} -> {ticker_currency(symbol)}  x{scale}  {rows:>6} rows"
        )

    print(f"\ntotal rows: {total_rows}")
    if not args.apply:
        print("mode: DRY RUN")
        return 0

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    migrated = dict(marker.get("migrated", {}))
    scaled_total = skipped_total = 0
    for path, symbol, scale in candidates:
        scaled, skipped = migrate_symbol(path, scale)
        scaled_total += scaled
        skipped_total += skipped
        migrated[symbol] = stamp

    marker_path = store_dir / MARKER_NAME
    marker_path.write_text(
        json.dumps(
            {
                "note": (
                    "Symbols whose stored prices were converted from a vendor "
                    "sub-unit (e.g. GBp pence) to their ISO currency. The store "
                    "is ISO-denominated; scaling happens at ingest, in "
                    "scripts.fetch_ohlcv._normalise_vendor_units. Do not re-run "
                    "the migration for a symbol listed here."
                ),
                "migrated": migrated,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"\nrows scaled: {scaled_total}   unparseable lines passed through: {skipped_total}"
    )
    print(f"marker: {marker_path}")
    print("mode: APPLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
