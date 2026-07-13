"""Refresh every committed universe file from its upstream source.

Index universes (S&P 500, CAC 40, etc.) re-fetch from Wikipedia; Nasdaq-100
re-fetches from Slickcharts (Wikipedia dropped its constituents table on
2026-07-13). Alternative universes (congressional, insider, high-short)
re-seed from the curated fallback constants in `engine/universes/alternative.py`.

Run manually after an upstream layout change, or via the weekly GitHub
Actions workflow `.github/workflows/refresh-universes.yml`. The cloud
trading-session sandbox has no outbound HTTP and MUST NOT call this — it
reads from the committed `data/universes/*.json` files instead.

Usage:
    python scripts/refresh_universes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from engine.universes.alternative import refresh_all_alternatives
from engine.universes.index import refresh_all_indexes


def main() -> int:
    print("Refreshing index universes from Wikipedia...")
    indexes = refresh_all_indexes()
    print(json.dumps(indexes, indent=2))

    print("\nRe-seeding alternative universes from curated fallbacks...")
    alts = refresh_all_alternatives()
    print(json.dumps(alts, indent=2))

    print("\nDone. Review `git diff data/universes/` and commit any changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
