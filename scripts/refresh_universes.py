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
from engine.universes.index import INDEX_REFRESHERS, refresh_all_indexes


def main() -> int:
    print("Refreshing index universes from Wikipedia/Slickcharts...")
    indexes = refresh_all_indexes()
    print(json.dumps(indexes, indent=2))
    skipped = [name for name in INDEX_REFRESHERS if name not in indexes]
    if skipped:
        print(
            f"\nSkipped (see warnings above): {', '.join(skipped)} — "
            "committed file left at its last known-good value."
        )

    print("\nRe-seeding alternative universes from curated fallbacks...")
    alts = refresh_all_alternatives()
    print(json.dumps(alts, indent=2))

    print("\nDone. Review `git diff data/universes/` and commit any changes.")
    # A skipped index must still fail the run: the weekly workflow's only
    # alert channel is a non-zero exit (GitHub failure email). The workflow
    # commits the successful indexes regardless — see refresh-universes.yml.
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
