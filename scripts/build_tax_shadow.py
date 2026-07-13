"""Build data/tax_shadow/{agent}.json for all trading agents.

Reads each agent's ``data/portfolios/{agent}/trades.json``, computes the
French PFU shadow ledger via ``engine.tax_shadow.compute_tax_shadow``, and
writes the result to ``data/tax_shadow/{agent}.json``.

This script is REPORTING ONLY — it never mutates portfolio state.

Usage:
    python scripts/build_tax_shadow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from engine.config import get_config
from engine.tax_shadow import compute_tax_shadow


def build_tax_shadow_all(
    portfolios_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[str]:
    """Compute and write tax shadow ledgers for all trading agents.

    Parameters
    ----------
    portfolios_dir:
        Directory containing per-agent portfolio subdirectories.  Defaults to
        ``data/portfolios/`` relative to the project root.  Pass an explicit
        path to redirect output (e.g. in tests).
    output_dir:
        Directory to write ``{agent}.json`` ledger files.  Defaults to
        ``data/tax_shadow/`` relative to the project root.  Pass an explicit
        path alongside ``portfolios_dir`` to keep all output in a tmp tree.

    Returns
    -------
    list[str]
        Agent IDs for which a ledger was written.
    """
    portfolios_dir = (
        portfolios_dir if portfolios_dir is not None else get_config().portfolios_dir
    )
    output_dir = output_dir if output_dir is not None else get_config().tax_shadow_dir

    if not portfolios_dir.exists():
        print("  No portfolios directory found — skipping.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    traders = set(get_config().trading_roster)

    for agent_dir in sorted(portfolios_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        if agent_dir.name not in traders:
            continue
        agent_id = agent_dir.name

        trades_path = agent_dir / "trades.json"
        if not trades_path.exists():
            print(f"  [SKIP] {agent_id}: no trades.json")
            continue

        try:
            trades = json.loads(trades_path.read_text())
        except (KeyError, ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] {agent_id}: could not read trades.json — {exc}")
            continue

        try:
            result = compute_tax_shadow(trades, agent=agent_id)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] {agent_id}: compute_tax_shadow failed — {exc}")
            continue

        out_path = output_dir / f"{agent_id}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

        sec_pfu = result["securities"]["lifetime_pfu"]
        crypto_pfu = result["crypto"]["lifetime_pfu"]
        print(
            f"  {agent_id}: sec_pfu={sec_pfu:.2f}  crypto_pfu={crypto_pfu:.2f}  → {out_path.name}"
        )
        written.append(agent_id)

    return written


def main() -> None:
    print(f"\n=== Build tax shadow ledgers ===")
    written = build_tax_shadow_all()
    print(f"\n  Done — wrote {len(written)} ledger(s).")


if __name__ == "__main__":
    main()
