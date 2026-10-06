"""Deterministic seeds for the random selectors, with no ``bt`` import.

The coin flip (``engine.baselines``) draws its picks without ``bt`` since plan
1.6 (2026-10-05); the ``bt`` algo in ``random_seeded`` re-exports this.
"""

from __future__ import annotations

import hashlib


def make_seed(agent_id: str, from_date_iso: str) -> int:
    """Deterministic 32-bit seed from (agent_id, date)."""
    h = hashlib.sha256(f"{agent_id}|{from_date_iso}".encode()).digest()
    return int.from_bytes(h[:4], "big")
