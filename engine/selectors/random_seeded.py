"""Seeded variant of the ``random`` selector.

Unlike bt.algos.SelectRandomly which reads numpy's global RNG, this
selector accepts an explicit seed and produces reproducible picks.
"""

from __future__ import annotations

import hashlib
import random as _pyrandom

import bt


class SelectRandomlySeeded(bt.Algo):
    def __init__(self, n: int, seed: int) -> None:
        super().__init__()
        self.n = n
        self._rng = _pyrandom.Random(seed)

    def __call__(self, target) -> bool:
        universe = target.universe.loc[target.now].dropna()
        candidates = list(universe.index)
        if not candidates:
            target.temp["selected"] = []
            return True
        picks = self._rng.sample(candidates, k=min(self.n, len(candidates)))
        target.temp["selected"] = picks
        return True


def make_seed(agent_id: str, from_date_iso: str) -> int:
    """Deterministic 32-bit seed from (agent_id, start_date)."""
    h = hashlib.sha256(f"{agent_id}|{from_date_iso}".encode()).digest()
    return int.from_bytes(h[:4], "big")
