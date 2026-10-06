"""Seeded variant of the ``random`` selector.

Unlike bt.algos.SelectRandomly which reads numpy's global RNG, this
selector accepts an explicit seed and produces reproducible picks. Its user is
the ``random`` selector of ``engine.adapter`` (factor research); the coin flip
stopped using it with plan 1.6 and draws with ``make_seed`` (now in the
bt-free ``engine.selectors.seeding``, re-exported here) without bt.
"""

from __future__ import annotations

import random as _pyrandom

import bt

from engine.selectors.seeding import make_seed

__all__ = ["SelectRandomlySeeded", "make_seed"]


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

