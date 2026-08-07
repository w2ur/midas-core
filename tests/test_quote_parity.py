"""Cross-language parity: engine/quotes.py vs site/src/lib/ohlcv.ts (W6.1).

Quote-currency resolution is implemented twice — once in Python for the
engine, once in TypeScript for the static site — because the site builds from
committed artifacts with no Python in the loop. Two implementations of one
rule drift, and this pair drifted in the worst possible way: each side pinned
the *opposite* answer as a passing test, so both suites were green while
disagreeing about what currency a ticker quotes in.

Found on 2026-08-07:
  - `.V` (TSX Venture) → CAD in TypeScript, absent from the Python table.
  - `…=X` → refused in TypeScript, answered `USD` in Python for *every* FX
    pair, since a pair carries no dot and fell through to the bare-ticker
    branch. Correct for `EURUSD=X` by accident; wrong for `EURGBP=X`,
    `GBPJPY=X`, `USDJPY=X`, `EURJPY=X`.

Both were resolved onto the TypeScript answer, which was the correct one in
each case. This test parses the TypeScript source rather than mirroring it in
a fixture: a fixture is a third copy, and a third copy drifts too.

Follows the `tests/test_reason_codes.py` pattern (a cross-artifact parity
check that already runs in CI). Reads source text only — no node, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from engine.config import get_config
from engine.quotes import _SUFFIX_UNITS, _heuristic_unit

_TS = Path(get_config().data_dir) / "site" / "src" / "lib" / "ohlcv.ts"

pytestmark = pytest.mark.skipif(
    not _TS.exists(), reason="site/ is not present at this data root (core mirror)"
)


def _ts_suffix_table() -> dict[str, str]:
    """Parse the `switch (suffix)` table out of ohlcv.ts `quoteUnit`.

    Deliberately strict: if the TypeScript is restructured so this parse stops
    finding entries, `test_the_parser_actually_parses` fails rather than the
    parity check silently comparing an empty table against a full one.
    """
    src = _TS.read_text(encoding="utf-8")
    body = src.split("const suffix = ticker.slice(dot + 1).toUpperCase();", 1)[1]
    body = body.split("}\n", 1)[0]

    table: dict[str, str] = {}
    pending: list[str] = []
    for line in body.splitlines():
        pending.extend(re.findall(r'case "([A-Z]+)":', line))
        returned = re.search(r'return "([A-Za-z]{3})"', line)
        if returned and pending:
            for suffix in pending:
                table[suffix] = returned.group(1)
            pending = []
    return table


def test_the_parser_actually_parses() -> None:
    """The control. A parity test whose parser silently returns {} passes
    forever — that is the failure mode this whole file exists to prevent."""
    table = _ts_suffix_table()
    assert len(table) >= 20
    assert table["PA"] == "EUR"
    assert table["L"] == "GBp"


def test_suffix_tables_agree() -> None:
    ts = _ts_suffix_table()
    py = dict(_SUFFIX_UNITS)
    assert py == ts, (
        "engine/quotes.py `_SUFFIX_UNITS` and site/src/lib/ohlcv.ts `quoteUnit` "
        "disagree. Both are read by production code; a divergence means the "
        "engine and the site price the same ticker in different currencies.\n"
        f"  only in Python: { ({k: v for k, v in py.items() if k not in ts}) }\n"
        f"  only in TS:     { ({k: v for k, v in ts.items() if k not in py}) }\n"
        f"  differing:      "
        f"{ ({k: (py[k], ts[k]) for k in py.keys() & ts.keys() if py[k] != ts[k]}) }"
    )


@pytest.mark.parametrize(
    "ticker",
    ["EURUSD=X", "EURGBP=X", "GBPJPY=X", "USDJPY=X", "EURJPY=X", "AUDUSD=X"],
)
def test_fx_pairs_refuse_on_the_heuristic(ticker: str) -> None:
    """A pair quotes in its second leg, which no suffix rule can see. Python
    answered USD for all of these; TypeScript refused. Refusing wins — a
    mislabelled price still prices, and the registry answers these anyway."""
    assert _heuristic_unit(ticker) is None


def test_ts_also_refuses_fx_pairs() -> None:
    """Pin the TypeScript side of the same rule, so a future edit that makes
    it guess is caught here rather than by a mispriced book."""
    src = _TS.read_text(encoding="utf-8")
    assert re.search(r"if \(/=X\$/\.test\(ticker\)\) return null;", src), (
        "ohlcv.ts no longer refuses FX pairs in quoteUnit — it must, to stay "
        "in step with engine.quotes._heuristic_unit"
    )


def test_bare_ticker_is_usd_on_both_sides() -> None:
    """Not a guess about an unknown exchange — Yahoo's US-listing convention,
    and the one place both sides deliberately do answer."""
    assert _heuristic_unit("AAPL") == "USD"
    src = _TS.read_text(encoding="utf-8")
    assert 'if (dot === -1) return "USD";' in src
