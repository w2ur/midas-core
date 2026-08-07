"""Every store read path takes raw ``close``, never ``adj_close``.

2026-08-07 review §5.2 / §7.3. The five readers named in the review, plus the
three the sweep turned up, had each independently chosen "``adj_close`` when
present, else ``close``". Two reasons that preference is wrong here:

- The paper broker never credits dividend cash, so a book holding a payer has
  no dividend in its cash. Pricing its position on a dividend-*reinvested*
  series values a return the book did not receive. Price return on ``close``
  is the internally consistent basis.
- Yahoo re-bases ``adj_close`` across a symbol's whole history after every
  payout, so the same date prices differently on two different days. That is
  structurally incompatible with ``add_snapshot`` / ``merge_baseline_series``'
  append-or-refuse contract.

Every fixture below writes an ``adj_close`` that **differs** from ``close``.
That is the point: with equal fields these assertions could not fail, and a
check that cannot fail is not evidence.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

# Raw close vs. a dividend-reinvested adj_close ~1.6% below it — the shape
# Yahoo produces, and the order of magnitude METHODOLOGY measured in-window
# (worst SAN.PA at 5.30% below).
RAW_CLOSE = 100.0
ADJ_CLOSE = 98.4


def _write_store(ohlcv_dir: Path, ticker: str, rows: list[dict]) -> None:
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    (ohlcv_dir / f"{ticker}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def _diverging_rows() -> list[dict]:
    return [
        {"date": "2026-06-10", "close": RAW_CLOSE, "adj_close": ADJ_CLOSE},
        {"date": "2026-06-11", "close": RAW_CLOSE * 1.1, "adj_close": ADJ_CLOSE * 1.1},
    ]


@pytest.fixture
def diverging_store(midas_data_root: Path) -> Path:
    """A store whose ``adj_close`` is visibly not its ``close``."""
    from engine.config import get_config

    ohlcv = get_config().ohlcv_dir
    _write_store(ohlcv, "PAYER", _diverging_rows())
    return ohlcv


class TestReadPathsTakeRawClose:
    def test_ohlcv_store_latest_close_on_or_before(self, diverging_store: Path) -> None:
        from engine.ohlcv_store import latest_close_on_or_before

        assert latest_close_on_or_before("PAYER", date(2026, 6, 10)) == RAW_CLOSE

    def test_quotes_latest_price(self, diverging_store: Path) -> None:
        """The broker's fill price and ``portfolio_mtm``'s point valuation."""
        from engine.quotes import latest_price

        quote = latest_price("PAYER", on=date(2026, 6, 10))
        assert quote is not None
        assert quote.price == RAW_CLOSE

    def test_fx_load_store_series(self, diverging_store: Path) -> None:
        from engine import fx

        assert fx._load_store_series("PAYER")["2026-06-10"] == RAW_CLOSE

    def test_baselines_load_ohlcv(self, diverging_store: Path) -> None:
        """The passive benchmark and coin-flip controls agents are graded against."""
        from engine import baselines

        assert baselines._load_ohlcv("PAYER")["2026-06-10"] == RAW_CLOSE

    def test_market_data_store_series(self, diverging_store: Path) -> None:
        from engine.market_data import _store_series

        series = _store_series("PAYER", date(2026, 6, 10), date(2026, 6, 10))
        assert series is not None
        assert float(series.iloc[0]) == RAW_CLOSE

    def test_market_data_latest_close_from_store(self, diverging_store: Path) -> None:
        from engine.market_data import _latest_close_from_store

        assert _latest_close_from_store("PAYER") == pytest.approx(RAW_CLOSE * 1.1)

    def test_market_data_latest_close_and_date(self, diverging_store: Path) -> None:
        from engine.market_data import latest_close_and_date_from_store

        result = latest_close_and_date_from_store("PAYER")
        assert result is not None
        assert result[0] == pytest.approx(RAW_CLOSE * 1.1)

    def test_market_data_get_latest_price(self, diverging_store: Path) -> None:
        from engine.market_data import get_latest_price

        assert get_latest_price("PAYER") == pytest.approx(RAW_CLOSE * 1.1)

    def test_resolve_manager_outcomes_forward_rows(self, diverging_store: Path) -> None:
        from scripts.resolve_manager_outcomes import _forward_rows

        rows = _forward_rows("PAYER", "2026-06-10", diverging_store)
        assert rows == [("2026-06-11", pytest.approx(RAW_CLOSE * 1.1))]


class TestNoReaderReferencesAdjClose:
    """Source-level backstop: only the ingest/migration writers touch the field.

    The per-reader tests above cover the readers that exist today. This one
    catches a *new* reader written tomorrow with the old idiom, which is how
    the divergence spread to eight call sites in the first place.
    """

    # Reads the field's value, as opposed to writing it or naming it in prose.
    READ_PATTERN = re.compile(r"""(?:\.get\(|\[)\s*["']adj_close["']""")

    # Files that legitimately handle the field: the ingest writer, the unit
    # migration that must scale it alongside the other price columns, and the
    # split detector that explicitly reasons about it.
    ALLOWED = {
        "engine/ohlcv_ingest.py",
        "engine/corporate_actions.py",
        "scripts/normalise_store_units.py",
    }

    def test_only_writers_read_adj_close(self) -> None:
        root = Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for directory in ("engine", "scripts", "app", "backtester"):
            for path in (root / directory).rglob("*.py"):
                rel = path.relative_to(root).as_posix()
                if rel in self.ALLOWED:
                    continue
                for lineno, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if self.READ_PATTERN.search(line):
                        offenders.append(f"{rel}:{lineno}: {line.strip()}")
        assert offenders == [], (
            "read paths must take raw `close` (2026-08-07 review §5.2); "
            "found adj_close reads:\n" + "\n".join(offenders)
        )

    def test_pattern_matches_the_idiom_it_is_looking_for(self) -> None:
        """The control: this regex must actually match the shape it bans."""
        assert self.READ_PATTERN.search('val = row.get("adj_close")')
        assert self.READ_PATTERN.search("val = row['adj_close']")
        assert not self.READ_PATTERN.search('val = row.get("close")')
        assert not self.READ_PATTERN.search('"adj_close": safe_float(x)')
