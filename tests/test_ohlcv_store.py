"""Tests for engine.ohlcv_store.latest_close_on_or_before."""

from datetime import date
from pathlib import Path

from engine.ohlcv_store import DatedClose, latest_close_on_or_before


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_returns_none_when_ticker_absent(tmp_path: Path) -> None:
    assert latest_close_on_or_before("GHOST", date(2026, 4, 17), store=tmp_path) is None


def test_returns_exact_date_when_present(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "MSFT.jsonl",
        [
            {"date": "2026-04-15", "close": 300.0},
            {"date": "2026-04-16", "close": 310.0},
            {"date": "2026-04-17", "close": 320.0},
        ],
    )
    assert latest_close_on_or_before(
        "MSFT", date(2026, 4, 17), store=tmp_path
    ) == DatedClose(320.0, date(2026, 4, 17))


def test_returns_prior_date_when_target_missing(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "MSFT.jsonl",
        [
            {"date": "2026-04-15", "close": 300.0},
            {"date": "2026-04-16", "close": 310.0},
        ],
    )
    # Store has no 04-17 row; cron-before-OHLCV scenario. The close is the
    # 04-16 one AND says so: the row's date, not the date asked about, is
    # what lets a caller see that this price is a day old (2026-10-03, CTVA).
    assert latest_close_on_or_before(
        "MSFT", date(2026, 4, 17), store=tmp_path
    ) == DatedClose(310.0, date(2026, 4, 16))


def test_returns_none_when_all_dates_later(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "MSFT.jsonl",
        [
            {"date": "2026-04-17", "close": 320.0},
        ],
    )
    assert latest_close_on_or_before("MSFT", date(2026, 4, 14), store=tmp_path) is None


def test_reads_raw_close_and_ignores_adj_close(tmp_path: Path) -> None:
    """Raw `close`, never `adj_close` (2026-08-07 review §5.2).

    This test asserted the opposite until 2026-08-07 — `adj_close` was
    preferred everywhere. The basis moved to price return because the paper
    broker credits no dividend cash and because Yahoo re-bases `adj_close`
    retroactively, which append-or-refuse cannot tolerate. The two fields
    differ here precisely so the assertion can fail if a reader drifts back.
    """
    _write_jsonl(
        tmp_path / "MSFT.jsonl",
        [
            {"date": "2026-04-17", "close": 320.0, "adj_close": 318.5},
        ],
    )
    dated = latest_close_on_or_before("MSFT", date(2026, 4, 17), store=tmp_path)
    assert dated is not None
    assert dated.close == 320.0


def test_returns_none_when_close_missing_rather_than_using_adj_close(
    tmp_path: Path,
) -> None:
    """No silent basis switch when a row somehow lacks `close`.

    `ohlcv_ingest.build_new_rows` drops rows without a close, so this shape
    does not occur in the committed store (verified: 0 of 1,703,770 rows).
    Falling back to `adj_close` would reintroduce the mixed basis on exactly
    the rows nobody is watching.
    """
    _write_jsonl(
        tmp_path / "MSFT.jsonl",
        [
            {"date": "2026-04-17", "adj_close": 318.5},
        ],
    )
    assert latest_close_on_or_before("MSFT", date(2026, 4, 17), store=tmp_path) is None


def test_as_of_is_the_newest_row_not_the_last_line(tmp_path: Path) -> None:
    """Lines are not in date order in the committed store (a refill appends
    an interior date at the end of the file, 2026-09-26), so `as_of` must be
    the newest eligible date, not whatever line was read last."""
    _write_jsonl(
        tmp_path / "SESG.PA.jsonl",
        [
            {"date": "2026-09-24", "close": 10.0},
            {"date": "2026-09-25", "close": 11.0},
            {"date": "2026-09-17", "close": 9.0},
        ],
    )
    assert latest_close_on_or_before(
        "SESG.PA", date(2026, 9, 30), store=tmp_path
    ) == DatedClose(11.0, date(2026, 9, 25))
