"""Tests for engine.ohlcv_ingest — the row-normalization / merge / idempotency
logic behind scripts/fetch_ohlcv.py.

All frames here are SYNTHETIC yfinance-shaped DataFrames — the point of the
extraction is to exercise this logic without touching the network. The store
byte layout is committed to git and read by the sandboxed agent, so the append
tests assert the exact on-disk JSON, not just row counts.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine import ohlcv_ingest
from engine.ohlcv_ingest import (
    append_new_rows,
    build_new_rows,
    existing_dates,
    fetch_window_start,
    flatten_columns,
    merge_rows,
    row_to_record,
    safe_float,
    safe_int,
)

_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _yf_frame(rows: dict[str, list], *, multiindex: bool = False) -> pd.DataFrame:
    """Build a synthetic yfinance-shaped OHLCV frame.

    `rows` maps ISO date strings to [open, high, low, close, adj_close, volume].
    Index is a DatetimeIndex (yfinance's shape); columns are the six OHLCV fields,
    optionally wrapped in a single-symbol MultiIndex (yfinance's single-download shape).
    """
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in rows], name="Date")
    data = [rows[d] for d in rows]
    columns: pd.Index | pd.MultiIndex = pd.Index(_FIELDS)
    if multiindex:
        columns = pd.MultiIndex.from_product([_FIELDS, ["AAPL"]])
    return pd.DataFrame(data, index=idx, columns=columns)


def _write_store(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


# --- column normalization -------------------------------------------------


def test_flatten_columns_collapses_multiindex() -> None:
    df = _yf_frame({"2026-04-15": [1, 2, 0.5, 1.5, 1.5, 100]}, multiindex=True)
    assert isinstance(df.columns, pd.MultiIndex)
    out = flatten_columns(df)
    assert list(out.columns) == _FIELDS


def test_flatten_columns_passthrough_when_flat() -> None:
    df = _yf_frame({"2026-04-15": [1, 2, 0.5, 1.5, 1.5, 100]})
    out = flatten_columns(df)
    assert list(out.columns) == _FIELDS


def test_row_to_record_maps_yf_columns_in_order() -> None:
    df = flatten_columns(_yf_frame({"2026-04-15": [10.0, 12.0, 9.5, 11.0, 10.8, 1000]}))
    ((_, row),) = df.iterrows()
    record = row_to_record("2026-04-15", row)
    # Key order defines the committed byte layout.
    assert list(record.keys()) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert record == {
        "date": "2026-04-15",
        "open": 10.0,
        "high": 12.0,
        "low": 9.5,
        "close": 11.0,
        "adj_close": 10.8,
        "volume": 1000,
    }


# --- cell coercion --------------------------------------------------------


def test_safe_float_handles_none_nan_and_series() -> None:
    assert safe_float(None) is None
    assert safe_float(np.nan) is None
    assert safe_float(pd.Series([], dtype="float64")) is None
    assert safe_float(pd.Series([3.5])) == 3.5
    assert safe_float(2) == 2.0


def test_safe_int_handles_none_nan_and_series() -> None:
    assert safe_int(None) is None
    assert safe_int(np.nan) is None
    assert safe_int(pd.Series([], dtype="float64")) is None
    assert safe_int(pd.Series([7.0])) == 7
    assert safe_int(4.0) == 4


# --- existing_dates -------------------------------------------------------


def test_existing_dates_missing_file_is_empty(tmp_path: Path) -> None:
    assert existing_dates(tmp_path / "GHOST.jsonl") == set()


def test_existing_dates_skips_blank_and_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "AAPL.jsonl"
    path.write_text(
        '{"date": "2026-04-15", "close": 1.0}\n'
        "\n"
        "not-json\n"
        '{"date": "2026-04-16", "close": 2.0}\n',
        encoding="utf-8",
    )
    assert existing_dates(path) == {"2026-04-15", "2026-04-16"}


def test_existing_dates_skips_valid_json_that_is_not_an_object(tmp_path: Path) -> None:
    # `42` and `["a"]` parse fine, but `.get` on them raises AttributeError.
    # Uncaught, one corrupt line would abort the whole ~1,150-symbol nightly
    # run — the opposite of the graceful degradation the docstring promises.
    path = tmp_path / "AAPL.jsonl"
    path.write_text(
        '{"date": "2026-04-15", "close": 1.0}\n'
        "42\n"
        '["not", "a", "dict"]\n'
        '{"date": "2026-04-16", "close": 2.0}\n',
        encoding="utf-8",
    )
    assert existing_dates(path) == {"2026-04-15", "2026-04-16"}


# --- build_new_rows -------------------------------------------------------


def test_build_new_rows_sorts_by_date() -> None:
    df = flatten_columns(
        _yf_frame(
            {
                "2026-04-17": [3, 3, 3, 3.0, 3.0, 30],
                "2026-04-15": [1, 1, 1, 1.0, 1.0, 10],
                "2026-04-16": [2, 2, 2, 2.0, 2.0, 20],
            }
        )
    )
    rows = build_new_rows(df, existing=set())
    assert [d for d, _ in rows] == ["2026-04-15", "2026-04-16", "2026-04-17"]


def test_build_new_rows_drops_rows_with_null_close() -> None:
    df = flatten_columns(
        _yf_frame(
            {
                "2026-04-15": [1, 1, 1, np.nan, 1.0, 10],  # missing close → dropped
                "2026-04-16": [2, 2, 2, 2.0, 2.0, 20],
            }
        )
    )
    rows = build_new_rows(df, existing=set())
    assert [d for d, _ in rows] == ["2026-04-16"]


def test_build_new_rows_skips_already_present_dates() -> None:
    df = flatten_columns(
        _yf_frame(
            {
                "2026-04-15": [1, 1, 1, 1.0, 1.0, 10],
                "2026-04-16": [2, 2, 2, 2.0, 2.0, 20],
            }
        )
    )
    rows = build_new_rows(df, existing={"2026-04-15"})
    assert [d for d, _ in rows] == ["2026-04-16"]


def test_build_new_rows_warns_when_a_row_has_no_close(caplog) -> None:
    import logging

    import pandas as pd

    from engine.ohlcv_ingest import build_new_rows

    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [None, 2.0],
            "Adj Close": [None, 2.0],
            "Volume": [10, 20],
        },
        index=pd.to_datetime(["2026-08-10", "2026-08-11"]),
    )
    with caplog.at_level(logging.WARNING, logger="engine.ohlcv_ingest"):
        rows = build_new_rows(df, set(), symbol="SAP.DE")

    assert [d for d, _ in rows] == ["2026-08-11"]
    assert "SAP.DE" in caplog.text
    assert "2026-08-10" in caplog.text


def test_build_new_rows_is_silent_when_every_row_has_a_close(caplog) -> None:
    import logging

    import pandas as pd

    from engine.ohlcv_ingest import build_new_rows

    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [2.0],
            "Adj Close": [2.0],
            "Volume": [10],
        },
        index=pd.to_datetime(["2026-08-11"]),
    )
    with caplog.at_level(logging.WARNING, logger="engine.ohlcv_ingest"):
        build_new_rows(df, set(), symbol="SAP.DE")

    assert caplog.text == ""


def test_merge_rows_warns_when_a_row_has_no_close(tmp_path, caplog) -> None:
    import logging

    import pandas as pd

    from engine.ohlcv_ingest import merge_rows

    path = tmp_path / "SAP.DE.jsonl"
    path.write_text(
        '{"date": "2026-08-07", "open": 1.0, "high": 1.0, "low": 1.0, '
        '"close": 1.0, "adj_close": 1.0, "volume": 1}\n',
        encoding="utf-8",
    )
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [None, 2.0],
            "Adj Close": [None, 2.0],
            "Volume": [10, 20],
        },
        index=pd.to_datetime(["2026-08-10", "2026-08-11"]),
    )
    with caplog.at_level(logging.WARNING, logger="engine.ohlcv_ingest"):
        result = merge_rows(path, df, revise_from="2026-08-11")

    assert result.appended == 1
    assert "SAP.DE" in caplog.text
    assert "2026-08-10" in caplog.text


# --- append_new_rows: merge + idempotency ---------------------------------


def test_append_writes_to_empty_store(tmp_path: Path) -> None:
    path = tmp_path / "AAPL.jsonl"
    df = flatten_columns(
        _yf_frame(
            {
                "2026-04-15": [10.0, 12.0, 9.5, 11.0, 10.8, 1000],
                "2026-04-16": [11.0, 13.0, 10.5, 12.0, 11.9, 2000],
            }
        )
    )
    n = append_new_rows(path, df)
    assert n == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["date"] == "2026-04-15"
    assert json.loads(lines[1])["date"] == "2026-04-16"
    # Exact byte layout of the first record (committed store contract).
    assert lines[0] == json.dumps(
        {
            "date": "2026-04-15",
            "open": 10.0,
            "high": 12.0,
            "low": 9.5,
            "close": 11.0,
            "adj_close": 10.8,
            "volume": 1000,
        }
    )


def test_append_merges_without_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "AAPL.jsonl"
    _write_store(
        path,
        [
            {
                "date": "2026-04-15",
                "open": 10.0,
                "high": 12.0,
                "low": 9.5,
                "close": 11.0,
                "adj_close": 10.8,
                "volume": 1000,
            },
        ],
    )
    # Frame overlaps 04-15 (already stored) and adds 04-16 (new).
    df = flatten_columns(
        _yf_frame(
            {
                "2026-04-15": [10.0, 12.0, 9.5, 11.0, 10.8, 1000],
                "2026-04-16": [11.0, 13.0, 10.5, 12.0, 11.9, 2000],
            }
        )
    )
    n = append_new_rows(path, df)
    assert n == 1  # only 04-16 is new
    dates = [json.loads(line)["date"] for line in path.read_text().splitlines()]
    assert dates == ["2026-04-15", "2026-04-16"]  # no dupes


def test_append_is_idempotent_on_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "AAPL.jsonl"
    df = flatten_columns(
        _yf_frame(
            {
                "2026-04-15": [10.0, 12.0, 9.5, 11.0, 10.8, 1000],
                "2026-04-16": [11.0, 13.0, 10.5, 12.0, 11.9, 2000],
            }
        )
    )
    assert append_new_rows(path, df) == 2
    before = path.read_text(encoding="utf-8")
    # Re-writing the same frame must add nothing and leave the file byte-identical.
    assert append_new_rows(path, df) == 0
    assert path.read_text(encoding="utf-8") == before


def test_append_empty_frame_returns_zero_and_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "AAPL.jsonl"
    empty = pd.DataFrame(index=pd.DatetimeIndex([], name="Date"), columns=_FIELDS)
    assert append_new_rows(path, empty) == 0
    assert not path.exists()


def test_append_all_null_close_frame_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "AAPL.jsonl"
    df = flatten_columns(_yf_frame({"2026-04-15": [1, 1, 1, np.nan, np.nan, 10]}))
    assert append_new_rows(path, df) == 0
    assert not path.exists()


# --- fetch window ---------------------------------------------------------


def test_fetch_window_start_includes_today_after_close() -> None:
    """Regression: e98ea8cf5 — the cron runs 22:30 UTC, after the US close,
    so a store holding yesterday must still fetch TODAY's available bar. The
    old `last >= end - 1 day` skip dropped it, leaving every other daily
    session on a two-market-day-old equity price."""
    assert fetch_window_start(date(2026, 8, 3), date(2026, 8, 4), 730) == date(
        2026, 8, 4
    )


def test_fetch_window_start_skips_when_today_already_stored() -> None:
    assert fetch_window_start(date(2026, 8, 4), date(2026, 8, 4), 730) is None


def test_fetch_window_start_skips_when_store_runs_ahead() -> None:
    assert fetch_window_start(date(2026, 8, 6), date(2026, 8, 4), 730) is None


def test_fetch_window_start_empty_store_uses_history_window() -> None:
    assert fetch_window_start(None, date(2026, 8, 4), 730) == date(
        2026, 8, 4
    ) - timedelta(days=730)


def test_fetch_window_start_revise_days_refetches_trailing_rows() -> None:
    """revise_days=1 re-requests the last stored day so a partial 24/7 bar can
    be corrected by its final value."""
    assert fetch_window_start(
        date(2026, 8, 4), date(2026, 8, 6), 730, revise_days=1
    ) == date(2026, 8, 4)


def test_fetch_window_start_revise_days_ignored_on_empty_store() -> None:
    assert fetch_window_start(None, date(2026, 8, 4), 730, revise_days=1) == date(
        2026, 8, 4
    ) - timedelta(days=730)


# --- property: idempotence of the merge -----------------------------------

_iso_dates = st.lists(
    st.dates(
        min_value=pd.Timestamp("2020-01-01").date(),
        max_value=pd.Timestamp("2030-12-31").date(),
    ).map(lambda d: d.isoformat()),
    min_size=1,
    max_size=8,
    unique=True,
)


@given(dates=_iso_dates)
def test_append_twice_is_a_no_op_property(dates: list[str], tmp_path_factory) -> None:
    """Property: appending a frame, then appending it again, adds zero new rows
    and leaves the store byte-identical. Idempotence is the core store invariant."""
    path = tmp_path_factory.mktemp("store") / "SYNTH.jsonl"
    rows = {d: [float(i + 1)] * 5 + [(i + 1) * 100] for i, d in enumerate(dates)}
    df = flatten_columns(_yf_frame(rows))
    first = append_new_rows(path, df)
    assert first == len(dates)
    snapshot = path.read_text(encoding="utf-8")
    assert append_new_rows(path, df) == 0
    assert path.read_text(encoding="utf-8") == snapshot


# --- revision merge -------------------------------------------------------


def _rec(d: str, close: float) -> dict:
    return {
        "date": d,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "adj_close": close,
        "volume": 100,
    }


def test_merge_rows_revises_a_frozen_partial_bar(tmp_path: Path) -> None:
    """Regression: 0cb72052c — a 24/7 crypto bar written at ~23:44 UTC is
    still forming, and append_new_rows keeps only UNSEEN dates, so the
    partial value was frozen permanently. BTC-EUR 2026-08-04 was stored at
    55649.74 against a true close of 55545.09."""
    path = tmp_path / "BTC-EUR.jsonl"
    _write_store(path, [_rec("2026-08-03", 55149.36), _rec("2026-08-04", 55649.74)])
    df = _yf_frame(
        {
            "2026-08-04": [1, 2, 0.5, 55545.09, 55545.09, 100],
            "2026-08-06": [1, 2, 0.5, 56241.04, 56241.04, 100],
        }
    )

    appended, revised, _quarantined, *_ = merge_rows(path, df, revise_from="2026-08-04")

    assert (appended, revised) == (1, 1)
    closes = [json.loads(line)["close"] for line in path.read_text().splitlines()]
    assert closes == [55149.36, 55545.09, 56241.04]


def test_merge_rows_leaves_rows_before_revise_from_untouched(tmp_path: Path) -> None:
    path = tmp_path / "BTC-EUR.jsonl"
    _write_store(path, [_rec("2026-08-03", 55149.36), _rec("2026-08-04", 55649.74)])
    df = _yf_frame(
        {
            "2026-08-03": [1, 2, 0.5, 99999.0, 99999.0, 100],
            "2026-08-04": [1, 2, 0.5, 55545.09, 55545.09, 100],
        }
    )

    appended, revised, _quarantined, *_ = merge_rows(path, df, revise_from="2026-08-04")

    assert (appended, revised) == (0, 1)
    closes = [json.loads(line)["close"] for line in path.read_text().splitlines()]
    assert closes == [55149.36, 55545.09]


def test_merge_rows_without_revise_from_is_pure_append(tmp_path: Path) -> None:
    path = tmp_path / "AAPL.jsonl"
    _write_store(path, [_rec("2026-08-03", 303.42)])
    df = _yf_frame(
        {
            "2026-08-03": [1, 2, 0.5, 99999.0, 99999.0, 100],
            "2026-08-04": [1, 2, 0.5, 309.38, 309.38, 100],
        }
    )

    assert merge_rows(path, df)[:3] == (1, 0, 0)
    closes = [json.loads(line)["close"] for line in path.read_text().splitlines()]
    assert closes == [303.42, 309.38]


def test_merge_rows_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "BTC-EUR.jsonl"
    _write_store(path, [_rec("2026-08-04", 55545.09)])
    # The frame must reproduce the stored record field-for-field — _rec sets
    # every OHLC field to the same value, so a frame with different open/high/low
    # would legitimately count as a revision and this test would not be measuring
    # idempotency at all.
    df = _yf_frame({"2026-08-04": [55545.09] * 5 + [100]})

    assert merge_rows(path, df, revise_from="2026-08-04")[:3] == (0, 0, 0)
    before = path.read_text()
    assert merge_rows(path, df, revise_from="2026-08-04")[:3] == (0, 0, 0)
    assert path.read_text() == before


def test_merge_rows_preserves_byte_layout_and_key_order(tmp_path: Path) -> None:
    path = tmp_path / "BTC-EUR.jsonl"
    _write_store(path, [_rec("2026-08-04", 55649.74)])
    df = _yf_frame({"2026-08-04": [1.0, 2.0, 0.5, 55545.09, 55545.09, 100]})

    merge_rows(path, df, revise_from="2026-08-04")

    line = path.read_text().splitlines()[0]
    assert list(json.loads(line).keys()) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert line == json.dumps(
        {
            "date": "2026-08-04",
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 55545.09,
            "adj_close": 55545.09,
            "volume": 100,
        }
    )
    assert path.read_text().endswith("\n")


def test_merge_rows_keeps_an_untouched_row_byte_identical_during_a_mixed_rewrite(
    tmp_path: Path,
) -> None:
    """A rewrite triggered by one row revising must not reformat a row it did not
    touch — even a lossless-looking re-serialize through json.dumps would violate
    the load-bearing on-disk byte layout. The untouched line is deliberately
    written with non-default JSON spacing (no space after ':' or ',') so a
    re-serializing implementation is caught: json.dumps' default separators
    would normalize it back to spaced form, producing a detectable diff."""
    path = tmp_path / "BTC-EUR.jsonl"
    untouched_line = (
        '{"date":"2026-08-03","open":55149.36,"high":55149.36,"low":55149.36,'
        '"close":55149.36,"adj_close":55149.36,"volume":100}'
    )
    path.write_text(
        untouched_line + "\n" + json.dumps(_rec("2026-08-04", 55649.74)) + "\n",
        encoding="utf-8",
    )
    df = _yf_frame({"2026-08-04": [1, 2, 0.5, 55545.09, 55545.09, 100]})

    appended, revised, _quarantined, *_ = merge_rows(path, df, revise_from="2026-08-04")

    assert (appended, revised) == (0, 1)
    lines = path.read_text().splitlines()
    assert lines[0] == untouched_line


def test_merge_rows_preserves_an_out_of_order_store(tmp_path: Path) -> None:
    """The store is NOT in date order and must not be re-sorted by a cron.

    529 of the 1,046 committed OHLCV files have an order break: a later
    long-history backfill was appended behind the original window, so the file
    reads recent-dates-then-an-older-block (`3EUS.L` breaks at line 513 of
    2581, 2026-04-24 -> 2016-04-28). Now that the revision window is universal,
    `appended >= 1` on every nightly run, so every symbol takes this rewrite
    path — a `sorted()` here would make an unattended job emit a ~230 MB,
    1.3-million-line reorder commit. Readers are order-insensitive;
    canonicalising the store is a separate, deliberate decision.

    Mirrors the real shape: recent block first, older block behind it, with the
    revision landing in the middle of the file.
    """
    path = tmp_path / "3EUS.L.jsonl"
    recent = ["2026-04-23", "2026-04-24"]
    older = ["2016-04-28", "2016-04-29"]
    original = [json.dumps(_rec(d, 100.0)) for d in recent + older]
    path.write_text("\n".join(original) + "\n", encoding="utf-8")

    # Revises 2026-04-24 (row 2 of 4 — mid-file) and appends a new trailing day.
    df = _yf_frame(
        {
            "2026-04-24": [1, 2, 0.5, 111.11, 111.11, 100],
            "2026-04-25": [1, 2, 0.5, 222.22, 222.22, 100],
        }
    )

    appended, revised, _quarantined, *_ = merge_rows(path, df, revise_from="2026-04-24")

    assert (appended, revised) == (1, 1)
    dates = [json.loads(line)["date"] for line in path.read_text().splitlines()]
    # Original order, break and all — NOT sorted. New row at the end.
    assert dates == [
        "2026-04-23",
        "2026-04-24",
        "2016-04-28",
        "2016-04-29",
        "2026-04-25",
    ]
    assert dates != sorted(dates)
    lines = path.read_text().splitlines()
    assert lines[0] == original[0]  # untouched rows keep their exact bytes
    assert lines[2] == original[2]
    assert lines[3] == original[3]
    assert json.loads(lines[1])["close"] == 111.11  # revised in place


def test_merge_rows_appends_multiple_new_dates_in_ascending_order(
    tmp_path: Path,
) -> None:
    """New dates are the one thing that IS sorted — among themselves, so a
    multi-day catch-up lands chronologically at the end rather than in whatever
    order the frame happens to iterate."""
    path = tmp_path / "3EUS.L.jsonl"
    path.write_text(
        json.dumps(_rec("2026-04-24", 100.0))
        + "\n"
        + json.dumps(_rec("2016-04-28", 100.0))
        + "\n",
        encoding="utf-8",
    )
    df = _yf_frame(
        {
            "2026-04-27": [1, 2, 0.5, 3.0, 3.0, 100],
            "2026-04-25": [1, 2, 0.5, 1.0, 1.0, 100],
            "2026-04-26": [1, 2, 0.5, 2.0, 2.0, 100],
        }
    )

    assert merge_rows(path, df, revise_from="2026-04-24")[:3] == (3, 0, 0)
    dates = [json.loads(line)["date"] for line in path.read_text().splitlines()]
    assert dates == [
        "2026-04-24",
        "2016-04-28",  # pre-existing break survives
        "2026-04-25",
        "2026-04-26",
        "2026-04-27",
    ]


def test_merge_rows_writes_an_empty_store_in_ascending_order(tmp_path: Path) -> None:
    """On a new file every row is new, so order preservation degrades to
    ascending — a fresh store is never born out of order."""
    path = tmp_path / "NEW.jsonl"
    df = _yf_frame(
        {
            "2026-04-26": [1, 2, 0.5, 2.0, 2.0, 100],
            "2026-04-24": [1, 2, 0.5, 0.0, 0.0, 100],
            "2026-04-25": [1, 2, 0.5, 1.0, 1.0, 100],
        }
    )

    assert merge_rows(path, df, revise_from="2026-04-24")[:3] == (3, 0, 0)
    dates = [json.loads(line)["date"] for line in path.read_text().splitlines()]
    assert dates == sorted(dates) == ["2026-04-24", "2026-04-25", "2026-04-26"]


def test_merge_rows_refuses_to_rewrite_a_store_with_unparseable_lines(
    tmp_path: Path,
) -> None:
    # A rewrite is driven by a date-keyed map, so a line carrying no parseable
    # date has no key to survive under. Fall back to pure append, not loss.
    path = tmp_path / "BTC-EUR.jsonl"
    path.write_text(
        json.dumps(_rec("2026-08-04", 55649.74)) + "\nnot json at all\n",
        encoding="utf-8",
    )
    df = _yf_frame({"2026-08-04": [1, 2, 0.5, 55545.09, 55545.09, 100]})

    assert merge_rows(path, df, revise_from="2026-08-04")[:3] == (0, 0, 0)
    assert "not json at all" in path.read_text()


def test_merge_rows_treats_a_non_object_json_line_as_unparseable(
    tmp_path: Path,
) -> None:
    # Same defence as existing_dates: `42` is valid JSON, so json.loads succeeds
    # and `.get` raises AttributeError. merge_rows must degrade to append-only
    # rather than crash the run on one corrupt line.
    path = tmp_path / "BTC-EUR.jsonl"
    path.write_text(
        json.dumps(_rec("2026-08-04", 55649.74)) + "\n42\n",
        encoding="utf-8",
    )
    df = _yf_frame({"2026-08-04": [1, 2, 0.5, 55545.09, 55545.09, 100]})

    assert merge_rows(path, df, revise_from="2026-08-04")[:3] == (0, 0, 0)
    assert path.read_text().splitlines()[1] == "42"


def test_merge_rows_leaves_no_tmp_file_behind_when_the_write_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fetch-ohlcv.yml does a blanket `git add data/market/ohlcv/`, so a stray
    # `{SYMBOL}.jsonl.tmp` from a crashed rewrite would be committed into the
    # store permanently (.gitignore is the second line of defence, not the
    # first). Crash the rewrite mid-flight and assert nothing is left behind.
    path = tmp_path / "BTC-EUR.jsonl"
    original = json.dumps(_rec("2026-08-04", 55649.74)) + "\n"
    path.write_text(original, encoding="utf-8")
    df = _yf_frame({"2026-08-04": [1, 2, 0.5, 55545.09, 55545.09, 100]})

    def boom(fd: int) -> None:
        raise OSError("disk went away")

    monkeypatch.setattr(ohlcv_ingest.os, "fsync", boom)

    with pytest.raises(OSError):
        merge_rows(path, df, revise_from="2026-08-04")

    assert list(tmp_path.glob("*.tmp")) == []
    assert path.read_text(encoding="utf-8") == original  # store never truncated


def test_merge_rows_leaves_no_tmp_file_behind_on_success(tmp_path: Path) -> None:
    path = tmp_path / "BTC-EUR.jsonl"
    path.write_text(json.dumps(_rec("2026-08-04", 55649.74)) + "\n", encoding="utf-8")
    df = _yf_frame({"2026-08-04": [1, 2, 0.5, 55545.09, 55545.09, 100]})

    assert merge_rows(path, df, revise_from="2026-08-04")[:3] == (0, 1, 0)
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Ingest anomaly tripwire (reliability review W5.1)
#
# merge_rows is the only place in the system that holds BOTH the old value and
# the new one for the same date. That makes it the only place a vendor unit
# flip, a re-denomination or a bad tick can be caught *before* it becomes that
# evening's fill price. Every such defect so far was found by a human reading a
# number that looked wrong, weeks to months later.
#
# The design constraint is not detection — a 100x jump is trivial to spot. It
# is detecting that without refusing the legitimate revisions the store depends
# on, which the universal 1-day revision window produces every single night.
# ---------------------------------------------------------------------------


def _store(path: Path, rows: list[tuple[str, float]]) -> None:
    path.write_text(
        "\n".join(json.dumps({"date": d, "close": c}) for d, c in rows) + "\n",
        encoding="utf-8",
    )


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [c for _, c in rows]},
        index=pd.to_datetime([d for d, _ in rows]),
    )


class TestQuarantineCatches:
    def test_a_hundred_times_unit_flip_on_a_revision(self, tmp_path):
        """The GBp shape: the vendor starts serving pence where it served pounds."""
        from engine.ohlcv_ingest import merge_rows

        path = tmp_path / "LLOY.L.jsonl"
        quarantine = tmp_path / "q" / "LLOY.L.jsonl"
        _store(path, [("2026-08-05", 1.15), ("2026-08-06", 1.16)])

        result = merge_rows(
            path,
            _frame([("2026-08-06", 116.0)]),
            revise_from="2026-08-06",
            quarantine=quarantine,
        )

        assert result.quarantined == 1
        assert result.revised == 0
        # The store is untouched — this is the whole point. A quarantined row
        # that still landed would be a warning, not a guard.
        assert json.loads(path.read_text().splitlines()[-1])["close"] == 1.16
        record = json.loads(quarantine.read_text().splitlines()[0])
        assert record["kind"] == "revision"
        assert record["ratio"] == pytest.approx(100.0)

    def test_the_fcit_bad_tick_as_a_new_row(self, tmp_path):
        """The real 2026-05-08 FCIT.L tick: 5,275.02 against a 320-330 range.

        Named in the review as a must-catch fixture. It reached the committed
        store and was only corrected months later by a full resweep.
        """
        from engine.ohlcv_ingest import merge_rows

        path = tmp_path / "FCIT.L.jsonl"
        quarantine = tmp_path / "q" / "FCIT.L.jsonl"
        _store(path, [("2026-05-06", 322.4), ("2026-05-07", 320.8)])

        result = merge_rows(
            path,
            _frame([("2026-05-08", 5275.02)]),
            revise_from="2026-05-08",
            quarantine=quarantine,
        )

        assert result.quarantined == 1
        assert result.appended == 0
        assert "2026-05-08" not in path.read_text()

    def test_the_quarantine_record_is_diagnosable(self, tmp_path):
        """ "anomaly" is not actionable; the old value, the new one and the
        ratio are."""
        from engine.ohlcv_ingest import merge_rows

        path = tmp_path / "AAA.jsonl"
        quarantine = tmp_path / "q" / "AAA.jsonl"
        _store(path, [("2026-05-06", 100.0)])
        merge_rows(
            path,
            _frame([("2026-05-07", 10_000.0)]),
            revise_from="2026-05-07",
            quarantine=quarantine,
        )

        record = json.loads(quarantine.read_text().splitlines()[0])
        assert record["symbol"] == "AAA"
        assert record["stored_close"] == 100.0
        assert record["incoming_close"] == 10_000.0
        assert record["ratio"] == pytest.approx(100.0)


class TestQuarantineDoesNotCatch:
    """The half that decides whether this guard survives contact with the cron.

    Thresholds were set from measured legitimate revisions: futures drift up to
    +3.37% between the 22:30 UTC fetch and the final close, FX up to -1.56% on
    a Friday. A guard that fires on those files an issue every night, and gets
    turned off.
    """

    @pytest.mark.parametrize(
        "label, stored, incoming",
        [
            ("commodity futures partial bar, worst measured", 100.0, 103.37),
            ("FX Friday roll, worst measured", 100.0, 98.44),
            ("a crypto bar still forming", 100.0, 108.0),
            ("right at the edge, inside", 100.0, 119.0),
        ],
    )
    def test_legitimate_revisions_pass(self, tmp_path, label, stored, incoming):
        from engine.ohlcv_ingest import merge_rows

        path = tmp_path / "GC=F.jsonl"
        quarantine = tmp_path / "q" / "GC=F.jsonl"
        _store(path, [("2026-08-05", 99.0), ("2026-08-06", stored)])

        result = merge_rows(
            path,
            _frame([("2026-08-06", incoming)]),
            revise_from="2026-08-06",
            quarantine=quarantine,
        )

        assert result.quarantined == 0, label
        assert result.revised == 1, label
        assert not quarantine.exists()

    def test_an_ordinary_new_bar_passes(self, tmp_path):
        from engine.ohlcv_ingest import merge_rows

        path = tmp_path / "AAPL.jsonl"
        quarantine = tmp_path / "q" / "AAPL.jsonl"
        _store(path, [("2026-08-05", 190.0)])

        result = merge_rows(
            path,
            _frame([("2026-08-06", 193.5)]),
            revise_from="2026-08-06",
            quarantine=quarantine,
        )

        assert result.quarantined == 0
        assert result.appended == 1

    def test_a_first_ever_row_has_no_reference_and_passes(self, tmp_path):
        """An empty store cannot produce an anomaly — only a bootstrap."""
        from engine.ohlcv_ingest import merge_rows

        path = tmp_path / "NEW.jsonl"
        quarantine = tmp_path / "q" / "NEW.jsonl"

        result = merge_rows(
            path,
            _frame([("2026-08-06", 4321.0)]),
            revise_from="2026-08-06",
            quarantine=quarantine,
        )

        assert result.quarantined == 0
        assert result.appended == 1

    def test_the_guard_is_off_unless_asked_for(self, tmp_path):
        """Opt-in: the resweep path must keep rewriting history freely, because
        that is where detect_split does the adjudicating."""
        from engine.ohlcv_ingest import merge_rows

        path = tmp_path / "LLOY.L.jsonl"
        _store(path, [("2026-08-06", 1.16)])

        result = merge_rows(
            path, _frame([("2026-08-06", 116.0)]), revise_from="2026-08-06"
        )

        assert result.quarantined == 0
        assert result.revised == 1


def test_quarantine_lives_outside_the_store_directory():
    """A refused row must never be one glob("*.jsonl") away from being a price.

    Every reader opens `{ohlcv_dir}/{TICKER}.jsonl` by name today, so a sidecar
    inside that directory would be safe *today* — which is exactly the kind of
    reasoning that stops being true later.
    """
    import inspect

    from scripts import fetch_ohlcv

    source = inspect.getsource(fetch_ohlcv._write_rows)
    assert '"quarantine"' in source
    assert "ohlcv_dir" not in source.split("quarantine")[1]


# ---------------------------------------------------------------------------
# Corrupt store lines: loud, and no duplicate re-append (W2.5)
# ---------------------------------------------------------------------------


class TestUnparseableLineDegradation:
    """A store file with one bad line silently disabled revision for that
    symbol — permanently, for every future run — and then re-appended the
    date the bad line carried, because `existing_dates` could not see it."""

    def test_degradation_warns(self, tmp_path: Path, caplog) -> None:
        import logging

        path = tmp_path / "BTC-EUR.jsonl"
        path.write_text(
            json.dumps(_rec("2026-08-03", 100.0)) + "\n" + '{"date": "2026-08-04"\n',
            encoding="utf-8",
        )
        df = _yf_frame({"2026-08-05": [1, 2, 0.5, 110.0, 110.0, 100]})

        with caplog.at_level(logging.WARNING, logger="engine.ohlcv_ingest"):
            merge_rows(path, df, revise_from="2026-08-04")

        assert "revision is DISABLED" in caplog.text
        assert "BTC-EUR.jsonl" in caplog.text

    def test_the_broken_lines_date_is_not_re_appended(self, tmp_path: Path) -> None:
        """The duplicate. The bad line carries 2026-08-04; the fetched frame
        also carries it; `existing_dates` cannot parse it, so without the
        salvage the store gains a SECOND row for that date."""
        path = tmp_path / "BTC-EUR.jsonl"
        path.write_text(
            json.dumps(_rec("2026-08-03", 100.0))
            + "\n"
            + '{"date": "2026-08-04", "close": 105.0, "open"\n',
            encoding="utf-8",
        )
        df = _yf_frame(
            {
                "2026-08-04": [1, 2, 0.5, 106.0, 106.0, 100],
                "2026-08-05": [1, 2, 0.5, 110.0, 110.0, 100],
            }
        )

        merge_rows(path, df, revise_from="2026-08-04")

        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        dates = []
        for ln in lines:
            try:
                dates.append(json.loads(ln)["date"])
            except json.JSONDecodeError:
                dates.append("2026-08-04")  # the broken line, by inspection
        assert sorted(dates) == ["2026-08-03", "2026-08-04", "2026-08-05"]

    def test_the_broken_line_survives(self, tmp_path: Path) -> None:
        """Append-only degradation exists to preserve it — a rewrite from a
        date-keyed map would drop the line entirely."""
        path = tmp_path / "BTC-EUR.jsonl"
        broken = '{"date": "2026-08-04", "close": 105.0, "open"'
        path.write_text(
            json.dumps(_rec("2026-08-03", 100.0)) + "\n" + broken + "\n",
            encoding="utf-8",
        )
        merge_rows(
            path,
            _yf_frame({"2026-08-05": [1, 2, 0.5, 110.0, 110.0, 100]}),
            revise_from="2026-08-04",
        )
        assert broken in path.read_text()

    def test_a_clean_store_still_revises(self, tmp_path: Path, caplog) -> None:
        """The control: the warning must not fire, and revision must still
        work, on a store with no bad lines."""
        import logging

        path = tmp_path / "BTC-EUR.jsonl"
        _write_store(path, [_rec("2026-08-03", 100.0), _rec("2026-08-04", 105.0)])

        with caplog.at_level(logging.WARNING, logger="engine.ohlcv_ingest"):
            result = merge_rows(
                path,
                _yf_frame({"2026-08-04": [1, 2, 0.5, 106.0, 106.0, 100]}),
                revise_from="2026-08-04",
            )

        assert result.revised == 1
        assert "revision is DISABLED" not in caplog.text
