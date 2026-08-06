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


@settings(max_examples=200)
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
    """Regression: 63970d933 — a 24/7 crypto bar written at ~23:44 UTC is
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

    appended, revised = merge_rows(path, df, revise_from="2026-08-04")

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

    appended, revised = merge_rows(path, df, revise_from="2026-08-04")

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

    assert merge_rows(path, df) == (1, 0)
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

    assert merge_rows(path, df, revise_from="2026-08-04") == (0, 0)
    before = path.read_text()
    assert merge_rows(path, df, revise_from="2026-08-04") == (0, 0)
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

    appended, revised = merge_rows(path, df, revise_from="2026-08-04")

    assert (appended, revised) == (0, 1)
    lines = path.read_text().splitlines()
    assert lines[0] == untouched_line


def test_merge_rows_refuses_to_rewrite_a_store_with_unparseable_lines(
    tmp_path: Path,
) -> None:
    # A rewrite reorders by date and would silently drop a line carrying no
    # parseable date. Fall back to pure append instead of losing it.
    path = tmp_path / "BTC-EUR.jsonl"
    path.write_text(
        json.dumps(_rec("2026-08-04", 55649.74)) + "\nnot json at all\n",
        encoding="utf-8",
    )
    df = _yf_frame({"2026-08-04": [1, 2, 0.5, 55545.09, 55545.09, 100]})

    assert merge_rows(path, df, revise_from="2026-08-04") == (0, 0)
    assert "not json at all" in path.read_text()
