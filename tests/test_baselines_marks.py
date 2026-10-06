"""Passive benchmark rows record their marks; the merge classifies mismatches.

Background (plan subtask 1.5, revised 2026-10-04). `merge_baseline_series`
refused every published point a recomputation disagreed with as one
undifferentiated count: 1,830 refusals on the 2026-10-01 session, of which
11 were genuinely wrong prices. Most of the rest were benchmark points that
forward-filled a close which had not landed yet (the store filled it later,
so the recomputation moves forever) and coin-flip path recomputes. A count
nobody can act on is a guard nobody reads.

So a benchmark row now records which closes it was priced from
(`mark_date`/`mark_close` and `base_date`/`base_close`), and a mismatch is
classified against the store:

- ``concern``: the store's mark/base ratio differs from the recorded one —
  a published close was revised. The only class that reaches the session's
  ``Concerns:`` path.
- ``stale_mark``: the row forward-filled, and the store now holds a close
  after its mark. Expected; the published row is right for what it saw.
- ``rescaled``: the ratio holds but the closes changed (a units or split
  rebase of the whole history). A ratio series cancels a constant factor.
- ``unclassified``: a legacy row with no entry in a readable marks sidecar.
  A legacy row whose sidecar is missing or unreadable is a ``concern``: the
  sidecar is what classifies it, so its absence fails toward the finding.

The coin flip no longer comes through this merge: since plan 1.6 it advances
from a persisted state (``tests/test_coinflip_state.py``).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from engine.baselines import (
    MergeCounts,
    build_all_baselines,
    compute_passive_benchmark,
    merge_baseline_series,
)
from engine.config import CASH_FLAT_TICKER, BenchmarkSpec, get_config

_SPEC = BenchmarkSpec("Test", "TEST", "EUR")


def _write_store(ticker: str, rows: list[tuple[str, float]]) -> None:
    ohlcv = get_config().ohlcv_dir
    ohlcv.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"date": d, "close": c}) for d, c in rows]
    (ohlcv / f"{ticker}.jsonl").write_text("\n".join(lines) + "\n")


def _closes(rows: list[tuple[str, float]]) -> dict[str, float]:
    return {d: c for d, c in rows}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


def _strip_marks(rows: list[dict]) -> list[dict]:
    keep = ("date", "portfolio_value", "cash", "positions_value", "currency")
    return [{k: r[k] for k in keep} for r in rows]


def _sidecar(rows: list[dict]) -> list[dict]:
    keys = ("date", "mark_date", "mark_close", "base_date", "base_close")
    return [{k: r[k] for k in keys} for r in rows]


# Thu 04-16 is the base, Fri 04-17 trades, Mon 04-20 has not landed yet when
# the row is first published, so the Mon row forward-fills Friday's close.
_FIRST_STORE = [("2026-04-16", 100.0), ("2026-04-17", 110.0)]
_FROM, _TO = date(2026, 4, 16), date(2026, 4, 20)


def _publish(path: Path, store: list[tuple[str, float]], *, legacy: bool = False):
    _write_store("TEST", store)
    rows = compute_passive_benchmark(_SPEC, _FROM, _TO)
    if legacy:
        _write(path, _strip_marks(rows))
        _write(path.with_name("benchmark_marks.json"), _sidecar(rows))
    else:
        _write(path, rows)
    return rows


def _remerge(path: Path, store: list[tuple[str, float]], **kw) -> MergeCounts:
    _write_store("TEST", store)
    computed = compute_passive_benchmark(_SPEC, _FROM, _TO)
    return merge_baseline_series(path, computed, closes=_closes(store), **kw)


# ---------------------------------------------------------------------------
# compute_passive_benchmark records its marks
# ---------------------------------------------------------------------------


def test_rows_record_the_closes_they_were_priced_from(midas_data_root):
    _write_store("TEST", _FIRST_STORE)
    rows = {r["date"]: r for r in compute_passive_benchmark(_SPEC, _FROM, _TO)}

    assert rows["2026-04-16"]["mark_date"] == "2026-04-16"
    assert rows["2026-04-16"]["base_date"] == "2026-04-16"
    # Saturday..Monday forward-fill Friday: the mark says so.
    for d in ("2026-04-18", "2026-04-19", "2026-04-20"):
        assert rows[d]["mark_date"] == "2026-04-17"
        assert rows[d]["mark_close"] == 110.0
        assert rows[d]["base_date"] == "2026-04-16"
        assert rows[d]["base_close"] == 100.0
    # The value is exactly what the marks say, by construction.
    r = rows["2026-04-20"]
    assert r["portfolio_value"] == get_config().initial_capital * (
        r["mark_close"] / r["base_close"]
    )


def test_cash_flat_rows_carry_no_marks(midas_data_root):
    """EUR_CASH_FLAT reads no price: there is no close to record."""
    spec = BenchmarkSpec("Cash", CASH_FLAT_TICKER, "EUR")
    rows = compute_passive_benchmark(spec, _FROM, _TO)
    assert all("mark_date" not in r for r in rows)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_identical_replay_counts_nothing(midas_data_root, tmp_path, capsys):
    path = tmp_path / "benchmark.json"
    _publish(path, _FIRST_STORE)
    assert _remerge(path, _FIRST_STORE) == MergeCounts()
    assert "[WARN]" not in capsys.readouterr().out


def test_a_late_close_is_a_stale_mark_not_a_concern(midas_data_root, tmp_path, capsys):
    path = tmp_path / "benchmark.json"
    published = _publish(path, _FIRST_STORE)

    counts = _remerge(path, _FIRST_STORE + [("2026-04-20", 120.0)])

    # Only the Monday row forward-filled across a close that later landed.
    assert counts == MergeCounts(stale_mark=1)
    assert counts.concern == 0
    assert "[WARN]" not in capsys.readouterr().out
    assert json.loads(path.read_text()) == published, "published rows must not move"


def test_a_revised_close_on_a_recorded_mark_is_a_concern(midas_data_root, tmp_path, capsys):
    """The fail-once check: a planted revision of a recorded mark fires."""
    path = tmp_path / "benchmark.json"
    published = _publish(path, _FIRST_STORE)

    counts = _remerge(path, [("2026-04-16", 100.0), ("2026-04-17", 111.0)])

    # 04-17 and the three days forward-filling it; 04-16 is its own base.
    assert counts == MergeCounts(revised=4) and counts.concern == 4
    out = capsys.readouterr().out
    assert out.count("[WARN]") == 4
    assert "2026-04-17" in out and "110.0" in out and "111.0" in out
    assert json.loads(path.read_text()) == published
    # Review M1: the concern names its remedy, as the scope that would fix it.
    assert f"{path.parent.name}/benchmark@2026-04-17" in out
    assert "METHODOLOGY changelog anchor" in out
    assert "human-authored" in out and "[restate]" in out


def test_a_revised_base_is_a_concern(midas_data_root, tmp_path):
    path = tmp_path / "benchmark.json"
    _publish(path, _FIRST_STORE)
    counts = _remerge(path, [("2026-04-16", 99.0), ("2026-04-17", 110.0)])
    # 04-16 itself is excluded (mark_date == base_date); the four later rows
    # all divide by the revised base.
    assert counts.concern == 4
    assert counts.rescaled == 1


def test_a_whole_history_rescale_is_not_a_concern(midas_data_root, tmp_path, capsys):
    """The other half of the fail-once check: x2 across the store cancels."""
    path = tmp_path / "benchmark.json"
    _publish(path, _FIRST_STORE)

    counts = _remerge(path, [(d, c * 2) for d, c in _FIRST_STORE])

    assert counts == MergeCounts(rescaled=5)
    assert "[WARN]" not in capsys.readouterr().out


def test_an_inexact_rescale_is_still_not_a_concern(midas_data_root, tmp_path):
    """x3 does not divide out exactly in binary floating point; 1e-12 relative does."""
    path = tmp_path / "benchmark.json"
    store = [("2026-04-16", 100.1), ("2026-04-17", 123.45)]
    _publish(path, store)
    scaled = [(d, c * 3) for d, c in store]
    assert (scaled[1][1] / scaled[0][1]) != (store[1][1] / store[0][1]), (
        "fixture must exercise the tolerance, not exact equality"
    )
    counts = _remerge(path, scaled)
    assert counts.concern == 0
    assert counts.rescaled == 5


def test_a_rescale_plus_a_late_close_is_a_stale_mark(midas_data_root, tmp_path):
    path = tmp_path / "benchmark.json"
    _publish(path, _FIRST_STORE)
    counts = _remerge(
        path, [(d, c * 2) for d, c in _FIRST_STORE] + [("2026-04-20", 240.0)]
    )
    assert counts == MergeCounts(stale_mark=1, rescaled=4)


def test_a_mark_the_store_no_longer_holds_is_a_concern(midas_data_root, tmp_path):
    """An unconfirmable mark is not a confirmed one: fail toward the concern."""
    path = tmp_path / "benchmark.json"
    _publish(path, _FIRST_STORE)
    counts = _remerge(path, [("2026-04-16", 100.0), ("2026-04-20", 110.0)])
    assert counts.concern == 4


def test_the_initial_row_is_never_a_concern(midas_data_root, tmp_path):
    """mark_date == base_date: the value is the initial capital by construction."""
    path = tmp_path / "benchmark.json"
    _write_store("TEST", [("2026-04-16", 100.0)])
    _write(path, compute_passive_benchmark(_SPEC, _FROM, _FROM))

    _write_store("TEST", [("2026-04-16", 50.0)])
    counts = merge_baseline_series(
        path,
        compute_passive_benchmark(_SPEC, _FROM, _FROM),
        closes={"2026-04-16": 50.0},
    )
    assert counts == MergeCounts(rescaled=1)


# ---------------------------------------------------------------------------
# Legacy rows: no fields on the row, marks in the sidecar
# ---------------------------------------------------------------------------


def test_adding_the_fields_refuses_nothing_on_a_legacy_row(midas_data_root, tmp_path):
    """A legacy row compares on value and currency only, and is never rewritten."""
    path = tmp_path / "benchmark.json"
    _write_store("TEST", _FIRST_STORE)
    legacy = _strip_marks(compute_passive_benchmark(_SPEC, _FROM, _TO))
    _write(path, legacy)
    before = path.read_bytes()

    counts = _remerge(path, _FIRST_STORE)

    assert counts == MergeCounts()
    assert path.read_bytes() == before


def test_a_legacy_row_whose_sidecar_is_missing_is_a_concern(midas_data_root, tmp_path, capsys):
    """Review fix 3: a missing sidecar used to read as `{}`, so every
    mismatched legacy row was `unclassified` ([INFO], not a concern) — the
    guard failed open on the one file that classifies those rows."""
    path = tmp_path / "benchmark.json"
    _write_store("TEST", _FIRST_STORE)
    _write(path, _strip_marks(compute_passive_benchmark(_SPEC, _FROM, _TO)))

    counts = _remerge(path, [("2026-04-16", 100.0), ("2026-04-17", 111.0)])

    assert counts == MergeCounts(sidecar=4) and counts.concern == 4
    warns = [l for l in capsys.readouterr().out.splitlines() if "[WARN]" in l]
    assert len(warns) == 4 and all("benchmark_marks.json is missing" in l for l in warns)


def test_a_legacy_row_absent_from_a_readable_sidecar_is_unclassified(
    midas_data_root, tmp_path, capsys
):
    path = tmp_path / "benchmark.json"
    _write_store("TEST", _FIRST_STORE)
    _write(path, _strip_marks(compute_passive_benchmark(_SPEC, _FROM, _TO)))
    _write(path.with_name("benchmark_marks.json"), [])

    counts = _remerge(path, [("2026-04-16", 100.0), ("2026-04-17", 111.0)])

    assert counts == MergeCounts(unclassified=4)
    assert "[WARN]" not in capsys.readouterr().out


def test_a_missing_sidecar_beside_mark_bearing_rows_only_is_no_concern(
    midas_data_root, tmp_path, capsys
):
    """No legacy row, no sidecar needed: the rows carry their own marks."""
    path = tmp_path / "benchmark.json"
    _publish(path, _FIRST_STORE)
    assert _remerge(path, _FIRST_STORE + [("2026-04-20", 120.0)]) == MergeCounts(
        stale_mark=1
    )
    assert "[WARN]" not in capsys.readouterr().out


def test_the_sidecar_classifies_a_legacy_row(midas_data_root, tmp_path):
    path = tmp_path / "benchmark.json"
    _publish(path, _FIRST_STORE, legacy=True)
    assert _remerge(path, _FIRST_STORE + [("2026-04-20", 120.0)]) == MergeCounts(
        stale_mark=1
    )
    assert _remerge(path, [("2026-04-16", 100.0), ("2026-04-17", 111.0)]).concern == 4
    assert _remerge(path, [(d, c * 2) for d, c in _FIRST_STORE]) == MergeCounts()


def test_row_fields_outrank_the_sidecar(midas_data_root, tmp_path):
    path = tmp_path / "benchmark.json"
    rows = _publish(path, _FIRST_STORE)
    # A sidecar that disagrees with the row: the row's own fields win.
    wrong = [dict(r, mark_close=999.0) for r in _sidecar(rows)]
    _write(path.with_name("benchmark_marks.json"), wrong)
    assert _remerge(path, _FIRST_STORE + [("2026-04-20", 120.0)]) == MergeCounts(
        stale_mark=1
    )


def test_an_unreadable_sidecar_is_a_counted_concern(midas_data_root, tmp_path, capsys):
    """Review fix 3: the unreadable-sidecar [WARN] is one concern of its own,
    and each mismatched legacy row it can no longer classify is another."""
    path = tmp_path / "benchmark.json"
    _publish(path, _FIRST_STORE, legacy=True)
    path.with_name("benchmark_marks.json").write_text("{not json")
    counts = _remerge(path, [("2026-04-16", 100.0), ("2026-04-17", 111.0)])
    assert counts == MergeCounts(sidecar=4, sidecar_file=1) and counts.concern == 5
    assert counts.mismatched == 4, "the file is a concern, not a mismatched row"
    out = capsys.readouterr().out
    assert "benchmark_marks.json is unreadable" in out


def test_an_unreadable_sidecar_is_a_concern_of_the_build(midas_data_root, capsys):
    """The build's totals and aggregate line carry it, even with no mismatch."""
    cfg = get_config()
    universes = _seed_desk(cfg, {})
    _build(universes)
    agent = next(a for a in cfg.trading_roster if cfg.roster[a].benchmark is not None
                 and not cfg.roster[a].benchmark.is_cash_flat)
    path = cfg.baselines_dir / agent / "benchmark.json"
    _write(path, _strip_marks(json.loads(path.read_text())))
    path.with_name("benchmark_marks.json").write_text("{not json")
    capsys.readouterr()
    totals = _build(universes)
    assert totals == MergeCounts(sidecar_file=1)
    out = capsys.readouterr().out
    # Regression: round-3 review, 2026-10-06 — the aggregate line called
    # this a price revision. It names the sidecar and never "revised".
    aggregate = next(l for l in out.splitlines() if "[WARN] baselines:" in l)
    assert aggregate.startswith("  [WARN] baselines: 1 concern(s) — 1 marks sidecar problem(s)")
    assert "since revised" not in aggregate


def test_msci_world_sidecar_name(midas_data_root, tmp_path):
    path = tmp_path / "global" / "msci_world.json"
    _write_store("TEST", _FIRST_STORE)
    rows = compute_passive_benchmark(_SPEC, _FROM, _TO)
    _write(path, _strip_marks(rows))
    _write(path.with_name("msci_world_marks.json"), _sidecar(rows))
    assert _remerge(path, [("2026-04-16", 100.0), ("2026-04-17", 111.0)]).concern == 4


# ---------------------------------------------------------------------------
# Appends, restatement
# ---------------------------------------------------------------------------


def test_the_merge_has_no_coin_flip_kind():
    """Plan 1.6: the coin flip left this merge (it advances from a state), so
    the ``kind`` switch and its ``path_recompute`` class are gone."""
    import inspect

    assert "kind" not in inspect.signature(merge_baseline_series).parameters
    assert "path_recompute" not in {f.name for f in __import__("dataclasses").fields(MergeCounts)}


def test_appends_are_counted(midas_data_root, tmp_path):
    path = tmp_path / "benchmark.json"
    _write_store("TEST", _FIRST_STORE)
    assert merge_baseline_series(
        path, compute_passive_benchmark(_SPEC, _FROM, _TO), closes=_closes(_FIRST_STORE)
    ) == MergeCounts(appended=5)


def test_restate_overwrites_and_classifies_nothing(midas_data_root, tmp_path):
    path = tmp_path / "benchmark.json"
    _publish(path, _FIRST_STORE)
    revised = [("2026-04-16", 100.0), ("2026-04-17", 111.0)]
    assert _remerge(path, revised, restate=True) == MergeCounts()
    assert json.loads(path.read_text())[1]["mark_close"] == 111.0


def test_counts_add_up():
    a = MergeCounts(appended=1, stale_mark=2, revised=1, sidecar=2)
    b = MergeCounts(rescaled=3, unclassified=5, cash_flat=1, coinflip=4, sidecar_file=6)
    total = a + b
    assert total == MergeCounts(1, 2, 3, 1, 5, 2, 1, 4, 6)
    assert total.concern == 1 + 2 + 1 + 4 + 6
    # Regression: round-4 review, 2026-10-06. `mismatched` added `concern`,
    # so a coin-flip concern and an unreadable sidecar file (neither a
    # published row) counted as disagreeing rows: this pinned 2+3+5+8. It
    # counts rows only: stale_mark + rescaled + unclassified + revised +
    # legacy sidecar rows + cash_flat.
    assert total.mismatched == 2 + 3 + 5 + 1 + 2 + 1


# ---------------------------------------------------------------------------
# The consumer: what step 9a prints
# ---------------------------------------------------------------------------

_DAYS = ["2026-04-17", "2026-04-18", "2026-04-19", "2026-04-20", "2026-04-21"]


def _seed_desk(cfg, closes: dict[str, list[float]]) -> dict[str, list[str]]:
    ramp = [100.0, 101.0, 102.0, 103.0, 104.0]
    tickers = {
        cfg.roster[a].benchmark.ticker
        for a in cfg.trading_roster
        if cfg.roster[a].benchmark is not None
    } | {cfg.global_reference.ticker}
    tickers.discard(CASH_FLAT_TICKER)
    for t in tickers:
        _write_store(t, list(zip(_DAYS, closes.get(t, ramp))))
    _write_store("FAKE-A", list(zip(_DAYS, [10.0, 10.5, 11.0, 11.5, 12.0])))
    _write_store("FAKE-B", list(zip(_DAYS, [20.0, 20.5, 20.0, 19.5, 19.0])))
    # The EUR books' coin flips hold the bare (USD) fake tickers; a flat rate
    # keeps them drawable, so these tests stay about the benchmark merge.
    _write_store("EURUSD=X", [(d, 1.0) for d in _DAYS])
    return {a: ["FAKE-A", "FAKE-B"] for a in cfg.trading_roster}


def _build(universes):
    return build_all_baselines(
        universes_by_agent=universes,
        from_date=date(2026, 4, 17),
        to_date=date(2026, 4, 21),
    )


def test_only_concerns_reach_the_warn_path(midas_data_root, capsys):
    cfg = get_config()
    universes = _seed_desk(cfg, {})
    _build(universes)
    capsys.readouterr()

    ref = cfg.global_reference.ticker
    # Revise one recorded mark of the global reference only.
    _seed_desk(cfg, {ref: [100.0, 101.0, 102.0, 999.0, 104.0]})
    totals = _build(universes)

    out = capsys.readouterr().out
    assert totals.concern >= 1
    assert f"[WARN] baselines: {totals.concern} concern(s)" in out
    warn_lines = [ln for ln in out.splitlines() if "[WARN]" in ln]
    assert all("concern" in ln or "revised" in ln for ln in warn_lines)


def test_expected_classes_print_one_info_line_each(midas_data_root, capsys):
    cfg = get_config()
    universes = _seed_desk(cfg, {})
    _build(universes)
    capsys.readouterr()

    # x2 across every benchmark history: rescaled, never a concern.
    doubled = [200.0, 202.0, 204.0, 206.0, 208.0]
    tickers = {
        cfg.roster[a].benchmark.ticker
        for a in cfg.trading_roster
        if cfg.roster[a].benchmark is not None
    } | {cfg.global_reference.ticker}
    _seed_desk(cfg, {t: doubled for t in tickers})
    totals = _build(universes)

    out = capsys.readouterr().out
    assert totals.concern == 0
    assert totals.rescaled > 0
    assert "[WARN]" not in out
    info = [ln for ln in out.splitlines() if "[INFO] baselines:" in ln]
    assert len(info) == 1
    assert "rescaled" in info[0] and "not a concern" in info[0]


def test_the_build_reads_each_benchmark_file_once(midas_data_root, monkeypatch):
    """Cleanup 6a: the benchmark was priced from one read of its file and
    classified against a second. One read now serves both."""
    from collections import Counter

    import engine.baselines as baselines

    cfg = get_config()
    universes = _seed_desk(cfg, {})
    reads: Counter[str] = Counter()
    real = baselines._load_ohlcv

    def counting(ticker):
        reads[ticker] += 1
        return real(ticker)

    monkeypatch.setattr(baselines, "_load_ohlcv", counting)
    _build(universes)
    agents = [a for a in cfg.trading_roster if cfg.roster[a].benchmark is not None]
    per_ticker = Counter(
        cfg.roster[a].benchmark.ticker
        for a in agents
        if not cfg.roster[a].benchmark.is_cash_flat
    )
    per_ticker[cfg.global_reference.ticker] += 1
    assert per_ticker, "the fixture must price at least one benchmark"
    for ticker, series in per_ticker.items():
        assert reads[ticker] == series, ticker


_ISO = st.dates(min_value=date(2026, 1, 1), max_value=date(2026, 3, 1)).map(date.isoformat)


@given(
    store=st.sets(_ISO, max_size=40),
    mark=_ISO,
    row=_ISO,
    priced=st.booleans(),
)
def test_has_later_close_bisect_matches_the_scan(store, mark, row, priced):
    """Cleanup 6b: the bisect answers exactly what the linear scan did."""
    from engine.baselines import _has_later_close

    closes = {d: 1.0 for d in store} if priced else None
    scan = closes is not None and mark < row and any(mark < d <= row for d in closes)
    dates = sorted(closes) if closes is not None else None
    assert _has_later_close({"mark_date": mark}, row, dates) == scan


# ---------------------------------------------------------------------------
# A cash-flat series (EUR_CASH_FLAT) and a restatement read no sidecar
# ---------------------------------------------------------------------------

_CASH_FLAT = BenchmarkSpec("Cash", CASH_FLAT_TICKER, "EUR")


def test_a_cash_flat_mismatch_is_its_own_concern_not_a_sidecar_one(
    midas_data_root, tmp_path, capsys
):
    """Regression: round-3 review, 2026-10-06. EUR_CASH_FLAT records no
    marks and has no sidecar, so a mismatch became a legacy-row concern
    whose remedy (restore the sidecar) cannot be carried out. A cash-flat
    series marks no price: a mismatch means the initial capital or the
    currency changed, and the [WARN] says that."""
    path = tmp_path / "benchmark.json"
    published = compute_passive_benchmark(_CASH_FLAT, _FROM, _TO)
    _write(path, [dict(r, portfolio_value=12_000.0, cash=12_000.0) for r in published])
    path.with_name("benchmark_marks.json").write_text("{not json")  # never read

    counts = merge_baseline_series(
        path, compute_passive_benchmark(_CASH_FLAT, _FROM, _TO), cash_flat=True
    )

    assert counts == MergeCounts(cash_flat=5) and counts.concern == 5
    warns = [l for l in capsys.readouterr().out.splitlines() if "[WARN]" in l]
    assert len(warns) == 5
    assert all("initial capital or the series currency changed" in l for l in warns)
    assert not any("sidecar is" in l or "unreadable" in l for l in warns)


def test_a_cash_flat_series_that_matches_is_silent(midas_data_root, tmp_path, capsys):
    path = tmp_path / "benchmark.json"
    rows = compute_passive_benchmark(_CASH_FLAT, _FROM, _TO)
    _write(path, rows)
    assert merge_baseline_series(path, rows, cash_flat=True) == MergeCounts()
    assert "[WARN]" not in capsys.readouterr().out


def test_a_cash_flat_row_differing_only_in_other_fields_is_no_concern(
    midas_data_root, tmp_path, capsys
):
    """Regression: round-4 review, 2026-10-06. The cash-flat branch compared
    the whole row, so a field added to (or dropped from) the writer's rows
    made every published row a CASH_FLAT_MISMATCH concern, though no number
    moved. It compares value and currency, as a legacy row does."""
    path = tmp_path / "benchmark.json"
    rows = compute_passive_benchmark(_CASH_FLAT, _FROM, _TO)
    _write(path, [{k: v for k, v in r.items() if k != "positions_value"} for r in rows])
    recomputed = [dict(r, note="a field the writer added since") for r in rows]
    assert merge_baseline_series(path, recomputed, cash_flat=True) == MergeCounts()
    assert "[WARN]" not in capsys.readouterr().out


# The demo desk has no cash-flat benchmark agent; the live cast does.
@pytest.mark.live_cast
def test_the_build_names_a_cash_flat_mismatch_as_such(midas_data_root, capsys):
    cfg = get_config()
    universes = _seed_desk(cfg, {})
    _build(universes)
    agent = next(
        a for a in cfg.trading_roster
        if cfg.roster[a].benchmark is not None
        and cfg.roster[a].benchmark.is_cash_flat
    )
    path = cfg.baselines_dir / agent / "benchmark.json"
    rows = json.loads(path.read_text())
    rows[0] = dict(rows[0], portfolio_value=9_999.0)
    _write(path, rows)
    capsys.readouterr()
    totals = _build(universes)
    assert totals == MergeCounts(cash_flat=1)
    aggregate = next(
        l for l in capsys.readouterr().out.splitlines() if "[WARN] baselines:" in l
    )
    assert "1 cash-flat benchmark point(s)" in aggregate
    assert "since revised" not in aggregate and "sidecar" not in aggregate


def test_a_restatement_reads_no_sidecar(midas_data_root, tmp_path, capsys):
    """Regression: round-3 review, 2026-10-06. `restate=True` overwrites
    every row it is handed unclassified, yet still loaded the sidecar and
    printed its unreadable [WARN] (and counted it)."""
    path = tmp_path / "benchmark.json"
    _publish(path, _FIRST_STORE, legacy=True)
    path.with_name("benchmark_marks.json").write_text("{not json")
    revised = [("2026-04-16", 100.0), ("2026-04-17", 111.0)]
    assert _remerge(path, revised, restate=True) == MergeCounts()
    assert "[WARN]" not in capsys.readouterr().out


def test_the_cash_flat_flag_is_the_spec_property_not_a_literal():
    """Round-4 review, 2026-10-06: the literal was compared in four places;
    one constant and one property now."""
    assert BenchmarkSpec("Cash", CASH_FLAT_TICKER, "EUR").is_cash_flat
    assert not BenchmarkSpec("S&P", "SPY", "USD").is_cash_flat


# The demo desk has no cash-flat benchmark agent; the live cast does.
@pytest.mark.live_cast
def test_a_restatement_passes_the_cash_flat_flag(midas_data_root, monkeypatch):
    """Round-4 review, 2026-10-06: the restatement path called the merge
    without the flag, so a cash-flat series it rewrote was treated as a
    priced one. It passes the spec's own flag."""
    import engine.baselines as baselines

    cfg = get_config()
    universes = _seed_desk(cfg, {})
    _build(universes)
    agent = next(
        a for a in cfg.trading_roster
        if cfg.roster[a].benchmark is not None and cfg.roster[a].benchmark.is_cash_flat
    )
    seen: list[tuple[str, bool]] = []
    real = baselines.merge_baseline_series

    def spy(path, rows, **kw):
        seen.append((path.parent.name, kw.get("cash_flat", False)))
        return real(path, rows, **kw)

    monkeypatch.setattr(baselines, "merge_baseline_series", spy)
    monkeypatch.setattr(baselines, "require_changelog_entry", lambda *a, **k: None)
    build_all_baselines(
        universes, date.fromisoformat(_DAYS[0]), date.fromisoformat(_DAYS[-1]),
        restate_series={f"{agent}/benchmark"}, changelog_entry="x",
    )
    assert seen == [(agent, True)]
