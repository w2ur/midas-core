"""Append-or-refuse contract for baseline files.

Background: `build_all_baselines` used to full-rewrite every baseline file
from `cfg.day_one` on every session, while `PortfolioManager.add_snapshot`
refuses to let a later session replace an already-published row. A revised
OHLCV price therefore silently moved the benchmark curve while the agent
curve stayed frozen — both plotted on the same dossier chart.

`merge_baseline_series` closes that gap: a published date is kept unless
`restate=True` is passed explicitly (the one-time, owner-approved
restatement escape hatch). New dates are always appended.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.baselines import MergeCounts, merge_baseline_series


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


def test_merge_baseline_series_appends_new_dates(tmp_path):
    path = tmp_path / "benchmark.json"
    _write(
        path,
        [
            {
                "date": "2026-08-04",
                "portfolio_value": 100.0,
                "cash": 0.0,
                "positions_value": 100.0,
                "currency": "EUR",
            }
        ],
    )
    computed = [
        {
            "date": "2026-08-04",
            "portfolio_value": 100.0,
            "cash": 0.0,
            "positions_value": 100.0,
            "currency": "EUR",
        },
        {
            "date": "2026-08-05",
            "portfolio_value": 101.0,
            "cash": 0.0,
            "positions_value": 101.0,
            "currency": "EUR",
        },
    ]

    assert merge_baseline_series(path, computed) == MergeCounts(appended=1)
    on_disk = json.loads(path.read_text())
    assert [row["date"] for row in on_disk] == ["2026-08-04", "2026-08-05"]


def test_merge_baseline_series_refuses_to_move_a_published_point(tmp_path):
    """Regression: pre-fix, build_all_baselines full-rewrote from day one every
    session while snapshots were append-or-refuse, so a revised price silently
    moved the benchmark curve under a frozen agent curve."""
    path = tmp_path / "benchmark.json"
    _write(
        path,
        [
            {
                "date": "2026-08-04",
                "portfolio_value": 8695.39,
                "cash": 0.0,
                "positions_value": 8695.39,
                "currency": "EUR",
            }
        ],
    )
    computed = [
        {
            "date": "2026-08-04",
            "portfolio_value": 8679.04,
            "cash": 0.0,
            "positions_value": 8679.04,
            "currency": "EUR",
        }
    ]

    counts = merge_baseline_series(path, computed)

    # A legacy row with no recorded marks and no sidecar beside it: kept, and
    # a concern of the sidecar cause, since nothing can classify it
    # (classification lives in tests/test_baselines_marks.py).
    assert counts == MergeCounts(sidecar=1) and counts.concern == 1
    assert json.loads(path.read_text())[0]["portfolio_value"] == 8695.39


def test_merge_baseline_series_restate_flag_overwrites(tmp_path):
    """The one-time restatement path — used deliberately, logged publicly."""
    path = tmp_path / "benchmark.json"
    _write(
        path,
        [
            {
                "date": "2026-08-04",
                "portfolio_value": 8695.39,
                "cash": 0.0,
                "positions_value": 8695.39,
                "currency": "EUR",
            }
        ],
    )
    computed = [
        {
            "date": "2026-08-04",
            "portfolio_value": 8679.04,
            "cash": 0.0,
            "positions_value": 8679.04,
            "currency": "EUR",
        }
    ]

    assert merge_baseline_series(path, computed, restate=True) == MergeCounts()
    assert json.loads(path.read_text())[0]["portfolio_value"] == 8679.04


def test_merge_baseline_series_creates_file_when_none_exists(tmp_path):
    """First-ever build for a fresh agent dir: no prior file to refuse against."""
    path = tmp_path / "benchmark.json"
    computed = [
        {
            "date": "2026-08-04",
            "portfolio_value": 100.0,
            "cash": 0.0,
            "positions_value": 100.0,
            "currency": "EUR",
        }
    ]

    assert merge_baseline_series(path, computed) == MergeCounts(appended=1)
    assert json.loads(path.read_text()) == computed


def test_merge_baseline_series_identical_replay_is_not_a_refusal(tmp_path, capsys):
    """Re-running the same session with unchanged prices must not warn."""
    path = tmp_path / "benchmark.json"
    rows = [
        {
            "date": "2026-08-04",
            "portfolio_value": 100.0,
            "cash": 0.0,
            "positions_value": 100.0,
            "currency": "EUR",
        }
    ]
    _write(path, rows)

    assert merge_baseline_series(path, rows) == MergeCounts()
    assert "[WARN]" not in capsys.readouterr().out


def test_merge_baseline_series_warns_on_total_fetch_failure_against_history(
    tmp_path, capsys
):
    """An empty computed series against an established baseline is a whole
    missing ticker file (within-range gaps are already forward-filled), not
    a transient blip — it must be loud, not a silent freeze."""
    path = tmp_path / "benchmark.json"
    _write(
        path,
        [
            {
                "date": "2026-08-04",
                "portfolio_value": 100.0,
                "cash": 0.0,
                "positions_value": 100.0,
                "currency": "EUR",
            }
        ],
    )

    result = merge_baseline_series(path, [])

    assert result == MergeCounts()
    out = capsys.readouterr().out
    assert "[WARN]" in out
    assert "benchmark.json" in out
    # The published file must survive completely untouched.
    on_disk = json.loads(path.read_text())
    assert on_disk == [
        {
            "date": "2026-08-04",
            "portfolio_value": 100.0,
            "cash": 0.0,
            "positions_value": 100.0,
            "currency": "EUR",
        }
    ]


def test_merge_baseline_series_silent_for_brand_new_agent_with_no_data(
    tmp_path, capsys
):
    """A brand-new agent with no prior file and no OHLCV data yet is the
    ordinary 'no line to draw' case — it must stay silent, not warn."""
    path = tmp_path / "benchmark.json"

    result = merge_baseline_series(path, [])

    assert result == MergeCounts()
    assert "[WARN]" not in capsys.readouterr().out
    assert json.loads(path.read_text()) == []


@pytest.mark.live_cast
def test_build_all_baselines_prints_one_aggregate_summary_on_concern(
    midas_data_root, capsys
):
    """A revised price across many baseline files must surface as one
    aggregated line, not one scattered [WARN] per file.

    The "2 concern(s)" figure is a live-roster-specific
    fact (the ``world`` agent's own benchmark shares the URTH ticker with
    the global reference, so one revision flags two files) — the demo
    desk has no such agent and would flag exactly one. Hence
    ``live_cast``, matching the convention in ``tests/conftest.py``.
    """
    from datetime import date as _date

    from engine.baselines import build_all_baselines
    from engine.config import get_config

    cfg = get_config()
    ohlcv = cfg.ohlcv_dir
    ohlcv.mkdir(parents=True, exist_ok=True)

    def _seed(ticker: str, rows: list[tuple[str, float]]) -> None:
        lines = [f'{{"date":"{d}","close":{c}}}' for d, c in rows]
        (ohlcv / f"{ticker}.jsonl").write_text("\n".join(lines) + "\n")

    agents_with_bench = {
        aid: cfg.roster[aid].benchmark
        for aid in cfg.trading_roster
        if cfg.roster[aid].benchmark is not None
    }
    for bench in agents_with_bench.values():
        if bench.is_cash_flat:
            continue
        _seed(bench.ticker, [("2026-04-17", 100.0), ("2026-04-18", 105.0)])

    global_ref = cfg.global_reference
    assert not global_ref.is_cash_flat, (
        "test needs a price-driven global reference to force a refusal"
    )
    _seed(global_ref.ticker, [("2026-04-17", 100.0), ("2026-04-18", 105.0)])

    universes_by_agent = {aid: ["FAKE-A", "FAKE-B"] for aid in agents_with_bench}
    _seed("FAKE-A", [("2026-04-17", 10.0), ("2026-04-18", 12.0)])
    _seed("FAKE-B", [("2026-04-17", 20.0), ("2026-04-18", 19.0)])
    # The EUR books' coin flips hold the bare (USD) fake tickers: a flat rate
    # keeps them drawable, so this test is about the benchmark revision only.
    _seed("EURUSD=X", [("2026-04-17", 1.0), ("2026-04-18", 1.0)])

    build_all_baselines(
        universes_by_agent=universes_by_agent,
        from_date=_date(2026, 4, 17),
        to_date=_date(2026, 4, 18),
    )
    capsys.readouterr()  # discard the first (all-append) build's output

    # A price revision on an already-published date: same date, new close.
    _seed(global_ref.ticker, [("2026-04-17", 100.0), ("2026-04-18", 999.0)])

    build_all_baselines(
        universes_by_agent=universes_by_agent,
        from_date=_date(2026, 4, 17),
        to_date=_date(2026, 4, 18),
    )

    out = capsys.readouterr().out
    # One aggregate line, even though the world agent's own benchmark.json
    # and the global msci_world.json share the URTH ticker and both flag it.
    assert out.count("[WARN] baselines:") == 1
    assert "[WARN] baselines: 2 concern(s) — 2 benchmark point(s) priced from" in out


# ---------------------------------------------------------------------------
# Scoped restatement (reliability review W4.3)
#
# `build_all_baselines` used to take a bool, which could only say "restate
# everything". On 2026-08-07 the coin flips genuinely needed restating onto
# normalised units and the passive benchmarks did not; the blanket flag moved
# eight benchmarks anyway — on fresher *prices*, not on units — and they had
# to be restored by hand. An API that cannot express the intended scope will
# eventually be used outside it.
# ---------------------------------------------------------------------------


# The coin flip needs a few bars before it moves: over a two-day
# window every value comes out flat at initial capital, and a "did it move?"
# assertion would then be unfalsifiable in both directions.
_DAYS = ["2026-04-17", "2026-04-18", "2026-04-19", "2026-04-20", "2026-04-21"]


def _seed_desk(cfg, last_close: float = 105.0, fake_a_last: float = 12.0) -> dict[str, list[str]]:
    """Seed enough OHLCV for every benchmark, coin flip and the global ref."""
    ohlcv = cfg.ohlcv_dir
    ohlcv.mkdir(parents=True, exist_ok=True)

    def _seed(ticker: str, closes: list[float]) -> None:
        lines = [f'{{"date":"{d}","close":{c}}}' for d, c in zip(_DAYS, closes)]
        (ohlcv / f"{ticker}.jsonl").write_text("\n".join(lines) + "\n")

    ramp = [100.0, 101.0, 102.0, 103.0, last_close]
    agents = {
        aid: cfg.roster[aid].benchmark
        for aid in cfg.trading_roster
        if cfg.roster[aid].benchmark is not None
    }
    for bench in agents.values():
        if not bench.is_cash_flat:
            _seed(bench.ticker, ramp)
    _seed(cfg.global_reference.ticker, ramp)
    _seed("FAKE-A", [10.0, 10.5, 11.0, 11.5, fake_a_last])
    _seed("FAKE-B", [20.0, 20.5, 20.0, 19.5, 19.0])
    _seed("EURUSD=X", [1.0] * len(_DAYS))  # the EUR books' coin flips hold USD names
    return {aid: ["FAKE-A", "FAKE-B"] for aid in agents}


def _build(cfg, universes, restate_series=None):
    from datetime import date as _date

    from engine.baselines import build_all_baselines

    build_all_baselines(
        universes_by_agent=universes,
        from_date=_date(2026, 4, 17),
        to_date=_date(2026, 4, 21),
        restate_series=restate_series,
    )


def _values(path):
    import json

    return {row["date"]: row["portfolio_value"] for row in json.loads(path.read_text())}


def _priced_agents(cfg):
    return [
        aid
        for aid in cfg.trading_roster
        if cfg.roster[aid].benchmark is not None
        and not cfg.roster[aid].benchmark.is_cash_flat
    ]


def test_restating_the_global_reference_leaves_benchmarks_frozen(midas_data_root, capsys):
    """The exact 2026-08-07 requirement, as an executable assertion: a scope
    moves only what it names. Under the old bool, `restate=True` moved these.

    The 2026-08-07 case itself scoped the coin flips, and a coin-flip scope is
    now refused outright (plan 1.6, `tests/test_coinflip_state.py`), so the
    narrowest series left to scope is the global reference.
    """
    from engine.config import get_config

    cfg = get_config()
    universes = _seed_desk(cfg)
    _build(cfg, universes)

    agent = _priced_agents(cfg)[0]
    bench_path = cfg.baselines_dir / agent / "benchmark.json"
    bench_before = _values(bench_path)
    global_path = cfg.baselines_dir / "global" / "msci_world.json"
    global_before = _values(global_path)

    _seed_desk(cfg, last_close=999.0, fake_a_last=40.0)
    _build(cfg, universes, restate_series={"global/msci_world"})
    capsys.readouterr()

    assert _values(global_path) != global_before, "the scope restated nothing"
    assert _values(bench_path) == bench_before, (
        "a global-reference-scoped restatement moved a passive benchmark — the "
        "exact over-reach the bool API allowed on 2026-08-07"
    )


def test_the_scope_does_restate_what_it_names(midas_data_root, capsys):
    """The other half: without this, a scope matching nothing would pass above."""
    from engine.config import get_config

    cfg = get_config()
    universes = _seed_desk(cfg)
    _build(cfg, universes)

    agent = _priced_agents(cfg)[0]
    bench_path = cfg.baselines_dir / agent / "benchmark.json"
    bench_before = _values(bench_path)

    _seed_desk(cfg, last_close=999.0)
    _build(cfg, universes, restate_series={"benchmark"})
    capsys.readouterr()

    assert _values(bench_path) != bench_before


def test_a_fully_qualified_series_restates_only_that_agent(midas_data_root, capsys):
    """`<agent>/<kind>` narrows to one file; other agents stay frozen."""
    from engine.config import get_config

    cfg = get_config()
    universes = _seed_desk(cfg)
    _build(cfg, universes)

    priced = _priced_agents(cfg)
    target, bystander = priced[0], priced[1]
    target_path = cfg.baselines_dir / target / "benchmark.json"
    bystander_path = cfg.baselines_dir / bystander / "benchmark.json"
    target_before = _values(target_path)
    bystander_before = _values(bystander_path)

    _seed_desk(cfg, last_close=999.0)
    _build(cfg, universes, restate_series={f"{target}/benchmark"})
    capsys.readouterr()

    assert _values(target_path) != target_before
    assert _values(bystander_path) == bystander_before


def test_no_scope_means_nothing_restates(midas_data_root, capsys):
    """Default is append-or-refuse — the same posture as passing nothing."""
    from engine.config import get_config

    cfg = get_config()
    universes = _seed_desk(cfg)
    _build(cfg, universes)

    agent = _priced_agents(cfg)[0]
    bench_path = cfg.baselines_dir / agent / "benchmark.json"
    before = _values(bench_path)

    _seed_desk(cfg, last_close=999.0)
    _build(cfg, universes)
    capsys.readouterr()

    assert _values(bench_path) == before


# ---------------------------------------------------------------------------
# Date-scoped restatement (plan 1.5, owner decision 2026-10-05)
#
# 11 published benchmark rows were priced from provisional vendor bars that
# were later revised. Restating their whole series would also move the ~400
# rows that merely forward-filled a close which landed later (`stale_mark`,
# right for what they saw), so the scope has to name a single date:
# `"<agent>/<kind>@<YYYY-MM-DD>"`.
# ---------------------------------------------------------------------------

_ANCHOR = "dated-scope-test"


def _disclose(root) -> None:
    (root / "METHODOLOGY.md").write_text(
        f'- <a id="{_ANCHOR}"></a>**A dated restatement.**\n', encoding="utf-8"
    )


def _build_dated(cfg, universes, scope, anchor=_ANCHOR):
    from datetime import date as _date

    from engine.baselines import build_all_baselines

    return build_all_baselines(
        universes_by_agent=universes,
        from_date=_date(2026, 4, 17),
        to_date=_date(2026, 4, 21),
        restate_series=scope,
        changelog_entry=anchor,
    )


def test_a_dated_scope_overwrites_only_that_date_of_that_series(midas_data_root, capsys):
    from engine.config import get_config

    cfg = get_config()
    _disclose(midas_data_root)
    universes = _seed_desk(cfg)
    _build(cfg, universes)

    priced = _priced_agents(cfg)
    target, bystander = priced[0], priced[1]
    target_path = cfg.baselines_dir / target / "benchmark.json"
    bystander_path = cfg.baselines_dir / bystander / "benchmark.json"
    before = _values(target_path)
    bystander_before = _values(bystander_path)

    # Every close from 04-19 on is revised, so 04-19, 04-20 and 04-21 all
    # disagree with what is published.
    _seed_desk(cfg, last_close=999.0)
    import json

    ramp_changed = cfg.ohlcv_dir / f"{cfg.roster[target].benchmark.ticker}.jsonl"
    lines = [
        '{"date":"%s","close":%s}' % (d, c)
        for d, c in zip(_DAYS, [100.0, 101.0, 150.0, 160.0, 170.0])
    ]
    ramp_changed.write_text("\n".join(lines) + "\n")

    _build_dated(cfg, universes, {f"{target}/benchmark@2026-04-19"})
    capsys.readouterr()

    after = _values(target_path)
    assert after["2026-04-19"] != before["2026-04-19"], "the dated scope restated nothing"
    # Fail-once control: the neighbours disagree with a recomputation too, so
    # "unchanged" below is the scope holding them, not an absence of drift.
    assert after["2026-04-20"] == before["2026-04-20"]
    assert after["2026-04-21"] == before["2026-04-21"]
    assert after["2026-04-17"] == before["2026-04-17"]
    assert _values(bystander_path) == bystander_before
    # The restated row carries its new marks like any new row.
    row = {r["date"]: r for r in json.loads(target_path.read_text())}["2026-04-19"]
    assert {"mark_date", "mark_close", "base_date", "base_close"} <= set(row)
    assert row["mark_close"] == 150.0


def test_a_dated_scope_needs_a_changelog_entry(midas_data_root):
    from engine.config import get_config
    from engine.disclosure import UndisclosedRestatementError

    cfg = get_config()
    _disclose(midas_data_root)  # a fork without METHODOLOGY.md is exempt
    universes = _seed_desk(cfg)
    agent = _priced_agents(cfg)[0]
    with pytest.raises(UndisclosedRestatementError):
        _build_dated(cfg, universes, {f"{agent}/benchmark@2026-04-19"}, anchor=None)


@pytest.mark.parametrize(
    "scope, message",
    [
        ("{agent}/benchmark@2026-4-19", "date"),
        ("{agent}/benchmark@not-a-date", "date"),
        ("{agent}/benchmark@", "date"),
        ("{agent}/mystery@2026-04-19", "kind"),
        ("{agent}/coinflip@2026-04-19", "coin flip"),
        ("nobody/benchmark@2026-04-19", "agent"),
        ("benchmark@2026-04-19", "<agent>/<kind>@<YYYY-MM-DD>"),
        ("{agent}/benchmark@2026-06-01", "not a published date"),
    ],
)
def test_a_malformed_dated_scope_is_refused_before_anything_is_written(
    midas_data_root, scope, message
):
    from engine.config import get_config

    cfg = get_config()
    _disclose(midas_data_root)
    universes = _seed_desk(cfg)
    _build(cfg, universes)
    agent = _priced_agents(cfg)[0]
    path = cfg.baselines_dir / agent / "benchmark.json"
    snapshot = path.read_text()

    with pytest.raises(ValueError, match=message):
        _build_dated(cfg, universes, {scope.format(agent=agent)})
    assert path.read_text() == snapshot


# ---------------------------------------------------------------------------
# A restatement is restate-only (whole-branch review I1, M4)
#
# A scoped restatement used to run the whole routine build around its scope:
# every benchmark gained its new dates and every coin flip advanced, so a call
# meant to move one disclosed row also published days of unrelated rows under
# a `[restate]` commit. And a dated scope was checked against the computed
# series only when its own file came up, after earlier files had been written.
# ---------------------------------------------------------------------------


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under ``root``, by relative path."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _build_to(cfg, universes, to_day, scope=None, anchor=_ANCHOR):
    from datetime import date as _date

    from engine.baselines import build_all_baselines

    return build_all_baselines(
        universes_by_agent=universes,
        from_date=_date(2026, 4, 17),
        to_date=_date.fromisoformat(to_day),
        restate_series=scope,
        changelog_entry=anchor if scope else None,
    )


@pytest.mark.parametrize(
    "scope",
    ["{agent}/benchmark@2026-04-19", "{agent}/benchmark", "benchmark", "global/msci_world"],
)
def test_a_restatement_writes_only_its_scoped_rows(midas_data_root, capsys, scope):
    """No new date is appended to any series and no coin flip advances: the
    rows a restatement writes are exactly the published rows it names."""
    from engine.config import get_config

    cfg = get_config()
    _disclose(midas_data_root)
    universes = _seed_desk(cfg)
    _build_to(cfg, universes, "2026-04-19")
    agent = _priced_agents(cfg)[0]
    before = _tree(cfg.baselines_dir)
    coin = {k: v for k, v in before.items() if "coinflip" in k}
    assert coin, "the routine build wrote no coin flip — the check below is vacuous"

    _seed_desk(cfg, last_close=999.0, fake_a_last=40.0)
    _build_to(cfg, universes, "2026-04-21", scope={scope.format(agent=agent)})
    capsys.readouterr()

    after = _tree(cfg.baselines_dir)
    assert set(after) == set(before), "a restatement created or removed a file"
    for name in before:
        if name.endswith(".json") and "coinflip" not in name and "_marks" not in name:
            dates_before = [r["date"] for r in json.loads(before[name])]
            dates_after = [r["date"] for r in json.loads(after[name])]
            assert dates_after == dates_before, f"{name}: a restatement appended a date"
    assert {k: after[k] for k in coin} == coin, "a restatement advanced a coin flip"


def test_a_routine_build_over_the_same_window_does_append(midas_data_root, capsys):
    """Fail-once control for the test above: the same call without a scope
    appends 04-20 and 04-21 and advances every coin flip."""
    from engine.config import get_config

    cfg = get_config()
    universes = _seed_desk(cfg)
    _build_to(cfg, universes, "2026-04-19")
    before = _tree(cfg.baselines_dir)
    _build_to(cfg, universes, "2026-04-21")
    capsys.readouterr()
    after = _tree(cfg.baselines_dir)
    assert any(after[k] != v for k, v in before.items() if "coinflip" in k)
    agent = _priced_agents(cfg)[0]
    rows = json.loads(after[f"{agent}/benchmark.json"])
    assert rows[-1]["date"] == "2026-04-21"


@pytest.mark.parametrize(
    "bad, message",
    [
        ("global/msci_world@2026-06-01", "not a published date"),
        ("{agent}/benchmark@2026-04-20", "not a published date"),
        ("{other}/benchmark@2026-04-16", "not a published date"),
        ("nobody/benchmark", "agent"),
        ("{agent}/mystery", "kind"),
        ("mystery", "scope"),
    ],
)
def test_one_bad_entry_in_a_scope_writes_nothing_at_all(midas_data_root, capsys, bad, message):
    """Every entry is validated before the first write: a valid entry whose
    file comes up first is not restated because a later entry is bad."""
    from engine.config import get_config

    cfg = get_config()
    _disclose(midas_data_root)
    universes = _seed_desk(cfg)
    _build_to(cfg, universes, "2026-04-19")
    priced = _priced_agents(cfg)
    agent, other = priced[0], priced[-1]
    before = _tree(cfg.baselines_dir)

    _seed_desk(cfg, last_close=999.0, fake_a_last=40.0)
    ticker = cfg.ohlcv_dir / f"{cfg.roster[agent].benchmark.ticker}.jsonl"
    ticker.write_text(
        "\n".join(
            '{"date":"%s","close":%s}' % (d, c)
            for d, c in zip(_DAYS, [100.0, 101.0, 150.0, 160.0, 170.0])
        )
        + "\n"
    )
    scope = {f"{agent}/benchmark@2026-04-19", bad.format(agent=agent, other=other)}
    with pytest.raises(ValueError, match=message):
        _build_to(cfg, universes, "2026-04-21", scope=scope)
    assert _tree(cfg.baselines_dir) == before


def _strip_benchmark(root: Path, agent: str) -> None:
    import yaml

    from engine.config import reset_config_cache

    roster = yaml.safe_load((root / "roster.yaml").read_text())
    del roster["agents"][agent]["benchmark"]
    (root / "roster.yaml").write_text(yaml.safe_dump(roster, sort_keys=False))
    reset_config_cache()


@pytest.mark.parametrize("scope", ["{agent}/benchmark@2026-04-19", "{agent}/benchmark"])
def test_a_scope_naming_an_agent_with_no_benchmark_is_refused(midas_data_root, scope):
    """Review M4: such an agent has no benchmark series, so the scope would
    restate nothing — and silently, since the build skips the agent."""
    from engine.config import get_config

    cfg = get_config()
    _disclose(midas_data_root)
    universes = _seed_desk(cfg)
    _build_to(cfg, universes, "2026-04-19")
    agent = _priced_agents(cfg)[0]
    _strip_benchmark(midas_data_root, agent)
    cfg = get_config()
    assert agent in cfg.trading_roster and cfg.roster[agent].benchmark is None
    before = _tree(cfg.baselines_dir)

    with pytest.raises(ValueError, match="no benchmark"):
        _build_to(cfg, universes, "2026-04-21", scope={scope.format(agent=agent)})
    assert _tree(cfg.baselines_dir) == before
