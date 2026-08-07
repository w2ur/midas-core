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

from engine.baselines import merge_baseline_series


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

    assert merge_baseline_series(path, computed) == (1, 0)
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

    appended, refused = merge_baseline_series(path, computed)

    assert (appended, refused) == (0, 1)
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

    assert merge_baseline_series(path, computed, restate=True) == (0, 0)
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

    assert merge_baseline_series(path, computed) == (1, 0)
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

    assert merge_baseline_series(path, rows) == (0, 0)
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

    assert result == (0, 0)
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

    assert result == (0, 0)
    assert "[WARN]" not in capsys.readouterr().out
    assert json.loads(path.read_text()) == []


@pytest.mark.live_cast
def test_build_all_baselines_prints_one_aggregate_summary_on_refusal(
    midas_data_root, capsys
):
    """A revised price across many baseline files must surface as one
    aggregated line, not one scattered [WARN] per file.

    The "2 published point(s) refused" figure is a live-roster-specific
    fact (the ``world`` agent's own benchmark shares the URTH ticker with
    the global reference, so one revision refuses two files) — the demo
    desk has no such agent and would refuse exactly one. Hence
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
        if bench.ticker == "EUR_CASH_FLAT":
            continue
        _seed(bench.ticker, [("2026-04-17", 100.0), ("2026-04-18", 105.0)])

    global_ref = cfg.global_reference
    assert global_ref.ticker != "EUR_CASH_FLAT", (
        "test needs a price-driven global reference to force a refusal"
    )
    _seed(global_ref.ticker, [("2026-04-17", 100.0), ("2026-04-18", 105.0)])

    universes_by_agent = {aid: ["FAKE-A", "FAKE-B"] for aid in agents_with_bench}
    _seed("FAKE-A", [("2026-04-17", 10.0), ("2026-04-18", 12.0)])
    _seed("FAKE-B", [("2026-04-17", 20.0), ("2026-04-18", 19.0)])

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
    # and the global msci_world.json share the URTH ticker and both refuse.
    assert out.count("[WARN] baselines:") == 1
    assert "2 published point(s) refused" in out


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


# bt needs a few bars before the coin flip holds anything: over a two-day
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
        if bench.ticker != "EUR_CASH_FLAT":
            _seed(bench.ticker, ramp)
    _seed(cfg.global_reference.ticker, ramp)
    _seed("FAKE-A", [10.0, 10.5, 11.0, 11.5, fake_a_last])
    _seed("FAKE-B", [20.0, 20.5, 20.0, 19.5, 19.0])
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
        and cfg.roster[aid].benchmark.ticker != "EUR_CASH_FLAT"
    ]


def test_restating_coin_flips_leaves_benchmarks_frozen(midas_data_root, capsys):
    """The exact 2026-08-07 requirement, as an executable assertion.

    Asserted on the benchmark rather than the coin flip because the benchmark
    is the series that provably moves with a revised price (the coin flip sits
    flat at initial capital over a fixture this small, so a "did it move?"
    assertion on it would be unfalsifiable). This is the direction that
    matters anyway: under the old bool, `restate=True` moved these.
    """
    from engine.config import get_config

    cfg = get_config()
    universes = _seed_desk(cfg)
    _build(cfg, universes)

    agent = _priced_agents(cfg)[0]
    bench_path = cfg.baselines_dir / agent / "benchmark.json"
    bench_before = _values(bench_path)

    _seed_desk(cfg, last_close=999.0, fake_a_last=40.0)
    _build(cfg, universes, restate_series={"coinflip"})
    capsys.readouterr()

    assert _values(bench_path) == bench_before, (
        "a coin-flip-scoped restatement moved a passive benchmark — the exact "
        "over-reach the bool API allowed on 2026-08-07"
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
