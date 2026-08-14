import math
from datetime import date, datetime, timezone

import pytest

from engine.leaderboard import (
    annualized_sharpe,
    build_current_leaderboard_artifact,
    build_leaderboard_rows,
    max_drawdown,
)


def test_annualized_sharpe_none_when_too_few_points():
    # Fewer than 3 NAV points -> fewer than 2 returns -> undefined.
    assert annualized_sharpe([10000.0]) is None
    assert annualized_sharpe([10000.0, 10100.0]) is None


def test_annualized_sharpe_none_when_zero_variance():
    # A perfectly flat book (e.g. the Manager holding 100% cash) has no return
    # dispersion -> Sharpe undefined. This is the realistic degenerate case.
    assert annualized_sharpe([2000.0, 2000.0, 2000.0, 2000.0]) is None


def test_annualized_sharpe_matches_manual_formula():
    values = [10000.0, 10100.0, 10050.0, 10200.0, 10150.0]
    returns = [b / a - 1.0 for a, b in zip(values, values[1:])]
    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    expected = mean / math.sqrt(var) * math.sqrt(252)
    assert annualized_sharpe(values) == expected


def test_annualized_sharpe_survives_zero_priced_point():
    # A zero NAV would divide-by-zero on the step *out* of it; the ``if prev``
    # guard drops that one return rather than crashing. The result is still a
    # finite number (or None), never an exception.
    result = annualized_sharpe([10000.0, 0.0, 10100.0, 10200.0, 10150.0])
    assert result is None or math.isfinite(result)


def test_max_drawdown_none_when_too_few_points():
    # A single point (or empty) has no peak-to-trough span to measure.
    assert max_drawdown([]) is None
    assert max_drawdown([10000.0]) is None


def test_max_drawdown_zero_for_monotonic_rise():
    # A book that only ever climbs has, by definition, no drawdown.
    # Unlike Sharpe, a flat/rising series is 0.0, not undefined.
    assert max_drawdown([10000.0, 10100.0, 10200.0]) == 0.0
    assert max_drawdown([2000.0, 2000.0, 2000.0]) == 0.0


def test_max_drawdown_matches_peak_to_trough():
    # Peak 12000 -> trough 9000 -> recovery. Worst decline is (9000-12000)/12000.
    values = [10000.0, 12000.0, 9000.0, 11000.0]
    assert max_drawdown(values) == (9000.0 - 12000.0) / 12000.0


def test_max_drawdown_uses_running_peak_not_global_max():
    # The trough at 8000 is measured against the *prior* peak 10000 (-20%),
    # not against the later, higher peak 15000 (which would overstate it).
    values = [10000.0, 8000.0, 15000.0, 14000.0]
    assert max_drawdown(values) == (8000.0 - 10000.0) / 10000.0


def test_max_drawdown_none_when_no_positive_peak():
    # No positive reference value to measure a decline against -> undefined,
    # never a division by zero.
    assert max_drawdown([0.0, 0.0, 0.0]) is None


def _patch_baselines_absent(monkeypatch):
    """Simulate a desk with no baseline series (fresh fork / demo desk)."""
    from engine import leaderboard as lb

    monkeypatch.setattr(lb, "_benchmark_return_pct", lambda agent_id, on: None)
    monkeypatch.setattr(lb, "_coinflip_return_pct", lambda agent_id, on: None)
    monkeypatch.setattr(lb, "_local_return_pct", lambda agent_id, summary, on: None)
    monkeypatch.setattr(lb, "_fx_translation_pp", lambda currency, on: None)


def test_build_leaderboard_rows_without_baselines_sorts_by_eur_mtm(monkeypatch):
    # Degradation contract: a desk with no baseline series (demo desk, fresh
    # fork) ranks exactly as before this field existed — raw EUR return desc.
    from engine import leaderboard as lb

    monkeypatch.setattr(
        lb,
        "portfolio_mtm_eur",
        lambda summary, on: {"a": 12000.0, "b": 9000.0, "c": 11000.0}[
            summary["agent_id"]
        ],
    )
    _patch_baselines_absent(monkeypatch)
    summaries = {
        "a": {"agent_id": "a"},
        "b": {"agent_id": "b"},
        "c": {"agent_id": "c"},
    }
    rows = build_leaderboard_rows(summaries, on=date(2026, 5, 23))
    assert [r["agent"] for r in rows] == ["a", "c", "b"]
    assert rows[0] == {
        "rank": 1,
        "agent": "a",
        "return_pct": 20.0,
        "vs_benchmark_pp": None,
        "vs_coinflip_pp": None,
    }
    assert rows[1]["rank"] == 2
    assert rows[2]["return_pct"] == -10.0


def test_build_leaderboard_rows_skips_agents_with_none_mtm(monkeypatch):
    from engine import leaderboard as lb

    monkeypatch.setattr(
        lb,
        "portfolio_mtm_eur",
        lambda summary, on: {"a": 10500.0, "b": None}[summary["agent_id"]],
    )
    _patch_baselines_absent(monkeypatch)
    summaries = {"a": {"agent_id": "a"}, "b": {"agent_id": "b"}}
    rows = build_leaderboard_rows(summaries, on=date(2026, 5, 23))
    assert [r["agent"] for r in rows] == ["a"]


def test_build_leaderboard_rows_ranks_on_vs_benchmark_not_raw_return(monkeypatch):
    # The metric change (2026-08-14): agent `a` has the higher raw EUR return
    # but trails its own benchmark; `b` beats its benchmark. `b` must lead.
    from engine import leaderboard as lb

    monkeypatch.setattr(
        lb,
        "portfolio_mtm_eur",
        lambda summary, on: {"a": 12000.0, "b": 11000.0}[summary["agent_id"]],
    )
    monkeypatch.setattr(
        lb,
        "_local_return_pct",
        lambda agent_id, summary, on: {"a": 20.0, "b": 10.0}[agent_id],
    )
    monkeypatch.setattr(
        lb,
        "_benchmark_return_pct",
        lambda agent_id, on: {"a": 25.0, "b": 5.0}[agent_id],
    )
    monkeypatch.setattr(lb, "_coinflip_return_pct", lambda agent_id, on: None)
    monkeypatch.setattr(lb, "_fx_translation_pp", lambda currency, on: None)
    summaries = {"a": {"agent_id": "a"}, "b": {"agent_id": "b"}}
    rows = build_leaderboard_rows(summaries, on=date(2026, 8, 14))
    assert [r["agent"] for r in rows] == ["b", "a"]
    assert rows[0]["rank"] == 1
    assert rows[0]["vs_benchmark_pp"] == 5.0
    assert rows[1]["vs_benchmark_pp"] == -5.0
    # Raw EUR return is retained untouched on every row.
    assert rows[1]["return_pct"] == 20.0


def test_vs_benchmark_uses_local_returns_not_eur_translation(monkeypatch):
    # The decontamination assertion. A USD book at +13.9456% local vs SPY at
    # +8.78% local is +5.1656pp — NOT the EUR-translated return minus the
    # local benchmark (16.31 - 8.78 = +7.53pp), which is what the site used
    # to display: that difference silently embeds ~2.4pp of EURUSD movement.
    from engine import leaderboard as lb

    monkeypatch.setattr(lb, "portfolio_mtm_eur", lambda summary, on: 11631.0)
    monkeypatch.setattr(lb, "mtm_base_currency", lambda summary, on: 13426.31)
    monkeypatch.setattr(
        lb, "_initial_capital_base", lambda agent_id, currency: 11783.09
    )
    monkeypatch.setattr(lb, "_benchmark_return_pct", lambda agent_id, on: 8.78)
    monkeypatch.setattr(lb, "_coinflip_return_pct", lambda agent_id, on: None)
    monkeypatch.setattr(lb, "_fx_translation_pp", lambda currency, on: None)
    summaries = {"usd-book": {"agent_id": "usd-book", "currency": "USD"}}
    rows = build_leaderboard_rows(summaries, on=date(2026, 8, 14))
    local = (13426.31 / 11783.09 - 1) * 100
    assert rows[0]["vs_benchmark_pp"] == round(local - 8.78, 4)
    assert abs(rows[0]["vs_benchmark_pp"] - 5.16) < 0.01


def test_rows_without_benchmark_rank_after_rows_with_one(monkeypatch):
    # Null-last, mirroring site board-sort: a missing baseline is an absence
    # of information and must never rank as best or worst on the metric.
    # Among the null rows, raw EUR return decides — deterministically.
    from engine import leaderboard as lb

    monkeypatch.setattr(
        lb,
        "portfolio_mtm_eur",
        lambda summary, on: {"lo": 10100.0, "hi": 13000.0, "b": 10500.0}[
            summary["agent_id"]
        ],
    )
    monkeypatch.setattr(
        lb,
        "_local_return_pct",
        lambda agent_id, summary, on: {"lo": 1.0, "hi": None, "b": 5.0}[agent_id],
    )
    monkeypatch.setattr(
        lb,
        "_benchmark_return_pct",
        lambda agent_id, on: {"lo": 0.0, "hi": None, "b": 4.5}[agent_id],
    )
    monkeypatch.setattr(lb, "_coinflip_return_pct", lambda agent_id, on: None)
    monkeypatch.setattr(lb, "_fx_translation_pp", lambda currency, on: None)
    summaries = {k: {"agent_id": k} for k in ("lo", "hi", "b")}
    rows = build_leaderboard_rows(summaries, on=date(2026, 8, 14))
    # `hi` has +30% raw return but no benchmark -> after both measured rows.
    assert [r["agent"] for r in rows] == ["lo", "b", "hi"]
    assert rows[2]["vs_benchmark_pp"] is None


def test_fx_translation_pp_present_only_on_non_eur_books(monkeypatch):
    from engine import leaderboard as lb

    monkeypatch.setattr(lb, "portfolio_mtm_eur", lambda summary, on: 10500.0)
    _patch_baselines_absent(monkeypatch)
    monkeypatch.setattr(
        lb,
        "_fx_translation_pp",
        lambda currency, on: 2.0704 if currency == "USD" else None,
    )
    summaries = {
        "eur-book": {"agent_id": "eur-book", "currency": "EUR"},
        "usd-book": {"agent_id": "usd-book", "currency": "USD"},
    }
    rows = build_leaderboard_rows(summaries, on=date(2026, 8, 14))
    by_agent = {r["agent"]: r for r in rows}
    assert "fx_translation_pp" not in by_agent["eur-book"]
    assert by_agent["usd-book"]["fx_translation_pp"] == 2.0704


def test_baseline_return_pct_reads_series_as_of_date(midas_data_root):
    import json

    from engine import leaderboard as lb
    from engine.config import get_config

    series = [
        {"date": "2026-04-17", "portfolio_value": 10000.0},
        {"date": "2026-05-01", "portfolio_value": 10500.0},
        {"date": "2026-06-01", "portfolio_value": 12000.0},
    ]
    bdir = get_config().baselines_dir / "a"
    bdir.mkdir(parents=True)
    (bdir / "benchmark.json").write_text(json.dumps(series))

    # As-of cutoff: the June row must not leak into a May valuation.
    assert lb._benchmark_return_pct("a", date(2026, 5, 23)) == 5.0
    assert lb._benchmark_return_pct("a", date(2026, 6, 2)) == 20.0
    # No series on disk -> None, never a raise (demo desk / fresh fork).
    assert lb._benchmark_return_pct("missing", date(2026, 5, 23)) is None


def test_local_return_pct_converts_inception_capital(monkeypatch, midas_data_root):
    from engine import leaderboard as lb

    # USD book: inception capital is EUR 10k converted at day one, so the
    # local return is measured off ~$11,783 — not $10,000.
    monkeypatch.setattr(lb, "mtm_base_currency", lambda summary, on: 13426.31)
    monkeypatch.setattr(
        lb, "_initial_capital_base", lambda agent_id, currency: 11783.09
    )
    got = lb._local_return_pct("usd-book", {"currency": "USD"}, date(2026, 8, 14))
    assert abs(got - 13.9456) < 0.001

    # EUR book: no conversion involved.
    monkeypatch.setattr(lb, "mtm_base_currency", lambda summary, on: 10127.0)
    monkeypatch.setattr(lb, "_initial_capital_base", lambda agent_id, currency: 10000.0)
    assert (
        abs(
            lb._local_return_pct("eur-book", {"currency": "EUR"}, date(2026, 8, 14))
            - 1.27
        )
        < 1e-9
    )


def test_fx_translation_pp_is_the_rate_move_since_day_one(monkeypatch, midas_data_root):
    from engine import leaderboard as lb

    # EUR value of 1 USD on `on` vs at day one: 1/1.1544 vs 1/1.1783
    # -> (1.1783/1.1544 - 1) * 100 = +2.0704pp leaderboard tailwind.
    def fake_to_eur(amount, from_currency, on=None):
        assert from_currency == "USD"
        rate = {date(2026, 4, 17): 1 / 1.1783}.get(on, 1 / 1.1544)
        return amount * rate

    monkeypatch.setattr(lb, "to_eur", fake_to_eur)
    got = lb._fx_translation_pp("USD", date(2026, 8, 14))
    assert abs(got - 2.0704) < 0.001
    # EUR books have no translation leg at all.
    assert lb._fx_translation_pp("EUR", date(2026, 8, 14)) is None


def test_build_current_leaderboard_artifact_shape(monkeypatch):
    from engine import leaderboard as lb

    monkeypatch.setattr(lb, "portfolio_mtm_eur", lambda summary, on: 10500.0)
    _patch_baselines_absent(monkeypatch)
    summaries = {"a": {"agent_id": "a"}}
    fixed_now = datetime(2026, 5, 23, 20, 0, 0, tzinfo=timezone.utc)
    artifact = build_current_leaderboard_artifact(
        summaries,
        on=date(2026, 5, 23),
        trigger="scheduled-weekend-refresh",
        updated_at=fixed_now,
    )
    assert artifact == {
        "updated_at": "2026-05-23T20:00:00Z",
        "trigger": "scheduled-weekend-refresh",
        "rows": [
            {
                "rank": 1,
                "agent": "a",
                "return_pct": 5.0,
                "vs_benchmark_pp": None,
                "vs_coinflip_pp": None,
            }
        ],
    }


def test_baseline_return_pct_survives_non_dict_rows(midas_data_root):
    # "Returns None — never raises": a truncated/corrupt series that parses
    # to a list of non-dicts must degrade, not crash the leaderboard write
    # in the session, the watcher, and the weekend refresh.
    import json as _json

    from engine import leaderboard as lb
    from engine.config import get_config

    bdir = get_config().baselines_dir / "a"
    bdir.mkdir(parents=True)
    (bdir / "benchmark.json").write_text(_json.dumps(["junk", 3, None]))
    assert lb._benchmark_return_pct("a", date(2026, 5, 23)) is None


@pytest.mark.live_cast
def test_initial_capital_base_uses_per_agent_spec(midas_data_root):
    # The Manager's roster entry declares initial_capital 2000; the traders
    # inherit the global 10000. A fork overriding one agent's capital must
    # not have that agent's local return measured off the global anchor.
    # live_cast: hardcodes live roster ids — on the demo desk every agent
    # inherits the global anchor and the override branch has no fixture.
    from engine import leaderboard as lb

    assert lb._initial_capital_base("the-manager", "EUR") == 2000.0
    assert lb._initial_capital_base("steady-eddie-eur", "EUR") == 10000.0
    # Unknown agent (bundle-derived summary on a fork) → global anchor.
    assert lb._initial_capital_base("no-such-agent", "EUR") == 10000.0


def test_rank_leaderboard_rows_is_shared_and_era_tolerant():
    # Public ranking helper used by restate_bundles: rows carrying
    # vs_benchmark_pp rank on it; rows without the key (pre-2026-08-14
    # bundles) fall back to raw return — missing key and null are the same.
    from engine.leaderboard import rank_leaderboard_rows

    rows = [
        {"agent": "old", "return_pct": 9.0},
        {"agent": "meas", "return_pct": 1.0, "vs_benchmark_pp": 2.0},
    ]
    ranked = rank_leaderboard_rows(rows)
    assert [r["agent"] for r in ranked] == ["meas", "old"]
    assert ranked[0]["rank"] == 1 and ranked[1]["rank"] == 2
