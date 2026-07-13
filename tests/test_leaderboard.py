import math
from datetime import date, datetime, timezone

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


def test_build_leaderboard_rows_sorts_by_eur_mtm_descending(monkeypatch):
    from engine import leaderboard as lb

    monkeypatch.setattr(
        lb,
        "portfolio_mtm_eur",
        lambda summary, on: {"a": 12000.0, "b": 9000.0, "c": 11000.0}[
            summary["agent_id"]
        ],
    )
    summaries = {
        "a": {"agent_id": "a"},
        "b": {"agent_id": "b"},
        "c": {"agent_id": "c"},
    }
    rows = build_leaderboard_rows(summaries, on=date(2026, 5, 23))
    assert [r["agent"] for r in rows] == ["a", "c", "b"]
    assert rows[0] == {"rank": 1, "agent": "a", "return_pct": 20.0}
    assert rows[1]["rank"] == 2
    assert rows[2]["return_pct"] == -10.0


def test_build_leaderboard_rows_skips_agents_with_none_mtm(monkeypatch):
    from engine import leaderboard as lb

    monkeypatch.setattr(
        lb,
        "portfolio_mtm_eur",
        lambda summary, on: {"a": 10500.0, "b": None}[summary["agent_id"]],
    )
    summaries = {"a": {"agent_id": "a"}, "b": {"agent_id": "b"}}
    rows = build_leaderboard_rows(summaries, on=date(2026, 5, 23))
    assert [r["agent"] for r in rows] == ["a"]


def test_build_current_leaderboard_artifact_shape(monkeypatch):
    from engine import leaderboard as lb

    monkeypatch.setattr(lb, "portfolio_mtm_eur", lambda summary, on: 10500.0)
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
        "rows": [{"rank": 1, "agent": "a", "return_pct": 5.0}],
    }
