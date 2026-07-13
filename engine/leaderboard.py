"""Leaderboard builders.

Pure functions extracted from scripts/daily_session.step_build_leaderboard
so the same logic powers the weekday session, the weekend refresh cron,
and the in-watcher live update — all anchored to the €10,000 inception
baseline.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from engine.valuation import portfolio_mtm_eur

# Trading days per year — annualization factor for the daily Sharpe ratio,
# matching the ``daily_sharpe`` convention bt/ffn use on the backtest path
# (``engine.backtest``), so live and backtest Sharpe values are comparable.
_TRADING_DAYS = 252


def annualized_sharpe(values: list[float], risk_free: float = 0.0) -> float | None:
    """Annualized Sharpe ratio of a NAV series (EUR risk-free ~0).

    Computes period-over-period returns from ``values`` (a portfolio-value
    series, oldest first), then ``mean / stdev * sqrt(252)``. Returns ``None``
    when there are fewer than two returns or the returns have zero variance —
    cases where a Sharpe ratio is undefined rather than zero.
    """
    if len(values) < 3:
        return None

    returns: list[float] = []
    for prev, curr in zip(values, values[1:]):
        if prev:
            returns.append(curr / prev - 1.0)

    n = len(returns)
    if n < 2:
        return None

    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)  # sample stdev
    if variance <= 0:
        return None

    return (mean - risk_free) / (variance**0.5) * (_TRADING_DAYS**0.5)


def max_drawdown(values: list[float]) -> float | None:
    """Maximum peak-to-trough decline of a NAV series, as a negative fraction.

    Walks ``values`` (a portfolio-value series, oldest first) tracking the
    running peak, and returns the most negative ``(value - peak) / peak`` ratio
    seen — measured against the peak *prior* to each trough, never the global
    maximum. A book that only ever climbs (or is flat) returns ``0.0``: a real,
    well-defined "no drawdown", unlike Sharpe which is undefined on a flat book.

    Returns ``None`` when there are fewer than two points, or when the running
    peak is never positive (no valid reference to measure a decline against —
    avoids dividing by zero).
    """
    if len(values) < 2:
        return None

    peak = values[0]
    worst = 0.0
    seen_positive_peak = False
    for v in values:
        if v > peak:
            peak = v
        if peak > 0:
            seen_positive_peak = True
            drawdown = (v - peak) / peak
            if drawdown < worst:
                worst = drawdown

    return worst if seen_positive_peak else None


def build_leaderboard_rows(
    portfolio_summaries: dict[str, dict],
    on: date | None,
) -> list[dict]:
    """Return ranked rows: [{rank, agent, return_pct}, ...] sorted desc.

    Agents whose EUR-MTM cannot be computed (e.g. missing FX rate) are dropped.
    """
    rows: list[dict] = []
    for agent_id, summary in portfolio_summaries.items():
        eur_mtm = portfolio_mtm_eur(summary, on)
        if eur_mtm is None:
            continue
        rows.append(
            {
                "agent": agent_id,
                "return_pct": round((eur_mtm / 10_000 - 1) * 100, 4),
            }
        )
    rows.sort(key=lambda r: r["return_pct"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def build_current_leaderboard_artifact(
    portfolio_summaries: dict[str, dict],
    *,
    on: date | None,
    trigger: str,
    updated_at: datetime | None = None,
) -> dict:
    """Build the current leaderboard artifact payload."""
    ts = updated_at or datetime.now(timezone.utc)
    iso = (
        ts.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    return {
        "updated_at": iso,
        "trigger": trigger,
        "rows": build_leaderboard_rows(portfolio_summaries, on=on),
    }
