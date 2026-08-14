"""Leaderboard builders.

Pure functions extracted from scripts/daily_session.step_build_leaderboard
so the same logic powers the weekday session, the weekend refresh cron,
and the in-watcher live update — all anchored to the €10,000 inception
baseline.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

from engine.config import get_config
from engine.fx import convert as fx_convert
from engine.fx import to_eur
from engine.valuation import mtm_base_currency, portfolio_mtm_eur

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


def _baseline_return_pct(agent_id: str, filename: str, on: date | None) -> float | None:
    """Return of an agent's baseline series (benchmark/coin flip) as of `on`.

    The series lives in ``data/baselines/<agent>/<filename>`` in the book's
    own currency, indexed at the initial capital. Measured from the series'
    first row to the last row dated on-or-before ``on`` (the whole series
    when ``on`` is None). Returns None — never raises — when the series is
    missing, empty, unreadable, or starts before ``on``: a fresh fork and the
    demo desk have no baselines, and the leaderboard must still rank.
    """
    path = get_config().baselines_dir / agent_id / filename
    try:
        series = json.loads(path.read_text())
        if not isinstance(series, list) or not series:
            return None
        rows = series
        if on is not None:
            iso = on.isoformat()
            rows = [r for r in series if str(r.get("date", "")) <= iso]
            if not rows:
                return None
        first = float(series[0]["portfolio_value"])
        last = float(rows[-1]["portfolio_value"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # AttributeError: a corrupt series can parse to a list of non-dicts,
        # and `.get` on those must degrade like every other malformed shape.
        return None
    if first == 0:
        return None
    return (last - first) / first * 100.0


def _benchmark_return_pct(agent_id: str, on: date | None) -> float | None:
    return _baseline_return_pct(agent_id, "benchmark.json", on)


def _coinflip_return_pct(agent_id: str, on: date | None) -> float | None:
    return _baseline_return_pct(agent_id, "coinflip.json", on)


def _initial_capital_base(agent_id: str, currency: str | None) -> float | None:
    """The EUR inception anchor expressed in the book's own currency.

    The anchor is the agent's own ``initial_capital`` from ``roster.yaml``
    (global fallback when the agent is not in the roster — e.g. a
    bundle-derived summary on a fork); non-EUR books received it converted
    at day one, so their local return is measured off that converted base —
    not off a flat 10,000.
    """
    cfg = get_config()
    spec = cfg.roster.get(agent_id)
    initial = float(spec.initial_capital if spec else cfg.initial_capital)
    if currency in (None, "EUR"):
        return initial
    return fx_convert(initial, "EUR", currency, cfg.day_one)


def _local_return_pct(agent_id: str, summary: dict, on: date | None) -> float | None:
    """Book-currency return since inception, in percent. None if unvaluable."""
    mtm = mtm_base_currency(summary, on)
    if mtm is None:
        return None
    # Same missing-currency default as portfolio_mtm_eur — the two halves of
    # a row must never disagree about which currency the book is in.
    initial = _initial_capital_base(agent_id, summary.get("currency", "USD"))
    if initial is None or initial <= 0:
        return None
    return (mtm / initial - 1.0) * 100.0


def _fx_translation_pp(currency: str | None, on: date | None) -> float | None:
    """Leaderboard tailwind/headwind from the book currency's move vs EUR.

    The EUR value of one unit of the book's currency on ``on`` versus at day
    one — the return a *flat* non-EUR book would show on the EUR-normalised
    leaderboard. None for EUR books (no translation leg) and when either
    rate is unavailable.
    """
    if currency in (None, "EUR"):
        return None
    now_eur = to_eur(1.0, currency, on)
    day_one_eur = to_eur(1.0, currency, get_config().day_one)
    if now_eur is None or day_one_eur is None or day_one_eur == 0:
        return None
    return (now_eur / day_one_eur - 1.0) * 100.0


def build_leaderboard_rows(
    portfolio_summaries: dict[str, dict],
    on: date | None,
) -> list[dict]:
    """Return ranked rows sorted by benchmark-relative return, best first.

    Row shape: ``{rank, agent, currency, return_pct, return_local_pct,
    vs_benchmark_pp, vs_coinflip_pp}`` plus ``fx_translation_pp`` on non-EUR
    books.

    ``currency`` is the book's denomination — the unit ``return_local_pct``
    is measured in, published so a consumer can label that figure without
    re-deriving the book's currency itself. The site's roster answers
    ``"mixed"`` for ``world`` (a property of its holdings, not of its book),
    so this is the only place the question is answered once.

    ``return_local_pct`` is the book-currency return the two ``vs_*`` fields
    are actually subtractions on, published (2026-08-14) so a reader can
    reconstruct the board's arithmetic rather than take it on trust. It
    completes an exact identity with the other two published legs::

        (1 + return_pct) = (1 + return_local_pct) x (1 + fx_translation_pp)

    — so a USD book's EUR return decomposes into what it earned and what the
    euro did. Emitted on EUR books too, where it equals ``return_pct`` by
    construction unless the agent's ``initial_capital`` differs from the
    global anchor; a row must not have to be inspected for its currency
    before its own fields can be read.

    ``return_pct`` stays the EUR-normalised return off the €10,000 anchor —
    unchanged since inception. Ranking moved to ``vs_benchmark_pp``
    (2026-08-14): the book's own-currency return minus its pre-registered
    benchmark's return, which is FX-free by construction — raw EUR ranking
    mixed market beta and currency translation into a skill ordering (the
    steady-eddie twins' 15pp gap was ~5pp market + ~2.4pp FX). Rows without
    a benchmark series rank after measured rows, by raw EUR return — a
    missing baseline must never read as best or worst (same null-last rule
    as the site's board-sort). A desk with no baselines at all (demo desk,
    fresh fork) therefore ranks exactly as before this field existed.

    Agents whose EUR-MTM cannot be computed (e.g. missing FX rate) are dropped.
    """
    rows: list[dict] = []
    # The translation leg is a property of (currency, on), not of the row —
    # memoised so the watcher's 15-minute builds don't re-derive the same
    # rate pair once per non-EUR book.
    fx_pp_by_currency: dict[str, float | None] = {}
    for agent_id, summary in portfolio_summaries.items():
        eur_mtm = portfolio_mtm_eur(summary, on)
        if eur_mtm is None:
            continue
        # Same missing-currency default as portfolio_mtm_eur (see
        # _local_return_pct) — one currency assumption per row.
        currency = summary.get("currency", "USD")
        local = _local_return_pct(agent_id, summary, on)
        bench = _benchmark_return_pct(agent_id, on)
        coin = _coinflip_return_pct(agent_id, on)
        # The EUR anchor is the agent's OWN initial capital, not a literal
        # 10,000. Both denominators have to name the same inception stake or
        # the identity below is false: `return_local_pct` divides by the
        # agent's `initial_capital` (converted at day one), so a hard-coded
        # 10,000 here made the two legs reconcile only because every book on
        # this desk happens to start at exactly 10,000. A fork running
        # `initial_capital: 5000` published a -45% return on a book that was
        # up 10% — wrong on its own terms, before any identity was claimed.
        # Zero effect on this desk: all ten ranked books are at 10,000, and
        # the 2,000 Manager is not on the ranked board.
        eur_anchor = _initial_capital_base(agent_id, "EUR")
        if not eur_anchor:
            continue
        row = {
            "agent": agent_id,
            "currency": currency,
            "return_pct": round((eur_mtm / eur_anchor - 1) * 100, 4),
            "return_local_pct": round(local, 4) if local is not None else None,
            "vs_benchmark_pp": (
                round(local - bench, 4)
                if local is not None and bench is not None
                else None
            ),
            "vs_coinflip_pp": (
                round(local - coin, 4)
                if local is not None and coin is not None
                else None
            ),
        }
        if currency not in fx_pp_by_currency:
            fx_pp_by_currency[currency] = _fx_translation_pp(currency, on)
        fx_pp = fx_pp_by_currency[currency]
        if fx_pp is not None:
            row["fx_translation_pp"] = round(fx_pp, 4)
        rows.append(row)

    return rank_leaderboard_rows(rows)


def rank_leaderboard_rows(rows: list[dict]) -> list[dict]:
    """Sort rows on the ranking metric and assign ``rank`` in place.

    The single definition of the board's order, shared with
    ``scripts.restate_bundles`` so a restated bundle cannot drift onto a
    different metric than the live builder. A row without a
    ``vs_benchmark_pp`` key (pre-2026-08-14 bundles) ranks exactly like one
    where it is null: after every measured row, by raw EUR return.
    """

    def _sort_key(r: dict) -> tuple:
        vs = r.get("vs_benchmark_pp")
        if vs is not None:
            return (0, -vs, r["agent"])
        return (1, -r["return_pct"], r["agent"])

    rows.sort(key=_sort_key)
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
