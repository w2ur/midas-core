"""Per-agent benchmark + coin-flip phantom competitors.

Data model: each baseline is a list of daily snapshots
{date, portfolio_value, cash, positions_value, currency} mirroring
the shape of data/portfolios/<agent>/snapshots.json so the site can
consume baselines with minimal new code.

Ticker choices:
- VGK  (Vanguard FTSE Europe ETF, USD-listed) replaces IMEU.L / IWDA.L UCITS
  variants which are not reliably available via yfinance. VGK tracks FTSE
  Developed Europe, consistent with engine/market_data.py conventions.
- URTH (iShares MSCI World ETF, USD-listed) replaces IWDA.L for world / global
  reference. URTH is the same proxy already used for msci_world in
  engine/market_data.py BENCHMARK_TICKERS.

Currency is the DISPLAY currency for the series (matches the agent's home
currency). The price ratio used to compute daily value is currency-invariant,
so the ETF's actual trading currency (USD for VGK/URTH) is not relevant to
the comparison. FX-noise over the short observation window is accepted as
de minimis, matching the existing snapshot-benchmark pattern in the site.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd

from engine.config import BenchmarkSpec, get_config


def _initial() -> float:
    """Return the initial capital from config."""
    return get_config().initial_capital


def _daterange(start: date, end: date) -> Iterator[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _load_ohlcv(ticker: str) -> dict[str, float]:
    """Return date_iso -> close for the ticker, empty if file missing."""
    path = get_config().ohlcv_dir / f"{ticker}.jsonl"
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[row["date"]] = float(row.get("adj_close") or row["close"])
    return out


def compute_passive_benchmark(
    spec: BenchmarkSpec,
    from_date: date,
    to_date: date,
) -> list[dict]:
    """€10k (or $10k) buy-and-hold of spec.ticker from from_date to to_date inclusive.

    Non-trading days carry the last observed close. Missing OHLCV data returns
    an empty list (caller treats as "no line to draw").
    """
    initial = _initial()
    if spec.ticker == "EUR_CASH_FLAT":
        return [
            {
                "date": d.isoformat(),
                "portfolio_value": initial,
                "cash": initial,
                "positions_value": 0.0,
                "currency": spec.currency,
            }
            for d in _daterange(from_date, to_date)
        ]

    closes = _load_ohlcv(spec.ticker)
    if not closes:
        return []

    first_close: float | None = None
    last_close: float | None = None
    out: list[dict] = []
    for d in _daterange(from_date, to_date):
        iso = d.isoformat()
        if iso in closes:
            last_close = closes[iso]
            if first_close is None:
                first_close = last_close
        if first_close is None or last_close is None:
            continue  # no data yet for the range
        value = initial * (last_close / first_close)
        out.append(
            {
                "date": iso,
                "portfolio_value": value,
                "cash": 0.0,
                "positions_value": value,
                "currency": spec.currency,
            }
        )
    return out


def _load_price_frame(
    tickers: list[str], from_date: date, to_date: date
) -> pd.DataFrame:
    """Build a DataFrame of daily closes over the date range for the given tickers.

    Missing rows are forward-filled; tickers with no file at all are dropped.
    """
    series_by_ticker: dict[str, pd.Series] = {}
    for t in tickers:
        closes = _load_ohlcv(t)
        if not closes:
            continue
        s = pd.Series({pd.Timestamp(d): v for d, v in closes.items()}).sort_index()
        series_by_ticker[t] = s
    if not series_by_ticker:
        return pd.DataFrame()
    df = pd.DataFrame(series_by_ticker)
    idx = pd.date_range(from_date, to_date, freq="D")
    return df.reindex(idx).ffill()


def compute_coin_flip(
    agent_id: str,
    tickers: list[str],
    currency: str,
    max_positions: int,
    from_date: date,
    to_date: date,
) -> list[dict]:
    """Random-trader-in-same-universe, €10k or $10k start, deterministic per agent.

    Builds the bt pipeline directly to avoid build_bt_strategy's StatTotalReturn
    + SelectN insertion, which would override the seeded picks with return-rank
    ordering. The seeded selector already caps picks at max_positions so no
    SelectN step is needed; LimitWeights stays as a safety valve for days when
    the available universe (after dropna) is smaller than max_positions, which
    would otherwise let WeighEqually allocate >1/max_positions to a single name.
    """
    import bt as _bt

    from engine.selectors.random_seeded import SelectRandomlySeeded, make_seed

    price_data = _load_price_frame(tickers, from_date, to_date)
    if price_data.empty:
        return []

    seed = make_seed(agent_id, from_date.isoformat())
    strategy_id = f"coinflip-{agent_id}"
    max_weight = 1.0 / max(max_positions, 1)

    pipeline = [
        _bt.algos.RunDaily(),
        SelectRandomlySeeded(n=max_positions, seed=seed),
        _bt.algos.WeighEqually(),
        _bt.algos.LimitWeights(max_weight),
        _bt.algos.Rebalance(),
    ]
    strategy = _bt.Strategy(strategy_id, pipeline)
    backtest = _bt.Backtest(strategy, price_data, initial_capital=_initial())
    bt_result = _bt.run(backtest)

    daily_values: pd.Series = bt_result.backtests[strategy_id].strategy.values
    snaps = [
        {"date": idx.date().isoformat(), "portfolioValue": float(val)}
        for idx, val in daily_values.items()
    ]
    return [
        {
            "date": s["date"],
            "portfolio_value": float(s["portfolioValue"]),
            "cash": 0.0,
            "positions_value": float(s["portfolioValue"]),
            "currency": currency,
        }
        for s in snaps
        if from_date.isoformat() <= s["date"] <= to_date.isoformat()
    ]


def compute_global_reference(from_date: date, to_date: date) -> list[dict]:
    """€10k buy-and-hold of MSCI World, the site's global reference line."""
    return compute_passive_benchmark(get_config().global_reference, from_date, to_date)


def _write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def merge_baseline_series(
    path: Path, computed: list[dict], *, restate: bool = False
) -> tuple[int, int]:
    """Append-or-refuse merge of a freshly computed series onto a baseline file.

    Reaches the same outcome as ``PortfolioManager.add_snapshot`` on the other
    curve plotted on the same dossier chart — a published point is immutable
    to a later run — by a simpler mechanism than that function's, and safely
    so: ``add_snapshot`` keys replacement on ``session_date`` identity because
    a trading day's valuation can be legitimately re-run intraday; a baseline
    date has no such concept, it is a deterministic function of a
    same-day-immutable OHLCV store, so any value mismatch on an already-
    published date is itself the signal that the upstream store changed
    retroactively — refusing on any mismatch is therefore equivalent to
    ``add_snapshot``'s session-identity check, not a looser stand-in for it.
    A date not yet on disk is appended; a date already on disk is refused —
    the on-disk row is kept exactly as published, even if ``computed`` now
    disagrees with it because an OHLCV price was revised upstream.
    ``restate=True`` is the explicit, one-time escape hatch: every date in
    ``computed`` overwrites its on-disk counterpart, used only for a
    deliberate, publicly logged restatement.

    Refusals are printed, not raised — a session that dies because one
    benchmark point drifted is worse than one that surfaces it. The same
    posture covers the case where ``computed`` is empty but the file already
    holds history: within-range gaps are already forward-filled (see
    ``compute_passive_benchmark`` above), so an empty ``computed`` against an
    established baseline means a whole ticker file is missing — a persistent
    condition, not a blip — and silently freezing the file without a word
    would look identical to success. That case is a no-op merge (nothing to
    append, nothing to refuse) but still prints, unlike a brand-new agent's
    first-ever build, where an empty ``computed`` against no prior file is
    the ordinary "no data yet" case and stays silent.

    Parameters
    ----------
    path:
        Target baseline file (created if it does not exist yet).
    computed:
        Freshly computed series for the full [from_date, to_date] window.
    restate:
        When True, every date overwrites the on-disk row instead of being
        refused. Reserved for a deliberate restatement.

    Returns
    -------
    tuple[int, int]
        (appended, refused) counts.
    """
    existing: list[dict] = json.loads(path.read_text()) if path.exists() else []
    if not computed and existing:
        print(
            f"  [WARN] {path.name}: computed series is empty against "
            f"{len(existing)} published row(s) — likely a missing OHLCV "
            f"ticker file (within-range gaps are already forward-filled), "
            f"not a transient blip. Keeping the published file as-is."
        )
        return 0, 0
    by_date = {row["date"]: row for row in existing}
    appended = 0
    refused = 0
    for row in computed:
        date_key = row["date"]
        if date_key not in by_date:
            by_date[date_key] = row
            appended += 1
            continue
        if restate:
            by_date[date_key] = row
            continue
        if by_date[date_key] != row:
            refused += 1
            print(
                f"  [WARN] {path.name}: {date_key} already published — "
                f"refusing to overwrite with a revised value."
            )
    merged = [by_date[d] for d in sorted(by_date)]
    _write_json(path, merged)
    return appended, refused


def build_all_baselines(
    universes_by_agent: dict[str, list[str]],
    from_date: date,
    to_date: date,
    max_positions_by_agent: dict[str, int] | None = None,
    *,
    restate: bool = False,
) -> None:
    """Produce all per-agent baseline files + the global reference file.

    Iterates get_config().trading_roster; agents whose benchmark is None are
    skipped. Append-or-refuse: an already-published date is kept as-is
    unless ``restate=True``. Missing OHLCV data for a brand-new agent (no
    prior file) yields an empty file — "no line to draw" for the site.
    Missing OHLCV data against an *established* baseline is not empty: the
    old file is kept frozen and a [WARN] is printed by
    ``merge_baseline_series`` (see its docstring for why those two cases
    differ). Per-date refusal warnings print as they occur; this function
    also prints one aggregated (appended, refused) summary across every
    file it merges, mirroring the shape of
    ``scripts.daily_session.step_update_snapshots``'s aggregate line, so a
    session with refusals scattered across many agents surfaces one count
    instead of dozens of individual prints.
    """
    cfg = get_config()
    max_positions_by_agent = max_positions_by_agent or {}
    baselines_dir = cfg.baselines_dir
    total_appended = 0
    total_refused = 0
    for agent_id in cfg.trading_roster:
        spec = cfg.roster[agent_id].benchmark
        if spec is None:
            continue
        agent_dir = baselines_dir / agent_id
        bench = compute_passive_benchmark(spec, from_date, to_date)
        appended, refused = merge_baseline_series(
            agent_dir / "benchmark.json", bench, restate=restate
        )
        total_appended += appended
        total_refused += refused

        tickers = universes_by_agent.get(agent_id, [])
        max_pos = max_positions_by_agent.get(agent_id, 5)
        coin = compute_coin_flip(
            agent_id=agent_id,
            tickers=tickers,
            currency=spec.currency,
            max_positions=max_pos,
            from_date=from_date,
            to_date=to_date,
        )
        appended, refused = merge_baseline_series(
            agent_dir / "coinflip.json", coin, restate=restate
        )
        total_appended += appended
        total_refused += refused

    appended, refused = merge_baseline_series(
        baselines_dir / "global" / "msci_world.json",
        compute_global_reference(from_date, to_date),
        restate=restate,
    )
    total_appended += appended
    total_refused += refused

    if total_refused:
        print(
            f"  [WARN] baselines: {total_refused} published point(s) refused, "
            f"{total_appended} new point(s) appended, across the baseline "
            f"files this build touched — an OHLCV price likely changed "
            f"since it was last published; the published values were kept."
        )
