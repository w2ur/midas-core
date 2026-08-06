"""Daily log generator — produces a human-readable markdown report of each trading session."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from engine.config import get_config
from engine.fx import to_eur
from engine.posts import display_name as _display_name
from engine.valuation import portfolio_mtm, portfolio_mtm_eur

_CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£", "JPY": "¥", "CHF": "CHF "}


def _fmt(amount: float, currency: str) -> str:
    sym = _CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    return f"{sym}{amount:,.2f}"


def _fmt_with_eur(amount: float, currency: str, on: date) -> str:
    """Format an amount in its native currency, adding the EUR equivalent if not already EUR."""
    native = _fmt(amount, currency)
    if currency == "EUR":
        return native
    eur_val = to_eur(amount, currency, on)
    if eur_val is None:
        return f"{native} (EUR rate unavailable)"
    return f"{native} ≈ {_fmt(eur_val, 'EUR')}"


def generate_daily_log(
    log_date: date,
    market_data: dict,
    agent_results: dict[str, dict],
    portfolio_summaries: dict[str, dict],
) -> Path:
    """Generate a markdown daily log and save to data/logs/YYYY-MM-DD.md.

    Parameters
    ----------
    log_date
        The trading date.
    market_data
        Benchmark values: {"sp500": ..., "gold": ..., "btc": ..., "msci_world": ...}
    agent_results
        Per-agent output: {"steady-eddie": {"commentary": "...", "trades": [...]}, ...}
    portfolio_summaries
        Per-agent portfolio state: {"steady-eddie": {"cash": ..., "deployed": ..., "positions": [...]}, ...}

    Returns
    -------
    Path to the generated log file.
    """
    logs_dir = get_config().logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{log_date.isoformat()}.md"

    lines: list[str] = []
    lines.append(f"# Midas Daily Log — {log_date.isoformat()}\n")

    # Market overview
    lines.append("## Market Snapshot\n")
    lines.append("| Index | Value |")
    lines.append("|-------|-------|")
    for name, value in market_data.items():
        if isinstance(value, (int, float)):
            lines.append(f"| {name} | {value:,.2f} |")
    lines.append("")

    # Agent sections
    for agent_id, result in agent_results.items():
        display_name = _display_name(agent_id)
        lines.append(f"## {display_name}\n")

        # Commentary
        commentary = result.get("commentary", "No commentary provided.")
        lines.append(f"> {commentary}\n")

        # Trades
        trades = result.get("trades", [])
        if trades:
            lines.append("### Trades\n")
            lines.append("| Action | Ticker | Shares | Reasoning |")
            lines.append("|--------|--------|--------|-----------|")
            for t in trades:
                action = t.get("action", "?")
                ticker = t.get("ticker", "?")
                shares = t.get("shares", "?")
                reasoning = t.get("reasoning", "")
                lines.append(f"| {action} | {ticker} | {shares} | {reasoning} |")
            lines.append("")
        else:
            lines.append("*No trades today.*\n")

        # Portfolio state
        summary = portfolio_summaries.get(agent_id)
        if summary:
            currency = summary.get(
                "currency", "USD"
            )  # default USD for legacy portfolios
            cash = summary.get("cash", 0)
            deployed = summary.get("deployed", 0)
            lines.append("### Portfolio\n")
            lines.append(f"- **Cash:** {_fmt_with_eur(cash, currency, log_date)}")
            lines.append(
                f"- **Deployed (cost basis):** {_fmt_with_eur(deployed, currency, log_date)}"
            )
            positions = summary.get("positions", [])
            if positions:
                # Positions may be a list of ticker strings OR dicts {ticker, shares}.
                # The leaderboard (below) needs dicts to value MTM; the display here
                # only cares about tickers. Accept both shapes defensively.
                tickers = [p["ticker"] if isinstance(p, dict) else p for p in positions]
                lines.append(f"- **Positions ({len(tickers)}):** {', '.join(tickers)}")
            lines.append("")

    # Leaderboard — MTM valuation in EUR for cross-agent comparison
    lines.append("## Leaderboard (mark-to-market, EUR-equivalent)\n")
    lines.append(
        "> Positions priced at latest close from the committed OHLCV store, "
        "converted to EUR at today's FX. All agents started at €10,000 equivalent.\n"
    )
    lines.append("| Rank | Agent | Native MTM | ≈ EUR | vs €10k |")
    lines.append("|------|-------|------------|-------|---------|")
    rows = []
    for agent_id, summary in portfolio_summaries.items():
        currency = summary.get("currency", "USD")
        native_mtm = portfolio_mtm(summary, log_date)
        eur_mtm = portfolio_mtm_eur(summary, log_date)
        rows.append((agent_id, currency, native_mtm, eur_mtm))
    rows.sort(key=lambda r: r[3] if r[3] is not None else -1, reverse=True)
    for rank, (agent_id, currency, native_mtm, eur_mtm) in enumerate(rows, start=1):
        display = _display_name(agent_id)
        # native_mtm can be None too now (portfolio_mtm returns None when a
        # held position's currency can't be converted into the book's own —
        # see engine.valuation.portfolio_mtm). eur_mtm is then also None,
        # since portfolio_mtm_eur derives from portfolio_mtm.
        native_str = (
            _fmt(native_mtm, currency)
            if native_mtm is not None
            else "— (rate unavailable)"
        )
        eur_str = (
            _fmt(eur_mtm, "EUR") if eur_mtm is not None else "— (rate unavailable)"
        )
        if eur_mtm is None:
            pnl_str = "—"
        else:
            pnl_pct = (eur_mtm / 10_000 - 1) * 100
            pnl_str = f"{pnl_pct:+.2f}%"
        lines.append(f"| {rank} | {display} | {native_str} | {eur_str} | {pnl_str} |")
    lines.append("")

    # Footer
    lines.append("---\n")
    lines.append("*Generated by Midas — AI Fund Manager*\n")

    content = "\n".join(lines)
    path.write_text(content)
    return path
