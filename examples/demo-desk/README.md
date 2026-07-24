# Demo Desk — define your own cast

The **demo desk** is a minimal, runnable Midas configuration. It exists so you
can stand up your own trading desk from a blank sheet: everything the engine
needs is declared in a single `roster.yaml`, and the same `midas` command line
that drives the reference desk drives this one.

It is a **paper-trading** simulation. Fills are booked against a committed daily
price store; nothing touches a real broker or real money. Treat any output as a
mechanics demonstration, not investment advice or evidence of an edge.

## The cast

Four agents, all declared in [`roster.yaml`](./roster.yaml):

| id               | role        | what it is                                             |
| ---------------- | ----------- | ------------------------------------------------------ |
| `demo-momentum`  | `trader`    | Trend-follower on `SPY / QQQ / IWM`, 3 positions max.  |
| `demo-value`     | `trader`    | Patient value buyer on `VTV / SCHD / BRK-B`.           |
| `demo-oracle`    | `narrator`  | Watches the desk and narrates; does not trade.         |
| `demo-allocator` | `allocator` | Allocates capital across traders on its own channel.   |

Each agent's voice and behaviour is a persona in
[`.claude/agents/`](./.claude/agents/) — swap those files (and the roster rows)
for your own strategies and you have a new desk. Nothing else is hardcoded.

## roster.yaml schema

The engine loads this file through `engine/config.py`. Fields and their
defaults, exactly as parsed:

### `globals`

| key                | meaning                                                        |
| ------------------ | -------------------------------------------------------------- |
| `day_one`          | ISO date the experiment starts.                                |
| `currencies`       | List of accounting currencies (default `[EUR, USD]`).          |
| `initial_capital`  | Default starting cash for agents that omit their own (default `10000`). |
| `global_reference` | Desk-wide reference benchmark `{label, ticker, currency}`.     |
| `agents_dir`       | Where persona files live (default `.claude/agents`).           |
| `jurisdiction`     | Optional `{tax_rate_pct, fees}`; omit for the fee/tax defaults. |

### Per-agent fields

| key               | meaning                                                                    |
| ----------------- | -------------------------------------------------------------------------- |
| `display_name`    | Human name shown in output (required).                                     |
| `voice`           | One-line style hint for the persona (default empty).                       |
| `post_time`       | Local time the agent "posts" its note (default empty).                     |
| `home_currency`   | Base currency for its portfolio (default `EUR`).                           |
| `initial_capital` | Starting cash (default = `globals.initial_capital`).                       |
| `max_positions`   | Soft position-count target (default `5`).                                  |
| `universe`        | A registered universe **name**, a list of names, or a list of literal tickers (e.g. `[SPY, QQQ, IWM]`). |
| `benchmark`       | Per-agent `{label, ticker, currency}` (optional).                          |
| `persona`         | Persona filename (default `{id}.md`).                                       |
| `role`            | `trader` \| `narrator` \| `allocator` (default `trader`).                   |
| `safety`          | Broker-enforced rails — see below.                                         |
| `allocator`       | Allocator sub-block — only for `role: allocator`.                          |

### `safety` — enforced by the broker, not the prompt

These rails live in the **Hands** (`engine/paper_broker.py`), so they hold even
if a persona is coaxed into misbehaving. A persona is aspirational; the broker
is enforcing.

| key                      | meaning                                                                       |
| ------------------------ | ----------------------------------------------------------------------------- |
| `max_order_notional`     | Reject any order whose base-currency notional exceeds this (default `500`).    |
| `max_orders_per_day`     | Per-agent cap on same-day fills (default `5`).                                 |
| `daily_drawdown_halt_pct`| **Negative**; halt all the agent's orders once daily drawdown is worse than this (default `-5.0`, `0.0` disables). |
| `allowed_universe`       | List of universe names; **empty means allow-all** (default `[]`).             |
| `dry_run`                | If `true`, validate and report but never mutate the portfolio (default `false`). |

### `allocator` sub-block (`role: allocator` only)

| key                       | meaning                                                                     |
| ------------------------- | --------------------------------------------------------------------------- |
| `channels_prefix`         | Isolates the allocator's order channels under `data/orders/{prefix}-*` (default `manager`). |
| `outcome_resolution_days` | Days before an allocation's outcome is scored (default `10`).               |
| `outcome_memory`          | `{same_ticker_max, other_ticker_max}` — how many past outcomes to recall (defaults `5` / `3`). |
| `baseline`                | `{enabled}` — whether to track a passive baseline (default `true`).         |
| `risk_budget`             | `{max_positions, per_position_cap, cash_floor, max_trades_per_week, min_conviction}`. |
| `policy`                  | `{blocklist, prose_override}` — tickers to avoid and an optional policy prose override. |

## Walkthrough — Brain writes, Hands fills

This drives the full loop end-to-end: bootstrap prices, hand-author an order
(the **Brain**), run the broker (the **Hands**), and inspect the fills and the
mutated portfolio. It uses `demo-momentum`.

Run the commands from a Midas checkout. To get the `midas` command, install the
package once with `pip install -e .`; from a checkout without installing, every
`midas <cmd>` below is identical to `python -m engine.cli <cmd>`.

```bash
# 1. Copy the desk to a working directory and point Midas at it.
cp -R examples/demo-desk /tmp/my-desk
export MIDAS_DATA_DIR=/tmp/my-desk

# 2. One-time: open a portfolio for demo-momentum (USD 10,000).
python -c "
from engine.config import get_config
from engine.portfolio import PortfolioManager
pm = PortfolioManager(base_dir=get_config().portfolios_dir)
pm.initialize('demo-momentum', initial_capital=10000, currency='USD')
"

# 3. Bootstrap the price store from yfinance. This is the one networked step and
#    is normal for local dev; the live desk populates the store from a cron.
midas fetch-ohlcv --symbols SPY,QQQ,IWM,VTV,SCHD,BRK-B,URTH --history-days 90
```

### Hand-author an order (the Brain)

Orders are append-only JSONL, one per line, in
`$MIDAS_DATA_DIR/data/orders/outbox/<TODAY>.jsonl`, where `<TODAY>` is today's
date (`YYYY-MM-DD`) — the broker fills for `date.today()`. The `Order` contract
(`engine/orders.py`): every field below is required, including a non-empty
`reasoning` (no silent trades). `shares` must be strictly positive (Midas is
long-only). Write two orders — one that fills and one that trips a rail:

```jsonl
{"order_id": "ord_demo_001", "ts": "2026-04-17T20:00:00Z", "agent_id": "demo-momentum", "action": "BUY", "ticker": "SPY", "shares": 3, "reasoning": "Trend intact above the 50-day; adding core exposure.", "currency": "USD"}
{"order_id": "ord_demo_002", "ts": "2026-04-17T20:00:00Z", "agent_id": "demo-momentum", "action": "BUY", "ticker": "SPY", "shares": 10, "reasoning": "Deliberately oversized clip to demonstrate the notional rail.", "currency": "USD"}
```

`demo-momentum`'s `max_order_notional` is `5000`. With SPY near \$740 the first
order (~\$2,200) clears; the second (~\$7,400) breaches the cap.

### Run the broker (the Hands)

```bash
midas fill-day
# fill-day: 1 filled, 1 rejected out of 2
```

Inspect the inbox — the durable Hands-side record of both outcomes:

```bash
cat $MIDAS_DATA_DIR/data/orders/inbox/<TODAY>.jsonl
```

```jsonl
{"order_id": "ord_demo_001", "ts_filled": "...Z", "status": "filled", "fill_price": 738.18, "fill_currency": "USD", "notional_base": 2214.54, "fees": 1.25, "reason": null}
{"order_id": "ord_demo_002", "ts_filled": "...Z", "status": "rejected", "fill_price": null, "fill_currency": null, "notional_base": null, "fees": null, "reason": "MAX_ORDER_NOTIONAL"}
```

The filled order mutated the portfolio — cash debited by notional + fee, a SPY
position opened:

```bash
cat $MIDAS_DATA_DIR/data/portfolios/demo-momentum/portfolio.json
```

```json
{
  "cash": 7784.21,
  "currency": "USD",
  "last_updated": "...",
  "positions": [
    { "ticker": "SPY", "shares": 3.0, "avg_cost": 738.18, "date_opened": "...", "grid_level": 0 }
  ]
}
```

`MAX_ORDER_NOTIONAL` is one of the broker's reason codes; the full set is
documented at the top of `engine/paper_broker.py`. Exact prices depend on the
day you fetch — the shape (one fill, one rejection, a debited portfolio) is what
matters.

> Note: fills are stamped with `executed_sha` (the git HEAD they ran against)
> when `MIDAS_DATA_DIR` is a git repo; in a throwaway `/tmp` directory the field
> is simply omitted.
