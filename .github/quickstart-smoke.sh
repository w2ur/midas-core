#!/usr/bin/env bash
# The README quickstart ("reach a real fill"), run against an installed wheel.
#
# The test job installs the checkout editable, so a module missing from the
# wheel, a package-data glob that drops a file, or a runtime dependency that
# only requirements.txt provides all pass there and break `pip install
# midas-core`. This drives the documented reader path from a directory outside
# the checkout, so nothing can be imported from the source tree.
#
# Usage: quickstart-smoke.sh <venv holding only the wheel>
# Keep the orders and the expected summary in step with README.md.
set -euo pipefail

venv=$(cd "$1" && pwd)
work=$(mktemp -d)
cd "$work"

"$venv/bin/midas" init-demo ./my-desk
export MIDAS_DATA_DIR="$work/my-desk"

"$venv/bin/python" -c "
from engine.config import get_config
from engine.portfolio import PortfolioManager
pm = PortfolioManager(base_dir=get_config().portfolios_dir)
pm.initialize('demo-momentum', initial_capital=10000, currency='USD')
"

"$venv/bin/midas" fetch-ohlcv --symbols SPY,QQQ,IWM,VTV,SCHD,BRK-B,URTH --history-days 90

today=$(date -u +%F)
outbox="$MIDAS_DATA_DIR/data/orders/outbox"
mkdir -p "$outbox"
cat > "$outbox/$today.jsonl" <<JSONL
{"order_id": "ord_demo_001", "ts": "${today}T20:00:00Z", "agent_id": "demo-momentum", "action": "BUY", "ticker": "SPY", "shares": 3, "reasoning": "Trend intact above the 50-day; adding core exposure.", "currency": "USD"}
{"order_id": "ord_demo_002", "ts": "${today}T20:00:00Z", "agent_id": "demo-momentum", "action": "BUY", "ticker": "SPY", "shares": 10, "reasoning": "Deliberately oversized clip to demonstrate the notional rail.", "currency": "USD"}
JSONL

out=$("$venv/bin/midas" fill-day)
printf '%s\n' "$out"
grep -qxF "fill-day: 1 filled, 1 rejected out of 2" <<< "$out" \
  || { echo "::error::fill-day did not report one fill and one rejection"; exit 1; }
grep -qF '"reason": "MAX_ORDER_NOTIONAL"' "$MIDAS_DATA_DIR/data/orders/inbox/$today.jsonl" \
  || { echo "::error::the oversized order was not rejected on MAX_ORDER_NOTIONAL"; exit 1; }
echo "quickstart smoke: ok"
