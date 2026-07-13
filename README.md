# midas-core

The engine behind **Midas** — an open, multi-agent paper-trading fund-manager framework
built on a Brain/Hands split (agents author decisions; a broker enforces safety rails and
executes). This repo is the reusable core: the engine, the config-driven roster/allocator
model, a runnable demo desk, and the deterministic Hands pipeline. Bring your own agents.

> Paper-trading by default. This is a framework, not a service: it never executes against
> your broker keys, never pools funds, never gives per-user advice.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pinned lockfile (incl. pytest + hypothesis)
pip install -e .                  # the midas-core package + `midas` CLI

# Point the engine at the bundled demo desk and run its test suite
export MIDAS_DATA_DIR="$PWD/examples/demo-desk"
pytest -q
```

`requirements.txt` is the fully-pinned lockfile every consumer installs (same
convention as the upstream live run) — it reproduces the exact tested
environment. `pyproject.toml`'s looser `dependencies` are for `pip install`
resolution only.

`examples/demo-desk/` is a minimal two-trader + allocator + oracle desk driven entirely by
`roster.yaml`. Copy it, edit the roster and the persona files under `.claude/agents/`, and
point `MIDAS_DATA_DIR` at your copy to run your own desk.

The deterministic pipeline (baselines, broker fills, valuations) is headless and fully
tested. The full LLM trading session (persona dispatch, journals, posts) runs under
Claude Code — see `scripts/daily_session.py`.

## Layout

- `engine/` — types, market data, bt adapter, paper broker, config (`roster.yaml` loader), universes.
- `scripts/` — session orchestration (`daily_session.py`), the conditional-order watcher
  (`check_triggers.py`), backtest runners, universe refresh.
- `examples/demo-desk/` — the forkable reference desk.
- `data/{strategies,universes}/` — generic strategy specs + index constituents (refresh with
  `scripts/refresh_universes.py`).

Apache-2.0 (LICENSE added at public release).
