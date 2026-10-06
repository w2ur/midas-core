"""The plan 1.6 migration: each coin flip's first state, from its last published row.

The state date is the last published row's date and the first repick runs at
that date's close, sized from the published value (no flat cash day). The
script refuses to overwrite an existing state unless given ``--force``, and
exits 2 when it cannot run.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from engine.baselines import build_all_baselines, coin_flip_state_path, load_coin_flip_state
from engine.config import get_config

_DAYS = [(date(2026, 4, 17) + timedelta(days=i)).isoformat() for i in range(8)]


def _store(ticker: str, closes: list[float]) -> None:
    ohlcv = get_config().ohlcv_dir
    ohlcv.mkdir(parents=True, exist_ok=True)
    (ohlcv / f"{ticker}.jsonl").write_text(
        "\n".join(json.dumps({"date": d, "close": c}) for d, c in zip(_DAYS, closes)) + "\n"
    )


def _agents(cfg) -> list[str]:
    return [a for a in cfg.trading_roster if cfg.roster[a].benchmark is not None]


@pytest.fixture
def desk(midas_data_root, monkeypatch):
    """Every agent has a published, pre-1.6 coin flip (no state) over 04-17..04-20."""
    import scripts.backfill_baselines as backfill

    cfg = get_config()
    for t, base in (("AAA", 10.0), ("BBB", 20.0), ("CCC", 7.0)):
        _store(t, [base * (1 + 0.01 * i) for i in range(len(_DAYS))])
    _store("EURUSD=X", [1.0] * len(_DAYS))  # the EUR books hold bare (USD) tickers
    universes = {a: ["AAA", "BBB", "CCC"] for a in _agents(cfg)}
    monkeypatch.setattr(backfill, "_universes_by_agent", lambda: universes)
    monkeypatch.setattr(backfill, "_max_positions_by_agent", lambda: {a: 2 for a in universes})
    for i, a in enumerate(_agents(cfg)):
        path = cfg.baselines_dir / a / "coinflip.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"date": d, "portfolio_value": 10_000.0 + 13.37 * k + i, "cash": 0.0,
             "positions_value": 10_000.0 + 13.37 * k + i, "currency": "EUR"}
            for k, d in enumerate(_DAYS[:4])
        ]
        path.write_text(json.dumps(rows, indent=2) + "\n")
    return cfg, universes


def _main(*argv: str) -> int:
    from scripts.init_coinflip_state import main

    return main(list(argv))


def test_the_state_starts_on_the_last_published_row_already_invested(desk):
    cfg, _ = desk
    assert _main() == 0
    for a in _agents(cfg):
        series = cfg.baselines_dir / a / "coinflip.json"
        last = json.loads(series.read_text())[-1]
        state = load_coin_flip_state(coin_flip_state_path(series))
        assert state.date == last["date"] == _DAYS[3]
        assert state.portfolio_value == last["portfolio_value"]
        assert len(state.holdings) == 2, "the first repick runs on the state date itself"
        assert all(h.mark_date == _DAYS[3] for h in state.holdings.values())
        assert state.cash < 0.5 * state.portfolio_value


def test_the_next_build_advances_from_the_migrated_state_with_no_concern(desk, capsys):
    cfg, universes = desk
    published = {
        a: (cfg.baselines_dir / a / "coinflip.json").read_text() for a in _agents(cfg)
    }
    assert _main() == 0
    capsys.readouterr()
    build_all_baselines(universes, date(2026, 4, 17), date(2026, 4, 23))
    out = capsys.readouterr().out
    assert not [l for l in out.splitlines() if "[WARN]" in l and "coinflip" in l]
    for a, before in published.items():
        rows = json.loads((cfg.baselines_dir / a / "coinflip.json").read_text())
        assert rows[:4] == json.loads(before)
        assert [r["date"] for r in rows[4:]] == _DAYS[4:7]


def test_without_the_migration_the_build_refuses_to_advance(desk, capsys):
    """The control for the test above: a pre-1.6 series with no state is a concern."""
    cfg, universes = desk
    build_all_baselines(universes, date(2026, 4, 17), date(2026, 4, 23))
    warns = [l for l in capsys.readouterr().out.splitlines() if "[WARN]" in l and "coinflip" in l]
    assert len(warns) == len(_agents(cfg))


def test_an_existing_state_is_not_overwritten_without_force(desk, capsys):
    cfg, _ = desk
    assert _main() == 0
    a = _agents(cfg)[0]
    state_path = coin_flip_state_path(cfg.baselines_dir / a / "coinflip.json")
    state_path.write_text('{"sentinel": true}')
    assert _main() == 2
    assert json.loads(state_path.read_text()) == {"sentinel": True}
    assert _main("--force") == 0
    assert load_coin_flip_state(state_path).date == _DAYS[3]


def test_a_dry_run_writes_nothing(desk):
    cfg, _ = desk
    assert _main("--dry-run") == 0
    assert not list(cfg.baselines_dir.glob("*/state/coinflip.json"))


@pytest.mark.parametrize("content", [None, "[]", "{broken"])
def test_an_agent_without_a_readable_published_row_cannot_run(desk, content):
    cfg, _ = desk
    path = cfg.baselines_dir / _agents(cfg)[0] / "coinflip.json"
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    assert _main() == 2
    assert not list(cfg.baselines_dir.glob("*/state/coinflip.json"))
