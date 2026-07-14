"""Lock the midas-live roster shape so future edits can't silently drift it."""

import pytest

from engine.config import get_config, reset_config_cache

pytestmark = pytest.mark.live_cast


def test_ten_traders_plus_oracle(monkeypatch):
    monkeypatch.delenv("MIDAS_DATA_DIR", raising=False)
    reset_config_cache()
    cfg = get_config()
    assert len(cfg.trading_roster) == 10
    assert "the-oracle" in cfg.roster
    assert cfg.roster["the-oracle"].role == "narrator"
    # spot-check a known agent survived the migration
    assert cfg.roster["satoshi"].benchmark.ticker == "BTC-EUR"
    reset_config_cache()
