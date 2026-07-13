from __future__ import annotations
import pytest
from engine.config import get_config, reset_config_cache


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.delenv("MIDAS_DATA_DIR", raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


def test_default_prefix_matches_legacy():
    import engine.orders as o
    import engine.triggers as t

    base = get_config().orders_dir
    assert o.allocator_channel_dir("manager", "outbox") == base / "manager-outbox"
    assert o.allocator_channel_dir("manager", "outbox") == o.MANAGER_OUTBOX_DIR
    assert o.allocator_channel_dir("manager", "inbox") == o.MANAGER_INBOX_DIR
    assert o.allocator_channel_dir("manager", "review") == o.MANAGER_REVIEW_DIR
    assert t.allocator_channel_dir("manager", "pending") == t.MANAGER_PENDING_DIR
    assert t.allocator_channel_dir("manager", "cancels") == t.MANAGER_CANCELS_DIR


def test_second_prefix_is_isolated():
    import engine.orders as o

    base = get_config().orders_dir
    assert o.allocator_channel_dir("crypto", "outbox") == base / "crypto-outbox"
    assert o.allocator_channel_dir("crypto", "outbox") != o.allocator_channel_dir(
        "manager", "outbox"
    )
