"""Live kill switch — global gate for real-money execution.

is_live_enabled() is the first check any real-money Hands worker must call.
Fail-safe principle: any unexpected error returns False (default OFF).
It is impossible for this function to error its way into returning True.

Config file: data/agent_config/live_switch.json
  {"live_enabled": false}  ← shipped default (always OFF)

Env override (wins over file):
  MIDAS_LIVE=0|false  → False
  MIDAS_LIVE=1|true   → True
  any other value     → fall back to file
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from engine.config import get_config

_ENV_FALSE = {"0", "false"}
_ENV_TRUE = {"1", "true"}


def is_live_enabled(*, path: Path | None = None) -> bool:
    """Return True only when live trading is explicitly enabled.

    Parameters
    ----------
    path:
        Path to the live_switch.json config file. Defaults to the production
        location. Override in tests via the ``path`` keyword argument.

    Returns
    -------
    bool
        True only when both the config and the env agree live is enabled.
        Defaults to False on any error.
    """
    try:
        # Note: env can flip the committed default-OFF to ON; the broker worker's environment is the trust boundary.
        env_val = os.environ.get("MIDAS_LIVE", "").strip().lower()
        if env_val in _ENV_FALSE:
            return False
        if env_val in _ENV_TRUE:
            return True

        config_path = (
            path
            if path is not None
            else get_config().agent_config_dir / "live_switch.json"
        )
        if not config_path.is_file():
            return False

        data = json.loads(config_path.read_text(encoding="utf-8"))
        return bool(data.get("live_enabled", False))
    except Exception:
        return False
