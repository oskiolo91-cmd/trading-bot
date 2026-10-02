"""Atomic JSON persistence for per-symbol live-trading settings."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from threading import RLock

LOG = logging.getLogger(__name__)
STATE_PATH = Path(os.environ.get("BOT_STATE_PATH", Path(__file__).resolve().parent / "bot_state.json"))
_STATE_LOCK = RLock()


def load_bot_state(path: str | Path = STATE_PATH) -> dict:
    """Load persisted state; report malformed files and return an empty recoverable state."""
    state_path = Path(path)
    with _STATE_LOCK:
        try:
            with state_path.open(encoding="utf-8") as state_file:
                state = json.load(state_file)
        except FileNotFoundError:
            return {"symbols": {}}
        except (OSError, json.JSONDecodeError) as exc:
            LOG.error("Cannot read bot state from %s: %s; using empty state", state_path, exc)
            return {"symbols": {}}

    if not isinstance(state, dict) or not isinstance(state.get("symbols", {}), dict):
        LOG.error("Invalid bot state structure in %s; using empty state", state_path)
        return {"symbols": {}}
    return {"symbols": state.get("symbols", {})}


def get_symbol_state(symbol: str, path: str | Path = STATE_PATH) -> dict:
    """Return a copy of one symbol's persisted settings."""
    record = load_bot_state(path)["symbols"].get(symbol, {})
    return dict(record) if isinstance(record, dict) else {}


def update_symbol_state(
    symbol: str,
    updates: dict,
    path: str | Path = STATE_PATH,
) -> None:
    """Atomically merge and persist validated JSON-compatible values for one symbol."""
    if not symbol:
        raise ValueError("symbol must not be empty")
    clean_updates = {}
    for key, value in updates.items():
        if value is None:
            clean_updates[key] = None
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            clean_updates[key] = value
        elif isinstance(value, (str, int, bool)):
            clean_updates[key] = value
        elif isinstance(value, (dict, list)):
            json.dumps(value, allow_nan=False)
            clean_updates[key] = value
        else:
            raise TypeError(f"{key} is not a supported bot-state value")

    state_path = Path(path)
    with _STATE_LOCK:
        state = load_bot_state(state_path)
        record = state["symbols"].setdefault(symbol, {})
        if not isinstance(record, dict):
            record = {}
            state["symbols"][symbol] = record
        record.update(clean_updates)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=state_path.parent,
                prefix=f".{state_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as state_file:
                temporary_path = Path(state_file.name)
                json.dump(state, state_file, indent=2, sort_keys=True, allow_nan=False)
                state_file.write("\n")
                state_file.flush()
                os.fsync(state_file.fileno())
            os.replace(temporary_path, state_path)
        except OSError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            LOG.exception("Cannot persist bot state to %s", state_path)
            raise
