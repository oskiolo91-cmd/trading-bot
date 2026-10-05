"""SQLite persistence for per-symbol live-trading settings."""

from __future__ import annotations

import logging
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path

LOG = logging.getLogger(__name__)
STATE_PATH = Path(os.environ.get("BOT_STATE_PATH", Path(__file__).resolve().parent / "bot_state.db"))

_COLUMNS = ("profilo_rischio", "custom_trailing_pct", "high_water_mark", "last_buy_date")
_ALIASES = {"profile": "profilo_rischio", **{column: column for column in _COLUMNS}}
_PROFILE_TRAILING = {
    "🐢 Conservativo": 0.03,
    "⚖️ Bilanciato": 0.06,
    "🚀 Speculativo": 0.12,
}


def _connect(path: str | Path) -> sqlite3.Connection:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS tickers_state (
            symbol TEXT PRIMARY KEY,
            profilo_rischio TEXT,
            custom_trailing_pct REAL,
            high_water_mark REAL,
            last_buy_date TEXT
        )"""
    )


def _public_record(row: sqlite3.Row) -> dict:
    record = {
        "profile": row["profilo_rischio"],
        "custom_trailing_pct": row["custom_trailing_pct"],
        "high_water_mark": row["high_water_mark"],
        "last_buy_date": row["last_buy_date"],
    }
    record["trailing_pct"] = record["custom_trailing_pct"]
    if record["trailing_pct"] is None:
        record["trailing_pct"] = _PROFILE_TRAILING.get(record["profile"], 0.06)
    return record


def load_bot_state(path: str | Path = STATE_PATH) -> dict:
    """Load all ticker settings, returning an empty state when no rows exist."""
    try:
        with closing(_connect(path)) as connection:
            with connection:
                _ensure_schema(connection)
                rows = connection.execute("SELECT * FROM tickers_state").fetchall()
        return {"symbols": {row["symbol"]: _public_record(row) for row in rows}}
    except sqlite3.Error:
        LOG.exception("Cannot read bot state from %s", path)
        raise


def get_symbol_state(symbol: str, path: str | Path = STATE_PATH) -> dict:
    """Return a copy of one symbol's persisted settings."""
    with closing(_connect(path)) as connection:
        with connection:
            _ensure_schema(connection)
            row = connection.execute(
                "SELECT * FROM tickers_state WHERE symbol = ?", (symbol,)
            ).fetchone()
    return _public_record(row) if row is not None else {}


def update_symbol_state(
    symbol: str,
    updates: dict,
    path: str | Path = STATE_PATH,
) -> None:
    """Atomically merge supported ticker settings into SQLite."""
    if not symbol:
        raise ValueError("symbol must not be empty")

    clean_updates = {}
    for key, value in updates.items():
        column = _ALIASES.get(key)
        if column is None:
            raise ValueError(f"unsupported bot-state field: {key}")
        if column in ("custom_trailing_pct", "high_water_mark") and value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{key} must be numeric or None")
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            value = float(value)
        elif value is not None and not isinstance(value, str):
            raise TypeError(f"{key} must be a string or None")
        clean_updates[column] = value

    with closing(_connect(path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _ensure_schema(connection)
            columns = list(clean_updates)
            insert_columns = ["symbol", *columns]
            values = [symbol, *(clean_updates[column] for column in columns)]
            placeholders = ", ".join("?" for _ in insert_columns)
            if columns:
                assignments = ", ".join(f"{column} = excluded.{column}" for column in columns)
                conflict = f"DO UPDATE SET {assignments}"
            else:
                conflict = "DO NOTHING"
            connection.execute(
                f"INSERT INTO tickers_state ({', '.join(insert_columns)}) "
                f"VALUES ({placeholders}) ON CONFLICT(symbol) {conflict}",
                values,
            )