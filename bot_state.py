"""SQLite persistence for per-symbol live-trading settings."""

from __future__ import annotations

import logging
import json
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path

LOG = logging.getLogger(__name__)
STATE_PATH = Path(os.environ.get("BOT_STATE_PATH", Path(__file__).resolve().parent / "bot_state.db"))

_COLUMNS = (
    "profilo_rischio",
    "custom_trailing_pct",
    "high_water_mark",
    "last_buy_date",
    "custom_settings_json",
    "bot_enabled",
    "active_ticker",
)
_ALIASES = {
    "profile": "profilo_rischio",
    "trailing_pct": "custom_trailing_pct",
    "custom_settings": "custom_settings_json",
    **{column: column for column in _COLUMNS},
}
_PROFILE_TRAILING = {
    "🐢 Conservativo": 0.03,
    "⚖️ Bilanciato": 0.06,
    "🚀 Speculativo": 0.12,
}


def connect_db(path: str | Path = STATE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection to the shared bot-state database."""
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_path, timeout=10, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def _connect(path: str | Path) -> sqlite3.Connection:
    return connect_db(path)


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS tickers_state (
            symbol TEXT PRIMARY KEY,
            profilo_rischio TEXT,
            custom_trailing_pct REAL,
            high_water_mark REAL,
            last_buy_date TEXT,
            custom_settings_json TEXT,
            bot_enabled INTEGER NOT NULL DEFAULT 0,
            active_ticker INTEGER NOT NULL DEFAULT 0
        )"""
    )
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(tickers_state)")}
    if "custom_settings_json" not in columns:
        connection.execute("ALTER TABLE tickers_state ADD COLUMN custom_settings_json TEXT")
    if "bot_enabled" not in columns:
        connection.execute("ALTER TABLE tickers_state ADD COLUMN bot_enabled INTEGER NOT NULL DEFAULT 0")
    if "active_ticker" not in columns:
        connection.execute("ALTER TABLE tickers_state ADD COLUMN active_ticker INTEGER NOT NULL DEFAULT 0")


def _public_record(row: sqlite3.Row) -> dict:
    custom_settings = {}
    if row["custom_settings_json"]:
        try:
            decoded_settings = json.loads(row["custom_settings_json"])
            if isinstance(decoded_settings, dict):
                custom_settings = decoded_settings
            else:
                LOG.warning("Ignoring non-object custom settings for %s", row["symbol"])
        except (json.JSONDecodeError, TypeError):
            LOG.warning("Ignoring invalid custom settings JSON for %s", row["symbol"])
    record = {
        "profile": row["profilo_rischio"],
        "custom_trailing_pct": row["custom_trailing_pct"],
        "high_water_mark": row["high_water_mark"],
        "last_buy_date": row["last_buy_date"],
        "custom_settings": custom_settings,
        "bot_enabled": bool(row["bot_enabled"]),
        "active_ticker": bool(row["active_ticker"]),
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
        if column == "custom_settings_json" and value is not None:
            if not isinstance(value, dict):
                raise TypeError(f"{key} must be an object or None")
            try:
                value = json.dumps(value, allow_nan=False, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must contain JSON-compatible finite values") from exc
        elif column in ("bot_enabled", "active_ticker"):
            if not isinstance(value, bool):
                raise TypeError(f"{key} must be a boolean")
            value = int(value)
        elif column in ("custom_trailing_pct", "high_water_mark") and value is not None:
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