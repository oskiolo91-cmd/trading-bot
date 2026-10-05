"""Best-effort Telegram notifications for live trading events."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests

LOG = logging.getLogger(__name__)


def _dotenv_values() -> dict[str, str]:
    values = {}
    dotenv_paths = (Path(__file__).resolve().parent / ".env", Path.cwd() / ".env")
    for dotenv_path in dict.fromkeys(dotenv_paths):
        try:
            lines = dotenv_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            name, separator, value = stripped.partition("=")
            if not separator or name.strip() not in {"TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"}:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            values[name.strip()] = value
    return values


def _telegram_credentials() -> tuple[str | None, str | None]:
    secrets = {}
    try:
        import streamlit as st
        secrets = st.secrets
    except Exception:
        pass

    dotenv = _dotenv_values()
    token = (
        os.environ.get("TELEGRAM_BOT_TOKEN")
        or secrets.get("TELEGRAM_BOT_TOKEN")
        or dotenv.get("TELEGRAM_BOT_TOKEN")
    )
    chat_id = (
        os.environ.get("TELEGRAM_CHAT_ID")
        or secrets.get("TELEGRAM_CHAT_ID")
        or dotenv.get("TELEGRAM_CHAT_ID")
    )
    return token, chat_id


def send_telegram_message(message: str) -> bool:
    """Send a Telegram message; telemetry failures never interrupt trading."""
    try:
        token, chat_id = _telegram_credentials()
        if not token or not chat_id:
            LOG.warning("Telegram telemetry skipped: bot token or chat ID is not configured")
            return False
        if not message.strip():
            return False
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=3,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok", False):
            LOG.warning("Telegram rejected a telemetry message")
            return False
        return True
    except Exception:
        LOG.warning("Telegram telemetry delivery failed")
        return False


def send_critical_alert(source: str, error: BaseException | str) -> bool:
    """Report an operational failure without exposing the traceback or credentials."""
    detail = " ".join(str(error).split())[:500]
    message = f"⚠️ Rilevata anomalia ({source}): {detail or type(error).__name__}"
    return send_telegram_message(message)
