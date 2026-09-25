"""Macro / corporate-actions calendar filter."""
from __future__ import annotations
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
import pandas as pd

def _to_date(value):
    return pd.Timestamp(value).date()

def load_calendar(path):
    with open(path, encoding="utf-8") as fh: data = json.load(fh)
    if not isinstance(data, list): raise ValueError("Calendar must be a list")
    return data

def is_risk_off_day(day, calendar, symbol=None, impact_level="high"):
    target = _to_date(day)
    for event in calendar:
        if str(event.get("impact","")).lower() != impact_level: continue
        ev_sym = event.get("symbol")
        if ev_sym is not None and ev_sym != symbol: continue
        if _to_date(event["date"]) == target: return True
    return False

def has_risk_off_within(start, calendar, days_ahead=1, symbol=None, impact_level="high"):
    base = _to_date(start)
    return any(is_risk_off_day(base+timedelta(days=o), calendar, symbol, impact_level) for o in range(days_ahead+1))
