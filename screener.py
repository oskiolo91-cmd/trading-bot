"""Offline batch screener selecting liquid, range-bound assets without imminent macro risk."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Any

from backtest import load_ohlcv_csv, prepare_data
from macro_filter import has_risk_off_within


def evaluate_asset(csv_path, min_avg_volume=1_000_000, lookback=20, adx_max=25.0, min_lateral_ratio=0.5, min_atr_pct=0.005):
    data = prepare_data(load_ohlcv_csv(csv_path))
    recent = data.tail(lookback)
    avg_volume = float(recent["Volume"].mean()) if len(recent) else 0.0
    adx = recent["ADX"].dropna()
    lateral_ratio = float((adx < adx_max).sum() / lookback) if lookback > 0 else 0.0
    last = data.iloc[-1] if len(data) else None
    atr_pct = float(last["ATR"] / last["Close"]) if last is not None and last["Close"] > 0 else float("nan")
    passes = len(recent) >= lookback and avg_volume >= min_avg_volume and lateral_ratio > min_lateral_ratio and atr_pct >= min_atr_pct
    return {"avg_volume": avg_volume, "atr_pct": atr_pct, "lateral_ratio": lateral_ratio, "passes": bool(passes)}


def screen_assets(csv_paths, min_avg_volume=1_000_000, calendar=None, as_of=None, macro_days_ahead=1, lookback=20, adx_max=25.0, min_lateral_ratio=0.5, min_atr_pct=0.005):
    check_day = as_of or date.today()
    selected = []
    for path in csv_paths:
        p = Path(path)
        if calendar and has_risk_off_within(check_day, calendar, macro_days_ahead, symbol=p.stem): continue
        stats = evaluate_asset(p, min_avg_volume, lookback, adx_max, min_lateral_ratio, min_atr_pct)
        if stats["passes"]: selected.append(p.name)
    return selected


async def screen_assets_async(csv_paths, min_avg_volume=1_000_000, calendar=None, as_of=None, macro_days_ahead=1):
    results = await asyncio.gather(*(asyncio.to_thread(screen_assets, [p], min_avg_volume, calendar, as_of, macro_days_ahead) for p in csv_paths))
    return [n for chunk in results for n in chunk]
