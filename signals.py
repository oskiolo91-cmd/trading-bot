"""Pure entry / exit signal functions (no side effects)."""

from __future__ import annotations
import math
from dataclasses import replace
from models import ExitDecision, ExitReason, Position

def _valid(*values):
    return all(v is not None and math.isfinite(v) for v in values)

def entry_limit_price(close, bb_lower, rsi, adx, adx_max=25.0, rsi_max=35.0):
    if not _valid(close, bb_lower, rsi, adx): return None
    if adx < adx_max and close <= bb_lower and rsi < rsi_max: return float(bb_lower)
    return None

def stop_loss_price(entry_price, atr, atr_mult=2.0):
    return entry_price - atr_mult * atr

def trailing_stop_price(high_water_mark, trailing_pct):
    if not _valid(high_water_mark, trailing_pct) or high_water_mark <= 0 or not 0 < trailing_pct < 1:
        return None
    return high_water_mark * (1.0 - trailing_pct)

def limit_order_filled(limit_price, next_low):
    return _valid(limit_price, next_low) and next_low < limit_price

def evaluate_exit(position, open_, high, low, close, bb_mid, trailing_pct=0.03, time_stop=10):
    pos = replace(position, candles_in_trade=position.candles_in_trade + 1)
    if _valid(pos.peak_price, high):
        pos = replace(pos, peak_price=max(pos.peak_price, high))
    if low <= pos.stop_loss:
        return pos, ExitDecision(ExitReason.STOP_LOSS, min(open_, pos.stop_loss))
    trail_level = trailing_stop_price(pos.peak_price, trailing_pct)
    if trail_level is not None:
        if close <= trail_level:
            return pos, ExitDecision(ExitReason.TRAILING_STOP, min(open_, trail_level))
    if not pos.trailing_active and _valid(bb_mid) and close > bb_mid:
        pos = replace(pos, trailing_active=True)
    if not pos.trailing_active and pos.candles_in_trade >= time_stop:
        return pos, ExitDecision(ExitReason.TIME_STOP, close)
    return pos, None
