"""Pure entry / exit signal functions (no side effects)."""

from __future__ import annotations

import math
from dataclasses import replace

from models import ExitDecision, ExitReason, Position


def _valid(*values: float) -> bool:
    """True if every value is a finite number."""
    return all(v is not None and math.isfinite(v) for v in values)


def entry_limit_price(
    close: float,
    bb_lower: float,
    rsi: float,
    adx: float,
    adx_max: float = 25.0,
    rsi_max: float = 35.0,
) -> float | None:
    """Return the buy-limit price (BB_lower) if all entry conditions hold, else None.

    Conditions: ADX < adx_max (range market), close <= BB_lower, RSI < rsi_max.
    """
    if not _valid(close, bb_lower, rsi, adx):
        return None
    if adx < adx_max and close <= bb_lower and rsi < rsi_max:
        return float(bb_lower)
    return None


def stop_loss_price(entry_price: float, atr: float, atr_mult: float = 2.0) -> float:
    """Fixed stop-loss ``atr_mult`` ATRs below the entry price."""
    return entry_price - atr_mult * atr


def limit_order_filled(limit_price: float, next_low: float) -> bool:
    """Maker fill rule: a buy limit fills only if the next candle's Low trades strictly below it."""
    return _valid(limit_price, next_low) and next_low < limit_price


def evaluate_exit(
    position: Position,
    open_: float,
    high: float,
    low: float,
    close: float,
    bb_mid: float,
    trailing_pct: float = 0.03,
    time_stop: int = 10,
) -> tuple[Position, ExitDecision | None]:
    """Advance the position by one candle and decide whether to exit.

    Priority: stop-loss -> trailing stop -> time stop. Gaps through a stop fill at the Open.
    The trailing stop arms only once Close crosses above BB_mid; from then on it tracks the
    running High peak and exits if Low falls ``trailing_pct`` below the peak set by prior candles.
    The time stop closes at Close after ``time_stop`` candles if the trailing stop never armed.

    Returns:
        (updated position, ExitDecision or None).
    """
    pos = replace(position, candles_in_trade=position.candles_in_trade + 1)

    if low <= pos.stop_loss:
        return pos, ExitDecision(ExitReason.STOP_LOSS, min(open_, pos.stop_loss))

    if pos.trailing_active:
        trail_level = pos.peak_price * (1.0 - trailing_pct)
        if low <= trail_level:
            return pos, ExitDecision(ExitReason.TRAILING_STOP, min(open_, trail_level))
        pos = replace(pos, peak_price=max(pos.peak_price, high))
    elif _valid(bb_mid) and close > bb_mid:
        pos = replace(pos, trailing_active=True, peak_price=high)

    if not pos.trailing_active and pos.candles_in_trade >= time_stop:
        return pos, ExitDecision(ExitReason.TIME_STOP, close)

    return pos, None
