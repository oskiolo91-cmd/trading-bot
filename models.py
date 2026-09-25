"""Lightweight live-state dataclasses shared by the backtest engine and future live-trading phases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class ExitReason(str, Enum):
    """Reason a position was closed."""

    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    END_OF_DATA = "end_of_data"


@dataclass(frozen=True)
class Order:
    """A resting buy limit order, valid for the next candle only."""

    created_date: pd.Timestamp
    limit_price: float
    stop_loss: float
    shares: int
    side: str = "BUY"


@dataclass(frozen=True)
class Position:
    """State of an open long position."""

    entry_date: pd.Timestamp
    entry_price: float
    stop_loss: float
    shares: int
    entry_commission: float
    peak_price: float
    trailing_active: bool = False
    candles_in_trade: int = 0


@dataclass(frozen=True)
class ExitDecision:
    """Outcome of an exit check that triggered a close."""

    reason: ExitReason
    price: float


@dataclass(frozen=True)
class StrategyParams:
    """All tunable strategy / execution parameters in one place."""

    risk_pct: float = 0.01
    max_cap_pct: float = 0.10
    commission_pct: float = 0.001
    trailing_pct: float = 0.03
    time_stop: int = 10
    atr_mult: float = 2.0
    adx_max: float = 25.0
    rsi_max: float = 35.0
