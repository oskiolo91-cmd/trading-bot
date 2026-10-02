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
    """A resting buy limit order, including optional scale-out quantities."""

    created_date: pd.Timestamp
    limit_price: float
    stop_loss: float
    shares: float
    side: str = "BUY"
    take_profit_price: float | None = None
    shares_1: float | None = None
    shares_2: float | None = None


@dataclass(frozen=True)
class Position:
    """State of an open long position."""

    entry_date: pd.Timestamp
    entry_price: float
    stop_loss: float
    shares: float
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

    bot_mode: str = "TREND_FOLLOWER"
    trade_budget_usd: float = 100.0
    commission_pct: float = 0.001
    trailing_pct: float = 0.03
    time_stop: int = 10
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    adx_max: float = 25.0
    rsi_max: float = 35.0
    daily_target_usd: float = 10.0
    max_daily_drawdown_usd: float = 5.0


def strategy_params_for_mode(mode: str) -> StrategyParams:
    """Build the fixed-budget defaults for one supported live strategy."""
    normalized_mode = mode.strip().upper()
    if normalized_mode == "TREND_FOLLOWER":
        return StrategyParams(
            bot_mode=normalized_mode,
            stop_loss_atr_mult=2.0,
            take_profit_atr_mult=3.0,
            daily_target_usd=10.0,
            max_daily_drawdown_usd=5.0,
        )
    if normalized_mode == "DAILY_SCALPER":
        return StrategyParams(
            bot_mode=normalized_mode,
            stop_loss_atr_mult=1.0,
            take_profit_atr_mult=0.5,
            daily_target_usd=5.0,
            max_daily_drawdown_usd=5.0,
        )
    raise ValueError("BOT_MODE must be TREND_FOLLOWER or DAILY_SCALPER")
