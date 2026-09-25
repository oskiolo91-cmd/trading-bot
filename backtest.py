"""Candle-by-candle backtest engine with maker limit fills and walk-forward validation."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from indicators import INDICATOR_COLUMNS, add_indicators
from macro_filter import is_risk_off_day
from models import ExitDecision, ExitReason, Order, Position, StrategyParams
from risk import calc_position_size
from signals import entry_limit_price, evaluate_exit, limit_order_filled, stop_loss_price

OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
TRADE_COLUMNS: tuple[str, ...] = (
    "entry_date", "exit_date", "entry_price", "exit_price", "shares",
    "gross_pnl", "commission", "net_pnl", "exit_reason",
)


@dataclass(frozen=True)
class BacktestResult:
    """Output of a backtest run."""

    trades: pd.DataFrame
    metrics: dict[str, Any]
    equity: pd.Series


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load an OHLCV CSV with columns Date, Open, High, Low, Close, Adj Close, Volume."""
    return pd.read_csv(path)


def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw OHLCV data and move all prices into the adjusted (Adj Close) price space.

    - parses/sorts/deduplicates Date and uses it as the index
    - coerces numerics, drops rows without a usable Adj Close / Close
    - fills missing Open/High/Low from Close and missing Volume with 0
    - rescales Open/High/Low/Close by Adj Close / Close (raw close kept in "Raw Close")
    - enforces Low <= min(Open, Close) and High >= max(Open, Close)
    """
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"]).set_index("Date")
    out.index = pd.DatetimeIndex(out.index, name="Date")
    out = out[~out.index.duplicated(keep="last")].sort_index()

    if "Adj Close" not in out.columns:
        out["Adj Close"] = out["Close"]
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in OHLCV_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[(out["Close"] > 0) & (out["Adj Close"] > 0)]
    for col in ("Open", "High", "Low"):
        out[col] = out[col].where(out[col] > 0, out["Close"])
    out["Volume"] = out["Volume"].fillna(0.0)

    factor = out["Adj Close"] / out["Close"]
    out["Raw Close"] = out["Close"]
    for col in ("Open", "High", "Low", "Close"):
        out[col] = out[col] * factor
    out["Close"] = out["Adj Close"]
    out["High"] = out[["High", "Open", "Close"]].max(axis=1)
    out["Low"] = out[["Low", "Open", "Close"]].min(axis=1)
    return out[[*OHLCV_COLUMNS, "Raw Close"]]


def prepare_data(df: pd.DataFrame, **indicator_kwargs: Any) -> pd.DataFrame:
    """Sanitize raw OHLCV data and append indicators."""
    return add_indicators(sanitize_ohlcv(df), **indicator_kwargs)


def _close_trade(position: Position, exit_date: pd.Timestamp, decision: ExitDecision, commission_pct: float) -> dict[str, Any]:
    """Build a trade record for a closed position."""
    exit_commission = position.shares * decision.price * commission_pct
    gross = position.shares * (decision.price - position.entry_price)
    commission = position.entry_commission + exit_commission
    return {
        "entry_date": position.entry_date,
        "exit_date": exit_date,
        "entry_price": position.entry_price,
        "exit_price": decision.price,
        "shares": position.shares,
        "gross_pnl": gross,
        "commission": commission,
        "net_pnl": gross - commission,
        "exit_reason": decision.reason.value,
    }


def _exit_proceeds(position: Position, decision: ExitDecision, commission_pct: float) -> float:
    """Cash received when closing a position, net of exit commission."""
    notional = position.shares * decision.price
    return notional - notional * commission_pct


def run_backtest(
    data: pd.DataFrame,
    params: StrategyParams = StrategyParams(),
    portfolio_value: float = 100_000.0,
    calendar: list[dict[str, Any]] | None = None,
    symbol: str | None = None,
) -> BacktestResult:
    """Simulate the strategy candle by candle on prepared data."""
    missing = [c for c in INDICATOR_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Data is missing indicator columns {missing}; call prepare_data first")

    rows = data.to_dict("records")
    dates = list(data.index)
    cash = float(portfolio_value)
    position: Position | None = None
    order: Order | None = None
    trades: list[dict[str, Any]] = []
    equity = np.empty(len(rows))

    for i, (day, row) in enumerate(zip(dates, rows)):
        if position is None and order is not None:
            if limit_order_filled(order.limit_price, row["Low"]):
                fill_price = order.limit_price
                entry_commission = order.shares * fill_price * params.commission_pct
                cash -= order.shares * fill_price + entry_commission
                position = Position(
                    entry_date=day, entry_price=fill_price, stop_loss=order.stop_loss,
                    shares=order.shares, entry_commission=entry_commission, peak_price=fill_price,
                )
            order = None

        if position is not None:
            position, decision = evaluate_exit(
                position, row["Open"], row["High"], row["Low"], row["Close"], row["BB_mid"],
                trailing_pct=params.trailing_pct, time_stop=params.time_stop,
            )
            if decision is not None:
                trade = _close_trade(position, day, decision, params.commission_pct)
                cash += _exit_proceeds(position, decision, params.commission_pct)
                trades.append(trade)
                position = None

        if position is None and i < len(rows) - 1:
            order = _maybe_place_order(row, day, dates[i + 1], cash, params, calendar, symbol)

        equity[i] = cash + (position.shares * row["Close"] if position is not None else 0.0)

    if position is not None and rows:
        decision = ExitDecision(ExitReason.END_OF_DATA, rows[-1]["Close"])
        trade = _close_trade(position, dates[-1], decision, params.commission_pct)
        cash += _exit_proceeds(position, decision, params.commission_pct)
        trades.append(trade)
        equity[-1] = cash

    trades_df = pd.DataFrame(trades, columns=list(TRADE_COLUMNS))
    equity_s = pd.Series(equity, index=data.index, name="equity")
    return BacktestResult(trades_df, compute_metrics(trades_df, equity_s, portfolio_value), equity_s)


def _maybe_place_order(row, day, next_day, cash, params, calendar, symbol):
    limit = entry_limit_price(row["Close"], row["BB_lower"], row["RSI"], row["ADX"], params.adx_max, params.rsi_max)
    if limit is None or not math.isfinite(row["ATR"]): return None
    if calendar and (is_risk_off_day(day, calendar, symbol) or is_risk_off_day(next_day, calendar, symbol)): return None
    stop = stop_loss_price(limit, row["ATR"], params.atr_mult)
    shares = calc_position_size(cash, limit, stop, params.risk_pct, params.max_cap_pct)
    affordable = math.floor(cash / (limit * (1.0 + params.commission_pct)))
    shares = min(shares, affordable)
    if shares <= 0: return None
    return Order(created_date=day, limit_price=limit, stop_loss=stop, shares=shares)


def compute_metrics(trades, equity, initial_capital):
    total = int(len(trades))
    net = trades["net_pnl"] if total else pd.Series(dtype=float)
    gross_profit = float(net[net > 0].sum())
    gross_loss = float(-net[net < 0].sum())
    curve = pd.concat([pd.Series([initial_capital]), equity.reset_index(drop=True)])
    drawdown = 1.0 - curve / curve.cummax()
    final_equity = float(curve.iloc[-1])
    return {
        "net_pnl_total": float(net.sum()),
        "win_rate": float((net > 0).sum() / total) if total else 0.0,
        "max_drawdown": float(drawdown.max()),
        "total_trades": total,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "return_pct": final_equity / initial_capital - 1.0,
    }


def walk_forward_split(data, in_sample_pct=0.7):
    if not 0.0 < in_sample_pct < 1.0: raise ValueError("in_sample_pct must be strictly between 0 and 1")
    cut = int(len(data) * in_sample_pct)
    return data.iloc[:cut], data.iloc[cut:]


def run_walk_forward(raw, params=None, portfolio_value=100_000.0, in_sample_pct=0.7, calendar=None, symbol=None):
    if params is None: params = StrategyParams()
    prepared = prepare_data(raw)
    _, out_sample = walk_forward_split(prepared, in_sample_pct)
    return run_backtest(out_sample, params, portfolio_value, calendar, symbol)


async def run_walk_forward_async(raw, params=None, portfolio_value=100_000.0, in_sample_pct=0.7, calendar=None, symbol=None):
    return await asyncio.to_thread(run_walk_forward, raw, params, portfolio_value, in_sample_pct, calendar, symbol)
