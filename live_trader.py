"""Alpaca paper-trading execution for the daily SPY strategy."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest, StopLossRequest
import pandas as pd
import yfinance as yf

BOT_DIR = Path(__file__).resolve().parent
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from backtest import prepare_data
from models import ExitDecision, ExitReason, Order, Position, StrategyParams
from risk import calc_position_size
from signals import entry_limit_price, evaluate_exit, stop_loss_price

EASTERN = ZoneInfo("America/New_York")
LOG = logging.getLogger(__name__)


def get_alpaca_client() -> TradingClient:
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ValueError("API keys not configured: set ALPACA_API_KEY and ALPACA_SECRET_KEY")
    return TradingClient(api_key=key, secret_key=secret, paper=True)


def get_account_value(client: TradingClient) -> float:
    return float(client.get_account().equity)


def get_open_position(client: TradingClient, symbol: str):
    try:
        return client.get_open_position(symbol)
    except APIError as exc:
        if exc.status_code == 404:
            return None
        raise


def has_open_position(client: TradingClient, symbol: str) -> bool:
    position = get_open_position(client, symbol)
    return position is not None and float(position.qty) != 0


def place_limit_buy(client: TradingClient, symbol: str, limit_price: float, shares: int, stop_loss_price: float) -> str:
    if not (math.isfinite(limit_price) and math.isfinite(stop_loss_price)) or not (0 < stop_loss_price < limit_price) or shares <= 0:
        raise ValueError("Invalid buy order price, stop or share count")
    limit = round(limit_price, 2)
    stop = round(stop_loss_price, 2)
    if not 0 < stop < limit:
        raise ValueError("Stop must be below the rounded limit price")
    request = LimitOrderRequest(
        symbol=symbol, qty=shares, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        limit_price=limit, order_class=OrderClass.OTO,
        take_profit=None, stop_loss=StopLossRequest(stop_price=stop),
    )
    return str(client.submit_order(order_data=request).id)


def place_market_sell(client: TradingClient, symbol: str, shares: int) -> str:
    if shares <= 0:
        raise ValueError("Shares must be positive")
    request = MarketOrderRequest(symbol=symbol, qty=shares, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
    return str(client.submit_order(order_data=request).id)


def get_latest_bars(symbol: str, n: int = 60) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be positive")
    frame = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if frame.empty:
        raise ValueError(f"No daily prices returned for {symbol}")
    frame = frame.tail(n).copy()
    frame["Adj Close"] = frame["Close"]
    return frame[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


def run_signal_check(symbol: str = "SPY", portfolio_value: float = 100_000.0, params: StrategyParams = StrategyParams()) -> Order | None:
    bars = get_latest_bars(symbol)
    data = prepare_data(bars)
    today = datetime.now(EASTERN).date()
    finished = data.loc[pd.DatetimeIndex(data.index).date < today]
    if finished.empty:
        return None
    row = finished.iloc[-1]
    limit = entry_limit_price(row["Close"], row["BB_lower"], row["RSI"], row["ADX"], params.adx_max, params.rsi_max)
    if limit is None or not math.isfinite(row["ATR"]):
        return None
    stop = stop_loss_price(limit, row["ATR"], params.atr_mult)
    shares = calc_position_size(portfolio_value, limit, stop, params.risk_pct, params.max_cap_pct)
    shares = min(shares, math.floor(portfolio_value / (limit * (1 + params.commission_pct))))
    if shares <= 0 or not (0 < round(stop, 2) < round(limit, 2)):
        return None
    return Order(created_date=finished.index[-1], limit_price=float(limit), stop_loss=float(stop), shares=shares)


def get_recent_orders(client: TradingClient, limit: int = 10) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit))
    return [{
        "id": str(order.id), "symbol": order.symbol,
        "side": order.side.value if hasattr(order.side, "value") else str(order.side),
        "qty": str(order.qty),
        "status": order.status.value if hasattr(order.status, "value") else str(order.status),
        "submitted_at": str(order.submitted_at) if order.submitted_at else None,
        "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price is not None else None,
    } for order in orders]


def _open_orders(client: TradingClient, symbol: str) -> list:
    orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[symbol]))
    result = []
    for order in orders:
        if order.symbol == symbol:
            result.append(order)
            result.extend(
                leg for leg in (getattr(order, "legs", None) or [])
                if leg.symbol == symbol and str(getattr(leg.status, "value", leg.status))
                in {"new", "accepted", "pending_new", "partially_filled", "held"}
            )
    return result


def _initial_position(client: TradingClient, symbol: str) -> Position | None:
    position = get_open_position(client, symbol)
    if position is None or float(position.qty) <= 0:
        return None
    stops = [order for order in _open_orders(client, symbol)
             if order.side == OrderSide.SELL and getattr(order, "stop_price", None)]
    if not stops:
        LOG.warning("No protective stop found for %s; skipping unmanaged position", symbol)
        return None
    price = float(position.avg_entry_price)
    filled = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=100, nested=False, symbols=[symbol]))
    buys = [order for order in filled if order.symbol == symbol and order.side == OrderSide.BUY and getattr(order, "filled_at", None)]
    entry_date = pd.Timestamp(max(buys, key=lambda order: order.filled_at).filled_at) if buys else pd.Timestamp.now(tz=EASTERN)
    return Position(entry_date=entry_date, entry_price=price, stop_loss=float(stops[0].stop_price),
                    shares=int(float(position.qty)), entry_commission=0.0, peak_price=price)


def _check_exit_once(client, symbol, position_state, params):
    if not has_open_position(client, symbol):
        position_state.clear(); return None
    if position_state.get("pending_exit"):
        status = client.get_order_by_id(position_state["pending_exit"]).status
        if str(getattr(status, "value", status)) not in {"canceled", "rejected", "expired"}:
            return None
        position_state.pop("pending_exit", None)
    bars = get_latest_bars(symbol)
    data = prepare_data(bars)
    if data.empty: return None
    current = data.iloc[-1]
    day = pd.Timestamp(data.index[-1]).date()
    today = datetime.now(EASTERN).date()
    if day != today or not math.isfinite(float(current["BB_mid"])): return None
    if day != position_state.get("day"):
        position_state["day"] = day
        position_state["base"] = position_state["position"]
    base = position_state["base"]
    if day <= base.entry_date.date(): return None
    updated, decision = evaluate_exit(
        base, float(current["Open"]), float(current["High"]), float(current["Low"]),
        float(current["Close"]), float(current["BB_mid"]), params.trailing_pct, params.time_stop,
    )
    if decision is None:
        position_state["position"] = updated; return None
    if decision.reason is ExitReason.STOP_LOSS: return decision
    for order in _open_orders(client, symbol):
        if order.side == OrderSide.SELL: client.cancel_order_by_id(order.id)
    if any(o.side == OrderSide.SELL for o in _open_orders(client, symbol)): return None
    pos = get_open_position(client, symbol)
    if pos is None: position_state.clear(); return None
    sell_id = place_market_sell(client, symbol, int(float(pos.qty)))
    position_state["pending_exit"] = sell_id
    position_state["position"] = updated
    return decision


async def run_exit_check(client, symbol, params=None, poll_seconds=60):
    if params is None: params = StrategyParams()
    if poll_seconds <= 0: raise ValueError("poll_seconds must be positive")
    position_state: dict = {}
    while True:
        try:
            clock = await asyncio.to_thread(client.get_clock)
            if clock.is_open:
                if not await asyncio.to_thread(has_open_position, client, symbol):
                    position_state.clear()
                else:
                    if "position" not in position_state:
                        p = await asyncio.to_thread(_initial_position, client, symbol)
                        if p: position_state["position"] = p
                    if "position" in position_state:
                        d = await asyncio.to_thread(_check_exit_once, client, symbol, position_state, params)
                        if d: LOG.info("Exit %s: %s", symbol, d.reason.value)
        except Exception:
            LOG.exception("Exit check failed; will retry")
        await asyncio.sleep(poll_seconds)


async def run_live_loop(symbol="SPY", params=None, poll_seconds=60):
    if params is None: params = StrategyParams()
    if poll_seconds <= 0: raise ValueError("poll_seconds must be positive")
    client = get_alpaca_client()
    last_screen_day = None
    position_state: dict = {}
    while True:
        try:
            clock = await asyncio.to_thread(client.get_clock)
            if clock.is_open:
                now = datetime.now(EASTERN)
                if now.hour == 9 and now.minute == 30 and last_screen_day != now.date():
                    last_screen_day = now.date()
                    if not await asyncio.to_thread(has_open_position, client, symbol) and not await asyncio.to_thread(_open_orders, client, symbol):
                        value = await asyncio.to_thread(get_account_value, client)
                        order = await asyncio.to_thread(run_signal_check, symbol, value, params)
                        if order:
                            oid = await asyncio.to_thread(place_limit_buy, client, symbol, order.limit_price, order.shares, order.stop_loss)
                            LOG.info("Submitted paper buy %s for %s", oid, symbol)
                if await asyncio.to_thread(has_open_position, client, symbol):
                    if "position" not in position_state:
                        p = await asyncio.to_thread(_initial_position, client, symbol)
                        if p: position_state["position"] = p
                    if "position" in position_state:
                        d = await asyncio.to_thread(_check_exit_once, client, symbol, position_state, params)
                        if d: LOG.info("Exit %s: %s", symbol, d.reason.value)
                else:
                    position_state.clear()
            else:
                LOG.debug("Market closed; next open: %s", clock.next_open)
        except Exception:
            LOG.exception("Live loop failed; will retry")
        now = datetime.now(EASTERN)
        await asyncio.sleep(min(poll_seconds, max(1, 60 - now.second - now.microsecond / 1_000_000)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_live_loop(os.environ.get("TRADING_SYMBOL", "SPY")))
