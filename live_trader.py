"""Alpaca paper-trading execution for the daily SPY strategy."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
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


def close_all_positions(client: TradingClient) -> bool:
    """Close every open long position when the daily profit target is reached."""
    positions = client.get_all_positions()
    if not positions:
        return False

    for position in positions:
        if float(position.qty) <= 0:
            continue
        for order in _open_orders(client, position.symbol):
            if order.side == OrderSide.SELL:
                client.cancel_order_by_id(order.id)

    closed = []
    for position in client.get_all_positions():
        if float(position.qty) <= 0:
            continue
        sell_id = place_market_sell(client, position.symbol, int(float(position.qty)))
        closed.append(sell_id)
    return bool(closed)


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


def place_limit_buy(
    client: TradingClient,
    symbol: str,
    limit_price: float,
    shares: int,
    stop_loss_price: float,
    take_profit_price: float = None,
) -> str:
    if not (math.isfinite(limit_price) and math.isfinite(stop_loss_price)) or not (0 < stop_loss_price < limit_price) or shares <= 0:
        raise ValueError("Invalid buy order price, stop or share count")
    limit = round(limit_price, 2)
    stop = round(stop_loss_price, 2)
    if not 0 < stop < limit:
        raise ValueError("Stop must be below the rounded limit price")

    if take_profit_price is not None:
        if not math.isfinite(take_profit_price) or take_profit_price <= limit:
            raise ValueError("Invalid take profit price")
        request = LimitOrderRequest(
            symbol=symbol, qty=shares, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            limit_price=limit, order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
            stop_loss=StopLossRequest(stop_price=stop),
        )
    else:
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
    frame = yf.download(symbol, period="2y", interval="1d", auto_adjust=False, progress=False)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if frame.empty:
        raise ValueError(f"No daily prices returned for {symbol}")
    # Calculate the regime filter before truncating so its 200-bar history remains available.
    frame["SMA_200"] = frame["Close"].rolling(window=200, min_periods=200).mean()
    frame = frame.tail(n).copy()
    frame["Adj Close"] = frame["Close"]
    return frame[["Open", "High", "Low", "Close", "Adj Close", "Volume", "SMA_200"]]


def run_signal_check(symbol: str = "SPY", portfolio_value: float = 100_000.0, params: StrategyParams = StrategyParams()) -> Order | None:
    bars = get_latest_bars(symbol)
    data = prepare_data(bars)
    data["SMA_200"] = bars["SMA_200"].reindex(data.index)
    today = datetime.now(EASTERN).date()
    finished = data.loc[pd.DatetimeIndex(data.index).date < today]
    if finished.empty:
        return None
    row = finished.iloc[-1]
    if not math.isfinite(float(row["SMA_200"])) or float(row["Raw Close"]) < float(row["SMA_200"]):
        return None
    limit = entry_limit_price(row["Close"], row["BB_lower"], row["RSI"], row["ADX"], params.adx_max, params.rsi_max)
    if limit is None or not math.isfinite(row["ATR"]):
        return None
    stop = stop_loss_price(limit, row["ATR"], params.atr_mult)
    risk = limit - stop
    take_profit = limit + (risk * params.risk_reward)
    shares = calc_position_size(portfolio_value, limit, stop, params.risk_pct, params.max_cap_pct)
    shares = min(shares, math.floor(portfolio_value / (limit * (1 + params.commission_pct))))
    if shares <= 0 or not (0 < round(stop, 2) < round(limit, 2)):
        return None
    # Split size into a profit-taking tranche and a smaller runner tranche.
    shares_1 = (shares + 1) // 2
    shares_2 = shares // 2
    return Order(
        created_date=finished.index[-1],
        limit_price=float(limit),
        stop_loss=float(stop),
        shares=shares,
        take_profit_price=float(limit + (risk * 1.5)),
        shares_1=shares_1,
        shares_2=shares_2,
    )


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
        "limit_price": str(order.limit_price) if getattr(order, "limit_price", None) is not None else None,
        "stop_loss_price": (
            str(order.stop_loss.stop_price)
            if getattr(order, "stop_loss", None) is not None and hasattr(order.stop_loss, "stop_price")
            else str(order.stop_loss) if getattr(order, "stop_loss", None) is not None else None
        ),
        "take_profit_price": (
            str(order.take_profit.limit_price)
            if getattr(order, "take_profit", None) is not None and hasattr(order.take_profit, "limit_price")
            else str(order.take_profit) if getattr(order, "take_profit", None) is not None else None
        ),
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


def _runner_stop_order(client: TradingClient, symbol: str, runner_order_id: str | None):
    """Find the protective stop attached to the OTO runner, never the bracket tranche."""
    orders = _open_orders(client, symbol)
    runner_ids = {str(runner_order_id)} if runner_order_id else set()
    runner_stop_ids = set()
    for order in orders:
        order_class = getattr(order, "order_class", None)
        order_class = getattr(order_class, "value", order_class)
        if str(order_class).lower() == OrderClass.OTO.value:
            runner_ids.add(str(order.id))
            for leg in getattr(order, "legs", None) or []:
                if leg.side == OrderSide.SELL and getattr(leg, "stop_price", None) is not None:
                    runner_stop_ids.add(str(leg.id))
    for order in orders:
        if order.side != OrderSide.SELL or getattr(order, "stop_price", None) is None:
            continue
        parent_id = getattr(order, "parent_order_id", None)
        if str(order.id) in runner_stop_ids or (parent_id is not None and str(parent_id) in runner_ids):
            return order
    return None


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
    updated, decision = evaluate_exit(
        base, float(current["Open"]), float(current["High"]), float(current["Low"]),
        float(current["Close"]), float(current["BB_mid"]), params.trailing_pct, params.time_stop,
    )
    if decision is None:
        # Move only the runner's Alpaca stop: first to break-even, then upward with ATR.
        initial_stop = position_state.setdefault("initial_stop", float(base.stop_loss))
        stop_order = _runner_stop_order(client, symbol, position_state.get("runner_order_id"))
        if stop_order is not None and math.isfinite(float(current["ATR"])):
            entry_price = float(base.entry_price)
            current_price = float(current["Close"])
            current_stop = float(stop_order.stop_price)
            next_stop = current_stop
            break_even_trigger = entry_price + (entry_price - initial_stop)
            if current_price >= break_even_trigger:
                next_stop = max(next_stop, entry_price)
                atr_stop = current_price - (float(current["ATR"]) * params.atr_mult)
                next_stop = max(next_stop, atr_stop)
            rounded_stop = round(next_stop, 2)
            if rounded_stop > current_stop and rounded_stop < current_price:
                client.replace_order_by_id(
                    stop_order.id,
                    ReplaceOrderRequest(stop_price=rounded_stop),
                )
                updated = replace(updated, stop_loss=rounded_stop)
                position_state["base"] = updated
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
    runner_order_id = None
    daily_state = {"day": None, "baseline_equity": None, "halted": False, "target_reached": False}
    while True:
        try:
            clock = await asyncio.to_thread(client.get_clock)
            if clock.is_open:
                now = datetime.now(EASTERN)
                if daily_state["day"] != now.date():
                    daily_state.update(day=now.date(), baseline_equity=None, halted=False, target_reached=False)
                if daily_state["baseline_equity"] is None:
                    daily_state["baseline_equity"] = await asyncio.to_thread(get_account_value, client)
                if not daily_state["halted"]:
                    current_value = await asyncio.to_thread(get_account_value, client)
                    drawdown_floor = daily_state["baseline_equity"] * (1 - params.max_daily_drawdown_pct)
                    if current_value <= drawdown_floor:
                        await asyncio.to_thread(client.cancel_orders)
                        await asyncio.to_thread(client.close_all_positions)
                        daily_state["halted"] = True
                        position_state.clear()
                        LOG.error(
                            "Daily max drawdown reached: %.2f <= %.2f; orders canceled, positions closed, bot paused until tomorrow.",
                            current_value,
                            drawdown_floor,
                        )
                    elif not daily_state["target_reached"]:
                        target_threshold = daily_state["baseline_equity"] * (1 + params.daily_target_pct)
                        if current_value >= target_threshold:
                            await asyncio.to_thread(client.cancel_orders)
                            await asyncio.to_thread(client.close_all_positions)
                            daily_state["target_reached"] = True
                            LOG.info(
                                "Daily profit target reached: %.2f >= %.2f; orders canceled and positions closed.",
                                current_value,
                                target_threshold,
                            )
                if daily_state["halted"]:
                    LOG.debug("Daily risk lock active for %s; operational checks skipped.", now.date())
                elif daily_state["target_reached"]:
                    LOG.debug("Daily profit target active for %s; operational checks skipped.", now.date())
                else:
                    if now.hour == 9 and now.minute == 30 and last_screen_day != now.date():
                        last_screen_day = now.date()
                        if not await asyncio.to_thread(has_open_position, client, symbol) and not await asyncio.to_thread(_open_orders, client, symbol):
                            value = await asyncio.to_thread(get_account_value, client)
                            order = await asyncio.to_thread(run_signal_check, symbol, value, params)
                            if order:
                                bracket_id = await asyncio.to_thread(
                                    place_limit_buy,
                                    client,
                                    symbol,
                                    order.limit_price,
                                    order.shares_1 or order.shares,
                                    order.stop_loss,
                                    order.take_profit_price,
                                )
                                runner_id = None
                                if order.shares_2:
                                    runner_id = await asyncio.to_thread(
                                        place_limit_buy,
                                        client,
                                        symbol,
                                        order.limit_price,
                                        order.shares_2,
                                        order.stop_loss,
                                    )
                                position_state["runner_order_id"] = runner_id
                                runner_order_id = runner_id
                                LOG.info("Submitted scale-out orders %s and %s for %s", bracket_id, runner_id, symbol)
                    if await asyncio.to_thread(has_open_position, client, symbol):
                        if runner_order_id:
                            position_state["runner_order_id"] = runner_order_id
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
