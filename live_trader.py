"""Alpaca paper-trading execution for the daily SPY strategy."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from datetime import datetime, timedelta, timezone
from threading import RLock, Thread
from zoneinfo import ZoneInfo

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import TradeUpdate
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from alpaca.trading.stream import TradingStream
import pandas as pd

BOT_DIR = Path(__file__).resolve().parent
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from backtest import download_daily_bars, prepare_data
from models import ExitDecision, ExitReason, Order, Position, StrategyParams, strategy_params_for_mode
from pnl_manager import SymbolState
from signals import entry_limit_price, evaluate_exit, stop_loss_price

EASTERN = ZoneInfo("America/New_York")
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradeFill:
    id: str
    symbol: str
    side: str
    qty: float
    price: float
    transaction_time: datetime


_ACTIVITY_CACHE: dict[int, tuple[datetime, list[TradeFill]]] = {}
_SYMBOL_STATES: dict[str, SymbolState] = {}
_SYMBOL_STATE_LOCK = RLock()
_TRADE_STREAM: TradingStream | None = None
_TRADE_STREAM_THREAD: Thread | None = None
_TRADE_STREAM_LOCK = RLock()
_TRADE_UPDATE_HANDLERS: dict[str, list[Callable[[TradeUpdate, SymbolState], Awaitable[None]]]] = {}


def get_alpaca_client() -> TradingClient:
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ValueError("API keys not configured: set ALPACA_API_KEY and ALPACA_SECRET_KEY")
    return TradingClient(api_key=key, secret_key=secret, paper=True)


def get_account_value(client: TradingClient) -> float:
    return float(client.get_account().equity)


def cancel_symbol_orders(client: TradingClient, symbol: str) -> None:
    """Cancel open orders for one symbol without touching other strategies."""
    for order in _open_orders(client, symbol):
        if order.symbol == symbol:
            try:
                client.cancel_order_by_id(order.id)
            except APIError as exc:
                if exc.status_code not in {404, 422}:
                    raise


def close_symbol_position(client: TradingClient, symbol: str) -> bool:
    """Cancel this symbol's orders and close only this symbol's position."""
    cancel_symbol_orders(client, symbol)
    if not has_open_position(client, symbol):
        return False
    client.close_position(symbol)
    return True


def _parse_trade_fill(payload: Mapping) -> TradeFill:
    """Parse only fields required for FIFO, independent of optional Alpaca activity fields."""
    timestamp = payload.get("transaction_time") or payload.get("date")
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if not isinstance(timestamp, datetime):
        raise ValueError("Alpaca fill activity has no valid transaction timestamp")

    side = str(getattr(payload.get("side"), "value", payload.get("side", ""))).lower()
    if side not in {OrderSide.BUY.value, OrderSide.SELL.value}:
        raise ValueError("Alpaca fill activity has an invalid side")

    quantity = float(payload.get("qty", 0))
    price = float(payload.get("price", 0))
    symbol = str(payload.get("symbol", ""))
    if not symbol or not math.isfinite(quantity) or not math.isfinite(price) or quantity <= 0 or price <= 0:
        raise ValueError("Alpaca fill activity has invalid symbol, quantity, or price")

    activity_id = payload.get("id")
    if not activity_id:
        activity_id = f"{payload.get('order_id', symbol)}:{payload.get('cum_qty', quantity)}:{timestamp.isoformat()}"
    return TradeFill(str(activity_id), symbol, side, quantity, price, timestamp)


def _get_trade_activities(client: TradingClient) -> list[TradeFill]:
    now = datetime.now(timezone.utc)
    cached = _ACTIVITY_CACHE.get(id(client))
    if cached and now - cached[0] < timedelta(seconds=30):
        return cached[1]

    activities: list[TradeFill] = []
    page_token = None
    has_more_pages = True
    while has_more_pages:
        query = {"direction": "asc", "page_size": 100}
        if page_token:
            query["page_token"] = page_token
        response = client.get("/account/activities/FILL", query)
        if not isinstance(response, list):
            raise ValueError("Alpaca FILL activities response is not a list")
        if not response:
            break
        page = [_parse_trade_fill(payload) for payload in response if isinstance(payload, Mapping)]
        if len(page) != len(response):
            raise ValueError("Alpaca FILL activities response contains a non-object record")
        activities.extend(page)
        if len(response) < 100:
            has_more_pages = False
        else:
            page_token = str(response[-1].get("id") or page[-1].id)

    _ACTIVITY_CACHE[id(client)] = (now, activities)
    return activities


def get_symbol_daily_pnl(
    client: TradingClient,
    symbol: str,
    now: datetime | None = None,
    commission_pct: float = 0.0,
) -> float:
    """Return the SymbolState realized FIFO P&L for the current session day."""
    state = get_symbol_state(client, symbol, commission_pct)
    return float(state.daily_realized_pnl(now))


def get_symbol_state(
    client: TradingClient,
    symbol: str,
    commission_pct: float = 0.001,
) -> SymbolState:
    """Create a per-symbol ledger and replay historical fills once on first access."""
    with _SYMBOL_STATE_LOCK:
        state = _SYMBOL_STATES.get(symbol)
        if state is not None:
            return state
        state = SymbolState(symbol, commission_pct=commission_pct, timezone_=EASTERN)
        activities = sorted(
            (activity for activity in _get_trade_activities(client) if activity.symbol == symbol),
            key=lambda activity: activity.transaction_time,
        )
        for activity in activities:
            try:
                state.record_fill(
                    activity.side,
                    activity.qty,
                    activity.price,
                    activity.transaction_time,
                    execution_id=activity.id,
                )
            except ValueError:
                LOG.warning("Unable to replay unmatched %s fill for %s", activity.side, symbol)
        _SYMBOL_STATES[symbol] = state
        return state


def apply_trade_update(
    client: TradingClient,
    update: TradeUpdate,
    commission_pct: float = 0.001,
) -> SymbolState | None:
    """Apply new or partial execution events to the matching symbol ledger."""
    event = str(getattr(update.event, "value", update.event)).lower()
    if event not in {"fill", "partial_fill"}:
        return None
    order = update.order
    if update.qty is None or update.price is None:
        LOG.warning("Fill update for %s has no incremental quantity or price", order.symbol)
        return None
    state = get_symbol_state(client, order.symbol, commission_pct)
    execution_id = update.execution_id or (
        f"{order.id}:{order.filled_qty}:{update.timestamp.isoformat()}"
    )
    state.record_fill(
        order.side,
        update.qty,
        update.price,
        update.timestamp,
        execution_id=str(execution_id),
    )
    return state


def register_trade_update_handler(
    symbol: str,
    handler: Callable[[TradeUpdate, SymbolState], Awaitable[None]],
) -> None:
    with _TRADE_STREAM_LOCK:
        handlers = _TRADE_UPDATE_HANDLERS.setdefault(symbol, [])
        if handler not in handlers:
            handlers.append(handler)


def start_trade_update_stream(
    api_key: str,
    secret_key: str,
    client: TradingClient,
    symbols: list[str],
    commission_pct: float = 0.001,
) -> None:
    """Hydrate symbol ledgers, then listen for account fill events in one daemon thread."""
    global _TRADE_STREAM, _TRADE_STREAM_THREAD
    for symbol in symbols:
        get_symbol_state(client, symbol, commission_pct)

    with _TRADE_STREAM_LOCK:
        if _TRADE_STREAM_THREAD is not None and _TRADE_STREAM_THREAD.is_alive():
            return
        stream = TradingStream(api_key, secret_key, paper=True)

        async def handle_update(update: TradeUpdate) -> None:
            try:
                state = await asyncio.to_thread(apply_trade_update, client, update, commission_pct)
                if state is None:
                    return
                with _TRADE_STREAM_LOCK:
                    handlers = tuple(_TRADE_UPDATE_HANDLERS.get(update.order.symbol, ()))
                for handler in handlers:
                    await handler(update, state)
            except Exception:
                LOG.exception("Could not apply trade update for %s", update.order.symbol)

        stream.subscribe_trade_updates(handle_update)
        thread = Thread(target=stream.run, name="alpaca-trade-updates", daemon=True)
        _TRADE_STREAM = stream
        _TRADE_STREAM_THREAD = thread
        thread.start()


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
    qty: float,
    stop_loss_price: float,
    take_profit_price: float = None,
) -> str:
    if not (math.isfinite(limit_price) and math.isfinite(stop_loss_price) and math.isfinite(qty)) or not (0 < stop_loss_price < limit_price) or qty <= 0:
        raise ValueError("Invalid buy order price, stop or share count")
    limit = round(limit_price, 2)
    stop = round(stop_loss_price, 2)
    if not 0 < stop < limit:
        raise ValueError("Stop must be below the rounded limit price")

    if take_profit_price is not None:
        if not math.isfinite(take_profit_price) or take_profit_price <= limit:
            raise ValueError("Invalid take profit price")
        request = LimitOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            limit_price=limit, order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
            stop_loss=StopLossRequest(stop_price=stop),
        )
    else:
        request = LimitOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            limit_price=limit, order_class=OrderClass.OTO,
            take_profit=None, stop_loss=StopLossRequest(stop_price=stop),
        )
    return str(client.submit_order(order_data=request).id)


def place_market_sell(client: TradingClient, symbol: str, qty: float) -> str:
    if not math.isfinite(qty) or qty <= 0:
        raise ValueError("Quantity must be positive")
    request = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
    return str(client.submit_order(order_data=request).id)


def get_latest_bars(symbol: str, n: int = 60) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be positive")
    frame = download_daily_bars(symbol, lookback_days=730)
    if frame.empty:
        raise ValueError(f"No daily Alpaca bars returned for {symbol}")
    # Calculate the regime filter before truncating so its 200-bar history remains available.
    frame["SMA_200"] = frame["Close"].rolling(window=200, min_periods=200).mean()
    frame = frame.tail(n).copy()
    frame["Adj Close"] = frame["Close"]
    return frame[["Open", "High", "Low", "Close", "Adj Close", "Volume", "SMA_200"]]


def run_signal_check(symbol: str = "SPY", params: StrategyParams | None = None) -> Order | None:
    if params is None:
        params = strategy_params_for_mode(os.environ.get("BOT_MODE", "TREND_FOLLOWER"))
    bars = get_latest_bars(symbol)
    data = prepare_data(bars)
    data["SMA_200"] = bars["SMA_200"].reindex(data.index)
    today = datetime.now(EASTERN).date()
    finished = data.loc[pd.DatetimeIndex(data.index).date < today]
    if finished.empty:
        return None
    row = finished.iloc[-1]
    if params.bot_mode == "TREND_FOLLOWER" and (
        not math.isfinite(float(row["SMA_200"])) or float(row["Raw Close"]) <= float(row["SMA_200"])
    ):
        return None
    limit = entry_limit_price(row["Close"], row["BB_lower"], row["RSI"], row["ADX"], params.adx_max, params.rsi_max)
    if limit is None or not math.isfinite(row["ATR"]):
        return None
    stop = stop_loss_price(limit, row["ATR"], params.stop_loss_atr_mult)
    fractional_qty = round(params.trade_budget_usd / limit, 4)
    take_profit = limit + (float(row["ATR"]) * params.take_profit_atr_mult)
    if fractional_qty <= 0 or not (0 < round(stop, 2) < round(limit, 2)):
        return None
    if params.bot_mode == "TREND_FOLLOWER":
        shares_1 = round(fractional_qty / 2, 4)
        shares_2 = shares_1
    else:
        shares_1 = fractional_qty
        shares_2 = None
    return Order(
        created_date=finished.index[-1],
        limit_price=float(limit),
        stop_loss=float(stop),
        shares=fractional_qty,
        take_profit_price=float(take_profit),
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
                    shares=float(position.qty), entry_commission=0.0, peak_price=price)


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
                atr_stop = current_price - (float(current["ATR"]) * params.stop_loss_atr_mult)
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
        if order.symbol == symbol and order.side == OrderSide.SELL:
            client.cancel_order_by_id(order.id)
    if any(o.symbol == symbol and o.side == OrderSide.SELL for o in _open_orders(client, symbol)): return None
    pos = get_open_position(client, symbol)
    if pos is None: position_state.clear(); return None
    sell_id = place_market_sell(client, symbol, float(pos.qty))
    position_state["pending_exit"] = sell_id
    position_state["position"] = updated
    return decision


class LiveTradingEngine:
    """Event-driven single-symbol strategy runner backed by Alpaca WebSockets."""

    def __init__(self, symbol: str, params: StrategyParams, client: TradingClient | None = None) -> None:
        self.symbol = symbol
        self.params = params
        self.client = client or get_alpaca_client()
        self.api_key = os.environ["ALPACA_API_KEY"]
        self.secret_key = os.environ["ALPACA_SECRET_KEY"]
        feed = DataFeed(os.environ.get("ALPACA_DATA_FEED", "iex").lower())
        self.data_stream = StockDataStream(self.api_key, self.secret_key, feed=feed)
        self.symbol_state = get_symbol_state(self.client, symbol, params.commission_pct)
        self.position_state: dict = {}
        self.runner_order_id: str | None = None
        self.pending_entry_order_ids = {
            str(order.id)
            for order in _open_orders(self.client, self.symbol)
            if order.symbol == self.symbol and order.side == OrderSide.BUY
        }
        self.halted_day = None
        self.last_entry_day = None
        self.clock = None
        self._event_lock = asyncio.Lock()

    def start(self) -> None:
        self.data_stream.subscribe_daily_bars(self.on_daily_bar, self.symbol)
        self.data_stream.subscribe_bars(self.on_minute_bar, self.symbol)
        register_trade_update_handler(self.symbol, self.on_trade_update)
        start_trade_update_stream(
            self.api_key, self.secret_key, self.client, [self.symbol], self.params.commission_pct
        )

    @staticmethod
    def _report_task(task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            LOG.exception("WebSocket strategy task failed")

    def _schedule(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        task.add_done_callback(self._report_task)

    async def on_daily_bar(self, bar) -> None:
        if bar.symbol == self.symbol:
            self._schedule(self._process_daily_bar(bar))

    async def on_minute_bar(self, bar) -> None:
        if bar.symbol == self.symbol:
            self._schedule(self._process_minute_bar(bar))

    async def on_trade_update(self, update: TradeUpdate, state: SymbolState) -> None:
        if update.order.symbol != self.symbol:
            return
        async with self._event_lock:
            try:
                self._update_pending_entry_orders(update)
                if state is self.symbol_state:
                    await self._enforce_daily_limits(update.timestamp)
            except Exception:
                LOG.exception("Could not process %s trade update for %s", update.event, self.symbol)

    def _update_pending_entry_orders(self, update: TradeUpdate) -> None:
        if update.order.side != OrderSide.BUY:
            return
        order_id = str(update.order.id)
        event = str(getattr(update.event, "value", update.event)).lower()
        if event in {"new", "accepted", "pending_new", "partial_fill", "pending_replace", "replaced"}:
            self.pending_entry_order_ids.add(order_id)
        elif event in {"fill", "canceled", "expired", "rejected"}:
            self.pending_entry_order_ids.discard(order_id)

    async def _enforce_daily_limits(self, timestamp: datetime) -> bool:
        day = timestamp.astimezone(EASTERN).date()
        if self.halted_day == day:
            return True
        pnl = float(self.symbol_state.daily_realized_pnl(timestamp))
        if pnl <= -self.params.max_daily_drawdown_usd or pnl >= self.params.daily_target_usd:
            self.halted_day = day
            await asyncio.to_thread(close_symbol_position, self.client, self.symbol)
            self.pending_entry_order_ids.clear()
            self.position_state.clear()
            LOG.warning("Daily P&L limit reached for %s: $%.2f; bot halted for %s", self.symbol, pnl, day)
            return True
        return False

    async def _process_daily_bar(self, bar) -> None:
        async with self._event_lock:
            await self._process_daily_bar_locked(bar)

    async def _process_daily_bar_locked(self, bar) -> None:
        day = bar.timestamp.astimezone(EASTERN).date()
        if await self._enforce_daily_limits(bar.timestamp):
            return
        if self.halted_day is not None and self.halted_day != day:
            self.halted_day = None
        has_position = self.symbol_state.position_qty > 0
        if has_position:
            if "position" not in self.position_state:
                position = await asyncio.to_thread(_initial_position, self.client, self.symbol)
                if position is not None:
                    self.position_state["position"] = position
            if "position" in self.position_state:
                self.position_state["runner_order_id"] = self.runner_order_id
                decision = await asyncio.to_thread(
                    _check_exit_once, self.client, self.symbol, self.position_state, self.params
                )
                if decision is not None:
                    LOG.info("Exit %s: %s", self.symbol, decision.reason.value)
            return

        self.position_state.clear()

    async def _submit_signal_order(self) -> None:
        order = await asyncio.to_thread(run_signal_check, self.symbol, self.params)
        if order is None:
            return
        bracket_id = await asyncio.to_thread(
            place_limit_buy,
            self.client,
            self.symbol,
            order.limit_price,
            order.shares_1 or order.shares,
            order.stop_loss,
            order.take_profit_price,
        )
        runner_id = None
        if order.shares_2:
            runner_id = await asyncio.to_thread(
                place_limit_buy,
                self.client,
                self.symbol,
                order.limit_price,
                order.shares_2,
                order.stop_loss,
            )
        self.pending_entry_order_ids.add(bracket_id)
        if runner_id:
            self.pending_entry_order_ids.add(runner_id)
        self.runner_order_id = runner_id
        self.position_state["runner_order_id"] = runner_id
        LOG.info("Submitted orders %s and %s for %s", bracket_id, runner_id, self.symbol)

    async def _process_minute_bar(self, bar) -> None:
        async with self._event_lock:
            await self._process_minute_bar_locked(bar)

    async def _process_minute_bar_locked(self, bar) -> None:
        timestamp = bar.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        day = timestamp.astimezone(EASTERN).date()
        if self.halted_day == day:
            return
        clock_close = self.clock.next_close if self.clock is not None else None
        if clock_close is not None and clock_close.tzinfo is None:
            clock_close = clock_close.replace(tzinfo=timezone.utc)
        if clock_close is None or clock_close.astimezone(EASTERN).date() != day:
            self.clock = await asyncio.to_thread(self.client.get_clock)
        if not self.clock.is_open:
            return
        next_close = self.clock.next_close
        if next_close.tzinfo is None:
            next_close = next_close.replace(tzinfo=timezone.utc)
        minutes_to_close = (next_close - timestamp.astimezone(timezone.utc)).total_seconds() / 60
        if self.params.bot_mode == "DAILY_SCALPER" and minutes_to_close <= 15:
            self.halted_day = day
            await asyncio.to_thread(close_symbol_position, self.client, self.symbol)
            self.pending_entry_order_ids.clear()
            self.position_state.clear()
            LOG.info("Scalper closing-window lock for %s", self.symbol)
            return

        if self.params.bot_mode != "DAILY_SCALPER":
            local_timestamp = timestamp.astimezone(EASTERN)
            if local_timestamp.hour != 9 or local_timestamp.minute != 30:
                return
        if self.last_entry_day == day:
            return
        if self.symbol_state.position_qty > 0:
            return
        if self.pending_entry_order_ids:
            return
        self.last_entry_day = day
        await self._submit_signal_order()


def run_live_loop(symbol: str | None = None, params: StrategyParams | None = None) -> None:
    """Run a single-symbol strategy from Alpaca market-data and trade-update streams."""
    symbol = symbol or os.environ.get("TRADING_SYMBOL", "SPY")
    params = params or strategy_params_for_mode(os.environ.get("BOT_MODE", "TREND_FOLLOWER"))
    engine = LiveTradingEngine(symbol, params)
    engine.start()
    data_thread = Thread(target=engine.data_stream.run, name=f"alpaca-bars-{symbol}", daemon=True)
    data_thread.start()
    try:
        data_thread.join()
    except KeyboardInterrupt:
        engine.data_stream.stop()
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_live_loop()
