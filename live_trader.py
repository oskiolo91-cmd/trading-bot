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
from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.models import TradeUpdate
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.stream import TradingStream
import pandas as pd

BOT_DIR = Path(__file__).resolve().parent
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from backtest import download_daily_bars, prepare_data
from bot_state import (
    get_symbol_state as get_persisted_symbol_state,
    update_symbol_state as update_persisted_symbol_state,
)
from models import Order, Position, StrategyParams, strategy_params_for_mode
from pnl_manager import SymbolState
from signals import entry_limit_price, stop_loss_price

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
_TRAILING_FILL_LOCK = RLock()
_TRAILING_FILL_EXECUTIONS: set[str] = set()


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


def get_total_realized_pnl(client: TradingClient, commission_pct: float = 0.001) -> float:
    """Replay fills through the FIFO ledgers and return account-wide realized P&L."""
    symbols = {activity.symbol for activity in _get_trade_activities(client)}
    return sum(
        float(get_symbol_state(client, symbol, commission_pct).realized_pnl)
        for symbol in symbols
    )


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
) -> str:
    if not (math.isfinite(limit_price) and math.isfinite(qty)) or limit_price <= 0 or qty <= 0:
        raise ValueError("Invalid buy order price or share count")
    if not math.isclose(qty, round(qty), abs_tol=1e-9):
        raise ValueError("Native Alpaca trailing stops require whole-share entry quantities")
    limit = round(limit_price, 2)
    request = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        limit_price=limit,
    )
    order = client.submit_order(order_data=request)
    if not getattr(order, "id", None):
        raise RuntimeError("Alpaca accepted no identifiable buy order")
    return str(order.id)


def _trailing_stop_orders(client: TradingClient, symbol: str) -> list:
    return [
        order for order in _open_orders(client, symbol)
        if order.symbol == symbol
        and order.side == OrderSide.SELL
        and str(getattr(getattr(order, "type", None), "value", getattr(order, "type", ""))).lower()
        == OrderType.TRAILING_STOP.value
    ]


def replace_trailing_stop_percent(client: TradingClient, symbol: str, trailing_pct: float) -> int:
    """Update every open native trailing order; Alpaca's `trail` is in percentage points."""
    if not math.isfinite(trailing_pct) or not 0 < trailing_pct < 1:
        raise ValueError("Trailing percentage must be between 0 and 1")
    trail_percent = round(trailing_pct * 100, 2)
    orders = _trailing_stop_orders(client, symbol)
    for order in orders:
        client.replace_order_by_id(order.id, ReplaceOrderRequest(trail=trail_percent))
    return len(orders)


def ensure_native_trailing_stop(
    client: TradingClient,
    symbol: str,
    trailing_pct: float,
    target_qty: float | None = None,
) -> str | None:
    """Cover an open long position with a broker-managed trailing stop."""
    if not math.isfinite(trailing_pct) or not 0 < trailing_pct < 1:
        raise ValueError("Trailing percentage must be between 0 and 1")
    position = get_open_position(client, symbol)
    if target_qty is None:
        if position is None:
            return None
        target_qty = abs(float(position.qty))
    if not math.isfinite(target_qty) or target_qty <= 0:
        return None
    if not math.isclose(target_qty, round(target_qty), abs_tol=1e-9):
        raise ValueError(
            f"Alpaca trailing stops do not support fractional shares; {symbol} remains unchanged"
        )

    open_orders = _open_orders(client, symbol)
    trailing_orders = [
        order for order in open_orders
        if order.symbol == symbol
        and order.side == OrderSide.SELL
        and str(getattr(getattr(order, "type", None), "value", getattr(order, "type", ""))).lower()
        == OrderType.TRAILING_STOP.value
    ]
    covered_qty = sum(
        max(
            0.0,
            float(getattr(order, "qty", 0) or 0)
            - float(getattr(order, "filled_qty", 0) or 0),
        )
        for order in trailing_orders
    )
    trail_percent = round(trailing_pct * 100, 2)
    for order in trailing_orders:
        current_trail = getattr(order, "trail_percent", None)
        if current_trail is not None and not math.isclose(float(current_trail), trail_percent):
            client.replace_order_by_id(order.id, ReplaceOrderRequest(trail=trail_percent))
    remaining_qty = round(target_qty - covered_qty, 8)
    if remaining_qty <= 0:
        return str(trailing_orders[0].id) if trailing_orders else None

    legacy_sell_orders = [
        order for order in open_orders
        if order.symbol == symbol
        and order.side == OrderSide.SELL
        and str(getattr(getattr(order, "type", None), "value", getattr(order, "type", ""))).lower()
        != OrderType.TRAILING_STOP.value
    ]
    if legacy_sell_orders:
        raise RuntimeError(
            f"Legacy sell protection for {symbol} was left untouched; reconcile it before adding a trailing stop"
        )

    request = TrailingStopOrderRequest(
        symbol=symbol,
        qty=remaining_qty,
        side=OrderSide.SELL,
        type=OrderType.TRAILING_STOP,
        time_in_force=TimeInForce.GTC,
        trail_percent=round(trailing_pct * 100, 2),
    )
    order = client.submit_order(order_data=request)
    if not getattr(order, "id", None):
        raise RuntimeError("Alpaca accepted no identifiable trailing-stop order")
    return str(order.id)


async def attach_trailing_stop_on_fill(
    client: TradingClient,
    update: TradeUpdate,
    trailing_pct: float,
) -> None:
    """Attach native protection once per buy execution, including partial fills."""
    event = str(getattr(update.event, "value", update.event)).lower()
    if update.order.side != OrderSide.BUY or event not in {"fill", "partial_fill"}:
        return
    execution_id = str(
        update.execution_id
        or f"{update.order.id}:{update.order.filled_qty}:{update.timestamp.isoformat()}"
    )
    with _TRAILING_FILL_LOCK:
        if execution_id in _TRAILING_FILL_EXECUTIONS:
            return
        _TRAILING_FILL_EXECUTIONS.add(execution_id)
    try:
        await asyncio.to_thread(
            ensure_native_trailing_stop,
            client,
            update.order.symbol,
            trailing_pct,
        )
    except Exception:
        with _TRAILING_FILL_LOCK:
            _TRAILING_FILL_EXECUTIONS.discard(execution_id)
        raise


def place_market_sell(client: TradingClient, symbol: str, qty: float) -> str:
    if not math.isfinite(qty) or qty <= 0:
        raise ValueError("Quantity must be positive")
    request = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
    return str(client.submit_order(order_data=request).id)


def whole_share_quantity(budget: float, price: float) -> int:
    """Size buys to whole shares because Alpaca trailing-stop orders cannot protect fractions."""
    if not math.isfinite(budget) or not math.isfinite(price) or budget <= 0 or price <= 0:
        return 0
    return math.floor(budget / price)


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
    fractional_qty = whole_share_quantity(params.trade_budget_usd, limit)
    if fractional_qty < 1:
        return None
    take_profit = limit + (float(row["ATR"]) * params.take_profit_atr_mult)
    if fractional_qty <= 0 or not (0 < round(stop, 2) < round(limit, 2)):
        return None
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


def recover_high_water_mark(symbol: str, entry_date, entry_price: float) -> float:
    """Rebuild a missing local HWM from Alpaca daily highs since the position entry."""
    entry_timestamp = pd.Timestamp(entry_date)
    now = pd.Timestamp.now(tz="UTC")
    days = max(2, (now.date() - entry_timestamp.date()).days + 2)
    try:
        bars = download_daily_bars(symbol, lookback_days=days)
        if bars.empty:
            raise ValueError(f"No historical price bars returned for {symbol}")
        bar_dates = pd.DatetimeIndex(bars.index)
        entry_utc = entry_timestamp.tz_localize("UTC") if entry_timestamp.tzinfo is None else entry_timestamp.tz_convert("UTC")
        highs = pd.to_numeric(bars.loc[bar_dates >= entry_utc, "High"], errors="coerce").dropna()
        peak = float(highs.max()) if not highs.empty else float(entry_price)
        return max(float(entry_price), peak)
    except Exception:
        LOG.exception("Could not recover high-water mark for %s; using average entry price", symbol)
        return float(entry_price)


def _initial_position(
    client: TradingClient,
    symbol: str,
    high_water_mark: float | None = None,
) -> Position | None:
    position = get_open_position(client, symbol)
    if position is None or float(position.qty) <= 0:
        return None
    stops = [
        order for order in _open_orders(client, symbol)
        if order.side == OrderSide.SELL
        and (
            getattr(order, "stop_price", None) is not None
            or str(getattr(getattr(order, "type", None), "value", getattr(order, "type", ""))).lower()
            == OrderType.TRAILING_STOP.value
        )
    ]
    if not stops:
        LOG.warning("No protective stop found for %s; skipping unmanaged position", symbol)
        return None
    price = float(position.avg_entry_price)
    filled = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=100, nested=False, symbols=[symbol]))
    buys = [order for order in filled if order.symbol == symbol and order.side == OrderSide.BUY and getattr(order, "filled_at", None)]
    entry_date = pd.Timestamp(max(buys, key=lambda order: order.filled_at).filled_at) if buys else pd.Timestamp.now(tz=EASTERN)
    try:
        saved_peak = float(high_water_mark)
    except (TypeError, ValueError):
        saved_peak = float("nan")
    peak_price = (
        max(price, saved_peak)
        if math.isfinite(saved_peak) and saved_peak > 0
        else recover_high_water_mark(symbol, entry_date, price)
    )
    broker_stop = getattr(stops[0], "stop_price", None)
    return Position(entry_date=entry_date, entry_price=price, stop_loss=float(broker_stop or price),
                    shares=float(position.qty), entry_commission=0.0, peak_price=peak_price)


def _check_exit_once(client, symbol, position_state, params):
    if not has_open_position(client, symbol):
        position_state.clear()
        return None
    position = get_open_position(client, symbol)
    ensure_native_trailing_stop(
        client,
        symbol,
        params.trailing_pct,
        abs(float(position.qty)) if position is not None else None,
    )
    return None


class LiveTradingEngine:
    """Event-driven single-symbol strategy runner backed by Alpaca WebSockets."""

    def __init__(self, symbol: str, params: StrategyParams, client: TradingClient | None = None) -> None:
        self.symbol = symbol
        self.params = params
        persisted = get_persisted_symbol_state(symbol)
        stored_trail = persisted.get("trailing_pct")
        if stored_trail is not None:
            try:
                stored_trail = float(stored_trail)
            except (TypeError, ValueError):
                stored_trail = float("nan")
            if math.isfinite(stored_trail) and 0 < stored_trail < 1:
                self.params = replace(params, trailing_pct=stored_trail)
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
        saved_buy_date = persisted.get("last_buy_date")
        try:
            self.last_entry_day = datetime.fromisoformat(saved_buy_date).date() if saved_buy_date else None
        except (TypeError, ValueError):
            LOG.warning("Ignoring invalid persisted last-buy date for %s", symbol)
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
                await attach_trailing_stop_on_fill(self.client, update, self.params.trailing_pct)
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
                open_position = await asyncio.to_thread(get_open_position, self.client, self.symbol)
                if open_position is not None:
                    await asyncio.to_thread(
                        ensure_native_trailing_stop,
                        self.client,
                        self.symbol,
                        self.params.trailing_pct,
                        abs(float(open_position.qty)),
                    )
                persisted = get_persisted_symbol_state(self.symbol)
                position = await asyncio.to_thread(
                    _initial_position,
                    self.client,
                    self.symbol,
                    persisted.get("high_water_mark"),
                )
                if position is not None:
                    self.position_state["position"] = position
                    update_persisted_symbol_state(
                        self.symbol,
                        {"high_water_mark": float(position.peak_price), "trailing_pct": self.params.trailing_pct},
                    )
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
        )
        self.pending_entry_order_ids.add(bracket_id)
        runner_id = None
        if order.shares_2:
            runner_id = await asyncio.to_thread(
                place_limit_buy,
                self.client,
                self.symbol,
                order.limit_price,
                order.shares_2,
            )
            self.pending_entry_order_ids.add(runner_id)
        self.runner_order_id = runner_id
        self.position_state["runner_order_id"] = runner_id
        LOG.info("Submitted entry orders %s and %s for %s; trailing protection follows fill", bracket_id, runner_id, self.symbol)

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
        if self.symbol_state.position_qty > 0:
            position = await asyncio.to_thread(get_open_position, self.client, self.symbol)
            if position is not None:
                await asyncio.to_thread(
                    ensure_native_trailing_stop,
                    self.client,
                    self.symbol,
                    self.params.trailing_pct,
                    abs(float(position.qty)),
                )
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
        try:
            await self._submit_signal_order()
        finally:
            if self.pending_entry_order_ids:
                self.last_entry_day = day
                update_persisted_symbol_state(
                    self.symbol,
                    {"last_buy_date": day.isoformat(), "trailing_pct": self.params.trailing_pct},
                )


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
