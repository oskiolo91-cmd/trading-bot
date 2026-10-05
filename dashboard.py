"""Watchlist trading dashboard: live indicators, per-ticker bots and manual paper orders."""

from __future__ import annotations

import asyncio
from contextlib import closing
import math
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus, OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetAssetsRequest, GetOrdersRequest
from alpaca.trading.enums import OrderType
from bot_state import (
    STATE_PATH as BOT_STATE_DB,
    connect_db,
    get_symbol_state as get_persisted_symbol_state,
    load_bot_state,
    update_symbol_state,
)


BOT_DIR = Path(__file__).resolve().parent
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))


def load_market_radar_data(database_path: str | Path = BOT_STATE_DB) -> pd.DataFrame:
    """Load the screener table and join persisted per-symbol bot flags."""
    with closing(connect_db(database_path)) as connection:
        try:
            radar = pd.read_sql_query("SELECT * FROM market_radar", connection)
        except pd.errors.DatabaseError:
            return pd.DataFrame()
        try:
            bot_flags = pd.read_sql_query(
                "SELECT symbol, bot_enabled, active_ticker FROM tickers_state", connection
            )
        except pd.errors.DatabaseError:
            bot_flags = pd.DataFrame(columns=["symbol", "bot_enabled", "active_ticker"])

    if "Symbol" not in radar.columns:
        return radar
    if bot_flags.empty:
        radar["bot_enabled"] = False
        radar["active_ticker"] = False
        return radar
    joined = radar.merge(
        bot_flags,
        how="left",
        left_on="Symbol",
        right_on="symbol",
    ).drop(columns="symbol")
    for column in ("bot_enabled", "active_ticker"):
        joined[column] = joined[column].fillna(0).astype(bool)
    return joined

from backtest import BacktestResult, download_daily_bars, prepare_data, run_backtest
from alpaca_data import fetch_daily_bars
from models import Position, StrategyParams
from live_trader import (
    _initial_position, _open_orders, _trailing_stop_orders,
    attach_trailing_stop_on_fill, close_symbol_position, ensure_native_trailing_stop,
    get_account_value, get_alpaca_client, get_symbol_daily_pnl,
    get_latest_bars, get_recent_orders, place_limit_buy, place_market_sell,
    register_trade_update_handler, replace_trailing_stop_percent, run_signal_check,
    start_trade_update_stream, whole_share_quantity,
)

DEFAULT_SYMBOLS = ("SPY", "QQQ")
REFRESH_SECONDS = 60
REFRESH_WORKERS = 8
APP_VERSION = os.environ.get("BOT_VERSION", "v2.1")
LOG_LIMIT = 200
ROW_WIDTHS = [1.6, 1, 1, 1.2, 1.5, 1.8]
PARAMS = StrategyParams()
PROFILES = {
    "🐢 Conservativo": StrategyParams(adx_max=20, rsi_max=30, trade_budget_usd=50, stop_loss_atr_mult=2.5, take_profit_atr_mult=3.0, daily_target_usd=2.0, trailing_pct=0.03),
    "⚖️ Bilanciato": StrategyParams(adx_max=25, rsi_max=35, trade_budget_usd=100, stop_loss_atr_mult=2.0, take_profit_atr_mult=3.0, daily_target_usd=5.0, trailing_pct=0.06),
    "🚀 Speculativo": StrategyParams(adx_max=35, rsi_max=45, trade_budget_usd=200, stop_loss_atr_mult=1.5, take_profit_atr_mult=2.5, daily_target_usd=10.0, trailing_pct=0.12),
    "🎛️ Custom": None,
}
PROFILE_NAMES = list(PROFILES.keys())
DEFAULT_PROFILE = "⚖️ Bilanciato"
CUSTOM_PROFILE = "🎛️ Custom"
BACKTEST_PERIOD = "90d"
EASTERN = ZoneInfo("America/New_York")
_ACTIVE_TRADING_CLIENT = None
_REGISTERED_TRAILING_SYMBOLS: set[str] = set()


@st.cache_resource
def get_ticker_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=REFRESH_WORKERS, thread_name_prefix="ticker-refresh")


def _is_active_tradable_equity(asset) -> bool:
    return (
        getattr(asset.status, "value", asset.status) == AssetStatus.ACTIVE.value
        and bool(getattr(asset, "tradable", False))
        and getattr(asset.asset_class, "value", asset.asset_class) == AssetClass.US_EQUITY.value
    )


def filter_tradable_equity_assets(assets) -> list[str]:
    return sorted({
        asset.symbol
        for asset in assets
        if _is_active_tradable_equity(asset)
    })


def filter_fractional_assets(assets) -> list[str]:
    return sorted({
        asset.symbol
        for asset in assets
        if _is_active_tradable_equity(asset)
        and getattr(asset, "fractionable", getattr(asset, "fractional_enabled", False))
    })


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_active_equity_symbols(_trading_client, credential_scope: str) -> tuple[list[str], list[str]]:
    asset_filter = GetAssetsRequest(
        status=AssetStatus.ACTIVE,
        asset_class=AssetClass.US_EQUITY,
    )
    assets = _trading_client.get_all_assets(filter=asset_filter)
    return filter_tradable_equity_assets(assets), filter_fractional_assets(assets)


def get_tradable_equity_symbols(trading_client, api_key: str, secret_key: str) -> list[str]:
    scope = hashlib.sha256(f"{api_key}:{secret_key}".encode()).hexdigest()
    available, _ = _cached_active_equity_symbols(trading_client, scope)
    return available


def get_fractional_asset_symbols(trading_client, api_key: str, secret_key: str) -> list[str]:
    scope = hashlib.sha256(f"{api_key}:{secret_key}".encode()).hexdigest()
    _, fractional = _cached_active_equity_symbols(trading_client, scope)
    return fractional


NO_KEYS_MESSAGE = (
    "Chiavi Alpaca non configurate: aggiungi ALPACA_API_KEY e ALPACA_SECRET_KEY nei Secrets "
    "dell'app (Streamlit Cloud → Settings → Secrets). Bot e ordini sono disattivati, "
    "la watchlist resta consultabile."
)


def to_alpaca_symbol(symbol: str) -> str:
    """Yahoo uses BRK-B, Alpaca uses BRK.B."""
    return symbol.replace("-", ".")


def fetch_ticker_data(symbol: str) -> dict:
    alpaca_symbol = to_alpaca_symbol(symbol)
    frame = download_daily_bars(alpaca_symbol, lookback_days=400, end=datetime.now(timezone.utc))[alpaca_symbol]
    return _ticker_data_from_frame(symbol, frame)


def _ticker_data_from_frame(symbol: str, frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError(f"Nessuna barra Alpaca per {symbol}")
    frame = frame.copy()
    # Orders are placed on the raw market tape, so indicators use unadjusted prices (as live_trader does).
    frame["Adj Close"] = frame["Close"]
    data = prepare_data(frame)
    if data.empty:
        raise ValueError(f"Nessuna candela valida per {symbol}")
    data["SMA_200"] = data["Close"].rolling(window=200, min_periods=200).mean()
    row = data.iloc[-1]
    values = {
        "price": float(row["Close"]),
        "prev_close": float(data["Close"].iloc[-2]) if len(data) > 1 else float("nan"),
        "adx": float(row["ADX"]), "rsi": float(row["RSI"]),
        "bb_lower": float(row["BB_lower"]), "bb_mid": float(row["BB_mid"]), "bb_upper": float(row["BB_upper"]),
        "atr": float(row["ATR"]),
        "chart_data": data.tail(260),
    }
    values["signal"] = compute_signal(values, PARAMS)
    return values


@st.cache_data(ttl=900, show_spinner=False)
def _cached_market_detail_data(symbol: str, credential_scope: str, _data_client) -> dict:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=100)
    alpaca_symbol = to_alpaca_symbol(symbol)
    frame = fetch_daily_bars(alpaca_symbol, start, end, client=_data_client)[alpaca_symbol]
    return _ticker_data_from_frame(symbol, frame)


def compute_signal(data: dict, params: StrategyParams) -> bool:
    """Entry signal (ADX < max, RSI < max, price <= BB lower) evaluated with the given profile."""
    values = [data.get(name) for name in ("price", "adx", "rsi", "bb_lower")]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return False
    price, adx, rsi, bb_lower = values
    return bool(adx < params.adx_max and rsi < params.rsi_max and price <= bb_lower)


def get_ticker_params(symbol: str, state=None) -> StrategyParams:
    """Strategy params of the risk profile selected for ``symbol`` (custom profile reads its sliders)."""
    state = st.session_state if state is None else state
    name = state.get("profile", {}).get(symbol, DEFAULT_PROFILE)
    params = PROFILES.get(name, PROFILES[DEFAULT_PROFILE])
    if params is None:
        base = PROFILES[DEFAULT_PROFILE]
        params = replace(
            base,
            adx_max=float(state.get(f"custom_adx_{symbol}", base.adx_max)),
            rsi_max=float(state.get(f"custom_rsi_{symbol}", base.rsi_max)),
            trade_budget_usd=float(state.get(f"custom_budget_{symbol}", base.trade_budget_usd)),
            stop_loss_atr_mult=float(state.get(f"custom_stop_atr_{symbol}", base.stop_loss_atr_mult)),
            take_profit_atr_mult=float(state.get(f"custom_take_profit_atr_{symbol}", base.take_profit_atr_mult)),
            trailing_pct=float(state.get(f"custom_trailing_pct_{symbol}", base.trailing_pct)),
            daily_target_usd=float(state.get(f"custom_daily_target_{symbol}", base.daily_target_usd)),
            max_daily_drawdown_usd=float(state.get(f"custom_max_drawdown_{symbol}", base.max_daily_drawdown_usd)),
        )
    trailing_override = state.get("trailing_stop_pct", {}).get(symbol)
    if trailing_override is not None:
        params = replace(params, trailing_pct=float(trailing_override))
    return params


def persist_symbol_settings(state: dict, symbol: str) -> None:
    """Persist the currently selected profile, custom risk values, HWM, and daily entry guard."""
    custom_keys = (
        "custom_adx", "custom_rsi", "custom_budget", "custom_stop_atr",
        "custom_take_profit_atr", "custom_trailing_pct", "custom_daily_target",
        "custom_max_drawdown",
    )
    record = {
        "profile": state.get("profile", {}).get(symbol, DEFAULT_PROFILE),
        "custom_trailing_pct": state.get("trailing_stop_pct", {}).get(symbol),
        "bot_enabled": bool(state.get("bot_enabled", {}).get(symbol, False)),
        "active_ticker": symbol in state.get("active_tickers", []),
        "custom_settings": {
            key: float(state[f"{key}_{symbol}"])
            for key in custom_keys
            if f"{key}_{symbol}" in state
        },
    }
    entry_override = state.get("bot_entry_price_overrides", {}).get(symbol)
    if entry_override is not None:
        record["custom_settings"]["bot_entry_price_override"] = float(entry_override)
    symbol_state = state.get("bot_state", {}).get(symbol, {})
    managed_position = symbol_state.get("position", symbol_state.get("base"))
    if isinstance(managed_position, Position):
        record["high_water_mark"] = float(managed_position.peak_price)
    last_buy = state.get("bot_last_buy", {}).get(symbol)
    if last_buy is not None:
        record["last_buy_date"] = last_buy.isoformat() if hasattr(last_buy, "isoformat") else str(last_buy)
    update_symbol_state(symbol, record)


async def _attach_dashboard_trailing_stop(update, _symbol_state) -> None:
    client = _ACTIVE_TRADING_CLIENT
    if client is None:
        raise RuntimeError("Alpaca client is unavailable while handling a buy fill")
    symbol = update.order.symbol
    persisted = get_persisted_symbol_state(_display_symbol(symbol))
    trailing_pct = persisted.get("trailing_pct", PROFILES[DEFAULT_PROFILE].trailing_pct)
    await attach_trailing_stop_on_fill(client, update, float(trailing_pct))


def run_profile_backtest(symbol: str) -> dict[str, BacktestResult]:
    """Run backtest for all 3 preset profiles on 90 days of data."""
    alpaca_symbol = to_alpaca_symbol(symbol)
    frame = download_daily_bars(alpaca_symbol, lookback_days=90, end=datetime.now(timezone.utc))[alpaca_symbol]
    if frame.empty:
        raise ValueError(f"Nessuna barra Alpaca per {symbol}")
    frame = frame.copy()
    frame["Adj Close"] = frame["Close"]
    data = prepare_data(frame)
    results = {}
    for name, params in PROFILES.items():
        if params is None:
            continue
        try:
            results[name] = run_backtest(data, params, symbol=symbol)
        except Exception:
            pass
    return results


def backtest_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        metrics = result.metrics
        rows.append({
            "Profilo": name,
            "Rendimento %": round(metrics["return_pct"] * 100, 2),
            "Win rate %": round(metrics["win_rate"] * 100, 1),
            "Max drawdown %": round(metrics["max_drawdown"] * 100, 2),
            "N. trade": int(metrics["total_trades"]),
        })
    return pd.DataFrame(rows, columns=["Profilo", "Rendimento %", "Win rate %", "Max drawdown %", "N. trade"])


def render_candlestick_chart(
    client,
    symbol: str,
    ticker_data: dict,
    params: StrategyParams,
    chart_key: str | None = None,
) -> None:
    data = ticker_data.get("chart_data")
    if not isinstance(data, pd.DataFrame) or data.empty:
        st.info(f"Grafico di {symbol} in attesa delle barre Alpaca.")
        return

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
    )
    figure.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )
    for column, label, color in (
        ("BB_upper", "Bollinger upper", "#d9822b"),
        ("BB_mid", "Bollinger mid", "#687386"),
        ("BB_lower", "Bollinger lower", "#d9822b"),
    ):
        figure.add_trace(
            go.Scatter(x=data.index, y=data[column], name=label, line={"color": color, "width": 1}),
            row=1,
            col=1,
        )
    if params.bot_mode == "TREND_FOLLOWER" and data["SMA_200"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=data.index,
                y=data["SMA_200"],
                name="SMA 200",
                line={"color": "#2878b5", "width": 1.5},
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color="#91a3b0"),
        row=2,
        col=1,
    )

    if client is not None:
        alpaca_symbol = to_alpaca_symbol(symbol)
        try:
            orders = client.get_orders(
                filter=GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    limit=500,
                    nested=False,
                    symbols=[alpaca_symbol],
                    side=OrderSide.BUY,
                )
            )
            entries = [
                order for order in orders
                if order.symbol == alpaca_symbol
                and order.side == OrderSide.BUY
                and order.filled_at is not None
                and float(order.filled_qty or 0) > 0
                and order.filled_avg_price is not None
            ]
            if entries:
                figure.add_trace(
                    go.Scatter(
                        x=[order.filled_at for order in entries],
                        y=[float(order.filled_avg_price) for order in entries],
                        mode="markers",
                        name="Entry eseguite",
                        marker={"symbol": "triangle-up", "size": 11, "color": "#16834a"},
                        customdata=[str(order.id) for order in entries],
                        hovertemplate="Entry %{y:.2f}<br>%{x}<br>Ordine %{customdata}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
        except Exception as exc:
            st.warning(f"Storico entry non disponibile per {symbol}: {exc}")

        try:
            sell_orders = [
                order for order in _open_orders(client, alpaca_symbol)
                if order.symbol == alpaca_symbol
                and order.side == OrderSide.SELL
            ]
            for order in sell_orders:
                stop_price = getattr(order, "stop_price", None)
                if stop_price is not None:
                    figure.add_hline(
                        y=float(stop_price),
                        line_dash="dash",
                        line_color="#c83232",
                        annotation_text="Stop / trailing",
                        row=1,
                        col=1,
                    )
                target_price = getattr(order, "limit_price", None)
                if target_price is not None:
                    figure.add_hline(
                        y=float(target_price),
                        line_dash="dot",
                        line_color="#16834a",
                        annotation_text="Take profit",
                        row=1,
                        col=1,
                    )
        except Exception as exc:
            st.warning(f"Stop attivo non disponibile per {symbol}: {exc}")

    figure.update_layout(
        title=f"{symbol} · candele daily",
        height=620,
        autosize=True,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 10, "r": 20, "t": 75, "b": 10},
    )
    figure.update_yaxes(title_text="Prezzo (USD)", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(
        figure,
        width="stretch",
        config={"responsive": True, "displaylogo": False},
        key=chart_key or f"candlestick_{symbol}",
    )


def _display_symbol(alpaca_symbol: str) -> str:
    return alpaca_symbol


def get_active_symbols(selected_symbols: list[str]) -> list[str]:
    return list(dict.fromkeys(symbol for symbol in selected_symbols if symbol))


def prepare_ticker_selection(
    available_symbols: list[str],
    current_selection: list[str] | None,
    positions: dict,
    bot_enabled: dict[str, bool],
    active_tickers: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    available = set(available_symbols)
    available_by_alpaca = {to_alpaca_symbol(symbol): symbol for symbol in available_symbols}
    if current_selection is None:
        selected = [symbol for symbol in DEFAULT_SYMBOLS if symbol in available]
        if not selected:
            selected = available_symbols[:2]
    else:
        selected = [symbol for symbol in current_selection if symbol in available]

    required = [
        symbol for symbol, enabled in bot_enabled.items()
        if enabled and symbol in available
    ]
    required.extend(symbol for symbol in (active_tickers or []) if symbol in available)
    required.extend(available_by_alpaca[symbol] for symbol in positions if symbol in available_by_alpaca)
    required = list(dict.fromkeys(required))
    missing = [symbol for symbol in required if symbol not in selected]
    return list(dict.fromkeys([*selected, *required])), missing


def render_symbol_card(client, equity: float | None, symbol: str, positions: dict) -> None:
    data = st.session_state["live_data"].get(symbol, {})
    st.session_state["profile"].setdefault(symbol, DEFAULT_PROFILE)
    st.session_state["bot_enabled"].setdefault(symbol, False)
    st.session_state.setdefault(f"profile_{symbol}", st.session_state["profile"][symbol])
    st.session_state.setdefault(f"bot_{symbol}", st.session_state["bot_enabled"][symbol])
    params = get_ticker_params(symbol)
    position = positions.get(to_alpaca_symbol(symbol))
    trading_ok = client is not None and equity is not None

    with st.container(border=True):
        price = data.get("price")
        change = data.get("prev_close")
        price_delta = (price / change - 1) * 100 if price and change else None
        header = st.columns([1.1, 2.4, 1.2], vertical_alignment="center")
        header[0].markdown(f"### {symbol}")
        header[0].metric(
            "Prezzo",
            f"${price:,.2f}" if isinstance(price, (int, float)) else "—",
            delta=f"{price_delta:+.2f}%" if price_delta is not None else None,
        )
        if trading_ok:
            with header[1]:
                render_symbol_risk_status(client, symbol, data, params)
            header[2].button(
                "🛑 KILL SIMBOLO",
                key=f"card_kill_{symbol}",
                type="primary",
                on_click=_on_kill_symbol,
                args=(client, symbol),
                disabled=client is None,
            )
        else:
            header[1].caption("P&L non disponibile senza connessione Alpaca.")

        if position is not None:
            pnl = float(position.unrealized_pl or 0)
            header[0].caption(f"Posizione {float(position.qty):.4f} · P&L non realizzato ${pnl:+.2f}")

        profile_controls = st.columns([1.5, 1], vertical_alignment="center")
        profile_controls[0].selectbox(
            "Profilo",
            PROFILE_NAMES,
            key=f"profile_{symbol}",
            on_change=_on_profile_change,
            args=(symbol, client),
        )
        profile_controls[1].toggle(
            "Bot",
            key=f"bot_{symbol}",
            disabled=not trading_ok,
            on_change=_on_toggle,
            args=(symbol,),
        )

        default_limit = max(0.01, float(data.get("bb_lower", price or 0.01)))
        has_quote = all(
            isinstance(data.get(key), (int, float)) and math.isfinite(data[key])
            for key in ("price", "atr", "bb_lower")
        )
        if has_quote and not st.session_state.get(f"buy_defaults_loaded_{symbol}"):
            st.session_state[f"buy_limit_{symbol}"] = default_limit
            st.session_state[f"buy_defaults_loaded_{symbol}"] = True
        st.session_state.setdefault(f"buy_limit_{symbol}", 0.01)
        st.session_state.setdefault(f"buy_budget_{symbol}", params.trade_budget_usd)
        st.session_state.setdefault(f"buy_confirm_{symbol}", False)
        order_controls = st.columns([1, 1, 1, 1], vertical_alignment="bottom")
        order_controls[0].number_input(
            "Budget (USD)", min_value=1.0, step=25.0, key=f"buy_budget_{symbol}"
        )
        order_controls[1].number_input(
            "Prezzo limite ($)", min_value=0.01, step=0.01, key=f"buy_limit_{symbol}"
        )
        limit_price = st.session_state[f"buy_limit_{symbol}"]
        estimated_qty = (
            whole_share_quantity(st.session_state[f"buy_budget_{symbol}"], limit_price)
            if has_quote and limit_price > 0
            else None
        )
        order_controls[2].caption(
            f"Quantità: {estimated_qty:.4f} az." if estimated_qty is not None else "Quantità: in attesa dei dati"
        )
        confirmed = order_controls[3].checkbox("Conferma", key=f"card_confirm_{symbol}")
        order_controls[3].button(
            "Invia ordine",
            key=f"card_buy_{symbol}",
            on_click=_on_submit_buy,
            args=(client, symbol),
            disabled=(
                not trading_ok
                or not has_quote
                or position is not None
                or estimated_qty is None
                or estimated_qty <= 0
                or not confirmed
            ),
            type="primary",
        )

        message = st.session_state["panel_msg"].get(symbol)
        if message:
            (st.success if message[0] == "success" else st.error)(message[1])
        if data.get("chart_data") is not None:
            render_candlestick_chart(client, symbol, data, params)
        elif data.get("error"):
            st.warning(f"Dati di {symbol} non disponibili: {data['error']}")
        else:
            st.info(f"Grafico di {symbol} in attesa delle barre Alpaca.")


def _fetch_ticker_result(symbol: str) -> dict:
    try:
        return {**fetch_ticker_data(symbol), "last_updated": datetime.now()}
    except Exception as exc:
        return {"error": str(exc), "signal": False, "last_updated": datetime.now()}


def fetch_all_tickers(symbols: list, on_progress=None) -> dict:
    results = _fetch_ticker_batch(symbols)
    for index, symbol in enumerate(symbols, start=1):
        if on_progress is not None:
            on_progress(index, len(symbols), symbol)
    return results


def _fetch_ticker_batch(symbols: list[str]) -> dict[str, dict]:
    return _cached_ticker_batch(tuple(symbols), _alpaca_cache_scope())


def _alpaca_cache_scope() -> str:
    api_key, secret_key, _ = _live_credentials()
    return hashlib.sha256(f"{api_key or ''}:{secret_key or ''}".encode()).hexdigest()


@st.cache_data(ttl=60, show_spinner=False)
def _cached_ticker_batch(
    symbols: tuple[str, ...],
    credential_scope: str,
) -> dict[str, dict]:
    """Fetch one Alpaca OHLCV batch, cached briefly per credential scope."""
    _ = credential_scope
    alpaca_symbols = [to_alpaca_symbol(symbol) for symbol in symbols]
    try:
        end = datetime.now(timezone.utc)
        frames = fetch_daily_bars(alpaca_symbols, start=end - timedelta(days=400), end=end)
    except Exception as exc:
        return {symbol: {"error": str(exc), "signal": False, "last_updated": datetime.now()} for symbol in symbols}
    results = {}
    for symbol, alpaca_symbol in zip(symbols, alpaca_symbols):
        try:
            results[symbol] = {
                **_ticker_data_from_frame(symbol, frames[alpaca_symbol]),
                "last_updated": datetime.now(),
            }
        except Exception as exc:
            results[symbol] = {"error": str(exc), "signal": False, "last_updated": datetime.now()}
    return results


def start_ticker_refresh(state, symbols: list[str]) -> None:
    executor = get_ticker_executor()
    batches = [symbols[index:index + REFRESH_WORKERS] for index in range(0, len(symbols), REFRESH_WORKERS)]
    credential_scope = _alpaca_cache_scope()
    state["refresh_jobs"] = {
        index: executor.submit(_cached_ticker_batch, tuple(batch), credential_scope)
        for index, batch in enumerate(batches)
    }
    state["refresh_completed"] = 0
    state["refresh_total"] = len(symbols)
    state["refresh_errors"] = []


@st.fragment(run_every="2s")
def render_refresh_controller(client, equity: float | None, positions: dict, refresh_symbols: list[str]) -> None:
    state = st.session_state
    if not refresh_symbols:
        st.caption("Seleziona almeno un ticker per caricare dati e grafici.")
        return
    if "refresh_jobs" not in state:
        refresh_requested = state.pop("refresh_requested", False)
        missing_symbols = [symbol for symbol in refresh_symbols if symbol not in state["live_data"]]
        if refresh_requested or (state.get("auto_refresh", True) and needs_refresh()):
            start_ticker_refresh(state, refresh_symbols)
        elif missing_symbols:
            start_ticker_refresh(state, missing_symbols)

    refresh_jobs = state.get("refresh_jobs")
    if refresh_jobs is not None:
        for batch_id, future in list(refresh_jobs.items()):
            if not future.done():
                continue
            results = future.result()
            state["live_data"].update(results)
            state["refresh_errors"].extend(symbol for symbol, result in results.items() if result.get("error"))
            state["refresh_completed"] += len(results)
            del refresh_jobs[batch_id]

        if not refresh_jobs:
            state.pop("refresh_jobs", None)
            state["last_refresh"] = datetime.now()
            if client is not None and equity is not None:
                run_bot_cycle(client, state, positions, equity)
            st.rerun(scope="app")

        completed = state["refresh_completed"]
        total = max(1, state.get("refresh_total", len(refresh_symbols)))
        st.progress(completed / total, text=f"Aggiornamento dati in background: {completed}/{total}")
    else:
        last_refresh = state.get("last_refresh")
        if last_refresh:
            st.caption(f"Ultimo aggiornamento dati: {last_refresh:%H:%M:%S}")
        else:
            st.caption("Primo aggiornamento dati in corso…")
        refresh_errors = state.get("refresh_errors", [])
        if refresh_errors:
            st.warning(f"Dati non disponibili per {', '.join(refresh_errors)}.")


def get_positions_map(client) -> dict:
    return {position.symbol: position for position in client.get_all_positions()}


def _live_credentials() -> tuple[str | None, str | None, bool]:
    try:
        secret_key = st.secrets.get("ALPACA_API_KEY")
        secret_value = st.secrets.get("ALPACA_SECRET_KEY")
    except Exception:
        secret_key = secret_value = None
    key = os.environ.get("ALPACA_API_KEY") or secret_key
    secret = os.environ.get("ALPACA_SECRET_KEY") or secret_value
    return key, secret, bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


def connect_alpaca() -> tuple[object | None, str | None]:
    key, secret, from_env = _live_credentials()
    if not key or not secret:
        return None, NO_KEYS_MESSAGE
    try:
        client = get_alpaca_client() if from_env else TradingClient(api_key=key, secret_key=secret, paper=True)
        return client, None
    except Exception as exc:
        return None, f"Connessione ad Alpaca non riuscita: {exc}"


def order_plan(
    data: dict,
    params: StrategyParams,
    entry_price_override: float | None = None,
) -> tuple[float, float, float]:
    """Return the entry limit, native trailing percentage, and whole-share quantity."""
    override = _number_or_nan(entry_price_override)
    entry_price = override if math.isfinite(override) and override > 0 else data["bb_lower"]
    limit = round(entry_price, 2)
    quantity = whole_share_quantity(params.trade_budget_usd, limit)
    return limit, params.trailing_pct, quantity


def add_log(state, message: str) -> None:
    log = state.setdefault("bot_log", [])
    log.append(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}")
    del log[:-LOG_LIMIT]


def reconcile_open_position_stops(client, positions: dict, state: dict) -> None:
    """Reconcile every account position, including manually traded symbols, with Alpaca stops."""
    for alpaca_symbol, position in positions.items():
        try:
            quantity = float(getattr(position, "qty", 0))
        except (TypeError, ValueError):
            continue
        if quantity <= 0:
            continue
        symbol = _display_symbol(alpaca_symbol)
        persisted = get_persisted_symbol_state(symbol)
        trailing_pct = get_ticker_params(symbol, state).trailing_pct
        stored_trail = persisted.get("trailing_pct")
        try:
            stored_trail = float(stored_trail) if stored_trail is not None else None
        except (TypeError, ValueError):
            stored_trail = None
        if stored_trail is not None and math.isfinite(stored_trail) and 0 < stored_trail < 1:
            trailing_pct = stored_trail
        try:
            ensure_native_trailing_stop(client, alpaca_symbol, trailing_pct, quantity)
        except Exception as exc:
            add_log(state, f"🛡️ {symbol}: trailing stop Alpaca non riconciliato: {exc}")
        try:
            symbol_state = state.setdefault("bot_state", {}).setdefault(symbol, {})
            if "position" not in symbol_state:
                local_position = _initial_position(
                    client,
                    alpaca_symbol,
                    persisted.get("high_water_mark"),
                )
                if local_position is not None:
                    symbol_state["position"] = local_position
                    state.setdefault("bot_high_water_marks", {})[symbol] = local_position.peak_price
                    persist_symbol_settings(state, symbol)
        except Exception as exc:
            add_log(state, f"🛡️ {symbol}: high-water mark non recuperato: {exc}")


def cancel_open_sells(client, alpaca_symbol: str, attempts: int = 10) -> None:
    """Protective stop legs lock the shares, so they must be cancelled before a market sell."""
    for order in _open_orders(client, alpaca_symbol):
        if order.symbol == alpaca_symbol and order.side == OrderSide.SELL:
            client.cancel_order_by_id(order.id)
    for _ in range(attempts):
        if not any(order.symbol == alpaca_symbol and order.side == OrderSide.SELL for order in _open_orders(client, alpaca_symbol)):
            return
        time.sleep(0.5)
    raise RuntimeError("lo stop-loss non risulta ancora annullato, riprova tra qualche secondo")


def run_bot_cycle(client, state, positions: dict, equity: float) -> None:
    """One bot pass over every enabled ticker: exit checks on open positions, entries on signals."""
    enabled = [symbol for symbol, on in state.get("bot_enabled", {}).items() if on]
    if positions:
        reconcile_open_position_stops(client, positions, state)
    try:
        market_open = bool(client.get_clock().is_open)
    except Exception as exc:
        add_log(state, f"🤖 Impossibile leggere l'orario di mercato: {exc}")
        return
    today = datetime.now(EASTERN).date()
    daily_risk = state.setdefault("daily_risk", {"day": None, "halted_symbols": {}})
    daily_risk.setdefault("halted_symbols", {})
    if daily_risk["day"] != today:
        daily_risk.update(day=today, halted_symbols={})
    if not market_open or not enabled:
        return
    bot_state = state.setdefault("bot_state", {})
    last_buy = state.setdefault("bot_last_buy", {})
    runner_order_ids = state.setdefault("bot_runner_order_ids", {})
    for symbol in enabled:
        alpaca_symbol = to_alpaca_symbol(symbol)
        data = state.get("live_data", {}).get(symbol, {})
        persisted = get_persisted_symbol_state(symbol)
        saved_profile = persisted.get("profile")
        if saved_profile in PROFILE_NAMES:
            state.setdefault("profile", {})[symbol] = saved_profile
        saved_override = persisted.get("custom_trailing_pct")
        if saved_override is None:
            state.get("trailing_stop_pct", {}).pop(symbol, None)
        else:
            try:
                saved_override = float(saved_override)
            except (TypeError, ValueError):
                saved_override = float("nan")
            if math.isfinite(saved_override) and 0 < saved_override < 1:
                state.setdefault("trailing_stop_pct", {})[symbol] = saved_override
        saved_custom_settings = persisted.get("custom_settings", {})
        saved_entry_override = saved_custom_settings.get("bot_entry_price_override")
        if saved_entry_override is None:
            state.get("bot_entry_price_overrides", {}).pop(symbol, None)
        else:
            state.setdefault("bot_entry_price_overrides", {})[symbol] = saved_entry_override
        for key, value in saved_custom_settings.items():
            if key != "bot_entry_price_override":
                state[f"{key}_{symbol}"] = value
        saved_buy_date = persisted.get("last_buy_date")
        if symbol not in last_buy and saved_buy_date:
            try:
                last_buy[symbol] = datetime.fromisoformat(saved_buy_date).date()
            except (TypeError, ValueError):
                add_log(state, f"🤖 {symbol}: data last-buy persistito non valido; lo ignoro")
        params = get_ticker_params(symbol, state)
        try:
            if daily_risk["halted_symbols"].get(symbol):
                continue
            daily_pnl = get_symbol_daily_pnl(client, alpaca_symbol, commission_pct=params.commission_pct)
            if daily_pnl <= -params.max_daily_drawdown_usd or daily_pnl >= params.daily_target_usd:
                close_symbol_position(client, alpaca_symbol)
                daily_risk["halted_symbols"][symbol] = True
                bot_state.pop(symbol, None)
                reason = "drawdown" if daily_pnl < 0 else "target"
                add_log(state, f"🤖 {symbol}: limite P&L giornaliero ({reason}) raggiunto: ${daily_pnl:,.2f}; ordini e posizione del solo simbolo chiusi")
                continue
            if alpaca_symbol in positions:
                if not market_open:
                    continue
                position = positions[alpaca_symbol]
                ensure_native_trailing_stop(
                    client,
                    alpaca_symbol,
                    params.trailing_pct,
                    abs(float(position.qty)),
                )
                exit_state = bot_state.setdefault(symbol, {})
                if runner_order_ids.get(symbol):
                    exit_state["runner_order_id"] = runner_order_ids[symbol]
                if "position" not in exit_state:
                    position = _initial_position(
                        client,
                        alpaca_symbol,
                        persisted.get("high_water_mark"),
                    )
                    if position is None:
                        if not exit_state.get("warned"):
                            add_log(state, f"🤖 {symbol}: posizione senza stop protettivo, gestiscila a mano")
                            exit_state["warned"] = True
                        continue
                    exit_state["position"] = position
                    state.setdefault("bot_high_water_marks", {})[symbol] = position.peak_price
                    persist_symbol_settings(state, symbol)
                continue
            bot_state.pop(symbol, None)
            if not compute_signal(data, params) or last_buy.get(symbol) == today:
                continue
            if any(order.side == OrderSide.BUY for order in _open_orders(client, alpaca_symbol)):
                continue
            entry_override = state.get("bot_entry_price_overrides", {}).get(symbol)
            limit, trailing_pct, shares = order_plan(data, params, entry_override)
            if shares < 1 or not 0 < trailing_pct < 1:
                add_log(state, f"🤖 {symbol}: segnale attivo ma ordine non dimensionabile")
                continue
            order_id = place_limit_buy(client, alpaca_symbol, limit, shares)
            last_buy[symbol] = today
            if state is st.session_state:
                persist_symbol_settings(state, symbol)
            add_log(state, f"🤖 {symbol}: {params.bot_mode} qty {shares:.0f} @ ${limit:,.2f}; trailing broker {params.trailing_pct:.1%} after fill (ID {order_id})")
        except Exception as exc:
            add_log(state, f"🤖 {symbol}: errore {exc}")


def init_state() -> None:
    defaults = {
        "bot_enabled": {},
        "active_tickers": [],
        "live_data": {}, "bot_log": [], "bot_state": {}, "bot_last_buy": {}, "bot_high_water_marks": {},
        "bot_entry_price_overrides": {},
        "panels": {}, "panel_msg": {}, "bot_notice": {}, "auto_refresh": True,
        "profile": {},
        "backtest_open": {},
        "backtest_cache": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.get("bot_state_loaded"):
        persisted_symbols = load_bot_state()["symbols"]
        for symbol, record in persisted_symbols.items():
            if not isinstance(record, dict):
                continue
            bot_enabled = bool(record.get("bot_enabled", False))
            st.session_state["bot_enabled"][symbol] = bot_enabled
            if bot_enabled or bool(record.get("active_ticker", False)):
                st.session_state["active_tickers"] = sorted(
                    set(st.session_state["active_tickers"]) | {symbol}
                )
            profile = record.get("profile")
            if profile in PROFILE_NAMES:
                st.session_state["profile"][symbol] = profile
            custom_trailing = record.get("custom_trailing_pct")
            try:
                custom_trailing = float(custom_trailing) if custom_trailing is not None else None
            except (TypeError, ValueError):
                custom_trailing = None
            if custom_trailing is not None and math.isfinite(custom_trailing) and 0 < custom_trailing < 1:
                st.session_state.setdefault("trailing_stop_pct", {})[symbol] = custom_trailing
            for key, value in record.get("custom_settings", {}).items():
                if key == "bot_entry_price_override":
                    st.session_state["bot_entry_price_overrides"][symbol] = value
                else:
                    st.session_state[f"{key}_{symbol}"] = value
            try:
                saved_hwm = float(record.get("high_water_mark"))
            except (TypeError, ValueError):
                saved_hwm = float("nan")
            if math.isfinite(saved_hwm) and saved_hwm > 0:
                st.session_state["bot_high_water_marks"][symbol] = saved_hwm
            last_buy_date = record.get("last_buy_date")
            if last_buy_date:
                try:
                    st.session_state["bot_last_buy"][symbol] = datetime.fromisoformat(last_buy_date).date()
                except (TypeError, ValueError):
                    add_log(st.session_state, f"🤖 {symbol}: data last-buy persistito non valido; lo ignoro")
        st.session_state["bot_state_loaded"] = True


def needs_refresh() -> bool:
    last = st.session_state.get("last_refresh")
    return last is None or (datetime.now() - last).total_seconds() >= REFRESH_SECONDS


def _on_refresh_now() -> None:
    st.session_state["refresh_requested"] = True


def _on_toggle(symbol: str) -> None:
    enabled = bool(st.session_state.get(f"bot_{symbol}"))
    set_ticker_bot_enabled(st.session_state, symbol, enabled)
    st.session_state["bot_notice"][symbol] = enabled
    add_log(st.session_state, f"🤖 {symbol}: bot {'ATTIVATO' if enabled else 'disattivato'}")


def set_ticker_bot_enabled(state: dict, symbol: str, enabled: bool) -> None:
    enabled = bool(enabled)
    was_enabled = bool(state.setdefault("bot_enabled", {}).get(symbol, False))
    state["bot_enabled"][symbol] = enabled
    active_tickers = set(state.get("active_tickers", []))
    was_active = symbol in active_tickers
    if enabled:
        active_tickers.add(symbol)
        state.setdefault("profile", {}).setdefault(symbol, DEFAULT_PROFILE)
    else:
        active_tickers.discard(symbol)
    state["active_tickers"] = sorted(active_tickers)
    if state is st.session_state and (was_enabled != enabled or was_active != (symbol in active_tickers)):
        persist_symbol_settings(state, symbol)


def _on_profile_change(symbol: str, client=None) -> None:
    name = st.session_state.get(f"profile_{symbol}", DEFAULT_PROFILE)
    previous_name = st.session_state["profile"].get(symbol, DEFAULT_PROFILE)
    if name != previous_name:
        st.session_state.get("trailing_stop_pct", {}).pop(symbol, None)
    st.session_state["profile"][symbol] = name
    persist_symbol_settings(st.session_state, symbol)
    try:
        has_trailing = client is not None and bool(_trailing_stop_orders(client, to_alpaca_symbol(symbol)))
        has_position = client is not None and to_alpaca_symbol(symbol) in get_positions_map(client)
        if client is not None and (has_trailing or has_position):
            updated = replace_trailing_stop_percent(
                client,
                to_alpaca_symbol(symbol),
                get_ticker_params(symbol).trailing_pct,
            )
            if not updated:
                add_log(st.session_state, f"⚠️ {symbol}: nessun trailing stop aperto da aggiornare su Alpaca")
    except Exception as exc:
        st.session_state.setdefault("panel_msg", {})[symbol] = (
            "error",
            f"Profilo salvato ma trailing stop Alpaca non aggiornato: {exc}",
        )
        add_log(st.session_state, f"⚠️ {symbol}: replace del trailing Alpaca fallito: {exc}")
    add_log(st.session_state, f"⚙️ {symbol}: profilo di rischio {name}")


def _on_toggle_backtest(symbol: str) -> None:
    opened = st.session_state["backtest_open"]
    opened[symbol] = not opened.get(symbol, False)


def _on_rerun_backtest(symbol: str) -> None:
    st.session_state["backtest_cache"].pop(symbol, None)


def _on_open_panel(symbol: str, kind: str, defaults: dict) -> None:
    panels = st.session_state["panels"]
    st.session_state["panel_msg"].pop(symbol, None)
    if panels.get(symbol) == kind:
        panels.pop(symbol)
        return
    panels[symbol] = kind
    for key, value in defaults.items():
        st.session_state[key] = value


def resolve_detail_symbol_from_click(table: pd.DataFrame | None, click_state: dict | None) -> str | None:
    if table is None or click_state is None:
        return None
    try:
        row_index = int(click_state.get("row", 0))
    except (TypeError, ValueError):
        return None
    if row_index < 0 or row_index >= len(table):
        return None
    ticker = table.iloc[row_index].get("Ticker", table.iloc[row_index].get("Symbol"))
    return str(ticker) if ticker is not None else None


def _on_select_detail(table_key: str, click_key: str) -> None:
    click = st.session_state.get(click_key)
    target = st.session_state.get(table_key)
    symbol = resolve_detail_symbol_from_click(target, click)
    if symbol:
        st.session_state["detail_symbol_selected"] = symbol


def _on_submit_buy(client, symbol: str) -> None:
    limit = float(st.session_state[f"buy_limit_{symbol}"])
    budget = float(st.session_state[f"buy_budget_{symbol}"])
    shares = whole_share_quantity(budget, limit)
    try:
        if shares <= 0:
            raise ValueError("Budget insufficiente per almeno un'azione intera protetta dal trailing Alpaca")
        order_id = place_limit_buy(client, to_alpaca_symbol(symbol), limit, shares)
    except Exception as exc:
        st.session_state["panel_msg"][symbol] = ("error", f"Ordine non inviato: {exc}")
        return
    st.session_state["panels"].pop(symbol, None)
    text = (
        f"Ordine limite inviato: ${budget:,.2f} ~ {shares:.0f} az. @ ${limit:,.2f}; "
        f"trailing Alpaca {get_ticker_params(symbol).trailing_pct:.1%} dopo il fill (ID {order_id})"
    )
    st.session_state["panel_msg"][symbol] = ("success", text)
    add_log(st.session_state, f"👤 {symbol}: {text}")


def _on_submit_sell(client, symbol: str) -> None:
    shares = float(st.session_state[f"sell_shares_{symbol}"])
    alpaca_symbol = to_alpaca_symbol(symbol)
    try:
        cancel_open_sells(client, alpaca_symbol)
        order_id = place_market_sell(client, alpaca_symbol, shares)
    except Exception as exc:
        st.session_state["panel_msg"][symbol] = ("error", f"Vendita non inviata: {exc}")
        return
    st.session_state["panels"].pop(symbol, None)
    st.session_state["bot_state"].pop(symbol, None)
    text = f"Vendita a mercato inviata: {shares:.4f} az. (ID {order_id})"
    st.session_state["panel_msg"][symbol] = ("success", text)
    add_log(st.session_state, f"👤 {symbol}: {text}")


def _on_kill_symbol(client, symbol: str) -> None:
    alpaca_symbol = to_alpaca_symbol(symbol)
    try:
        close_symbol_position(client, alpaca_symbol)
        set_ticker_bot_enabled(st.session_state, symbol, False)
        st.session_state["bot_state"].pop(symbol, None)
        add_log(st.session_state, f"🛑 {symbol}: KILL SIMBOLO, ordini annullati e posizione chiusa")
    except Exception as exc:
        add_log(st.session_state, f"🛑 {symbol}: KILL SIMBOLO fallito: {exc}")


def _fmt(value: float, pattern: str = "{:,.2f}") -> str:
    return pattern.format(value) if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def adx_label(adx: float, params: StrategyParams = PARAMS) -> str:
    if not math.isfinite(adx):
        return "ADX —"
    return f":{'green' if adx < params.adx_max else 'red'}[ADX **{adx:.1f}**]"


def rsi_label(rsi: float, params: StrategyParams = PARAMS) -> str:
    if not math.isfinite(rsi):
        return "RSI —"
    color = "green" if rsi < params.rsi_max else ("orange" if rsi <= 50 else "red")
    return f":{color}[RSI **{rsi:.1f}**]"


def render_header(client, account_error: str | None, equity: float | None, positions: dict) -> None:
    cols = st.columns(5)
    bots = sum(bool(on) for on in st.session_state["bot_enabled"].values())
    if client is None or equity is None:
        cols[0].metric("Account Equity", "—")
        cols[1].metric("Capitale investito", "—")
        cols[2].metric("Posizioni aperte", "—")
        cols[3].metric("P&L oggi (non realizzato)", "—")
        cols[4].metric("Bot attivi", bots)
        st.warning(account_error or NO_KEYS_MESSAGE)
        return
    today_pl = 0.0
    invested_capital = 0.0
    for position in positions.values():
        value = getattr(position, "unrealized_intraday_pl", None)
        if value is None:
            value = getattr(position, "unrealized_pl", 0)
        today_pl += float(value or 0)
        cost_basis = _number_or_nan(getattr(position, "cost_basis", None))
        if not math.isfinite(cost_basis):
            entry_price = _number_or_nan(getattr(position, "avg_entry_price", None))
            quantity = _number_or_nan(getattr(position, "qty", None))
            if math.isfinite(entry_price) and math.isfinite(quantity):
                cost_basis = entry_price * quantity
        if math.isfinite(cost_basis):
            invested_capital += abs(cost_basis)
    cols[0].metric("Account Equity", f"${equity:,.2f}")
    cols[1].metric("Capitale investito", f"${invested_capital:,.2f}")
    cols[2].metric("Posizioni aperte", len(positions))
    cols[3].metric("P&L oggi (non realizzato)", f"${today_pl:,.2f}", delta=f"{today_pl:,.2f}")
    cols[4].metric("Bot attivi", bots)
    if account_error:
        st.warning(account_error)


def render_buy_panel(client, symbol: str) -> None:
    with st.container(border=True):
        params = get_ticker_params(symbol)
        st.markdown(f"**Compra {symbol}** ~ ordine limite DAY con trailing stop Alpaca {params.trailing_pct:.1%} dopo il fill")
        cols = st.columns(2)
        cols[0].number_input("Prezzo limite ($)", min_value=0.0, step=0.01, key=f"buy_limit_{symbol}")
        budget = cols[1].number_input("Budget (USD)", min_value=1.0, step=25.0, key=f"buy_budget_{symbol}")
        limit = float(st.session_state.get(f"buy_limit_{symbol}", 0.0))
        estimate = whole_share_quantity(budget, limit)
        cols[1].caption(f"Quantità intera stimata: {estimate} az.")
        confirmed = st.checkbox("Confermo l'ordine", key=f"buy_confirm_{symbol}")
        st.button("Invia ordine", key=f"buy_submit_{symbol}", type="primary", disabled=not confirmed,
                  on_click=_on_submit_buy, args=(client, symbol))


def render_sell_panel(client, symbol: str, qty: float) -> None:
    with st.container(border=True):
        st.markdown(f"**Vendi {symbol}** ~ ordine a mercato")
        st.number_input(
            "Azioni da vendere", min_value=0.0001, max_value=max(0.0001, qty),
            step=0.0001, format="%.4f", key=f"sell_shares_{symbol}"
        )
        st.caption("Gli stop-loss aperti su questo titolo verranno annullati prima della vendita.")
        st.button("Conferma vendita", key=f"sell_submit_{symbol}", type="primary",
                  on_click=_on_submit_sell, args=(client, symbol))


def render_symbol_risk_status(client, symbol: str, data: dict, params: StrategyParams) -> None:
    alpaca_symbol = to_alpaca_symbol(symbol)
    pnl = get_symbol_daily_pnl(client, alpaca_symbol, commission_pct=params.commission_pct)
    lower = -params.max_daily_drawdown_usd
    upper = params.daily_target_usd
    progress = min(1.0, max(0.0, (pnl - lower) / (upper - lower))) if upper > lower else 0.0
    color = "#16834a" if pnl >= 0 else "#c83232"
    st.markdown(
        f'<div role="meter" aria-valuenow="{pnl:.2f}" aria-valuemin="{lower:.2f}" '
        f'aria-valuemax="{upper:.2f}" aria-label="P&L realizzato giornaliero {symbol}" '
        'style="height:8px;border-radius:4px;background:#e6e8eb;overflow:hidden">'
        f'<div style="height:100%;width:{progress * 100:.2f}%;background:{color}"></div></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"P&L realizzato oggi: ${pnl:+.2f} ({lower:+.2f} / +{upper:.2f})")

    price = data.get("price")
    if params.bot_mode == "TREND_FOLLOWER" and isinstance(price, (int, float)) and price > 0:
        stops = [
            float(order.stop_price)
            for order in _trailing_stop_orders(client, alpaca_symbol)
            if order.symbol == alpaca_symbol
            and getattr(order, "stop_price", None) is not None
        ]
        if stops:
            stop_price = max(stops)
            distance_pct = max(0.0, (price - stop_price) / price * 100)
            st.caption(f"Stop trailing broker: ${stop_price:.2f}, distanza {distance_pct:.2f}%")


def render_custom_sliders(symbol: str) -> None:
    previous_trailing = get_persisted_symbol_state(symbol).get("trailing_pct")
    base = PROFILES[DEFAULT_PROFILE]
    defaults = {f"custom_adx_{symbol}": float(base.adx_max), f"custom_rsi_{symbol}": float(base.rsi_max),
                f"custom_budget_{symbol}": float(base.trade_budget_usd),
                f"custom_stop_atr_{symbol}": float(base.stop_loss_atr_mult),
                f"custom_take_profit_atr_{symbol}": float(base.take_profit_atr_mult),
                f"custom_trailing_pct_{symbol}": float(base.trailing_pct),
                f"custom_daily_target_{symbol}": float(base.daily_target_usd),
                f"custom_max_drawdown_{symbol}": float(base.max_daily_drawdown_usd)}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    with st.container(border=True):
        st.caption(f"🎛️ Profilo Custom ~ {symbol}")
        cols = st.columns(8)
        cols[0].slider("ADX max", 15.0, 50.0, step=1.0, key=f"custom_adx_{symbol}")
        cols[1].slider("RSI max", 25.0, 60.0, step=1.0, key=f"custom_rsi_{symbol}")
        cols[2].number_input("Budget $", min_value=1.0, step=25.0, key=f"custom_budget_{symbol}")
        cols[3].slider("Stop ATR mult", 0.5, 4.0, step=0.1, key=f"custom_stop_atr_{symbol}")
        cols[4].slider("Target ATR mult", 0.1, 5.0, step=0.1, key=f"custom_take_profit_atr_{symbol}")
        cols[5].slider("Trailing stop %", 0.01, 0.20, step=0.01, key=f"custom_trailing_pct_{symbol}")
        cols[6].number_input("Daily target $", min_value=0.1, step=1.0, key=f"custom_daily_target_{symbol}")
        cols[7].number_input("Daily loss cap $", min_value=0.1, step=1.0, key=f"custom_max_drawdown_{symbol}")
    params = get_ticker_params(symbol)
    persist_symbol_settings(st.session_state, symbol)
    try:
        previous_trailing = float(previous_trailing)
    except (TypeError, ValueError):
        previous_trailing = params.trailing_pct
    if _ACTIVE_TRADING_CLIENT is not None and not math.isclose(previous_trailing, params.trailing_pct):
        try:
            replace_trailing_stop_percent(client=_ACTIVE_TRADING_CLIENT, symbol=to_alpaca_symbol(symbol),
                                          trailing_pct=params.trailing_pct)
        except Exception as exc:
            st.session_state.setdefault("panel_msg", {})[symbol] = (
                "error",
                f"Impostazione custom salvata ma trailing stop Alpaca non aggiornato: {exc}",
            )


def backtest_chart(table: pd.DataFrame) -> go.Figure:
    metrics = ["Rendimento %", "Win rate %", "Max drawdown %", "N. trade"]
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)
    colors = ["#2e7d32", "#1565c0", "#e65100"]
    for index, row in table.reset_index(drop=True).iterrows():
        for position, metric in enumerate(metrics, start=1):
            fig.add_trace(go.Bar(x=[row["Profilo"]], y=[row[metric]], name=row["Profilo"],
                                 marker_color=colors[index % len(colors)], showlegend=position == 1,
                                 legendgroup=row["Profilo"], text=[row[metric]], textposition="auto"),
                          row=1, col=position)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.1))
    return fig


def render_backtest_panel(symbol: str) -> None:
    cache = st.session_state["backtest_cache"]
    with st.container(border=True):
        st.markdown(f"**📊 Backtest comparativo {symbol}** ~ ultimi 90 giorni, capitale simulato $100.000")
        if symbol not in cache:
            with st.spinner(f"Backtest di {symbol} sui 3 profili…"):
                try:
                    cache[symbol] = run_profile_backtest(symbol)
                except Exception as exc:
                    cache[symbol] = str(exc)
        results = cache[symbol]
        if isinstance(results, str) or not results:
            st.error(f"Backtest non disponibile: {results or 'nessun risultato'}")
        else:
            table = backtest_table(results)
            st.plotly_chart(backtest_chart(table), key=f"backtest_chart_{symbol}")
            st.dataframe(table, hide_index=True)
            st.caption("Uscite: trailing stop 3%, time stop 10 candele, stop-loss ATR del profilo.")
        st.button("🔄 Ricalcola", key=f"backtest_rerun_{symbol}", on_click=_on_rerun_backtest, args=(symbol,))


def render_ticker_row(symbol: str, client, equity: float | None, positions: dict) -> None:
    data = st.session_state["live_data"].get(symbol, {})
    trading_ok = client is not None and equity is not None
    position = positions.get(to_alpaca_symbol(symbol))
    params = get_ticker_params(symbol)
    signal = compute_signal(data, params)
    cols = st.columns(ROW_WIDTHS, vertical_alignment="center")

    if "price" not in data:
        cols[0].markdown(f"**{symbol}**  \n:gray[{str(data.get('error', 'in caricamento…'))[:60]}]")
    else:
        change = (data["price"] / data["prev_close"] - 1) * 100 if data["prev_close"] else float("nan")
        delta = f":{'green' if change >= 0 else 'red'}[{change:+.2f}%]" if math.isfinite(change) else ""
        cols[0].markdown(f"**{symbol}**  \n${_fmt(data['price'])} {delta}")
        cols[1].markdown(adx_label(data["adx"], params))
        cols[2].markdown(rsi_label(data["rsi"], params))
    cols[3].markdown("🟢 **Segnale**" if signal else "🔴 No segnale")

    with cols[4]:
        if f"profile_{symbol}" not in st.session_state:
            st.session_state[f"profile_{symbol}"] = st.session_state["profile"].get(symbol, DEFAULT_PROFILE)
        st.selectbox(f"Profilo {symbol}", PROFILE_NAMES, key=f"profile_{symbol}", label_visibility="collapsed",
                     on_change=_on_profile_change, args=(symbol, client))
        st.toggle(f"Bot {symbol}", key=f"bot_{symbol}", disabled=not trading_ok,
                  on_change=_on_toggle, args=(symbol,))
        if st.session_state["bot_enabled"].get(symbol):
            st.markdown(":blue[🤖 Bot attivo]")
        notice = st.session_state["bot_notice"].pop(symbol, None)
        if notice:
            st.caption("✅ Attivato: agirà al prossimo aggiornamento")

    with cols[5]:
        if position is not None:
            pnl = float(position.unrealized_pl or 0)
            pct = float(position.unrealized_plpc or 0) * 100
            color = "green" if pnl >= 0 else "red"
            st.markdown(f"{position.qty} az. ~ :{color}[${pnl:,.2f} ({pct:+.2f}%)]")
            qty = float(position.qty)
            st.button("Vendi", key=f"sell_{symbol}", disabled=not trading_ok, on_click=_on_open_panel,
                      args=(symbol, "sell", {f"sell_shares_{symbol}": max(0.0001, qty)}))
            if trading_ok:
                st.button("🛑 KILL SIMBOLO", key=f"kill_{symbol}", type="primary",
                          on_click=_on_kill_symbol, args=(client, symbol))
                render_symbol_risk_status(client, symbol, data, params)
        else:
            defaults = {}
            if trading_ok and "price" in data and math.isfinite(data["atr"]) and math.isfinite(data["bb_lower"]):
                limit, _, _ = order_plan(data, params)
                defaults = {
                    f"buy_limit_{symbol}": max(0.0, limit),
                    f"buy_budget_{symbol}": params.trade_budget_usd,
                    f"buy_confirm_{symbol}": False,
                }
                st.caption(f"Stop trailing Alpaca: {params.trailing_pct:.1%} dopo il fill")
            st.button("Compra", key=f"buy_{symbol}", disabled=not defaults, on_click=_on_open_panel,
                      args=(symbol, "buy", defaults))
            if trading_ok and st.session_state["bot_enabled"].get(symbol):
                st.button("🛑 KILL SIMBOLO", key=f"kill_{symbol}", type="primary",
                          on_click=_on_kill_symbol, args=(client, symbol))
        st.button("📊 Backtest", key=f"backtest_{symbol}", on_click=_on_toggle_backtest, args=(symbol,))

    if st.session_state["profile"].get(symbol) == CUSTOM_PROFILE:
        render_custom_sliders(symbol)

    message = st.session_state["panel_msg"].get(symbol)
    if message:
        (st.success if message[0] == "success" else st.error)(message[1])
    panel = st.session_state["panels"].get(symbol)
    if panel == "buy" and trading_ok and position is None:
        render_buy_panel(client, symbol)
    elif panel == "sell" and trading_ok and position is not None:
        render_sell_panel(client, symbol, float(position.qty))
    if st.session_state["backtest_open"].get(symbol):
        render_backtest_panel(symbol)


def render_watchlist(
    client,
    equity: float | None,
    positions: dict,
    symbols: list[str],
    search_query: str = "",
) -> None:
    query = search_query.strip().casefold()
    visible_symbols = [symbol for symbol in symbols if query in symbol.casefold()]
    if not visible_symbols:
        st.info("Nessun ticker corrisponde alla ricerca.")
        return
    signals = sum(
        compute_signal(st.session_state["live_data"].get(symbol, {}), get_ticker_params(symbol))
        for symbol in visible_symbols
    )
    with st.expander(f"Ticker selezionati · {len(visible_symbols)} · {signals} segnali", expanded=True):
        head = st.columns(ROW_WIDTHS)
        for column, label in zip(head, ("Titolo / Prezzo", "ADX", "RSI", "Segnale", "Profilo / Bot", "Azioni")):
            column.caption(label)
        for symbol in visible_symbols:
            render_ticker_row(symbol, client, equity, positions)


def build_watchlist_table(
    symbols: list[str],
    positions: dict,
    state=None,
    broker_orders: dict[str, list] | None = None,
) -> pd.DataFrame:
    state = st.session_state if state is None else state
    broker_orders = broker_orders or {}
    rows = []
    for symbol in symbols:
        data = state.get("live_data", {}).get(symbol, {})
        position = positions.get(to_alpaca_symbol(symbol))
        bot_state = state.get("bot_state", {}).get(symbol, {})
        managed_position = bot_state.get("position", bot_state.get("base"))
        price = data.get("price")
        if price is None and position is not None:
            price = getattr(position, "current_price", None)
        price = _number_or_nan(price)
        previous_close = _number_or_nan(data.get("prev_close"))
        change = (
            (price / previous_close - 1) * 100
            if math.isfinite(price) and math.isfinite(previous_close) and previous_close != 0
            else float("nan")
        )
        quantity = _number_or_nan(getattr(position, "qty", None)) if position is not None else float("nan")
        entry_price = _number_or_nan(getattr(position, "avg_entry_price", None)) if position is not None else float("nan")
        invested = _number_or_nan(getattr(position, "cost_basis", None)) if position is not None else float("nan")
        if not math.isfinite(invested) and math.isfinite(entry_price) and math.isfinite(quantity):
            invested = abs(entry_price * quantity)
        current_value = _number_or_nan(getattr(position, "market_value", None)) if position is not None else float("nan")
        if not math.isfinite(current_value) and math.isfinite(price) and math.isfinite(quantity):
            current_value = price * quantity
        pnl = _number_or_nan(getattr(position, "unrealized_pl", None)) if position is not None else float("nan")
        if not math.isfinite(pnl) and math.isfinite(current_value) and math.isfinite(invested):
            pnl = current_value - invested
        pnl_pct = _number_or_nan(getattr(position, "unrealized_plpc", None)) if position is not None else float("nan")
        if math.isfinite(pnl_pct):
            pnl_pct *= 100
        elif math.isfinite(pnl) and math.isfinite(invested) and invested != 0:
            pnl_pct = pnl / invested * 100
        else:
            pnl_pct = float("nan")
        params = get_ticker_params(symbol, state)
        saved_hwm = state.get("bot_high_water_marks", {}).get(symbol, entry_price)
        high_water_mark = _number_or_nan(getattr(managed_position, "peak_price", saved_hwm))
        if position is not None and not math.isfinite(high_water_mark):
            high_water_mark = entry_price
        trailing_orders = [
            order for order in broker_orders.get(to_alpaca_symbol(symbol), [])
            if order.symbol == to_alpaca_symbol(symbol)
            and order.side == OrderSide.SELL
            and str(getattr(getattr(order, "type", None), "value", getattr(order, "type", ""))).lower()
            == OrderType.TRAILING_STOP.value
        ]
        broker_stop = max(
            (
                _number_or_nan(getattr(order, "stop_price", None))
                for order in trailing_orders
                if math.isfinite(_number_or_nan(getattr(order, "stop_price", None)))
            ),
            default=float("nan"),
        )
        bot_enabled = bool(state.get("bot_enabled", {}).get(symbol, False))
        entry_override = _number_or_nan(state.get("bot_entry_price_overrides", {}).get(symbol))
        if math.isfinite(entry_override) and entry_override > 0:
            bot_entry_price = entry_override
            bot_entry_status = "Prossimo ingresso" if position is not None else "Personalizzato"
        elif position is not None:
            bot_entry_price = entry_price
            bot_entry_status = "Eseguito"
        elif bot_enabled:
            entry_inputs = (data.get("atr"), data.get("bb_lower"))
            if all(isinstance(value, (int, float)) and math.isfinite(value) for value in entry_inputs):
                bot_entry_price, _, _ = order_plan(data, params)
                bot_entry_status = "Previsto"
            else:
                bot_entry_price = float("nan")
                bot_entry_status = "In attesa dati"
        else:
            bot_entry_price = float("nan")
            bot_entry_status = "Bot spento"
        rows.append({
            "Ticker": symbol,
            "Prezzo attuale ($)": price,
            "Var. giornaliera %": change,
            "Prezzo medio acquisto ($)": entry_price,
            "Prezzo ingresso bot ($)": bot_entry_price,
            "Stato ingresso bot": bot_entry_status,
            "Quantità": quantity,
            "Capitale investito ($)": invested,
            "Valore attuale ($)": current_value,
            "ADX": data.get("adx", float("nan")),
            "RSI": data.get("rsi", float("nan")),
            "Segnale": "Sì" if compute_signal(data, get_ticker_params(symbol, state)) else "No",
            "Bot": bool(state.get("bot_enabled", {}).get(symbol, False)),
            "Profilo di Rischio": state.get("profile", {}).get(symbol, DEFAULT_PROFILE),
            "Trailing stop %": params.trailing_pct * 100,
            "Prezzo Massimo Raggiunto ($)": high_water_mark,
            "Stop broker ($)": broker_stop,
            "Posizione": "Aperta" if position is not None else "Assente",
            "P&L ($)": pnl,
            "P&L posizione %": pnl_pct,
        })
    return pd.DataFrame(rows)


def update_bot_high_water_mark(state: dict, symbol: str, requested_value) -> bool:
    value = _number_or_nan(requested_value)
    if not math.isfinite(value) or value <= 0:
        return False
    symbol_state = state.get("bot_state", {}).get(symbol)
    if not symbol_state:
        return False
    changed = False
    for key in ("position", "base"):
        position = symbol_state.get(key)
        if not isinstance(position, Position) or value <= position.peak_price:
            continue
        symbol_state[key] = replace(position, peak_price=max(value, position.entry_price))
        changed = True
    if changed:
        state.setdefault("bot_high_water_marks", {})[symbol] = max(
            _number_or_nan(getattr(symbol_state.get("position"), "peak_price", value)),
            _number_or_nan(getattr(symbol_state.get("base"), "peak_price", value)),
        )
        if state is st.session_state:
            persist_symbol_settings(state, symbol)
    return changed


def _number_or_nan(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def build_market_radar_table(
    radar: pd.DataFrame,
    active_tickers: set[str] | list[str] | None = None,
    live_data: dict[str, dict] | None = None,
) -> pd.DataFrame:
    table = radar.copy()
    active = set(active_tickers or [])
    live_data = live_data or {}
    live_prices = []
    live_changes = []
    for symbol in table["Symbol"].astype(str):
        data = live_data.get(symbol, {})
        price = _number_or_nan(data.get("price"))
        previous_close = _number_or_nan(data.get("prev_close"))
        live_prices.append(price)
        live_changes.append(
            (price / previous_close - 1) * 100
            if math.isfinite(price) and math.isfinite(previous_close) and previous_close != 0
            else float("nan")
        )
    table["Prezzo dinamico ($)"] = live_prices
    table["Var. dinamica %"] = live_changes
    table["Segnale"] = table.apply(
        lambda row: "Scalper + Trend" if bool(row.get("Validatore_Scalper")) and bool(row.get("Validatore_Trend"))
        else "Scalper" if bool(row.get("Validatore_Scalper"))
        else "Trend" if bool(row.get("Validatore_Trend"))
        else "—",
        axis=1,
    )
    table["Attiva Bot"] = table["Symbol"].astype(str).isin(active)
    table["Validatore_Scalper"] = table["Validatore_Scalper"].map({True: "🟢", False: "🔴"})
    table["Validatore_Trend"] = table["Validatore_Trend"].map({True: "🟢", False: "🔴"})
    return table


def paginate_market_radar(filtered: pd.DataFrame, page: int, page_size: int) -> tuple[pd.DataFrame, int]:
    page_count = max(1, math.ceil(len(filtered) / page_size))
    page = min(max(page, 0), page_count - 1)
    start = page * page_size
    return filtered.iloc[start:start + page_size].copy(), page_count


def search_market_radar_by_security(radar: pd.DataFrame, query: str) -> pd.DataFrame:
    query = query.strip()
    if not query:
        return radar.copy()
    security_names = radar.get("SecurityName", pd.Series("", index=radar.index))
    matches = security_names.fillna("").astype(str).str.contains(query, case=False, regex=False)
    return radar.loc[matches].copy()


def resolve_market_radar_selected_symbol(table: pd.DataFrame, selected_rows) -> str | None:
    if table.empty or not selected_rows:
        return None
    try:
        row_index = int(selected_rows[0])
    except (TypeError, ValueError):
        return None
    if row_index < 0 or row_index >= len(table):
        return None
    symbol = table.iloc[row_index].get("Symbol")
    return str(symbol) if symbol is not None else None


def render_market_radar_detail(row: pd.Series, symbol: str) -> None:
    st.markdown("---")
    st.subheader(f"Dettaglio {symbol}")
    info_column, chart_column = st.columns([1, 2], gap="large")

    with info_column:
        name = row.get("SecurityName", "")
        if pd.notna(name) and str(name).strip():
            st.markdown(f"**{name}**")
        sector = row.get("Sector", "—")
        industry = row.get("Industry", "—")
        quote_type = row.get("QuoteType", "—")
        st.write(f"**Settore:** {sector if pd.notna(sector) else '—'}")
        st.write(f"**Industria:** {industry if pd.notna(industry) else '—'}")
        st.write(f"**Tipo:** {quote_type if pd.notna(quote_type) else '—'}")
        metrics = st.columns(2)
        for column, label, field, suffix in (
            (metrics[0], "Volume medio 20g", "Volume_SMA20", ""),
            (metrics[1], "Market cap", "MarketCap", ""),
            (metrics[0], "RSI", "RSI", ""),
            (metrics[1], "ADX", "ADX", ""),
            (metrics[0], "ATR", "ATR_pct", "%"),
            (metrics[1], "Prezzo", "Close", " $"),
        ):
            value = _number_or_nan(row.get(field))
            display_value = f"{value:,.2f}{suffix}" if math.isfinite(value) else "—"
            column.metric(label, display_value)

    with chart_column:
        ticker_data = st.session_state.get("live_data", {}).get(symbol, {})
        chart_data = ticker_data.get("chart_data")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=100)
        if isinstance(chart_data, pd.DataFrame) and not chart_data.empty:
            recent_chart = chart_data.loc[pd.to_datetime(chart_data.index, utc=True) >= cutoff]
            ticker_data = {**ticker_data, "chart_data": recent_chart}
        else:
            recent_chart = pd.DataFrame()

        try:
            if recent_chart.empty:
                key, secret, _ = _live_credentials()
                if not key or not secret:
                    raise ValueError("Configura le chiavi Alpaca per caricare lo storico.")
                credential_scope = hashlib.sha256(f"{key}:{secret}".encode()).hexdigest()
                data_client = StockHistoricalDataClient(key, secret)
                with st.spinner(f"Caricamento candele {symbol}..."):
                    ticker_data = _cached_market_detail_data(symbol, credential_scope, data_client)
            render_candlestick_chart(
                None,
                symbol,
                ticker_data,
                get_ticker_params(symbol),
                chart_key=f"market_radar_candlestick_{symbol}",
            )
        except Exception as exc:
            st.error(f"Grafico non disponibile per {symbol}: {exc}")

        active_tickers = set(st.session_state.get("active_tickers", []))
        if st.button(
            f"➕ Aggiungi {symbol} ai Bot Attivi",
            key=f"add_market_radar_ticker_{symbol}",
            disabled=symbol in active_tickers,
        ):
            active_tickers.add(symbol)
            st.session_state["active_tickers"] = sorted(active_tickers)
            update_symbol_state(symbol, {"active_ticker": True})
            st.success(f"{symbol} aggiunto ai Bot Attivi.")


def render_market_explorer(positions: dict, available_symbols: list[str], trading_client=None) -> list[str]:
    st.subheader("Tutti i ticker")
    st.session_state["market_radar_visible_symbols"] = []
    search_col, sector_col, type_col = st.columns([1.6, 1.4, 1.2])
    selected_symbols = []
    search_query = search_col.text_input("Cerca Security", key="radar_search")
    if st.session_state.pop("nasdaq_sync_message", None):
        st.success("Database Nasdaq aggiornato.")
    if st.button("Aggiorna Anagrafica Nasdaq", key="update_nasdaq_security_master"):
        try:
            from screener import sync_market_radar_metadata, update_nasdaq_db

            with st.spinner("Download dati dal Nasdaq in corso..."):
                update_nasdaq_db()
                sync_market_radar_metadata()
            st.session_state["nasdaq_sync_message"] = True
            st.rerun(scope="app")
        except Exception as exc:
            st.error(f"Aggiornamento anagrafica Nasdaq non riuscito: {exc}")

    radar_service = None
    if trading_client is not None and available_symbols:
        try:
            from screener import (
                get_market_radar_worker_status,
                initialize_market_radar,
                start_market_radar_worker,
            )

            initialize_market_radar(available_symbols)
            key, secret, _ = _live_credentials()
            if key and secret:
                start_market_radar_worker(
                    available_symbols,
                    trading_client=trading_client,
                    data_client=StockHistoricalDataClient(key, secret),
                    batch_size=25,
                )
            radar_service = get_market_radar_worker_status()
        except Exception as exc:
            st.warning(f"Bootstrap del radar non riuscito: {exc}")

    status_column, refresh_column = st.columns([4, 1])
    if radar_service:
        state = radar_service.get("state", "idle")
        if state == "running":
            status_column.caption(
                f"Aggiornamento in background: {radar_service.get('completed', 0)} / "
                f"{radar_service.get('total', 0)} ticker elaborati."
            )
        elif state == "failed":
            status_column.warning(f"Aggiornamento interrotto: {radar_service.get('error')}")
            if refresh_column.button("Riprova aggiornamento", key="retry_market_radar"):
                key, secret, _ = _live_credentials()
                if key and secret:
                    from screener import start_market_radar_worker

                    start_market_radar_worker(
                        available_symbols,
                        force=True,
                        trading_client=trading_client,
                        data_client=StockHistoricalDataClient(key, secret),
                        batch_size=25,
                    )
                    st.rerun(scope="app")
        elif state == "complete":
            status_column.caption("Catalogo aggiornato; dati letti dal database SQLite.")
    if refresh_column.button("Aggiorna Tabella", key="refresh_market_radar_table"):
        st.rerun(scope="app")

    try:
        radar = load_market_radar_data()
    except Exception as exc:
        st.error(f"Impossibile leggere la tabella market_radar: {exc}")
        return selected_symbols
    if radar.empty:
        if trading_client is None:
            st.info("Connessione Alpaca non disponibile. Configura ALPACA_API_KEY e ALPACA_SECRET_KEY nei Secrets dell'app.")
        elif not available_symbols:
            st.info("Connessione Alpaca attiva, ma non risultano asset USA attivi, tradabili e frazionabili.")
        elif radar_service and radar_service.get("error"):
            st.error(f"Aggiornamento radar non riuscito: {radar_service['error']}")
        else:
            st.info("Il Market Radar non contiene ancora ticker.")
        return selected_symbols

    required = {
        "Symbol", "Sector", "QuoteType", "MarketCap", "Close", "Volume_SMA20",
        "ADX", "ATR_pct", "RSI", "BB_lower", "SMA_200",
        "Validatore_Scalper", "Validatore_Trend",
    }
    missing = sorted(required - set(radar.columns))
    if missing:
        st.error(f"La tabella market_radar non contiene: {', '.join(missing)}")
        return selected_symbols
    for column in ("SecurityName", "Industry"):
        if column not in radar.columns:
            radar[column] = "Unknown"

    sectors = sorted(radar["Sector"].dropna().astype(str).unique())
    quote_types = sorted(radar["QuoteType"].dropna().astype(str).unique())
    selected_sectors = sector_col.multiselect("Settore", sectors, default=sectors, key="radar_sectors")
    selected_types = type_col.multiselect("Tipo", quote_types, default=quote_types, key="radar_types")
    buy_the_dip_only = st.checkbox(
        "🎯 Mostra solo titoli sotto la Banda di Bollinger",
        value=False,
        key="filter_bb_dip",
    )

    filter_controls = st.columns([1.5, 1.5, 1.2, 1.2, 1])
    scalper_only = filter_controls[0].toggle("Solo segnali Scalper", key="radar_scalper_only")
    trend_only = filter_controls[1].toggle("Solo segnali Trend", key="radar_trend_only")
    sort_by = filter_controls[2].selectbox(
        "Ordina per",
        ["Symbol", "MarketCap", "Close", "Volume_SMA20", "ADX", "ATR_pct", "RSI"],
        format_func=lambda column: {"Symbol": "Ticker", "MarketCap": "Market Cap", "Close": "Prezzo",
                                    "Volume_SMA20": "Volume", "ATR_pct": "ATR %"}.get(column, column),
        key="radar_sort_by",
    )
    descending = filter_controls[3].toggle("Decrescente", key="radar_sort_desc", value=sort_by != "Symbol")
    page_size = filter_controls[4].selectbox("Righe", [10, 25, 50, 100], index=1, key="radar_page_size")

    filtered = filter_market_radar(
        radar, selected_sectors, selected_types, scalper_only, trend_only,
        buy_the_dip_only=buy_the_dip_only,
    )
    filtered = search_market_radar_by_security(filtered, search_query)
    filtered = filtered.sort_values(sort_by, ascending=not descending, na_position="last", kind="stable")
    if filtered.empty:
        st.info("Nessun ticker corrisponde ai filtri selezionati.")
        return selected_symbols

    page_key = "radar_page"
    radar_page, page_count = paginate_market_radar(
        filtered, int(st.session_state.get(page_key, 0)), page_size
    )
    st.session_state[page_key] = min(max(int(st.session_state.get(page_key, 0)), 0), page_count - 1)
    previous_col, page_label_col, next_col = st.columns([1, 2, 1])
    if previous_col.button("Precedente", disabled=st.session_state[page_key] == 0, key="radar_previous"):
        st.session_state[page_key] -= 1
    page_label_col.caption(f"Pagina {st.session_state[page_key] + 1} di {page_count} · {len(filtered)} ticker")
    if next_col.button("Successiva", disabled=st.session_state[page_key] >= page_count - 1, key="radar_next"):
        st.session_state[page_key] += 1
    radar_page, _ = paginate_market_radar(filtered, st.session_state[page_key], page_size)
    st.session_state["market_radar_visible_symbols"] = radar_page["Symbol"].astype(str).tolist()

    active_tickers = {
        symbol for symbol, enabled in st.session_state["bot_enabled"].items() if enabled
    }
    radar_table = build_market_radar_table(
        radar_page,
        active_tickers,
        st.session_state.get("live_data", {}),
    )
    selection_key = hashlib.sha256(
        "\0".join(radar_table["Symbol"].astype(str)).encode()
    ).hexdigest()[:12]
    st.session_state["market_radar_detail_table"] = radar_table
    visible_symbols = set(radar_table["Symbol"].astype(str))
    edited = st.data_editor(
        radar_table,
        hide_index=True,
        width="stretch",
        disabled=[column for column in radar_table.columns if column not in {"Symbol", "Attiva Bot"}],
        column_config={
            "Symbol": st.column_config.ButtonColumn(
                "Ticker",
                help="Clicca il ticker per aprire il dettaglio.",
                on_click=_on_select_detail,
                args=("market_radar_detail_table", "market_radar_table_click"),
                key="market_radar_table_click",
            ),
            "Segnale": st.column_config.TextColumn("Segnale"),
            "Attiva Bot": st.column_config.CheckboxColumn(
                "Bot attivo",
                help="Clicca la casella per attivare o disattivare il bot.",
                disabled=trading_client is None,
            ),
            "MarketCap": st.column_config.NumberColumn("Market Cap", format="$%d"),
            "Close": st.column_config.NumberColumn("Close", format="$%.2f"),
            "Prezzo dinamico ($)": st.column_config.NumberColumn("Prezzo attuale", format="$%.2f"),
            "Var. dinamica %": st.column_config.NumberColumn("Var. oggi", format="%.2f%%"),
            "Volume_SMA20": st.column_config.NumberColumn("Volume SMA20", format="%d"),
            "ATR_pct": st.column_config.NumberColumn("ATR %", format="%.2f%%"),
            "RSI": st.column_config.NumberColumn("RSI", format="%.2f"),
            "ADX": st.column_config.NumberColumn("ADX", format="%.2f"),
            "BB_lower": st.column_config.NumberColumn("BB Lower", format="$%.2f"),
            "SMA_200": st.column_config.NumberColumn("SMA 200", format="$%.2f"),
            "Validatore_Scalper": st.column_config.TextColumn("Scalper"),
            "Validatore_Trend": st.column_config.TextColumn("Trend"),
        },
        key=f"market_radar_editor_{selection_key}",
    )
    updated_active = sync_market_editor_selection(
        set(st.session_state.get("active_tickers", [])),
        visible_symbols,
        edited,
        positions,
        st.session_state["bot_enabled"],
        persist_changes=True,
    )
    if updated_active != set(st.session_state.get("active_tickers", [])):
        st.session_state["active_tickers"] = sorted(updated_active)
    selected_symbol = st.session_state.get("detail_symbol_selected")
    if selected_symbol:
        selected_rows = radar_table[radar_table["Symbol"].astype(str) == str(selected_symbol)]
        if not selected_rows.empty:
            render_market_radar_detail(selected_rows.iloc[0], str(selected_symbol))
    return selected_symbols


def _radar_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def filter_market_radar(
    radar: pd.DataFrame,
    sectors: list[str],
    quote_types: list[str],
    scalper_only: bool = False,
    trend_only: bool = False,
    buy_the_dip_only: bool = False,
) -> pd.DataFrame:
    sector_values = radar["Sector"].astype(str)
    quote_type_values = radar["QuoteType"].astype(str)
    if not sectors:
        sectors = sorted(sector_values.unique())
    if not quote_types:
        quote_types = sorted(quote_type_values.unique())

    filtered = radar[
        sector_values.isin(sectors)
        & quote_type_values.isin(quote_types)
    ].copy()
    filtered["Validatore_Scalper"] = filtered["Validatore_Scalper"].map(_radar_bool)
    filtered["Validatore_Trend"] = filtered["Validatore_Trend"].map(_radar_bool)
    if scalper_only:
        filtered = filtered[filtered["Validatore_Scalper"]]
    if trend_only:
        filtered = filtered[filtered["Validatore_Trend"]]
    if buy_the_dip_only:
        current_price = pd.to_numeric(filtered["Close"], errors="coerce")
        bollinger_lower = pd.to_numeric(filtered["BB_lower"], errors="coerce")
        filtered = filtered.loc[current_price <= bollinger_lower]
    return filtered


def merge_market_editor_selection(
    active_tickers: set[str],
    visible_symbols: set[str],
    edited: pd.DataFrame,
) -> set[str]:
    updated = active_tickers - visible_symbols
    updated.update(
        str(row["Symbol"])
        for _, row in edited.iterrows()
        if _radar_bool(row["Attiva Bot"])
    )
    return updated


def sync_market_editor_selection(
    active_tickers: set[str],
    visible_symbols: set[str],
    edited: pd.DataFrame,
    positions: dict,
    bot_enabled: dict[str, bool],
    persist_changes: bool = False,
) -> set[str]:
    updated = merge_market_editor_selection(active_tickers, visible_symbols, edited)
    open_symbols = {_display_symbol(symbol) for symbol in positions}
    for _, row in edited.iterrows():
        symbol = str(row["Symbol"])
        was_enabled = bool(bot_enabled.get(symbol, False))
        was_active = symbol in active_tickers
        checked = _radar_bool(row["Attiva Bot"])
        if checked:
            bot_enabled[symbol] = True
        elif symbol in open_symbols:
            bot_enabled[symbol] = True
            updated.add(symbol)
        else:
            bot_enabled[symbol] = False
        if persist_changes and (
            was_enabled != bool(bot_enabled[symbol])
            or was_active != (symbol in updated)
        ):
            update_symbol_state(symbol, {
                "bot_enabled": bool(bot_enabled[symbol]),
                "active_ticker": symbol in updated,
            })
    return updated


def load_account(client) -> tuple[float | None, dict, str | None]:
    if client is None:
        return None, {}, None
    try:
        return get_account_value(client), get_positions_map(client), None
    except Exception as exc:
        return None, {}, f"Alpaca non raggiungibile (bot e ordini disattivati): {exc}"


def main() -> None:
    global _ACTIVE_TRADING_CLIENT
    st.set_page_config(page_title="Trading Dashboard ~ Watchlist", page_icon="📈", layout="wide")
    init_state()
    client, connect_error = connect_alpaca()
    _ACTIVE_TRADING_CLIENT = client
    equity, positions, account_error = load_account(client)
    available_symbols: list[str] = []
    fractional_symbols: list[str] = []
    if client is not None:
        key, secret, _ = _live_credentials()
        if key and secret:
            try:
                available_symbols = get_tradable_equity_symbols(client, key, secret)
                fractional_symbols = get_fractional_asset_symbols(client, key, secret)
            except Exception as exc:
                st.error(f"Impossibile caricare gli asset Alpaca: {exc}")
    else:
        st.warning(connect_error or NO_KEYS_MESSAGE)

    title_column, version_column = st.columns([5, 1], vertical_alignment="center")
    title_column.title("📈 Trading Dashboard")
    version_column.metric("Versione", APP_VERSION)
    st.caption("Alpaca Paper Trading ~ strategia range: ADX < max, RSI < max, prezzo ≤ Bollinger inferiore "
               "(soglie e rischio dal profilo di ogni titolo)")
    render_header(client, account_error or connect_error, equity, positions)

    selected_defaults, missing_required = prepare_ticker_selection(
        fractional_symbols,
        st.session_state.get("selected_tickers"),
        positions,
        st.session_state["bot_enabled"],
        st.session_state["active_tickers"],
    )
    st.session_state["selected_tickers"] = selected_defaults
    if missing_required:
        st.info(
            "Ticker con bot attivo o posizione aperta mantenuti nella selezione: "
            + ", ".join(missing_required)
        )
    for symbol in selected_defaults:
        st.session_state["bot_enabled"].setdefault(symbol, False)
        st.session_state["profile"].setdefault(symbol, DEFAULT_PROFILE)
    if client is not None:
        key, secret, _ = _live_credentials()
        if key and secret:
            tracked_symbols = set(positions)
            tracked_symbols.update(
                to_alpaca_symbol(symbol)
                for symbol, enabled in st.session_state["bot_enabled"].items()
                if enabled
            )
            tracked_symbols.update(to_alpaca_symbol(symbol) for symbol in st.session_state["active_tickers"])
            tracked_symbols.update(to_alpaca_symbol(symbol) for symbol in selected_defaults)
            for symbol in tracked_symbols:
                if symbol not in _REGISTERED_TRAILING_SYMBOLS:
                    register_trade_update_handler(symbol, _attach_dashboard_trailing_stop)
                    _REGISTERED_TRAILING_SYMBOLS.add(symbol)
            start_trade_update_stream(key, secret, client, sorted(tracked_symbols))

    controls = st.columns([1, 1, 5], vertical_alignment="center")
    controls[0].button("🔄 Aggiorna ora", on_click=_on_refresh_now, width="stretch")
    controls[1].toggle("Auto-refresh", key="auto_refresh")
    selected_symbols = render_market_explorer(positions, available_symbols, client)

    position_symbols = [
        symbol for symbol, position in positions.items()
        if position is not None and getattr(position, "qty", 0) not in (None, 0)
    ]
    active_symbols = get_active_symbols(
        [
            *[symbol for symbol, enabled in st.session_state["bot_enabled"].items() if enabled],
            *st.session_state.get("active_tickers", []),
            *position_symbols,
            *selected_symbols,
            st.session_state.get("detail_symbol_selected"),
        ]
    )
    if client is not None:
        for symbol in active_symbols:
            alpaca_symbol = to_alpaca_symbol(symbol)
            if alpaca_symbol not in _REGISTERED_TRAILING_SYMBOLS:
                register_trade_update_handler(alpaca_symbol, _attach_dashboard_trailing_stop)
                _REGISTERED_TRAILING_SYMBOLS.add(alpaca_symbol)
    refresh_symbols = get_active_symbols(
        [*active_symbols, *st.session_state.get("market_radar_visible_symbols", [])]
    )
    render_refresh_controller(client, equity, positions, refresh_symbols)

    if active_symbols:
        open_orders_by_symbol = {}
        if client is not None:
            try:
                open_orders_by_symbol = {
                    to_alpaca_symbol(symbol): _open_orders(client, to_alpaca_symbol(symbol))
                    for symbol in active_symbols
                }
            except Exception as exc:
                st.error(f"Impossibile leggere gli ordini stop effettivi da Alpaca: {exc}")
        personal_table = build_watchlist_table(
            active_symbols,
            positions,
            broker_orders=open_orders_by_symbol,
        )
        st.session_state["detail_source_table"] = personal_table
        edited_personal = st.data_editor(
            personal_table,
            hide_index=True,
            use_container_width=True,
            disabled=[
                column for column in personal_table.columns
                if column not in {
                    "Bot", "Ticker", "Profilo di Rischio", "Trailing stop %",
                    "Prezzo Massimo Raggiunto ($)", "Prezzo ingresso bot ($)",
                }
            ],
            key="my_ticker_table",
            column_config={
                "Ticker": st.column_config.ButtonColumn(
                    "Ticker",
                    help="Clicca il ticker per aprire il dettaglio.",
                    on_click=_on_select_detail,
                    args=("detail_source_table", "my_ticker_table_click"),
                    key="my_ticker_table_click",
                ),
                "Profilo di Rischio": st.column_config.SelectboxColumn(
                    "Profilo di Rischio",
                    options=PROFILE_NAMES,
                    required=True,
                ),
                "Trailing stop %": st.column_config.NumberColumn(
                    format="%.1f%%",
                    min_value=0.1,
                    max_value=99.0,
                    step=0.1,
                ),
                "Prezzo Massimo Raggiunto ($)": st.column_config.NumberColumn(
                    format="$%.2f",
                    min_value=0.0,
                    step=0.01,
                ),
                "Stop broker ($)": st.column_config.NumberColumn(
                    format="$%.2f",
                    help="Prezzo stop corrente riportato dall'ordine trailing aperto su Alpaca.",
                ),
                "Prezzo attuale ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Var. giornaliera %": st.column_config.NumberColumn(format="%.2f%%"),
                "Prezzo medio acquisto ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Prezzo ingresso bot ($)": st.column_config.NumberColumn(
                    format="$%.2f",
                    min_value=0.01,
                    step=0.01,
                    help="Ingresso eseguito o limite personalizzato per il prossimo ordine bot.",
                ),
                "Stato ingresso bot": st.column_config.TextColumn("Stato ingresso bot"),
                "Quantità": st.column_config.NumberColumn(format="%.4f"),
                "Capitale investito ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Valore attuale ($)": st.column_config.NumberColumn(format="$%.2f"),
                "ADX": st.column_config.NumberColumn(format="%.2f"),
                "RSI": st.column_config.NumberColumn(format="%.2f"),
                "Bot": st.column_config.CheckboxColumn("Bot", help="Attiva o disattiva il bot per questo ticker."),
                "P&L ($)": st.column_config.NumberColumn(format="$%.2f"),
                "P&L posizione %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        edited_personal = edited_personal.copy()
        for _, row in edited_personal.iterrows():
            symbol = str(row["Ticker"])
            set_ticker_bot_enabled(st.session_state, symbol, bool(row["Bot"]))
            profile = str(row["Profilo di Rischio"])
            previous_profile = st.session_state["profile"].get(symbol, DEFAULT_PROFILE)
            original_row = personal_table.loc[personal_table["Ticker"] == symbol].iloc[0]
            requested_trailing_pct = _number_or_nan(row["Trailing stop %"])
            original_trailing_pct = _number_or_nan(original_row["Trailing stop %"])
            trailing_pct_changed = (
                math.isfinite(requested_trailing_pct)
                and not math.isclose(requested_trailing_pct, original_trailing_pct)
            )
            trailing_overrides = st.session_state.setdefault("trailing_stop_pct", {})
            if profile != previous_profile and not trailing_pct_changed:
                trailing_overrides.pop(symbol, None)
            elif math.isfinite(requested_trailing_pct) and 0.1 <= requested_trailing_pct <= 99.0:
                trailing_overrides[symbol] = requested_trailing_pct / 100.0
            if profile in PROFILE_NAMES:
                st.session_state["profile"][symbol] = profile
                st.session_state[f"profile_{symbol}"] = profile
            if profile != previous_profile or trailing_pct_changed:
                try:
                    updated = replace_trailing_stop_percent(
                        client,
                        to_alpaca_symbol(symbol),
                        get_ticker_params(symbol).trailing_pct,
                    ) if client is not None else 0
                    if updated == 0 and positions.get(to_alpaca_symbol(symbol)) is not None:
                        add_log(st.session_state, f"⚠️ {symbol}: nessun trailing stop aperto da aggiornare su Alpaca")
                except Exception as exc:
                    st.session_state["panel_msg"][symbol] = (
                        "error",
                        f"Impostazione salvata ma trailing stop Alpaca non aggiornato: {exc}",
                    )
                    add_log(st.session_state, f"⚠️ {symbol}: replace del trailing Alpaca fallito: {exc}")
                persist_symbol_settings(st.session_state, symbol)
            update_bot_high_water_mark(
                st.session_state,
                symbol,
                row["Prezzo Massimo Raggiunto ($)"],
            )
            requested_entry = _number_or_nan(row["Prezzo ingresso bot ($)"])
            original_entry = _number_or_nan(original_row["Prezzo ingresso bot ($)"])
            entry_overrides = st.session_state.setdefault("bot_entry_price_overrides", {})
            if math.isfinite(requested_entry) and requested_entry > 0:
                if not math.isfinite(original_entry) or not math.isclose(requested_entry, original_entry):
                    entry_overrides[symbol] = requested_entry
                    persist_symbol_settings(st.session_state, symbol)
            elif not math.isfinite(requested_entry):
                if entry_overrides.pop(symbol, None) is not None:
                    persist_symbol_settings(st.session_state, symbol)
        active_tickers = set(st.session_state.get("active_tickers", []))
        active_tickers = {symbol for symbol in active_tickers if symbol in set(edited_personal["Ticker"].astype(str))} | set(
            edited_personal[edited_personal["Bot"] == True]["Ticker"].astype(str)
        )
        st.session_state["active_tickers"] = sorted(active_tickers)

        detail_options = edited_personal["Ticker"].astype(str).tolist()
        detail_symbol = st.session_state.get("detail_symbol_selected")
        if detail_symbol not in detail_options:
            detail_symbol = detail_options[0]
            st.session_state["detail_symbol_selected"] = detail_symbol
        actions = st.columns([1, 1, 1], vertical_alignment="center")
        actions[0].button(
            "Compra",
            key=f"quick_buy_{detail_symbol}",
            on_click=_on_open_panel,
            args=(detail_symbol, "buy", {
                f"buy_limit_{detail_symbol}": 0.0,
                f"buy_budget_{detail_symbol}": 100.0,
                f"buy_confirm_{detail_symbol}": False,
            }),
        )
        actions[1].button(
            "Vendi",
            key=f"quick_sell_{detail_symbol}",
            disabled=positions.get(to_alpaca_symbol(detail_symbol)) is None,
            on_click=_on_open_panel,
            args=(detail_symbol, "sell", {f"sell_shares_{detail_symbol}": 0.0001}),
        )
        actions[2].button("📊 Backtest", key=f"quick_backtest_{detail_symbol}", on_click=_on_toggle_backtest, args=(detail_symbol,))
        st.markdown(f"### Dettaglio {detail_symbol}")
        render_symbol_card(client, equity, detail_symbol, positions)
    else:
        st.info("Nessun ticker attivo. Attiva un bot o un ticker dalla tabella radar.")

    if any(st.session_state["bot_enabled"].values()):
        st.info("🤖 I bot girano solo mentre questa pagina è aperta nel browser.")

    with st.expander("Log Bot", expanded=False):
        entries = st.session_state["bot_log"][-20:]
        if entries:
            st.code("\n".join(reversed(entries)), language=None)
        else:
            st.caption("Nessuna azione registrata.")

if __name__ == "__main__":
    main()
