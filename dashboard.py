"""Watchlist trading dashboard: live indicators, per-ticker bots and manual paper orders."""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide

BOT_DIR = Path(__file__).resolve().parent
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from backtest import BacktestResult, load_ohlcv_csv, prepare_data, run_backtest
from models import StrategyParams
from live_trader import (
    _check_exit_once, _initial_position, _open_orders, get_account_value, get_alpaca_client,
    get_latest_bars, get_recent_orders, place_limit_buy, place_market_sell, run_signal_check,
)
from risk import calc_position_size

# Auto-download SPY data on first run
try:
    _data_path = BOT_DIR / "data" / "SPY.csv"
    if not _data_path.exists():
        _data_path.parent.mkdir(exist_ok=True)
        _df = yf.download("SPY", start="2020-01-01", auto_adjust=False, progress=False)
        _df.columns = [c[0] if isinstance(c, tuple) else c for c in _df.columns]
        _df.index.name = "Date"
        _df.reset_index().to_csv(_data_path, index=False)
except Exception:
    pass


WATCHLIST = {
    "Big Tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "AMD", "INTC", "ORCL"],
    "Difesa": ["LMT", "RTX", "NOC", "GD", "BA", "HII", "LHX", "AXON", "KTOS"],
    "Energia / Materie prime": ["XOM", "CVX", "COP", "SLB", "OXY", "FCX", "NEM", "AA", "CLF"],
    "ETF": ["SPY", "QQQ", "DIA", "GLD", "SLV", "USO", "XLE", "XLK", "XLI"],
    "Finance": ["JPM", "GS", "BRK-B", "V", "MA"],
}
ALL_SYMBOLS = [symbol for symbols in WATCHLIST.values() for symbol in symbols]
REFRESH_SECONDS = 60
LOG_LIMIT = 200
ROW_WIDTHS = [1.6, 1, 1, 1.2, 1.5, 1.8]
PARAMS = StrategyParams()
PROFILES = {
    "🐢 Conservativo": StrategyParams(adx_max=20, rsi_max=30, risk_pct=0.005, atr_mult=2.5),
    "⚖️ Bilanciato": StrategyParams(adx_max=25, rsi_max=35, risk_pct=0.010, atr_mult=2.0),
    "🚀 Speculativo": StrategyParams(adx_max=35, rsi_max=45, risk_pct=0.020, atr_mult=1.5),
    "🎛️ Custom": None,
}
PROFILE_NAMES = list(PROFILES.keys())
DEFAULT_PROFILE = "⚖️ Bilanciato"
CUSTOM_PROFILE = "🎛️ Custom"
BACKTEST_PERIOD = "90d"
EASTERN = ZoneInfo("America/New_York")
NO_KEYS_MESSAGE = (
    "Chiavi Alpaca non configurate: aggiungi ALPACA_API_KEY e ALPACA_SECRET_KEY nei Secrets "
    "dell'app (Streamlit Cloud → Settings → Secrets). Bot e ordini sono disattivati, "
    "la watchlist resta consultabile."
)


def to_alpaca_symbol(symbol: str) -> str:
    """Yahoo uses BRK-B, Alpaca uses BRK.B."""
    return symbol.replace("-", ".")


def fetch_ticker_data(symbol: str) -> dict:
    frame = yf.download(symbol, period="60d", interval="1d", auto_adjust=False, progress=False)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if frame.empty:
        raise ValueError(f"Nessun dato da Yahoo Finance per {symbol}")
    frame = frame.copy()
    # Orders are placed on the raw market tape, so indicators use unadjusted prices (as live_trader does).
    frame["Adj Close"] = frame["Close"]
    data = prepare_data(frame)
    if data.empty:
        raise ValueError(f"Nessuna candela valida per {symbol}")
    row = data.iloc[-1]
    values = {
        "price": float(row["Close"]),
        "prev_close": float(data["Close"].iloc[-2]) if len(data) > 1 else float("nan"),
        "adx": float(row["ADX"]), "rsi": float(row["RSI"]),
        "bb_lower": float(row["BB_lower"]), "bb_mid": float(row["BB_mid"]), "bb_upper": float(row["BB_upper"]),
        "atr": float(row["ATR"]),
    }
    values["signal"] = compute_signal(values, PARAMS)
    return values


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
    if params is not None:
        return params
    base = PROFILES[DEFAULT_PROFILE]
    return replace(
        base,
        adx_max=float(state.get(f"custom_adx_{symbol}", base.adx_max)),
        rsi_max=float(state.get(f"custom_rsi_{symbol}", base.rsi_max)),
        risk_pct=float(state.get(f"custom_risk_{symbol}", base.risk_pct * 100)) / 100,
        atr_mult=float(state.get(f"custom_atr_{symbol}", base.atr_mult)),
    )


def run_profile_backtest(symbol: str) -> dict[str, BacktestResult]:
    """Run backtest for all 3 preset profiles on 90 days of data."""
    frame = yf.download(symbol, period=BACKTEST_PERIOD, interval="1d", auto_adjust=False, progress=False)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if frame.empty:
        raise ValueError(f"Nessun dato da Yahoo Finance per {symbol}")
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


def fetch_all_tickers(symbols: list, on_progress=None) -> dict:
    results: dict[str, dict] = {}
    for index, symbol in enumerate(symbols, start=1):
        try:
            results[symbol] = {**fetch_ticker_data(symbol), "last_updated": datetime.now()}
        except Exception as exc:
            results[symbol] = {"error": str(exc), "signal": False, "last_updated": datetime.now()}
        if on_progress is not None:
            on_progress(index, len(symbols), symbol)
    return results


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


def order_plan(data: dict, equity: float, params: StrategyParams) -> tuple[float, float, int]:
    """Default limit (BB lower), stop (limit - atr_mult ATR) and share count for a buy."""
    limit = round(data["bb_lower"], 2)
    stop = round(data["bb_lower"] - params.atr_mult * data["atr"], 2)
    shares = calc_position_size(equity, limit, stop, params.risk_pct, params.max_cap_pct)
    if limit > 0:
        shares = min(shares, math.floor(equity / (limit * (1 + params.commission_pct))))
    return limit, stop, max(0, shares)


def add_log(state, message: str) -> None:
    log = state.setdefault("bot_log", [])
    log.append(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}")
    del log[:-LOG_LIMIT]


def cancel_open_sells(client, alpaca_symbol: str, attempts: int = 10) -> None:
    """Protective stop legs lock the shares, so they must be cancelled before a market sell."""
    for order in _open_orders(client, alpaca_symbol):
        if order.side == OrderSide.SELL:
            client.cancel_order_by_id(order.id)
    for _ in range(attempts):
        if not any(order.side == OrderSide.SELL for order in _open_orders(client, alpaca_symbol)):
            return
        time.sleep(0.5)
    raise RuntimeError("lo stop-loss non risulta ancora annullato, riprova tra qualche secondo")


def run_bot_cycle(client, state, positions: dict, equity: float) -> None:
    """One bot pass over every enabled ticker: exit checks on open positions, entries on signals."""
    enabled = [symbol for symbol, on in state.get("bot_enabled", {}).items() if on]
    if not enabled:
        return
    try:
        market_open = bool(client.get_clock().is_open)
    except Exception as exc:
        add_log(state, f"🤖 Impossibile leggere l'orario di mercato: {exc}")
        return
    today = datetime.now(EASTERN).date()
    bot_state = state.setdefault("bot_state", {})
    last_buy = state.setdefault("bot_last_buy", {})
    for symbol in enabled:
        alpaca_symbol = to_alpaca_symbol(symbol)
        data = state.get("live_data", {}).get(symbol, {})
        params = get_ticker_params(symbol, state)
        try:
            if alpaca_symbol in positions:
                if not market_open:
                    continue
                exit_state = bot_state.setdefault(symbol, {})
                if "position" not in exit_state:
                    position = _initial_position(client, alpaca_symbol)
                    if position is None:
                        if not exit_state.get("warned"):
                            add_log(state, f"🤖 {symbol}: posizione senza stop protettivo, gestiscila a mano")
                            exit_state["warned"] = True
                        continue
                    exit_state["position"] = position
                decision = _check_exit_once(client, alpaca_symbol, exit_state, params)
                if decision is not None:
                    add_log(state, f"🤖 {symbol}: uscita {decision.reason.value} a ~${decision.price:,.2f}")
                continue
            bot_state.pop(symbol, None)
            if not compute_signal(data, params) or last_buy.get(symbol) == today:
                continue
            if any(order.side == OrderSide.BUY for order in _open_orders(client, alpaca_symbol)):
                continue
            limit, stop, shares = order_plan(data, equity, params)
            last_buy[symbol] = today
            if shares <= 0 or not 0 < stop < limit:
                add_log(state, f"🤖 {symbol}: segnale attivo ma ordine non dimensionabile")
                continue
            order_id = place_limit_buy(client, alpaca_symbol, limit, shares, stop)
            add_log(state, f"🤖 {symbol}: COMPRA limite {shares} az. @ ${limit:,.2f} ~ stop ${stop:,.2f} (ID {order_id})")
        except Exception as exc:
            add_log(state, f"🤖 {symbol}: errore {exc}")


def init_state() -> None:
    defaults = {
        "bot_enabled": {symbol: False for symbol in ALL_SYMBOLS},
        "live_data": {}, "bot_log": [], "bot_state": {}, "bot_last_buy": {},
        "panels": {}, "panel_msg": {}, "bot_notice": {}, "auto_refresh": True,
        "profile": {symbol: DEFAULT_PROFILE for symbol in ALL_SYMBOLS},
        "backtest_open": {},
        "backtest_cache": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def needs_refresh() -> bool:
    last = st.session_state.get("last_refresh")
    return last is None or (datetime.now() - last).total_seconds() >= REFRESH_SECONDS


def _on_refresh_now() -> None:
    st.session_state["last_refresh"] = None


def _on_toggle(symbol: str) -> None:
    enabled = bool(st.session_state.get(f"bot_{symbol}"))
    st.session_state["bot_enabled"][symbol] = enabled
    st.session_state["bot_notice"][symbol] = enabled
    add_log(st.session_state, f"🤖 {symbol}: bot {'ATTIVATO' if enabled else 'disattivato'}")


def _on_profile_change(symbol: str) -> None:
    name = st.session_state.get(f"profile_{symbol}", DEFAULT_PROFILE)
    st.session_state["profile"][symbol] = name
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


def _on_submit_buy(client, symbol: str) -> None:
    limit = float(st.session_state[f"buy_limit_{symbol}"])
    stop = float(st.session_state[f"buy_stop_{symbol}"])
    shares = int(st.session_state[f"buy_shares_{symbol}"])
    try:
        order_id = place_limit_buy(client, to_alpaca_symbol(symbol), limit, shares, stop)
    except Exception as exc:
        st.session_state["panel_msg"][symbol] = ("error", f"Ordine non inviato: {exc}")
        return
    st.session_state["panels"].pop(symbol, None)
    text = f"Ordine limite inviato: {shares} az. @ ${limit:,.2f} ~ stop ${stop:,.2f} (ID {order_id})"
    st.session_state["panel_msg"][symbol] = ("success", text)
    add_log(st.session_state, f"👤 {symbol}: {text}")


def _on_submit_sell(client, symbol: str) -> None:
    shares = int(st.session_state[f"sell_shares_{symbol}"])
    alpaca_symbol = to_alpaca_symbol(symbol)
    try:
        cancel_open_sells(client, alpaca_symbol)
        order_id = place_market_sell(client, alpaca_symbol, shares)
    except Exception as exc:
        st.session_state["panel_msg"][symbol] = ("error", f"Vendita non inviata: {exc}")
        return
    st.session_state["panels"].pop(symbol, None)
    st.session_state["bot_state"].pop(symbol, None)
    text = f"Vendita a mercato inviata: {shares} az. (ID {order_id})"
    st.session_state["panel_msg"][symbol] = ("success", text)
    add_log(st.session_state, f"👤 {symbol}: {text}")


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
    cols = st.columns(4)
    bots = sum(bool(on) for on in st.session_state["bot_enabled"].values())
    if client is None or equity is None:
        cols[0].metric("Account Equity", "—")
        cols[1].metric("Posizioni aperte", "—")
        cols[2].metric("P&L oggi (non realizzato)", "—")
        cols[3].metric("Bot attivi", bots)
        st.warning(account_error or NO_KEYS_MESSAGE)
        return
    today_pl = 0.0
    for position in positions.values():
        value = getattr(position, "unrealized_intraday_pl", None)
        if value is None:
            value = getattr(position, "unrealized_pl", 0)
        today_pl += float(value or 0)
    cols[0].metric("Account Equity", f"${equity:,.2f}")
    cols[1].metric("Posizioni aperte", len(positions))
    cols[2].metric("P&L oggi (non realizzato)", f"${today_pl:,.2f}", delta=f"{today_pl:,.2f}")
    cols[3].metric("Bot attivi", bots)
    if account_error:
        st.warning(account_error)


def render_buy_panel(client, symbol: str) -> None:
    with st.container(border=True):
        st.markdown(f"**Compra {symbol}** ~ ordine limite DAY con stop-loss protettivo")
        cols = st.columns(3)
        cols[0].number_input("Prezzo limite ($)", min_value=0.0, step=0.01, key=f"buy_limit_{symbol}")
        cols[1].number_input("Stop loss ($)", min_value=0.0, step=0.01, key=f"buy_stop_{symbol}")
        cols[2].number_input("Azioni", min_value=0, step=1, key=f"buy_shares_{symbol}")
        confirmed = st.checkbox("Confermo l'ordine", key=f"buy_confirm_{symbol}")
        st.button("Invia ordine", key=f"buy_submit_{symbol}", type="primary", disabled=not confirmed,
                  on_click=_on_submit_buy, args=(client, symbol))


def render_sell_panel(client, symbol: str, qty: int) -> None:
    with st.container(border=True):
        st.markdown(f"**Vendi {symbol}** ~ ordine a mercato")
        st.number_input("Azioni da vendere", min_value=1, max_value=max(1, qty), step=1, key=f"sell_shares_{symbol}")
        st.caption("Gli stop-loss aperti su questo titolo verranno annullati prima della vendita.")
        st.button("Conferma vendita", key=f"sell_submit_{symbol}", type="primary",
                  on_click=_on_submit_sell, args=(client, symbol))


def render_custom_sliders(symbol: str) -> None:
    base = PROFILES[DEFAULT_PROFILE]
    defaults = {f"custom_adx_{symbol}": float(base.adx_max), f"custom_rsi_{symbol}": float(base.rsi_max),
                f"custom_risk_{symbol}": base.risk_pct * 100, f"custom_atr_{symbol}": float(base.atr_mult)}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    with st.container(border=True):
        st.caption(f"🎛️ Profilo Custom ~ {symbol}")
        cols = st.columns(4)
        cols[0].slider("ADX max", 15.0, 50.0, step=1.0, key=f"custom_adx_{symbol}")
        cols[1].slider("RSI max", 25.0, 60.0, step=1.0, key=f"custom_rsi_{symbol}")
        cols[2].slider("Risk %", 0.1, 5.0, step=0.1, key=f"custom_risk_{symbol}")
        cols[3].slider("Stop ATR mult", 1.0, 3.0, step=0.1, key=f"custom_atr_{symbol}")


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
                     on_change=_on_profile_change, args=(symbol,))
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
            qty = int(float(position.qty))
            st.button("Vendi", key=f"sell_{symbol}", disabled=not trading_ok, on_click=_on_open_panel,
                      args=(symbol, "sell", {f"sell_shares_{symbol}": max(1, qty)}))
        else:
            defaults = {}
            if trading_ok and "price" in data and math.isfinite(data["atr"]) and math.isfinite(data["bb_lower"]):
                limit, stop, shares = order_plan(data, equity, params)
                defaults = {f"buy_limit_{symbol}": max(0.0, limit), f"buy_stop_{symbol}": max(0.0, stop),
                            f"buy_shares_{symbol}": shares, f"buy_confirm_{symbol}": False}
            st.button("Compra", key=f"buy_{symbol}", disabled=not defaults, on_click=_on_open_panel,
                      args=(symbol, "buy", defaults))
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
        render_sell_panel(client, symbol, int(float(position.qty)))
    if st.session_state["backtest_open"].get(symbol):
        render_backtest_panel(symbol)


def render_watchlist(client, equity: float | None, positions: dict) -> None:
    for category, symbols in WATCHLIST.items():
        signals = sum(compute_signal(st.session_state["live_data"].get(symbol, {}), get_ticker_params(symbol))
                      for symbol in symbols)
        with st.expander(f"{category} ~ {len(symbols)} titoli ~ {signals} segnali", expanded=True):
            head = st.columns(ROW_WIDTHS)
            for column, label in zip(head, ("Titolo / Prezzo", "ADX", "RSI", "Segnale", "Profilo / Bot", "Azioni")):
                column.caption(label)
            for symbol in symbols:
                render_ticker_row(symbol, client, equity, positions)


def load_account(client) -> tuple[float | None, dict, str | None]:
    if client is None:
        return None, {}, None
    try:
        return get_account_value(client), get_positions_map(client), None
    except Exception as exc:
        return None, {}, f"Alpaca non raggiungibile (bot e ordini disattivati): {exc}"


def auto_refresh_countdown() -> None:
    placeholder = st.empty()
    while True:
        last = st.session_state.get("last_refresh") or datetime.now()
        remaining = REFRESH_SECONDS - (datetime.now() - last).total_seconds()
        if remaining <= 0:
            break
        placeholder.caption(f"⏱️ Prossimo aggiornamento automatico tra {math.ceil(remaining)}s")
        time.sleep(1)
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Trading Dashboard ~ Watchlist", page_icon="📈", layout="wide")
    init_state()
    st.title("📈 Trading Dashboard")
    st.caption("Alpaca Paper Trading ~ strategia range: ADX < max, RSI < max, prezzo ≤ Bollinger inferiore "
               "(soglie e rischio dal profilo di ogni titolo)")

    client, connect_error = connect_alpaca()
    equity, positions, account_error = load_account(client)
    render_header(client, account_error or connect_error, equity, positions)

    controls = st.columns([3, 1, 1], vertical_alignment="center")
    controls[1].button("🔄 Aggiorna ora", on_click=_on_refresh_now, width="stretch")
    controls[2].toggle("Auto-refresh 60s", key="auto_refresh")

    if needs_refresh():
        progress = st.progress(0.0, text="Scarico i dati della watchlist…")
        st.session_state["live_data"] = fetch_all_tickers(
            ALL_SYMBOLS,
            lambda done, total, symbol: progress.progress(done / total, text=f"Scarico {symbol} ({done}/{total})…"),
        )
        progress.empty()
        if client is not None and equity is not None:
            run_bot_cycle(client, st.session_state, positions, equity)
        st.session_state["last_refresh"] = datetime.now()
    controls[0].caption(f"Ultimo aggiornamento dati: {st.session_state['last_refresh']:%H:%M:%S}")
    if any(st.session_state["bot_enabled"].values()):
        st.info("🤖 I bot girano solo mentre questa pagina è aperta nel browser.")

    render_watchlist(client, equity, positions)

    with st.expander("Log Bot", expanded=False):
        entries = st.session_state["bot_log"][-20:]
        if entries:
            st.code("\n".join(reversed(entries)), language=None)
        else:
            st.caption("Nessuna azione registrata.")

    if st.session_state["auto_refresh"]:
        auto_refresh_countdown()


if __name__ == "__main__":
    main()
