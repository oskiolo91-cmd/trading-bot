"""Interactive dashboard for the existing candle-by-candle trading backtest."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from alpaca.trading.client import TradingClient

BOT_DIR = Path(__file__).resolve().parent
DATA_DIR = BOT_DIR / "data"

if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from backtest import BacktestResult, load_ohlcv_csv, prepare_data, run_backtest
from models import StrategyParams
from risk import calc_position_size
from live_trader import (
    get_account_value, get_alpaca_client, get_latest_bars,
    get_recent_orders, place_limit_buy, run_signal_check,
)

TRADE_VIEW_COLUMNS = (
    "entry_date", "exit_date", "entry_price", "exit_price", "shares", "net_pnl", "exit_reason",
)
EXIT_COLORS = {
    "trailing_stop": "#22c55e",
    "time_stop": "#f59e0b",
    "stop_loss": "#ef4444",
    "end_of_data": "#94a3b8",
}


@st.cache_data
def load_data(path: str, modified_ns: int) -> pd.DataFrame:
    return load_ohlcv_csv(path)


def run_dashboard_backtest(raw: pd.DataFrame, params: StrategyParams, portfolio_value: float):
    if raw.empty: raise ValueError("No valid OHLCV rows.")
    prepared = prepare_data(raw)
    if prepared.empty: raise ValueError("No valid OHLCV rows.")
    return prepared, run_backtest(prepared, params, portfolio_value, symbol="SPY")


def price_figure(prices: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=prices.index, open=prices["Open"], high=prices["High"],
        low=prices["Low"], close=prices["Close"], name="Price",
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
    ))
    for col, color in (("BB_upper", "#818cf8"), ("BB_mid", "#f59e0b"), ("BB_lower", "#818cf8")):
        fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode="lines", name=col.replace("_", " "), line=dict(color=color, width=1.5, dash="dot" if col != "BB_mid" else "solid")))
    if not trades.empty:
        fig.add_trace(go.Scatter(x=trades["entry_date"], y=trades["entry_price"], mode="markers", name="Buy", marker=dict(symbol="triangle-up", color="#22c55e", size=12)))
        for reason, color in EXIT_COLORS.items():
            exits = trades.loc[trades["exit_reason"] == reason]
            if not exits.empty:
                fig.add_trace(go.Scatter(x=exits["exit_date"], y=exits["exit_price"], mode="markers", name=f"Sell {reason}", marker=dict(symbol="triangle-down", color=color, size=12)))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)", xaxis_rangeslider_visible=False, hovermode="x unified", height=580)
    return fig


def equity_figure(equity: pd.Series) -> go.Figure:
    peak = equity.cummax()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=peak, mode="lines", name="peak", line=dict(color="rgba(239,68,68,0)", width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=equity.index, y=equity, mode="lines", name="Drawdown", line=dict(color="rgba(239,68,68,0)", width=0), fill="tonexty", fillcolor="rgba(239,68,68,0.16)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=equity.index, y=equity, mode="lines", name="Portfolio", line=dict(color="#2563eb", width=2), hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio ($)", hovermode="x unified", height=380)
    return fig


def trades_table(trades: pd.DataFrame):
    view = trades.loc[:, list(TRADE_VIEW_COLUMNS)].copy()
    for c in ("entry_date", "exit_date"): view[c] = pd.to_datetime(view[c]).dt.strftime("%Y-%m-%d")
    def color_row(row):
        c = "background-color:rgba(34,197,94,0.15)" if row["net_pnl"] > 0 else ("background-color:rgba(239,68,68,0.15)" if row["net_pnl"] < 0 else "")
        return [c] * len(row)
    return view.style.apply(color_row, axis=1).format({"entry_price": "${:,.2f}", "exit_price": "${:,.2f}", "net_pnl": "${:,.2f}"})


def _live_keys():
    try:
        k = st.secrets.get("ALPACA_API_KEY"); s = st.secrets.get("ALPACA_SECRET_KEY")
    except Exception:
        k = s = None
    return os.environ.get("ALPACA_API_KEY") or k, os.environ.get("ALPACA_SECRET_KEY") or s


def _indicator_color_adx(value: float) -> str:
    return "#22c55e" if value < 25 else "#ef4444"


def _indicator_color_rsi(value: float) -> str:
    if value < 35: return "#22c55e"
    if value < 50: return "#f59e0b"
    return "#ef4444"


def render_signal_monitor(client: TradingClient, account_value: float) -> None:
    """Signal monitor + manual trading override panel."""
    st.subheader("Monitor segnale & ordine manuale")
    sym = st.text_input("Ticker", value="SPY", placeholder="es. AAPL, SPY").strip().upper()
    analyze = st.button("Analizza", type="secondary")
    if not analyze:
        return
    if not sym:
        st.warning("Inserisci un ticker."); return
    try:
        bars = get_latest_bars(sym)
        data = prepare_data(bars)
    except Exception as exc:
        st.error(f"Errore nel caricamento dati per {sym}: {exc}"); return
    if data.empty:
        st.warning(f"Nessun dato per {sym}."); return
    row = data.iloc[-1]
    adx = float(row["ADX"]) if not pd.isna(row["ADX"]) else None
    rsi = float(row["RSI"]) if not pd.isna(row["RSI"]) else None
    bb_lower = float(row["BB_lower"]) if not pd.isna(row["BB_lower"]) else None
    bb_mid = float(row["BB_mid"]) if not pd.isna(row["BB_mid"]) else None
    bb_upper = float(row["BB_upper"]) if not pd.isna(row["BB_upper"]) else None
    atr = float(row["ATR"]) if not pd.isna(row["ATR"]) else None
    price = float(row["Close"])
    st.write("**Indicatori ultima candela giornaliera**")
    cols = st.columns(6)
    cols[0].metric("Prezzo", f"${price:,.2f}")
    cols[1].metric("ADX", f"{adx:.1f}" if adx is not None else "n/a", help="<25: mercato laterale (ok)")
    cols[2].metric("RSI", f"{rs:&.1f}" if rsi is not None else "n/a", help="<35: ipervenduto (ok)")
    cols[3].metric("BB Lower", f"${bb_lower:,.2f}" if bb_lower is not None else "n/a")
    cols[4].metric("BB Mid", f"${bb_mid:,.2f}" if bb_mid is not None else "n/a")
    cols[5].metric("ATR", f"{atr:,.2f}" if atr is not None else "n/a")
    c1 = adx is not None and adx < 25
    c2 = bsol= price <= bb_lower if bb_lower is not None else False
    c3 = rsi is not None and rsi < 35
    score = sum([c1, c2, c3])
    if score == 3:
        st.success("<b>😂&৳ Segnale ATTIVO -- tutte le condizioni soddisfatte</b>", icon="")
    elif score > 0:
        conds = [f"ADX<25: {'✓' if c1 else 'x'}", f"Prezzo≥BB Lower: {'✓' if c2 else 'x'}", f"RSI<35: {'✓' if c3 else 'x'}"]
        st.warning(f"Segnale PARZIALE -- {score}/3 condizioni: {', '.join(conds)}")
    else:
        st.error("Nessun segnale -- condizioni non soddisfatte")
    if bb_lower is not None and atr is not None:
        default_stop = round(max(bb_lower - 2 * atr, 0.01), 2)
        default_shares = calc_position_size(account_value, bb_lower, default_stop)
    else:
        default_stop = round(max(price * 0.97, 0.01), 2)
        default_shares = 1
    st.divider()
    st.write("**Ordine automatico (segnale strategia)**")
    if score == 3:
        if st.button(f"Esegui ordine automatico {sym}", type="primary"):
            try:
                signal = run_signal_check(sym, account_value)
                if signal is None:
                    st.warning("Segnale non validato al novo controllo.")
                else:
                    oid = place_limit_buy(client, sym, signal.limit_price, signal.shares, signal.stop_loss)
                    st.success(f"Ordine piaziato -- id: {oid} | limit: ${signal.limit_price:.2f} | stop: ${signal.stop_loss:.2f} | {_signal.shares} azioni")
        except Exception as exc:
            st.error(f"Errore ordine: {exc}")
    else:
        st.button(f"Esegui ordine automatico {sym}", disabled=True, help="Disabilitato: non tutte le condizioni soddisfatte")
    with st.expander("Override manuale"):
        st.caption("Usa questa sezione per piazjare un ordine anche senza segnale automatico.")
        man_limit = st.number_input("Prezfo limite ($)", min_value=0.01, value=float(round(bb_lower or price * 0.99, 2)), step=0.01)
        man_stop = st.number_input("Stop loss ($)", min_value=0.01, value=float(default_stop), step=0.01)
        man_shares = st.number_input("Numero azioni", min_value=1, value=int(max(default_shares, 1)), step=1)
        confirm = st.checkbox("Confermo di voler comprare anche senza segnale automatico")
        if st.button("Compra ora (manuale)", type="primary", disabled=not confirm):
            if man_stop >= man_limit:
                st.error("Lo stop loss deve essere inferiore al prezzo limite.")
            else:
                try:
                    oid = place_limit_buy(client, sym, man_limit, int(man_shares), man_stop)
                    st.success(f"Ordine manuale piazzato -- id: {oid} | {sym} | limit: ${man_limit:.2f} | stop: ${man_stop:.2f} | {int(man_shares)} azioni")
                except Exception as exc:
                    st.error(f"Errore ordine manuale: {exc}")


def render_live_trading() -> None:
    with st.expander("Live Trading (Paper)"):
        st.caption("Read-only account view & ordini su Alpaca Paper")
        key, secret = _live_keys()
        if not key or not secret:
            st.error("API keys not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in Streamlit secrets.")
            return
        try:
            client = TradingClient(api_key=key, secret_key=secret, paper=True)
            account_value = get_account_value(client)
            positions = client.get_all_positions()
            orders = get_recent_orders(client)
        except Exception:
            st.error("Could not reach Alpaca. Check credentials and connection.")
            return
        st.metric("Account equity", f"${account_value:,.2f}")
        st.subheader("Open positions")
        if positions:
            st.dataframe(pd.DataFrame([{"Symbol": p.symbol, "Shares": p.qty, "Avg entry ($)": p.avg_entry_price, "Market value ($)": p.market_value, "Unrealized P&L ($)": p.unrealized_pl} for p in positions]), hide_index=True, use_container_width=True)
        else:
            st.info("No open positions.")
        st.subheader("Recent orders (last 10)")
        if orders:
            st.dataframe(pd.DataFrame(orders), hide_index=True, use_container_width=True)
        else:
            st.info("No recent orders.")
        st.divider()
        render_signal_monitor(client, account_value)


def main() -> None:
    st.set_page_config(page_title="Trading Bot · Backtest", page_icon="📈", layout="wide")
    st.title("Trading Bot · Backtest")
    st.caption("SPY daily data · Bollinger range strategy with trailing stop")
    files = sorted(DATA_DIR.rglob("*.csv")) if DATA_DIR.exists() else []
    if not files:
        st.warning("No CSV files found in data/."); render_live_trading(); return
    options = [str(p.name) for p in files]
    with st.sidebar:
        st.header("Backtest settings")
        selected = st.selectbox("CSV file", options, index=0)
        risk = st.slider("Risk %", 0.1, 3.0, 1.0, 0.1, format="%.1f%%")
        trailing = st.slider("Trailing stop %", 1.0, 10.0, 3.0, 0.5, format="%.1f%%")
        time_stop = st.slider("Time-stop candles", 5, 30, 10)
        commission = st.slider("Commission %", 0.05, 0.5, 0.1, 0.05, format="%.2f%%")
        portfolio = st.number_input("Portfolio ($)", min_value=1.0, value=100_000.0, step=1_000.0)
        run = st.button("Run Backtest", type="primary")
    if not run:
        st.info("Set parameters and click Run Backtest."); render_live_trading(); return
    params = StrategyParams(risk_pct=risk/100, trailing_pct=trailing/100, time_stop=time_stop, commission_pct=commission/100)
    selected_path = DATA_DIR / selected
    try:
        raw = load_data(str(selected_path), selected_path.stat().st_mtime_ns)
        prepared, result = run_dashboard_backtest(raw, params, portfolio)
    except Exception as exc:
        st.error(f"Error: {exc}"); return
    m = result.metrics
    cols = st.columns(4)
    cols[0].metric("Net P&L", f"${m['net_pnl_total']:,.2f}")
    cols[1].metric("Win Rate", f"{m['win_rate']:.1%}")
    cols[2].metric("Max Drawdown", f"{m['max_drawdown']:.1%}")
    cols[3].metric("Total Trades", str(m['total_trades']))
    if result.trades.empty: st.warning("No trades found.")
    st.subheader("Price & signals")
    st.plotly_chart(price_figure(prepared, result.trades), use_container_width=True)
    st.subheader("Equity curve")
    st.plotly_chart(equity_figure(result.equity), use_container_width=True)
    st.subheader("Trades")
    st.dataframe(trades_table(result.trades), use_container_width=True, hide_index=True)
    render_live_trading()


if __name__ == "__main__":
    main()
