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
from live_trader import get_account_value, get_alpaca_client, get_recent_orders, run_signal_check

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


def run_dashboard_backtest(raw: pd.DataFrame, params: StrategyParams, portfolio_value: float) -> tuple[pd.DataFrame, BacktestResult]:
    if raw.empty: raise ValueError("No valid OHLCV rows.")
    prepared = prepare_data(raw)
    if prepared.empty: raise ValueError("No valid OHLCV rows.")
    return prepared, run_backtest(prepared, params, portfolio_value, symbol="SPY")


def price_figure(prices: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=prices.index, open=prices["Open"], high=prices["High"], low=prices["Low"], close=prices["Close"], name="Price", increasing_line_color="#16a34a", decreasing_line_color="#dc2626"))
    for column, color in (("BB_upper", "#818cf8"), ("BB_mid", "#f59e0b"), ("BB_lower", "#818cf8")):
        fig.add_trace(go.Scatter(x=prices.index, y=prices[column], mode="lines", name=column.replace("_", " "), line=dict(color=color, width=1.5, dash="dot" if column != "BB_mid" else "solid")))
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


def trades_table(trades: pd.DataFrame) -> pd.io.formats.style.Styler:
    view = trades.loc[:, list(TRADE_VIEW_COLUMNS)].copy()
    for c in ("entry_date", "exit_date"): view[c] = pd.to_datetime(view[c]).dt.strftime("%Y-%m-%d")
    def color_row(row):
        c = "background-color:rgba(34,197,94,0.15)" if row["net_pnl"]>0 else ("background-color:rgba(239,68,68,0.15)" if row["net_pnl"]<0 else "")
        return [c]*len(row)
    return view.style.apply(color_row, axis=1).format({"entry_price":"${:,.2f}","exit_price":"${:,.2f}","net_pnl":"${:,.2f}"})


def _live_keys():
    try:
        k = st.secrets.get("ALPACA_API_KEY"); s = st.secrets.get("ALPACA_SECRET_KEY")
    except Exception:
        k = s = None
    return os.environ.get("ALPACA_API_KEY") or k, os.environ.get("ALPACA_SECRET_KEY") or s


def render_live_trading():
    with st.expander("Live Trading (Paper)"):
        st.caption("Read-only account view · Check Signal Now does not place an order")
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
        sym = st.text_input("Symbol for signal check", value="SPY").strip().upper()
        if st.button("Check Signal Now"):
            if not sym: st.warning("Enter a symbol first."); return
            try:
                signal = run_signal_check(sym, account_value, StrategyParams())
            except Exception:
                st.error("Could not check signal."); return
            if signal is None:
                st.info(f"No entry signal for {sym} on the latest daily bar.")
            else:
                st.success(f"{sym}: buy limit ${signal.limit_price:.2f} · {signal.shares} shares · stop ${signal.stop_loss:.2f} (not submitted)")


def main():
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
