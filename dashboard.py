"""Interactive dashboard for the existing candle-by-candle trading backtest."""

from __future__ import annotations

import sys
from pathlib import Path

# Auto-download SPY if not present
try:
    from download_data import ensure_spy_data
    ensure_spy_data()
except Exception:
    pass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BOT_DIR = Path(__file__).resolve().parent
DATA_DIR = BOT_DIR / "data"

if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from backtest import BacktestResult, load_ohlcv_csv, prepare_data, run_backtest
from models import StrategyParams

TRADE_VIEW_COLUMNS = ("entry_date", "exit_date", "entry_price", "exit_price", "shares", "net_pnl", "exit_reason")
EXIT_COLORS = {"trailing_stop": "#22c55e", "time_stop": "#f59e0b", "stop_loss": "#ef4444", "end_of_data": "#94a3b8"}


@st.cache_data
def load_data(path, modified_ns):
    return load_ohlcv_csv(path)


def run_dashboard_backtest(raw, params, portfolio_value):
    if raw.empty: raise ValueError("No valid OHLCV rows.")
    prepared = prepare_data(raw)
    if prepared.empty: raise ValueError("No valid OHLCV rows.")
    return prepared, run_backtest(prepared, params, portfolio_value, symbol="SPY")


def price_figure(prices, trades):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=prices.index, open=prices["Open"], high=prices["High"], low=prices["Low"], close=prices["Close"], name="Price", increasing_line_color="#16a34a", decreasing_line_color="#dc2626"))
    for col, color in (("BB_upper", "#818cf8"), ("BB_mid", "#f59e0b"), ("BB_lower", "#818cf8")):
        fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode="lines", name=col.replace("_"," "), line=dict(color=color,width=1.5,dash="dot" if col!="BB_mid" else "solid")))
    if not trades.empty:
        fig.add_trace(go.Scatter(x=trades["entry_date"],y=trades["entry_price"],mode="markers",name="Buy",marker=dict(symbol="triangle-up",color="#22c55e",size=12)))
        for reason, color in EXIT_COLORS.items():
            exits = trades.loc[trades["exit_reason"]==reason]
            if not exits.empty:
                fig.add_trace(go.Scatter(x=exits["exit_date"],y=exits["exit_price"],mode="markers",name=f"Sell {reason}",marker=dict(symbol="triangle-down",color=color,size=12)))
    fig.update_layout(xaxis_title="Date",yaxis_title="Price ($)",xaxis_rangeslider_visible=False,hovermode="x unified",height=580)
    return fig


def equity_figure(equity):
    peak = equity.cummax()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index,y=peak,mode="lines",line=dict(color="rgba(239,68,68,0)",width=0),showlegend=False,hoverinfo="skip", name="peak"))
    fig.add_trace(go.Scatter(x=equity.index,y=equity,mode="lines",name="Drawdown",line=dict(color="rgba(239,68,68,0)",width=0),fill="tonexty",fillcolor="rgba(239,68,68,0.16)",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=equity.index,y=equity,mode="lines",name="Portfolio",line=dict(color="#2563eb",width=2),hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="Date",yaxis_title="Portfolio ($)",hovermode="x unified",height=380)
    return fig


def trades_table(trades):
    view = trades.loc[:, list(TRADE_VIEW_COLUMNS)].copy()
    for c in ("entry_date", "exit_date"): view[c] = pd.to_datetime(view[c]).dt.strftime("%Y-%m-%d")
    def color_row(row):
        c = "background-color:rgba(34,197,94,0.15)" if row["net_pnl"]>0 else ("background-color:rgba(239,68,68,0.15)" if row["net_pnl"]<0 else "")
        return [c]*len(row)
    return view.style.apply(color_row,axis=1).format({"entry_price":"${:,.2f}","exit_price":"${:,.2f}","net_pnl":"${:,.2f}"})


def main():
    st.set_page_config(page_title="Trading Bot · Backtest", page_icon="📈", layout="wide")
    st.title("Trading Bot · Backtest")
    st.caption("SPY daily data · Bollinger range strategy with trailing stop")
    files = sorted(DATA_DIR.rglob("*.csv")) if DATA_DIR.exists() else []
    if not files:
        st.warning("No CSV files found in data/."); return
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
        st.info("Set parameters and click Run Backtest."); return
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
    if result.trades.empty: st.warning("No trades found for these settings.")
    st.subheader("Price & signals")
    st.plotly_chart(price_figure(prepared, result.trades), use_container_width=True)
    st.subheader("Equity curve")
    st.plotly_chart(equity_figure(result.equity), use_container_width=True)
    st.subheader("Trades")
    st.dataframe(trades_table(result.trades), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
