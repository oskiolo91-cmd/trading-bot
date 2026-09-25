"""Interactive dashboard for the existing candle-by-candle trading backtest."""

from __future__ import annotations

import math
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
from live_trader import get_account_value, get_alpaca_client, get_latest_bars, get_recent_orders, place_limit_buy, run_signal_check
from risk import calc_position_size

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
    """Cache a CSV until its modification timestamp changes."""
    return load_ohlcv_csv(path)


def run_dashboard_backtest(raw: pd.DataFrame, params: StrategyParams, portfolio_value: float) -> tuple[pd.DataFrame, BacktestResult]:
    if raw.empty:
        raise ValueError("The selected CSV contains no valid OHLCV rows.")
    prepared = prepare_data(raw)
    if prepared.empty:
        raise ValueError("The selected CSV contains no valid OHLCV rows.")
    return prepared, run_backtest(prepared, params, portfolio_value, symbol="SPY")


def price_figure(prices: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=prices.index, open=prices["Open"], high=prices["High"],
        low=prices["Low"], close=prices["Close"], name="Price",
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
    ))
    for column, color in (("BB_upper", "#818cf8"), ("BB_mid", "#f59e0b"), ("BB_lower", "#818cf8")):
        fig.add_trace(go.Scatter(
            x=prices.index, y=prices[column], mode="lines", name=column.replace("_", " "),
            line=dict(color=color, width=1.5, dash="dot" if column != "BB_mid" else "solid"),
        ))
    if not trades.empty:
        fig.add_trace(go.Scatter(
            x=trades["entry_date"], y=trades["entry_price"], mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", color="#22c55e", size=12, line=dict(color="#14532d", width=1)),
            hovertemplate="Buy · %{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
        ))
        for reason, color in EXIT_COLORS.items():
            exits = trades.loc[trades["exit_reason"] == reason]
            if exits.empty:
                continue
            fig.add_trace(go.Scatter(
                x=exits["exit_date"], y=exits["exit_price"], mode="markers",
                name=f"Sell · {reason.replace('_', ' ')}",
                marker=dict(symbol="triangle-down", color=color, size=12, line=dict(color="#1e293b", width=1)),
                hovertemplate=f"Sell ({reason.replace('_', ' ')}) · %{{x|%Y-%m-%d}}<br>$%{{y:,.2f}}<extra></extra>",
            ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Adjusted price ($)", xaxis_rangeslider_visible=False,
        hovermode="x unified", height=580, margin=dict(l=8, r=8, t=20, b=8),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def equity_figure(equity: pd.Series) -> go.Figure:
    peak = equity.cummax()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=peak, mode="lines", name="Previous peak",
        line=dict(color="rgba(239,68,68,0)", width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity, mode="lines", name="Drawdown",
        line=dict(color="rgba(239,68,68,0)", width=0), fill="tonexty",
        fillcolor="rgba(239,68,68,0.16)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity, mode="lines", name="Portfolio value",
        line=dict(color="#2563eb", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Portfolio value ($)", hovermode="x unified",
        height=380, margin=dict(l=8, r=8, t=20, b=8),
    )
    return fig


def trades_table(trades: pd.DataFrame) -> pd.io.formats.style.Styler:
    view = trades.loc[:, TRADE_VIEW_COLUMNS].copy()
    for column in ("entry_date", "exit_date"):
        view[column] = pd.to_datetime(view[column]).dt.strftime("%Y-%m-%d")

    def color_row(row: pd.Series) -> list[str]:
        color = "background-color: rgba(34, 197, 94, 0.15)" if row["net_pnl"] > 0 else (
            "background-color: rgba(239, 68, 68, 0.15)" if row["net_pnl"] < 0 else ""
        )
        return [color] * len(row)

    return view.style.apply(color_row, axis=1).format({
        "entry_price": "${:,.2f}", "exit_price": "${:,.2f}", "net_pnl": "${:,.2f}",
    })


def _live_credentials() -> tuple[str | None, str | None, bool]:
    try:
        secret_key = st.secrets.get("ALPACA_API_KEY")
        secret_value = st.secrets.get("ALPACA_SECRET_KEY")
    except (FileNotFoundError, KeyError):
        secret_key = secret_value = None
    return (
        os.environ.get("ALPACA_API_KEY") or secret_key,
        os.environ.get("ALPACA_SECRET_KEY") or secret_value,
        bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")),
    )


def render_live_trading() -> None:
    with st.expander("Live Trading (Paper)"):
        st.caption("Alpaca Paper Trading · gli ordini qui vengono inviati al conto paper")
        key, secret, from_env = _live_credentials()
        if not key or not secret:
            st.error("API keys not configured (ALPACA_API_KEY and ALPACA_SECRET_KEY)")
            return
        try:
            client = get_alpaca_client() if from_env else TradingClient(api_key=key, secret_key=secret, paper=True)
            account_value = get_account_value(client)
            positions = client.get_all_positions()
            orders = get_recent_orders(client)
        except Exception:
            st.error("Could not reach Alpaca Paper Trading. Check credentials and connection.")
            return
        st.metric("Account equity", f"${account_value:,.2f}")
        st.subheader("Open positions")
        if positions:
            st.dataframe(pd.DataFrame([{
                "Symbol": item.symbol, "Shares": item.qty, "Avg entry ($)": item.avg_entry_price,
                "Market value ($)": item.market_value, "Unrealized P&L ($)": item.unrealized_pl,
            } for item in positions]), hide_index=True, width="stretch")
        else:
            st.info("No open positions.")
        st.subheader("Recent orders (last 10)")
        if orders:
            st.dataframe(pd.DataFrame(orders), hide_index=True, width="stretch")
        else:
            st.info("No recent orders.")

        st.subheader("Signal Monitor & Manual Trading")
        symbol = st.text_input("Ticker", value="SPY").strip().upper()
        if st.button("Analyze"):
            st.session_state.pop("signal_monitor", None)
            if not symbol:
                st.warning("Inserisci un ticker prima di analizzare.")
            else:
                try:
                    data = prepare_data(get_latest_bars(symbol))
                    if data.empty:
                        raise ValueError("Nessuna candela giornaliera disponibile.")
                    row = data.iloc[-1]
                    values = {name: float(row[name]) for name in
                              ("Close", "ADX", "RSI", "BB_lower", "BB_mid", "BB_upper", "ATR")}
                    if not all(math.isfinite(value) for value in values.values()):
                        raise ValueError("Indicatori non disponibili per l'ultima candela.")
                    st.session_state["signal_monitor"] = (symbol, data.index[-1], values)
                    st.session_state["manual_limit"] = max(0.0, round(values["BB_lower"], 2))
                    st.session_state["manual_stop"] = max(0.0, round(values["BB_lower"] - 2 * values["ATR"], 2))
                    st.session_state["manual_confirm"] = False
                    st.session_state["manual_shares"] = calc_position_size(
                        account_value, values["BB_lower"], values["BB_lower"] - 2 * values["ATR"],
                    )
                except Exception as exc:
                    st.error(f"Impossibile analizzare {symbol}: {exc}")

        analysis = st.session_state.get("signal_monitor")
        if analysis is None:
            return
        if analysis[0] != symbol:
            st.session_state.pop("signal_monitor", None)
            return
        _, bar_date, values = analysis
        st.caption(f"Ultima candela giornaliera: {bar_date:%Y-%m-%d}")
        params = StrategyParams()
        conditions = {
            f"ADX < {params.adx_max:g} (mercato laterale)": values["ADX"] < params.adx_max,
            f"RSI < {params.rsi_max:g} (ipervenduto)": values["RSI"] < params.rsi_max,
            "Close ≤ BB Lower": values["Close"] <= values["BB_lower"],
        }
        adx_color = "#16a34a" if values["ADX"] < params.adx_max else "#dc2626"
        rsi_color = "#16a34a" if values["RSI"] < params.rsi_max else (
            "#ea580c" if values["RSI"] <= 50 else "#dc2626"
        )
        cols = st.columns(4)
        cols[0].markdown(
            f'<div style="border-left:4px solid {adx_color};padding:8px 12px;color:{adx_color}">'
            f'ADX<br><strong style="font-size:1.6rem">{values["ADX"]:.2f}</strong></div>',
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            f'<div style="border-left:4px solid {rsi_color};padding:8px 12px;color:{rsi_color}">'
            f'RSI<br><strong style="font-size:1.6rem">{values["RSI"]:.2f}</strong></div>',
            unsafe_allow_html=True,
        )
        cols[2].metric("Current Price (Close)", f'${values["Close"]:,.2f}')
        cols[3].metric("ATR", f'${values["ATR"]:,.2f}')
        bands = st.columns(3)
        for column, name in zip(bands, ("BB_lower", "BB_mid", "BB_upper")):
            column.metric(name.replace("_", " "), f'${values[name]:,.2f}')

        passed = sum(conditions.values())
        if passed == 3:
            st.success("🟢 Segnale ATTIVO — tutte le condizioni soddisfatte")
        elif passed:
            details = " · ".join(f'{"✅" if ok else "❌"} {name}' for name, ok in conditions.items())
            st.warning(f"🟡 Segnale PARZIALE — {passed}/3 condizioni soddisfatte\n\n{details}")
        else:
            st.error("🔴 Nessun segnale — condizioni non soddisfatte")

        limit = values["BB_lower"]
        stop = limit - params.atr_mult * values["ATR"]
        shares = calc_position_size(account_value, limit, stop, params.risk_pct, params.max_cap_pct)
        if limit > 0:
            shares = min(shares, math.floor(account_value / (limit * (1 + params.commission_pct))))
        can_buy = (passed == 3 and shares > 0 and round(stop, 2) > 0
                   and round(stop, 2) < round(limit, 2))
        if passed == 3 and not can_buy:
            st.info("Segnale attivo, ma non è possibile dimensionare un ordine valido.")
        if st.button("Esegui ordine automatico", disabled=not can_buy):
            try:
                signal = run_signal_check(symbol, account_value, params)
                if signal is None:
                    st.error("Segnale non validato al novo controllo. Riprova.")
                else:
                    order_id = place_limit_buy(client, symbol, signal.limit_price, signal.shares, signal.stop_loss)
                    st.success(
                        f"Ordine paper inviato per {symbol} (ID {order_id}): "
                        f"limite ${signal.limit_price:.2f} · {signal.shares} azioni · stop loss ${signal.stop_loss:.2f}"
                    )
            except Exception as exc:
                st.error(f"Ordine automatico non inviato: {exc}")

        with st.expander("Override manuale"):
            manual_limit = st.number_input("Prezzo limite ($)", min_value=0.0, step=0.01, key="manual_limit")
            manual_stop = st.number_input("Stop loss ($)", min_value=0.0, step=0.01, key="manual_stop")
            manual_shares = st.number_input("Numero azioni", min_value=0, step=1, key="manual_shares")
            confirmed = st.checkbox("Confermo di voler comprare anche senza segnale automatico", key="manual_confirm")
            if st.button("Compra ora (manuale)", disabled=not confirmed):
                try:
                    order_id = place_limit_buy(client, symbol, manual_limit, manual_shares, manual_stop)
                    st.success(
                        f"Ordine paper manuale inviato per {symbol} (ID {order_id}): "
                        f"limite ${manual_limit:.2f} · {manual_shares} azioni · stop loss ${manual_stop:.2f}"
                    )
                except Exception as exc:
                    st.error(f"Ordine manuale non inviato: {exc}")


def main() -> None:
    st.set_page_config(page_title="Trading Bot · Backtest", page_icon="📈", layout="wide")
    st.title("Trading Bot · Backtest")
    st.caption("SPY daily data · Bollinger range strategy · adjusted prices")

    files = sorted(DATA_DIR.rglob("*.csv")) if DATA_DIR.exists() else []
    if not files:
        st.warning("No CSV files found in data/. Download SPY.csv before running the backtest.")
        return
    options = [str(path.relative_to(BOT_DIR)) for path in files]
    default = options.index("data/SPY.csv") if "data/SPY.csv" in options else 0
    with st.sidebar:
        st.header("Backtest settings")
        selected = st.selectbox("CSV file", options, index=default)
        risk = st.slider("Risk %", 0.1, 3.0, 1.0, 0.1, format="%.1f%%")
        trailing = st.slider("Trailing stop %", 1.0, 10.0, 3.0, 0.5, format="%.1f%%")
        time_stop = st.slider("Time-stop candles", 5, 30, 10)
        commission = st.slider("Commission %", 0.05, 0.5, 0.1, 0.05, format="%.2f%%")
        portfolio = st.number_input("Portfolio value ($)", min_value=1.0, value=100_000.0, step=1_000.0)
        run = st.button("Run Backtest", type="primary", width="stretch")

    if not run:
        st.info("Set your parameters, then click Run Backtest.")
        render_live_trading()
        return

    params = StrategyParams(
        risk_pct=risk / 100, trailing_pct=trailing / 100,
        time_stop=time_stop, commission_pct=commission / 100,
    )
    selected_path = BOT_DIR / selected
    try:
        raw = load_data(str(selected_path), selected_path.stat().st_mtime_ns)
        prepared, result = run_dashboard_backtest(raw, params, portfolio)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not run backtest: {exc}")
        return

    metrics = result.metrics
    cols = st.columns(4)
    cols[0].metric("Net P&L", f"${metrics['net_pnl_total']:,.2f}")
    cols[1].metric("Win Rate", f"{metrics['win_rate']:.1%}")
    cols[2].metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
    cols[3].metric("Total Trades", f"{metrics['total_trades']:,}")

    if result.trades.empty:
        st.warning("No trades found for these settings and this data range.")
    st.subheader("Price & signals")
    st.plotly_chart(price_figure(prepared, result.trades), width="stretch")
    st.subheader("Equity curve")
    st.plotly_chart(equity_figure(result.equity), width="stretch")
    st.subheader("Trades")
    st.dataframe(trades_table(result.trades), width="stretch", hide_index=True)
    render_live_trading()


if __name__ == "__main__":
    main()
