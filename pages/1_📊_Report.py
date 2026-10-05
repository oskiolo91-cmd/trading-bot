"""Institutional portfolio tear sheet based on Alpaca history and SQLite state."""

from __future__ import annotations

import math
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest

from alpaca_data import fetch_daily_bars
from bot_state import load_bot_state
from live_trader import get_trade_activities


@st.cache_resource
def _clients(api_key: str, secret_key: str):
    return (
        TradingClient(api_key=api_key, secret_key=secret_key, paper=True),
        StockHistoricalDataClient(api_key=api_key, secret_key=secret_key),
    )


def _credentials() -> tuple[str | None, str | None]:
    try:
        secret_key = st.secrets.get("ALPACA_API_KEY")
        secret_value = st.secrets.get("ALPACA_SECRET_KEY")
    except Exception:
        secret_key = secret_value = None
    return (
        os.environ.get("ALPACA_API_KEY") or secret_key,
        os.environ.get("ALPACA_SECRET_KEY") or secret_value,
    )


def build_portfolio_history_frame(history) -> pd.DataFrame:
    """Normalize Alpaca's portfolio history response to a timestamp-indexed frame."""
    timestamps = getattr(history, "timestamp", [])
    equities = getattr(history, "equity", [])
    if len(timestamps) == 0 or len(timestamps) != len(equities):
        return pd.DataFrame(columns=["Equity"])

    if isinstance(timestamps[0], (datetime, pd.Timestamp)):
        dates = pd.to_datetime(timestamps, utc=True, errors="coerce")
    else:
        dates = pd.to_datetime(timestamps, unit="s", utc=True, errors="coerce")
    frame = pd.DataFrame({
        "Timestamp": dates,
        "Equity": pd.to_numeric(equities, errors="coerce"),
    }).dropna()
    return frame.sort_values("Timestamp").set_index("Timestamp")


def build_equity_comparison(history: pd.DataFrame, spy_bars: pd.DataFrame) -> pd.DataFrame:
    """Align portfolio and SPY daily closes and normalize both series to 100."""
    if history.empty or spy_bars.empty or "Close" not in spy_bars:
        return pd.DataFrame(columns=["Portfolio", "SPY"])
    portfolio = pd.to_numeric(history["Equity"], errors="coerce").dropna()
    benchmark = pd.to_numeric(spy_bars["Close"], errors="coerce").dropna()
    portfolio.index = pd.to_datetime(portfolio.index, utc=True).normalize()
    benchmark.index = pd.to_datetime(benchmark.index, utc=True).normalize()
    aligned = pd.concat(
        [portfolio.groupby(level=0).last().rename("Portfolio"), benchmark.groupby(level=0).last().rename("SPY")],
        axis=1,
        join="inner",
    ).dropna()
    aligned = aligned[(aligned["Portfolio"] > 0) & (aligned["SPY"] > 0)]
    if aligned.empty:
        return pd.DataFrame(columns=["Portfolio", "SPY"])
    return aligned.div(aligned.iloc[0]).mul(100)


def realized_trade_pnls(fills) -> list[float]:
    """Calculate realized long-only outcomes by matching sell fills FIFO to buys."""
    lots = defaultdict(deque)
    outcomes = []
    for fill in sorted(fills, key=lambda item: item.transaction_time):
        symbol = fill.symbol
        if fill.side == "buy":
            lots[symbol].append([float(fill.qty), float(fill.price)])
            continue
        if fill.side != "sell":
            continue
        remaining = float(fill.qty)
        realized = 0.0
        matched = 0.0
        while remaining > 1e-9 and lots[symbol]:
            lot = lots[symbol][0]
            quantity = min(remaining, lot[0])
            realized += quantity * (float(fill.price) - lot[1])
            matched += quantity
            remaining -= quantity
            lot[0] -= quantity
            if lot[0] <= 1e-9:
                lots[symbol].popleft()
        if matched > 0:
            outcomes.append(realized)
    return outcomes


def performance_metrics(outcomes: list[float]) -> tuple[float | None, float | None]:
    if not outcomes:
        return None, None
    winners = sum(value for value in outcomes if value > 0)
    losers = abs(sum(value for value in outcomes if value < 0))
    win_rate = sum(value > 0 for value in outcomes) / len(outcomes) * 100
    profit_factor = winners / losers if losers else (math.inf if winners else None)
    return win_rate, profit_factor


def max_drawdown_pct(equity: pd.Series) -> float | None:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    values = values[values > 0]
    if values.empty:
        return None
    return float(((values / values.cummax()) - 1).min() * -100)


def portfolio_figure(comparison: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for name, color in (("Portfolio", "#147d64"), ("SPY", "#315a75")):
        if name in comparison:
            figure.add_trace(go.Scatter(
                x=comparison.index,
                y=comparison[name],
                name=name,
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}<extra>%{fullData.name}</extra>",
            ))
    figure.update_layout(
        title="Crescita normalizzata · base 100",
        height=380,
        hovermode="x unified",
        margin={"l": 12, "r": 18, "t": 48, "b": 12},
        legend={"orientation": "h", "y": 1.12},
        yaxis_title="Valore indice",
    )
    return figure


def underwater_figure(history: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if not history.empty:
        equity = pd.to_numeric(history["Equity"], errors="coerce").dropna()
        drawdown = (equity / equity.cummax() - 1) * 100
        figure.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line={"color": "#c73c46", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(199, 60, 70, 0.20)",
            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>",
        ))
    figure.update_layout(
        title="Underwater · drawdown dal massimo precedente",
        height=260,
        margin={"l": 12, "r": 18, "t": 48, "b": 12},
        showlegend=False,
        yaxis_title="Drawdown (%)",
    )
    return figure


def _render_database_state() -> None:
    symbols = load_bot_state()["symbols"]
    st.subheader("Ticker monitorati nel database")
    if not symbols:
        st.caption("Nessun ticker persistito nel database del bot.")
        return
    frame = pd.DataFrame.from_dict(symbols, orient="index").rename_axis("Ticker").reset_index()
    frame = frame.rename(columns={
        "profile": "Profilo rischio",
        "custom_trailing_pct": "Trailing personalizzato",
        "high_water_mark": "High-water mark",
        "last_buy_date": "Ultimo acquisto",
    })
    frame["Trailing personalizzato"] = pd.to_numeric(
        frame["Trailing personalizzato"], errors="coerce"
    ).mul(100)
    st.dataframe(
        frame[["Ticker", "Profilo rischio", "Trailing personalizzato", "High-water mark", "Ultimo acquisto"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Trailing personalizzato": st.column_config.NumberColumn(format="%.1f%%"),
            "High-water mark": st.column_config.NumberColumn(format="$%.2f"),
        },
    )


def render_report() -> None:
    st.set_page_config(page_title="Report istituzionale", page_icon="📊", layout="wide")
    st.title("Tear Sheet · Portafoglio")
    st.caption("Dati storici Alpaca e stato operativo SQLite")
    _render_database_state()
    st.divider()

    api_key, secret_key = _credentials()
    if not api_key or not secret_key:
        st.warning("Configura ALPACA_API_KEY e ALPACA_SECRET_KEY per caricare lo storico del broker.")
        metrics = st.columns(3)
        metrics[0].metric("Win Rate", "—")
        metrics[1].metric("Profit Factor", "—")
        metrics[2].metric("Max Drawdown", "—")
        return

    trading_client, market_client = _clients(api_key, secret_key)
    history_frame = pd.DataFrame(columns=["Equity"])
    outcomes = []
    errors = []
    try:
        request = GetPortfolioHistoryRequest(period="all", timeframe="1D")
        history = trading_client.get_portfolio_history(history_filter=request)
        history_frame = build_portfolio_history_frame(history)
    except Exception as exc:
        errors.append(f"Storico equity non disponibile: {exc}")
    try:
        outcomes = realized_trade_pnls(get_trade_activities(trading_client))
    except Exception as exc:
        errors.append(f"Storico eseguiti non disponibile: {exc}")

    win_rate, profit_factor = performance_metrics(outcomes)
    drawdown = max_drawdown_pct(history_frame.get("Equity", pd.Series(dtype=float)))
    metrics = st.columns(3)
    metrics[0].metric("Win Rate", f"{win_rate:.1f}%" if win_rate is not None else "—")
    metrics[1].metric(
        "Profit Factor",
        "∞" if profit_factor is not None and math.isinf(profit_factor)
        else f"{profit_factor:.2f}" if profit_factor is not None else "—",
    )
    metrics[2].metric("Max Drawdown", f"{drawdown:.2f}%" if drawdown is not None else "—")

    comparison = pd.DataFrame(columns=["Portfolio", "SPY"])
    if not history_frame.empty:
        start = history_frame.index.min().to_pydatetime()
        end = history_frame.index.max().to_pydatetime() + timedelta(days=1)
        try:
            spy_bars = fetch_daily_bars("SPY", start, end, client=market_client)["SPY"]
            comparison = build_equity_comparison(history_frame, spy_bars)
        except Exception as exc:
            errors.append(f"Benchmark SPY non disponibile: {exc}")
    if not comparison.empty:
        st.plotly_chart(portfolio_figure(comparison), use_container_width=True)
    else:
        st.info("Storico sufficiente non disponibile per confrontare portafoglio e SPY.")
    st.plotly_chart(underwater_figure(history_frame), use_container_width=True)

    if outcomes:
        st.caption(f"Win rate e profit factor calcolati su {len(outcomes)} chiusure FIFO; commissioni escluse.")
    for error in errors:
        st.warning(error)


if __name__ == "__main__":
    render_report()
