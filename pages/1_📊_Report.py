"""Institutional portfolio tear sheet based on Alpaca history and SQLite state."""

from __future__ import annotations

import math
import os
from collections import defaultdict, deque
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca_data import fetch_daily_bars
from bot_state import load_bot_state
from live_trader import get_trade_activities

EASTERN = ZoneInfo("America/New_York")
ESTIMATED_COMMISSION_PCT = 0.001
STARTING_BOT_EQUITY = 10_000.0


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


def _eastern_date(value) -> date:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(EASTERN)
    else:
        timestamp = timestamp.tz_convert(EASTERN)
    return timestamp.date()


def build_synthetic_equity_history(
    fills,
    daily_bars: dict[str, pd.DataFrame],
    starting_equity: float = STARTING_BOT_EQUITY,
    commission_pct: float = ESTIMATED_COMMISSION_PCT,
    current_unrealized_pnl: float | None = None,
    as_of: datetime | None = None,
) -> pd.DataFrame:
    """Build a cash-flow-independent equity curve from FIFO fills and daily marks."""
    if not math.isfinite(starting_equity) or starting_equity <= 0:
        raise ValueError("starting_equity must be positive and finite")
    if not math.isfinite(commission_pct) or commission_pct < 0:
        raise ValueError("commission_pct must be finite and non-negative")

    prices_by_date: dict[date, dict[str, float]] = defaultdict(dict)
    for symbol, bars in daily_bars.items():
        if bars.empty or "Close" not in bars:
            continue
        closes = pd.to_numeric(bars["Close"], errors="coerce").dropna()
        for timestamp, close in closes.items():
            prices_by_date[_eastern_date(timestamp)][symbol] = float(close)

    ordered_fills = sorted(fills, key=lambda item: item.transaction_time)
    fill_dates = [_eastern_date(fill.transaction_time) for fill in ordered_fills]
    dates = sorted(set(prices_by_date) | set(fill_dates))
    if dates:
        dates = sorted(set(dates) | {dates[0] - timedelta(days=1)})
    if current_unrealized_pnl is not None:
        current_date = _eastern_date(as_of or datetime.now(EASTERN))
        dates = sorted(set(dates) | {current_date})
    if not dates:
        return pd.DataFrame(columns=["Equity", "Realized P&L", "Unrealized P&L"])

    lots: dict[str, deque] = defaultdict(deque)
    last_prices: dict[str, float] = {}
    fill_index = 0
    realized_total = 0.0
    rows = []
    for current_date in dates:
        last_prices.update(prices_by_date.get(current_date, {}))
        realized_today = 0.0
        while fill_index < len(ordered_fills) and fill_dates[fill_index] <= current_date:
            fill = ordered_fills[fill_index]
            fill_index += 1
            quantity = float(fill.qty)
            price = float(fill.price)
            if fill.side == "buy":
                lots[fill.symbol].append([quantity, price, quantity * price * commission_pct])
                continue
            if fill.side != "sell":
                continue

            matched_quantity = 0.0
            fill_pnl = 0.0
            while quantity > 1e-9 and lots[fill.symbol]:
                lot = lots[fill.symbol][0]
                matched = min(quantity, lot[0])
                entry_fee = lot[2] if math.isclose(matched, lot[0]) else lot[2] * matched / lot[0]
                fill_pnl += matched * (price - lot[1]) - entry_fee
                lot[0] -= matched
                lot[2] -= entry_fee
                quantity -= matched
                matched_quantity += matched
                if lot[0] <= 1e-9:
                    lots[fill.symbol].popleft()
            if matched_quantity:
                fill_pnl -= matched_quantity * price * commission_pct
                realized_today += fill_pnl
        realized_total += realized_today
        unrealized = 0.0
        open_entry_fees = 0.0
        for symbol, symbol_lots in lots.items():
            open_entry_fees += sum(lot[2] for lot in symbol_lots)
            mark = last_prices.get(symbol)
            if mark is None:
                continue
            unrealized += sum((mark - lot[1]) * lot[0] - lot[2] for lot in symbol_lots)

        if current_unrealized_pnl is not None and current_date == dates[-1]:
            unrealized = float(current_unrealized_pnl) - open_entry_fees
        rows.append({
            "Timestamp": pd.Timestamp(datetime.combine(current_date, time.min), tz="UTC"),
            "Equity": starting_equity + realized_total + unrealized,
            "Realized P&L": realized_total,
            "Unrealized P&L": unrealized,
        })

    return pd.DataFrame(rows).set_index("Timestamp").sort_index()


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


def realized_trade_pnls(
    fills,
    commission_pct: float = ESTIMATED_COMMISSION_PCT,
) -> list[float]:
    """Calculate realized long-only outcomes FIFO, net of estimated entry/exit fees."""
    lots = defaultdict(deque)
    outcomes = []
    for fill in sorted(fills, key=lambda item: item.transaction_time):
        symbol = fill.symbol
        if fill.side == "buy":
            quantity = float(fill.qty)
            price = float(fill.price)
            lots[symbol].append([quantity, price, quantity * price * commission_pct])
            continue
        if fill.side != "sell":
            continue
        remaining = float(fill.qty)
        realized = 0.0
        matched = 0.0
        while remaining > 1e-9 and lots[symbol]:
            lot = lots[symbol][0]
            quantity = min(remaining, lot[0])
            entry_fee = lot[2] if math.isclose(quantity, lot[0]) else lot[2] * quantity / lot[0]
            realized += quantity * (float(fill.price) - lot[1]) - entry_fee
            lot[2] -= entry_fee
            matched += quantity
            remaining -= quantity
            lot[0] -= quantity
            if lot[0] <= 1e-9:
                lots[symbol].popleft()
        if matched > 0:
            realized -= matched * float(fill.price) * commission_pct
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
    if values.empty:
        return None
    return float(((values / values.cummax()) - 1).min() * -100)


def sharpe_ratio(equity: pd.Series, periods_per_year: int = 252) -> float | None:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    returns = values.pct_change().dropna()
    if len(returns) < 2:
        return None
    volatility = float(returns.std(ddof=1))
    if not math.isfinite(volatility) or volatility == 0:
        return None
    return float(returns.mean() / volatility * math.sqrt(periods_per_year))


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
        title="True Bot Equity vs SPY · base 100",
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
        metrics = st.columns(4)
        metrics[0].metric("Win Rate", "—")
        metrics[1].metric("Profit Factor", "—")
        metrics[2].metric("Max Drawdown", "—")
        metrics[3].metric("Sharpe Ratio", "—")
        return

    trading_client, market_client = _clients(api_key, secret_key)
    errors = []
    fills = []
    positions = None
    try:
        fills = get_trade_activities(trading_client)
    except Exception as exc:
        errors.append(f"Fill Alpaca non disponibili: {exc}")
    try:
        positions = trading_client.get_all_positions()
    except Exception as exc:
        errors.append(f"Posizioni aperte non disponibili: {exc}")

    current_unrealized = (
        sum(float(getattr(position, "unrealized_pl", 0) or 0) for position in positions)
        if positions is not None
        else None
    )
    traded_symbols = {fill.symbol for fill in fills}
    if positions is not None:
        traded_symbols.update(str(position.symbol) for position in positions)
    data_symbols = sorted(traded_symbols | {"SPY"})
    now = datetime.now(timezone.utc)
    if fills:
        start = min(fill.transaction_time for fill in fills)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        start -= timedelta(days=5)
    else:
        start = now - timedelta(days=365)
    daily_bars = {}
    try:
        daily_bars = fetch_daily_bars(data_symbols, start, now, client=market_client)
    except Exception as exc:
        errors.append(f"Barre storiche non disponibili: {exc}")

    history_frame = build_synthetic_equity_history(
        fills,
        daily_bars,
        current_unrealized_pnl=current_unrealized,
        as_of=datetime.now(EASTERN),
    )
    outcomes = realized_trade_pnls(fills)
    win_rate, profit_factor = performance_metrics(outcomes)
    drawdown = max_drawdown_pct(history_frame.get("Equity", pd.Series(dtype=float)))
    sharpe = sharpe_ratio(history_frame.get("Equity", pd.Series(dtype=float)))
    metrics = st.columns(4)
    metrics[0].metric("Win Rate", f"{win_rate:.1f}%" if win_rate is not None else "—")
    metrics[1].metric(
        "Profit Factor",
        "∞" if profit_factor is not None and math.isinf(profit_factor)
        else f"{profit_factor:.2f}" if profit_factor is not None else "—",
    )
    metrics[2].metric("Max Drawdown", f"{drawdown:.2f}%" if drawdown is not None else "—")
    metrics[3].metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "—")

    comparison = pd.DataFrame(columns=["Portfolio", "SPY"])
    spy_bars = daily_bars.get("SPY", pd.DataFrame())
    if not history_frame.empty and not spy_bars.empty:
        comparison = build_equity_comparison(history_frame, spy_bars)
    if not comparison.empty:
        st.plotly_chart(portfolio_figure(comparison), use_container_width=True)
    else:
        st.info("Storico sufficiente non disponibile per confrontare portafoglio e SPY.")
    st.plotly_chart(underwater_figure(history_frame), use_container_width=True)

    if outcomes:
        st.caption(
            f"{len(outcomes)} chiusure FIFO · commissioni stimate {ESTIMATED_COMMISSION_PCT:.2%} per lato · "
            f"equity iniziale ${STARTING_BOT_EQUITY:,.0f}."
        )
    elif not fills:
        st.caption("Nessun fill disponibile: la curva sintetica parte dal capitale base.")
    for error in errors:
        st.warning(error)


if __name__ == "__main__":
    render_report()
