"""Shared Alpaca market-data access for backtests and live indicators."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


@lru_cache(maxsize=1)
def get_stock_data_client() -> StockHistoricalDataClient:
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ValueError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY for Alpaca market data")
    return StockHistoricalDataClient(key, secret)


def fetch_daily_bars(
    symbols: str | list[str],
    start: datetime,
    end: datetime | None = None,
    client: StockHistoricalDataClient | None = None,
) -> dict[str, pd.DataFrame]:
    """Return raw OHLCV daily bars indexed by timestamp for one or more symbols."""
    symbol_list = [symbols] if isinstance(symbols, str) else list(symbols)
    if not symbol_list:
        return {}

    feed_name = os.environ.get("ALPACA_DATA_FEED", "iex").lower()
    try:
        feed = DataFeed(feed_name)
    except ValueError as exc:
        raise ValueError("ALPACA_DATA_FEED must be 'iex' or 'sip'") from exc

    request = StockBarsRequest(
        symbol_or_symbols=symbol_list,
        timeframe=TimeFrame.Day,
        start=start,
        end=end or datetime.now(timezone.utc),
        limit=10000,
        adjustment=Adjustment.RAW,
        feed=feed,
    )
    bar_set = (client or get_stock_data_client()).get_stock_bars(request)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbol_list:
        rows = [
            {
                "Date": bar.timestamp,
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Adj Close": bar.close,
                "Volume": bar.volume,
            }
            for bar in bar_set.data.get(symbol, [])
        ]
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.set_index("Date").sort_index()
            frame.index.name = "Date"
        frames[symbol] = frame
    return frames