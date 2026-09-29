from datetime import datetime, timezone

import numpy as np
import pandas as pd

import screener


def test_radar_row_calculates_validators_from_last_closed_bar(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=200, freq="B", tz="UTC")
    close = np.full(200, 10.0)
    close[-1] = 11.0
    prepared = pd.DataFrame(
        {
            "Close": close,
            "Volume": np.full(200, 1000.0),
            "ADX": np.full(200, 24.0),
            "ATR": np.full(200, 0.5),
            "RSI": np.full(200, 34.0),
            "BB_lower": np.full(200, 11.0),
        },
        index=dates,
    )
    monkeypatch.setattr(screener, "prepare_data", lambda frame: prepared)

    row = screener._radar_row("SPY", pd.DataFrame({"Close": close}), {
        "Sector": "ETF", "QuoteType": "ETF", "MarketCap": 1_000_000,
    })

    assert row["Volume_SMA20"] == 1000
    assert row["ATR_pct"] == 0.5 / 11 * 100
    assert row["Validatore_Scalper"] is True
    assert row["Validatore_Trend"] is True


def test_radar_exports_schema_and_metadata(tmp_path, monkeypatch):
    dates = pd.date_range("2025-01-01", periods=220, freq="B", tz="UTC")
    close = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
    frame = pd.DataFrame({
        "Open": close - 0.1,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Adj Close": close,
        "Volume": 100_000,
    }, index=dates)

    monkeypatch.setattr(screener, "_fetch_yahoo_metadata", lambda symbol: {
        "Sector": "Technology", "QuoteType": "Equity", "MarketCap": 123_000_000,
    })
    monkeypatch.setattr(screener, "fetch_daily_bars", lambda symbols, **kwargs: {symbol: frame for symbol in symbols})

    class FakeTradingClient:
        def get_clock(self):
            return type("Clock", (), {"is_open": False})()

    output = tmp_path / "market_radar.csv"
    radar = screener.generate_market_radar(["SPY"], output, trading_client=FakeTradingClient())
    stored = pd.read_csv(output)

    assert output.exists()
    assert list(radar.columns) == screener.RADAR_COLUMNS
    assert stored.loc[0, "Sector"] == "Technology"
    assert stored.loc[0, "QuoteType"] == "Equity"
    assert stored.loc[0, "Symbol"] == "SPY"
