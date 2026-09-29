import numpy as np
import pandas as pd

import dashboard


def _daily_frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=220, freq="B", tz="UTC")
    close = pd.Series(100 + np.arange(len(dates)) * 0.1, index=dates)
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": 100_000,
        },
        index=dates,
    )


def test_ticker_batch_requests_ohlcv_once_for_multiple_symbols(monkeypatch):
    calls = []
    frame = _daily_frame()

    def fake_download(symbols, **kwargs):
        calls.append(symbols)
        return {symbol: frame.copy() for symbol in symbols}

    monkeypatch.setattr(dashboard, "fetch_daily_bars", fake_download)
    results = dashboard._fetch_ticker_batch(["SPY", "QQQ", "BRK-B"])

    assert calls == [["SPY", "QQQ", "BRK.B"]]
    assert set(results) == {"SPY", "QQQ", "BRK-B"}
    assert len(results["SPY"]["chart_data"]) == 220
    assert pd.notna(results["SPY"]["chart_data"]["SMA_200"].iloc[-1])


def test_active_modules_combine_configured_symbol_bots_and_positions():
    state = {"bot_enabled": {"QQQ": True, "GLD": False}}
    symbols = dashboard.get_active_symbols({"AAPL": object()}, state)

    assert symbols == ["SPY", "QQQ", "AAPL"]