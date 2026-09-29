import numpy as np
import pandas as pd
from uuid import uuid4
from types import SimpleNamespace
from alpaca.trading.enums import AssetClass, AssetStatus

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


def test_asset_filter_keeps_only_active_tradable_fractional_equities():
    assets = [
        SimpleNamespace(symbol="SPY", status=AssetStatus.ACTIVE, tradable=True,
                        fractionable=True, asset_class=AssetClass.US_EQUITY),
        SimpleNamespace(symbol="OLD", status=AssetStatus.INACTIVE, tradable=True,
                        fractionable=True, asset_class=AssetClass.US_EQUITY),
        SimpleNamespace(symbol="LOCKED", status=AssetStatus.ACTIVE, tradable=False,
                        fractionable=True, asset_class=AssetClass.US_EQUITY),
        SimpleNamespace(symbol="WHOLE", status=AssetStatus.ACTIVE, tradable=True,
                        fractionable=False, asset_class=AssetClass.US_EQUITY),
        SimpleNamespace(symbol="BTC/USD", status=AssetStatus.ACTIVE, tradable=True,
                        fractionable=True, asset_class="crypto"),
    ]

    assert dashboard.filter_fractional_assets(assets) == ["SPY"]


def test_fractional_asset_fetch_is_cached_once_per_account():
    class FakeClient:
        calls = 0

        def get_all_assets(self):
            self.calls += 1
            return [SimpleNamespace(
                symbol="QQQ", status=AssetStatus.ACTIVE, tradable=True,
                fractionable=True, asset_class=AssetClass.US_EQUITY,
            )]

    client = FakeClient()
    scope = str(uuid4())

    first = dashboard._cached_fractional_asset_symbols(client, scope)
    second = dashboard._cached_fractional_asset_symbols(client, scope)

    assert first == second == ["QQQ"]
    assert client.calls == 1


def test_selection_defaults_and_keeps_active_bot_and_positions_selected():
    available = ["SPY", "QQQ", "AAPL", "MSFT"]
    selected, missing = dashboard.prepare_ticker_selection(
        available,
        None,
        {"MSFT": object()},
        {"AAPL": True, "QQQ": False},
    )
    symbols = dashboard.get_active_symbols(selected)

    assert symbols == ["SPY", "QQQ", "AAPL", "MSFT"]
    assert missing == ["AAPL", "MSFT"]