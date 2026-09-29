from datetime import datetime, timezone
from types import SimpleNamespace

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.timeframe import TimeFrame

from alpaca_data import fetch_daily_bars


class FakeHistoricalClient:
    def get_stock_bars(self, request):
        self.request = request
        bar = SimpleNamespace(
            timestamp=datetime(2026, 9, 28, tzinfo=timezone.utc),
            open=100.0,
            high=103.0,
            low=99.0,
            close=102.0,
            volume=12345.0,
        )
        return SimpleNamespace(data={"SPY": [bar]})


def test_fetch_daily_bars_uses_raw_alpaca_daily_request(monkeypatch):
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    client = FakeHistoricalClient()

    result = fetch_daily_bars(
        "SPY",
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 29, tzinfo=timezone.utc),
        client=client,
    )["SPY"]

    assert client.request.timeframe.amount == TimeFrame.Day.amount
    assert client.request.timeframe.unit == TimeFrame.Day.unit
    assert client.request.adjustment == Adjustment.RAW
    assert client.request.feed == DataFeed.IEX
    assert result.iloc[0]["Close"] == 102.0
    assert result.iloc[0]["Adj Close"] == 102.0
    assert result.iloc[0]["Volume"] == 12345.0