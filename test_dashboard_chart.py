from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
from alpaca.trading.enums import OrderSide

import dashboard
from models import StrategyParams


class FakeTradingClient:
    def get_orders(self, filter):
        return [
            SimpleNamespace(
                id="entry-1",
                symbol="SPY",
                side=OrderSide.BUY,
                filled_at=datetime(2026, 9, 28, tzinfo=timezone.utc),
                filled_qty="0.25",
                filled_avg_price="100.50",
            )
        ]


def test_candlestick_chart_has_overlays_entries_and_active_stop(monkeypatch):
    dates = pd.date_range("2026-09-24", periods=5, tz="UTC")
    frame = pd.DataFrame(
        {
            "Open": [100, 101, 102, 101, 103],
            "High": [102, 103, 104, 103, 105],
            "Low": [99, 100, 101, 100, 102],
            "Close": [101, 102, 103, 102, 104],
            "Volume": [1000, 1100, 1200, 1300, 1400],
            "BB_upper": [105, 106, 107, 106, 108],
            "BB_mid": [101, 102, 103, 102, 104],
            "BB_lower": [97, 98, 99, 98, 100],
            "SMA_200": [95, 95.5, 96, 96.5, 97],
        },
        index=dates,
    )
    captured = {}
    monkeypatch.setattr(dashboard, "_open_orders", lambda client, symbol: [
        SimpleNamespace(symbol=symbol, side=OrderSide.SELL, stop_price=99.5)
    ])
    monkeypatch.setattr(dashboard.st, "plotly_chart", lambda figure, **kwargs: captured.update(figure=figure))

    dashboard.render_candlestick_chart(
        FakeTradingClient(), "SPY", {"chart_data": frame}, StrategyParams(bot_mode="TREND_FOLLOWER")
    )

    names = {trace.name for trace in captured["figure"].data}
    assert {"SPY", "Bollinger upper", "Bollinger mid", "Bollinger lower", "SMA 200", "Volume", "Entry eseguite"} <= names
    assert any(shape.line.dash == "dash" for shape in captured["figure"].layout.shapes)