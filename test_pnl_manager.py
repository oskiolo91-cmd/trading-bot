from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pnl_manager import SymbolState


def test_fractional_partial_fills_use_fifo_and_include_commissions():
    state = SymbolState("SPY", commission_pct=0.001, timezone_=timezone.utc)
    now = datetime(2026, 9, 29, 15, tzinfo=timezone.utc)

    state.record_fill("buy", 0.25, 100, now, "buy-1")
    state.record_fill("buy", 0.75, 100, now, "buy-2")
    state.record_fill("sell", 0.50, 110, now, "sell-1")
    state.record_fill("sell", 0.50, 105, now, "sell-2")

    assert state.position_qty == Decimal("0.00")
    assert state.daily_realized_pnl(now) == Decimal("7.28")


def test_carried_position_realizes_today_with_original_entry_commission():
    state = SymbolState("SPY", commission_pct=0.001, timezone_=timezone.utc)
    yesterday = datetime(2026, 9, 28, 15, tzinfo=timezone.utc)
    today = yesterday + timedelta(days=1)

    state.record_fill("buy", 1, 100, yesterday, "buy-yesterday")
    state.record_fill("sell", 1, 110, today, "sell-today")

    assert state.daily_realized_pnl(today) == Decimal("9.79")
    assert state.position_qty == Decimal("0")


def test_duplicate_execution_id_does_not_change_position_or_pnl():
    state = SymbolState("SPY", timezone_=timezone.utc)
    now = datetime(2026, 9, 29, 15, tzinfo=timezone.utc)

    state.record_fill("buy", 0.5, 100, now, "execution-1")
    state.record_fill("buy", 0.5, 100, now, "execution-1")

    assert state.position_qty == Decimal("0.5")
    assert state.daily_realized_pnl(now) == Decimal("0.00")


def test_sell_fill_cannot_exceed_tracked_long_position():
    state = SymbolState("SPY", timezone_=timezone.utc)
    now = datetime(2026, 9, 29, 15, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="exceeds the tracked"):
        state.record_fill("sell", 0.1, 100, now, "unmatched-sell")


def test_websocket_partial_fill_updates_incrementally_and_deduplicates():
    import live_trader

    now = datetime(2026, 9, 29, 15, tzinfo=timezone.utc)
    state = SymbolState("SPY", timezone_=timezone.utc)
    order = SimpleNamespace(symbol="SPY", side="buy", id="order-1", filled_qty=0.25)
    updates = [
        SimpleNamespace(
            event="partial_fill", execution_id="execution-1", order=order,
            qty=0.25, price=100.0, timestamp=now,
        ),
        SimpleNamespace(
            event="partial_fill", execution_id="execution-2", order=order,
            qty=0.25, price=101.0, timestamp=now,
        ),
    ]

    with patch.dict(live_trader._SYMBOL_STATES, {"SPY": state}):
        live_trader.apply_trade_update(object(), updates[0])
        live_trader.apply_trade_update(object(), updates[1])
        live_trader.apply_trade_update(object(), updates[1])

    assert state.position_qty == Decimal("0.50")
    assert state.average_entry_price == Decimal("100.5")


def test_historical_fill_parser_accepts_minimal_alpaca_activity_payload():
    import live_trader

    class MinimalActivityClient:
        def get(self, path, query):
            assert path == "/account/activities/FILL"
            return [{
                "id": "activity-1",
                "symbol": "SPY",
                "side": "buy",
                "qty": "0.25",
                "price": "100.00",
                "transaction_time": "2026-09-29T15:00:00Z",
            }]

    client = MinimalActivityClient()
    live_trader._ACTIVITY_CACHE.pop(id(client), None)
    fills = live_trader._get_trade_activities(client)

    assert len(fills) == 1
    assert fills[0].symbol == "SPY"
    assert fills[0].qty == 0.25
    assert fills[0].price == 100.0