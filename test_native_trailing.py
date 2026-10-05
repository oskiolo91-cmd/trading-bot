import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import pandas as pd
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

import live_trader
from bot_state import get_symbol_state, load_bot_state, update_symbol_state


class FakeTradingClient:
    def __init__(self, orders=None, quantity=2):
        self.orders = list(orders or [])
        self.quantity = quantity
        self.submitted = []
        self.replaced = []
        self.canceled = []

    def get_open_position(self, symbol):
        return SimpleNamespace(qty=str(self.quantity))

    def get_orders(self, filter):
        return self.orders

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        return SimpleNamespace(id=f"order-{len(self.submitted)}")

    def replace_order_by_id(self, order_id, order_data):
        self.replaced.append((order_id, order_data))

    def cancel_order_by_id(self, order_id):
        self.canceled.append(order_id)


def test_buy_order_is_submitted_without_a_python_managed_exit_leg():
    client = FakeTradingClient()

    order_id = live_trader.place_limit_buy(client, "SPY", 100.125, 2)

    request = client.submitted[0]
    assert order_id == "order-1"
    assert request.symbol == "SPY"
    assert request.qty == 2
    assert request.limit_price == 100.12
    assert request.order_class is None
    assert request.stop_loss is None
    assert request.take_profit is None


def test_buy_with_fractional_quantity_is_rejected_before_submission():
    client = FakeTradingClient()

    with pytest.raises(ValueError, match="whole-share"):
        live_trader.place_limit_buy(client, "SPY", 100, 0.5)

    assert client.submitted == []


def test_open_position_gets_native_trailing_stop_using_percentage_points():
    client = FakeTradingClient()

    order_id = live_trader.ensure_native_trailing_stop(client, "SPY", 0.06)

    request = client.submitted[0]
    assert order_id == "order-1"
    assert request.type == OrderType.TRAILING_STOP
    assert request.side == OrderSide.SELL
    assert request.qty == 2
    assert request.trail_percent == 6.0
    assert request.time_in_force == TimeInForce.GTC


def test_fractional_open_position_is_not_changed_when_native_stop_is_unsupported():
    client = FakeTradingClient(quantity=0.5)
    client.orders = [
        SimpleNamespace(
            id="old-protection",
            symbol="SPY",
            side=OrderSide.SELL,
            type=OrderType.STOP,
            stop_price=95.0,
        )
    ]

    with pytest.raises(ValueError, match="fractional shares"):
        live_trader.ensure_native_trailing_stop(client, "SPY", 0.06)

    assert client.canceled == []
    assert client.submitted == []


def test_legacy_sell_protection_is_not_canceled_if_native_migration_cannot_be_safe():
    legacy_stop = SimpleNamespace(
        id="old-protection",
        symbol="SPY",
        side=OrderSide.SELL,
        type=OrderType.STOP,
        stop_price=95.0,
    )
    client = FakeTradingClient(orders=[legacy_stop])

    with pytest.raises(RuntimeError, match="left untouched"):
        live_trader.ensure_native_trailing_stop(client, "SPY", 0.06)

    assert client.canceled == []
    assert client.submitted == []


def test_missing_high_water_mark_recovers_from_historical_daily_highs(monkeypatch):
    index = pd.date_range("2026-10-01", periods=3, tz="UTC")
    bars = pd.DataFrame({"High": [102.0, 108.0, 105.0]}, index=index)
    monkeypatch.setattr(live_trader, "download_daily_bars", lambda *args, **kwargs: bars)

    high_water_mark = live_trader.recover_high_water_mark(
        "SPY",
        pd.Timestamp("2026-10-01", tz="UTC"),
        100.0,
    )

    assert high_water_mark == 108.0


def test_risk_profile_change_replaces_open_native_trail():
    trailing_order = SimpleNamespace(
        id="trail-1",
        symbol="SPY",
        side=OrderSide.SELL,
        type=OrderType.TRAILING_STOP,
        trail_percent=6.0,
    )
    client = FakeTradingClient(orders=[trailing_order])

    count = live_trader.replace_trailing_stop_percent(client, "SPY", 0.047)

    assert count == 1
    assert client.replaced[0][0] == "trail-1"
    assert client.replaced[0][1].trail == pytest.approx(4.7)


def test_sqlite_state_round_trips_risk_fields_and_last_buy(tmp_path):
    path = tmp_path / "bot_state.db"

    update_symbol_state(
        "SPY",
        {
            "profile": "⚖️ Bilanciato",
            "custom_trailing_pct": 0.047,
            "high_water_mark": 123.45,
            "last_buy_date": datetime(2026, 10, 2, tzinfo=timezone.utc).date().isoformat(),
            "custom_settings": {"custom_adx": 31.0, "custom_budget": 150.0},
        },
        path,
    )

    assert get_symbol_state("SPY", path)["high_water_mark"] == 123.45
    record = load_bot_state(path)["symbols"]["SPY"]
    assert record["profile"] == "⚖️ Bilanciato"
    assert record["custom_trailing_pct"] == pytest.approx(0.047)
    assert record["trailing_pct"] == pytest.approx(0.047)
    assert record["last_buy_date"] == "2026-10-02"
    assert record["custom_settings"] == {"custom_adx": 31.0, "custom_budget": 150.0}
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(tickers_state)")}
    assert columns == {
        "symbol", "profilo_rischio", "custom_trailing_pct", "high_water_mark",
        "last_buy_date", "custom_settings_json",
    }


def test_sqlite_state_updates_fields_without_overwriting_others(tmp_path):
    path = tmp_path / "bot_state.db"
    update_symbol_state("SPY", {"profile": "⚖️ Bilanciato", "high_water_mark": 123.45}, path)
    update_symbol_state("SPY", {"last_buy_date": "2026-10-02"}, path)
    assert load_bot_state(path)["symbols"]["SPY"]["profile"] == "⚖️ Bilanciato"
    state = get_symbol_state("SPY", path)
    assert state["profile"] == "⚖️ Bilanciato"
    assert state["high_water_mark"] == 123.45
    assert state["last_buy_date"] == "2026-10-02"


def test_sqlite_migrates_existing_table_and_preserves_rows(tmp_path):
    path = tmp_path / "bot_state.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE tickers_state (
                symbol TEXT PRIMARY KEY,
                profilo_rischio TEXT,
                custom_trailing_pct REAL,
                high_water_mark REAL,
                last_buy_date TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO tickers_state VALUES (?, ?, ?, ?, ?)",
            ("SPY", "⚖️ Bilanciato", 0.047, 123.45, "2026-10-02"),
        )

    update_symbol_state("SPY", {"custom_settings": {"custom_budget": 150.0}}, path)

    state = get_symbol_state("SPY", path)
    assert state["profile"] == "⚖️ Bilanciato"
    assert state["high_water_mark"] == 123.45
    assert state["custom_settings"] == {"custom_budget": 150.0}
