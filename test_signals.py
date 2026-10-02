import pandas as pd
import pytest
from datetime import datetime
from types import SimpleNamespace

import live_trader
from models import ExitReason, Position
from signals import evaluate_exit, trailing_stop_price


def test_trailing_stop_tracks_high_water_mark_and_exits_at_profile_level():
    position = Position(
        entry_date=pd.Timestamp("2026-01-01"),
        entry_price=100.0,
        stop_loss=80.0,
        shares=1.0,
        entry_commission=0.0,
        peak_price=100.0,
    )

    updated, decision = evaluate_exit(
        position, open_=100.0, high=110.0, low=105.0, close=109.0,
        bb_mid=105.0, trailing_pct=0.06,
    )

    assert decision is None
    assert updated.peak_price == 110.0
    assert trailing_stop_price(updated.peak_price, 0.06) == pytest.approx(103.4)

    updated, decision = evaluate_exit(
        updated, open_=109.0, high=112.0, low=104.0, close=105.0,
        bb_mid=106.0, trailing_pct=0.06,
    )

    assert updated.peak_price == 112.0
    assert decision.reason is ExitReason.TRAILING_STOP
    assert decision.price == pytest.approx(105.28)


def test_bollinger_trailing_activation_does_not_lower_high_water_mark():
    position = Position(
        entry_date=pd.Timestamp("2026-01-01"),
        entry_price=100.0,
        stop_loss=80.0,
        shares=1.0,
        entry_commission=0.0,
        peak_price=120.0,
    )

    updated, decision = evaluate_exit(
        position, open_=110.0, high=115.0, low=108.0, close=110.0,
        bb_mid=105.0, trailing_pct=0.12,
    )

    assert decision is None
    assert updated.trailing_active
    assert updated.peak_price == 120.0


def test_live_exit_submits_market_sell_when_trailing_stop_is_crossed(monkeypatch):
    position = Position(
        entry_date=pd.Timestamp("2026-01-01"),
        entry_price=100.0,
        stop_loss=80.0,
        shares=2.0,
        entry_commission=0.0,
        peak_price=110.0,
    )
    sell_orders = []
    cancelled_orders = []
    protective_order = SimpleNamespace(symbol="SPY", side=live_trader.OrderSide.SELL, id="stop-order")
    open_order_snapshots = iter([[protective_order], []])
    monkeypatch.setattr(live_trader, "has_open_position", lambda client, symbol: True)
    monkeypatch.setattr(live_trader, "get_latest_bars", lambda symbol: pd.DataFrame())
    monkeypatch.setattr(live_trader, "prepare_data", lambda bars: pd.DataFrame(
        [{"Open": 104.0, "High": 110.0, "Low": 102.0, "Close": 103.0, "BB_mid": 100.0, "ATR": 1.0}],
        index=[pd.Timestamp.now(tz=live_trader.EASTERN)],
    ))
    monkeypatch.setattr(
        live_trader,
        "_open_orders",
        lambda client, symbol: next(open_order_snapshots),
    )
    monkeypatch.setattr(live_trader, "get_open_position", lambda client, symbol: SimpleNamespace(qty="2"))
    monkeypatch.setattr(
        live_trader,
        "place_market_sell",
        lambda client, symbol, qty: sell_orders.append((symbol, qty)) or "sell-order-id",
    )
    state = {
        "position": position,
        "base": position,
        "day": datetime.now(live_trader.EASTERN).date(),
    }

    decision = live_trader._check_exit_once(
        SimpleNamespace(cancel_order_by_id=cancelled_orders.append),
        "SPY",
        state,
        live_trader.StrategyParams(trailing_pct=0.06),
    )

    assert decision.reason is ExitReason.TRAILING_STOP
    assert cancelled_orders == ["stop-order"]
    assert sell_orders == [("SPY", 2.0)]
    assert state["pending_exit"] == "sell-order-id"