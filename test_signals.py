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


def test_live_position_exit_is_delegated_to_native_broker_trailing_stop(monkeypatch):
    position = Position(
        entry_date=pd.Timestamp("2026-01-01"),
        entry_price=100.0,
        stop_loss=80.0,
        shares=2.0,
        entry_commission=0.0,
        peak_price=110.0,
    )
    attached = []
    monkeypatch.setattr(live_trader, "has_open_position", lambda client, symbol: True)
    monkeypatch.setattr(live_trader, "get_open_position", lambda client, symbol: SimpleNamespace(qty="2"))
    monkeypatch.setattr(
        live_trader,
        "ensure_native_trailing_stop",
        lambda client, symbol, pct, qty: attached.append((symbol, pct, qty)),
    )
    state = {"position": position, "base": position}

    decision = live_trader._check_exit_once(
        object(),
        "SPY",
        state,
        live_trader.StrategyParams(trailing_pct=0.06),
    )

    assert decision is None
    assert attached == [("SPY", 0.06, 2.0)]
    assert state["position"] == position