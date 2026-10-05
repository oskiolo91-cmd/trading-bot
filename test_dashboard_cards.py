import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from uuid import uuid4
from types import SimpleNamespace
from alpaca.trading.enums import AssetClass, AssetStatus, OrderSide, OrderType

import dashboard

_REPORT_SPEC = importlib.util.spec_from_file_location(
    "institutional_report", Path(__file__).parent / "pages" / "1_📊_Report.py"
)
report = importlib.util.module_from_spec(_REPORT_SPEC)
_REPORT_SPEC.loader.exec_module(report)


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
    dashboard._cached_ticker_batch.clear()
    results = dashboard._fetch_ticker_batch(["SPY", "QQQ", "BRK-B"])
    cached_results = dashboard._fetch_ticker_batch(["SPY", "QQQ", "BRK-B"])

    assert calls == [["SPY", "QQQ", "BRK.B"]]
    assert set(results) == {"SPY", "QQQ", "BRK-B"}
    assert set(cached_results) == set(results)
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

    assert dashboard.filter_tradable_equity_assets(assets) == ["SPY", "WHOLE"]
    assert dashboard.filter_fractional_assets(assets) == ["SPY"]


def test_fractional_asset_fetch_is_cached_once_per_account():
    class FakeClient:
        calls = 0
        asset_filter = None

        def get_all_assets(self, filter):
            self.calls += 1
            self.asset_filter = filter
            return [
                SimpleNamespace(
                    symbol="QQQ", status=AssetStatus.ACTIVE, tradable=True,
                    fractionable=True, asset_class=AssetClass.US_EQUITY,
                ),
                SimpleNamespace(
                    symbol="WHOLE", status=AssetStatus.ACTIVE, tradable=True,
                    fractionable=False, asset_class=AssetClass.US_EQUITY,
                ),
            ]

    client = FakeClient()
    scope = str(uuid4())

    first = dashboard._cached_active_equity_symbols(client, scope)
    second = dashboard._cached_active_equity_symbols(client, scope)

    assert first == second == (["QQQ", "WHOLE"], ["QQQ"])
    assert client.calls == 1
    assert client.asset_filter.status == AssetStatus.ACTIVE
    assert client.asset_filter.asset_class == AssetClass.US_EQUITY
    assert client.asset_filter.attributes is None


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


def test_risk_profiles_set_trailing_stop_percentages():
    profiles = {
        "🐢 Conservativo": 0.03,
        "⚖️ Bilanciato": 0.06,
        "🚀 Speculativo": 0.12,
    }

    for profile, trailing_pct in profiles.items():
        params = dashboard.get_ticker_params("SPY", {"profile": {"SPY": profile}})
        assert params.trailing_pct == trailing_pct


def test_watchlist_table_shows_profile_hwm_and_broker_stop():
    state = {
        "live_data": {"SPY": {"price": 110.0, "prev_close": 109.0}},
        "bot_enabled": {"SPY": True},
        "profile": {"SPY": "⚖️ Bilanciato"},
        "bot_state": {"SPY": {"position": SimpleNamespace(peak_price=120.0)}},
    }
    positions = {"SPY": SimpleNamespace(avg_entry_price=100.0)}

    broker_orders = {
        "SPY": [
            SimpleNamespace(
                symbol="SPY",
                side=OrderSide.SELL,
                type=OrderType.TRAILING_STOP,
                stop_price=113.25,
            )
        ]
    }
    table = dashboard.build_watchlist_table(["SPY"], positions, state, broker_orders)

    assert table.loc[0, "Profilo di Rischio"] == "⚖️ Bilanciato"
    assert table.loc[0, "Trailing stop %"] == 6.0
    assert table.loc[0, "Prezzo Massimo Raggiunto ($)"] == 120.0
    assert table.loc[0, "Stop broker ($)"] == 113.25
    assert table.loc[0, "Prezzo ingresso bot ($)"] == 100.0
    assert table.loc[0, "Stato ingresso bot"] == "Eseguito"


def test_active_bot_without_position_shows_planned_entry_price():
    state = {
        "live_data": {"NVDA": {"price": 94.0, "prev_close": 93.0, "atr": 2.0, "bb_lower": 95.0}},
        "bot_enabled": {"NVDA": True},
        "profile": {},
    }

    table = dashboard.build_watchlist_table(["NVDA"], {}, state)

    assert table.loc[0, "Prezzo ingresso bot ($)"] == 95.0
    assert table.loc[0, "Stato ingresso bot"] == "Previsto"


def test_bot_entry_value_is_editable_and_override_changes_next_order_plan():
    state = {
        "live_data": {"NVDA": {"price": 94.0, "prev_close": 93.0, "atr": 2.0, "bb_lower": 95.0}},
        "bot_enabled": {"NVDA": True},
        "profile": {},
        "bot_entry_price_overrides": {"NVDA": 90.0},
    }

    table = dashboard.build_watchlist_table(["NVDA"], {}, state)
    limit, trailing_pct, shares = dashboard.order_plan(
        state["live_data"]["NVDA"],
        dashboard.get_ticker_params("NVDA", state),
        entry_price_override=state["bot_entry_price_overrides"]["NVDA"],
    )

    assert table.loc[0, "Prezzo ingresso bot ($)"] == 90.0
    assert table.loc[0, "Stato ingresso bot"] == "Personalizzato"
    assert (limit, trailing_pct, shares) == (90.0, 0.06, 1)


def test_manual_trailing_percentage_overrides_profile_without_local_stop_calculation():
    state = {
        "profile": {"SPY": "🐢 Conservativo"},
        "trailing_stop_pct": {"SPY": 0.047},
        "bot_state": {"SPY": {"position": SimpleNamespace(peak_price=120.0)}},
    }
    positions = {"SPY": SimpleNamespace(avg_entry_price=100.0)}

    params = dashboard.get_ticker_params("SPY", state)
    table = dashboard.build_watchlist_table(
        ["SPY"],
        positions,
        state,
        {"SPY": [SimpleNamespace(
            symbol="SPY",
            side=OrderSide.SELL,
            type=OrderType.TRAILING_STOP,
            stop_price=114.11,
        )]},
    )

    assert params.trailing_pct == 0.047
    assert table.loc[0, "Trailing stop %"] == 4.7
    assert table.loc[0, "Stop broker ($)"] == 114.11


def test_report_builds_synthetic_equity_benchmark_and_drawdown():
    fills = [
        SimpleNamespace(
            symbol="SPY", side="buy", qty=1, price=100,
            transaction_time=pd.Timestamp("2026-09-01T14:00:00Z"),
        ),
        SimpleNamespace(
            symbol="SPY", side="sell", qty=1, price=110,
            transaction_time=pd.Timestamp("2026-09-03T14:00:00Z"),
        ),
    ]
    dates = pd.date_range("2026-09-01 20:00", periods=3, freq="D", tz="UTC")
    market_bars = {"SPY": pd.DataFrame({"Close": [100.0, 105.0, 110.0]}, index=dates)}
    frame = report.build_synthetic_equity_history(fills, market_bars)
    spy_bars = pd.DataFrame({"Close": [495.0, 500.0, 505.0, 510.0]}, index=frame.index)
    comparison = report.build_equity_comparison(frame, spy_bars)
    chart = report.portfolio_figure(comparison)

    assert len(frame) == 4
    np.testing.assert_allclose(frame["Equity"].iloc[2], 10_004.9)
    np.testing.assert_allclose(frame["Equity"].iloc[-1], 10_009.79)
    np.testing.assert_allclose(
        comparison["Portfolio"].iloc[-1], frame["Equity"].iloc[-1] / frame["Equity"].iloc[0] * 100
    )
    np.testing.assert_allclose(comparison["SPY"].iloc[-1], 510 / 495 * 100)
    np.testing.assert_allclose(list(chart.data[0].y), list(comparison["Portfolio"]))
    np.testing.assert_allclose(report.max_drawdown_pct(pd.Series([100.0, 90.0, 95.0])), 10.0)
    assert report.sharpe_ratio(frame["Equity"]) is not None


def test_synthetic_equity_includes_open_position_unrealized_pnl():
    fill = SimpleNamespace(
        symbol="SPY", side="buy", qty=1, price=100,
        transaction_time=pd.Timestamp("2026-09-01T14:00:00Z"),
    )
    bars = {
        "SPY": pd.DataFrame(
            {"Close": [101.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-09-01T20:00:00Z")]),
        )
    }

    frame = report.build_synthetic_equity_history(
        [fill], bars, current_unrealized_pnl=7.0,
        as_of=pd.Timestamp("2026-09-01T20:00:00Z").to_pydatetime(),
    )

    np.testing.assert_allclose(frame["Equity"].iloc[-1], 10_006.9)


def test_report_calculates_realized_outcomes_fifo():
    fills = [
        SimpleNamespace(symbol="SPY", side="buy", qty=2, price=100, transaction_time=pd.Timestamp("2026-09-01")),
        SimpleNamespace(symbol="SPY", side="sell", qty=1, price=110, transaction_time=pd.Timestamp("2026-09-02")),
        SimpleNamespace(symbol="SPY", side="sell", qty=1, price=90, transaction_time=pd.Timestamp("2026-09-03")),
    ]

    outcomes = report.realized_trade_pnls(fills)

    np.testing.assert_allclose(outcomes, [9.79, -10.19])
    win_rate, profit_factor = report.performance_metrics(outcomes)
    assert win_rate == 50.0
    np.testing.assert_allclose(profit_factor, 9.79 / 10.19)


def test_watchlist_can_raise_but_not_lower_bot_high_water_mark():
    position = dashboard.Position(
        entry_date=pd.Timestamp("2026-01-01"),
        entry_price=100.0,
        stop_loss=90.0,
        shares=1.0,
        entry_commission=0.0,
        peak_price=110.0,
    )
    state = {"bot_state": {"SPY": {"position": position, "base": position}}}

    assert dashboard.update_bot_high_water_mark(state, "SPY", 120.0)
    assert state["bot_state"]["SPY"]["position"].peak_price == 120.0
    assert state["bot_state"]["SPY"]["base"].peak_price == 120.0
    assert not dashboard.update_bot_high_water_mark(state, "SPY", 105.0)


def test_bot_does_not_mark_buy_day_when_broker_rejects_order(monkeypatch):
    state = {
        "bot_enabled": {"SPY": True},
        "live_data": {"SPY": {"price": 100.0, "atr": 2.0, "bb_lower": 100.0}},
        "profile": {"SPY": dashboard.DEFAULT_PROFILE},
    }
    client = SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=True))

    def reject_order(*args, **kwargs):
        raise RuntimeError("rejected")

    monkeypatch.setattr(dashboard, "get_symbol_daily_pnl", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(dashboard, "compute_signal", lambda *args, **kwargs: True)
    monkeypatch.setattr(dashboard, "_open_orders", lambda *args, **kwargs: [])
    monkeypatch.setattr(dashboard, "place_limit_buy", reject_order)

    dashboard.run_bot_cycle(client, state, {}, 10_000)

    assert "SPY" not in state.get("bot_last_buy", {})