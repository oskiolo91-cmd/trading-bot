import pandas as pd

import dashboard


def test_market_radar_filters_sector_type_and_combined_validators():
    radar = pd.DataFrame([
        {"Symbol": "A", "Sector": "Tech", "QuoteType": "Equity", "Validatore_Scalper": True, "Validatore_Trend": True},
        {"Symbol": "B", "Sector": "Tech", "QuoteType": "Equity", "Validatore_Scalper": True, "Validatore_Trend": False},
        {"Symbol": "C", "Sector": "Energy", "QuoteType": "ETF", "Validatore_Scalper": True, "Validatore_Trend": True},
    ])

    filtered = dashboard.filter_market_radar(
        radar,
        sectors=["Tech"],
        quote_types=["Equity"],
        scalper_only=True,
        trend_only=True,
    )

    assert filtered["Symbol"].tolist() == ["A"]


def test_editor_selection_preserves_active_symbols_hidden_by_filters():
    active = {"A", "HIDDEN"}
    visible = {"A", "B"}
    edited = pd.DataFrame({"Symbol": ["A", "B"], "Attiva Bot": [False, True]})

    updated = dashboard.merge_market_editor_selection(active, visible, edited)

    assert updated == {"B", "HIDDEN"}


def test_active_explorer_symbols_are_added_to_control_tab_selection():
    selected, missing = dashboard.prepare_ticker_selection(
        available_symbols=["SPY", "QQQ", "NVDA"],
        current_selection=["SPY"],
        positions={},
        bot_enabled={},
        active_tickers=["NVDA"],
    )

    assert selected == ["SPY", "NVDA"]
    assert missing == ["NVDA"]


def test_editor_checkbox_enables_bot_and_open_position_cannot_be_disarmed():
    bot_enabled = {"A": False, "B": True, "C": True}
    edited = pd.DataFrame({
        "Symbol": ["A", "B", "C"],
        "Attiva Bot": [True, False, False],
    })

    active = dashboard.sync_market_editor_selection(
        active_tickers={"B", "C"},
        visible_symbols={"A", "B", "C"},
        edited=edited,
        positions={"B": object()},
        bot_enabled=bot_enabled,
    )

    assert active == {"A", "B"}
    assert bot_enabled == {"A": True, "B": True, "C": False}