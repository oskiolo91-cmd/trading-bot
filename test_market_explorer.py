import pandas as pd
from types import SimpleNamespace

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


def test_market_radar_pagination_returns_page_and_total_pages():
    radar = pd.DataFrame({"Symbol": ["A", "B", "C", "D", "E"]})

    page, page_count = dashboard.paginate_market_radar(radar, page=1, page_size=2)

    assert page["Symbol"].tolist() == ["C", "D"]
    assert page_count == 3


def test_personal_ticker_table_includes_live_metrics_signal_and_position_pnl():
    state = {
        "live_data": {
            "SPY": {"price": 90.0, "prev_close": 100.0, "adx": 10.0, "rsi": 20.0, "bb_lower": 95.0},
        },
        "bot_enabled": {"SPY": True},
        "profile": {},
    }
    positions = {"SPY": SimpleNamespace(unrealized_pl=2.5)}

    table = dashboard.build_watchlist_table(["SPY"], positions, state)

    assert round(table.loc[0, "Var. %"], 2) == -10.0
    assert table.loc[0, "Segnale"] == "Sì"
    assert table.loc[0, "Bot"] == "Attivo"
    assert table.loc[0, "P&L ($)"] == 2.5


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