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


def test_resolve_market_radar_path_prefers_existing_local_csv(tmp_path):
    existing = tmp_path / "market_radar.csv"
    existing.write_text("Symbol,Sector\nAAPL,Technology\n")

    resolved = dashboard.resolve_market_radar_path(tmp_path / "missing.csv")

    assert resolved == existing


def test_market_radar_with_empty_filters_shows_everything():
    radar = pd.DataFrame([
        {"Symbol": "A", "Sector": "Tech", "QuoteType": "Equity", "Validatore_Scalper": True, "Validatore_Trend": True},
        {"Symbol": "B", "Sector": "Energy", "QuoteType": "ETF", "Validatore_Scalper": False, "Validatore_Trend": False},
    ])

    filtered = dashboard.filter_market_radar(
        radar,
        sectors=[],
        quote_types=[],
        scalper_only=False,
        trend_only=False,
    )

    assert filtered["Symbol"].tolist() == ["A", "B"]


def test_resolve_detail_symbol_from_click_uses_selected_row():
    table = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "NVDA"], "Dettaglio": ["Dettaglio"] * 3})

    symbol = dashboard.resolve_detail_symbol_from_click(table, {"row": 1})

    assert symbol == "MSFT"


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

    assert round(table.loc[0, "Var. giornaliera %"], 2) == -10.0
    assert table.loc[0, "Segnale"] == "Sì"
    assert bool(table.loc[0, "Bot"]) is True
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


def test_build_watchlist_table_exposes_bot_toggle_and_useful_metrics():
    state = {
        "live_data": {
            "SPY": {"price": 120.0, "prev_close": 100.0, "adx": 10.0, "rsi": 20.0, "bb_lower": 125.0},
        },
        "bot_enabled": {"SPY": True},
        "profile": {},
    }
    positions = {"SPY": SimpleNamespace(
        unrealized_pl="60.00",
        qty="3",
        unrealized_plpc="0.20",
        avg_entry_price="100.00",
        cost_basis="300.00",
        market_value="360.00",
    )}

    table = dashboard.build_watchlist_table(["SPY"], positions, state)

    assert bool(table.loc[0, "Bot"]) is True
    assert set(table.columns) >= {
        "Ticker", "Prezzo attuale ($)", "Var. giornaliera %", "Prezzo medio acquisto ($)",
        "Quantità", "Capitale investito ($)", "Valore attuale ($)", "ADX", "RSI", "Segnale",
        "Bot", "Posizione", "P&L ($)", "P&L posizione %",
    }
    assert table.loc[0, "Segnale"] == "Sì"
    assert table.loc[0, "Prezzo attuale ($)"] == 120.0
    assert table.loc[0, "Prezzo medio acquisto ($)"] == 100.0
    assert table.loc[0, "Quantità"] == 3.0
    assert table.loc[0, "Capitale investito ($)"] == 300.0
    assert table.loc[0, "Valore attuale ($)"] == 360.0
    assert table.loc[0, "P&L ($)"] == 60.0
    assert table.loc[0, "P&L posizione %"] == 20.0


def test_build_market_radar_table_adds_signal_and_activation_flags():
    radar = pd.DataFrame([
        {
            "Symbol": "A",
            "Sector": "Tech",
            "QuoteType": "Equity",
            "MarketCap": 100,
            "Close": 100.0,
            "Volume_SMA20": 1000,
            "ADX": 12.0,
            "ATR_pct": 1.2,
            "RSI": 31.0,
            "BB_lower": 90.0,
            "SMA_200": 95.0,
            "Validatore_Scalper": True,
            "Validatore_Trend": True,
        },
        {
            "Symbol": "B",
            "Sector": "Energy",
            "QuoteType": "ETF",
            "MarketCap": 200,
            "Close": 200.0,
            "Volume_SMA20": 2000,
            "ADX": 30.0,
            "ATR_pct": 2.0,
            "RSI": 60.0,
            "BB_lower": 180.0,
            "SMA_200": 190.0,
            "Validatore_Scalper": False,
            "Validatore_Trend": False,
        },
    ])

    table = dashboard.build_market_radar_table(
        radar,
        {"A"},
        {
            "A": {"price": 105.0, "prev_close": 100.0},
            "B": {"price": 198.0, "prev_close": 200.0},
        },
    )

    assert table["Attiva Bot"].tolist() == [True, False]
    assert table["Segnale"].tolist() == ["Scalper + Trend", "—"]
    assert table["Prezzo dinamico ($)"].tolist() == [105.0, 198.0]
    assert [round(value, 2) for value in table["Var. dinamica %"]] == [5.0, -1.0]