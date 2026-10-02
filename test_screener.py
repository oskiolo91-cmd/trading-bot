from datetime import datetime, timezone

import numpy as np
import pandas as pd

import screener


def test_radar_row_calculates_validators_from_last_closed_bar(monkeypatch):
    dates = pd.date_range("2026-01-01", periods=200, freq="B", tz="UTC")
    close = np.full(200, 10.0)
    close[-1] = 11.0
    prepared = pd.DataFrame(
        {
            "Close": close,
            "Volume": np.full(200, 1000.0),
            "ADX": np.full(200, 24.0),
            "ATR": np.full(200, 0.5),
            "RSI": np.full(200, 34.0),
            "BB_lower": np.full(200, 11.0),
        },
        index=dates,
    )
    monkeypatch.setattr(screener, "prepare_data", lambda frame: prepared)

    row = screener._radar_row("SPY", pd.DataFrame({"Close": close}), {
        "Sector": "ETF", "QuoteType": "ETF", "MarketCap": 1_000_000,
    })

    assert row["Volume_SMA20"] == 1000
    assert row["ATR_pct"] == 0.5 / 11 * 100
    assert row["Validatore_Scalper"] is True
    assert row["Validatore_Trend"] is True


def test_radar_exports_schema_and_metadata(tmp_path, monkeypatch):
    dates = pd.date_range("2025-01-01", periods=220, freq="B", tz="UTC")
    close = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
    frame = pd.DataFrame({
        "Open": close - 0.1,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Adj Close": close,
        "Volume": 100_000,
    }, index=dates)

    monkeypatch.setattr(screener, "_fetch_yahoo_metadata", lambda symbol: {
        "Sector": "Technology", "QuoteType": "Equity", "MarketCap": 123_000_000,
    })
    data_client = object()
    received_clients = []

    def fake_fetch_daily_bars(symbols, **kwargs):
        received_clients.append(kwargs.get("client"))
        return {symbol: frame for symbol in symbols}

    monkeypatch.setattr(screener, "fetch_daily_bars", fake_fetch_daily_bars)

    class FakeTradingClient:
        def get_clock(self):
            return type("Clock", (), {"is_open": False})()

    output = tmp_path / "market_radar.csv"
    radar = screener.generate_market_radar(
        ["SPY"], output, trading_client=FakeTradingClient(), data_client=data_client
    )
    stored = pd.read_csv(output)

    assert output.exists()
    assert list(radar.columns) == screener.RADAR_COLUMNS
    assert stored.loc[0, "Sector"] == "Technology"
    assert stored.loc[0, "QuoteType"] == "Equity"
    assert stored.loc[0, "Symbol"] == "SPY"
    assert received_clients == [data_client]


def test_fetch_yahoo_metadata_retries_after_transient_yahoo_error(monkeypatch):
    attempts = {"count": 0}

    class FakeTicker:
        def get_info(self):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("temporary Yahoo issue")
            return {
                "sector": "Technology",
                "quoteType": "EQUITY",
                "marketCap": 123_000_000,
            }

    monkeypatch.setattr(screener.yf, "Ticker", lambda symbol: FakeTicker())
    monkeypatch.setattr(screener.time_module, "sleep", lambda *_args, **_kwargs: None)

    metadata = screener._fetch_yahoo_metadata("AAPL")

    assert metadata["Sector"] == "Technology"
    assert metadata["QuoteType"] == "Equity"
    assert metadata["MarketCap"] == 123_000_000
    assert attempts["count"] == 2


def test_fetch_yahoo_metadata_uses_fast_info_fallback_when_get_info_fails(monkeypatch):
    class FakeTicker:
        fast_info = type("FastInfo", (), {"market_cap": 456_000_000})()

        def get_info(self):
            raise RuntimeError("Yahoo blocked request")

    monkeypatch.setattr(screener.yf, "Ticker", lambda symbol: FakeTicker())
    monkeypatch.setattr(screener.time_module, "sleep", lambda *_args, **_kwargs: None)

    metadata = screener._fetch_yahoo_metadata("AAPL")

    assert metadata["Sector"] == "Unknown"
    assert metadata["QuoteType"] == "Equity"
    assert metadata["MarketCap"] == 456_000_000


def test_generate_market_radar_keeps_cached_sector_metadata_on_refresh(tmp_path, monkeypatch):
    dates = pd.date_range("2025-01-01", periods=220, freq="B", tz="UTC")
    close = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
    frame = pd.DataFrame({
        "Open": close - 0.1,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Adj Close": close,
        "Volume": 100_000,
    }, index=dates)

    output = tmp_path / "market_radar.csv"
    pd.DataFrame([
        {
            "Symbol": "AAPL",
            "Sector": "Technology",
            "QuoteType": "Equity",
            "MarketCap": 123_000_000,
            "Close": 100.0,
            "Volume_SMA20": 1000,
            "ADX": 20.0,
            "ATR_pct": 1.0,
            "RSI": 30.0,
            "BB_lower": 90.0,
            "SMA_200": 95.0,
            "Validatore_Scalper": True,
            "Validatore_Trend": True,
        }
    ]).to_csv(output, index=False)

    monkeypatch.setattr(screener, "_fetch_yahoo_metadata", lambda symbol: {
        "Sector": "Unknown",
        "QuoteType": "Equity",
        "MarketCap": None,
    })
    monkeypatch.setattr(screener, "fetch_daily_bars", lambda symbols, **kwargs: {symbol: frame for symbol in symbols})

    class FakeTradingClient:
        def get_clock(self):
            return type("Clock", (), {"is_open": False})()

    screener.generate_market_radar(["AAPL"], output, trading_client=FakeTradingClient(), data_client=object())
    stored = pd.read_csv(output)

    assert stored.loc[0, "Sector"] == "Technology"
    assert stored.loc[0, "QuoteType"] == "Equity"


def test_generate_market_radar_creates_missing_output_directory(tmp_path, monkeypatch):
    dates = pd.date_range("2025-01-01", periods=220, freq="B", tz="UTC")
    close = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
    frame = pd.DataFrame({
        "Open": close - 0.1,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Adj Close": close,
        "Volume": 100_000,
    }, index=dates)

    output = tmp_path / "subdir" / "market_radar.csv"

    monkeypatch.setattr(screener, "_fetch_yahoo_metadata", lambda symbol: {
        "Sector": "Technology",
        "QuoteType": "Equity",
        "MarketCap": 123_000_000,
    })
    monkeypatch.setattr(screener, "fetch_daily_bars", lambda symbols, **kwargs: {symbol: frame for symbol in symbols})

    class FakeTradingClient:
        def get_clock(self):
            return type("Clock", (), {"is_open": False})()

    screener.generate_market_radar(["AAPL"], output, trading_client=FakeTradingClient(), data_client=object())

    assert output.exists()
    stored = pd.read_csv(output)
    assert stored.loc[0, "Sector"] == "Technology"


def test_initialize_market_radar_bootstraps_and_preserves_existing_data(tmp_path):
    output = tmp_path / "market_radar.csv"
    output.write_text("")

    initial = screener.initialize_market_radar(["SPY", "AAPL"], output)

    assert initial["Symbol"].tolist() == ["SPY", "AAPL"]
    assert set(initial.columns) == set(screener.RADAR_COLUMNS)
    assert initial["Sector"].tolist() == ["Unknown", "Unknown"]
    assert pd.isna(initial.loc[0, "Close"])

    initial.loc[initial["Symbol"] == "AAPL", "Sector"] = "Technology"
    initial.loc[initial["Symbol"] == "AAPL", "Close"] = 200.0
    initial.to_csv(output, index=False)
    updated = screener.initialize_market_radar(["AAPL", "MSFT"], output)

    apple = updated.loc[updated["Symbol"] == "AAPL"].iloc[0]
    microsoft = updated.loc[updated["Symbol"] == "MSFT"].iloc[0]
    assert apple["Sector"] == "Technology"
    assert apple["Close"] == 200.0
    assert microsoft["Sector"] == "Unknown"
    assert pd.isna(microsoft["Close"])


def test_generate_market_radar_writes_progressively_to_csv(monkeypatch, tmp_path):
    dates = pd.date_range("2025-01-01", periods=220, freq="B", tz="UTC")
    close = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
    frame = pd.DataFrame({
        "Open": close - 0.1,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Adj Close": close,
        "Volume": 100_000,
    }, index=dates)

    output = tmp_path / "progressive_market_radar.csv"
    calls = []

    monkeypatch.setattr(screener, "_fetch_yahoo_metadata", lambda symbol: {
        "Sector": "Technology",
        "QuoteType": "Equity",
        "MarketCap": 123_000_000,
    })
    def fake_fetch_daily_bars(symbols, **kwargs):
        bootstrapped = pd.read_csv(output)
        assert set(bootstrapped["Symbol"]) == {"AAPL", "MSFT"}
        assert bootstrapped["Close"].isna().all()
        return {symbol: frame for symbol in symbols}

    monkeypatch.setattr(screener, "fetch_daily_bars", fake_fetch_daily_bars)

    class FakeTradingClient:
        def get_clock(self):
            return type("Clock", (), {"is_open": False})()

    screener.generate_market_radar(
        ["AAPL", "MSFT"],
        output,
        trading_client=FakeTradingClient(),
        data_client=object(),
        batch_size=1,
        progress_callback=lambda partial: calls.append(len(partial)),
    )

    assert len(calls) >= 2
    stored = pd.read_csv(output)
    assert set(stored["Symbol"]) == {"AAPL", "MSFT"}
