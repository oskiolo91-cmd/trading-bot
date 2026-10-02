from datetime import datetime, timezone

import numpy as np
import pandas as pd

import screener


def _seed_nasdaq_db(radar_path, rows):
    database_path = radar_path.parent / "nasdaq_screener.csv"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(database_path, index=False)
    return database_path


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


def test_update_nasdaq_db_downloads_and_normalizes_reference_data(tmp_path, monkeypatch):
    request = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"rows": [{
                "symbol": "brk.b", "name": "Berkshire Hathaway", "sector": "Financial Services",
                "industry": "Insurance", "marketCap": "$1,234,000",
            }]}}

    def fake_get(url, **kwargs):
        request.update(url=url, **kwargs)
        return FakeResponse()

    monkeypatch.setattr(screener.requests, "get", fake_get)
    output = tmp_path / "nasdaq_screener.csv"
    database = screener.update_nasdaq_db(output)
    stored = pd.read_csv(output)

    assert request["url"] == screener.NASDAQ_SCREENER_URL
    assert request["headers"] == screener.NASDAQ_HEADERS
    assert database.loc[0, "symbol"] == "BRK-B"
    assert database.loc[0, "marketCap"] == 1_234_000
    assert stored.loc[0, "name"] == "Berkshire Hathaway"


def test_radar_exports_schema_and_nasdaq_metadata(tmp_path, monkeypatch):
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
    nasdaq_requests = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"rows": [{
                "symbol": "SPY", "name": "SPDR S&P 500 ETF", "sector": "ETF",
                "industry": "Large Blend", "marketCap": "123000000",
            }]}}

    def fake_get(*args, **kwargs):
        nasdaq_requests.append((args, kwargs))
        return FakeResponse()

    monkeypatch.setattr(screener.requests, "get", fake_get)
    data_client = object()
    received_clients = []

    def fake_fetch_daily_bars(symbols, **kwargs):
        received_clients.append(kwargs.get("client"))
        return {symbol: frame for symbol in symbols}

    monkeypatch.setattr(screener, "fetch_daily_bars", fake_fetch_daily_bars)

    class FakeTradingClient:
        def get_clock(self):
            return type("Clock", (), {"is_open": False})()

    radar = screener.generate_market_radar(
        ["SPY"], output, trading_client=FakeTradingClient(), data_client=data_client
    )
    stored = pd.read_csv(output)

    assert output.exists()
    assert list(radar.columns) == screener.RADAR_COLUMNS
    assert stored.loc[0, "Sector"] == "ETF"
    assert stored.loc[0, "Industry"] == "Large Blend"
    assert stored.loc[0, "SecurityName"] == "SPDR S&P 500 ETF"
    assert stored.loc[0, "MarketCap"] == 123_000_000
    assert stored.loc[0, "Symbol"] == "SPY"
    assert received_clients == [data_client]
    assert len(nasdaq_requests) == 1


def test_nasdaq_metadata_lookup_uses_index_and_unknown_fallback():
    database = pd.DataFrame([{
        "name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
        "marketCap": 456_000_000,
    }], index=pd.Index(["AAPL"], name="symbol"))

    found = screener._nasdaq_metadata(database, "AAPL")
    missing = screener._nasdaq_metadata(database, "MSFT")

    assert found == {
        "SecurityName": "Apple Inc.", "Sector": "Technology",
        "Industry": "Consumer Electronics", "MarketCap": 456_000_000,
    }
    assert missing == {
        "SecurityName": "Unknown", "Sector": "Unknown", "Industry": "Unknown", "MarketCap": None,
    }


def test_sync_market_radar_metadata_preserves_financial_fields(tmp_path):
    output = tmp_path / "market_radar.csv"
    row = screener._empty_radar_row("AAPL")
    row.update({
        "SecurityName": "Old name", "Sector": "Old sector", "Industry": "Old industry",
        "MarketCap": 100, "Close": 190.0, "Volume_SMA20": 5000,
        "ADX": 20.0, "ATR_pct": 1.5, "RSI": 35.0, "BB_lower": 180.0,
        "SMA_200": 170.0, "Validatore_Scalper": True, "Validatore_Trend": False,
    })
    screener._write_market_radar(pd.DataFrame([row]), output)
    _seed_nasdaq_db(output, [{
        "symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology",
        "industry": "Consumer Electronics", "marketCap": 3_000_000,
    }])

    screener.sync_market_radar_metadata(output)
    stored = pd.read_csv(output).iloc[0]

    assert stored["SecurityName"] == "Apple Inc."
    assert stored["Sector"] == "Technology"
    assert stored["Industry"] == "Consumer Electronics"
    assert stored["MarketCap"] == 3_000_000
    assert stored["Close"] == 190.0
    assert stored["Volume_SMA20"] == 5000
    assert stored["RSI"] == 35.0
    assert bool(stored["Validatore_Scalper"]) is True
    assert bool(stored["Validatore_Trend"]) is False


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

    _seed_nasdaq_db(output, [{
        "symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology",
        "industry": "Software", "marketCap": 200_000_000,
    }])
    monkeypatch.setattr(screener, "fetch_daily_bars", lambda symbols, **kwargs: {symbol: frame for symbol in symbols})

    class FakeTradingClient:
        def get_clock(self):
            return type("Clock", (), {"is_open": False})()

    screener.generate_market_radar(["AAPL"], output, trading_client=FakeTradingClient(), data_client=object())
    stored = pd.read_csv(output)

    assert stored.loc[0, "Sector"] == "Technology"
    assert stored.loc[0, "QuoteType"] == "Equity"
    assert stored.loc[0, "Close"] == 120.0
    assert stored.loc[0, "MarketCap"] == 123_000_000
    assert stored.loc[0, "SecurityName"] == "Unknown"


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

    monkeypatch.setattr(
        screener, "_load_nasdaq_db",
        lambda *_args: pd.DataFrame(columns=["name", "sector", "industry", "marketCap"], index=pd.Index([], name="symbol")),
    )
    monkeypatch.setattr(screener, "fetch_daily_bars", lambda symbols, **kwargs: {symbol: frame for symbol in symbols})

    class FakeTradingClient:
        def get_clock(self):
            return type("Clock", (), {"is_open": False})()

    screener.generate_market_radar(["AAPL"], output, trading_client=FakeTradingClient(), data_client=object())

    assert output.exists()
    stored = pd.read_csv(output)
    assert stored.loc[0, "Sector"] == "Unknown"


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

    _seed_nasdaq_db(output, [
        {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "industry": "Hardware", "marketCap": 100},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "industry": "Software", "marketCap": 200},
    ])
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
