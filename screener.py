"""Offline batch screener selecting liquid, range-bound assets without imminent macro risk."""

from __future__ import annotations

import asyncio
import argparse
import logging
import math
import sqlite3
import threading
from contextlib import closing

import requests
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from alpaca_data import fetch_daily_bars
from backtest import download_daily_bars, load_ohlcv_csv, prepare_data
from macro_filter import has_risk_off_within
from bot_state import STATE_PATH, connect_db

LOG = logging.getLogger(__name__)
DATABASE_PATH = STATE_PATH
LEGACY_DATA_DIR = Path(__file__).resolve().parent
MARKET_RADAR_TABLE = "market_radar"
NASDAQ_SCREENER_TABLE = "nasdaq_screener"
NASDAQ_SCREENER_URL = (
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true"
)
NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/114.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
}
_RADAR_FILE_LOCK = threading.RLock()
_RADAR_WORKER_LOCK = threading.Lock()
_RADAR_WORKERS: dict[str, threading.Thread] = {}
_RADAR_WORKER_STATUS: dict[str, dict[str, Any]] = {}
RADAR_COLUMNS = [
    "Symbol", "SecurityName", "Sector", "Industry", "QuoteType", "MarketCap", "Close", "Volume_SMA20",
    "ADX", "ATR_pct", "RSI", "BB_lower", "SMA_200",
    "Validatore_Scalper", "Validatore_Trend",
]


def _empty_radar_row(symbol: str) -> dict[str, Any]:
    return {
        "Symbol": symbol,
        "SecurityName": "Unknown",
        "Sector": "Unknown",
        "Industry": "Unknown",
        "QuoteType": "Equity",
        "MarketCap": None,
        "Close": float("nan"),
        "Volume_SMA20": float("nan"),
        "ADX": float("nan"),
        "ATR_pct": float("nan"),
        "RSI": float("nan"),
        "BB_lower": float("nan"),
        "SMA_200": float("nan"),
        "Validatore_Scalper": False,
        "Validatore_Trend": False,
    }


def resolve_database_path(database_path: str | Path | None = None) -> Path:
    return Path(database_path) if database_path is not None else DATABASE_PATH


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _migrate_legacy_csv(table_name: str, filename: str, database_path: Path) -> None:
    """Import a legacy CSV once, only when its SQLite table has not been created."""
    if database_path.resolve() != DATABASE_PATH.resolve():
        return
    with _RADAR_FILE_LOCK:
        with closing(connect_db(database_path)) as connection:
            if _table_exists(connection, table_name):
                return
        legacy_path = LEGACY_DATA_DIR / filename
        if not legacy_path.exists():
            return
        try:
            frame = pd.read_csv(legacy_path)
            with closing(connect_db(database_path)) as connection:
                with connection:
                    if not _table_exists(connection, table_name):
                        frame.to_sql(table_name, connection, if_exists="fail", index=False)
            LOG.info("Imported legacy %s into SQLite table %s", filename, table_name)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, sqlite3.Error):
            LOG.exception("Could not migrate legacy CSV %s", legacy_path)


def _write_market_radar(frame: pd.DataFrame, output_path: str | Path | None = None) -> None:
    normalized = frame.reindex(columns=RADAR_COLUMNS)
    with closing(connect_db(resolve_database_path(output_path))) as connection:
        with connection:
            normalized.to_sql(MARKET_RADAR_TABLE, connection, if_exists="replace", index=False)


def _read_market_radar(output_path: str | Path | None = None) -> pd.DataFrame:
    database_path = resolve_database_path(output_path)
    _migrate_legacy_csv(MARKET_RADAR_TABLE, "market_radar.csv", database_path)
    try:
        with closing(connect_db(database_path)) as connection:
            if not _table_exists(connection, MARKET_RADAR_TABLE):
                return pd.DataFrame(columns=RADAR_COLUMNS)
            frame = pd.read_sql_query(f"SELECT * FROM {MARKET_RADAR_TABLE}", connection)
    except (OSError, sqlite3.Error, pd.errors.DatabaseError):
        LOG.exception("Could not read market radar from SQLite")
        return pd.DataFrame(columns=RADAR_COLUMNS)
    for column in RADAR_COLUMNS:
        if column not in frame.columns:
            frame[column] = _empty_radar_row("")[column]
    frame = frame[RADAR_COLUMNS]
    if "Symbol" in frame:
        frame = frame.dropna(subset=["Symbol"]).drop_duplicates("Symbol", keep="last")
        frame["Symbol"] = frame["Symbol"].astype(str)
    return frame


def initialize_market_radar(
    symbols: list[str], output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Persist catalog rows immediately, retaining any existing metadata and financial data."""
    path = resolve_database_path(output_path)
    catalog = list(dict.fromkeys(str(symbol).strip() for symbol in symbols if str(symbol).strip()))
    with _RADAR_FILE_LOCK:
        current = _read_market_radar(path)
        existing_symbols = set(current["Symbol"].astype(str)) if not current.empty else set()
        missing_rows = [_empty_radar_row(symbol) for symbol in catalog if symbol not in existing_symbols]
        with closing(connect_db(path)) as connection:
            table_exists = _table_exists(connection, MARKET_RADAR_TABLE)
        if missing_rows or not table_exists:
            initialized = pd.concat([current, pd.DataFrame(missing_rows, columns=RADAR_COLUMNS)], ignore_index=True)
            _write_market_radar(initialized, path)
            current = initialized
    return current


def resolve_nasdaq_db_path(output_path: str | Path | None = None) -> Path:
    return resolve_database_path(output_path)


def update_nasdaq_db(output_path: str | Path | None = None) -> pd.DataFrame:
    """Download and persist Nasdaq's public stock reference data in SQLite."""
    response = requests.get(NASDAQ_SCREENER_URL, headers=NASDAQ_HEADERS, timeout=30)
    response.raise_for_status()
    rows = response.json()["data"]["rows"]
    if not rows:
        raise ValueError("Nasdaq returned an empty symbol catalog")

    frame = pd.DataFrame(rows)
    required_fields = {"symbol", "name", "sector", "industry", "marketCap"}
    missing = required_fields - set(frame.columns)
    if missing:
        raise ValueError(f"Nasdaq response is missing fields: {', '.join(sorted(missing))}")
    frame = frame[["symbol", "name", "sector", "industry", "marketCap"]].copy()
    frame["symbol"] = frame["symbol"].fillna("").astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    frame = frame[frame["symbol"].ne("")].drop_duplicates("symbol", keep="last")
    frame["marketCap"] = pd.to_numeric(
        frame["marketCap"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )
    for column in ("name", "sector", "industry"):
        frame[column] = frame[column].replace({"N/A": pd.NA, "": pd.NA})
    if frame.empty:
        raise ValueError("Nasdaq response contained no usable symbols")

    with closing(connect_db(resolve_nasdaq_db_path(output_path))) as connection:
        with connection:
            frame.to_sql(NASDAQ_SCREENER_TABLE, connection, if_exists="replace", index=False)
    return frame


def _load_nasdaq_db(output_path: str | Path | None = None) -> pd.DataFrame:
    path = resolve_nasdaq_db_path(output_path)
    _migrate_legacy_csv(NASDAQ_SCREENER_TABLE, "nasdaq_screener.csv", path)
    with closing(connect_db(path)) as connection:
        table_exists = _table_exists(connection, NASDAQ_SCREENER_TABLE)
    if not table_exists:
        update_nasdaq_db(path)
    with closing(connect_db(path)) as connection:
        frame = pd.read_sql_query(f"SELECT * FROM {NASDAQ_SCREENER_TABLE}", connection)
    if "symbol" not in frame.columns:
        raise ValueError(f"Nasdaq database has no symbol column: {path}")
    frame["symbol"] = frame["symbol"].fillna("").astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    frame = frame.dropna(subset=["symbol"]).drop_duplicates("symbol", keep="last")
    return frame.set_index("symbol")


def _nasdaq_metadata(nasdaq_db: pd.DataFrame, symbol: str) -> dict[str, Any]:
    symbol = str(symbol).strip().upper().replace(".", "-")
    if symbol not in nasdaq_db.index:
        return {"SecurityName": "Unknown", "Sector": "Unknown", "Industry": "Unknown", "MarketCap": None}
    record = nasdaq_db.loc[symbol]
    if isinstance(record, pd.DataFrame):
        record = record.iloc[-1]

    def value_or_unknown(field: str) -> str:
        value = record.get(field)
        return "Unknown" if _metadata_value_is_missing(value) else str(value)

    market_cap = record.get("marketCap")
    try:
        market_cap = int(float(market_cap)) if not _metadata_value_is_missing(market_cap) else None
    except (TypeError, ValueError, OverflowError):
        market_cap = None
    return {
        "SecurityName": value_or_unknown("name"),
        "Sector": value_or_unknown("sector"),
        "Industry": value_or_unknown("industry"),
        "MarketCap": market_cap,
    }


def _upsert_market_radar_rows(rows: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    with _RADAR_FILE_LOCK:
        current = _read_market_radar(output_path)
        if not current.empty:
            current = current.set_index("Symbol").astype(object)
        else:
            current = pd.DataFrame(columns=RADAR_COLUMNS).set_index("Symbol").astype(object)
        for row in rows:
            symbol = str(row["Symbol"])
            if symbol not in current.index:
                current.loc[symbol] = {
                    column: value
                    for column, value in _empty_radar_row(symbol).items()
                    if column != "Symbol"
                }
            for column, value in row.items():
                if column in RADAR_COLUMNS and column != "Symbol":
                    current.loc[symbol, column] = value
        updated = current.reset_index().reindex(columns=RADAR_COLUMNS)
        _write_market_radar(updated, output_path)
    return updated


def get_market_radar_worker_status(output_path: str | Path | None = None) -> dict[str, Any]:
    path = str(resolve_database_path(output_path).resolve())
    with _RADAR_WORKER_LOCK:
        return dict(_RADAR_WORKER_STATUS.get(path, {"state": "idle", "completed": 0, "total": 0, "error": None}))


def start_market_radar_worker(
    symbols: list[str],
    output_path: str | Path | None = None,
    force: bool = False,
    **generation_options,
) -> bool:
    """Start one daemon worker per SQLite database; safe on every Streamlit rerun."""
    path = resolve_database_path(output_path)
    key = str(path.resolve())
    catalog = list(dict.fromkeys(str(symbol).strip() for symbol in symbols if str(symbol).strip()))
    if not catalog:
        return False
    initialize_market_radar(catalog, path)
    with _RADAR_WORKER_LOCK:
        worker = _RADAR_WORKERS.get(key)
        if worker is not None and worker.is_alive():
            return False
        previous_status = _RADAR_WORKER_STATUS.get(key, {})
        same_catalog = previous_status.get("symbols") == catalog
        if previous_status.get("state") in {"complete", "failed"} and same_catalog and not force:
            return False
        _RADAR_WORKER_STATUS[key] = {
            "state": "running", "completed": 0, "total": len(catalog),
            "error": None, "symbols": catalog,
        }

        def update_progress(partial: pd.DataFrame) -> None:
            with _RADAR_WORKER_LOCK:
                _RADAR_WORKER_STATUS[key]["completed"] = len(partial)

        def run_worker() -> None:
            try:
                generate_market_radar(
                    symbols=catalog,
                    output_path=path,
                    progress_callback=update_progress,
                    **generation_options,
                )
            except Exception as exc:
                LOG.exception("Market radar background worker failed")
                with _RADAR_WORKER_LOCK:
                    _RADAR_WORKER_STATUS[key].update(state="failed", error=str(exc))
            else:
                with _RADAR_WORKER_LOCK:
                    _RADAR_WORKER_STATUS[key].update(state="complete", completed=len(catalog))

        worker = threading.Thread(
            target=run_worker,
            name=f"market-radar-{path.stem}",
            daemon=True,
        )
        _RADAR_WORKERS[key] = worker
        worker.start()
    return True


def resolve_market_radar_path(output_path: str | Path | None = None) -> Path:
    """Compatibility wrapper returning the SQLite database path, not a CSV path."""
    return resolve_database_path(output_path)


def evaluate_asset(csv_path, min_avg_volume=1_000_000, lookback=20, adx_max=25.0, min_lateral_ratio=0.5, min_atr_pct=0.005):
    data = prepare_data(load_ohlcv_csv(csv_path))
    recent = data.tail(lookback)
    avg_volume = float(recent["Volume"].mean()) if len(recent) else 0.0
    adx = recent["ADX"].dropna()
    lateral_ratio = float((adx < adx_max).sum() / lookback) if lookback > 0 else 0.0
    last = data.iloc[-1] if len(data) else None
    atr_pct = float(last["ATR"] / last["Close"]) if last is not None and last["Close"] > 0 else float("nan")
    passes = len(recent) >= lookback and avg_volume >= min_avg_volume and lateral_ratio > min_lateral_ratio and atr_pct >= min_atr_pct
    return {"avg_volume": avg_volume, "atr_pct": atr_pct, "lateral_ratio": lateral_ratio, "passes": bool(passes)}


def screen_assets(csv_paths, min_avg_volume=1_000_000, calendar=None, as_of=None, macro_days_ahead=1, lookback=20, adx_max=25.0, min_lateral_ratio=0.5, min_atr_pct=0.005):
    check_day = as_of or date.today()
    selected = []
    for path in csv_paths:
        p = Path(path)
        if calendar and has_risk_off_within(check_day, calendar, macro_days_ahead, symbol=p.stem): continue
        stats = evaluate_asset(p, min_avg_volume, lookback, adx_max, min_lateral_ratio, min_atr_pct)
        if stats["passes"]: selected.append(p.name)
    return selected


async def screen_assets_async(csv_paths, min_avg_volume=1_000_000, calendar=None, as_of=None, macro_days_ahead=1):
    results = await asyncio.gather(*(asyncio.to_thread(screen_assets, [p], min_avg_volume, calendar, as_of, macro_days_ahead) for p in csv_paths))
    return [n for chunk in results for n in chunk]


def load_index_universe() -> list[str]:
    """Load S&P 500 and Nasdaq 100 constituents from their public component tables."""
    urls = (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/Nasdaq-100",
    )
    symbols: list[str] = []
    for url in urls:
        tables = pd.read_html(url)
        for table in tables:
            symbol_column = next(
                (column for column in table.columns
                 if str(column).strip().lower() in {"symbol", "ticker", "ticker symbol"}),
                None,
            )
            if symbol_column is None:
                continue
            symbols.extend(
                str(value).strip().replace(".", "-")
                for value in table[symbol_column].dropna()
                if str(value).strip()
            )
            break
    if not symbols:
        raise RuntimeError("Could not load S&P 500/Nasdaq 100 constituents")
    return list(dict.fromkeys(symbols))


def _metadata_value_is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "unknown"


def _last_closed_bars(frame: pd.DataFrame, market_is_open: bool) -> pd.DataFrame:
    if frame.empty or not market_is_open:
        return frame
    today = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date()
    eastern_dates = pd.Index([pd.Timestamp(timestamp).tz_convert("America/New_York").date() for timestamp in frame.index])
    return frame.loc[eastern_dates < today]


def _radar_row(symbol: str, bars: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    if bars.empty:
        raise ValueError(f"No Alpaca daily bars for {symbol}")
    frame = bars.copy()
    frame["Adj Close"] = frame["Close"]
    data = prepare_data(frame)
    if data.empty:
        raise ValueError(f"No usable daily candles for {symbol}")

    row = data.iloc[-1]
    close = float(row["Close"])
    atr = float(row["ATR"])
    adx = float(row["ADX"])
    rsi = float(row["RSI"])
    bb_lower = float(row["BB_lower"])
    sma_200 = float(data["Close"].rolling(window=200, min_periods=200).mean().iloc[-1])
    volume_sma20 = float(data["Volume"].rolling(window=20, min_periods=20).mean().iloc[-1])

    scalper = all(math.isfinite(value) for value in (close, adx, rsi, bb_lower)) and (
        adx < 25 and rsi < 35 and close <= bb_lower
    )
    trend = scalper and math.isfinite(sma_200) and close > sma_200
    return {
        "Symbol": symbol,
        **metadata,
        "Close": close,
        "Volume_SMA20": volume_sma20,
        "ADX": adx,
        "ATR_pct": atr / close * 100 if close > 0 and math.isfinite(atr) else float("nan"),
        "RSI": rsi,
        "BB_lower": bb_lower,
        "SMA_200": sma_200,
        "Validatore_Scalper": bool(scalper),
        "Validatore_Trend": bool(trend),
    }


def sync_market_radar_metadata(output_path: str | Path | None = None) -> None:
    """Apply the SQLite Nasdaq master to radar rows without changing financial fields."""
    radar_path = resolve_database_path(output_path)
    try:
        nasdaq_db = _load_nasdaq_db(radar_path)
    except Exception:
        LOG.exception("Could not load Nasdaq security master for radar sync")
        raise

    radar = _read_market_radar(radar_path)
    rows: list[dict[str, Any]] = []
    for record in radar.to_dict("records"):
        symbol = str(record["Symbol"])
        metadata = _nasdaq_metadata(nasdaq_db, symbol)
        for field in ("SecurityName", "Sector", "Industry"):
            if _metadata_value_is_missing(metadata[field]):
                metadata[field] = record.get(field, "Unknown")
        if metadata["MarketCap"] is None:
            metadata["MarketCap"] = record.get("MarketCap")
        rows.append({"Symbol": symbol, **metadata})
    if rows:
        _upsert_market_radar_rows(rows, radar_path)


def generate_market_radar(
    symbols: list[str] | None = None,
    output_path: str | Path | None = None,
    metadata_workers: int = 8,
    trading_client=None,
    data_client=None,
    progress_callback=None,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Enrich the SQLite catalog in small, durable chunks."""
    output_path = resolve_database_path(output_path)
    symbols = list(dict.fromkeys(symbols if symbols is not None else load_index_universe()))
    if not symbols:
        raise ValueError("At least one ticker is required")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    initialize_market_radar(symbols, output_path)
    try:
        nasdaq_db = _load_nasdaq_db(output_path)
    except Exception:
        LOG.exception("Nasdaq security master unavailable; continuing with cached metadata")
        nasdaq_db = pd.DataFrame(columns=["name", "sector", "industry", "marketCap"])
        nasdaq_db.index.name = "symbol"
    cached_radar = _read_market_radar(Path(output_path))
    cached_metadata = cached_radar.set_index("Symbol").to_dict("index") if not cached_radar.empty else {}
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=400)
    if trading_client is None:
        from live_trader import get_alpaca_client

        trading_client = get_alpaca_client()
    try:
        market_is_open = bool(trading_client.get_clock().is_open)
    except Exception:
        LOG.warning("Could not read Alpaca market clock; today's bar will be excluded")
        market_is_open = True

    completed_rows: list[dict[str, Any]] = []
    for offset in range(0, len(symbols), batch_size):
        batch = symbols[offset:offset + batch_size]
        metadata_by_symbol: dict[str, dict[str, Any]] = {}
        for symbol in batch:
            metadata = _nasdaq_metadata(nasdaq_db, symbol)
            cached = cached_metadata.get(symbol, {})
            for field in ("SecurityName", "Sector", "Industry"):
                if _metadata_value_is_missing(metadata[field]):
                    metadata[field] = cached.get(field, "Unknown")
            if metadata["MarketCap"] is None:
                metadata["MarketCap"] = cached.get("MarketCap")
            metadata["QuoteType"] = cached.get("QuoteType", "Equity")
            metadata_by_symbol[symbol] = metadata
        alpaca_symbols = [symbol.replace("-", ".") for symbol in batch]
        try:
            frames = fetch_daily_bars(alpaca_symbols, start=start, end=end, client=data_client)
        except Exception:
            LOG.exception("Could not load Alpaca bars for market radar batch %s", batch)
            frames = {}

        rows_to_write: list[dict[str, Any]] = []
        for symbol, alpaca_symbol in zip(batch, alpaca_symbols):
            frame = _last_closed_bars(frames.get(alpaca_symbol, pd.DataFrame()), market_is_open)
            try:
                if frame.empty:
                    raise ValueError("No Alpaca bars available")
                row = _radar_row(symbol, frame, metadata_by_symbol[symbol])
            except Exception as exc:
                LOG.warning("Keeping %s in market radar without fresh financial data: %s", symbol, exc)
                row = {"Symbol": symbol, **metadata_by_symbol[symbol]}
            rows_to_write.append(row)
            completed_rows.append(row)

        current = _upsert_market_radar_rows(rows_to_write, Path(output_path))
        if progress_callback is not None:
            progress_callback(pd.DataFrame(completed_rows))

    return _read_market_radar(Path(output_path)).reindex(columns=RADAR_COLUMNS)


def main(argv=None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Build the Alpaca/Nasdaq market radar in SQLite")
    parser.add_argument("--symbols", help="Comma-separated symbols; defaults to S&P 500 + Nasdaq 100")
    parser.add_argument("--output", default=str(DATABASE_PATH), help="SQLite database path")
    args = parser.parse_args(argv)
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()] if args.symbols else None
    return generate_market_radar(symbols=symbols, output_path=args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
