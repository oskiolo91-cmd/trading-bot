"""Offline batch screener selecting liquid, range-bound assets without imminent macro risk."""

from __future__ import annotations

import asyncio
import argparse
import logging
import math
import os
import tempfile
import threading
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from alpaca_data import fetch_daily_bars
from backtest import download_daily_bars, load_ohlcv_csv, prepare_data
from macro_filter import has_risk_off_within

LOG = logging.getLogger(__name__)
# The local CSV is treated as a lightweight database: the dashboard reads it immediately and the
# background worker updates it asynchronously as data is fetched for each ticker.
MARKET_RADAR_PATH = Path(__file__).resolve().with_name("market_radar.csv")
_RADAR_FILE_LOCK = threading.RLock()
_RADAR_WORKER_LOCK = threading.Lock()
_RADAR_WORKERS: dict[str, threading.Thread] = {}
_RADAR_WORKER_STATUS: dict[str, dict[str, Any]] = {}
RADAR_COLUMNS = [
    "Symbol", "Sector", "QuoteType", "MarketCap", "Close", "Volume_SMA20",
    "ADX", "ATR_pct", "RSI", "BB_lower", "SMA_200",
    "Validatore_Scalper", "Validatore_Trend",
]


def _empty_radar_row(symbol: str) -> dict[str, Any]:
    return {
        "Symbol": symbol,
        "Sector": "Unknown",
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


def _write_market_radar(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = frame.reindex(columns=RADAR_COLUMNS)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv",
            prefix=f".{output_path.name}.", dir=output_path.parent, delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            normalized.to_csv(temporary_file, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _read_market_radar(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame(columns=RADAR_COLUMNS)
    try:
        frame = pd.read_csv(output_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        LOG.exception("Could not read market radar CSV at %s", output_path)
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
    path = resolve_market_radar_path(output_path)
    catalog = list(dict.fromkeys(str(symbol).strip() for symbol in symbols if str(symbol).strip()))
    with _RADAR_FILE_LOCK:
        current = _read_market_radar(path)
        existing_symbols = set(current["Symbol"].astype(str)) if not current.empty else set()
        missing_rows = [_empty_radar_row(symbol) for symbol in catalog if symbol not in existing_symbols]
        try:
            existing_columns = pd.read_csv(path, nrows=0).columns.tolist() if path.exists() else []
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            existing_columns = []
        if missing_rows or existing_columns != RADAR_COLUMNS:
            initialized = pd.concat([current, pd.DataFrame(missing_rows, columns=RADAR_COLUMNS)], ignore_index=True)
            _write_market_radar(initialized, path)
            current = initialized
    return current


def _upsert_market_radar_rows(rows: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    with _RADAR_FILE_LOCK:
        current = _read_market_radar(output_path)
        if not current.empty:
            current = current.set_index("Symbol")
        else:
            current = pd.DataFrame(columns=RADAR_COLUMNS).set_index("Symbol")
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
    path = str(resolve_market_radar_path(output_path).resolve())
    with _RADAR_WORKER_LOCK:
        return dict(_RADAR_WORKER_STATUS.get(path, {"state": "idle", "completed": 0, "total": 0, "error": None}))


def start_market_radar_worker(
    symbols: list[str],
    output_path: str | Path | None = None,
    force: bool = False,
    **generation_options,
) -> bool:
    """Start one daemon worker per CSV path; safe to call on every Streamlit rerun."""
    path = resolve_market_radar_path(output_path)
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
    candidates: list[Path] = []
    if output_path is not None:
        requested = Path(output_path)
        candidates.append(requested)
        candidates.append(requested.parent / "market_radar.csv")
    candidates.extend([
        Path.cwd() / "market_radar.csv",
        MARKET_RADAR_PATH,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(output_path) if output_path is not None else MARKET_RADAR_PATH


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


def _yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def _metadata_value_is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "unknown"


def _read_cached_market_radar_metadata(output_path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(output_path)
    if not path.exists():
        return {}
    try:
        cached = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return {}
    if cached.empty:
        return {}

    metadata_by_symbol: dict[str, dict[str, Any]] = {}
    for row in cached.to_dict("records"):
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol:
            continue
        sector = row.get("Sector")
        quote_type = row.get("QuoteType")
        market_cap = row.get("MarketCap")
        market_cap_value = None
        if market_cap is not None and not _metadata_value_is_missing(market_cap):
            try:
                market_cap_value = int(market_cap)
            except (TypeError, ValueError):
                market_cap_value = None
        metadata_by_symbol[symbol] = {
            "Sector": str(sector) if not _metadata_value_is_missing(sector) else "Unknown",
            "QuoteType": str(quote_type) if not _metadata_value_is_missing(quote_type) else "Equity",
            "MarketCap": market_cap_value,
        }
    return metadata_by_symbol


def _merge_metadata_with_cache(cached: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cached or {})
    for field, fallback in {
        "Sector": "Unknown",
        "QuoteType": "Equity",
        "MarketCap": None,
    }.items():
        value = fresh.get(field)
        if field == "MarketCap" and value is not None:
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = None
        if field in {"Sector", "QuoteType"} and _metadata_value_is_missing(value):
            value = merged.get(field) or fallback
        if field == "MarketCap" and _metadata_value_is_missing(value):
            value = merged.get(field)
        if value is None and field == "MarketCap":
            merged[field] = None
        elif not _metadata_value_is_missing(value):
            merged[field] = value
        elif field in merged and not _metadata_value_is_missing(merged.get(field)):
            continue
        else:
            merged[field] = fallback
    return merged


def _fetch_yahoo_metadata(symbol: str) -> dict[str, Any]:
    ticker = yf.Ticker(_yahoo_symbol(symbol))
    info: dict[str, Any] = {}
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            candidate = ticker.get_info() or {}
            if candidate:
                info = candidate
                break
        except Exception as exc:
            last_error = exc
            LOG.warning("Metadata fetch failed for %s (attempt %s/3): %s", symbol, attempt + 1, exc)

        try:
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info is not None:
                merged = dict(info)
                market_cap = getattr(fast_info, "market_cap", None)
                if market_cap is not None and "marketCap" not in merged:
                    merged["marketCap"] = market_cap
                quote_type = getattr(fast_info, "quote_type", None)
                if quote_type is not None and "quoteType" not in merged:
                    merged["quoteType"] = quote_type
                if merged:
                    info = merged
                    break
        except Exception:
            pass

        if attempt < 2:
            time_module.sleep(0.5 * (attempt + 1))

    if not info and last_error is not None:
        LOG.warning("Using fallback metadata for %s after repeated Yahoo failures", symbol)

    raw_quote_type = str(info.get("quoteType") or "").upper()
    quote_type = "ETF" if raw_quote_type == "ETF" else "Equity" if raw_quote_type in {"EQUITY", "", "STOCK"} else raw_quote_type
    market_cap = info.get("marketCap")
    if market_cap is None:
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info is not None:
            market_cap = getattr(fast_info, "market_cap", None)
    sector = info.get("sector") or info.get("industry") or "Unknown"

    return {
        "Sector": str(sector),
        "QuoteType": quote_type,
        "MarketCap": int(market_cap) if market_cap is not None else None,
    }


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


def generate_market_radar(
    symbols: list[str] | None = None,
    output_path: str | Path | None = None,
    metadata_workers: int = 8,
    trading_client=None,
    data_client=None,
    progress_callback=None,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Enrich the bootstrapped local catalog in small, durable chunks."""
    output_path = resolve_market_radar_path(output_path)
    symbols = list(dict.fromkeys(symbols if symbols is not None else load_index_universe()))
    if not symbols:
        raise ValueError("At least one ticker is required")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    initialize_market_radar(symbols, output_path)
    cached_metadata = _read_cached_market_radar_metadata(output_path)
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
        missing_metadata = [
            symbol for symbol in batch
            if _metadata_value_is_missing(cached_metadata.get(symbol, {}).get("Sector"))
        ]
        with ThreadPoolExecutor(max_workers=max(1, min(metadata_workers, len(missing_metadata)))) as executor:
            futures = {executor.submit(_fetch_yahoo_metadata, symbol): symbol for symbol in missing_metadata}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    fresh = future.result()
                except Exception:
                    LOG.exception("Could not load metadata for %s", symbol)
                    fresh = {"Sector": "Unknown", "QuoteType": "Equity", "MarketCap": None}
                cached_metadata[symbol] = _merge_metadata_with_cache(cached_metadata.get(symbol, {}), fresh)

        for symbol in batch:
            metadata_by_symbol[symbol] = cached_metadata.get(symbol, {
                "Sector": "Unknown", "QuoteType": "Equity", "MarketCap": None,
            })
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
    parser = argparse.ArgumentParser(description="Build the Alpaca/yfinance market radar CSV")
    parser.add_argument("--symbols", help="Comma-separated symbols; defaults to S&P 500 + Nasdaq 100")
    parser.add_argument("--output", default=str(MARKET_RADAR_PATH))
    args = parser.parse_args(argv)
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()] if args.symbols else None
    return generate_market_radar(symbols=symbols, output_path=args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
