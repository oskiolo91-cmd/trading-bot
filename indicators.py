"""Technical indicators computed on adjusted prices using pure pandas/numpy (no pandas_ta dependency)."""

from __future__ import annotations
import pandas as pd
import numpy as np

INDICATOR_COLUMNS: tuple[str, ...] = ("BB_upper", "BB_mid", "BB_lower", "RSI", "ADX", "ATR")


def adjusted_high_low(df: pd.DataFrame, price_col: str = "Adj Close") -> tuple[pd.Series, pd.Series]:
    """Scale High/Low by the Adj Close / Close ratio."""
    factor = (df[price_col] / df["Close"]).where(df["Close"] > 0, 1.0)
    return df["High"] * factor, df["Low"] * factor


def _bollinger_bands(close: pd.Series, length: int = 20, std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(length).mean()
    sigma = close.rolling(length).std(ddof=0)
    return mid + std * sigma, mid, mid - std * sigma


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = _atr(high, low, close, length)
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / length, adjust=False).mean()


def add_indicators(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    bb_length: int = 20,
    bb_std: float = 2.0,
    rsi_length: int = 14,
    adx_length: int = 14,
    atr_length: int = 14,
) -> pd.DataFrame:
    """Add BB_upper/BB_mid/BB_lower, RSI, ADX and ATR columns computed on adjusted prices."""
    out = df.copy()
    close = out[price_col].astype(float)
    high, low = adjusted_high_low(out, price_col)
    out["BB_upper"], out["BB_mid"], out["BB_lower"] = _bollinger_bands(close, bb_length, bb_std)
    out["RSI"] = _rsi(close, rsi_length)
    out["ATR"] = _atr(high, low, close, atr_length)
    out["ADX"] = _adx(high, low, close, adx_length)
    return out
