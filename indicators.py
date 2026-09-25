"""Technical indicators computed on split/dividend-adjusted prices via pandas_ta."""

from __future__ import annotations

import pandas as pd

try:
    import pandas_ta as ta
except ImportError:  # Python < 3.12: API-compatible community fork
    import pandas_ta_classic as ta

INDICATOR_COLUMNS: tuple[str, ...] = ("BB_upper", "BB_mid", "BB_lower", "RSI", "ADX", "ATR")


def _pick_column(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """Return the first column of ``frame`` whose name starts with ``prefix`` (pandas_ta names vary by version)."""
    for col in frame.columns:
        if str(col).startswith(prefix):
            return frame[col]
    raise KeyError(f"pandas_ta output has no column starting with {prefix!r}: {list(frame.columns)}")


def adjusted_high_low(df: pd.DataFrame, price_col: str = "Adj Close") -> tuple[pd.Series, pd.Series]:
    """Scale High/Low by the Adj Close / Close ratio so range-based indicators share the adjusted price space."""
    factor = (df[price_col] / df["Close"]).where(df["Close"] > 0, 1.0)
    return df["High"] * factor, df["Low"] * factor


def add_indicators(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    bb_length: int = 20,
    bb_std: float = 2.0,
    rsi_length: int = 14,
    adx_length: int = 14,
    atr_length: int = 14,
) -> pd.DataFrame:
    """Add BB_upper/BB_mid/BB_lower, RSI, ADX and ATR columns computed on adjusted prices.

    Args:
        df: OHLCV DataFrame containing at least High, Low, Close and ``price_col``.
        price_col: Adjusted close column used for all calculations.
        bb_length: Bollinger Bands lookback.
        bb_std: Bollinger Bands standard-deviation multiplier.
        rsi_length: RSI lookback.
        adx_length: ADX lookback.
        atr_length: ATR lookback.

    Returns:
        A copy of ``df`` with the indicator columns appended (NaN during warm-up).
    """
    out = df.copy()
    close = out[price_col].astype(float)
    high, low = adjusted_high_low(out, price_col)

    bands = ta.bbands(close, length=bb_length, std=bb_std)
    out["BB_lower"] = _pick_column(bands, "BBL_")
    out["BB_mid"] = _pick_column(bands, "BBM_")
    out["BB_upper"] = _pick_column(bands, "BBU_")
    out["RSI"] = ta.rsi(close, length=rsi_length)
    out["ADX"] = _pick_column(ta.adx(high, low, close, length=adx_length), "ADX_")
    out["ATR"] = ta.atr(high, low, close, length=atr_length)
    return out
