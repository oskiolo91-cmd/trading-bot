"""Download SPY daily bars from Alpaca if the local CSV does not exist."""
from datetime import datetime, timezone
from pathlib import Path

from backtest import download_daily_bars

DATA_PATH = Path(__file__).parent / "data" / "SPY.csv"

def ensure_spy_data():
    if DATA_PATH.exists(): return
    DATA_PATH.parent.mkdir(exist_ok=True)
    df = download_daily_bars("SPY", lookback_days=3650, end=datetime.now(timezone.utc))
    df.reset_index().to_csv(DATA_PATH, index=False)

if __name__ == "__main__":
    ensure_spy_data()
    print(f"SPY data saved to {DATA_PATH}")
