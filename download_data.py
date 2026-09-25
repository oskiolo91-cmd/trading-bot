"""Download SPY data on first run if not present."""
from pathlib import Path
import yfinance as yf

DATA_PATH = Path(__file__).parent / "data" / "SPY.csv"

def ensure_spy_data():
    if DATA_PATH.exists(): return
    DATA_PATH.parent.mkdir(exist_ok=True)
    df = yf.download("SPY", start="2020-01-01", auto_adjust=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index.name = "Date"
    df.reset_index().to_csv(DATA_PATH, index=False)

if __name__ == "__main__":
    ensure_spy_data()
    print(f"SPY data saved to {DATA_PATH}")
