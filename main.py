"""CLI entry point: walk-forward backtest of the range + trailing-stop strategy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import BacktestResult, load_ohlcv_csv, run_walk_forward
from macro_filter import load_calendar
from models import StrategyParams


def generate_synthetic_ohlcv(rows=400, seed=42, start_price=100.0, daily_vol=0.012, mean_reversion=0.06, intraday_vol=0.006, avg_volume=3_000_000, start_date="2024-01-02"):
    rng = np.random.default_rng(seed)
    log_anchor = math.log(start_price)
    log_price = np.empty(rows)
    log_price[0] = log_anchor
    shocks = rng.standard_t(df=5, size=rows) * daily_vol / math.sqrt(5 / 3)
    for t in range(1, rows):
        log_price[t] = log_price[t - 1] + mean_reversion * (log_anchor - log_price[t - 1]) + shocks[t]
    close = np.exp(log_price)
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, intraday_vol / 2, rows))
    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    high = body_high * (1 + np.abs(rng.normal(0, intraday_vol, rows)))
    low = body_low * (1 - np.abs(rng.normal(0, intraday_vol, rows)))
    volume = rng.lognormal(math.log(avg_volume), 0.3, rows).round()
    adj_factor = np.linspace(0.98, 1.0, rows)
    dates = pd.bdate_range(start=start_date, periods=rows)
    return pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Open": open_.round(4), "High": high.round(4), "Low": low.round(4), "Close": close.round(4), "Adj Close": (close * adj_factor).round(4), "Volume": volume.astype(np.int64)})


def parse_args(argv=None):
    defaults = StrategyParams()
    p = argparse.ArgumentParser(description="Walk-forward backtest: range trading + trailing stop")
    p.add_argument("--csv", default=None)
    p.add_argument("--risk", type=float, default=defaults.risk_pct)
    p.add_argument("--max-cap", type=float, default=defaults.max_cap_pct)
    p.add_argument("--trailing-pct", type=float, default=defaults.trailing_pct)
    p.add_argument("--commission", type=float, default=defaults.commission_pct)
    p.add_argument("--time-stop", type=int, default=defaults.time_stop)
    p.add_argument("--atr-mult", type=float, default=defaults.atr_mult)
    p.add_argument("--portfolio", type=float, default=100_000.0)
    p.add_argument("--in-sample-pct", type=float, default=0.7)
    p.add_argument("--calendar", default=None)
    p.add_argument("--rows", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="output")
    return p.parse_args(argv)


def format_summary(result):
    lines = ["", "=== OUT-OF-SAMPLE TPADES==="]
    if result.trades.empty:
        lines.append("(no trades)")
    else:
        view = result.trades.copy()
        for col in ("entry_date", "exit_date"):
            view[col] = pd.to_datetime(view[col]).dt.strftime("%Y-%m-%d")
        lines.append(view.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    lines += ["", "=== OUT-OF-SAMPLE METRICS ==="]
    for key, value in result.metrics.items():
        if isinstance(value, float) and key in ("win_rate", "max_drawdown", "return_pct"):
            shown = f"{value:.2%}"
        elif isinstance(value, float):
            shown = f"{value:,.2f}"
        else:
            shown = "n/a" if value is None else str(value)
        lines.append(f"{key:<16} {shown:>16}")
    return "\n".join(lines)


def save_outputs(result, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades_path, metrics_path = out / "trades.csv", out / "metrics.json"
    result.trades.to_csv(trades_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as fh: json.dump(result.metrics, fh, indent=2)
    return trades_path, metrics_path


def main(argv=None):
    args = parse_args(argv)
    if args.csv:
        raw, symbol = load_ohlcv_csv(args.csv), Path(args.csv).stem
        print(f"Loaded {len(raw)} rows from {args.csv}")
    else:
        raw, symbol = generate_synthetic_ohlcv(rows=args.rows, seed=args.seed), None
        print(f"No --csv given: generated {len(raw)} synthetic OHLCV rows (seed={args.seed})")
    calendar = load_calendar(args.calendar) if args.calendar else None
    params = StrategyParams(risk_pct=args.risk, max_cap_pct=args.max_cap, commission_pct=args.commission, trailing_pct=args.trailing_pct, time_stop=args.time_stop, atr_mult=args.atr_mult)
    result = run_walk_forward(raw, params, args.portfolio, args.in_sample_pct, calendar, symbol)
    print(format_summary(result))
    trades_path, metrics_path = save_outputs(result, args.output_dir)
    print(f"\nSaved {trades_path} and {metrics_path}")
    print(json.dumps(result.metrics, indent=2))
    return result.metrics


if __name__ == "__main__":
    main()
