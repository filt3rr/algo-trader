"""Run vectorized backtests on 2 years of historical data.

Usage:
    python scripts/run_backtest.py

Fetches daily OHLCV for BTC/USD and ETH/USD from Alpaca (2023-2025),
runs all three signal strategies, and prints a formatted results table.
Results are also saved to data/backtest_results.json for PHASE_2_REPORT.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd


def main() -> None:
    from agent.data.ingestion import MarketDataClient
    from agent.strategy.backtest import BacktestResult, run_all_signals

    client = MarketDataClient()

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=730)  # 2 years

    symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]

    all_results: dict[str, dict] = {}
    printed: list[str] = []

    header = f"{'Symbol':<10} {'Signal':<12} {'Return':>9} {'B&H':>9} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7}"
    sep = "-" * len(header)
    printed.append(sep)
    printed.append(header)
    printed.append(sep)

    for symbol in symbols:
        print(f"Fetching 2y daily data for {symbol}...", flush=True)
        try:
            df = client.fetch_bars_single(symbol, timeframe="1d", start=start, end=end)
        except Exception as exc:
            print(f"  ERROR fetching {symbol}: {exc}")
            continue

        if df is None or len(df) < 100:
            print(f"  Insufficient data for {symbol} ({len(df) if df is not None else 0} bars)")
            continue

        print(f"  {len(df)} bars fetched. Running backtests...", flush=True)

        results = run_all_signals(df, symbol=symbol)
        all_results[symbol] = {}

        for sig, r in results.items():
            row = (
                f"{symbol:<10} {sig:<12} "
                f"{r.total_return_pct:>+8.1f}% "
                f"{r.buy_hold_return_pct:>+8.1f}% "
                f"{r.sharpe_ratio:>8.3f} "
                f"{r.max_drawdown_pct:>7.1f}% "
                f"{r.win_rate:>8.1%} "
                f"{r.n_trades:>7}"
            )
            printed.append(row)
            all_results[symbol][sig] = {
                "total_return_pct": round(r.total_return_pct, 2),
                "buy_hold_return_pct": round(r.buy_hold_return_pct, 2),
                "sharpe_ratio": round(r.sharpe_ratio, 4),
                "max_drawdown_pct": round(r.max_drawdown_pct, 2),
                "win_rate": round(r.win_rate, 4),
                "n_trades": r.n_trades,
                "calmar_ratio": round(r.calmar_ratio, 4),
                "start_date": r.start_date,
                "end_date": r.end_date,
                "n_bars": r.n_bars,
            }

        printed.append("")

    printed.append(sep)
    output = "\n".join(printed)
    print("\n" + output)

    out_path = Path("data/backtest_results.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
