"""
A/B Test: Kronos vs Claude on NSE tickers

Compares:
  - Claude-only (baseline TradingAgents with market analyst)
  - Kronos-only (just the quantitative signal, no LLM debate)
  - Kronos + Claude (hybrid: Kronos pre-analysis injected as context)

Metrics collected:
  - Signal (BUY/SELL/HOLD/SKIP)
  - 5-day directional accuracy vs actual yfinance data
  - Latency (seconds per ticker)
  - Cost estimate (API calls vs self-hosted)

Usage:
    cd ~/trading_enhancement/TradingAgents
    source ../kronos_env/bin/activate
    pip install -e . -q
    python tradingagents/kronos/ab_test.py --tickers RELIANCE TCS INFY --date 2025-04-01
"""

import sys
import os
import time
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Add kronos to path
KRONOS_REPO = Path(__file__).parents[3] / "kronos"
sys.path.insert(0, str(KRONOS_REPO))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ab_test")

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def fetch_actual_prices(ticker: str, start: str, end: str) -> list:
    """Fetch actual closing prices from yfinance for validation."""
    import yfinance as yf
    sym = ticker + ".NS" if "." not in ticker else ticker
    df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return []
    closes = df["Close"].dropna()
    # Flatten MultiIndex columns (yfinance >= 0.2 returns MultiIndex for single tickers)
    if hasattr(closes, 'columns'):
        closes = closes.iloc[:, 0]
    return [float(v) for v in closes.values.flatten()]


def directional_accuracy(actual, predicted):
    """Fraction of days where direction matches."""
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    n = min(len(actual), len(predicted)) - 1
    if n <= 0:
        return float("nan")
    actual_dir = np.sign(np.diff(actual[:n+1]))
    pred_dir = np.sign(np.diff(predicted[:n+1]))
    return float(np.mean(actual_dir == pred_dir))


# ------------------------------------------------------------------ #
#  Kronos-only analysis
# ------------------------------------------------------------------ #

def run_kronos_only(tickers: list, pred_len: int = 5) -> dict:
    """Run Kronos standalone analysis (no LLM) on each ticker."""
    from tradingagents.kronos import KronosWrapper, KronosSignalParser

    print("\n[A] Kronos-only analysis")
    print("-" * 50)

    wrapper = KronosWrapper.get_instance(sample_count=3)
    wrapper.load_model()

    results = {}
    for ticker in tickers:
        t0 = time.time()
        try:
            prediction = wrapper.predict(ticker, pred_len=pred_len)
            signal = KronosSignalParser.parse(prediction)
            elapsed = time.time() - t0
            results[ticker] = {
                "signal": signal["signal"],
                "confidence": signal["confidence"],
                "pct_change_5d": signal["pct_change_5d"],
                "pred_close": prediction["pred_close_5d"],
                "latency_s": round(elapsed, 3),
                "cost_usd": 0.0,  # self-hosted
                "error": None,
            }
            print(
                f"  {ticker:15s}  {signal['signal']:4s}  conf={signal['confidence']:.0%}"
                f"  Δ={signal['pct_change_5d']:+.1f}%  lat={elapsed:.2f}s"
            )
        except Exception as e:
            results[ticker] = {"signal": "ERROR", "error": str(e), "latency_s": time.time()-t0}
            print(f"  {ticker:15s}  ERROR: {e}")

    return results


# ------------------------------------------------------------------ #
#  Post-hoc accuracy validation
# ------------------------------------------------------------------ #

def validate_predictions(results: dict, trade_date: str, pred_len: int = 5) -> dict:
    """Add directional accuracy by checking actual yfinance prices."""
    import pandas as pd
    from datetime import datetime, timedelta

    ref_date = datetime.strptime(trade_date, "%Y-%m-%d")
    end_date = (ref_date + timedelta(days=pred_len + 14)).strftime("%Y-%m-%d")

    validated = {}
    for ticker, data in results.items():
        if data.get("error") or "pred_close" not in data:
            validated[ticker] = data
            continue

        actuals = fetch_actual_prices(ticker, trade_date, end_date)
        if len(actuals) >= pred_len:
            actuals = actuals[:pred_len]
            da = directional_accuracy(actuals, data["pred_close"])
            data["directional_accuracy"] = round(da, 3)
            data["actual_close_5d"] = [round(float(v), 2) for v in actuals]
        else:
            data["directional_accuracy"] = float("nan")
            data["actual_close_5d"] = actuals

        validated[ticker] = data

    return validated


# ------------------------------------------------------------------ #
#  Summary table
# ------------------------------------------------------------------ #

def print_summary(kronos_results: dict, trade_date: str):
    print("\n" + "=" * 70)
    print(f"A/B TEST SUMMARY — Trade date: {trade_date}")
    print("=" * 70)

    print(f"\n{'Ticker':<15} {'Signal':<6} {'Conf':>6} {'Δ5d':>7} {'DA':>6} {'Lat(s)':>8} {'Cost':>8}")
    print("-" * 65)
    for t, d in kronos_results.items():
        if d.get("error"):
            print(f"  {t:<13}  ERROR")
            continue
        da = f"{d.get('directional_accuracy', float('nan')):.0%}" if d.get('directional_accuracy') == d.get('directional_accuracy') else "N/A"
        print(
            f"  {t:<13}  {d['signal']:<5} {d['confidence']:>5.0%}  "
            f"{d['pct_change_5d']:>+6.1f}%  {da:>5}  {d['latency_s']:>6.2f}s  "
            f"${d.get('cost_usd', 0):.4f}"
        )

    valid = [d for d in kronos_results.values() if not d.get("error") and d.get("directional_accuracy") == d.get("directional_accuracy")]
    if valid:
        avg_da = np.mean([d["directional_accuracy"] for d in valid])
        avg_lat = np.mean([d["latency_s"] for d in valid])
        total_cost = sum(d.get("cost_usd", 0) for d in valid)
        print("-" * 65)
        print(f"  {'AVERAGE':<13}  {'':5} {'':5}  {'':6}  {avg_da:>5.0%}  {avg_lat:>6.2f}s  ${total_cost:.4f}")

    print()


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Kronos A/B test vs Claude on NSE tickers")
    parser.add_argument("--tickers", nargs="+", default=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"])
    parser.add_argument("--date", default="2025-04-01", help="Trade date for validation (YYYY-MM-DD)")
    parser.add_argument("--pred-len", type=int, default=5, help="Days to forecast")
    parser.add_argument("--output", default=None, help="JSON output file path")
    args = parser.parse_args()

    print("=" * 70)
    print("KRONOS vs CLAUDE A/B TEST")
    print(f"Tickers: {', '.join(args.tickers)}")
    print(f"Trade date: {args.date}  |  Forecast horizon: {args.pred_len} days")
    print("=" * 70)

    # Run Kronos
    kronos_results = run_kronos_only(args.tickers, pred_len=args.pred_len)

    # Validate against actual prices
    print("\nFetching actual prices for directional accuracy...")
    kronos_results = validate_predictions(kronos_results, args.date, pred_len=args.pred_len)

    print_summary(kronos_results, args.date)

    # Note on Claude comparison
    print("\nNOTE: Claude comparison requires LLM API keys (ANTHROPIC_API_KEY / OPENAI_API_KEY).")
    print("To run Claude baseline, use the standard TradingAgents CLI:")
    print("  cd ~/trading_enhancement/TradingAgents && python main.py --ticker RELIANCE --date 2025-04-01")

    output = {
        "test_date": args.date,
        "tickers": args.tickers,
        "pred_len": args.pred_len,
        "kronos_results": {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in kronos_results.items()},
        "summary": {
            "model": "NeoQuasar/Kronos-base",
            "avg_latency_s": round(float(np.mean([d["latency_s"] for d in kronos_results.values() if not d.get("error")])), 3),
            "avg_directional_accuracy": round(float(np.nanmean([d.get("directional_accuracy", float("nan")) for d in kronos_results.values()])), 3),
            "total_cost_usd": 0.0,
        }
    }

    out_path = args.output or os.path.expanduser("~/trading_enhancement/ab_test_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
