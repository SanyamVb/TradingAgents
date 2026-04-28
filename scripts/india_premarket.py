#!/usr/bin/env python3
"""
NSE Pre-Market Analysis Runner
================================
Runs TradingAgents for a list of NSE tickers using the previous session's
end-of-day data and prints a clean recommendation summary before market open.

Usage:
    python scripts/india_premarket.py                         # uses DEFAULT_TICKERS
    python scripts/india_premarket.py RELIANCE.NS TCS.NS      # override from CLI args

Schedule (cron) — runs at 8:45 AM IST (03:15 UTC) Mon–Fri:
    15 3 * * 1-5  cd /path/to/TradingAgents && python scripts/india_premarket.py >> ~/.tradingagents/logs/premarket.log 2>&1

Requirements:
    pip install pandas-market-calendars   # for NSE holiday calendar (Option A)
    # OR use the built-in fallback below (Option B, zero extra deps)
"""

import sys
import os
import re
import logging
from datetime import date, timedelta

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file if present (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

logging.basicConfig(level=logging.WARNING)  # suppress verbose LangGraph logs

# ── Configuration ──────────────────────────────────────────────────────────────

# Edit this list or pass tickers as CLI arguments
# Top 10 high-performing Nifty 50 large-caps (as of 2026)
DEFAULT_TICKERS = [
    "RELIANCE.NS",      # Largest company by market cap — energy, retail, telecom
    "TCS.NS",           # Top IT; consistent earnings and dividend payer
    "HDFCBANK.NS",      # Largest private bank
    "ICICIBANK.NS",     # High-growth private bank
    "BHARTIARTL.NS",    # Telecom leader; strong subscriber growth
    "SBIN.NS",          # Largest PSU bank; high beta, high volume
    "INFY.NS",          # IT bellwether; large institutional ownership
    "HINDUNILVR.NS",    # FMCG defensive; consistent compounder
    "ITC.NS",           # Conglomerate; high dividend yield
    "LT.NS",            # Infrastructure & engineering; capex play
]

# LLM settings — set API key via environment variable
LLM_PROVIDER    = os.getenv("TRADINGAGENTS_LLM_PROVIDER", "openai")
DEEP_THINK_LLM  = os.getenv("TRADINGAGENTS_DEEP_LLM", "gpt-4o")
QUICK_THINK_LLM = os.getenv("TRADINGAGENTS_QUICK_LLM", "gpt-4o-mini")
BACKEND_URL     = os.getenv("TRADINGAGENTS_BACKEND_URL", None)  # custom proxy/Azure URL


# ── NSE Trading Day Logic ──────────────────────────────────────────────────────

# Option A: uses pandas_market_calendars (recommended — self-maintaining holiday list)
def _get_last_nse_day_mcal() -> str:
    import pandas_market_calendars as mcal
    nse = mcal.get_calendar("NSE")
    today = date.today()
    schedule = nse.schedule(
        start_date=(today - timedelta(days=10)).isoformat(),
        end_date=(today - timedelta(days=1)).isoformat(),
    )
    if schedule.empty:
        raise RuntimeError("No NSE trading days found in the last 10 calendar days")
    return schedule.index[-1].date().isoformat()


# Option B: zero-dependency fallback with hardcoded NSE holidays
# Update NSE_HOLIDAYS each year from: https://www.nseindia.com/resources/exchange-communication-holidays
NSE_HOLIDAYS = {
    # 2025
    date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
    date(2025, 4, 14), date(2025, 4, 18), date(2025, 8, 15),
    date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 24),
    date(2025, 11, 5), date(2025, 12, 25),
    # 2026
    date(2026, 1, 26), date(2026, 3, 25), date(2026, 4, 2),
    date(2026, 4, 10), date(2026, 4, 14), date(2026, 5, 1),
    date(2026, 8, 15), date(2026, 10, 2), date(2026, 10, 20),
    date(2026, 11, 19), date(2026, 12, 25),
}


def _get_last_nse_day_fallback() -> str:
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5 or d in NSE_HOLIDAYS:
        d -= timedelta(days=1)
    return d.isoformat()


def get_last_nse_trading_day() -> str:
    """Return the most recent completed NSE session date as YYYY-MM-DD."""
    try:
        return _get_last_nse_day_mcal()
    except ImportError:
        return _get_last_nse_day_fallback()


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_money(text: str, label: str):
    """Extract a numeric value following **Label**: from markdown text."""
    m = re.search(rf"\*\*{re.escape(label)}\*\*[:\s]*([\d,]+\.?\d*)", text)
    return float(m.group(1).replace(",", "")) if m else None


def format_inr(val) -> str:
    return f"₹{val:,.2f}" if val is not None else "N/A"


def build_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"]    = LLM_PROVIDER
    config["deep_think_llm"]  = DEEP_THINK_LLM
    config["quick_think_llm"] = QUICK_THINK_LLM
    config["backend_url"]     = BACKEND_URL
    config["max_debate_rounds"]       = 1   # keep fast; raise to 2–3 for more debate depth
    config["max_risk_discuss_rounds"] = 1
    config["data_vendors"] = {
        "core_stock_apis":      "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data":     "yfinance",
        "news_data":            "yfinance",
    }
    return config


def run_ticker(ta: TradingAgentsGraph, ticker: str, trade_date: str) -> None:
    print(f"\n  Analysing {ticker} …", flush=True)
    try:
        final_state, signal = ta.propagate(ticker, trade_date)

        trader_text = final_state.get("trader_investment_plan", "")
        pm_text     = final_state.get("final_trade_decision", "")

        entry_price  = parse_money(trader_text, "Entry Price")
        stop_loss    = parse_money(trader_text, "Stop Loss")
        price_target = parse_money(pm_text, "Price Target")

        # Extract first sentence of executive summary for a one-liner thesis
        thesis_block = ""
        m = re.search(r"\*\*Executive Summary\*\*[:\s]*(.+?)(?:\n|$)", pm_text, re.IGNORECASE)
        if m:
            thesis_block = m.group(1).strip().split(".")[0] + "."

        action = signal.upper()

        print(f"\n  {'─'*58}")
        print(f"  {ticker:<22}  [{action}]   {trade_date}")
        print(f"  {'─'*58}")
        print(f"  Entry Price  : {format_inr(entry_price)}")
        print(f"  Stop Loss    : {format_inr(stop_loss)}")
        print(f"  Price Target : {format_inr(price_target)}")
        print(f"  Rating       : {signal}")
        if thesis_block:
            print(f"  Thesis       : {thesis_block}")

    except Exception as exc:
        print(f"\n  FAILED {ticker}: {exc}")
        logging.exception("Error processing %s", ticker)


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    tickers    = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS
    trade_date = get_last_nse_trading_day()

    print(f"\n{'='*62}")
    print(f"  NSE Pre-Market Analysis  |  Reference date: {trade_date}")
    print(f"  LLM: {LLM_PROVIDER} / {DEEP_THINK_LLM}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"{'='*62}")

    config = build_config()
    ta = TradingAgentsGraph(debug=False, config=config)

    for ticker in tickers:
        run_ticker(ta, ticker, trade_date)

    print(f"\n{'='*62}")
    print("  All recommendations are informational only.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
