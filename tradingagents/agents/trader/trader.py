"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools
import logging

import pandas as pd

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)

logger = logging.getLogger(__name__)


def _fetch_latest_close(ticker: str, trade_date: str) -> str:
    """Return a human-readable latest-close string anchored to trade_date.

    Fetches the most recent close on or before trade_date so the Trader LLM
    always has an accurate numeric price to anchor entry / stop-loss levels.
    Falls back gracefully if data is unavailable.
    """
    try:
        import yfinance as yf
        trade_dt = pd.to_datetime(trade_date)
        # Fetch a 5-day window ending the day after trade_date to ensure we
        # capture the trade_date session even with timezone offsets.
        end_dt = (trade_dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        start_dt = (trade_dt - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        hist = yf.Ticker(ticker).history(start=start_dt, end=end_dt, auto_adjust=True)
        if hist.empty:
            return ""
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist = hist[hist.index <= trade_dt]
        if hist.empty:
            return ""
        latest_close = hist["Close"].iloc[-1]
        latest_date = hist.index[-1].strftime("%Y-%m-%d")
        return f"Latest verified close for {ticker} as of {latest_date}: {latest_close:.2f}"
    except Exception as exc:
        logger.warning("Could not fetch latest close for %s: %s", ticker, exc)
        return ""


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        trade_date = state.get("trade_date", "")
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]

        # Fetch a fresh, verified close price so the LLM doesn't anchor to
        # stale prices from analyst prose written against older data.
        latest_close_line = _fetch_latest_close(company_name, trade_date)
        price_anchor = (
            f"\n\n**Price anchor (use this for entry/stop-loss levels)**: "
            f"{latest_close_line}"
            if latest_close_line
            else ""
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan. "
                    "Always set Entry Price and Stop Loss relative to the provided price anchor — "
                    "never use prices from older analyst reports if a price anchor is supplied.\n\n"
                    "CRITICAL rules for entry_price, stop_loss, and price_target:\n"
                    "- For a SELL action: you MUST provide entry_price and stop_loss. entry_price >= current price, stop_loss > entry_price, price_target < entry_price.\n"
                    "- For a BUY action: you MUST provide entry_price and stop_loss. entry_price <= current price, stop_loss < entry_price, price_target > entry_price.\n"
                    "- For HOLD: entry_price and stop_loss may be omitted.\n"
                    "- NEVER leave entry_price or stop_loss null on a BUY or SELL action.\n"
                    "Violating these rules (e.g. stop_loss below entry on a sell) is always wrong."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\nProposed Investment Plan: {investment_plan}"
                    f"{price_anchor}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
