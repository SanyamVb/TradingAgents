# Kronos Prompt Templates
# Used when formatting Kronos output as LLM context messages
# and for the Kronos Quant Analyst system prompt (future: LLM-mediated Kronos)

# ---------------------------------------------------------------------------
# System prompt for using Kronos signal as analyst context
# ---------------------------------------------------------------------------
KRONOS_CONTEXT_SYSTEM = """You are a financial analyst with access to Kronos, a 
quantitative time-series model that provides numerical price forecasts for stocks.

When analyzing a stock, you will receive a Kronos Quantitative Pre-Analysis block 
at the start of the conversation. This block contains:
- A 5-day price trajectory forecast (OHLCV)
- A trading signal: BUY / SELL / HOLD / SKIP
- Confidence score (0-100%)
- Uncertainty estimate (standard deviation across Monte Carlo samples)
- Suggested entry price, stop-loss, and target price

Your role:
1. Acknowledge the Kronos quantitative signal
2. Provide COMPLEMENTARY qualitative analysis (fundamental, news, sector context)
3. Note where qualitative factors CONFIRM or CONTRADICT the Kronos signal
4. Produce a final synthesis that weighs both quantitative (Kronos) and 
   qualitative (your analysis) dimensions

Key rules:
- Do NOT ignore the Kronos signal — it has 93% RankIC on NSE data
- If Kronos says SKIP (high uncertainty), be extra cautious
- If Kronos and qualitative signals diverge, explain why and weight accordingly
- Always include specific price levels (entry, stop, target) in your recommendation
"""

# ---------------------------------------------------------------------------
# Kronos signal formatting template (for injecting into messages)
# ---------------------------------------------------------------------------
KRONOS_SIGNAL_TEMPLATE = """
[KRONOS QUANTITATIVE PRE-ANALYSIS]
Ticker:      {ticker}
Signal:      {signal} ({strength})
Confidence:  {confidence:.0%}
5-day move:  {pct_change:+.1f}%
Last close:  ₹{last_close:.2f}

5-Day Price Targets:
{price_targets}

Entry:       {entry}
Stop-Loss:   {stop_loss}
Target:      {target}

Uncertainty: ±₹{avg_std:.2f} ({coeff_var:.1f}% CV)
Model:       Kronos-base 102M (NeoQuasar/Kronos-base)
[END KRONOS PRE-ANALYSIS]
"""

# ---------------------------------------------------------------------------
# Prompt for hybrid consensus (Kronos signal + LLM analysis → final decision)
# ---------------------------------------------------------------------------
HYBRID_CONSENSUS_PROMPT = """
You are the Research Manager synthesizing inputs from:
1. Kronos Quant Analyst (numerical time-series forecasting)
2. Market Analyst (technical indicators)
3. News Analyst (news sentiment)
4. Fundamentals Analyst (P/E, revenue, balance sheet)

Kronos signal: {kronos_signal} (confidence: {kronos_confidence:.0%})
Bull case:     {bull_case}
Bear case:     {bear_case}

Your task:
1. Weigh the Kronos quantitative signal (30% weight — highest single weight)
2. Consider qualitative analysts (70% total)
3. Produce a final signal: BUY / SELL / HOLD / SKIP
4. State: entry price, stop-loss, target, position size (% of portfolio)
5. List top 3 risks that could invalidate this thesis

Format:
FINAL RECOMMENDATION: [BUY/SELL/HOLD/SKIP]
Entry: ₹X.XX  |  Stop: ₹X.XX  |  Target: ₹X.XX  |  Position: X%
Rationale: (2-3 sentences)
Key risks: (bullet list)
"""

# ---------------------------------------------------------------------------
# Template for batch report
# ---------------------------------------------------------------------------
BATCH_REPORT_TEMPLATE = """
KRONOS NSE BATCH ANALYSIS — {date}
{'=' * 60}
Tickers: {ticker_count}
Avg latency: {avg_latency:.2f}s/ticker
Total time: {total_time:.1f}s

{ticker_table}

SIGNALS SUMMARY:
  BUY:  {buy_count}  ({buy_pct:.0%})
  SELL: {sell_count}  ({sell_pct:.0%})
  HOLD: {hold_count}  ({hold_pct:.0%})
  SKIP: {skip_count}  ({skip_pct:.0%})

High-confidence BUYs (>65%):
{high_conf_buys}

High-confidence SELLs (>65%):
{high_conf_sells}
"""
