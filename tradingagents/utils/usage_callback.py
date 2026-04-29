"""Token usage tracking callback for LangChain/LangGraph runs.

Tracks input tokens, output tokens, and estimated cost per ticker and in
aggregate across a full run. Plug into TradingAgentsGraph via the callbacks
kwarg or pass directly to any LangChain LLM.

Usage:
    tracker = UsageTracker()
    ta = TradingAgentsGraph(config=config, callbacks=[tracker.handler])
    # ... run tickers ...
    tracker.print_summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Pricing per million tokens (USD) — update when Anthropic/OpenAI change rates
_PRICING: Dict[str, Dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-5-20250929": {"input": 3.0,  "output": 15.0},
    "claude-opus-4-5-20251101":   {"input": 15.0, "output": 75.0},
    "claude-haiku-4-5-20250929":  {"input": 0.8,  "output": 4.0},
    "claude-haiku-3-5-20241022":  {"input": 0.8,  "output": 4.0},
    "claude-sonnet-3-7-20250219": {"input": 3.0,  "output": 15.0},
    # OpenAI
    "gpt-4o":                     {"input": 2.5,  "output": 10.0},
    "gpt-4o-mini":                {"input": 0.15, "output": 0.6},
    "gpt-4.1":                    {"input": 2.0,  "output": 8.0},
}
def _get_pricing(model: str) -> Dict[str, float]:
    """Look up pricing, falling back to substring match for aliased model names."""
    if model in _PRICING:
        return _PRICING[model]
    # Partial match — e.g. LangChain may return 'claude-haiku-4-5-20251001' or just 'claude-haiku-4-5'
    for key in _PRICING:
        if key in model or model in key:
            return _PRICING[key]
    # Infer tier from model name
    name = model.lower()
    if "haiku" in name:
        return {"input": 0.8, "output": 4.0}
    if "sonnet" in name:
        return {"input": 3.0, "output": 15.0}
    if "opus" in name:
        return {"input": 15.0, "output": 75.0}
    return _DEFAULT_PRICING


@dataclass
class CallRecord:
    model: str
    input_tokens: int
    output_tokens: int

    def cost_usd(self) -> float:
        p = _get_pricing(self.model)
        return (self.input_tokens * p["input"] + self.output_tokens * p["output"]) / 1_000_000


@dataclass
class TickerUsage:
    ticker: str
    calls: List[CallRecord] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def llm_calls(self) -> int:
        return len(self.calls)

    def cost_usd(self) -> float:
        return sum(c.cost_usd() for c in self.calls)

    def model_breakdown(self) -> str:
        counts: Dict[str, dict] = {}
        for c in self.calls:
            m = c.model or "unknown"
            if m not in counts:
                counts[m] = {"calls": 0, "in": 0, "out": 0}
            counts[m]["calls"] += 1
            counts[m]["in"] += c.input_tokens
            counts[m]["out"] += c.output_tokens
        parts = []
        for m, v in counts.items():
            p = _get_pricing(m)
            cost = (v["in"] * p["input"] + v["out"] * p["output"]) / 1_000_000
            parts.append(f"{m}: {v['calls']} calls, {v['in']:,}in/{v['out']:,}out, ${cost:.4f}")
        return " | ".join(parts)


class UsageCallbackHandler(BaseCallbackHandler):
    """LangChain callback that accumulates token usage per LLM response."""

    def __init__(self, tracker: "UsageTracker"):
        super().__init__()
        self._tracker = tracker

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                usage = None
                # LangChain stores usage in generation_info or llm_output
                if hasattr(gen, "generation_info") and gen.generation_info:
                    usage = gen.generation_info.get("usage")
                if usage is None and response.llm_output:
                    usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")

                if usage:
                    it = (
                        usage.get("input_tokens")
                        or usage.get("prompt_tokens")
                        or 0
                    )
                    ot = (
                        usage.get("output_tokens")
                        or usage.get("completion_tokens")
                        or 0
                    )
                    model = (response.llm_output or {}).get("model_name", "") or (response.llm_output or {}).get("model", "")
                    self._tracker._record(int(it), int(ot), model)


class UsageTracker:
    """Aggregates token usage across tickers and prints a formatted summary."""

    def __init__(self):
        self._tickers: List[TickerUsage] = []
        self._current: Optional[TickerUsage] = None
        self.handler = UsageCallbackHandler(self)

    def start_ticker(self, ticker: str) -> None:
        self._current = TickerUsage(ticker=ticker)
        self._tickers.append(self._current)

    def _record(self, input_tokens: int, output_tokens: int, model: str) -> None:
        if self._current is None:
            return
        self._current.calls.append(CallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ))

    def print_ticker_usage(self, ticker: str) -> None:
        t = next((x for x in self._tickers if x.ticker == ticker), None)
        if not t:
            return
        print(
            f"  {'·'*58}\n"
            f"  Token usage  : {t.input_tokens:,} in / {t.output_tokens:,} out "
            f"({t.total_tokens:,} total, {t.llm_calls} calls)  ~${t.cost_usd():.4f}\n"
            f"  Models       : {t.model_breakdown()}"
        )

    def print_summary(self) -> None:
        if not self._tickers:
            return
        total_in  = sum(t.input_tokens  for t in self._tickers)
        total_out = sum(t.output_tokens for t in self._tickers)
        total_tok = total_in + total_out
        total_cost = sum(t.cost_usd() for t in self._tickers)
        total_calls = sum(t.llm_calls for t in self._tickers)

        print(f"\n  {'─'*58}")
        print(f"  TOKEN USAGE SUMMARY")
        print(f"  {'─'*58}")
        print(f"  {'Ticker':<22}  {'In':>8}  {'Out':>8}  {'Total':>8}  {'Cost':>8}  {'Calls':>5}")
        print(f"  {'─'*58}")
        for t in self._tickers:
            print(
                f"  {t.ticker:<22}  {t.input_tokens:>8,}  {t.output_tokens:>8,}  "
                f"{t.total_tokens:>8,}  ${t.cost_usd():>7.4f}  {t.llm_calls:>5}"
            )
        print(f"  {'─'*58}")
        print(
            f"  {'TOTAL':<22}  {total_in:>8,}  {total_out:>8,}  "
            f"{total_tok:>8,}  ${total_cost:>7.4f}  {total_calls:>5}"
        )
        print(f"  {'─'*58}\n")
