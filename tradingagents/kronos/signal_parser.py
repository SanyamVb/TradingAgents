"""
KronosSignalParser — converts raw Kronos prediction output into
a structured trading signal dictionary suitable for TradingAgents.

Also provides a formatted text report for use in the LangGraph message chain.
"""

from __future__ import annotations

from typing import Any


class KronosSignalParser:
    """Parse and format Kronos prediction results into trading signals.

    Input: dict returned by KronosWrapper.predict()
    Output: structured signal dict + human-readable report text
    """

    # Signal confidence thresholds
    HIGH_CONFIDENCE = 0.65
    LOW_CONFIDENCE = 0.30

    @classmethod
    def parse(cls, prediction: dict) -> dict:
        """Parse raw Kronos prediction into structured signal.

        Returns:
            {
                ticker, signal, confidence, strength,
                pct_change_5d, price_targets: {day_1..5},
                uncertainty: {avg_std, coeffvar},
                recommendation: str,
                rationale: str,
            }
        """
        ticker = prediction["ticker"]
        signal = prediction["signal"]
        confidence = prediction["confidence"]
        pct_change = prediction["pct_change_5d"]
        last_close = prediction["last_close"]
        pred_close = prediction["pred_close_5d"]
        pred_std = prediction["pred_std_5d"]

        # Price targets keyed by day
        price_targets = {f"day_{i+1}": pred_close[i] for i in range(len(pred_close))}
        # Entry / stop-loss / target suggestions
        if signal == "BUY":
            entry_price = round(pred_close[0] * 1.001, 2)          # slight premium
            stop_loss = round(last_close * 0.97, 2)                  # 3% stop
            target = round(pred_close[-1] * 1.01, 2)                 # day 5 target
        elif signal == "SELL":
            entry_price = round(pred_close[0] * 0.999, 2)
            stop_loss = round(last_close * 1.03, 2)
            target = round(pred_close[-1] * 0.99, 2)
        else:
            entry_price = stop_loss = target = None

        # Strength label
        if abs(pct_change) >= 4:
            strength = "STRONG"
        elif abs(pct_change) >= 2:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        # Uncertainty metrics
        avg_std = round(sum(pred_std) / len(pred_std), 2) if pred_std else 0.0
        coeffvar = round(avg_std / (abs(last_close) + 1e-9) * 100, 2)

        recommendation = cls._build_recommendation(signal, confidence, strength, pct_change, last_close)
        rationale = cls._build_rationale(ticker, signal, pct_change, pred_close, confidence)

        return {
            "ticker": ticker,
            "signal": signal,
            "confidence": confidence,
            "strength": strength,
            "pct_change_5d": pct_change,
            "last_close": last_close,
            "price_targets": price_targets,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target_price": target,
            "uncertainty": {"avg_std": avg_std, "coeff_variation_pct": coeffvar},
            "recommendation": recommendation,
            "rationale": rationale,
        }

    # ------------------------------------------------------------------

    @classmethod
    def format_report(cls, signal: dict) -> str:
        """Format structured signal as a text report for LangGraph messages."""
        sep = "=" * 60
        lines = [
            sep,
            f"KRONOS QUANTITATIVE ANALYSIS — {signal['ticker']}",
            sep,
            f"Signal:      {signal['signal']}  ({signal['strength']})",
            f"Confidence:  {signal['confidence']:.0%}",
            f"5-day move:  {signal['pct_change_5d']:+.2f}%",
            f"Last close:  ₹{signal['last_close']:.2f}",
            "",
            "5-Day Price Targets:",
        ]
        for k, v in signal["price_targets"].items():
            lines.append(f"  {k}: ₹{v:.2f}")

        if signal["entry_price"]:
            lines += [
                "",
                f"Entry:       ₹{signal['entry_price']:.2f}",
                f"Stop-Loss:   ₹{signal['stop_loss']:.2f}",
                f"Target:      ₹{signal['target_price']:.2f}",
            ]

        unc = signal["uncertainty"]
        lines += [
            "",
            f"Uncertainty: ±₹{unc['avg_std']:.2f} avg  ({unc['coeff_variation_pct']:.1f}% CV)",
            "",
            "Recommendation:",
            f"  {signal['recommendation']}",
            "",
            "Rationale:",
            f"  {signal['rationale']}",
            sep,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recommendation(signal, confidence, strength, pct_change, last_close) -> str:
        if signal == "SKIP":
            return "SKIP — high forecast uncertainty. Wait for clearer price action."
        if signal == "BUY":
            return (
                f"Kronos forecasts a {pct_change:+.2f}% gain over 5 days (confidence: {confidence:.0%}). "
                f"Consider entering a LONG position with disciplined stop-loss management."
            )
        if signal == "SELL":
            return (
                f"Kronos forecasts a {pct_change:+.2f}% decline over 5 days (confidence: {confidence:.0%}). "
                f"Consider reducing/exiting LONG or entering SHORT with risk controls."
            )
        return (
            f"Kronos forecasts a near-flat {pct_change:+.2f}% move (confidence: {confidence:.0%}). "
            f"HOLD existing positions; no new entry signal."
        )

    @staticmethod
    def _build_rationale(ticker, signal, pct_change, pred_close, confidence) -> str:
        direction = "upward" if pct_change > 0 else "downward"
        price_path = " → ".join([f"₹{v:.0f}" for v in pred_close])
        return (
            f"Kronos-base (102M param financial time-series model) predicts a {direction} "
            f"trajectory for {ticker} over the next 5 trading sessions: {price_path}. "
            f"Model confidence {confidence:.0%} based on {3} Monte Carlo samples."
        )
