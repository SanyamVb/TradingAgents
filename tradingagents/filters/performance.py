"""
Historical filter performance tracking.

Tracks which filters triggered, signal outcomes, and calibration data.
Persists to JSON for analysis across runs.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .models import FilterDecision, FilterResult, FilterStage

logger = logging.getLogger(__name__)


@dataclass
class TickerRecord:
    """Win/loss tracking per ticker."""
    ticker: str
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    total_pnl_pct: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_trades if self.n_trades > 0 else 0.5

    @property
    def avg_pnl_pct(self) -> float:
        return self.total_pnl_pct / self.n_trades if self.n_trades > 0 else 0.0


@dataclass
class FilterRunRecord:
    """Record of a single filter pipeline run."""
    ticker: str
    trade_date: str
    original_signal: str
    final_signal: str
    decision: str
    first_failure: Optional[str]
    failure_reason: str
    position_size_multiplier: float
    stages_passed: int
    stages_failed: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    # Outcome filled in later after trade resolves
    outcome: Optional[str] = None    # "win" / "loss" / "neutral"
    pnl_pct: Optional[float] = None


@dataclass
class FilterStageStats:
    """Aggregate stats for a single filter stage."""
    stage: str
    total_runs: int = 0
    total_triggered: int = 0
    outcomes_when_triggered: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def trigger_rate(self) -> float:
        return self.total_triggered / self.total_runs if self.total_runs > 0 else 0.0

    def win_rate_when_triggered(self) -> float:
        total = sum(self.outcomes_when_triggered.values())
        wins = self.outcomes_when_triggered.get("win", 0)
        return wins / total if total > 0 else 0.0


class FilterPerformanceTracker:
    """
    Tracks filter performance across runs.

    Usage:
        tracker = FilterPerformanceTracker("~/.tradingagents/filter_performance.json")
        tracker.record(filter_result)
        tracker.record_outcome("RELIANCE", "2025-03-01", "win", pnl_pct=2.5)
        report = tracker.generate_report()
    """

    def __init__(self, db_path: str = "~/.tradingagents/filter_performance.json"):
        self.db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._data: Dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load filter performance DB: {e}")
        return {
            "runs": [],
            "ticker_records": {},
            "stage_stats": {},
        }

    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def record(self, result: FilterResult):
        """Record a filter pipeline result."""
        run = FilterRunRecord(
            ticker=result.ticker,
            trade_date=result.trade_date,
            original_signal=result.original_signal,
            final_signal=result.final_signal,
            decision=result.decision.value,
            first_failure=result.first_failure.value if result.first_failure else None,
            failure_reason=result.failure_reason,
            position_size_multiplier=result.position_size_multiplier,
            stages_passed=result.stages_passed,
            stages_failed=result.stages_failed,
        )
        self._data["runs"].append(asdict(run))

        # Update stage stats
        for sr in result.stage_results:
            stage_key = sr.stage.value
            if stage_key not in self._data["stage_stats"]:
                self._data["stage_stats"][stage_key] = {
                    "total_runs": 0,
                    "total_triggered": 0,
                    "outcomes_when_triggered": {},
                }
            stats = self._data["stage_stats"][stage_key]
            stats["total_runs"] += 1
            if sr.decision != FilterDecision.PASS:
                stats["total_triggered"] += 1

        self._save()

    def record_outcome(
        self,
        ticker: str,
        trade_date: str,
        outcome: str,
        pnl_pct: Optional[float] = None,
    ):
        """
        Record trade outcome after it resolves.

        Args:
            ticker: The ticker symbol.
            trade_date: The trade date (ISO string).
            outcome: "win", "loss", or "neutral".
            pnl_pct: Percentage P&L.
        """
        # Update run records
        for run in self._data["runs"]:
            if run["ticker"] == ticker and run["trade_date"] == trade_date:
                run["outcome"] = outcome
                if pnl_pct is not None:
                    run["pnl_pct"] = pnl_pct
                break

        # Update ticker aggregate
        if ticker not in self._data["ticker_records"]:
            self._data["ticker_records"][ticker] = {
                "ticker": ticker,
                "n_trades": 0,
                "n_wins": 0,
                "n_losses": 0,
                "total_pnl_pct": 0.0,
            }
        rec = self._data["ticker_records"][ticker]
        rec["n_trades"] += 1
        if outcome == "win":
            rec["n_wins"] += 1
        elif outcome == "loss":
            rec["n_losses"] += 1
        if pnl_pct is not None:
            rec["total_pnl_pct"] += pnl_pct

        # Update stage stats with outcome
        for run in self._data["runs"]:
            if run["ticker"] == ticker and run["trade_date"] == trade_date:
                if run.get("first_failure"):
                    stage_stats = self._data["stage_stats"].get(run["first_failure"], {})
                    outcomes = stage_stats.get("outcomes_when_triggered", {})
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1
                break

        self._save()

    def get_win_rates(self) -> Dict[str, float]:
        """Return {ticker: win_rate} for all tracked tickers."""
        return {
            ticker: (rec["n_wins"] / rec["n_trades"] if rec["n_trades"] > 0 else 0.5)
            for ticker, rec in self._data["ticker_records"].items()
        }

    def get_trade_counts(self) -> Dict[str, int]:
        """Return {ticker: n_trades} for all tracked tickers."""
        return {
            ticker: rec["n_trades"]
            for ticker, rec in self._data["ticker_records"].items()
        }

    def generate_report(self) -> str:
        """Generate a markdown report of filter performance."""
        runs = self._data["runs"]
        n_runs = len(runs)
        if n_runs == 0:
            return "No filter runs recorded yet."

        n_passed = sum(1 for r in runs if r["decision"] == "PASS")
        n_filtered = n_runs - n_passed
        filter_rate = n_filtered / n_runs

        lines = [
            "# Filter Performance Report",
            "",
            f"Total runs      : {n_runs}",
            f"Passed          : {n_passed} ({n_passed/n_runs:.0%})",
            f"Filtered out    : {n_filtered} ({filter_rate:.0%})",
            "",
            "## Filter Trigger Rates",
            "",
        ]

        for stage, stats in self._data["stage_stats"].items():
            t_runs = stats["total_runs"]
            triggered = stats["total_triggered"]
            rate = triggered / t_runs if t_runs > 0 else 0.0
            lines.append(f"  {stage:<30} triggered {triggered:>4}/{t_runs:<4} = {rate:.0%}")

        # Stage effectiveness: win rate delta when triggered
        lines += ["", "## Ticker Performance", ""]
        for ticker, rec in sorted(
            self._data["ticker_records"].items(),
            key=lambda x: -x[1].get("n_trades", 0),
        )[:20]:
            n = rec["n_trades"]
            wr = rec["n_wins"] / n if n > 0 else 0
            avg_pnl = rec["total_pnl_pct"] / n if n > 0 else 0
            lines.append(
                f"  {ticker:<15} trades={n:>4}  win_rate={wr:.0%}  avg_pnl={avg_pnl:+.2f}%"
            )

        return "\n".join(lines)
