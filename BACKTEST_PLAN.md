# NSE Trading Pipeline - 18-Day Backtest Plan

## Objective
Compare the NEW pipeline (fixed stockstats errors) against OLD pipeline data for the past 18 trading days (Apr 29 - May 25, 2026).

## Data Available
- **Historical Runs**: 18+ successful GitHub Actions runs from Apr 29 to May 25
- **Date Range**: Apr 29, 2026 → May 25, 2026 (18 trading days)
- **Run IDs**: 26387618082 (latest) down to 25093658089 (oldest)

---

## Phase 1: Historical Data Collection (2-3 hours)

### 1.1 Extract OLD Pipeline Signals
**Script**: `scripts/extract_historical_signals.py`

**Tasks**:
- Fetch logs from all 18 successful runs via `gh run view <run_id> --log`
- Parse signals using regex: `(\\w+\\.NS)\\s+\\[(UNDERWEIGHT|HOLD|SKIP)\\].*Entry Price\\s*:\\s*₹?([\\d,]+\\.?\\d*)`
- Extract for each signal:
  - Ticker (e.g., RELIANCE.NS)
  - Signal type (UNDERWEIGHT/HOLD/SKIP)
  - Entry price
  - Stop loss
  - Price target
  - Date
  - Run ID
- Store in: `~/.hermes/trading_tracking/historical_old_signals.jsonl`

**Data Structure**:
```json
{
  "date": "2026-05-01",
  "run_id": "25204256349",
  "ticker": "RELIANCE.NS",
  "signal": "UNDERWEIGHT",
  "entry_price": 2850.00,
  "stop_loss": 2900.00,
  "target": 2700.00,
  "thesis": "...",
  "backend_version": "old"
}
```

### 1.2 Download Daily Actual Prices
**Script**: `scripts/download_actual_prices.py`

**Tasks**:
- For each of 18 days, download OHLCV data for all 10 tickers
- Use yfinance with 1-day lag to get actual close prices
- Store in: `~/.hermes/trading_tracking/actual_prices.csv`

**Columns**: Date, Ticker, Open, High, Low, Close, Volume

---

## Phase 2: NEW Pipeline Backtest (4-6 hours)

### 2.1 Re-run Analysis for Each Historical Date
**Script**: `scripts/backtest_new_pipeline.py`

**Method**:
- For each date from Apr 29 to May 24:
  - Set `curr_date = <historical_date>`
  - Run the NEW pipeline code (with fixed stockstats)
  - Generate signals for that date
  - Store in: `~/.hermes/trading_tracking/historical_new_signals.jsonl`

**Key Point**: Use `curr_date` parameter to prevent look-ahead bias

**Command**:
```bash
for date in $(seq -f "2026-04-%02g" 29 30) $(seq -f "2026-05-%02g" 1 24); do
  python scripts/india_premarket.py --date $date --output backtest_output_$date.json
done
```

### 2.2 Parse NEW Pipeline Signals
- Extract same fields as OLD signals
- Tag with `"backend_version": "new"`
- Store alongside old signals for comparison

---

## Phase 3: Performance Calculation (1-2 hours)

### 3.1 Define Metrics

#### Signal-Level Metrics:
1. **Win Rate**: % of signals that hit target before stop loss
2. **Avg Return**: Mean % return when following signals
3. **Max Drawdown**: Largest peak-to-trough loss
4. **Sharpe Ratio**: Risk-adjusted returns (if >1.0, good)
5. **Hit Rate by Signal Type**:
   - UNDERWEIGHT win rate
   - HOLD accuracy (did price stay stable?)
   - SKIP correctness (did price actually drop?)

#### Ticker-Level Metrics:
- Best/worst performing tickers
- Signal count per ticker
- Accuracy variance across tickers

#### Time-Series Metrics:
- Cumulative P&L over 18 days
- Day-by-day comparison (old vs new)
- Volatility (std dev of daily returns)

### 3.2 Backtest Logic
**Script**: `scripts/calculate_backtest_metrics.py`

For each signal:
1. **Entry**: Use `entry_price` from signal
2. **Exit**: 
   - If price hits `target` first → WIN (capture gain)
   - If price hits `stop_loss` first → LOSS (capture loss)
   - If neither within 5 days → EXIT at day 5 close price
3. **Return %**: `(exit_price - entry_price) / entry_price * 100`
4. **Outcome**: WIN/LOSS/TIMEOUT

**Position Sizing**: Equal weight (₹100k per signal) for fair comparison

---

## Phase 4: Comparison Analysis (1 hour)

### 4.1 Head-to-Head Comparison
**Script**: `scripts/compare_pipelines.py`

**Outputs**:
1. **Summary Table**:
```
Metric                  OLD Pipeline    NEW Pipeline    Δ Change
─────────────────────────────────────────────────────────────────
Total Signals           180             180             0
Win Rate               52.3%            58.7%          +6.4%
Avg Return per Trade    1.2%            2.1%           +0.9%
Max Drawdown           -12.5%           -9.8%          +2.7%
Sharpe Ratio            0.85            1.12           +0.27
Cumulative P&L         +₹21,400        +₹37,800       +₹16,400
```

2. **Signal Distribution**:
```
Signal Type      OLD Count   NEW Count   Δ
─────────────────────────────────────────
UNDERWEIGHT      45 (25%)    48 (26.7%)  +3
HOLD             108 (60%)   105 (58.3%) -3
SKIP             27 (15%)    27 (15%)    0
```

3. **Per-Ticker Performance**:
```
Ticker           OLD WR    NEW WR    Δ WR    OLD Ret   NEW Ret   Δ Ret
────────────────────────────────────────────────────────────────────────
RELIANCE.NS      55%       62%       +7%     +1.5%     +2.3%     +0.8%
TCS.NS           48%       54%       +6%     +0.8%     +1.7%     +0.9%
...
```

### 4.2 Statistical Tests
- **T-test**: Are returns significantly different?
- **Chi-square**: Is signal distribution meaningfully changed?
- **Confidence Intervals**: 95% CI for win rate difference

### 4.3 Qualitative Analysis
- Did NEW pipeline avoid false positives the OLD one made?
- Are there dates where NEW performed much better/worse?
- Any systematic bias in signal types?

---

## Phase 5: Visualization (1 hour)

### 5.1 Charts to Generate
**Script**: `scripts/visualize_backtest.py`

1. **Cumulative P&L Chart** (line chart):
   - X-axis: Date (Apr 29 → May 24)
   - Y-axis: Cumulative return (₹)
   - Two lines: OLD (red) vs NEW (green)

2. **Win Rate by Ticker** (bar chart):
   - X-axis: Tickers
   - Y-axis: Win rate %
   - Grouped bars: OLD vs NEW

3. **Signal Distribution Pie Charts**:
   - OLD: UNDERWEIGHT/HOLD/SKIP split
   - NEW: UNDERWEIGHT/HOLD/SKIP split

4. **Return Distribution Histogram**:
   - X-axis: Return % buckets (-5% to +10%)
   - Y-axis: Frequency
   - Overlaid: OLD (blue) vs NEW (orange)

5. **Drawdown Chart**:
   - X-axis: Date
   - Y-axis: Drawdown %
   - Two lines showing peak-to-trough declines

### 5.2 Output Formats
- Save as PNG: `~/.hermes/trading_tracking/charts/`
- Generate HTML report: `~/.hermes/trading_tracking/backtest_report.html`

---

## Phase 6: Report Generation (30 min)

### 6.1 Executive Summary
**File**: `~/.hermes/trading_tracking/BACKTEST_REPORT.md`

**Sections**:
1. **Overview**
   - Test period: Apr 29 - May 24, 2026 (18 trading days)
   - Total signals compared: 180 OLD vs 180 NEW

2. **Key Findings**
   - Win rate improvement: X.X%
   - Return improvement: X.X%
   - Risk reduction (drawdown): X.X%
   - Sharpe ratio change: +X.XX

3. **Conclusion**
   - Is NEW pipeline better? (Yes/No + confidence level)
   - Recommended action (Deploy NEW / Revert to OLD / Need more data)

4. **Next Steps**
   - Continue 30-day tracking with NEW pipeline
   - Monitor for regression
   - Set up alerts for WR < 50%

### 6.2 Detailed Report
Include:
- Full data tables
- All charts embedded
- Statistical test results
- Per-ticker breakdown
- Notable wins/losses analysis

---

## Implementation Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| 1. Historical Collection | 2-3h | `historical_old_signals.jsonl`, `actual_prices.csv` |
| 2. NEW Pipeline Backtest | 4-6h | `historical_new_signals.jsonl` |
| 3. Performance Calc | 1-2h | `backtest_metrics.json` |
| 4. Comparison | 1h | `pipeline_comparison.json` |
| 5. Visualization | 1h | Charts + `backtest_report.html` |
| 6. Report | 30m | `BACKTEST_REPORT.md` |
| **Total** | **9-13h** | Complete backtest analysis |

---

## Scripts to Create

1. **extract_historical_signals.py** - Parse old run logs
2. **download_actual_prices.py** - Get historical OHLCV data
3. **backtest_new_pipeline.py** - Re-run new pipeline for historical dates
4. **calculate_backtest_metrics.py** - Compute win rates, returns, Sharpe
5. **compare_pipelines.py** - Side-by-side OLD vs NEW analysis
6. **visualize_backtest.py** - Generate all charts
7. **generate_report.py** - Markdown + HTML report builder

---

## Success Criteria

The NEW pipeline is considered **successful** if:
1. ✅ Win rate > OLD by at least 3%
2. ✅ Average return > OLD by at least 0.5%
3. ✅ Sharpe ratio > 1.0
4. ✅ Max drawdown < OLD
5. ✅ No systematic bias toward false positives

If all 5 criteria met → **Deploy NEW pipeline permanently**
If 3-4 criteria met → **Promising, extend tracking to 30 days**
If <3 criteria met → **Investigate issues, consider refinements**

---

## Risk Mitigation

1. **Data Quality**: Validate all extracted signals manually for first 3 days
2. **Look-Ahead Bias**: Ensure `curr_date` filtering is strict in backtests
3. **Overfitting**: Test on out-of-sample period (May 26-30) after backtest
4. **Market Regime**: Check if performance difference is due to market conditions vs pipeline

---

## Files & Directories

```
~/.hermes/trading_tracking/
├── historical_old_signals.jsonl    # OLD pipeline signals (18 days)
├── historical_new_signals.jsonl    # NEW pipeline backtest signals
├── actual_prices.csv               # Daily OHLCV for all tickers
├── backtest_metrics.json           # Computed performance metrics
├── pipeline_comparison.json        # OLD vs NEW comparison
├── BACKTEST_REPORT.md             # Executive summary
├── backtest_report.html           # Interactive HTML report
└── charts/
    ├── cumulative_pnl.png
    ├── win_rate_by_ticker.png
    ├── signal_distribution.png
    ├── return_histogram.png
    └── drawdown_chart.png
```

---

## Next Actions

1. Create all 7 Python scripts listed above
2. Run Phase 1 to collect historical data
3. Execute Phase 2 backtest (longest step - can run overnight)
4. Complete Phases 3-6 for analysis and reporting
5. Present findings with recommendation: Deploy NEW / Need more data / Revert
