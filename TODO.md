# TradingAgents — TODO

## High Priority

- [ ] **Plug in real news API for better sentiment analysis**
  - yfinance news is shallow and often stale
  - Candidates: NewsAPI, Finnhub, Alpha Vantage News, Benzinga
  - NSE-specific: Economic Times, Moneycontrol, BSE announcements feed
  - Wire into `news_data` vendor slot in `data_vendors` config

## Signal Quality

- [ ] **Re-evaluate signals after 2 weeks (May 12)** — run predicted vs real comparison for the Apr 28 batch
- [ ] **Track signal accuracy over time** — build a simple CSV log of signals + outcomes for backtesting

## Cost & Performance

- [ ] **Run tickers in parallel** — currently sequential (~2-3 min/ticker); parallelising would cut 10-ticker run from ~25 min to ~5 min
- [ ] **Add Haiku for quick/cheap agents** — use `claude-haiku-4-5` for news/social analysts, reserve Sonnet for debate + PM

## Infrastructure

- [ ] **Top up Anthropic API credits** — ITC.NS and LT.NS failed on Apr 28 run due to quota exceeded (402)
- [ ] **Add GitHub Actions repo secrets** — `TRADINGAGENTS_LLM_PROVIDER`, `TRADINGAGENTS_DEEP_LLM`, `TRADINGAGENTS_QUICK_LLM` as Variables; `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` as Secrets
