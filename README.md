# D10 Strategy

A systematic BTC allocation strategy based on an ensemble of 59 on-chain, macro, and technical signals.

## Strategy Overview

| Metric | Value |
|---|---|
| Signals | 59 (on-chain, macro, technical) |
| Ensemble | 2Y + 3Y rolling windows |
| Pair Selection | T3 holdout (70/30 train/validate) |
| Rebalance | Quarterly |
| Allocation | Daily (0–100% BTC, neutral = 50%) |
| Sharpe Ratio | 1.28 |
| Ann. Return | +57.1% (Strategy) |
| Max Drawdown | -51.1% |
| Avg Exposure | 74.2% |

## How It Works

**Quarterly** — Signal pair selection (learning step)
- Every quarter, the best signal pairs are selected using a 70/30 train/validate holdout split
- Prevents overfitting by validating pair predictive power on unseen data

**Daily** — Allocation update (scoring step)
- Selected signal pairs are scored on today's fresh data
- Ensemble combo score determines BTC exposure:
  - Combo < 0 → bullish → increase BTC exposure (above 50%)
  - Combo > 0 → bearish → decrease BTC exposure (below 50%)
  - Step size: 10% per threshold level, max 5 steps (0–100%)

## Repository Structure

```
├── d10_production_v2_audited.py   # Core strategy (backtest + full run)
├── btc_combined_backtest_lib.py   # Signal library (data fetching, scoring)
├── d10_daily_score.py             # Daily scorer — saves outputs to data/
├── dashboard.py                   # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── combo_history.csv          # Daily combo score + exposure + BTC price
│   ├── signals_history.csv        # Historical per-signal values
│   ├── signals_today.json         # Today's signal breakdown
│   └── metrics.json               # Backtest performance metrics
└── .github/workflows/
    └── daily_update.yml           # GitHub Action — runs daily at 08:00 UTC
```

## Running Locally

Install dependencies:
```bash
pip3 install -r requirements.txt
```

Run daily scorer (fast — uses cache, only recomputes current quarter):
```bash
python3 d10_daily_score.py
```

Force full recompute:
```bash
FORCE_FULL=1 python3 d10_daily_score.py
```

Run the dashboard:
```bash
streamlit run dashboard.py
```

Run full backtest:
```bash
python3 d10_production_v2_audited.py
```

## Required API Keys

Set as environment variables or GitHub Actions secrets:

| Variable | Source |
|---|---|
| `CRYPTOQUANT_KEY` | [cryptoquant.com](https://cryptoquant.com) |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |

## Automation

A GitHub Action (`.github/workflows/daily_update.yml`) runs every day at 08:00 UTC:
1. Fetches latest market + on-chain data
2. Rescores the ensemble using today's signal values
3. Commits updated `data/` files to the repo
4. Streamlit Cloud auto-redeploys on each push