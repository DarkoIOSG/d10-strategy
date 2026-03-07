"""
btc_combined_backtest.py  —  v5.0
==================================
Nine strategies combining:
  A) signals_backtesting_ml.py  — 31 on-chain/macro signals, ternary ML walk-forward
  B) btc_regime_backtest.py     — technical signals + walk-forward combinatorics

Strategies:
  S1  Signals Alone           — exact signals_backtesting_ml.py methodology
      31 CryptoQuant on-chain/macro signals → ternary scoring (730-day rolling
      percentile) → monthly Spearman top-10 selection → equal-weight composite
      → sigmoid cash mapping (20–50%) → BTC exposure (≈50–80%)

  S2  Signals + Combinatorics — S1 signal universe + walk-forward pair ranking
      Same 31 signals, ternary scored → 6-month walk-forward combinatorics →
      pair-power composite → BTC exposure

  S3  Full Regime + All Data  — all signals (A+B) + standard combinatorics
      S1 (31) + technical (~30) signals → combined 6-month walk-forward
      combinatorics → BTC exposure

  S4  MVRV-Sensitive Regime   — S3 + MVRV-regime-filtered training window
      Training window restricted to dates in same MVRV Z-score regime

  S9  S6 + Confidence Gate    — S6 signal universe + S8 execution policy
      99-signal state+delta universe → MVRV-weighted pairs → gated hold-previous sizer

  BH  BTC Buy & Hold          — benchmark

Signal convention (consistent across both source files):
  Positive composite → bearish (increase cash / reduce BTC exposure)
  Negative composite → bullish (reduce cash / increase BTC exposure)

Requirements:
  pip install yfinance pandas numpy matplotlib scipy requests
  CRYPTOQUANT_KEY env var  — optional (enables on-chain signals; proxies used without it)
  FRED_API_KEY env var     — optional (enables real HY Spread)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass
from itertools import combinations
from abc import ABC, abstractmethod
from scipy.stats import spearmanr
import os
import pickle
import time
import hashlib
import warnings
import requests

warnings.filterwarnings("ignore")

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

# ============================================================
# 1. GLOBAL CONFIG
# ============================================================

EVAL_START  = "2020-01-01"
TRAIN_START = "2015-01-01"
OUT_DIR     = "/Users/darkobosnjak/Desktop/D10 Strategy App/output"

# --- Combinatorics engine (from btc_regime_backtest.py — unchanged) ---
FWD_HORIZONS    = [3, 7, 14, 30, 90, 180]
HORIZON_WEIGHTS = {3: 0.05, 7: 0.10, 14: 0.15, 30: 0.30, 90: 0.25, 180: 0.15}
MIN_EVENTS      = 5
CLUSTER_GAP     = 30
FEE_PER_UNIT    = 0.0010   # 10 bps round-trip per unit of exposure change

# --- S1 parameters (exact signals_backtesting_ml.py methodology) ---
S1_N_SIGNALS         = 10    # Top-N signals selected per monthly rebalance
S1_SELECTION_WINDOW  = 1095  # 3-year Spearman ranking window (days)
S1_HIST_RETURN_DAYS  = 365   # BTC log-return horizon used in ranking
S1_PERCENTILE_WINDOW = 730   # 2-year rolling window for ternary scoring
S1_PERCENTILE_HI     = 80    # Upper threshold → +1 (bearish)
S1_PERCENTILE_LO     = 20    # Lower threshold → -1 (bullish)
S1_MIN_VALID_FRAC    = 0.70  # Min non-NaN fraction to include a signal
S1_CASH_MIN          = 0.20  # Cash when composite = -1 (fully bullish)
S1_CASH_MID          = 0.35  # Cash when composite =  0 (neutral)
S1_CASH_MAX          = 0.50  # Cash when composite = +1 (fully bearish)
S1_SIGMOID_K         = 3.0   # Sigmoid steepness for cash mapping
# Wide variant: 0–50% cash → ≈52–98% BTC exposure
S1_WIDE_CASH_MIN     = 0.00
S1_WIDE_CASH_MAX     = 0.50

# --- S4 MVRV regime config ---
MVRV_REGIME_BOUNDS = [         # (label, lo_inclusive, hi_exclusive)
    ("COLD",    -np.inf,  0.0),
    ("NEUTRAL",  0.0,     2.0),
    ("HOT",      2.0,     5.0),
    ("EXTREME",  5.0,  np.inf),
]
MIN_REGIME_TRAIN_DAYS = 120   # Fall back to full window if fewer same-regime days

# --- Delta signal configuration (S5/S6) ---
DELTA_LOOKBACKS = [7, 30]
DELTA_SIGNAL_COLS = [
    "MVRV", "SOPR", "SOPR_Adj", "STH_SOPR", "LTH_SOPR",
    "NVT", "NVM", "S2F_Dev", "Exch_Reserve", "Exch_Netflow",
    "Whale_Ratio", "MPI", "Puell", "SOPR_Ratio", "SSR",
    "Dormancy", "Lev_Ratio", "Open_Interest", "Coinbase_Prem", "NRPL",
    "MVRV_Proxy", "Puell_Proxy", "RealVol_30", "RealVol_90",
    "LR_1Y", "LR_2Y_Z", "VIX", "DXY", "SP500_Trend", "Gold_90d",
]

# --- S7/S8 position sizing config ---
TIER_THRESHOLDS = [0.35, 0.60]   # |combo| cutoffs for tiered allocation
MIN_CONFIDENCE  = 0.20            # |combo| gate below which S8 holds previous position
PRESCREEN_N     = 20              # top-N signals to keep before pair-scoring (reduces C(114,2)=6441 → C(20,2)=190)
GATE_SWEEP_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# --- v5.0 Module experiment constants ---
# Module C: γ for conviction weighting within the dead-zone-preserving mapping.
# gamma=2.0: hr=0.60→0.04, hr=0.75→0.25, hr=0.90→0.64, hr=1.0→1.0
# gamma=1.0: hr=0.60→0.20, hr=0.75→0.50, hr=0.90→0.80, hr=1.0→1.0 (linear)
CONVICTION_GAMMA = 2.0
V5_VARIANTS = {           # label → (primary_horizon, use_magnitude, conviction_gamma)
    "V0 Base":     (30,  False, None),
    "V1 +M":       (30,  True,  None),
    "V2 +Cg2":     (30,  False, 2.0),
    "V2b +Cg1":    (30,  False, 1.0),
    "V3 +M+Cg2":   (30,  True,  2.0),
    "V4 +H90":     (90,  False, None),
    "V5 +H90+Cg2": (90,  False, 2.0),
}

# --- Walk-forward retraining interval ---
RETRAIN_MONTHS = 3   # quarterly rebalancing (6M → 13 rebalances; 3M → 25 rebalances)

# --- Naive benchmarks for each strategy group ---
# S1/S2 have mid_exposure=0.65 → benchmark: fixed 65% BTC
# S3-S8 have mid_exposure=0.75 → benchmark: fixed 75% BTC
STRATEGY_BENCHMARK = {
    "Base":                   "BTC 75% Fixed",
    "Base+Delta":             "BTC 75% Fixed",
    "Base+MVRV":              "BTC 75% Fixed",
    "Base+Delta+MVRV":        "BTC 75% Fixed",
    "Base+MVRV+Tiered":                "BTC 75% Fixed",
    "Base+MVRV+Gate":                 "BTC 75% Fixed",
    "Base+MVRV+Tiered+Gate":          "BTC 75% Fixed",
    "Base+Delta+MVRV+Gate":           "BTC 75% Fixed",
    "Base+Delta+MVRV+Gate (PS20)":    "BTC 75% Fixed",
}

# Set to "1" via env var to run the retrain frequency study (slow: 4× S4 runs)
RETRAIN_STUDY = os.environ.get("RETRAIN_STUDY", "0") == "1"

# --- CryptoQuant API ---
CQUANT_BASE = "https://api.cryptoquant.com/v1"

# --- Caching ---
CACHE_DIR           = os.path.expanduser("~/.btc_cache")
CACHE_MAX_AGE_HOURS = 24      # Live API/yfinance data treated as stale after this
FORCE_REFRESH       = os.environ.get("FORCE_REFRESH", "0") == "1"
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# CACHE HELPERS
# ============================================================

def _cache_path(key: str) -> str:
    safe = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def _is_stale(path: str, max_age_hours: float = CACHE_MAX_AGE_HOURS) -> bool:
    if not os.path.exists(path):
        return True
    return (time.time() - os.path.getmtime(path)) / 3600 > max_age_hours


def _load_cache(key: str):
    path = _cache_path(key)
    if FORCE_REFRESH or _is_stale(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(key: str, data) -> None:
    path = _cache_path(key)
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"  Cache write failed: {e}")


# ============================================================
# 2. CRYPTOQUANT API HELPERS
# ============================================================

def _fetch_cquant(endpoint: str, token: str, field: str,
                  start: str = TRAIN_START, exchange: str = None):
    """
    Fetch one daily metric from CryptoQuant API v1.
    Returns pd.Series indexed by date, or None on failure.
    """
    params = {"window": "day", "from": start.replace("-", ""), "limit": 10000}
    if exchange:
        params["exchange"] = exchange
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(f"{CQUANT_BASE}/{endpoint}",
                         params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"  CQ {endpoint}: HTTP {r.status_code}")
            return None
        rows = r.json().get("result", {}).get("data", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        if "date" not in df.columns or field not in df.columns:
            print(f"  CQ {endpoint}: missing col (have: {list(df.columns)[:6]})")
            return None
        df["date"] = pd.to_datetime(df["date"])
        s = pd.to_numeric(df.set_index("date")[field], errors="coerce")
        s.name = field
        return s.sort_index()
    except Exception as e:
        print(f"  CQ {endpoint}: {e}")
        return None


def fetch_cquant_signals(token: str, start: str = TRAIN_START) -> dict:
    """
    Fetch all CryptoQuant-sourced signals.
    Returns {signal_name: pd.Series}.
    """
    # (endpoint, field, exchange_override)
    FETCHES = {
        # Valuation
        "MVRV":          ("btc/market-indicator/mvrv",                       "mvrv",                       None),
        "SOPR":          ("btc/market-indicator/sopr",                       "sopr",                       None),
        "SOPR_Adj":      ("btc/market-indicator/sopr",                       "a_sopr",                     None),
        "STH_SOPR":      ("btc/market-indicator/sopr",                       "sth_sopr",                   None),
        "LTH_SOPR":      ("btc/market-indicator/sopr",                       "lth_sopr",                   None),
        "NVT":           ("btc/network-indicator/nvt",                       "nvt",                        None),
        "NVM":           ("btc/network-indicator/nvm",                       "nvm",                        None),
        "S2F_Dev":       ("btc/network-indicator/stock-to-flow",             "stock_to_flow_reversion",    None),
        # On-chain activity
        "Exch_Reserve":  ("btc/exchange-flows/reserve",                      "reserve",                    "binance"),
        "Exch_Netflow":  ("btc/exchange-flows/netflow",                      "netflow_total",              "binance"),
        "Whale_Ratio":   ("btc/flow-indicator/exchange-whale-ratio",         "exchange_whale_ratio",       "binance"),
        "MPI":           ("btc/flow-indicator/mpi",                          "mpi",                        None),
        "Puell":         ("btc/network-indicator/puell-multiple",            "puell_multiple",             None),
        "SOPR_Ratio":    ("btc/market-indicator/sopr-ratio",                 "sopr_ratio",                 None),
        "Dormancy":      ("btc/network-indicator/dormancy",                  "average_dormancy",           None),
        # Derivatives / sentiment
        "Lev_Ratio":     ("btc/market-indicator/estimated-leverage-ratio",   "estimated_leverage_ratio",   "binance"),
        "SSR":           ("btc/market-indicator/stablecoin-supply-ratio",    "stablecoin_supply_ratio",    None),
        "Open_Interest": ("btc/market-data/open-interest",                   "open_interest",              "binance"),
        "Coinbase_Prem": ("btc/market-data/coinbase-premium-index",          "coinbase_premium_index",     None),
        "NRPL":          ("btc/network-indicator/nrpl",                      "nrpl",                       None),
    }

    out = {}
    for name, (ep, fld, exch) in FETCHES.items():
        cache_key = f"cq_{name}_{start}"
        cached = _load_cache(cache_key)
        if cached is not None:
            out[name] = cached
            print(f"  CQ {name}: cached ({len(cached)} pts)")
            continue
        print(f"  Fetching CQ {name}...", end=" ", flush=True)
        s = _fetch_cquant(ep, token, fld, start=start, exchange=exch)
        if s is not None:
            out[name] = s
            _save_cache(cache_key, s)
            print(f"OK ({len(s)} pts)")
        else:
            print("FAILED")
    return out


# ============================================================
# 3. PROXY SIGNALS  (yfinance / computed — no API key needed)
# ============================================================

def compute_proxy_signals(close: pd.Series,
                          sp500: pd.Series,
                          vix:   pd.Series,
                          dxy:   pd.Series,
                          gold:  pd.Series,
                          hy:    pd.Series = None) -> dict:
    """
    Compute proxy signals from publicly available market data.
    These supplement (or replace, if no CQ key) CryptoQuant signals.
    Always included: macro signals (VIX, DXY, SP500, Gold, HY Spread,
    BTC realized vol, BTC momentum proxies).
    """
    out = {}
    idx = close.index

    # ---- BTC-derived proxies ----

    log_ret = np.log(close / close.shift(1))

    # MVRV proxy: Mayer Multiple (price / 200-day MA) — high → overvalued → normal
    ma200 = close.rolling(200, min_periods=100).mean()
    out["MVRV_Proxy"] = (close / ma200.replace(0, np.nan)).rename("MVRV_Proxy")

    # Puell Multiple proxy: 14d change / 365d change (relative miner revenue)
    ret14  = close.pct_change(14)
    ret365 = close.pct_change(365).replace(0, np.nan)
    out["Puell_Proxy"] = (ret14 / ret365).clip(-5, 5).rename("Puell_Proxy")

    # BTC realized vol 30d (high vol = risk-off = bearish) — normal
    out["RealVol_30"] = (log_ret.rolling(30, min_periods=10).std() * np.sqrt(365)).rename("RealVol_30")

    # BTC realized vol 90d — normal
    out["RealVol_90"] = (log_ret.rolling(90, min_periods=30).std() * np.sqrt(365)).rename("RealVol_90")

    # BTC 1-year log return (momentum / overbought proxy) — normal
    out["LR_1Y"] = np.log(close / close.shift(365)).rename("LR_1Y")

    # BTC 2-year z-score (long-run mean reversion) — normal
    lr = np.log(close)
    lr_mean = lr.rolling(730, min_periods=180).mean()
    lr_std  = lr.rolling(730, min_periods=180).std().replace(0, np.nan)
    out["LR_2Y_Z"] = ((lr - lr_mean) / lr_std).rename("LR_2Y_Z")

    # ---- Macro signals ----

    # VIX: extreme spikes = capitulation = bullish (inverse direction)
    if vix is not None and len(vix) > 30:
        out["VIX"] = vix.reindex(idx, method="ffill").rename("VIX")

    # DXY: strong USD = risk-off = bearish for BTC (normal direction)
    if dxy is not None and len(dxy) > 30:
        out["DXY"] = dxy.reindex(idx, method="ffill").rename("DXY")

    # S&P 500 trend: % above/below 200-day MA — inverse (bullish when above)
    if sp500 is not None and len(sp500) > 200:
        sp = sp500.reindex(idx, method="ffill")
        sp_ma200 = sp.rolling(200, min_periods=100).mean()
        out["SP500_Trend"] = ((sp - sp_ma200) / sp_ma200.replace(0, np.nan)).rename("SP500_Trend")

    # Gold 90-day return: rising gold = risk-off = bearish for BTC (normal)
    if gold is not None and len(gold) > 90:
        g = gold.reindex(idx, method="ffill")
        out["Gold_90d"] = g.pct_change(90).rename("Gold_90d")

    # HY Spread: high spread = credit risk = bearish (normal)
    if hy is not None and len(hy) > 30:
        out["HY_Spread"] = hy.reindex(idx, method="ffill").rename("HY_Spread")

    return out


# ============================================================
# 4. SIGNAL DIRECTION REGISTRY
# ============================================================
# 'normal':  high raw value → +1 (bearish)
# 'inverse': high raw value → -1 (bullish)

SIGNAL_DIRECTION = {
    # CryptoQuant — valuation (high = overbought = bearish)
    "MVRV":          "normal",
    "SOPR":          "normal",
    "SOPR_Adj":      "normal",
    "STH_SOPR":      "normal",
    "LTH_SOPR":      "normal",
    "NVT":           "normal",
    "NVM":           "normal",
    "S2F_Dev":       "normal",
    # CryptoQuant — on-chain activity (high = distribution = bearish)
    "Exch_Reserve":  "normal",
    "Exch_Netflow":  "normal",
    "Whale_Ratio":   "normal",
    "MPI":           "normal",
    "Puell":         "normal",
    "Supply_Profit": "normal",
    "SOPR_Ratio":    "normal",
    "Dormancy":      "normal",
    # CryptoQuant — derivatives
    "Lev_Ratio":     "normal",   # high leverage = fragile = bearish
    "SSR":           "normal",   # high SSR = less stablecoin buying power = bearish
    "Stable_Inflow": "inverse",  # high stablecoin inflow = buying power = bullish
    "Funding":       "normal",   # high funding = longs overextended = bearish
    "Open_Interest": "normal",   # high OI = overheated = bearish
    "Coinbase_Prem": "inverse",  # positive premium = US buyers = bullish
    "GBTC_Prem":     "inverse",  # positive premium = institutional demand = bullish
    "NRPL":          "normal",   # positive NRPL = profit-taking = bearish
    # Proxy signals
    "MVRV_Proxy":    "normal",
    "Puell_Proxy":   "normal",
    "RealVol_30":    "normal",
    "RealVol_90":    "normal",
    "LR_1Y":         "normal",
    "LR_2Y_Z":       "normal",
    # Macro proxies
    "VIX":           "inverse",  # extreme VIX spike = capitulation = contrarian bullish
    "DXY":           "normal",   # strong dollar = bearish for BTC
    "SP500_Trend":   "inverse",  # positive SP500 trend = risk-on = bullish for BTC
    "Gold_90d":      "normal",   # gold rising = risk-off = bearish for BTC
    "HY_Spread":     "normal",   # high credit spread = risk-off = bearish
}


# ============================================================
# 5. TERNARY SCORING
# ============================================================

def compute_ternary_matrix(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw signal values → ternary (+1, 0, -1) using rolling percentiles.

    For each signal at date t:
      p_hi = rolling S1_PERCENTILE_HI-th percentile over past S1_PERCENTILE_WINDOW days
      p_lo = rolling S1_PERCENTILE_LO-th percentile over past S1_PERCENTILE_WINDOW days

    If direction='normal':  raw > p_hi → +1 (bearish), raw < p_lo → -1 (bullish)
    If direction='inverse': signs are flipped

    Returns DataFrame with same shape as raw_df, values in {-1.0, 0.0, +1.0}.
    """
    result = pd.DataFrame(index=raw_df.index)
    min_p  = max(30, S1_PERCENTILE_WINDOW // 4)   # need at least 25% of window

    for col in raw_df.columns:
        s         = raw_df[col].copy()
        # For delta signals (e.g., MVRV_d7), inherit direction from base signal
        _base     = col.rsplit("_d", 1)[0] if (col not in SIGNAL_DIRECTION
                    and "_d" in col and col.split("_d")[-1].isdigit()) else col
        direction = SIGNAL_DIRECTION.get(col, SIGNAL_DIRECTION.get(_base, "normal"))

        p_hi = s.rolling(S1_PERCENTILE_WINDOW, min_periods=min_p).quantile(
            S1_PERCENTILE_HI / 100.0)
        p_lo = s.rolling(S1_PERCENTILE_WINDOW, min_periods=min_p).quantile(
            S1_PERCENTILE_LO / 100.0)

        ternary = pd.Series(0.0, index=s.index)
        valid   = s.notna() & p_hi.notna() & p_lo.notna()
        ternary[valid & (s > p_hi)] = +1.0
        ternary[valid & (s < p_lo)] = -1.0

        if direction == "inverse":
            ternary = -ternary

        result[col] = ternary

    return result


# ============================================================
# 5b. DELTA SIGNALS  (event-driven N-day differences)
# ============================================================

def compute_delta_signals(raw_df: pd.DataFrame,
                          lookbacks: list = None) -> pd.DataFrame:
    """
    For each signal column in raw_df, compute N-day differences.
    Returns continuous delta DataFrame with same index as raw_df.
    Column names: {signal}_d{N}  (e.g., MVRV_d7, LTH_SOPR_d30).
    """
    if lookbacks is None:
        lookbacks = DELTA_LOOKBACKS
    out = {}
    for col in raw_df.columns:
        s = raw_df[col].dropna()
        for N in lookbacks:
            out[f"{col}_d{N}"] = s.diff(N)
    return pd.DataFrame(out, index=raw_df.index)


# ============================================================
# 6. S1 WALK-FORWARD ENGINE
#    (exact signals_backtesting_ml.py methodology)
# ============================================================

def run_s1_walkforward(raw_df: pd.DataFrame,
                       ternary_df: pd.DataFrame,
                       close: pd.Series):
    """
    Monthly walk-forward Spearman signal selection for S1.

    For each month-start rebalance date rd:
      1. Training window = [rd - S1_SELECTION_WINDOW, rd]
      2. Target = log(close[t] / close[t - S1_HIST_RETURN_DAYS]) for t in window
      3. Exclude signals with <S1_MIN_VALID_FRAC non-NaN in window
      4. Rank by |Spearman(ternary[t], target[t])| over training window
      5. Select top-S1_N_SIGNALS signals
      6. For each day t in OOS period: composite = mean(ternary[top_signals, t])
         (signal selection is monthly; ternary values update daily)

    Returns: (composite: pd.Series, selection_history: dict[date → list[str]])
    """
    # Target series: 365-day backward log-return
    target = np.log(close / close.shift(S1_HIST_RETURN_DAYS))

    # Earliest possible first rebalance (need S1_SELECTION_WINDOW + percentile window)
    first_rd = (pd.Timestamp(TRAIN_START)
                + pd.Timedelta(days=S1_SELECTION_WINDOW + S1_PERCENTILE_WINDOW))
    first_rd = max(first_rd, pd.Timestamp(EVAL_START))

    rebalance_dates = pd.date_range(start=first_rd, end=close.index[-1], freq="MS")

    composite        = pd.Series(np.nan, index=close.index, name="S1_Composite")
    selection_history = {}  # rd.date() → list of selected signal names

    print(f"\nS1 walk-forward: {len(rebalance_dates)} monthly rebalances "
          f"(top-{S1_N_SIGNALS} signals, {S1_SELECTION_WINDOW}-day Spearman window, "
          f"{S1_PERCENTILE_WINDOW}-day ternary window)")

    last_top_signals = list(ternary_df.columns[:S1_N_SIGNALS])  # fallback

    for i, rd in enumerate(rebalance_dates):
        train_start = rd - pd.Timedelta(days=S1_SELECTION_WINDOW)
        next_rd     = (rebalance_dates[i + 1]
                       if i + 1 < len(rebalance_dates)
                       else close.index[-1] + pd.Timedelta(days=1))

        # ---- Training window ----
        in_window   = (ternary_df.index >= train_start) & (ternary_df.index <= rd)
        t_ternary   = ternary_df.loc[in_window]
        t_target    = target.loc[in_window]
        n_window    = int(in_window.sum())

        if n_window < 60:
            continue

        # ---- Rank signals by |Spearman corr| with 365-day return ----
        ranked = []
        for col in ternary_df.columns:
            # Validity filter: raw signal must have ≥70% non-NaN data in window
            raw_col    = raw_df[col] if col in raw_df.columns else t_ternary[col]
            valid_frac = raw_col.loc[in_window].notna().sum() / n_window
            if valid_frac < S1_MIN_VALID_FRAC:
                continue

            sig_vals = t_ternary[col].dropna()
            tgt_vals = t_target.reindex(sig_vals.index).dropna()
            common   = sig_vals.index.intersection(tgt_vals.index)

            if len(common) < MIN_EVENTS:
                continue

            try:
                corr, _ = spearmanr(sig_vals.loc[common], tgt_vals.loc[common])
                if not np.isnan(corr):
                    ranked.append((col, abs(corr)))
            except Exception:
                continue

        if not ranked:
            top_signals = last_top_signals
        else:
            ranked.sort(key=lambda x: -x[1])
            top_signals      = [c for c, _ in ranked[:S1_N_SIGNALS]]
            last_top_signals = top_signals

        selection_history[rd.date()] = top_signals

        # ---- Compute daily composite in OOS window ----
        oos_dates = close.index[(close.index >= rd) & (close.index < next_rd)]
        for t in oos_dates:
            if t not in ternary_df.index:
                continue
            vals = ternary_df[top_signals].loc[t].dropna()
            if len(vals) > 0:
                composite.loc[t] = float(vals.mean())

    composite = composite.ffill().fillna(0.0)
    active_pct = (composite.abs() > 0.01).mean()
    print(f"  S1 composite: mean={composite.mean():+.4f}, "
          f"std={composite.std():.4f}, non-zero={active_pct:.1%}")
    return composite, selection_history


# ============================================================
# 7. S1 COMPOSITE → BTC EXPOSURE
# ============================================================

def s1_composite_to_exposure(composite: pd.Series,
                              cash_min: float = S1_CASH_MIN,
                              cash_max: float = S1_CASH_MAX,
                              k: float        = S1_SIGMOID_K) -> pd.Series:
    """
    S1 cash → exposure mapping:

      cash% = cash_min + (cash_max - cash_min) * sigmoid(k * composite)

    Base (50–80%):  cash_min=0.20, cash_max=0.50, k=3.0
    Wide (≈52–98%): cash_min=0.00, cash_max=0.50, k=3.0
    """
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(x)))

    cash = composite.apply(
        lambda c: cash_min + (cash_max - cash_min)
                  * (_sigmoid(k * c) if not np.isnan(c) else _sigmoid(0))
    )
    return (1.0 - cash).rename("S1_Exposure")


# ============================================================
# 8. BASE SIGNAL CLASS  (technical signals)
# ============================================================

class BaseSignal(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, close: pd.Series, extra: dict = None) -> pd.Series:
        pass


# ============================================================
# 9. TECHNICAL SIGNAL CLASSES  (from btc_regime_backtest.py — unchanged)
# ============================================================

class MABreakSignal(BaseSignal):
    def __init__(self, ma_type="SMA", freq="daily", windows=None):
        self.ma_type = ma_type
        self.freq    = freq
        self.windows = windows or [7, 30, 50, 200]
        super().__init__(f"MABreak_{ma_type}_{freq}_{'_'.join(map(str, self.windows))}")
        self.min_hist      = 400
        self.search_window = 60
        self.decay_hl      = 15

    def _compute_mas(self, close):
        price = close.resample("W-FRI").last().dropna() if self.freq == "weekly" else close.copy()
        mas = {}
        for w in self.windows:
            mas[w] = (price.rolling(w, min_periods=w).mean() if self.ma_type == "SMA"
                      else price.ewm(span=w, adjust=False).mean())
        if self.freq == "weekly":
            mas = {w: s.reindex(close.index, method="ffill") for w, s in mas.items()}
        return mas

    def _state_vector(self, close, mas):
        ws   = sorted(mas.keys())
        bits = {f"P>{w}": (close > mas[w]).astype(int) for w in ws}
        for i, sw in enumerate(ws):
            for lw in ws[i + 1:]:
                bits[f"{sw}>{lw}"] = (mas[sw] > mas[lw]).astype(int)
        return pd.DataFrame(bits, index=close.index)

    def _get_priority(self, cols):
        def sk(c):
            p = c.replace("P>", "0>").split(">")
            return -(int(p[1]) * 1000 + (1000 - int(p[0])))
        return sorted(cols, key=sk)

    def _detect_breaks(self, sdf):
        recs, diff = [], sdf - sdf.shift(1)
        for col in sdf.columns:
            f = diff[col]
            for d in f[f == 1].index:
                recs.append({"date": d, "bit": col, "direction": 1,
                             "holding": sdf.loc[d].to_dict()})
            for d in f[f == -1].index:
                recs.append({"date": d, "bit": col, "direction": -1,
                             "holding": sdf.loc[d].to_dict()})
        return (pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
                if recs else pd.DataFrame())

    def _related_keys(self, bit, all_keys):
        nums = set()
        for p in bit.replace("P>", "0>").split(">"):
            try:
                nums.add(int(p))
            except Exception:
                pass
        rel = []
        for k in all_keys:
            if k == bit:
                continue
            for p in k.replace("P>", "0>").split(">"):
                try:
                    if int(p) in nums:
                        rel.append(k)
                        break
                except Exception:
                    pass
        return rel[:3]

    def _find_events(self, brk_df, close, bit, dirn, holding, before, horizons):
        m     = ((brk_df["bit"] == bit) & (brk_df["direction"] == dirn)
                 & (brk_df["date"] < before))
        cands = brk_df[m].copy()
        if cands.empty:
            return pd.DataFrame()
        for k in self._related_keys(bit, list(holding.keys())):
            if cands.empty:
                break
            if k in holding and k != bit:
                try:
                    cands = cands[cands["holding"].apply(
                        lambda h: isinstance(h, dict) and h.get(k) == holding[k])]
                except Exception:
                    break
        if cands.empty:
            return pd.DataFrame()
        clustered, ld = [], None
        for _, r in cands.iterrows():
            if ld is None or (r["date"] - ld).days >= CLUSTER_GAP:
                clustered.append(r)
                ld = r["date"]
        cands = pd.DataFrame(clustered)
        if cands.empty:
            return pd.DataFrame()
        evts = []
        for _, r in cands.iterrows():
            d = r["date"]
            if d not in close.index:
                continue
            p0  = close.loc[d]
            rec = {"date": d, "bit": bit, "direction": dirn, "price": p0}
            for h in horizons:
                fi = close.index.searchsorted(d) + h
                if fi < len(close):
                    rec[f"fwd_{h}d"] = close.iloc[fi] / p0 - 1
                else:
                    rec[f"fwd_{h}d"] = np.nan
            evts.append(rec)
        return pd.DataFrame(evts)

    def _eval_events(self, evts):
        scores = []
        for h in FWD_HORIZONS:
            rc = f"fwd_{h}d"
            if rc not in evts.columns:
                continue
            v = evts[rc].dropna()
            if len(v) < MIN_EVENTS:
                continue
            hr, med = (v > 0).mean(), v.median()
            if   hr >= 0.75 and med >  0.02: s = -1.0
            elif hr >= 0.60 and med >  0:    s = -0.5
            elif hr <= 0.25 and med < -0.02: s =  1.0
            elif hr <= 0.40 and med <  0:    s =  0.5
            else:                             s =  0.0
            scores.append((h, s))
        if not scores:
            return 0.0
        tw = sum(HORIZON_WEIGHTS.get(h, 0.1) for h, _ in scores)
        return sum(HORIZON_WEIGHTS.get(h, 0.1) * s for h, s in scores) / tw if tw > 0 else 0.0

    def compute(self, close, extra=None):
        mas    = self._compute_mas(close)
        sdf    = self._state_vector(close, mas)
        pri    = self._get_priority(list(sdf.columns))
        bdf    = self._detect_breaks(sdf)
        scores = pd.Series(0.0, index=close.index, name=self.name)
        start  = close.index[0] + pd.Timedelta(days=self.min_hist)
        for t in close.index[close.index >= start]:
            cutoff = t - pd.Timedelta(days=self.search_window)
            if bdf.empty:
                continue
            rec = bdf[(bdf["date"] > cutoff) & (bdf["date"] <= t)]
            if rec.empty:
                continue
            brk = None
            for b in pri:
                mm = rec[rec["bit"] == b]
                if not mm.empty:
                    r   = mm.iloc[-1]
                    brk = {"date": r["date"], "bit": r["bit"],
                           "direction": r["direction"], "holding": r["holding"],
                           "days_since": (t - r["date"]).days}
                    break
            if brk is None:
                continue
            evts = self._find_events(bdf, close, brk["bit"], brk["direction"],
                                     brk["holding"], t, FWD_HORIZONS)
            if evts.empty:
                continue
            raw = self._eval_events(evts)
            pi  = close.index.searchsorted(t)
            if pi >= 30:
                p30 = close.iloc[pi] / close.iloc[pi - 30] - 1
                if brk["direction"] == -1 and p30 >  0.10:
                    raw *= 0.5
                elif brk["direction"] ==  1 and p30 < -0.10:
                    raw *= 0.5
            scores.loc[t] = np.clip(raw * 0.5 ** (brk["days_since"] / self.decay_hl), -1, 1)
        return scores


class RSISignal(BaseSignal):
    def __init__(self, period=14, ob=70, os_=30, freq="daily"):
        self.period, self.ob, self.os, self.freq = period, ob, os_, freq
        super().__init__(f"RSI_{period}_{freq}")

    def _rsi(self, price):
        d  = price.diff()
        ag = d.clip(lower=0).ewm(span=self.period, adjust=False).mean()
        al = (-d.clip(upper=0)).ewm(span=self.period, adjust=False).mean()
        return 100 - 100 / (1 + ag / al.replace(0, np.nan))

    def compute(self, close, extra=None):
        price = close.resample("W-FRI").last().dropna() if self.freq == "weekly" else close
        rsi   = self._rsi(price)
        if self.freq == "weekly":
            rsi = rsi.reindex(close.index, method="ffill")
        mid, hr = (self.ob + self.os) / 2, (self.ob - self.os) / 2
        score   = ((rsi - mid) / hr).clip(-1, 1).fillna(0)
        score.name = self.name
        return score


class MACDSignal(BaseSignal):
    def __init__(self, fast=12, slow=26, signal_p=9, freq="daily"):
        self.fast, self.slow, self.signal_p, self.freq = fast, slow, signal_p, freq
        super().__init__(f"MACD_{fast}_{slow}_{signal_p}_{freq}")

    def compute(self, close, extra=None):
        price = close.resample("W-FRI").last().dropna() if self.freq == "weekly" else close
        hist  = (price.ewm(span=self.fast, adjust=False).mean()
                 - price.ewm(span=self.slow, adjust=False).mean())
        hist  = hist - hist.ewm(span=self.signal_p, adjust=False).mean()
        if self.freq == "weekly":
            hist = hist.reindex(close.index, method="ffill")
        rm    = hist.rolling(252, min_periods=60).mean()
        rs    = hist.rolling(252, min_periods=60).std()
        z     = ((hist - rm) / rs.replace(0, np.nan)).fillna(0)
        score = (-z / 3).clip(-1, 1)
        score.name = self.name
        return score


class BollingerSignal(BaseSignal):
    def __init__(self, period=20, num_std=2.0, freq="daily"):
        self.period, self.num_std, self.freq = period, num_std, freq
        super().__init__(f"Boll_{period}_{num_std}_{freq}")

    def compute(self, close, extra=None):
        price = close.resample("W-FRI").last().dropna() if self.freq == "weekly" else close
        sma   = price.rolling(self.period, min_periods=self.period).mean()
        std   = price.rolling(self.period, min_periods=self.period).std()
        pct_b = (price - (sma - self.num_std * std)) / (2 * self.num_std * std).replace(0, np.nan)
        if self.freq == "weekly":
            pct_b = pct_b.reindex(close.index, method="ffill")
        score = ((pct_b - 0.5) * 2).clip(-1, 1).fillna(0)
        score.name = self.name
        return score


class ZScoreMASignal(BaseSignal):
    def __init__(self, ma_period=50, z_window=252, ma_type="SMA", freq="daily"):
        self.ma_period, self.z_window, self.ma_type, self.freq = ma_period, z_window, ma_type, freq
        super().__init__(f"Zscore_{ma_type}{ma_period}_{freq}")

    def compute(self, close, extra=None):
        price = close.resample("W-FRI").last().dropna() if self.freq == "weekly" else close
        ma    = (price.rolling(self.ma_period, min_periods=self.ma_period).mean()
                 if self.ma_type == "SMA"
                 else price.ewm(span=self.ma_period, adjust=False).mean())
        dev   = (price - ma) / ma
        rm    = dev.rolling(self.z_window, min_periods=60).mean()
        rs    = dev.rolling(self.z_window, min_periods=60).std()
        z     = ((dev - rm) / rs.replace(0, np.nan)).fillna(0)
        if self.freq == "weekly":
            z = z.reindex(close.index, method="ffill")
        score = (z / 3).clip(-1, 1)
        score.name = self.name
        return score


class PercentileGapSignal(BaseSignal):
    """
    Rolling percentile rank of (price/MA - 1).
    More regime-adaptive than fixed ±0.3 threshold: normalizes by the historical
    distribution of gaps, so a 20% gap scores differently in bull vs bear markets.
    High gap (above 80th pctile of gap history) → +1 (bearish).
    """
    def __init__(self, ma_period: int, window: int = 252, ma_type: str = "SMA"):
        super().__init__(f"GapPct_{ma_type}{ma_period}_{window}d")
        self.ma_period = ma_period
        self.window    = window
        self.ma_type   = ma_type

    def compute(self, close: pd.Series, extra: dict = None) -> pd.Series:
        if self.ma_type == "SMA":
            ma = close.rolling(self.ma_period).mean()
        else:
            ma = close.ewm(span=self.ma_period, adjust=False).mean()
        gap  = (close / ma - 1).fillna(0)
        # rolling percentile rank → [0, 1] → center to [-1, +1]
        rank  = gap.rolling(self.window, min_periods=self.window // 4).rank(pct=True)
        score = (rank * 2 - 1).clip(-1, 1)  # +1 = high gap = bearish
        score.name = self.name
        return score


class SP500OverlaySignal(BaseSignal):
    def __init__(self, ma_type="SMA", windows=None):
        self.ma_type = ma_type
        self.windows = windows or [50, 200]
        super().__init__(f"SP500_{ma_type}_{'_'.join(map(str, self.windows))}")

    def compute(self, close, extra=None):
        sp = extra.get("sp500") if extra else None
        if sp is None:
            return pd.Series(0.0, index=close.index, name=self.name)
        sp  = sp.reindex(close.index, method="ffill").ffill()
        mas = {w: (sp.rolling(w, min_periods=w).mean() if self.ma_type == "SMA"
                   else sp.ewm(span=w, adjust=False).mean()) for w in self.windows}
        bits = {f"SP>MA{w}": (sp > mas[w]).astype(float) for w in sorted(self.windows)}
        ws   = sorted(self.windows)
        for i, sw in enumerate(ws):
            for lw in ws[i + 1:]:
                bits[f"SP_MA{sw}>MA{lw}"] = (mas[sw] > mas[lw]).astype(float)
        avg   = pd.DataFrame(bits).mean(axis=1)
        score = (-(avg - 0.5) * 2).clip(-1, 1).fillna(0)
        score.name = self.name
        return score


class SP500RSISignal(BaseSignal):
    def __init__(self, period=14):
        self.period = period
        super().__init__(f"SP500_RSI_{period}")

    def compute(self, close, extra=None):
        sp = extra.get("sp500") if extra else None
        if sp is None:
            return pd.Series(0.0, index=close.index, name=self.name)
        sp  = sp.reindex(close.index, method="ffill").ffill()
        d   = sp.diff()
        ag  = d.clip(lower=0).ewm(span=self.period, adjust=False).mean()
        al  = (-d.clip(upper=0)).ewm(span=self.period, adjust=False).mean()
        rsi = 100 - 100 / (1 + ag / al.replace(0, np.nan))
        score = -((rsi - 50) / 20).clip(-1, 1).fillna(0)
        score.name = self.name
        return score


# ============================================================
# 10. TECHNICAL SIGNAL REGISTRY
# ============================================================

def build_technical_signal_registry():
    """
    Principled signal universe — 18 configs (32 → 18):

    MA Breakout (both SMA + EMA):
      Daily crypto-native : [30, 90, 180, 365]  — 1m / 1q / 6m / 1y (crypto calendar)
      Daily traditional   : [50, 100, 200, 365]  — 50/200 industry standard + midpoint + crypto annual
      Weekly              : [4, 13, 26, 52]       — same horizons in weekly bars

    Z-Score (both SMA + EMA):
      Daily               : 50, 90, 200, 365      — traditional + crypto quarter/year
      Weekly              : 26                    — semi-annual

    SP500 macro regime (both SMA + EMA):
      [50, 200] — golden cross; traditional equity calendar retained for SP500
    """
    sigs = []

    # MA Breakout
    for mt in ["SMA", "EMA"]:
        sigs.append(MABreakSignal(mt, "daily",  [30, 90, 180, 365]))   # crypto-native
        sigs.append(MABreakSignal(mt, "daily",  [50, 100, 200, 365]))  # traditional + crypto annual
        sigs.append(MABreakSignal(mt, "weekly", [4, 13, 26, 52]))      # weekly equivalent

    # Z-Score
    for mt in ["SMA", "EMA"]:
        for mp in [50, 90, 200, 365]:
            sigs.append(ZScoreMASignal(mp, 252, mt, "daily"))
        sigs.append(ZScoreMASignal(26, 252, mt, "weekly"))

    # SP500 macro regime
    sigs.append(SP500OverlaySignal("SMA", [50, 200]))
    sigs.append(SP500OverlaySignal("EMA", [50, 200]))

    # Percentile Gap — magnitude-aware, regime-adaptive (6 variants)
    for mt in ["SMA", "EMA"]:
        for mp in [90, 200, 365]:
            sigs.append(PercentileGapSignal(mp, window=365, ma_type=mt))

    return sigs


def compute_technical_signals(close: pd.Series,
                               sp500: pd.Series,
                               sigs: list) -> pd.DataFrame:
    extra, results = {"sp500": sp500}, {}
    for sig in sigs:
        print(f"  {sig.name}...", end=" ", flush=True)
        try:
            s   = sig.compute(close, extra)
            act = (s.abs() > 0.01).mean()
            results[sig.name] = s
            print(f"active {act:.1%}")
        except Exception as e:
            print(f"FAILED: {e}")
    return pd.DataFrame(results, index=close.index)


# ============================================================
# 11. COMBINATORICS ENGINE  (from btc_regime_backtest.py — unchanged)
# ============================================================

def discretize_signals(score_df, thresholds=(-0.3, 0.3)):
    lo, hi = thresholds
    d = pd.DataFrame(0, index=score_df.index, columns=score_df.columns)
    d[score_df <= lo] = -1
    d[score_df >= hi] =  1
    return d


def compute_forward_returns(close, horizons=FWD_HORIZONS):
    return pd.DataFrame({h: close.shift(-h) / close - 1 for h in horizons},
                        index=close.index)


def evaluate_joint_state(disc, fwd, s1, s2, js, before):
    mask  = ((disc[s1] == js[0]) & (disc[s2] == js[1]) & (disc.index < before))
    dates = disc.index[mask]
    if len(dates) < MIN_EVENTS:
        return None
    out = {}
    for h in fwd.columns:
        safe = dates[dates + pd.Timedelta(days=h) < before]
        rets = fwd.loc[safe, h].dropna()
        if len(rets) < MIN_EVENTS:
            continue
        out[h] = {"hit_rate": (rets > 0).mean(), "median": rets.median(), "n": len(rets)}
    return out or None



# Regime adjacency order for soft weighting (5.2)
_REGIME_ORDER = ["COLD", "NEUTRAL", "HOT", "EXTREME"]
_REGIME_WEIGHT = {0: 1.0, 1: 0.5, 2: 0.1}   # distance 0/1/2+ → weight


def _regime_weights(mvrv_tr: pd.Series, current_regime: str) -> pd.Series:
    """
    Return a weight for each day in the training window based on regime distance.
    Same regime → 1.0, adjacent → 0.5, two steps away → 0.1.
    """
    try:
        cur_idx = _REGIME_ORDER.index(current_regime)
    except ValueError:
        cur_idx = 1  # default NEUTRAL

    def _w(v):
        reg = _get_regime(v)
        try:
            dist = abs(_REGIME_ORDER.index(reg) - cur_idx)
        except ValueError:
            dist = 2
        return _REGIME_WEIGHT.get(min(dist, 2), 0.1)

    return mvrv_tr.apply(_w)


def pair_power(disc_window, fwd_window, s1, s2, cutoff, weights=None,
               primary_horizon: int = 30, use_magnitude: bool = False):
    """
    Score a (s1, s2) pair by the best joint-state predictive power.
    If weights (pd.Series aligned to disc_window.index) are provided,
    hit-rate is computed as a weighted mean.

    primary_horizon: forward-return column to train on (default 30d).
    use_magnitude:   if True, multiply score by median(|ret|)/realized_vol
                     (Module M — rewards pairs that predict large moves).
    For primary_horizon > 30, n_eff is divided by (primary_horizon/30) as a
    surrogate Newey-West overlap correction.
    """
    best = 0
    if primary_horizon not in fwd_window.columns:
        return best
    # min_ev scaled to horizon: 90d requires 3× more observations
    min_ev = MIN_EVENTS * max(1, primary_horizon // 30)
    for js in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (0, -1), (1, 0), (0, 1)]:
        mask = (disc_window[s1] == js[0]) & (disc_window[s2] == js[1])
        hits = disc_window.index[mask]
        if len(hits) < MIN_EVENTS:
            continue
        safe = hits[hits + pd.Timedelta(days=primary_horizon) < cutoff]
        if len(safe) < min_ev:
            continue
        rets = fwd_window.loc[safe, primary_horizon].dropna()
        if len(rets) < min_ev:
            continue
        if weights is not None:
            w = weights.reindex(rets.index).fillna(0.1)
            w_sum = w.sum()
            if w_sum == 0:
                continue
            hit_rate = (w * (rets > 0)).sum() / w_sum
            n_eff    = w_sum ** 2 / (w ** 2).sum()   # effective sample size
        else:
            hit_rate = (rets > 0).mean()
            n_eff    = len(rets)
        # Overlap adjustment for long horizons (surrogate Newey-West)
        if primary_horizon > 30:
            n_eff = n_eff / (primary_horizon / 30)
        score = abs(hit_rate - 0.5) * n_eff
        # Module M: multiply by magnitude factor (median |ret| / realized vol)
        if use_magnitude:
            rv = fwd_window[primary_horizon].std()
            if rv > 0:
                score *= rets.abs().median() / rv
        best = max(best, score)
    return best



def score_at_date(disc, fwd, top_pairs, t, conviction_gamma=None):
    """
    conviction_gamma: if not None, use corrected Module C conviction weighting.
    Preserves the dead zone (0.40 < hr < 0.60) and median direction check from
    the discrete system, but replaces fixed {0, 0.5, 1.0} output with a continuous
    weight: sv = (2*|hr-0.5|)^gamma, scaled so hr=0.60→small, hr→1.0→large.
    gamma>1 sharpens focus on high-conviction cells (suppresses moderate ones).
    gamma=2.0: hr=0.60→0.04, hr=0.75→0.25, hr=0.90→0.64, hr=1.0→1.0
    """
    scores = []
    for s1, s2 in top_pairs:
        st1, st2 = disc.loc[t, s1], disc.loc[t, s2]
        res = evaluate_joint_state(disc, fwd, s1, s2, (st1, st2), t)
        if res is None:
            continue
        ws, tw = 0.0, 0.0
        for h, stats in res.items():
            w    = HORIZON_WEIGHTS.get(h, 0.1)
            hr   = stats["hit_rate"]
            med  = stats["median"]
            if conviction_gamma is not None:
                # Module C (corrected): preserve dead zone + median check,
                # replace discrete {0.5, 1.0} with continuous gamma weighting
                if hr >= 0.60 and med > 0:
                    sv = -(2 * (hr - 0.5)) ** conviction_gamma   # bullish, continuous
                elif hr <= 0.40 and med < 0:
                    sv =  (2 * (0.5 - hr)) ** conviction_gamma   # bearish, continuous
                else:
                    sv = 0.0                                       # dead zone unchanged
            else:
                # Existing discrete thresholds (unchanged)
                if   hr >= 0.75 and med >  0.02: sv = -1.0
                elif hr >= 0.60 and med >  0:    sv = -0.5
                elif hr <= 0.25 and med < -0.02: sv =  1.0
                elif hr <= 0.40 and med <  0:    sv =  0.5
                else:                             sv =  0.0
            ws += w * sv
            tw += w
        if tw > 0:
            scores.append(np.clip(ws / tw, -1, 1))
    return float(np.mean(scores)) if scores else 0.0


def _prescreen_signals(disc_tr: pd.DataFrame, fwd_tr: pd.DataFrame,
                       sigs: list, n: int) -> list:
    """
    Select top-n signals by solo predictive power on 30d forward returns.

    Uses |Spearman(ternary_signal, 30d_fwd_return)| over the training window.
    Signals with fewer than MIN_EVENTS valid overlapping observations are skipped.

    Returns a list of up to n signal names (falls back to all signals if data
    is insufficient to rank).
    """
    if 30 not in fwd_tr.columns or n >= len(sigs):
        return sigs
    fwd30 = fwd_tr[30].dropna()
    scores = []
    for s in sigs:
        common = disc_tr[s].dropna().index.intersection(fwd30.index)
        if len(common) < MIN_EVENTS:
            continue
        try:
            corr, _ = spearmanr(disc_tr[s].loc[common], fwd30.loc[common])
            if not np.isnan(corr):
                scores.append((s, abs(corr)))
        except Exception:
            pass
    if not scores:
        return sigs[:n]
    scores.sort(key=lambda x: -x[1])
    return [s for s, _ in scores[:n]]


def build_combo_score_wf(score_df: pd.DataFrame,
                          close: pd.Series,
                          train_years: int = 3,
                          retrain_months: int = 6,
                          top_n: int = 15,
                          label: str = "COMBO_WF",
                          prescreen_n: int = None,
                          primary_horizon: int = 30,
                          use_magnitude: bool = False,
                          conviction_gamma=None):
    """
    Walk-forward combinatorics (standard rolling window).
    Re-ranks signal pairs every retrain_months on trailing train_years of data.

    primary_horizon:  forward-return horizon used by pair_power for ranking (Module H).
    use_magnitude:    if True, pair_power multiplies score by magnitude factor (Module M).
    conviction_gamma: if not None, score_at_date uses continuous conviction weighting (Module C).

    Returns: (combo: pd.Series, pair_history: dict[date → list[tuple]])
    """
    disc  = discretize_signals(score_df)
    fwd   = compute_forward_returns(close)
    sigs  = list(score_df.columns)
    # FIX 2: removed full-sample activity filter (act/asigs/pairs) — moved inside loop

    eval_start      = pd.Timestamp(EVAL_START)
    rebalance_dates = pd.date_range(eval_start, close.index[-1],
                                    freq=f"{retrain_months}MS")

    ps_note = f", prescreen top-{prescreen_n}" if prescreen_n else ""
    mods = []
    if use_magnitude:    mods.append("M")
    if conviction_gamma: mods.append(f"C(γ={conviction_gamma})")
    if primary_horizon != 30: mods.append(f"H{primary_horizon}d")
    mod_str = f" [{'+'.join(mods)}]" if mods else ""
    print(f"\n  {label}{mod_str}: {len(sigs)} signals{ps_note}, "
          f"{len(rebalance_dates)} rebalances "
          f"(every {retrain_months}M, {train_years}Y window)")

    combo        = pd.Series(0.0, index=close.index, name=label)
    pair_history = {}   # rd.date() → list of (s1, s2) top pairs

    for i, rd in enumerate(rebalance_dates):
        train_start = rd - pd.DateOffset(years=train_years)
        next_rd     = (rebalance_dates[i + 1]
                       if i + 1 < len(rebalance_dates)
                       else close.index[-1] + pd.Timedelta(days=1))

        disc_tr = disc[(disc.index >= train_start) & (disc.index < rd)]
        fwd_tr  = fwd[(fwd.index  >= train_start) & (fwd.index  < rd)]

        # FIX 2: per-window activity filter — prevents full-sample lookahead bias
        act_tr   = {s: (disc_tr[s] != 0).mean() for s in sigs}
        asigs_tr = [s for s, a in act_tr.items() if a > 0.05]
        pairs_tr = list(combinations(asigs_tr, 2))

        # Optional signal pre-screening
        if prescreen_n and prescreen_n < len(asigs_tr):
            loop_sigs  = _prescreen_signals(disc_tr, fwd_tr, asigs_tr, prescreen_n)
            loop_pairs = list(combinations(loop_sigs, 2))
        else:
            loop_pairs = pairs_tr

        meta      = {(s1, s2): pair_power(disc_tr, fwd_tr, s1, s2, rd,
                                           primary_horizon=primary_horizon,
                                           use_magnitude=use_magnitude)
                     for s1, s2 in loop_pairs}
        ranked    = sorted(meta.items(), key=lambda x: -x[1])
        top_pairs = [p for p, _ in ranked[:top_n]]
        top_lbl   = f"{top_pairs[0][0][:15]}…" if top_pairs else "none"
        print(f"    [{rd.date()} → {next_rd.date()}] pairs={len(loop_pairs)}, top1: {top_lbl}", flush=True)

        pair_history[rd.date()] = top_pairs

        oos_dates = close.index[(close.index >= rd) & (close.index < next_rd)]
        for t in oos_dates:
            combo.loc[t] = score_at_date(disc, fwd, top_pairs, t,
                                          conviction_gamma=conviction_gamma)

    return combo, pair_history


# ============================================================
# 12. S4 — MVRV REGIME-SENSITIVE WALK-FORWARD COMBINATORICS
# ============================================================

def _get_regime(mvrv_value: float) -> str:
    """Classify a single MVRV Z-score value into a regime label."""
    if np.isnan(mvrv_value):
        return "NEUTRAL"
    for label, lo, hi in MVRV_REGIME_BOUNDS:
        if lo <= mvrv_value < hi:
            return label
    return "EXTREME"


def build_combo_score_wf_mvrv(score_df: pd.DataFrame,
                               close: pd.Series,
                               mvrv_series: pd.Series,
                               train_years: int = 3,
                               retrain_months: int = 6,
                               top_n: int = 15,
                               label: str = "S4_COMBO",
                               prescreen_n: int = None,
                               primary_horizon: int = 30):
    """
    S4 walk-forward combinatorics with soft MVRV regime weighting.

    At each rebalance date rd:
      1. Determine current MVRV regime (COLD / NEUTRAL / HOT / EXTREME)
      2. Compute soft weights over full training window:
           same regime → 1.0,  adjacent → 0.5,  two steps away → 0.1
      3. If same-regime days < MIN_REGIME_TRAIN_DAYS, fall back to uniform weights
         (avoids forcing regime signal when data is thin)
      4. [Optional] Pre-screen to top-prescreen_n signals by solo Spearman with 30d return
         (reduces search space from C(n,2) to C(prescreen_n,2) — critical for large universes)
      5. Rank pairs using weighted pair_power
      6. Apply top_n pairs to OOS window

    Returns: (combo: pd.Series, pair_history: dict[date → list[tuple]])
    """
    disc  = discretize_signals(score_df)
    fwd   = compute_forward_returns(close)
    sigs  = list(score_df.columns)
    # FIX 3: removed full-sample activity filter (act/asigs/pairs) — moved inside loop

    # Align MVRV to close index
    mvrv_aligned = (mvrv_series.reindex(close.index, method="ffill")
                    if mvrv_series is not None
                    else pd.Series(1.0, index=close.index))

    eval_start      = pd.Timestamp(EVAL_START)
    rebalance_dates = pd.date_range(eval_start, close.index[-1],
                                    freq=f"{retrain_months}MS")

    ps_note  = f", prescreen top-{prescreen_n}" if prescreen_n else ""
    h_note   = f", H{primary_horizon}d" if primary_horizon != 30 else ""
    print(f"\n  MVRV-regime combo [{label}]: {len(sigs)} signals{ps_note}{h_note}, "
          f"{len(rebalance_dates)} rebalances")

    combo        = pd.Series(0.0, index=close.index, name=label)
    pair_history = {}   # rd.date() → list of (s1, s2) top pairs

    for i, rd in enumerate(rebalance_dates):
        train_start  = rd - pd.DateOffset(years=train_years)
        next_rd      = (rebalance_dates[i + 1]
                        if i + 1 < len(rebalance_dates)
                        else close.index[-1] + pd.Timedelta(days=1))

        # Current MVRV regime
        try:
            mvrv_at_rd = float(mvrv_aligned.asof(rd))
        except Exception:
            mvrv_at_rd = np.nan
        regime = _get_regime(mvrv_at_rd)

        # Full training window
        in_train = (disc.index >= train_start) & (disc.index < rd)
        disc_tr  = disc[in_train]
        fwd_tr   = fwd[in_train]
        mvrv_tr  = mvrv_aligned[in_train]

        # FIX 3: per-window activity filter — prevents full-sample lookahead bias
        act_tr   = {s: (disc_tr[s] != 0).mean() for s in sigs}
        asigs_tr = [s for s, a in act_tr.items() if a > 0.05]
        pairs_tr = list(combinations(asigs_tr, 2))

        # Same-regime day count (for fallback decision)
        n_same_regime = int((mvrv_tr.apply(lambda v: _get_regime(v) == regime)).sum())

        if n_same_regime >= MIN_REGIME_TRAIN_DAYS:
            weights = _regime_weights(mvrv_tr, regime)
            print(f"    [{rd.date()}] regime={regime}, "
                  f"same-regime={n_same_regime}d -> soft-weighted", end="", flush=True)
        else:
            weights = None
            print(f"    [{rd.date()}] regime={regime}, "
                  f"same-regime={n_same_regime}d < {MIN_REGIME_TRAIN_DAYS} -> uniform", end="", flush=True)

        # Optional signal pre-screening
        if prescreen_n and prescreen_n < len(asigs_tr):
            loop_sigs  = _prescreen_signals(disc_tr, fwd_tr, asigs_tr, prescreen_n)
            loop_pairs = list(combinations(loop_sigs, 2))
            print(f", ps={len(loop_sigs)}sigs/{len(loop_pairs)}pairs", end="", flush=True)
        else:
            loop_pairs = pairs_tr

        print(flush=True)

        meta      = {(s1, s2): pair_power(disc_tr, fwd_tr, s1, s2, rd, weights=weights,
                                           primary_horizon=primary_horizon)
                     for s1, s2 in loop_pairs}
        ranked    = sorted(meta.items(), key=lambda x: -x[1])
        top_pairs = [p for p, _ in ranked[:top_n]]

        pair_history[rd.date()] = top_pairs

        oos_dates = close.index[(close.index >= rd) & (close.index < next_rd)]
        for t in oos_dates:
            combo.loc[t] = score_at_date(disc, fwd, top_pairs, t)

    return combo, pair_history


# ============================================================
# 12b. SIGNAL STABILITY ANALYSIS  (5.5)
# ============================================================

def analyze_signal_stability(s1_hist: dict,
                              s2_hist: dict,
                              s3_hist: dict,
                              s4_hist: dict,
                              s5_hist: dict = None,
                              s6_hist: dict = None) -> None:
    """
    Print signal/pair selection stability metrics for each strategy.

    For S1 (signal lists):
      - Turnover rate: fraction of signals that changed vs previous period
      - Persistence count: how many periods each signal appears

    For S2/S3/S4 (pair lists):
      - Pair turnover rate: fraction of pairs that changed vs previous period
      - Most persistent pairs across rebalances
    """
    print("\n" + "=" * 70)
    print("SIGNAL / PAIR STABILITY ANALYSIS")
    print("=" * 70)

    def _signal_turnover(hist: dict) -> float:
        dates  = sorted(hist.keys())
        if len(dates) < 2:
            return float("nan")
        changes = []
        for d1, d2 in zip(dates[:-1], dates[1:]):
            prev = set(hist[d1])
            curr = set(hist[d2])
            changes.append(len(prev.symmetric_difference(curr)) / max(len(prev | curr), 1))
        return float(np.mean(changes))

    def _pair_turnover(hist: dict) -> float:
        dates  = sorted(hist.keys())
        if len(dates) < 2:
            return float("nan")
        changes = []
        for d1, d2 in zip(dates[:-1], dates[1:]):
            prev = set(hist[d1])
            curr = set(hist[d2])
            changes.append(len(prev.symmetric_difference(curr)) / max(len(prev | curr), 1))
        return float(np.mean(changes))

    def _persistence(hist: dict) -> pd.Series:
        from collections import Counter
        cnt: Counter = Counter()
        for items in hist.values():
            for it in items:
                cnt[it] += 1
        return pd.Series(dict(cnt)).sort_values(ascending=False)

    # S1 signal turnover
    if s1_hist:
        to = _signal_turnover(s1_hist)
        print(f"\nS1 — signal turnover: {to:.1%} per period  "
              f"({len(s1_hist)} rebalances)")
        top_sigs = _persistence(s1_hist).head(10)
        print("  Most persistent signals (appearances / total periods):")
        for sig, cnt in top_sigs.items():
            print(f"    {sig:<25s} {cnt:>3d} / {len(s1_hist)}")

    # S2 pair turnover
    if s2_hist:
        to = _pair_turnover(s2_hist)
        print(f"\nS2 — pair turnover: {to:.1%} per period  "
              f"({len(s2_hist)} rebalances)")
        top_pairs = _persistence(s2_hist).head(5)
        print("  Most persistent pairs:")
        for pair, cnt in top_pairs.items():
            print(f"    {str(pair):<45s} {cnt:>3d} / {len(s2_hist)}")

    # S3 pair turnover
    if s3_hist:
        to = _pair_turnover(s3_hist)
        print(f"\nS3 — pair turnover: {to:.1%} per period  "
              f"({len(s3_hist)} rebalances)")
        top_pairs = _persistence(s3_hist).head(5)
        print("  Most persistent pairs:")
        for pair, cnt in top_pairs.items():
            print(f"    {str(pair):<45s} {cnt:>3d} / {len(s3_hist)}")

    # S4 pair turnover
    if s4_hist:
        to = _pair_turnover(s4_hist)
        print(f"\nS4 — pair turnover: {to:.1%} per period  "
              f"({len(s4_hist)} rebalances)")
        top_pairs = _persistence(s4_hist).head(5)
        print("  Most persistent pairs:")
        for pair, cnt in top_pairs.items():
            print(f"    {str(pair):<45s} {cnt:>3d} / {len(s4_hist)}")

    # S5 pair turnover (delta-only)
    if s5_hist:
        to = _pair_turnover(s5_hist)
        print(f"\nS5 — pair turnover: {to:.1%} per period  "
              f"({len(s5_hist)} rebalances)")
        top_pairs = _persistence(s5_hist).head(5)
        print("  Most persistent pairs:")
        for pair, cnt in top_pairs.items():
            print(f"    {str(pair):<45s} {cnt:>3d} / {len(s5_hist)}")

    # S6 pair turnover (state + delta)
    if s6_hist:
        to = _pair_turnover(s6_hist)
        print(f"\nS6 — pair turnover: {to:.1%} per period  "
              f"({len(s6_hist)} rebalances)")
        top_pairs = _persistence(s6_hist).head(5)
        print("  Most persistent pairs:")
        for pair, cnt in top_pairs.items():
            print(f"    {str(pair):<45s} {cnt:>3d} / {len(s6_hist)}")

    print("=" * 70)



# ============================================================
# 13. STRATEGY CONFIGS  (exposure range variations)
# ============================================================

@dataclass
class StrategyConfig:
    name:                 str
    min_exposure:         float = 0.50
    mid_exposure:         float = 0.75
    max_exposure:         float = 1.00
    steepness:            float = 3.0
    confidence_threshold: float = 0.20


# S2: S1 signals + combinatorics — use S1-aligned range
S2_CFG = StrategyConfig(
    "S2 (Signals+Combos)",
    min_exposure=0.50, mid_exposure=0.65, max_exposure=0.80,
    confidence_threshold=0.20)

# S3: All signals + combinatorics — wider regime range
S3_CFG = StrategyConfig(
    "S3 (All+Combos)",
    min_exposure=0.50, mid_exposure=0.75, max_exposure=1.00,
    confidence_threshold=0.20)

# S4: MVRV-sensitive — same range as S3
S4_CFG = StrategyConfig(
    "S4 (MVRV-Regime)",
    min_exposure=0.50, mid_exposure=0.75, max_exposure=1.00,
    confidence_threshold=0.20)

# S5: Delta-only signals — same bounds as S4 (MVRV framework)
S5_CFG = StrategyConfig(
    "S5 (Delta-only)",
    min_exposure=0.50, mid_exposure=0.75, max_exposure=1.00,
    confidence_threshold=0.20)

# S6: State + Delta signals — same bounds as S4 (MVRV framework)
S6_CFG = StrategyConfig(
    "S6 (State+Delta)",
    min_exposure=0.50, mid_exposure=0.75, max_exposure=1.00,
    confidence_threshold=0.20)

# S9: S6 signal universe + S8 confidence gate (hold-previous below |combo| < 0.20)
S9_CFG = StrategyConfig(
    "S9 (S6+Gate)",
    min_exposure=0.50, mid_exposure=0.75, max_exposure=1.00,
    confidence_threshold=0.20)


# ============================================================
# 14. EXPOSURE MAPPING  (for S2/S3/S4 — tanh sigmoid with dead-zone)
# ============================================================

def signal_to_exposure_scalar(s: float, cfg: StrategyConfig) -> float:
    """
    Scalar version: maps a single combo value ∈ [-1, +1] → BTC exposure ∈ [min, max].
    Used internally by signal_to_exposure and gated_exposure.
    """
    thr = cfg.confidence_threshold
    k   = cfg.steepness
    tk  = np.tanh(k)
    if np.isnan(s) or abs(s) < thr:
        return cfg.mid_exposure
    s_adj = (abs(s) - thr) / (1.0 - thr)
    if s < 0:   # bullish → raise exposure
        return cfg.mid_exposure + np.tanh(s_adj * k) / tk * (cfg.max_exposure - cfg.mid_exposure)
    else:       # bearish → lower exposure
        return cfg.mid_exposure - np.tanh(s_adj * k) / tk * (cfg.mid_exposure - cfg.min_exposure)


def signal_to_exposure(signal: pd.Series, cfg: StrategyConfig) -> pd.Series:
    """
    Maps combo signal ∈ [-1, +1] → BTC exposure ∈ [min, max].
    Dead zone: |signal| < confidence_threshold → mid_exposure (no trade).
    Beyond threshold, rescaled signal drives tanh sigmoid.
    """
    return signal.apply(lambda s: signal_to_exposure_scalar(s, cfg))


def tiered_exposure(combo: pd.Series, cfg: StrategyConfig,
                    thresholds: list = None) -> pd.Series:
    """
    S7: Maps combo → discrete 3-tier exposure (step function).
    Only commits capital at two conviction thresholds; reduces constant low-
    conviction noise vs the smooth sigmoid of signal_to_exposure.

    |combo| < thresholds[0]                 → neutral (mid_exposure)
    thresholds[0] ≤ |combo| < thresholds[1] → moderate (mid±half-range)
    |combo| ≥ thresholds[1]                 → strong (max or min)
    """
    if thresholds is None:
        thresholds = TIER_THRESHOLDS

    def _tier(c):
        if np.isnan(c) or abs(c) < thresholds[0]:
            return cfg.mid_exposure
        elif abs(c) < thresholds[1]:
            if c < 0:   # moderate bull → halfway between mid and max
                return (cfg.mid_exposure + cfg.max_exposure) / 2
            else:       # moderate bear → halfway between min and mid
                return (cfg.min_exposure + cfg.mid_exposure) / 2
        else:
            return cfg.max_exposure if c < 0 else cfg.min_exposure

    return combo.apply(_tier)


def gated_exposure(combo: pd.Series, cfg: StrategyConfig,
                   min_confidence: float = MIN_CONFIDENCE) -> pd.Series:
    """
    S8: Only update position when |combo| > min_confidence.
    Below the gate: hold previous committed position (do NOT revert to neutral).
    """
    result = []
    prev   = cfg.mid_exposure
    for c in combo:
        if np.isnan(c) or abs(c) < min_confidence:
            result.append(prev)
        else:
            new_exp = signal_to_exposure_scalar(c, cfg)
            prev    = new_exp
            result.append(new_exp)
    return pd.Series(result, index=combo.index)


def tiered_gated_exposure(combo: pd.Series, cfg: StrategyConfig,
                          min_confidence: float = MIN_CONFIDENCE,
                          thresholds=None) -> pd.Series:
    """
    Gate + Tiers: only update when |combo| > min_confidence,
    but use the discrete tier step function (not sigmoid) when updating.
    Combines S7's low-fee discretisation with S8's hold-previous noise filter.
    """
    if thresholds is None:
        thresholds = TIER_THRESHOLDS

    def _tier_scalar(c):
        if abs(c) < thresholds[0]:
            return cfg.mid_exposure
        elif abs(c) < thresholds[1]:
            return (cfg.mid_exposure + cfg.max_exposure) / 2 if c < 0 \
                   else (cfg.min_exposure + cfg.mid_exposure) / 2
        else:
            return cfg.max_exposure if c < 0 else cfg.min_exposure

    result = []
    prev   = cfg.mid_exposure
    for c in combo:
        if np.isnan(c) or abs(c) < min_confidence:
            result.append(prev)
        else:
            new_exp = _tier_scalar(c)
            prev    = new_exp
            result.append(new_exp)
    return pd.Series(result, index=combo.index)


def gate_level_sweep(combo: pd.Series, cfg: StrategyConfig, close: pd.Series,
                     gate_levels: list = None) -> pd.DataFrame:
    """
    Sweep gate confidence thresholds on a pre-computed combo series.

    For each gate level g ∈ gate_levels:
      - Apply gated_exposure(combo, cfg, min_confidence=g)
      - Run backtest at standard 10 bps fee
      - Compute: Ann.Ret, Sharpe, Max DD, Fees, position change count

    The implied win-rate proxy maps |combo| gate to historical hit-rate:
      implied_wr% ≈ 50 + gate × 50  (e.g., gate=0.20 → ~60% win rate)

    Reuses the existing backtest engine — no re-scoring of signal pairs needed.
    Returns a DataFrame with one row per gate level.
    """
    if gate_levels is None:
        gate_levels = GATE_SWEEP_LEVELS

    rows = []
    for g in gate_levels:
        exp = gated_exposure(combo, cfg, min_confidence=g)
        bt  = backtest_from_exposure(close, exp, f"Gate_{g:.2f}")
        oos = slice_rebase(bt, EVAL_START)

        cum  = oos["port_cum"]
        ar   = _ann_ret_from_cum(cum)
        vol  = oos["port_ret"].std() * np.sqrt(365)
        sh   = ar / vol if vol > 0 else 0.0   # geometric AR / ann.vol, matches full_metrics
        pk   = cum.cummax()
        md   = ((cum - pk) / pk).min()
        fees = oos["fee_drag"].sum()

        exp_oos    = exp[exp.index >= EVAL_START]
        n_changes  = int((exp_oos.diff().abs() > 1e-4).sum())
        implied_wr = 50.0 + g * 50.0   # linear proxy

        rows.append({
            "Gate |combo|":  f"{g:.2f}",
            "Impl. Win%":    f"~{implied_wr:.0f}%",
            "Ann.Ret":       f"{ar:.1%}",
            "Sharpe":        f"{sh:.2f}",
            "Max DD":        f"{md:.1%}",
            "Fees(%NAV)":    f"{fees:.2%}",
            "Pos.Changes":   n_changes,
        })

    return pd.DataFrame(rows)


# ============================================================
# 15. BACKTEST ENGINE
# ============================================================

def backtest_from_exposure(close: pd.Series,
                           exposure: pd.Series,
                           name: str = "strategy") -> pd.DataFrame:
    """
    Run backtest given a pre-computed daily BTC exposure series [0, 1].
    Applies 10 bps fee on every unit of exposure change (round-trip).
    """
    exp      = exposure.reindex(close.index).ffill().fillna(0.65)
    btc_ret  = close.pct_change().fillna(0)
    prev_exp = exp.shift(1).fillna(exp.iloc[0])

    # Gross portfolio return
    port_ret = prev_exp * btc_ret

    # Fee drag: 10 bps × |Δexposure|
    exp_chg  = exp.diff().abs().fillna(0)
    fee_drag = exp_chg * FEE_PER_UNIT
    port_ret = port_ret - fee_drag

    port_cum = (1 + port_ret).cumprod()
    btc_cum  = (1 + btc_ret).cumprod()

    return pd.DataFrame({
        "btc_cum":  btc_cum,
        "port_cum": port_cum,
        "exposure": exp,
        "port_ret": port_ret,
        "btc_ret":  btc_ret,
        "fee_drag": fee_drag,
    }, index=close.index)


# ============================================================
# 16. METRICS HELPERS
# ============================================================

def backtest_with_fee(close: pd.Series, exposure: pd.Series,
                      name: str, fee: float) -> pd.DataFrame:
    """backtest_from_exposure with a custom fee (for fee sensitivity study)."""
    exp      = exposure.reindex(close.index).ffill().fillna(0.65)
    btc_ret  = close.pct_change().fillna(0)
    prev_exp = exp.shift(1).fillna(exp.iloc[0])
    port_ret = prev_exp * btc_ret - exp.diff().abs().fillna(0) * fee
    port_cum = (1 + port_ret).cumprod()
    btc_cum  = (1 + btc_ret).cumprod()
    return pd.DataFrame({"btc_cum": btc_cum, "port_cum": port_cum,
                         "exposure": exp, "port_ret": port_ret,
                         "btc_ret": btc_ret, "fee_drag": exp.diff().abs().fillna(0) * fee},
                        index=close.index)


def slice_rebase(df: pd.DataFrame, start: str) -> pd.DataFrame:
    d = df[df.index >= start].copy()
    for c in ["btc_cum", "port_cum"]:
        if c in d.columns:
            d[c] = d[c] / d[c].iloc[0]
    return d


# ============================================================
# 16b. ROBUSTNESS HELPERS
# ============================================================

SUB_PERIODS = [
    ("Bull",      [("2020-01-01", "2021-12-31"), ("2024-01-01", "2024-12-31")]),
    ("Bear",      [("2022-01-01", "2022-12-31"), ("2025-01-01", "2026-12-31")]),
    ("Recovery",  [("2023-01-01", "2023-12-31")]),
]


def _sharpe_from_ret(ret: pd.Series) -> float:
    s = ret.std()
    return ret.mean() / s * np.sqrt(365) if s > 0 else 0.0


def _ann_ret_from_cum(cum: pd.Series) -> float:
    ny = (cum.index[-1] - cum.index[0]).days / 365.25
    return (cum.iloc[-1] / cum.iloc[0]) ** (1 / ny) - 1 if ny > 0 else 0.0


def bootstrap_sharpe_diff(ret_a: pd.Series, ret_b: pd.Series,
                          n: int = 5000, seed: int = 42) -> dict:
    """
    Bootstrap test: is Sharpe(B) - Sharpe(A) > 0?
    Returns observed diff, p-value (fraction of resamples where B ≤ A), 90% CI.
    """
    common = ret_a.index.intersection(ret_b.index)
    ra = ret_a.loc[common].values
    rb = ret_b.loc[common].values
    obs = _sharpe_from_ret(pd.Series(rb)) - _sharpe_from_ret(pd.Series(ra))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    idx = np.arange(len(ra))
    for i in range(n):
        bi = rng.choice(idx, len(idx), replace=True)
        diffs[i] = _sharpe_from_ret(pd.Series(rb[bi])) - _sharpe_from_ret(pd.Series(ra[bi]))
    p_val  = float(np.mean(diffs <= 0))
    ci_lo, ci_hi = np.percentile(diffs, [5, 95])
    return {"diff": obs, "p_value": p_val, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "significant": p_val < 0.10}


def sub_period_sharpe(oos_dict: dict) -> pd.DataFrame:
    """Compute Sharpe for each strategy within each sub-period."""
    rows = []
    for period_name, ranges in SUB_PERIODS:
        for strat, bt in oos_dict.items():
            col = "btc_ret" if strat == "BTC Buy & Hold" else "port_ret"
            if col not in bt.columns:
                continue
            combined = pd.concat([bt[col].loc[s:e]
                                   for s, e in ranges]).sort_index()
            if len(combined) < 20:
                sh = np.nan
            else:
                sh = _sharpe_from_ret(combined)
            rows.append({"Period": period_name, "Strategy": strat, "Sharpe": round(sh, 2)})
    return pd.DataFrame(rows).pivot(index="Strategy", columns="Period", values="Sharpe")


def fee_sensitivity_table(exposure_map: dict, close: pd.Series,
                           fee_levels: list = None) -> pd.DataFrame:
    """
    Run each execution strategy at multiple fee levels.
    exposure_map: {label: exposure_series}
    """
    if fee_levels is None:
        fee_levels = [0.0005, 0.0010, 0.0020]
    rows = []
    for fee in fee_levels:
        for label, exp in exposure_map.items():
            bt  = backtest_with_fee(close, exp, label, fee)
            oos = slice_rebase(bt, EVAL_START)
            rows.append({
                "Fee (bps)": int(fee * 10000),
                "Strategy":  label,
                "Ann.Ret":   f"{_ann_ret_from_cum(oos['port_cum']):.1%}",
                "Sharpe":    f"{_sharpe_from_ret(oos['port_ret']):.2f}",
            })
    df = pd.DataFrame(rows)
    return df.pivot(index="Strategy", columns="Fee (bps)", values=["Ann.Ret", "Sharpe"])


def print_layer_conclusion(layer: str, base_label: str, new_label: str,
                           boot: dict, win_years: int, total_years: int,
                           note: str = "") -> None:
    sig = "✓ significant (p<0.10)" if boot["significant"] else "✗ not significant"
    print(f"\n{'='*70}")
    print(f"  LAYER CONCLUSION — {layer}")
    print(f"  {base_label}  →  {new_label}")
    print(f"  Sharpe improvement: {boot['diff']:+.3f}  {sig}")
    print(f"  90% CI: [{boot['ci_lo']:+.3f}, {boot['ci_hi']:+.3f}]")
    print(f"  Year win rate: {win_years}/{total_years} years {new_label} beats {base_label}")
    if note:
        print(f"  Note: {note}")
    verdict = "ADD" if (boot["significant"] and win_years >= total_years // 2 + 1) \
              else ("MARGINAL" if boot["significant"] or win_years >= total_years // 2 + 1
                    else "SKIP")
    print(f"  Verdict: {verdict}")
    print(f"{'='*70}")


def full_metrics(cum: pd.Series, daily: pd.Series,
                 label: str, fee_drag: pd.Series = None,
                 naive_ar: float = None, naive_label: str = None) -> dict:
    tr  = cum.iloc[-1] / cum.iloc[0] - 1
    ny  = (cum.index[-1] - cum.index[0]).days / 365.25
    ar  = (1 + tr) ** (1 / ny) - 1 if ny > 0 else 0
    av  = daily.std() * np.sqrt(365)
    sh  = ar / av if av > 0 else 0
    pk  = cum.cummax()
    md  = ((cum - pk) / pk).min()
    ca  = ar / abs(md) if md != 0 else 0
    tf  = fee_drag.sum() if fee_drag is not None else 0
    result = {
        "Strategy":     label,
        "Ann.Ret":      f"{ar:.1%}",
        "Total Ret":    f"{tr:.1%}",
        "Volatility":   f"{av:.1%}",
        "Sharpe":       f"{sh:.2f}",
        "Calmar":       f"{ca:.2f}",
        "Max DD":       f"{md:.1%}",
        "Fees(%NAV)":   f"{tf:.2%}",
        "Final($100k)": f"${100_000 * (1 + tr):,.0f}",
    }
    if naive_ar is not None:
        diff = ar - naive_ar
        sign = "+" if diff >= 0 else ""
        flag = "✓" if diff > 0 else ("=" if abs(diff) < 0.001 else "✗")
        result["Naive"]    = naive_label or "?"
        result["vs Naive"] = f"{sign}{diff:.1%} {flag}"
    else:
        result["Naive"]    = "—"
        result["vs Naive"] = "—"
    return result


def yearly_breakdown(backtest_results: dict) -> pd.DataFrame:
    rows = []
    for name, bt in backtest_results.items():
        col = "btc_cum" if name == "BTC Buy & Hold" else "port_cum"
        if col not in bt.columns:
            continue
        for yr, grp in bt.groupby(bt.index.year):
            if grp.empty:
                continue
            rows.append({"Year": yr, "Strategy": name,
                         "Return": grp[col].iloc[-1] / grp[col].iloc[0] - 1})
    df    = pd.DataFrame(rows)
    pivot = df.pivot(index="Year", columns="Strategy", values="Return")
    order = ["BTC Buy & Hold"] + [s for s in pivot.columns if s != "BTC Buy & Hold"]
    return pivot[[c for c in order if c in pivot.columns]]


# ============================================================
# 17. VISUALIZATION
# ============================================================

COLORS = {
    "BTC Buy & Hold":            "orange",
    "BTC 75% Fixed":             "#FFD54F",
    # Layer 1
    "Base":                      "#9C27B0",
    # Layer 2 — delta
    "Base+Delta":                "#FF5722",
    # Layer 3 — MVRV regime
    "Base+MVRV":                 "#F44336",
    "Base+Delta+MVRV":           "#FF9800",
    # Layer 4 — execution policies
    "Base+MVRV+Tiered":               "#673AB7",
    "Base+MVRV+Gate":                 "#E91E63",
    "Base+MVRV+Tiered+Gate":          "#00BCD4",
    "Base+Delta+MVRV+Gate":           "#4CAF50",
    "Base+Delta+MVRV+Gate (PS20)":       "#00E676",
    "Base+Delta+MVRV+Gate (PS20+H90)":  "#69F0AE",
    # v5.0 Module experiment variants
    "V0 Base":     "#90A4AE",
    "V1 +M":       "#FF9800",
    "V2 +Cg2":     "#AB47BC",
    "V2b +Cg1":    "#CE93D8",
    "V3 +M+Cg2":   "#EF5350",
    "V4 +H90":     "#42A5F5",
    "V5 +H90+Cg2": "#26C6DA",
}


def plot_strategy_comparison(backtest_results: dict, title: str, out_path: str):
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    # Equity curves (log scale)
    ax = axes[0]
    for name, bt in backtest_results.items():
        col = "btc_cum" if name == "BTC Buy & Hold" else "port_cum"
        lw  = 2.5 if name in ("BTC Buy & Hold",) else 1.8
        ax.plot(bt.index, bt[col], label=name.strip(),
                color=COLORS.get(name, "gray"), lw=lw, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1  (log scale)")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Signal for first non-BH strategy
    ax = axes[1]
    non_bh = [(n, bt) for n, bt in backtest_results.items() if n != "BTC Buy & Hold"]
    if non_bh:
        name0, bt0 = non_bh[0]
        if "exposure" in bt0.columns:
            sig = bt0["exposure"] * 2 - 1   # scale exposure to [-1, +1] for display
            ax.fill_between(sig.index, sig, 0, where=sig > 0,
                            color="red",   alpha=0.35, label="Higher cash (reduce BTC)")
            ax.fill_between(sig.index, sig, 0, where=sig < 0,
                            color="green", alpha=0.35, label="Lower cash (add BTC)")
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_ylabel(f"Exposure signal — {name0.strip()}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    # Exposure overlay
    ax = axes[2]
    for name, bt in backtest_results.items():
        if "exposure" in bt.columns:
            ax.plot(bt.index, bt["exposure"] * 100,
                    label=name.strip(), color=COLORS.get(name, "gray"),
                    lw=1.2, alpha=0.8)
    ax.axhline(65, color="blue",  lw=0.7, ls="--", alpha=0.4, label="65% mid (S1/S2)")
    ax.axhline(75, color="gray",  lw=0.7, ls="--", alpha=0.4, label="75% mid (S3/S4)")
    ax.axhline(100, color="black", lw=0.5, ls=":", alpha=0.3)
    ax.set_ylabel("BTC Exposure %")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_yearly_heatmap(yearly_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(max(10, len(yearly_df) * 1.1),
                                    max(4, len(yearly_df.columns) * 0.65)))
    vals = yearly_df.values.astype(float)
    vmax = max(abs(np.nanmax(vals)), abs(np.nanmin(vals)), 0.01)
    im   = ax.imshow(vals.T, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(yearly_df.index)))
    ax.set_xticklabels(yearly_df.index.astype(str), fontsize=10)
    ax.set_yticks(range(len(yearly_df.columns)))
    ax.set_yticklabels([c.strip() for c in yearly_df.columns], fontsize=9)
    ax.set_title("Annual Returns by Strategy  (green=positive, red=negative)", fontsize=12)
    for r, yr in enumerate(yearly_df.index):
        for c, strat in enumerate(yearly_df.columns):
            v = yearly_df.loc[yr, strat]
            if not np.isnan(v):
                ax.text(r, c, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, color="black" if abs(v) < 0.4 else "white")
    plt.colorbar(im, ax=ax, label="Annual Return", shrink=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_yearly_bars(yearly_df: pd.DataFrame, out_path: str):
    years    = yearly_df.index.tolist()
    strats   = yearly_df.columns.tolist()
    n_strats = len(strats)
    x        = np.arange(len(years))
    width    = 0.75 / n_strats

    fig, ax = plt.subplots(figsize=(max(12, len(years) * 1.5), 6))
    for i, s in enumerate(strats):
        vals = [yearly_df.loc[yr, s] for yr in years]
        ax.bar(x + i * width - (n_strats - 1) * width / 2, vals,
               width=width * 0.9, label=s.strip(),
               color=COLORS.get(s, None), alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Annual Return")
    ax.set_title("Year-by-Year Returns: All Strategies vs BTC", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_signal_heatmap(ternary_df: pd.DataFrame,
                        close: pd.Series,
                        n_top: int = 20,
                        out_path: str = None):
    activity = ternary_df.abs().mean().sort_values(ascending=False)
    top      = activity.head(n_top).index.tolist()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [1, 3]}, sharex=True)
    axes[0].plot(close.index, close, color="orange", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("BTC Price")
    axes[0].set_title(f"Top {n_top} On-Chain/Macro Signal Heatmap  (ternary ±1/0)")
    axes[0].grid(True, alpha=0.3)

    weekly = ternary_df[top].resample("W").last()
    im = axes[1].imshow(
        weekly.T.values, aspect="auto", cmap="RdYlGn_r",
        vmin=-1, vmax=1,
        extent=[mdates.date2num(weekly.index[0]),
                mdates.date2num(weekly.index[-1]),
                len(top) - 0.5, -0.5])
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(top, fontsize=7)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.colorbar(im, ax=axes[1], label="Ternary (red=bearish +1, green=bullish -1)")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close()


# ============================================================
# 18. MARKET DATA FETCH
# ============================================================

def fetch_market_data():
    """Download BTC, SPY/^GSPC, VIX, DXY, Gold from yfinance. Returns clean Series."""
    cache_key = f"market_data_{TRAIN_START}"
    cached = _load_cache(cache_key)
    if cached is not None:
        close, sp500, vix, dxy, gold, hy = cached
        print(f"  Market data: loaded from cache "
              f"(BTC {len(close)} days through {close.index[-1].date()})")
        return close, sp500, vix, dxy, gold, hy

    def _dl(ticker, start=TRAIN_START):
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].dropna().squeeze()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s

    print("Fetching BTC-USD...", end=" "); close = _dl("BTC-USD"); print(f"{len(close)} days")
    print("Fetching SPY...",     end=" "); sp500  = _dl("SPY");     print(f"{len(sp500)} days")
    print("Fetching ^VIX...",   end=" ")
    try:
        vix = _dl("^VIX")
        print(f"{len(vix)} days")
    except Exception:
        vix = None
        print("FAILED (using None)")
    print("Fetching DX-Y.NYB...", end=" ")
    try:
        dxy = _dl("DX-Y.NYB")
        print(f"{len(dxy)} days")
    except Exception:
        dxy = None
        print("FAILED")
    print("Fetching GC=F (Gold)...", end=" ")
    try:
        gold = _dl("GC=F")
        print(f"{len(gold)} days")
    except Exception:
        gold = None
        print("FAILED")

    # HY Spread — try FRED
    hy = None
    fred_key = os.environ.get("FRED_API_KEY")
    if FRED_AVAILABLE and fred_key:
        try:
            fred = Fred(api_key=fred_key)
            raw_hy = fred.get_series("BAMLH0A0HYM2", observation_start=TRAIN_START)
            raw_hy.index = pd.to_datetime(raw_hy.index).tz_localize(None)
            hy = raw_hy.dropna()
            print(f"HY Spread (FRED): {len(hy)} days")
        except Exception as e:
            print(f"HY Spread FRED failed: {e}")

    _save_cache(cache_key, (close, sp500, vix, dxy, gold, hy))
    return close, sp500, vix, dxy, gold, hy


# ============================================================
# 18b. RETRAIN FREQUENCY STUDY
# ============================================================

def run_retrain_frequency_study(combined_df: pd.DataFrame,
                                 close: pd.Series,
                                 mvrv_raw: pd.Series) -> None:
    """
    Compare S4 (MVRV-regime combinatorics) across retrain intervals [1, 3, 6, 12] months.

    Holds constant: signal universe (combined_df), train_years=3, top_n=15.
    Varies: retrain_months — how often pair rankings are refreshed.

    Key trade-offs:
      1M  — most responsive but highest pair churn; overfits recent noise
      3M  — quarterly; adapts within a market phase; recommended middle ground
      6M  — current default; 13 OOS periods over 6.2 years; can lag regime shifts
      12M — very stable pair selection; may miss entire bear/bull phase

    Prints: Ann.Ret, Sharpe, Calmar, Max DD, Fees, Pair Turnover, N rebalances.
    Conclusion printed at the end.
    """
    print("\n" + "=" * 90)
    print("RETRAIN FREQUENCY STUDY  — S4 (MVRV-Regime)")
    print(f"Signals: {combined_df.shape[1]}  |  Train window: 3Y  |  Top-N pairs: 15")
    print(f"Tested intervals: 1M / 3M / 6M / 12M  (eval period: {EVAL_START} →)")
    print("=" * 90)

    study_rows = []
    study_bts  = {}

    for months in [1, 3, 6, 12]:
        lbl = f"S4 [{months}M]"
        print(f"\n--- {lbl} ---", flush=True)
        combo, ph = build_combo_score_wf_mvrv(
            combined_df, close, mvrv_raw,
            train_years=3, retrain_months=months, top_n=15,
            label=f"S4_{months}M"
        )
        exp = signal_to_exposure(combo, S4_CFG)
        bt  = backtest_from_exposure(close, exp, lbl)
        oos = slice_rebase(bt, EVAL_START)
        study_bts[lbl] = oos

        cum = oos["port_cum"]
        ny  = (cum.index[-1] - cum.index[0]).days / 365.25
        ar  = (cum.iloc[-1] / cum.iloc[0]) ** (1 / ny) - 1
        av  = oos["port_ret"].std() * np.sqrt(365)
        sh  = ar / av if av > 0 else 0
        pk  = cum.cummax()
        md  = ((cum - pk) / pk).min()
        ca  = ar / abs(md) if md != 0 else 0
        tf  = oos["fee_drag"].sum()
        n_reb = len(ph)

        # Pair selection turnover
        dates = sorted(ph.keys())
        tos   = []
        for d1, d2 in zip(dates[:-1], dates[1:]):
            prev, curr = set(ph[d1]), set(ph[d2])
            tos.append(len(prev.symmetric_difference(curr)) / max(len(prev | curr), 1))
        avg_to = float(np.mean(tos)) if tos else float("nan")

        study_rows.append({
            "Interval":      f"{months}M",
            "Rebalances":    n_reb,
            "Ann.Ret":       f"{ar:.1%}",
            "Sharpe":        f"{sh:.2f}",
            "Calmar":        f"{ca:.2f}",
            "Max DD":        f"{md:.1%}",
            "Fees(%NAV)":    f"{tf:.2%}",
            "Pair Turnover": f"{avg_to:.1%}",
            "_ar": ar,
            "_sh": sh,
        })

    sdf = pd.DataFrame(study_rows).drop(columns=["_ar", "_sh"])
    print("\n" + "=" * 90)
    print("RETRAIN FREQUENCY STUDY — RESULTS")
    print("=" * 90)
    print(sdf.to_string(index=False))
    print("=" * 90)

    best_sh  = max(study_rows, key=lambda r: r["_sh"])
    best_ar  = max(study_rows, key=lambda r: r["_ar"])
    print(f"\nConclusion:")
    print(f"  Best Sharpe  → {best_sh['Interval']} ({best_sh['Sharpe']})")
    print(f"  Best Ann.Ret → {best_ar['Interval']} ({best_ar['Ann.Ret']})")
    print(f"  Note: higher frequency = lower Pair Turnover stability but faster regime")
    print(f"  adaptation. Quarterly (3M) typically balances both in practice.")
    print("=" * 90)


# ============================================================
# 19. MAIN
# ============================================================

def main():
    print("=" * 80)
    print("BTC COMBINED BACKTEST  v4.1")
    print("Layered robustness framework:")
    print("  L1: Base (state+technical signals, standard combinatorics)")
    print("  L2: +Delta  — do N-day difference signals add net value?")
    print("  L3: +MVRV   — does regime conditioning add net value?")
    print("  L3b-PS: +Delta+MVRV w/ pre-screening (top-20 Spearman → 190 pairs)")
    print("  L4: +Exec   — Tiered / Gate / Tiered+Gate on final base")
    print("  Sweep: Gate sensitivity [0.05 → 0.50] on Base+MVRV combo")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # A. Market data
    # ------------------------------------------------------------------ #
    print("\n[1/8] Fetching market data...")
    close, sp500, vix, dxy, gold, hy = fetch_market_data()
    print(f"BTC range: {close.index[0].date()} → {close.index[-1].date()}")

    # ------------------------------------------------------------------ #
    # B. CryptoQuant signals — REQUIRED
    # ------------------------------------------------------------------ #
    print("\n[2/8] On-chain signals (CryptoQuant)...")
    cquant_token = os.environ.get("CRYPTOQUANT_KEY")
    if cquant_token:
        print("  API key found — fetching live on-chain signals")
    else:
        print("  No CRYPTOQUANT_KEY set — attempting to load from cache...")
    cq_signals = fetch_cquant_signals(cquant_token, start=TRAIN_START)
    if len(cq_signals) < 5:
        print("\n" + "!" * 80)
        print("FATAL: CRYPTOQUANT_KEY environment variable is not set and cache is empty.")
        print()
        print("This backtest requires real on-chain data. Without it:")
        print("  • Only 10 proxy/macro signals are available (intended: 34+).")
        print("  • S4 MVRV regime weighting uses Mayer Multiple proxy,")
        print("    which has the wrong scale for the Z-score regime bounds.")
        print("  • All strategy results are misleading and not comparable.")
        print()
        print("Fix: export CRYPTOQUANT_KEY=<your_key> and re-run.")
        print("!" * 80)
        raise SystemExit(1)
    if not cquant_token:
        print(f"  Cache-only mode: {len(cq_signals)} CQ signals loaded (key not set, using cached data)")

    # ------------------------------------------------------------------ #
    # C. Proxy signals (always computed)
    # ------------------------------------------------------------------ #
    print("\n[3/7] Computing proxy/macro signals...")
    proxy_signals = compute_proxy_signals(close, sp500, vix, dxy, gold, hy)
    for name in proxy_signals:
        print(f"  {name}: OK ({proxy_signals[name].notna().sum()} pts)")

    # Merge all raw signals into a single DataFrame
    all_raw_dict = {**cq_signals, **proxy_signals}
    raw_df = pd.DataFrame(all_raw_dict).reindex(close.index).ffill()
    print(f"\n  Total signals: {raw_df.shape[1]} "
          f"({len(cq_signals)} CryptoQuant + {len(proxy_signals)} proxy/macro)")

    # Delta signals (event-driven N-day differences) for S5/S6
    delta_cols    = [c for c in DELTA_SIGNAL_COLS if c in raw_df.columns]
    delta_raw_df  = compute_delta_signals(raw_df[delta_cols], lookbacks=DELTA_LOOKBACKS)
    delta_cache_key  = f"delta_ternary_{close.index[-1].date()}_{DELTA_LOOKBACKS}_{len(delta_cols)}"
    delta_ternary_df = _load_cache(delta_cache_key)
    if delta_ternary_df is not None:
        print(f"  Delta ternary: loaded from cache {delta_ternary_df.shape}")
    else:
        delta_ternary_df = compute_ternary_matrix(delta_raw_df)
        _save_cache(delta_cache_key, delta_ternary_df)
    print(f"  Delta signals: {len(delta_raw_df.columns)} "
          f"({len(delta_cols)} on-chain/proxy × {DELTA_LOOKBACKS})")

    # ------------------------------------------------------------------ #
    # D. Ternary matrix (shared by S1 and S2)
    # ------------------------------------------------------------------ #
    print("\n[4/7] Computing ternary signal matrix...")
    ternary_df = compute_ternary_matrix(raw_df)
    active_frac = (ternary_df.abs() > 0).mean()
    print(f"  Ternary matrix: {ternary_df.shape}, "
          f"avg activity={active_frac.mean():.1%}")

    # ------------------------------------------------------------------ #
    # E. Technical signals — cached by last BTC date
    # ------------------------------------------------------------------ #
    print("\n[5/7] Computing technical signals...")
    tech_sigs = build_technical_signal_registry()
    print(f"  {len(tech_sigs)} technical signal configs")
    tech_cache_key = f"tech_signals_{close.index[-1].date()}_{len(tech_sigs)}"
    tech_score_df  = _load_cache(tech_cache_key)
    if tech_score_df is not None:
        print(f"  Technical signals: loaded from cache {tech_score_df.shape}")
    else:
        tech_score_df = compute_technical_signals(close, sp500, tech_sigs)
        _save_cache(tech_cache_key, tech_score_df)
    print(f"  Technical signal matrix: {tech_score_df.shape}")

    # Base signal universe: on-chain ternary + technical
    combined_df = pd.concat([ternary_df, tech_score_df], axis=1)
    combined_df = combined_df.reindex(close.index).fillna(0)

    # Extended universe: Base + delta
    s6_df = pd.concat([combined_df, delta_ternary_df], axis=1)
    s6_df = s6_df.reindex(close.index).fillna(0)

    # ------------------------------------------------------------------ #
    # F. Walk-forward combinatorics  (all 4 universe × regime combos)
    # ------------------------------------------------------------------ #
    print("\n[6/7] Walk-forward combinatorics...")

    print("\n  --- L1/L3a: Base (state+technical, no MVRV) ---")
    base_combo, base_pair_hist = build_combo_score_wf(combined_df, close,
                                                       train_years=3, retrain_months=RETRAIN_MONTHS,
                                                       top_n=15, label="BASE_COMBO")
    base_exposure = signal_to_exposure(base_combo, S3_CFG)

    print("\n  --- L2: Base+Delta (state+tech+delta, no MVRV) ---")
    print(f"    Universe: {s6_df.shape[1]} signals ({combined_df.shape[1]} state+tech + {delta_ternary_df.shape[1]} delta)")
    bdelta_combo, bdelta_pair_hist = build_combo_score_wf(s6_df, close,
                                                           train_years=3, retrain_months=RETRAIN_MONTHS,
                                                           top_n=15, label="BDELTA_COMBO")
    bdelta_exposure = signal_to_exposure(bdelta_combo, S3_CFG)

    print("\n  --- L3a: Base+MVRV (state+tech, MVRV-weighted) ---")
    mvrv_raw = raw_df.get("MVRV")
    if mvrv_raw is None:
        print("  WARNING: MVRV Z-score not available — regime weighting disabled")
    bmvrv_combo, bmvrv_pair_hist = build_combo_score_wf_mvrv(combined_df, close, mvrv_raw,
                                                              train_years=3, retrain_months=RETRAIN_MONTHS,
                                                              top_n=15, label="BMVRV_COMBO")
    bmvrv_exposure = signal_to_exposure(bmvrv_combo, S4_CFG)

    print("\n  --- L3b: Base+Delta+MVRV (state+tech+delta, MVRV-weighted) ---")
    bdmvrv_combo, bdmvrv_pair_hist = build_combo_score_wf_mvrv(s6_df, close, mvrv_raw,
                                                                train_years=3, retrain_months=RETRAIN_MONTHS,
                                                                top_n=15, label="BDMVRV_COMBO")
    bdmvrv_exposure = signal_to_exposure(bdmvrv_combo, S4_CFG)

    print(f"\n  --- L3b-PS: Base+Delta+MVRV (pre-screened top-{PRESCREEN_N}) ---")
    print(f"    Reduces C({s6_df.shape[1]},2)={s6_df.shape[1]*(s6_df.shape[1]-1)//2} pairs "
          f"→ C({PRESCREEN_N},2)={PRESCREEN_N*(PRESCREEN_N-1)//2} pairs per rebalance")
    bdmvrv_ps_combo, bdmvrv_ps_hist = build_combo_score_wf_mvrv(
        s6_df, close, mvrv_raw,
        train_years=3, retrain_months=RETRAIN_MONTHS,
        top_n=15, label="BDMVRV_PS_COMBO", prescreen_n=PRESCREEN_N)
    bdmvrv_ps_gate = gated_exposure(bdmvrv_ps_combo, S4_CFG)

    print(f"\n  --- L3b-PS-H90: Base+Delta+MVRV+Gate (PS20, H90d) ---")
    print(f"    H90: 90-day pair-selection horizon + NW surrogate (n_eff /= 3)")
    bdmvrv_ps_h90_combo, bdmvrv_ps_h90_hist = build_combo_score_wf_mvrv(
        s6_df, close, mvrv_raw,
        train_years=3, retrain_months=RETRAIN_MONTHS,
        top_n=15, label="BDMVRV_PS_H90_COMBO", prescreen_n=PRESCREEN_N,
        primary_horizon=90)
    bdmvrv_ps_h90_gate = gated_exposure(bdmvrv_ps_h90_combo, S4_CFG)

    # ── v5.0: Module Experiment (V0–V5) ───────────────────────────────── #
    print("\n\n" + "═"*70)
    print("  [v5.0] MODULE EXPERIMENT: M / C / H variants on Base")
    print("  Signal universe: combined_df (state+tech, no delta, no MVRV)")
    print("  Variants: V0=baseline, V1=+M, V2=+C, V3=+M+C, V4=+H90, V5=+H90+C")
    print("═"*70)
    v5_combos    = {}
    v5_exposures = {}
    v5_bts       = {}
    v5_oos       = {}
    for _vlabel, (_ph, _um, _cg) in V5_VARIANTS.items():
        _v_combo, _ = build_combo_score_wf(
            combined_df, close,
            train_years=3, retrain_months=RETRAIN_MONTHS, top_n=15,
            label=f"COMBO_{_vlabel.replace(' ', '').replace('+', '').replace('(', '').replace(')', '')}",
            primary_horizon=_ph, use_magnitude=_um, conviction_gamma=_cg)
        _v_exp = signal_to_exposure(_v_combo, S3_CFG)
        _v_bt  = backtest_from_exposure(close, _v_exp, _vlabel)
        _v_oos = slice_rebase(_v_bt, EVAL_START)
        v5_combos[_vlabel]    = _v_combo
        v5_exposures[_vlabel] = _v_exp
        v5_bts[_vlabel]       = _v_bt
        v5_oos[_vlabel]       = _v_oos

    # ------------------------------------------------------------------ #
    # G. Backtests
    # ------------------------------------------------------------------ #
    print("\n[7/7] Running backtests and robustness analysis...")

    bh_bt   = backtest_from_exposure(close, pd.Series(1.0,  index=close.index), "BTC Buy & Hold")
    bm75_bt = backtest_from_exposure(close, pd.Series(0.75, index=close.index), "BTC 75% Fixed")
    bh_oos   = slice_rebase(bh_bt,   EVAL_START)
    bm75_oos = slice_rebase(bm75_bt, EVAL_START)
    bm75_ar  = _ann_ret_from_cum(bm75_oos["port_cum"])

    base_bt    = backtest_from_exposure(close, base_exposure,    "Base")
    bdelta_bt  = backtest_from_exposure(close, bdelta_exposure,  "Base+Delta")
    bmvrv_bt   = backtest_from_exposure(close, bmvrv_exposure,   "Base+MVRV")
    bdmvrv_bt  = backtest_from_exposure(close, bdmvrv_exposure,  "Base+Delta+MVRV")
    base_oos   = slice_rebase(base_bt,   EVAL_START)
    bdelta_oos = slice_rebase(bdelta_bt, EVAL_START)
    bmvrv_oos  = slice_rebase(bmvrv_bt,  EVAL_START)
    bdmvrv_oos = slice_rebase(bdmvrv_bt, EVAL_START)

    # Execution layer — applied to Base+MVRV combo (L3 winner from prior runs)
    tiered_exp   = tiered_exposure(bmvrv_combo,      S4_CFG)
    gate_exp     = gated_exposure(bmvrv_combo,        S4_CFG)
    tg_exp       = tiered_gated_exposure(bmvrv_combo, S4_CFG)
    bdmvrv_gate  = gated_exposure(bdmvrv_combo,       S4_CFG)

    tiered_bt    = backtest_from_exposure(close, tiered_exp,    "Base+MVRV+Tiered")
    gate_bt      = backtest_from_exposure(close, gate_exp,      "Base+MVRV+Gate")
    tg_bt        = backtest_from_exposure(close, tg_exp,        "Base+MVRV+Tiered+Gate")
    bdmvrv_g_bt  = backtest_from_exposure(close, bdmvrv_gate,   "Base+Delta+MVRV+Gate")
    bdmvrv_ps_bt    = backtest_from_exposure(close, bdmvrv_ps_gate,    "Base+Delta+MVRV+Gate (PS20)")
    bdmvrv_ps_h90_bt = backtest_from_exposure(close, bdmvrv_ps_h90_gate, "Base+Delta+MVRV+Gate (PS20+H90)")
    tiered_oos        = slice_rebase(tiered_bt,        EVAL_START)
    gate_oos          = slice_rebase(gate_bt,          EVAL_START)
    tg_oos            = slice_rebase(tg_bt,            EVAL_START)
    bdmvrv_g_oos      = slice_rebase(bdmvrv_g_bt,      EVAL_START)
    bdmvrv_ps_oos     = slice_rebase(bdmvrv_ps_bt,     EVAL_START)
    bdmvrv_ps_h90_oos = slice_rebase(bdmvrv_ps_h90_bt, EVAL_START)

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    all_bt = {
        "BTC Buy & Hold":               bh_oos,
        "BTC 75% Fixed":                bm75_oos,
        "Base":                         base_oos,
        "Base+Delta":                   bdelta_oos,
        "Base+MVRV":                    bmvrv_oos,
        "Base+Delta+MVRV":              bdmvrv_oos,
        "Base+MVRV+Tiered":             tiered_oos,
        "Base+MVRV+Gate":               gate_oos,
        "Base+MVRV+Tiered+Gate":        tg_oos,
        "Base+Delta+MVRV+Gate":         bdmvrv_g_oos,
        "Base+Delta+MVRV+Gate (PS20)":       bdmvrv_ps_oos,
        "Base+Delta+MVRV+Gate (PS20+H90)":  bdmvrv_ps_h90_oos,
    }

    metrics_list = [
        full_metrics(bh_oos["btc_cum"],      bh_oos["btc_ret"],      "BTC Buy & Hold"),
        full_metrics(bm75_oos["port_cum"],   bm75_oos["port_ret"],   "BTC 75% Fixed"),
        full_metrics(base_oos["port_cum"],   base_oos["port_ret"],   "Base",
                     base_oos["fee_drag"],   naive_ar=bm75_ar, naive_label="BTC 75% Fixed"),
        full_metrics(bdelta_oos["port_cum"], bdelta_oos["port_ret"], "Base+Delta",
                     bdelta_oos["fee_drag"], naive_ar=bm75_ar, naive_label="BTC 75% Fixed"),
        full_metrics(bmvrv_oos["port_cum"],  bmvrv_oos["port_ret"],  "Base+MVRV",
                     bmvrv_oos["fee_drag"],  naive_ar=bm75_ar, naive_label="BTC 75% Fixed"),
        full_metrics(bdmvrv_oos["port_cum"], bdmvrv_oos["port_ret"], "Base+Delta+MVRV",
                     bdmvrv_oos["fee_drag"], naive_ar=bm75_ar, naive_label="BTC 75% Fixed"),
        full_metrics(tiered_oos["port_cum"], tiered_oos["port_ret"], "Base+MVRV+Tiered",
                     tiered_oos["fee_drag"], naive_ar=bm75_ar, naive_label="BTC 75% Fixed"),
        full_metrics(gate_oos["port_cum"],   gate_oos["port_ret"],   "Base+MVRV+Gate",
                     gate_oos["fee_drag"],   naive_ar=bm75_ar, naive_label="BTC 75% Fixed"),
        full_metrics(tg_oos["port_cum"],     tg_oos["port_ret"],     "Base+MVRV+Tiered+Gate",
                     tg_oos["fee_drag"],     naive_ar=bm75_ar, naive_label="BTC 75% Fixed"),
        full_metrics(bdmvrv_g_oos["port_cum"],      bdmvrv_g_oos["port_ret"],      "Base+Delta+MVRV+Gate",
                     bdmvrv_g_oos["fee_drag"],      naive_ar=bm75_ar, naive_label="BTC 75% Fixed"),
        full_metrics(bdmvrv_ps_oos["port_cum"],     bdmvrv_ps_oos["port_ret"],     "Base+Delta+MVRV+Gate (PS20)",
                     bdmvrv_ps_oos["fee_drag"],     naive_ar=bm75_ar, naive_label="BTC 75% Fixed"),
        full_metrics(bdmvrv_ps_h90_oos["port_cum"], bdmvrv_ps_h90_oos["port_ret"], "Base+Delta+MVRV+Gate (PS20+H90)",
                     bdmvrv_ps_h90_oos["fee_drag"], naive_ar=bm75_ar, naive_label="BTC 75% Fixed"),
    ]

    mdf = pd.DataFrame(metrics_list)
    print("\n" + "=" * 130)
    print(f"OVERALL PERFORMANCE  [{EVAL_START} → {close.index[-1].date()}]"
          f"  (retraining: {RETRAIN_MONTHS}M  |  naive: BTC 75% Fixed)")
    print("=" * 130)
    print(mdf.to_string(index=False))
    print("=" * 130)
    print("vs Naive: ✓ = beats fixed-allocation baseline  ✗ = fails to beat it")

    # Year-by-year
    yearly = yearly_breakdown(all_bt)
    print("\n--- Year-by-Year Returns ---")
    print(yearly.applymap(lambda v: f"{v:+.1%}" if not np.isnan(v) else "-").to_string())

    # ------------------------------------------------------------------ #
    # Robustness analysis
    # ------------------------------------------------------------------ #
    print("\n\n" + "=" * 70)
    print("  ROBUSTNESS ANALYSIS")
    print("=" * 70)

    # Helper: count years where new > base
    def _year_winrate(oos_base, oos_new):
        yb = yearly_breakdown({"base": oos_base, "new": oos_new})
        if yb.empty:
            return 0, 0
        wins = (yb["new"] > yb["base"]).sum()
        return int(wins), len(yb)

    # --- Layer 2: does delta add value? ---
    print("\n[L2] Base  →  Base+Delta")
    boot_l2 = bootstrap_sharpe_diff(base_oos["port_ret"], bdelta_oos["port_ret"])
    wins_l2, total_yr = _year_winrate(base_oos, bdelta_oos)
    print(f"  Base Sharpe:        {_sharpe_from_ret(base_oos['port_ret']):.3f}")
    print(f"  Base+Delta Sharpe:  {_sharpe_from_ret(bdelta_oos['port_ret']):.3f}")
    print_layer_conclusion("L2: + Delta signals", "Base", "Base+Delta",
                           boot_l2, wins_l2, total_yr,
                           note="Delta signals add N-day rate-of-change; high fee if signals flip frequently")

    # --- Layer 3a: does MVRV add value (on Base)? ---
    print("\n[L3a] Base  →  Base+MVRV")
    boot_l3a = bootstrap_sharpe_diff(base_oos["port_ret"], bmvrv_oos["port_ret"])
    wins_l3a, _ = _year_winrate(base_oos, bmvrv_oos)
    print(f"  Base Sharpe:       {_sharpe_from_ret(base_oos['port_ret']):.3f}")
    print(f"  Base+MVRV Sharpe:  {_sharpe_from_ret(bmvrv_oos['port_ret']):.3f}")
    print_layer_conclusion("L3a: + MVRV regime weighting (on Base)", "Base", "Base+MVRV",
                           boot_l3a, wins_l3a, total_yr,
                           note="Regime weighting biases training toward same-MVRV-regime historical periods")

    # --- Layer 3b: does Delta+MVRV beat MVRV alone? ---
    print("\n[L3b] Base+MVRV  →  Base+Delta+MVRV")
    boot_l3b = bootstrap_sharpe_diff(bmvrv_oos["port_ret"], bdmvrv_oos["port_ret"])
    wins_l3b, _ = _year_winrate(bmvrv_oos, bdmvrv_oos)
    print(f"  Base+MVRV Sharpe:        {_sharpe_from_ret(bmvrv_oos['port_ret']):.3f}")
    print(f"  Base+Delta+MVRV Sharpe:  {_sharpe_from_ret(bdmvrv_oos['port_ret']):.3f}")
    print_layer_conclusion("L3b: + Delta on top of MVRV base", "Base+MVRV", "Base+Delta+MVRV",
                           boot_l3b, wins_l3b, total_yr,
                           note=f"Base+Delta+MVRV has {s6_df.shape[1]} signals / {s6_df.shape[1]*(s6_df.shape[1]-1)//2} pairs — high search space risk")

    # --- Layer 4: execution policies ---
    print("\n[L4] Execution policies on Base+MVRV")
    for label, oos in [("Base+MVRV+Tiered",     tiered_oos),
                        ("Base+MVRV+Gate",        gate_oos),
                        ("Base+MVRV+Tiered+Gate", tg_oos)]:
        boot = bootstrap_sharpe_diff(bmvrv_oos["port_ret"], oos["port_ret"])
        wins, _ = _year_winrate(bmvrv_oos, oos)
        fees = oos["fee_drag"].sum()
        print(f"\n  Base+MVRV  →  {label}  (fees: {fees:.2%}/yr)")
        print_layer_conclusion(f"L4: {label}", "Base+MVRV", label, boot, wins, total_yr)

    # --- Layer 4b: prescreened delta+MVRV+gate vs base+tiered+gate ---
    print("\n[L4b] Base+Delta+MVRV+Gate (PS20)  — prescreened large universe")
    boot_ps = bootstrap_sharpe_diff(tg_oos["port_ret"], bdmvrv_ps_oos["port_ret"])
    wins_ps, _ = _year_winrate(tg_oos, bdmvrv_ps_oos)
    fees_ps = bdmvrv_ps_oos["fee_drag"].sum()
    print(f"  Base+MVRV+Tiered+Gate Sharpe:       {_sharpe_from_ret(tg_oos['port_ret']):.3f}")
    print(f"  Base+Delta+MVRV+Gate (PS20) Sharpe: {_sharpe_from_ret(bdmvrv_ps_oos['port_ret']):.3f}")
    print_layer_conclusion("L4b: Pre-screened Delta+MVRV+Gate",
                           "Base+MVRV+Tiered+Gate", "Base+Delta+MVRV+Gate (PS20)",
                           boot_ps, wins_ps, total_yr,
                           note=f"PS20 reduces {s6_df.shape[1]*(s6_df.shape[1]-1)//2} → {PRESCREEN_N*(PRESCREEN_N-1)//2} pairs; "
                                f"gate fees: {fees_ps:.2%}/yr")

    # --- Gate level sensitivity sweep ---
    print("\n--- Gate Level Sensitivity (on Base+MVRV combo) ---")
    print(f"    Mapping: |combo| gate → implied win-rate proxy (50 + gate×50)")
    gate_sweep_df = gate_level_sweep(bmvrv_combo, S4_CFG, close)
    print(gate_sweep_df.to_string(index=False))

    # ── v5.0 Module Experiment Results ────────────────────────────────── #
    print("\n\n" + "═"*70)
    print("  [v5.0] MODULE EXPERIMENT RESULTS  (OOS 2020–2026)")
    print("  Decision criteria: p<0.10 AND ≥4/7 year wins AND 2022 Sharpe ≥ V0")
    print("═"*70)

    v5_metrics = []
    for _vlabel, _v_oos in v5_oos.items():
        _m = full_metrics(_v_oos["port_cum"], _v_oos["port_ret"], _vlabel,
                          _v_oos["fee_drag"], naive_ar=bm75_ar, naive_label="BTC 75% Fixed")
        v5_metrics.append(_m)
    print("\n  ── V0–V5 Performance ──")
    print(pd.DataFrame(v5_metrics).to_string(index=False))

    # Bootstrap Sharpe vs V0
    print("\n  ── Bootstrap Sharpe Difference vs V0 (5000 resamples) ──")
    _v0_ret = v5_oos["V0 Base"]["port_ret"]
    _v0_sh  = _sharpe_from_ret(_v0_ret)
    print(f"  V0 Base Sharpe: {_v0_sh:.3f}")
    for _vlabel in list(V5_VARIANTS.keys())[1:]:
        _vx_ret = v5_oos[_vlabel]["port_ret"]
        _boot   = bootstrap_sharpe_diff(_v0_ret, _vx_ret)
        _sig    = "✓ sig" if _boot["significant"] else "✗ not sig"
        print(f"  V0 → {_vlabel:12s}: Δ={_boot['diff']:+.3f}  p={_boot['p_value']:.3f}  "
              f"CI=[{_boot['ci_lo']:+.3f},{_boot['ci_hi']:+.3f}]  {_sig}")

    # Year-by-year for V0–V5
    _v5_yearly = yearly_breakdown(v5_oos)
    print("\n  ── V0–V5 Year-by-Year Returns ──")
    print(_v5_yearly.applymap(lambda v: f"{v:+.1%}" if not np.isnan(v) else "-").to_string())

    # Decision verdicts using pre-committed criteria
    print("\n  ── Module Verdicts (PROMOTE / MARGINAL / SKIP) ──")
    _bear_s = pd.Timestamp("2022-01-01")
    _bear_e = pd.Timestamp("2022-12-31")
    _v0_bear_ret = _v0_ret.loc[_bear_s:_bear_e]
    _v0_bear_sh  = _sharpe_from_ret(_v0_bear_ret) if len(_v0_bear_ret) > 10 else np.nan
    _v5_yr_all   = _v5_yearly  # already computed above
    for _vlabel in list(V5_VARIANTS.keys())[1:]:
        _vx_ret = v5_oos[_vlabel]["port_ret"]
        _boot   = bootstrap_sharpe_diff(_v0_ret, _vx_ret)
        _wins   = 0
        _total  = 0
        if _vlabel in _v5_yr_all.columns and "V0 Base" in _v5_yr_all.columns:
            _both   = _v5_yr_all[[_vlabel, "V0 Base"]].dropna()
            _wins   = int((_both[_vlabel] > _both["V0 Base"]).sum())
            _total  = len(_both)
        _vx_bear_ret = _vx_ret.loc[_bear_s:_bear_e]
        _vx_bear_sh  = _sharpe_from_ret(_vx_bear_ret) if len(_vx_bear_ret) > 10 else np.nan
        _p1 = _boot["significant"]
        _p2 = _wins >= 4
        _p3 = (not np.isnan(_vx_bear_sh)) and (not np.isnan(_v0_bear_sh)) and (_vx_bear_sh >= _v0_bear_sh)
        _verdict = ("PROMOTE"  if (_p1 and _p2 and _p3) else
                    "MARGINAL" if ((_p1 or _p2) and _p3) else "SKIP")
        print(f"  {_vlabel:12s}: sig={_p1}, yr_wins={_wins}/{_total}, "
              f"bear={_vx_bear_sh:.2f}≥{_v0_bear_sh:.2f}={_p3}  → {_verdict}")

    # --- H90 on Full Stack: head-to-head vs recommended ---
    print("\n\n" + "═"*70)
    print("  [v5.1] H90 ON FULL STACK: PS20 vs PS20+H90")
    print("  Does 90-day pair-selection survive MVRV regime conditioning?")
    print("═"*70)
    compare_labels_h90 = ["Base+Delta+MVRV+Gate (PS20)", "Base+Delta+MVRV+Gate (PS20+H90)"]
    h90_rows = [m for m in metrics_list if m["Strategy"] in compare_labels_h90]
    print(pd.DataFrame(h90_rows).to_string(index=False))

    # Bootstrap: PS20 vs PS20+H90
    _ps20_ret     = bdmvrv_ps_oos["port_ret"]
    _ps20_h90_ret = bdmvrv_ps_h90_oos["port_ret"]
    _h90_boot = bootstrap_sharpe_diff(_ps20_ret, _ps20_h90_ret)
    _h90_sig  = "✓ sig" if _h90_boot["significant"] else "✗ not sig"
    print(f"\n  Bootstrap Sharpe (PS20 → PS20+H90):")
    print(f"    Δ={_h90_boot['diff']:+.3f}  p={_h90_boot['p_value']:.3f}  "
          f"CI=[{_h90_boot['ci_lo']:+.3f},{_h90_boot['ci_hi']:+.3f}]  {_h90_sig}")

    # Year-by-year H90 vs PS20
    _h90_yearly = yearly_breakdown({
        "PS20":      bdmvrv_ps_oos,
        "PS20+H90":  bdmvrv_ps_h90_oos,
    })
    print("\n  Year-by-Year:")
    print(_h90_yearly.applymap(lambda v: f"{v:+.1%}" if not np.isnan(v) else "-").to_string())

    # H90 verdict vs PS20
    _h90_wins = 0
    _h90_total = 0
    if "PS20" in _h90_yearly.columns and "PS20+H90" in _h90_yearly.columns:
        _h90_both  = _h90_yearly[["PS20+H90", "PS20"]].dropna()
        _h90_wins  = int((_h90_both["PS20+H90"] > _h90_both["PS20"]).sum())
        _h90_total = len(_h90_both)
    _ps20_bear_ret    = _ps20_ret.loc[_bear_s:_bear_e]
    _ps20_h90_bear_rt = _ps20_h90_ret.loc[_bear_s:_bear_e]
    _ps20_bear_sh     = _sharpe_from_ret(_ps20_bear_ret) if len(_ps20_bear_ret) > 10 else np.nan
    _ps20_h90_bear_sh = _sharpe_from_ret(_ps20_h90_bear_rt) if len(_ps20_h90_bear_rt) > 10 else np.nan
    _h90_p1 = _h90_boot["significant"]
    _h90_p2 = _h90_wins >= 4
    _h90_p3 = (not np.isnan(_ps20_h90_bear_sh)) and (not np.isnan(_ps20_bear_sh)) and (_ps20_h90_bear_sh >= _ps20_bear_sh)
    _h90_verdict = ("PROMOTE"  if (_h90_p1 and _h90_p2 and _h90_p3) else
                    "MARGINAL" if ((_h90_p1 or _h90_p2) and _h90_p3) else "SKIP")
    print(f"\n  H90 on Full Stack: sig={_h90_p1}, yr_wins={_h90_wins}/{_h90_total}, "
          f"bear={_ps20_h90_bear_sh:.2f}≥{_ps20_bear_sh:.2f}={_h90_p3}  → {_h90_verdict}")

    # --- Final head-to-head comparison: Tiered+Gate vs PS20 ---
    print("\n--- Final Comparison: Best Execution Policy vs Pre-screened Delta Universe ---")
    compare_labels = ["Base+MVRV+Tiered+Gate", "Base+Delta+MVRV+Gate (PS20)"]
    compare_rows = [m for m in metrics_list if m["Strategy"] in compare_labels]
    print(pd.DataFrame(compare_rows).to_string(index=False))

    # --- Sub-period Sharpe table ---
    print("\n--- Sub-Period Sharpe (Bull / Bear / Recovery) ---")
    sp_df = sub_period_sharpe({
        "BTC Buy & Hold":                        bh_oos,
        "BTC 75% Fixed":                         bm75_oos,
        "Base":                                  base_oos,
        "Base+Delta":                            bdelta_oos,
        "Base+MVRV":                             bmvrv_oos,
        "Base+MVRV+Gate":                        gate_oos,
        "Base+MVRV+Tiered+Gate":                 tg_oos,
        "Base+Delta+MVRV+Gate":                  bdmvrv_g_oos,
        "Base+Delta+MVRV+Gate (PS20)":           bdmvrv_ps_oos,
        "Base+Delta+MVRV+Gate (PS20+H90)":       bdmvrv_ps_h90_oos,
    })
    print(sp_df.to_string())

    # --- Fee sensitivity (execution layer only) ---
    print("\n--- Fee Sensitivity (execution layer) ---")
    fee_df = fee_sensitivity_table({
        "Base+MVRV (sigmoid)":               bmvrv_exposure,
        "Base+MVRV+Tiered":                  tiered_exp,
        "Base+MVRV+Gate":                    gate_exp,
        "Base+MVRV+Tiered+Gate":             tg_exp,
        "Base+Delta+MVRV+Gate (PS20)":       bdmvrv_ps_gate,
        "Base+Delta+MVRV+Gate (PS20+H90)":   bdmvrv_ps_h90_gate,
    }, close, fee_levels=[0.0005, 0.0010, 0.0020])
    print(fee_df.to_string())

    # --- Pair stability summary ---
    analyze_signal_stability(
        {}, base_pair_hist, {}, bmvrv_pair_hist,
        s5_hist=bdelta_pair_hist, s6_hist=bdmvrv_pair_hist
    )

    # Optional: retrain frequency study
    if RETRAIN_STUDY:
        run_retrain_frequency_study(combined_df, close, mvrv_raw)

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    plot_strategy_comparison(
        all_bt,
        f"BTC Combined Backtest v5.0  [{EVAL_START} → present]  (10 bps fees)",
        os.path.join(OUT_DIR, "btc_combined_v50.png")
    )
    plot_yearly_heatmap(yearly, os.path.join(OUT_DIR, "btc_combined_yearly_heatmap.png"))
    plot_yearly_bars(yearly,   os.path.join(OUT_DIR, "btc_combined_yearly_bars.png"))

    oos_ternary = ternary_df[ternary_df.index >= EVAL_START]
    oos_close   = close[close.index >= EVAL_START]
    plot_signal_heatmap(oos_ternary, oos_close, n_top=20,
                        out_path=os.path.join(OUT_DIR, "btc_combined_signal_heatmap.png"))

    print(f"\nAll outputs saved to: {OUT_DIR}")
    return raw_df, ternary_df, delta_ternary_df, all_bt, mdf, yearly


if __name__ == "__main__":
    raw_df, ternary_df, delta_ternary_df, all_bt, metrics_df, yearly = main()
