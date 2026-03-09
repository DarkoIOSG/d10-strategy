#!/usr/bin/env python3
"""
D10 Daily Score Exporter (v2)
==============================
Runs the D10 v2 pipeline and saves outputs to data/ for the dashboard.

Fast mode (default): uses cached combo history and only recomputes the
current quarter (~2 pair-selection iterations instead of ~42).

Full mode: FORCE_FULL=1 python d10_daily_score.py

Outputs:
  data/combo_raw.csv        — full combo history cache (internal)
  data/combo_history.csv    — date, combo, exposure, btc_price (for dashboard)
  data/signals_today.json   — today's signal breakdown + score/exposure
  data/signals_history.csv  — historical per-signal values (for Signal Explorer)
  data/metrics.json         — backtest performance metrics
"""

import os, sys, json, importlib.util
from itertools import combinations
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

COMBO_RAW_FILE = os.path.join(DATA_DIR, "combo_raw.csv")
FORCE_FULL     = os.environ.get("FORCE_FULL", "0") == "1"

# Load d10_production_v2_audited as a module
spec = importlib.util.spec_from_file_location(
    "d10v2", os.path.join(SCRIPT_DIR, "d10_production_v2_audited.py")
)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

# Override API keys from environment if provided
if os.environ.get("CRYPTOQUANT_KEY"):
    v2.CQ_KEY = os.environ["CRYPTOQUANT_KEY"]
if os.environ.get("FRED_API_KEY"):
    v2.FRED_KEY = os.environ["FRED_API_KEY"]

import yfinance as yf


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_data():
    """Fetch all market data needed for the v2 pipeline."""
    close = yf.download("BTC-USD", start=v2.TRAIN_START, progress=False)["Close"].squeeze().dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)

    def _dl(ticker):
        try:
            r = yf.download(ticker, start=v2.TRAIN_START, progress=False)["Close"].squeeze().dropna()
            r.index = pd.to_datetime(r.index).tz_localize(None)
            return r.reindex(close.index, method="ffill")
        except:
            return pd.Series(np.nan, index=close.index)

    sp500 = _dl("^GSPC"); vix = _dl("^VIX"); dxy = _dl("DX-Y.NYB")
    gold  = _dl("GC=F");  hy  = _dl("HYG")

    cq  = v2._mod.fetch_cquant_signals(v2.CQ_KEY, start=v2.TRAIN_START)
    prx = v2._mod.compute_proxy_signals(close, sp500, vix, dxy, gold, hy)
    raw_df = pd.DataFrame({**cq, **prx}).reindex(close.index).ffill()

    ter = v2._mod.compute_ternary_matrix(raw_df)
    tsr = v2._mod.compute_technical_signals(close, sp500, v2._mod.build_technical_signal_registry())
    combined = pd.concat([ter, tsr], axis=1).reindex(close.index).fillna(0)

    mvrv_raw     = raw_df.get("MVRV")
    mvrv_aligned = mvrv_raw.reindex(close.index, method="ffill")
    disc = v2._mod.discretize_signals(combined)
    fwd  = v2._mod.compute_forward_returns(close)

    return close, disc, fwd, mvrv_aligned


def add_fred_signals(disc, close):
    """Add 4 FRED macro signals to the disc DataFrame."""
    from fredapi import Fred
    fred = Fred(api_key=v2.FRED_KEY)

    def fetch_fred(sid):
        s = fred.get_series(sid, observation_start=v2.TRAIN_START)
        return s.dropna().reindex(close.index, method="ffill")

    ffr  = fetch_fred("FEDFUNDS")
    ppi  = fetch_fred("PPIACO")
    cpi  = fetch_fred("CPIAUCSL")
    sofr = fetch_fred("SOFR")
    iorb = fetch_fred("IORB")
    rrp  = fetch_fred("RRPONTSYD")

    disc["FFR_vs_MA365"]          = np.sign(ffr - ffr.rolling(365).mean()).reindex(disc.index).fillna(0).clip(-1, 1).astype(int)
    disc["SOFR_IORB_Spread_Sign"] = np.sign(sofr - iorb).reindex(disc.index).fillna(0).clip(-1, 1).astype(int)
    disc["RRP_Trend_90d"]         = np.sign(rrp - rrp.rolling(90).mean()).reindex(disc.index).fillna(0).clip(-1, 1).astype(int)
    ppi_yoy = ppi.pct_change(252); cpi_yoy = cpi.pct_change(252)
    disc["PPI_vs_CPI"]            = np.sign(ppi_yoy - cpi_yoy).reindex(disc.index).fillna(0).clip(-1, 1).astype(int)

    return disc


# ── Full metrics computation ──────────────────────────────────────────────────

def compute_full_metrics(close_eval, exposure_eval, label):
    """Compute full metrics dict suitable for metrics.json / dashboard."""
    exp = exposure_eval.reindex(close_eval.index).ffill()
    bt  = v2._mod.backtest_from_exposure(close_eval, exp, label)
    r   = bt["port_ret"]

    tr     = (1 + r).prod() - 1
    ny     = len(r) / 365.25
    ar     = (1 + tr) ** (1 / ny) - 1
    av     = r.std() * np.sqrt(365)
    sharpe = ar / av if av > 0 else 0

    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()

    ho_r      = r[r.index.year.isin([2024, 2025])]
    ho_tr     = (1 + ho_r).prod() - 1
    ho_ny     = len(ho_r) / 365.25
    ho_ar     = (1 + ho_tr) ** (1 / ho_ny) - 1
    ho_av     = ho_r.std() * np.sqrt(365)
    ho_sharpe = ho_ar / ho_av if ho_av > 0 else 0

    yearly = {
        str(yr): float((1 + r[r.index.year == yr]).prod() - 1)
        for yr in sorted(r.index.year.unique())
    }

    fees_pct_yr = float(bt["fee_drag"].sum() / ny)

    return {
        "sharpe":       float(sharpe),
        "ho_sharpe":    float(ho_sharpe),
        "ann_ret":      float(ar),
        "ann_vol":      float(av),
        "max_dd":       float(mdd),
        "avg_exposure": float(exp.mean()),
        "fees":         float(fees_pct_yr * 10_000),  # in bps/yr
        "yearly":       yearly,
    }


# ── Incremental combo builder ─────────────────────────────────────────────────

def _build_combo_from(close, disc, fwd, mvrv_aligned, rebal_dates, start_from):
    """
    Build 2Y+3Y ensemble combo for rebalance periods >= start_from only.
    Returns a Series with NaN for periods before start_from.
    """
    sigs_all    = list(disc.columns)
    combo_parts = {}

    for ty in [2, 3]:
        combo = pd.Series(np.nan, index=close.index)
        for i, rd in enumerate(rebal_dates):
            if rd < start_from:
                continue
            train_start = rd - pd.DateOffset(years=ty)
            if train_start < pd.Timestamp(v2.TRAIN_START):
                continue
            next_rd = (rebal_dates[i + 1]
                       if i + 1 < len(rebal_dates)
                       else close.index[-1] + pd.Timedelta(days=1))

            in_train = (disc.index >= train_start) & (disc.index < rd)
            disc_tr  = disc[in_train]
            fwd_tr   = fwd[in_train]
            mvrv_tr  = mvrv_aligned[in_train]

            act         = {s: (disc_tr[s] != 0).mean() for s in sigs_all}
            active_sigs = [s for s, a in act.items() if a > v2.ACTIVITY_THRESHOLD]
            pairs_all   = list(combinations(active_sigs, 2))

            mv      = float(mvrv_aligned.asof(rd)) if not pd.isna(mvrv_aligned.asof(rd)) else np.nan
            regime  = v2._mod._get_regime(mv)
            n_same  = int((mvrv_tr.apply(lambda x: v2._mod._get_regime(x) == regime)).sum())
            weights = (v2._mod._regime_weights(mvrv_tr, regime)
                       if n_same >= v2._mod.MIN_REGIME_TRAIN_DAYS else None)

            top_pairs = v2.select_pairs_holdout(pairs_all, disc_tr, fwd_tr, rd, weights)

            oos_dates = close.index[(close.index >= rd) & (close.index < next_rd)]
            for t in oos_dates:
                combo.loc[t] = v2._mod.score_at_date(disc, fwd, top_pairs, t)

            print(f"    [{rd.date()}] {ty}Y  pairs={len(top_pairs)}")

        combo_parts[ty] = combo

    return sum(combo_parts.values()) / len(combo_parts)


def _build_full_combo(close, disc, fwd, mvrv_aligned):
    """Build full 2Y+3Y ensemble combo across all rebalance dates."""
    rebal_dates = list(
        pd.date_range(pd.Timestamp(v2.EVAL_START), close.index[-1], freq="3MS")
    )
    sigs_all    = list(disc.columns)
    combo_parts = {}

    for ty in [2, 3]:
        combo = pd.Series(0.0, index=close.index)
        print(f"  Building {ty}Y window...")
        for i, rd in enumerate(rebal_dates):
            train_start = rd - pd.DateOffset(years=ty)
            next_rd     = (rebal_dates[i + 1]
                           if i + 1 < len(rebal_dates)
                           else close.index[-1] + pd.Timedelta(days=1))

            in_train = (disc.index >= train_start) & (disc.index < rd)
            disc_tr  = disc[in_train]; fwd_tr = fwd[in_train]
            mvrv_tr  = mvrv_aligned[in_train]

            act         = {s: (disc_tr[s] != 0).mean() for s in sigs_all}
            active_sigs = [s for s, a in act.items() if a > v2.ACTIVITY_THRESHOLD]
            pairs_all   = list(combinations(active_sigs, 2))

            mv      = float(mvrv_aligned.asof(rd)) if not pd.isna(mvrv_aligned.asof(rd)) else np.nan
            regime  = v2._mod._get_regime(mv)
            n_same  = int((mvrv_tr.apply(lambda x: v2._mod._get_regime(x) == regime)).sum())
            weights = (v2._mod._regime_weights(mvrv_tr, regime)
                       if n_same >= v2._mod.MIN_REGIME_TRAIN_DAYS else None)

            top_pairs = v2.select_pairs_holdout(pairs_all, disc_tr, fwd_tr, rd, weights)

            oos_dates = close.index[(close.index >= rd) & (close.index < next_rd)]
            for t in oos_dates:
                combo.loc[t] = v2._mod.score_at_date(disc, fwd, top_pairs, t)

            if i % 4 == 0:
                print(f"    [{rd.date()}] pairs={len(top_pairs)}, n_sigs={len(active_sigs)}")

        combo_parts[ty] = combo

    return sum(combo_parts.values()) / len(combo_parts)


def build_combo_smart(close, disc, fwd, mvrv_aligned):
    """
    Build ensemble combo using cache when possible.
    Only recomputes the current quarter (fast mode) unless forced or no cache.
    """
    rebal_dates   = list(
        pd.date_range(pd.Timestamp(v2.EVAL_START), close.index[-1], freq="3MS")
    )
    past_rebals   = [rd for rd in rebal_dates if rd <= close.index[-1]]
    current_rebal = past_rebals[-1]

    cached = None
    if not FORCE_FULL and os.path.exists(COMBO_RAW_FILE):
        try:
            cached = pd.read_csv(COMBO_RAW_FILE, index_col=0, parse_dates=True)["combo"]
            cached = cached.reindex(close.index)
        except Exception as e:
            print(f"  Cache load failed ({e}), running full computation")
            cached = None

    if cached is not None:
        cache_before_rebal = cached[cached.index < current_rebal].dropna()
        if len(cache_before_rebal) > 100:
            print(f"  Fast mode: recomputing only current quarter from {current_rebal.date()}")
            new_part = _build_combo_from(close, disc, fwd, mvrv_aligned, rebal_dates, current_rebal)
            combo = cached.copy().reindex(close.index).fillna(0.0)
            mask  = new_part.notna()
            combo[mask] = new_part[mask]
            combo = combo.fillna(0.0)
        else:
            print("  Cache incomplete, running full computation...")
            combo = _build_full_combo(close, disc, fwd, mvrv_aligned)
    else:
        print("  No cache, running full computation...")
        combo = _build_full_combo(close, disc, fwd, mvrv_aligned)

    pd.DataFrame({"combo": combo}).to_csv(COMBO_RAW_FILE)
    return combo


# ── JSON serialisation helper ─────────────────────────────────────────────────

def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mode = "FULL" if FORCE_FULL else "FAST (incremental)"
    print("=" * 60)
    print(f"D10 Daily Score Exporter v2  [{mode}]")
    print("=" * 60)

    # 1. Fetch market data
    print("\n[1/5] Fetching data...")
    close, disc, fwd, mvrv_aligned = fetch_data()

    # 2. Add FRED macro signals
    print("\n[2/5] Adding FRED macro signals...")
    disc = add_fred_signals(disc, close)
    print(f"  Total signals: {disc.shape[1]}")

    # 3. Build ensemble combo (fast or full)
    print("\n[3/5] Building ensemble combo...")
    combo = build_combo_smart(close, disc, fwd, mvrv_aligned)

    # 4. Run D10 backtest
    print("\n[4/5] Running D10 backtest...")
    exposure = v2.run_d10_backtest(combo, close)

    # 5. Compute metrics
    print("\n[5/5] Computing metrics...")
    close_eval    = close.loc[v2.EVAL_START:]
    exposure_eval = exposure.loc[v2.EVAL_START:]
    m_d10 = compute_full_metrics(close_eval, exposure_eval, "D10 v2")
    m_bh  = compute_full_metrics(close_eval, pd.Series(1.0, index=close_eval.index), "BTC B&H")

    # ── Save combo_history.csv ────────────────────────────────────
    history = pd.DataFrame({
        "combo":     combo,
        "exposure":  exposure,
        "btc_price": close,
    }).loc[v2.EVAL_START:]
    history.index.name = "date"
    history.to_csv(os.path.join(DATA_DIR, "combo_history.csv"))
    print(f"  Saved combo_history.csv ({len(history)} rows)")

    # ── Save signals_history.csv ──────────────────────────────────
    sig_hist = disc.loc[v2.EVAL_START:]
    sig_hist.index.name = "date"
    sig_hist.to_csv(os.path.join(DATA_DIR, "signals_history.csv"))
    print(f"  Saved signals_history.csv ({len(sig_hist)} rows × {sig_hist.shape[1]} signals)")

    # ── Save signals_today.json ───────────────────────────────────
    today          = disc.index[-1]
    signals_values = disc.loc[today].to_dict()

    def _category(name):
        tech  = ("MABreak_", "Zscore_", "GapPct_", "SP500_")
        macro = ("FedBS", "M2_", "RRP_", "FFR_", "CPI_", "PPI_",
                 "Claims_", "SOFR", "IORB", "NASDAQ", "BTC_vs_", "Gold")
        if any(name.startswith(p) for p in tech):
            return "Technical"
        if any(k in name for k in macro):
            return "Macro"
        return "On-chain"

    signals_today = {
        "date":        str(today.date()),
        "combo_score": float(combo.iloc[-1]),
        "exposure":    float(exposure.iloc[-1]),
        "btc_price":   float(close.iloc[-1]),
        "signals": {
            k: {"value": int(v), "category": _category(k)}
            for k, v in signals_values.items()
        },
        "signal_counts": {
            "bullish": int((disc.loc[today] == -1).sum()),
            "bearish": int((disc.loc[today] ==  1).sum()),
            "neutral": int((disc.loc[today] ==  0).sum()),
        },
    }
    with open(os.path.join(DATA_DIR, "signals_today.json"), "w") as f:
        json.dump(_json_safe(signals_today), f, indent=2)
    print(f"  Saved signals_today.json (date={today.date()})")

    # ── Save metrics.json ─────────────────────────────────────────
    with open(os.path.join(DATA_DIR, "metrics.json"), "w") as f:
        json.dump(_json_safe({
            "d10":       m_d10,
            "bh":        m_bh,
            "generated": str(today.date()),
        }), f, indent=2)
    print("  Saved metrics.json")

    # ── Summary ───────────────────────────────────────────────────
    label = "BULLISH" if combo.iloc[-1] < 0 else "BEARISH" if combo.iloc[-1] > 0 else "NEUTRAL"
    print(f"\nToday ({today.date()})")
    print(f"  BTC Price:    ${close.iloc[-1]:,.0f}")
    print(f"  Combo Score:  {combo.iloc[-1]:.4f}")
    print(f"  BTC Exposure: {exposure.iloc[-1]*100:.0f}%")
    print(f"  Signal:       {label}")
    print(f"  Ann Return:   {m_d10['ann_ret']:+.1%} (Strategy) vs {m_bh['ann_ret']:+.1%} (B&H)")
    print(f"\nAll data saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()