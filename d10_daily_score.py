#!/usr/bin/env python3
"""
D10 Daily Score Exporter
========================
Runs the D10 v3 pipeline and saves outputs to data/ for the dashboard.

Fast mode (default): uses cached combo history and only recomputes the
current quarter (~3 pair-selection iterations instead of ~63).

Full mode: FORCE_FULL=1 python d10_daily_score.py

Outputs:
  data/combo_raw.csv       — full combo history cache (internal)
  data/combo_history.csv   — date, combo, exposure, btc_price (for dashboard)
  data/signals_today.json  — today's signal breakdown + score/exposure
  data/metrics.json        — backtest performance metrics
"""

import os, sys, json, importlib.util, pickle
from itertools import combinations
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

COMBO_RAW_FILE = os.path.join(DATA_DIR, "combo_raw.csv")
FORCE_FULL     = os.environ.get("FORCE_FULL", "0") == "1"

# Ensure v3 loads the lib from the same directory
os.environ.setdefault(
    "BTC_LIB_PATH",
    os.path.join(SCRIPT_DIR, "btc_combined_backtest_lib.py")
)

# Load d10_production_backtest_v3 as a module
spec = importlib.util.spec_from_file_location(
    "d10v3", os.path.join(SCRIPT_DIR, "d10_production_backtest_v3.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)


# ── Incremental combo builder ─────────────────────────────────────────────────

def _build_combo_from(close, disc, fwd, mvrv_aligned, rebal_dates, start_from):
    """
    Build ensemble combo for rebalance periods >= start_from only.
    Returns a Series with NaN for periods before start_from.
    """
    sigs_all = list(disc.columns)
    fwd_90   = close.pct_change(90).shift(-90)
    combo_parts = {}

    for ty in v3.ENSEMBLE_WINDOWS:
        combo = pd.Series(np.nan, index=close.index)
        v3._mod.HORIZON_WEIGHTS.clear()
        v3._mod.HORIZON_WEIGHTS.update(v3.WEIGHT_SCHEMES["uniform"])

        for i, rd in enumerate(rebal_dates):
            if rd < start_from:
                continue
            train_start = rd - pd.DateOffset(years=ty)
            if train_start < pd.Timestamp(v3.TRAIN_START):
                continue
            next_rd = (rebal_dates[i + 1]
                       if i + 1 < len(rebal_dates)
                       else close.index[-1] + pd.Timedelta(days=1))

            in_train  = (disc.index >= train_start) & (disc.index < rd)
            disc_tr   = disc[in_train]
            fwd_tr    = fwd[in_train]
            mvrv_tr   = mvrv_aligned[in_train]

            act         = {s: (disc_tr[s] != 0).mean() for s in sigs_all}
            active_sigs = [s for s, a in act.items() if a > v3.ACTIVITY_THRESHOLD]
            pairs_all   = list(combinations(active_sigs, 2))

            mv      = float(mvrv_aligned.asof(rd)) if not pd.isna(mvrv_aligned.asof(rd)) else np.nan
            regime  = v3._mod._get_regime(mv)
            n_same  = int((mvrv_tr.apply(lambda x: v3._mod._get_regime(x) == regime)).sum())
            weights = (v3._mod._regime_weights(mvrv_tr, regime)
                       if n_same >= v3._mod.MIN_REGIME_TRAIN_DAYS else None)

            ho_pairs, val_dates = v3.select_pairs_holdout(
                pairs_all, disc_tr, fwd_tr, rd, weights
            )

            best_scheme, best_ic = "uniform", -999
            for sname, sw in v3.WEIGHT_SCHEMES.items():
                ic = v3.validate_weight_scheme(disc, fwd, ho_pairs, val_dates, sw, close, fwd_90)
                if ic > best_ic:
                    best_ic, best_scheme = ic, sname
            best_weights = v3.WEIGHT_SCHEMES[best_scheme]

            oos_dates = close.index[(close.index >= rd) & (close.index < next_rd)]
            for t in oos_dates:
                combo.loc[t] = v3.score_with_weights(disc, fwd, ho_pairs, t, best_weights)

            print(f"    [{rd.date()}] {ty}Y  pairs={len(ho_pairs)}, wt={best_scheme} (IC={best_ic:.3f})")

        combo_parts[ty] = combo

    return sum(combo_parts.values()) / len(combo_parts)


def build_combo_smart(close, disc, fwd, mvrv_aligned):
    """
    Build ensemble combo using cache when possible.
    Only recomputes the current quarter (fast mode) unless forced or no cache.
    """
    rebal_dates = list(
        pd.date_range(pd.Timestamp(v3.EVAL_START), close.index[-1], freq="3MS")
    )
    past_rebals = [rd for rd in rebal_dates if rd <= close.index[-1]]
    current_rebal = past_rebals[-1]

    # Try to load cached combo
    cached = None
    if not FORCE_FULL and os.path.exists(COMBO_RAW_FILE):
        try:
            cached = pd.read_csv(COMBO_RAW_FILE, index_col=0, parse_dates=True)["combo"]
            cached = cached.reindex(close.index)
        except Exception as e:
            print(f"  Cache load failed ({e}), running full computation")
            cached = None

    if cached is not None:
        # Check cache has data up to (just before) current rebalance period
        cache_before_rebal = cached[cached.index < current_rebal].dropna()
        if len(cache_before_rebal) > 100:
            print(f"  Fast mode: recomputing only current quarter from {current_rebal.date()}")
            new_part = _build_combo_from(close, disc, fwd, mvrv_aligned, rebal_dates, current_rebal)
            # Merge: keep cache for pre-current-rebal, use new for current-rebal onwards
            combo = cached.copy().reindex(close.index).fillna(0.0)
            mask = new_part.notna()
            combo[mask] = new_part[mask]
            combo = combo.fillna(0.0)
        else:
            print("  Cache incomplete, running full computation...")
            combo = v3.build_ensemble_combo(close, disc, fwd, mvrv_aligned)
    else:
        print("  No cache, running full computation...")
        combo = v3.build_ensemble_combo(close, disc, fwd, mvrv_aligned)

    # Save updated combo cache
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
    print(f"D10 Daily Score Exporter  [{mode}]")
    print("=" * 60)

    # 1. Fetch data
    print("\n[1/5] Fetching data...")
    close, combined, mvrv_raw, sp500 = v3.fetch_data()
    mvrv_aligned = mvrv_raw.reindex(close.index, method="ffill")

    # 2. Prepare signals
    print("\n[2/5] Preparing signals...")
    disc, fwd = v3.prepare_signals(close, combined, sp500)

    # 3. Build ensemble combo (fast or full)
    print("\n[3/5] Building ensemble combo...")
    combo = build_combo_smart(close, disc, fwd, mvrv_aligned)

    # 4. Run backtest (needed for exposure)
    print("\n[4/5] Running D10 backtest...")
    exposure = v3.run_d10_backtest(combo, close)

    # 5. Compute metrics
    print("\n[5/5] Computing metrics...")
    close_eval    = close.loc[v3.EVAL_START:]
    exposure_eval = exposure.loc[v3.EVAL_START:]
    m    = v3.compute_metrics(close_eval, exposure_eval, "D10 v3")
    m_bh = v3.compute_metrics(
        close_eval, pd.Series(1.0, index=close_eval.index), "BTC B&H"
    )

    # ── Save combo_history.csv ────────────────────────────────────
    history = pd.DataFrame({
        "combo":     combo,
        "exposure":  exposure,
        "btc_price": close,
    }).loc[v3.EVAL_START:]
    history.index.name = "date"
    history.to_csv(os.path.join(DATA_DIR, "combo_history.csv"))
    print(f"  Saved combo_history.csv ({len(history)} rows)")

    # ── Save signals_today.json ───────────────────────────────────
    today          = disc.index[-1]
    signals_values = disc.loc[today].to_dict()

    def _category(name):
        tech = ("MABreak_", "Zscore_", "GapPct_", "SP500_")
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
    def _clean(r):
        out = {k: v for k, v in r.items() if k != "name"}
        out["yearly"] = {str(yr): float(ret) for yr, ret in out["yearly"].items()}
        return _json_safe(out)

    with open(os.path.join(DATA_DIR, "metrics.json"), "w") as f:
        json.dump({"d10": _clean(m), "bh": _clean(m_bh),
                   "generated": str(today.date())}, f, indent=2)
    print("  Saved metrics.json")

    # ── Summary ───────────────────────────────────────────────────
    label = "BULLISH" if combo.iloc[-1] < 0 else "BEARISH" if combo.iloc[-1] > 0 else "NEUTRAL"
    print(f"\nToday ({today.date()})")
    print(f"  BTC Price:    ${close.iloc[-1]:,.0f}")
    print(f"  Combo Score:  {combo.iloc[-1]:.4f}")
    print(f"  BTC Exposure: {exposure.iloc[-1]*100:.0f}%")
    print(f"  Signal:       {label}")
    print(f"\nAll data saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()