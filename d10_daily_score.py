#!/usr/bin/env python3
"""
D10 Daily Score Exporter
========================
Runs the full D10 v3 pipeline and saves outputs to data/ for the dashboard.
Designed to be called by GitHub Actions once per day.

Outputs:
  data/combo_history.csv   — full time series (date, combo, exposure, btc_price)
  data/signals_today.json  — today's signal breakdown + current score/exposure
  data/metrics.json        — backtest performance metrics
"""

import os, sys, json, importlib.util
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Ensure v3 loads the lib from the same directory
os.environ.setdefault("BTC_LIB_PATH", os.path.join(SCRIPT_DIR, "btc_combined_backtest_lib.py"))

# Load d10_production_backtest_v3 as a module
spec = importlib.util.spec_from_file_location(
    "d10v3", os.path.join(SCRIPT_DIR, "d10_production_backtest_v3.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)


def _json_safe(obj):
    """Recursively make an object JSON-serialisable."""
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


def main():
    print("=" * 60)
    print("D10 Daily Score Exporter")
    print("=" * 60)

    # ── 1. Fetch data ────────────────────────────────────────────
    print("\n[1/5] Fetching data...")
    close, combined, mvrv_raw, sp500 = v3.fetch_data()
    mvrv_aligned = mvrv_raw.reindex(close.index, method="ffill")

    # ── 2. Prepare signals ───────────────────────────────────────
    print("\n[2/5] Preparing signals...")
    disc, fwd = v3.prepare_signals(close, combined, sp500)

    # ── 3. Build ensemble combo ──────────────────────────────────
    print("\n[3/5] Building ensemble combo...")
    combo = v3.build_ensemble_combo(close, disc, fwd, mvrv_aligned)

    # ── 4. Run backtest (needed for exposure series) ─────────────
    print("\n[4/5] Running D10 backtest...")
    exposure = v3.run_d10_backtest(combo, close)

    # ── 5. Compute metrics ───────────────────────────────────────
    print("\n[5/5] Computing metrics...")
    close_eval    = close.loc[v3.EVAL_START:]
    exposure_eval = exposure.loc[v3.EVAL_START:]
    m   = v3.compute_metrics(close_eval, exposure_eval, "D10 v3")
    m_bh = v3.compute_metrics(close_eval, pd.Series(1.0, index=close_eval.index), "BTC B&H")

    # ── Save combo_history.csv ───────────────────────────────────
    history = pd.DataFrame({
        "combo":     combo,
        "exposure":  exposure,
        "btc_price": close,
    }).loc[v3.EVAL_START:]
    history.index.name = "date"
    history.to_csv(os.path.join(DATA_DIR, "combo_history.csv"))
    print(f"  Saved combo_history.csv ({len(history)} rows)")

    # ── Save signals_today.json ──────────────────────────────────
    today          = disc.index[-1]
    signals_values = disc.loc[today].to_dict()

    # Categorise signals
    def _category(name):
        tech_prefixes = ("MABreak_", "Zscore_", "GapPct_", "SP500_")
        macro_keywords = ("FedBS", "M2_", "RRP_", "FFR_", "CPI_", "PPI_",
                          "Claims_", "SOFR", "IORB", "NASDAQ", "BTC_vs_", "Gold")
        if any(name.startswith(p) for p in tech_prefixes):
            return "Technical"
        if any(k in name for k in macro_keywords):
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
            "bullish": int((disc.loc[today] == -1).sum()),  # combo<0 → bullish
            "bearish": int((disc.loc[today] ==  1).sum()),
            "neutral": int((disc.loc[today] ==  0).sum()),
        },
    }
    with open(os.path.join(DATA_DIR, "signals_today.json"), "w") as f:
        json.dump(_json_safe(signals_today), f, indent=2)
    print(f"  Saved signals_today.json (date={today.date()})")

    # ── Save metrics.json ────────────────────────────────────────
    def _clean_metrics(r):
        out = {k: v for k, v in r.items() if k != "name"}
        out["yearly"] = {str(yr): float(ret) for yr, ret in out["yearly"].items()}
        return _json_safe(out)

    metrics = {
        "d10": _clean_metrics(m),
        "bh":  _clean_metrics(m_bh),
        "generated": str(today.date()),
    }
    with open(os.path.join(DATA_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics.json")

    # ── Summary ──────────────────────────────────────────────────
    exp_pct = exposure.iloc[-1] * 100
    print(f"\nToday ({today.date()})")
    print(f"  BTC Price:    ${close.iloc[-1]:,.0f}")
    print(f"  Combo Score:  {combo.iloc[-1]:.4f}")
    print(f"  BTC Exposure: {exp_pct:.0f}%")
    label = "BULLISH" if combo.iloc[-1] < 0 else "BEARISH" if combo.iloc[-1] > 0 else "NEUTRAL"
    print(f"  Signal:       {label}")
    print(f"\nAll data saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()