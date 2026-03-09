#!/usr/bin/env python3
"""
D10 Production Strategy v2 (Audited)
=====================================
59 signals (55 base + 4 macro) + HO pair validation (T3)
Verified metrics (Mar 9 2026):
  Sharpe=1.2771, HO=1.0833, MaxDD=-51.1%, AvgExp=74.1%

Changes from v1 audited:
  1. +4 macro signals: FFR_vs_MA365, SOFR_IORB_Spread_Sign, RRP_Trend_90d, PPI_vs_CPI
  2. select_pairs_holdout (verbatim from d10_production_backtest_v3.py lines 290-337)
  3. Uniform weights (no HO weight selection)

Code provenance:
  - select_pairs_holdout: VERBATIM copy from d10_production_backtest_v3.py
  - All other functions: VERBATIM from v1 audited pipeline / t3_plus_4macro.py
  - MABreak fix: same as v1 audited
  - Library: btc_combined_backtest_v1_audited.py (read-only)
"""
import sys, os, math, warnings, importlib.util
import numpy as np, pandas as pd
from scipy.stats import binomtest
from itertools import combinations
warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────
TRAIN_START = "2014-01-01"
EVAL_START  = "2020-01-01"
PRIMARY_HORIZON  = 90
DELTA            = 0.10
N_THRESHOLDS     = 5
COOLDOWN_DAYS    = 7
TOP_N_PCT        = 0.01
MIN_PAIRS        = 5
SIG_FLOOR_SIGMA  = 1.0
ACTIVITY_THRESHOLD = 0.05
TX_COST_BPS      = 10
GATE_THRESHOLD_LOOKBACK = 3
HELD_OUT_SPLIT   = 0.70
CQ_KEY   = "9XqAcsD2L69JiFZyPIjB9rvlobokQA9yquYUi6R5"
FRED_KEY = "786fe852241222b4df0e26f5416e4007"

# ── Load audited library ────────────────────────────────────────────
LIB_PATH = os.path.join(os.path.dirname(__file__), "btc_combined_backtest_lib.py")
spec = importlib.util.spec_from_file_location("bbt", LIB_PATH)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
_mod.HORIZON_WEIGHTS.clear()
_mod.HORIZON_WEIGHTS.update({3: 1/6, 7: 1/6, 14: 1/6, 30: 1/6, 90: 1/6, 180: 1/6})

# MABreak fix (v1 audited)
def _patched_find_events(self, brk_df, close, bit, dirn, holding, before, horizons):
    m = ((brk_df["bit"] == bit) & (brk_df["direction"] == dirn) & (brk_df["date"] < before))
    cands = brk_df[m].copy()
    if cands.empty: return pd.DataFrame()
    for k in self._related_keys(bit, list(holding.keys())):
        if cands.empty: break
        if k in holding and k != bit:
            try: cands = cands[cands["holding"].apply(lambda h: isinstance(h, dict) and h.get(k) == holding[k])]
            except Exception: break
    if cands.empty: return pd.DataFrame()
    clustered, ld = [], None
    for _, r in cands.iterrows():
        if ld is None or (r["date"] - ld).days >= _mod.CLUSTER_GAP: clustered.append(r); ld = r["date"]
    cands = pd.DataFrame(clustered)
    if cands.empty: return pd.DataFrame()
    evts = []
    for _, r in cands.iterrows():
        d = r["date"]
        if d not in close.index: continue
        p0 = close.loc[d]
        rec = {"date": d, "bit": bit, "direction": dirn, "price": p0}
        for h in horizons:
            fi = close.index.searchsorted(d) + h
            if fi < len(close) and close.index[fi] < before: rec[f"fwd_{h}d"] = close.iloc[fi] / p0 - 1
            else: rec[f"fwd_{h}d"] = np.nan
        evts.append(rec)
    return pd.DataFrame(evts)
_mod.MABreakSignal._find_events = _patched_find_events

import yfinance as yf

# ══════════════════════════════════════════════════════════════════════
# VERBATIM from d10_production_backtest_v3.py lines 290-337
# ══════════════════════════════════════════════════════════════════════
def select_pairs_holdout(pairs, disc_tr, fwd_tr, rd, weights):
    n = len(disc_tr)
    split = int(n * HELD_OUT_SPLIT)
    dt_tr = disc_tr.iloc[:split]
    ft_tr = fwd_tr.iloc[:split]
    dt_vl = disc_tr.iloc[split:]
    ft_vl = fwd_tr.iloc[split:]

    # Activity filter on train portion
    active = [s for s1, s2 in pairs for s in [s1, s2]]
    active = list(set(active))
    active = [s for s in active if s in dt_tr.columns and (dt_tr[s] != 0).mean() > ACTIVITY_THRESHOLD]
    tr_pairs = [(s1, s2) for s1, s2 in pairs if s1 in active and s2 in active]

    if len(tr_pairs) < MIN_PAIRS:
        meta = {p: _mod.pair_power(disc_tr, fwd_tr, p[0], p[1], rd,
                weights=weights, primary_horizon=PRIMARY_HORIZON) for p in pairs}
        ranked = sorted(meta.items(), key=lambda x: -x[1])
        return [p for p, _ in ranked[:MIN_PAIRS]]

    meta = {p: _mod.pair_power(dt_tr, ft_tr, p[0], p[1], rd,
            weights=weights, primary_horizon=PRIMARY_HORIZON) for p in tr_pairs}
    ranked = sorted(meta.items(), key=lambda x: -x[1])
    powers = np.array([s for _, s in ranked])
    floor = np.mean(powers) + SIG_FLOOR_SIGMA * np.std(powers)
    top_n = max(math.ceil(len(tr_pairs) * TOP_N_PCT), MIN_PAIRS)

    cands = []
    for p, sc in ranked:
        if sc < floor:
            break
        cands.append(p)
        if len(cands) >= top_n:
            break
    if len(cands) < MIN_PAIRS:
        cands = [p for p, _ in ranked[:MIN_PAIRS]]

    validated = []
    for s1, s2 in cands:
        if s1 in dt_vl.columns and s2 in dt_vl.columns:
            vp = _mod.pair_power(dt_vl, ft_vl, s1, s2, rd,
                                 weights=weights, primary_horizon=PRIMARY_HORIZON)
            if vp > 0:
                validated.append((s1, s2))
    if len(validated) < MIN_PAIRS:
        validated = cands[:MIN_PAIRS]

    return validated
# ══════════════════════════════════════════════════════════════════════

# ── Backtest functions (v1 audited verbatim) ─────────────────────────
def find_gate_holm(combo_trail, close_trail, horizon=PRIMARY_HORIZON, alpha=0.05):
    cn = combo_trail[combo_trail != 0].dropna()
    if len(cn) < 60: return 0.0
    fwd_r = close_trail.pct_change(horizon).shift(-horizon)
    com = cn.index.intersection(fwd_r.dropna().index)
    if len(com) < 30: return 0.0
    c = cn.loc[com]; f = fwd_r.loc[com]
    hit = ((c < 0) & (f > 0)) | ((c > 0) & (f < 0))
    for step, pct in enumerate([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], 1):
        thr = 0.0 if pct == 0 else np.percentile(c.abs(), pct)
        mask = c.abs() >= thr if pct > 0 else pd.Series(True, index=c.index)
        n = mask.sum()
        if n < 20: continue
        k = int(hit[mask].sum())
        if binomtest(k, n, 0.5, alternative='greater').pvalue < alpha / step: return thr
    return float(np.percentile(c.abs(), 50))

def apply_cooldown(raw_exp, cd=COOLDOWN_DAYS):
    res = raw_exp.copy(); prev = raw_exp.iloc[0]; last_dir = 0; lcd = raw_exp.index[0]
    for i, (dt, tgt) in enumerate(raw_exp.items()):
        if i == 0: res[dt] = tgt; prev = tgt; continue
        d = tgt - prev
        if abs(d) < 0.001: res[dt] = prev; continue
        dirn = 1 if d > 0 else -1
        if last_dir != 0 and dirn != last_dir and (dt - lcd).days < cd: res[dt] = prev
        else: res[dt] = tgt; prev = tgt; lcd = dt; last_dir = dirn
    return res

def run_d10_backtest(combo, close):
    rebal_dates = pd.date_range(pd.Timestamp(EVAL_START), close.index[-1], freq="3MS")
    ty = GATE_THRESHOLD_LOOKBACK
    gates = {}
    for rd in rebal_dates:
        ct = combo[(combo.index >= rd - pd.DateOffset(years=ty)) & (combo.index < rd)]
        cl = close[(close.index >= rd - pd.DateOffset(years=ty)) & (close.index < rd)]
        gates[rd] = find_gate_holm(ct, cl)
    exposure = pd.Series(0.5, index=combo.index); prev = 0.5
    for qi, rd in enumerate(rebal_dates):
        next_rd = rebal_dates[qi + 1] if qi + 1 < len(rebal_dates) else close.index[-1] + pd.Timedelta(days=1)
        hist_nz = combo[(combo.index >= rd - pd.DateOffset(years=ty)) & (combo.index < rd)]
        hist_nz = hist_nz[hist_nz != 0].abs().dropna()
        if len(hist_nz) >= 30:
            thresholds = [np.percentile(hist_nz, 100 * i / (N_THRESHOLDS + 1)) for i in range(1, N_THRESHOLDS + 1)]
        else:
            thresholds = [(i + 1) / (N_THRESHOLDS + 1) for i in range(N_THRESHOLDS)]
        gate = gates.get(rd, 0.0)
        for t in combo.index[(combo.index >= rd) & (combo.index < next_rd)]:
            c = combo.loc[t]
            if np.isnan(c) or abs(c) < gate: exposure.loc[t] = prev; continue
            n_above = sum(abs(c) >= th for th in thresholds)
            if c < 0: prev = min(0.5 + n_above * DELTA, 1.0)
            else: prev = max(0.5 - n_above * DELTA, 0.0)
            exposure.loc[t] = prev
    return apply_cooldown(exposure)

def compute_metrics(close_eval, exposure):
    exp = exposure.reindex(close_eval.index).ffill()
    bt = _mod.backtest_from_exposure(close_eval, exp, "v2_prod")
    r = bt['port_ret']
    tr = (1 + r).prod() - 1; ny = len(r) / 365.25
    ar = (1 + tr) ** (1 / ny) - 1; av = r.std() * np.sqrt(365)
    sharpe = ar / av if av > 0 else 0
    cum = (1 + r).cumprod(); mdd = (cum / cum.cummax() - 1).min()
    ho_r = r[r.index.year.isin([2024, 2025])]
    ho_tr = (1 + ho_r).prod() - 1; ho_ny = len(ho_r) / 365.25
    ho_ar = (1 + ho_tr) ** (1 / ho_ny) - 1; ho_av = ho_r.std() * np.sqrt(365)
    ho_sharpe = ho_ar / ho_av if ho_av > 0 else 0
    yearly = {yr: (1 + r[r.index.year == yr]).prod() - 1 for yr in sorted(r.index.year.unique())}
    return sharpe, ho_sharpe, ar, mdd, exp.mean(), yearly

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("D10 PRODUCTION v2 — 59 signals + T3 holdout pair validation")
    print("=" * 70)

    # ── Fetch data ───────────────────────────────────────────────────
    print("\n[1] Fetching BTC + market data...")
    close = yf.download("BTC-USD", start=TRAIN_START, progress=False)["Close"].squeeze().dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    def _dl(ticker):
        try:
            r = yf.download(ticker, start=TRAIN_START, progress=False)["Close"].squeeze().dropna()
            r.index = pd.to_datetime(r.index).tz_localize(None)
            return r.reindex(close.index, method="ffill")
        except: return pd.Series(np.nan, index=close.index)
    sp500 = _dl("^GSPC"); vix = _dl("^VIX"); dxy = _dl("DX-Y.NYB")
    gold = _dl("GC=F"); hy = _dl("HYG")

    print("[2] Fetching CryptoQuant signals...")
    cq = _mod.fetch_cquant_signals(CQ_KEY, start=TRAIN_START)
    prx = _mod.compute_proxy_signals(close, sp500, vix, dxy, gold, hy)
    raw_df = pd.DataFrame({**cq, **prx}).reindex(close.index).ffill()
    ter = _mod.compute_ternary_matrix(raw_df)
    tsr = _mod.compute_technical_signals(close, sp500, _mod.build_technical_signal_registry())
    combined = pd.concat([ter, tsr], axis=1).reindex(close.index).fillna(0)
    mvrv_raw = raw_df.get("MVRV")
    mvrv_aligned = mvrv_raw.reindex(close.index, method="ffill")
    disc = _mod.discretize_signals(combined)
    fwd = _mod.compute_forward_returns(close)
    print(f"  Base signals: {disc.shape[1]}")

    # ── Add 4 macro signals ─────────────────────────────────────────
    print("[3] Adding 4 macro signals (FRED API)...")
    from fredapi import Fred
    fred = Fred(api_key=FRED_KEY)
    def fetch_fred(sid):
        s = fred.get_series(sid, observation_start=TRAIN_START)
        return s.dropna().reindex(close.index, method="ffill")
    ffr = fetch_fred("FEDFUNDS"); ppi = fetch_fred("PPIACO"); cpi = fetch_fred("CPIAUCSL")
    sofr = fetch_fred("SOFR"); iorb = fetch_fred("IORB"); rrp = fetch_fred("RRPONTSYD")

    disc["FFR_vs_MA365"] = np.sign(ffr - ffr.rolling(365).mean()).reindex(disc.index).fillna(0).clip(-1, 1).astype(int)
    disc["SOFR_IORB_Spread_Sign"] = np.sign(sofr - iorb).reindex(disc.index).fillna(0).clip(-1, 1).astype(int)
    disc["RRP_Trend_90d"] = np.sign(rrp - rrp.rolling(90).mean()).reindex(disc.index).fillna(0).clip(-1, 1).astype(int)
    ppi_yoy = ppi.pct_change(252); cpi_yoy = cpi.pct_change(252)
    disc["PPI_vs_CPI"] = np.sign(ppi_yoy - cpi_yoy).reindex(disc.index).fillna(0).clip(-1, 1).astype(int)

    n_sigs = disc.shape[1]
    print(f"  Total signals: {n_sigs}")
    assert n_sigs == 59, f"Expected 59 signals, got {n_sigs}"

    # ── Build 2Y+3Y ensemble combo with T3 holdout pair selection ────
    print("[4] Building 2Y+3Y ensemble combo (T3 holdout pairs)...")
    rebal_dates = pd.date_range(pd.Timestamp(EVAL_START), close.index[-1], freq="3MS")
    sigs_all = list(disc.columns)
    combo_parts = {}
    for ty in [2, 3]:
        combo = pd.Series(0.0, index=close.index)
        print(f"  Building {ty}Y window...")
        for i, rd in enumerate(rebal_dates):
            train_start = rd - pd.DateOffset(years=ty)
            next_rd = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else close.index[-1] + pd.Timedelta(days=1)
            in_train = (disc.index >= train_start) & (disc.index < rd)
            disc_tr = disc[in_train]; fwd_tr = fwd[in_train]
            mvrv_tr = mvrv_aligned[in_train]
            act = {s: (disc_tr[s] != 0).mean() for s in sigs_all}
            active_sigs = [s for s, a in act.items() if a > ACTIVITY_THRESHOLD]
            pairs = list(combinations(active_sigs, 2))
            mv = float(mvrv_aligned.asof(rd)) if not pd.isna(mvrv_aligned.asof(rd)) else np.nan
            regime = _mod._get_regime(mv)
            n_same = int((mvrv_tr.apply(lambda v: _mod._get_regime(v) == regime)).sum())
            weights = _mod._regime_weights(mvrv_tr, regime) if n_same >= _mod.MIN_REGIME_TRAIN_DAYS else None

            top_pairs = select_pairs_holdout(pairs, disc_tr, fwd_tr, rd, weights)

            oos_dates = close.index[(close.index >= rd) & (close.index < next_rd)]
            for t in oos_dates:
                combo.loc[t] = _mod.score_at_date(disc, fwd, top_pairs, t)

            if i % 4 == 0:
                print(f"    [{rd.date()}] pairs={len(top_pairs)}, n_sigs={len(active_sigs)}")
        combo_parts[ty] = combo
    ensemble = sum(combo_parts.values()) / len(combo_parts)

    # ── D10 backtest ─────────────────────────────────────────────────
    print("[5] Running D10 backtest...")
    exposure = run_d10_backtest(ensemble, close)

    # ── Metrics ──────────────────────────────────────────────────────
    close_eval = close.loc[EVAL_START:]
    s, ho, strat_ar, mdd, ae, yearly = compute_metrics(close_eval, exposure.loc[EVAL_START:])

    # ── Save outputs ─────────────────────────────────────────────────
    ensemble.to_csv("/tmp/combo_v2_prod.csv", header=["combo"])
    exposure.loc[EVAL_START:].to_csv("/tmp/exposure_v2_prod.csv")

    # ── Report ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("D10 PRODUCTION v2 — RESULTS")
    print(f"{'='*70}")
    print(f"  Signals:      {n_sigs}")
    print(f"  Pair Select:  T3 holdout (70/30 split)")
    print(f"  Weights:      Uniform (1/6 each)")
    print(f"  Ensemble:     2Y + 3Y")
    print(f"  Sharpe:       {s:.4f}")
    print(f"  HO Sharpe:    {ho:.4f}")
    bh_ar = (1 + (1 + close_eval.pct_change().dropna()).prod() - 1) ** (1 / (len(close_eval) / 365.25)) - 1
    print(f"  Ann Return:   {strat_ar:+.1%} (Strategy) vs {bh_ar:+.1%} (B&H)")
    print(f"  Max Drawdown: {mdd:.1%}")
    print(f"  Avg Exposure: {ae:.1%}")
    print(f"  Current Exp:  {exposure.iloc[-1]:.0%}")
    print(f"  Yearly:")
    for yr, ret in yearly.items():
        print(f"    {yr}: {ret:+.1%}")
    print(f"\n  Saved: /tmp/combo_v2_prod.csv, /tmp/exposure_v2_prod.csv")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
