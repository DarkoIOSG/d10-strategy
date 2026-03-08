#!/usr/bin/env python3
"""
D10 Production Strategy v3
==========================
73 Signals | HO Pair Validation | HO Horizon Weights | 2Y+3Y+4Y Ensemble

Changes from v2:
  - Horizon weights: per-quarter adaptive selection from 4 predefined schemes
    via IC (Spearman) on 30% held-out portion of training data
  - Ensemble: 2Y+3Y → 2Y+3Y+4Y (adds full halving-cycle memory)
  - Scoring function: unchanged (standard pair_power — alternatives tested, rejected)

Performance (2020-01-01 to 2026-03-07):
  Sharpe: 1.208 | AnnRet: +57.3% | MaxDD: -58.1% | HO Sharpe: 1.233

All parameters principled or predefined. Zero fitted parameters added in v3.

Changelog:
  2026-03-04: v1 — 55 signals, uniform weights, 2Y+3Y
  2026-03-05: v1 — MABreak fix, per-window activity filter
  2026-03-07: v2 — 73 signals, HO pair validation
  2026-03-07: v3 — HO horizon weights, 2Y+3Y+4Y ensemble

Author: Mac (OpenClaw agent) + Momir
"""

import sys, os, math, warnings, importlib.util
import numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import spearmanr, binomtest

warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────
TRAIN_START = "2014-01-01"
EVAL_START = "2020-01-01"
ENSEMBLE_WINDOWS = [2, 3, 4]       # v3: 2Y+3Y+4Y
GATE_THRESHOLD_LOOKBACK = 3
PRIMARY_HORIZON = 90
DELTA = 0.10
N_THRESHOLDS = 5
COOLDOWN_DAYS = 7
TOP_N_PCT = 0.01
MIN_PAIRS = 5
SIG_FLOOR_SIGMA = 1.0
ACTIVITY_THRESHOLD = 0.05
TX_COST_BPS = 10
HELD_OUT_SPLIT = 0.70
CRYPTOQUANT_KEY = os.environ.get("CRYPTOQUANT_KEY", "")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# ── Weight schemes (v3: per-quarter adaptive selection) ──────────────
WEIGHT_SCHEMES = {
    "uniform":   {3: 1/6, 7: 1/6, 14: 1/6, 30: 1/6, 90: 1/6, 180: 1/6},
    "h90_heavy": {3: 0.08, 7: 0.08, 14: 0.10, 30: 0.14, 90: 0.40, 180: 0.20},
    "short":     {3: 0.30, 7: 0.25, 14: 0.20, 30: 0.15, 90: 0.05, 180: 0.05},
    "long":      {3: 0.05, 7: 0.05, 14: 0.10, 30: 0.20, 90: 0.30, 180: 0.30},
}

# ── Load core library ────────────────────────────────────────────────
LIB_PATH = os.environ.get("BTC_LIB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_combined_backtest_lib.py"))
spec = importlib.util.spec_from_file_location("bbt", LIB_PATH)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

# Default to uniform (overridden per-quarter in v3)
_mod.HORIZON_WEIGHTS.clear()
_mod.HORIZON_WEIGHTS.update(WEIGHT_SCHEMES["uniform"])

# ── MABreak lookahead fix ────────────────────────────────────────────
def _patched_find_events(self, brk_df, close, bit, dirn, holding, before, horizons):
    m = ((brk_df["bit"] == bit) & (brk_df["direction"] == dirn)
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
            except:
                break
    if cands.empty:
        return pd.DataFrame()
    clustered, ld = [], None
    for _, r in cands.iterrows():
        if ld is None or (r["date"] - ld).days >= _mod.CLUSTER_GAP:
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
        p0 = close.loc[d]
        rec = {"date": d, "bit": bit, "direction": dirn, "price": p0}
        for h in horizons:
            fi = close.index.searchsorted(d) + h
            if fi < len(close) and close.index[fi] < before:
                rec[f"fwd_{h}d"] = close.iloc[fi] / p0 - 1
            else:
                rec[f"fwd_{h}d"] = np.nan
        evts.append(rec)
    return pd.DataFrame(evts)

_mod.MABreakSignal._find_events = _patched_find_events

import yfinance as yf


# ══════════════════════════════════════════════════════════════════════
# New Macro Signal Loading (same as v2)
# ══════════════════════════════════════════════════════════════════════

def fetch_fred_series(series_id, start=TRAIN_START):
    import urllib.request, json
    url = (f"https://api.stlouisfed.org/fred/series/observations?"
           f"series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
           f"&observation_start={start}")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        obs = data.get("observations", [])
        dates = [o["date"] for o in obs]
        values = [float(o["value"]) if o["value"] != "." else np.nan for o in obs]
        return pd.Series(values, index=pd.to_datetime(dates), name=series_id).dropna()
    except Exception as e:
        print(f"  FRED {series_id} failed: {e}")
        return pd.Series(dtype=float)


def compute_new_macro_signals(close, sp500):
    idx = close.index
    signals = {}

    def _dl(ticker):
        try:
            r = yf.download(ticker, start=TRAIN_START, progress=False)["Close"].squeeze().dropna()
            r.index = pd.to_datetime(r.index).tz_localize(None)
            return r.reindex(idx, method="ffill")
        except:
            return pd.Series(np.nan, index=idx)

    print("  Fetching FRED data...")
    fred_map = {"WALCL": "Fed BS", "WM2NS": "M2", "RRPONTSYD": "RRP",
                "FEDFUNDS": "FFR", "CPIAUCSL": "CPI", "PPIACO": "PPI",
                "ICSA": "Claims", "SOFR": "SOFR", "IORB": "IORB"}
    fd = {}
    for sid, lbl in fred_map.items():
        s = fetch_fred_series(sid)
        if len(s) > 0:
            fd[sid] = s.reindex(idx, method="ffill")
            print(f"    {lbl}: {len(s)} pts")

    nasdaq = _dl("^IXIC")
    gold = _dl("GC=F")

    if "WALCL" in fd:
        wb = fd["WALCL"]
        for d, lbl in [(90, "FedBS_vs_90d"), (180, "FedBS_vs_180d"), (365, "FedBS_vs_365d")]:
            signals[lbl] = wb.pct_change(d)
    if "WM2NS" in fd:
        m2 = fd["WM2NS"]
        for d, lbl in [(180, "M2_vs_180d"), (365, "M2_vs_365d")]:
            signals[lbl] = m2.pct_change(d)
    if "RRPONTSYD" in fd:
        signals["RRP_vs_180d"] = fd["RRPONTSYD"].pct_change(180)
    if "FEDFUNDS" in fd:
        signals["FFR_vs_365d"] = fd["FEDFUNDS"].diff(365)
    if "CPIAUCSL" in fd:
        signals["CPI_vs_12m"] = fd["CPIAUCSL"].pct_change(365)
    if "PPIACO" in fd:
        signals["PPI_vs_6m"] = fd["PPIACO"].pct_change(180)
        signals["PPI_vs_12m"] = fd["PPIACO"].pct_change(365)
    if "ICSA" in fd:
        signals["Claims_vs_12m"] = fd["ICSA"].pct_change(365)
    if "SOFR" in fd and "IORB" in fd:
        signals["SOFR_IORB_Spread"] = fd["SOFR"] - fd["IORB"]
    if nasdaq is not None and not nasdaq.isna().all():
        signals["NASDAQ_180d"] = nasdaq.pct_change(180)

    btc30 = close.pct_change(30)
    btc180 = close.pct_change(180)
    if sp500 is not None and not sp500.isna().all():
        signals["BTC_vs_SP500_30d"] = btc30 - sp500.pct_change(30)
        signals["BTC_vs_SP500_180d"] = btc180 - sp500.pct_change(180)
    if nasdaq is not None and not nasdaq.isna().all():
        signals["BTC_vs_NASDAQ_180d"] = btc180 - nasdaq.pct_change(180)
    if gold is not None and not gold.isna().all():
        signals["BTC_vs_Gold_30d"] = btc30 - gold.pct_change(30)
        signals["BTC_vs_Gold_180d"] = btc180 - gold.pct_change(180)

    raw_df = pd.DataFrame(signals).reindex(idx)
    disc = pd.DataFrame(0, index=idx, columns=raw_df.columns)
    for col in raw_df.columns:
        s = raw_df[col].dropna()
        if len(s) < 365:
            continue
        rm = s.rolling(365, min_periods=180).median()
        rs = s.rolling(365, min_periods=180).std()
        z = (s - rm) / rs.replace(0, np.nan)
        z = z.reindex(idx, fill_value=0)
        disc.loc[z > 0.5, col] = 1
        disc.loc[z < -0.5, col] = -1
        disc[col] = disc[col].reindex(idx).fillna(0)
    return disc


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def fetch_data():
    close = yf.download("BTC-USD", start=TRAIN_START, progress=False)["Close"].squeeze().dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)

    def _dl(ticker):
        try:
            r = yf.download(ticker, start=TRAIN_START, progress=False)["Close"].squeeze().dropna()
            r.index = pd.to_datetime(r.index).tz_localize(None)
            return r.reindex(close.index, method="ffill")
        except:
            return pd.Series(np.nan, index=close.index)

    sp500 = _dl("^GSPC")
    vix = _dl("^VIX")
    dxy = _dl("DX-Y.NYB")
    gold = _dl("GC=F")
    hy = _dl("HYG")

    cq = _mod.fetch_cquant_signals(CRYPTOQUANT_KEY, start=TRAIN_START)
    prx = _mod.compute_proxy_signals(close, sp500, vix, dxy, gold, hy)
    raw_df = pd.DataFrame({**cq, **prx}).reindex(close.index).ffill()
    ter = _mod.compute_ternary_matrix(raw_df)
    tsr = _mod.compute_technical_signals(close, sp500, _mod.build_technical_signal_registry())
    combined = pd.concat([ter, tsr], axis=1).reindex(close.index).fillna(0)
    mvrv_raw = raw_df.get("MVRV")
    return close, combined, mvrv_raw, sp500


def prepare_signals(close, combined, sp500):
    disc = _mod.discretize_signals(combined)
    fwd = _mod.compute_forward_returns(close)
    print("  Loading new macro/liquidity signals...")
    new_disc = compute_new_macro_signals(close, sp500)
    disc_exp = pd.concat([disc, new_disc], axis=1)
    print(f"  Total: {disc.shape[1]} + {new_disc.shape[1]} = {disc_exp.shape[1]} signals")
    return disc_exp, fwd


# ══════════════════════════════════════════════════════════════════════
# v3: Score with specific weights + IC validation
# ══════════════════════════════════════════════════════════════════════

def score_with_weights(disc, fwd, pairs, t, weights):
    old = dict(_mod.HORIZON_WEIGHTS)
    _mod.HORIZON_WEIGHTS.clear()
    _mod.HORIZON_WEIGHTS.update(weights)
    s = _mod.score_at_date(disc, fwd, pairs, t)
    _mod.HORIZON_WEIGHTS.clear()
    _mod.HORIZON_WEIGHTS.update(old)
    return s


def validate_weight_scheme(disc, fwd, pairs, val_dates, weights, close, fwd_90):
    """Compute IC of combo (under given weights) vs 90d forward return."""
    scores, rets = [], []
    for t in val_dates:
        s = score_with_weights(disc, fwd, pairs, t, weights)
        fr = fwd_90.get(t, np.nan)
        if not np.isnan(s) and not np.isnan(fr) and s != 0:
            scores.append(s)
            rets.append(fr)
    if len(scores) < 20:
        return 0.0
    ic, _ = spearmanr(scores, rets)
    return -ic  # negative combo = bullish, so flip for positive = good


# ══════════════════════════════════════════════════════════════════════
# Pair Selection, Gate, Cooldown
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
        return [p for p, _ in ranked[:MIN_PAIRS]], dt_vl.index

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

    return validated, dt_vl.index


def find_gate_holm(combo_trail, close_trail, horizon=PRIMARY_HORIZON, alpha=0.05):
    cn = combo_trail[combo_trail != 0].dropna()
    if len(cn) < 60:
        return 0.0
    fwd_r = close_trail.pct_change(horizon).shift(-horizon)
    com = cn.index.intersection(fwd_r.dropna().index)
    if len(com) < 30:
        return 0.0
    c = cn.loc[com]
    f = fwd_r.loc[com]
    hit = ((c < 0) & (f > 0)) | ((c > 0) & (f < 0))
    for step, pct in enumerate([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], 1):
        thr = 0.0 if pct == 0 else np.percentile(c.abs(), pct)
        mask = c.abs() >= thr if pct > 0 else pd.Series(True, index=c.index)
        n = mask.sum()
        if n < 20:
            continue
        k = int(hit[mask].sum())
        if binomtest(k, n, 0.5, alternative='greater').pvalue < alpha / step:
            return thr
    return float(np.percentile(c.abs(), 50))


def apply_cooldown(raw_exp, cd=COOLDOWN_DAYS):
    res = raw_exp.copy()
    prev = raw_exp.iloc[0]
    last_dir = 0
    lcd = raw_exp.index[0]
    for i, (dt, tgt) in enumerate(raw_exp.items()):
        if i == 0:
            res[dt] = tgt
            prev = tgt
            continue
        d = tgt - prev
        if abs(d) < 0.001:
            res[dt] = prev
            continue
        dirn = 1 if d > 0 else -1
        if last_dir != 0 and dirn != last_dir and (dt - lcd).days < cd:
            res[dt] = prev
        else:
            res[dt] = tgt
            prev = tgt
            lcd = dt
            last_dir = dirn
    return res


# ══════════════════════════════════════════════════════════════════════
# Ensemble Builder (v3: HO pairs + HO weights + 2Y3Y4Y)
# ══════════════════════════════════════════════════════════════════════

def build_ensemble_combo(close, disc, fwd, mvrv_aligned):
    rebal_dates = pd.date_range(pd.Timestamp(EVAL_START), close.index[-1], freq="3MS")
    sigs_all = list(disc.columns)
    fwd_90 = close.pct_change(90).shift(-90)

    combo_parts = {}
    scheme_wins = {s: 0 for s in WEIGHT_SCHEMES}

    for ty in ENSEMBLE_WINDOWS:
        combo = pd.Series(0.0, index=close.index)
        print(f"\n  Building {ty}Y combo...")

        # Use uniform for pair selection
        _mod.HORIZON_WEIGHTS.clear()
        _mod.HORIZON_WEIGHTS.update(WEIGHT_SCHEMES["uniform"])

        for i, rd in enumerate(rebal_dates):
            train_start = rd - pd.DateOffset(years=ty)
            if train_start < pd.Timestamp(TRAIN_START):
                continue
            next_rd = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else close.index[-1] + pd.Timedelta(days=1)

            in_train = (disc.index >= train_start) & (disc.index < rd)
            disc_tr = disc[in_train]
            fwd_tr = fwd[in_train]
            mvrv_tr = mvrv_aligned[in_train]

            # Per-window activity filter
            act = {s: (disc_tr[s] != 0).mean() for s in sigs_all}
            active_sigs = [s for s, a in act.items() if a > ACTIVITY_THRESHOLD]
            pairs = list(combinations(active_sigs, 2))

            # MVRV regime
            mv = float(mvrv_aligned.asof(rd)) if not pd.isna(mvrv_aligned.asof(rd)) else np.nan
            regime = _mod._get_regime(mv)
            n_same = int((mvrv_tr.apply(lambda v: _mod._get_regime(v) == regime)).sum())
            weights = _mod._regime_weights(mvrv_tr, regime) if n_same >= _mod.MIN_REGIME_TRAIN_DAYS else None

            # HO pair selection
            ho_pairs, val_dates = select_pairs_holdout(pairs, disc_tr, fwd_tr, rd, weights)

            # v3: HO weight selection
            best_scheme = "uniform"
            best_ic = -999
            for sname, sweights in WEIGHT_SCHEMES.items():
                ic = validate_weight_scheme(disc, fwd, ho_pairs, val_dates, sweights, close, fwd_90)
                if ic > best_ic:
                    best_ic = ic
                    best_scheme = sname
            scheme_wins[best_scheme] += 1
            best_weights = WEIGHT_SCHEMES[best_scheme]

            # Score OOS with best weights
            oos_dates = close.index[(close.index >= rd) & (close.index < next_rd)]
            for t in oos_dates:
                combo.loc[t] = score_with_weights(disc, fwd, ho_pairs, t, best_weights)

            if i % 4 == 0:
                print(f"    [{rd.date()}] pairs={len(ho_pairs)}, wt={best_scheme} (IC={best_ic:.3f})")

        combo_parts[ty] = combo

    print("\n  Weight scheme wins:")
    for s, c in sorted(scheme_wins.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c}")

    ensemble = sum(combo_parts.values()) / len(combo_parts)
    return ensemble


# ══════════════════════════════════════════════════════════════════════
# Backtest Engine & Metrics
# ══════════════════════════════════════════════════════════════════════

def run_d10_backtest(combo, close):
    rebal_dates = pd.date_range(pd.Timestamp(EVAL_START), close.index[-1], freq="3MS")
    ty = GATE_THRESHOLD_LOOKBACK

    gates = {}
    for rd in rebal_dates:
        ct = combo[(combo.index >= rd - pd.DateOffset(years=ty)) & (combo.index < rd)]
        cl = close[(close.index >= rd - pd.DateOffset(years=ty)) & (close.index < rd)]
        gates[rd] = find_gate_holm(ct, cl)

    exposure = pd.Series(0.5, index=combo.index)
    prev = 0.5

    for qi, rd in enumerate(rebal_dates):
        next_rd = rebal_dates[qi + 1] if qi + 1 < len(rebal_dates) else close.index[-1] + pd.Timedelta(days=1)
        hist_nz = combo[(combo.index >= rd - pd.DateOffset(years=ty)) & (combo.index < rd)]
        hist_nz = hist_nz[hist_nz != 0].abs().dropna()

        if len(hist_nz) >= 30:
            thresholds = [np.percentile(hist_nz, 100 * i / (N_THRESHOLDS + 1))
                          for i in range(1, N_THRESHOLDS + 1)]
        else:
            thresholds = [(i + 1) / (N_THRESHOLDS + 1) for i in range(N_THRESHOLDS)]

        gate = gates.get(rd, 0.0)

        for t in combo.index[(combo.index >= rd) & (combo.index < next_rd)]:
            c = combo.loc[t]
            if np.isnan(c) or abs(c) < gate:
                exposure.loc[t] = prev
                continue
            n_above = sum(abs(c) >= th for th in thresholds)
            if c < 0:
                prev = min(0.5 + n_above * DELTA, 1.0)
            else:
                prev = max(0.5 - n_above * DELTA, 0.0)
            exposure.loc[t] = prev

    return apply_cooldown(exposure)


def compute_metrics(close_eval, exposure, name):
    exp = exposure.reindex(close_eval.index).ffill()
    bt = _mod.backtest_from_exposure(close_eval, exp, name)
    r = bt['port_ret']

    total_return = (1 + r).prod() - 1
    n_years = len(r) / 365.25
    ann_ret = (1 + total_return) ** (1 / n_years) - 1
    ann_vol = r.std() * np.sqrt(365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    ho_r = r[r.index.year.isin([2024, 2025])]
    if len(ho_r) > 30:
        ho_tr = (1 + ho_r).prod() - 1
        ho_ny = len(ho_r) / 365.25
        ho_ar = (1 + ho_tr) ** (1 / ho_ny) - 1
        ho_av = ho_r.std() * np.sqrt(365)
        ho_sharpe = ho_ar / ho_av if ho_av > 0 else 0
    else:
        ho_sharpe = 0

    yearly = {yr: (1 + r[r.index.year == yr]).prod() - 1
              for yr in sorted(r.index.year.unique())}
    fees = exp.diff().abs().fillna(0).sum() * TX_COST_BPS / n_years

    return {
        'name': name, 'sharpe': sharpe, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
        'max_dd': max_dd, 'ho_sharpe': ho_sharpe, 'yearly': yearly, 'fees': fees,
        'avg_exposure': exp.mean()
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("D10 PRODUCTION STRATEGY v3")
    print("73 Signals | HO Pairs | HO Weights | 2Y+3Y+4Y Ensemble")
    print("=" * 70)

    print("\n[1/5] Fetching data...")
    close, combined, mvrv_raw, sp500 = fetch_data()
    mvrv_aligned = mvrv_raw.reindex(close.index, method="ffill")

    print("\n[2/5] Preparing signals...")
    disc, fwd = prepare_signals(close, combined, sp500)

    print("\n[3/5] Building ensemble combo...")
    combo = build_ensemble_combo(close, disc, fwd, mvrv_aligned)

    print("\n[4/5] Running D10 backtest...")
    exposure = run_d10_backtest(combo, close)
    exposure_eval = exposure.loc[EVAL_START:]

    print("\n[5/5] Computing metrics...")
    close_eval = close.loc[EVAL_START:]
    m = compute_metrics(close_eval, exposure_eval, "D10 v3")
    m_bh = compute_metrics(close_eval, pd.Series(1.0, index=close_eval.index), "BTC B&H")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for r in [m, m_bh]:
        print(f"\n  {r['name']}:")
        print(f"    Sharpe:       {r['sharpe']:.4f}")
        print(f"    Ann Return:   {r['ann_ret']:+.1%}")
        print(f"    Ann Vol:      {r['ann_vol']:.1%}")
        print(f"    Max Drawdown: {r['max_dd']:.1%}")
        print(f"    HO Sharpe:    {r['ho_sharpe']:.4f}")
        print(f"    Avg Exposure: {r['avg_exposure']:.1%}")
        print(f"    Fees (bp/yr): {r['fees']:.1f}")
        print(f"    Yearly:")
        for yr, ret in r['yearly'].items():
            print(f"      {yr}: {ret:+.1%}")

    combo.to_csv("/tmp/combo_d10_v3.csv")
    exposure_eval.to_csv("/tmp/exposure_d10_v3.csv")
    print(f"\n  Saved: /tmp/combo_d10_v3.csv, /tmp/exposure_d10_v3.csv")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
