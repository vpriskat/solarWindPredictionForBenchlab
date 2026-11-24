"""Experiment ladder for the BenchLab solar-wind stacker.

    python run_experiments.py --backend xgb --cache cache.pkl
    python run_experiments.py --backend xgb --cache cache.pkl --obs-latency 3        # live-like ACE latency
    python run_experiments.py --backend xgb --cache cache.pkl --enlil-path <re-downloaded 0-96h file>

Everything is scored on ONE fixed test block with the official metric (NaN obs dropped, NaN preds -> persistence).

Variants
  A    absolute target, heuristic weights, base features            (your old setup)
  D    residual vs mean(enlil, tcn, ts), metric weights, all feats   (best of the previous ladder)
  R    per-lead-group ridge blend of persistence / recurrence / TCN / Timestral / ENLIL        (NO trees)
  T    R + strongly regularised trees on the residual, 'lite' features; may choose 0 trees
  T+   same with ALL engineered features
  T -g T retrained without feature group g                    -> which feeds earn their place
and a stress test: T evaluated with group g set to NaN at inference (what a dead feed costs).

The trees use min_child_weight ~ 10 issue times of rows (72 rows share one t0) and 32 histogram bins, because
slowly-varying features (coronal-hole values) are otherwise near-unique date IDs the trees can memorise.
"""
import argparse
import dataclasses
import os
import pickle
import time

import numpy as np
import pandas as pd

import features as F
from features import Cfg, load_raw, make_features, coverage_report
from score_tools import score, per_lead, metric_weights, bootstrap_matrix
from blend import Blend
from tailcal import TailCal
from date_report import print_date_report

BLEND_COLS = ["persistence", "rec27_eff", "tcn_pred", "ts_speed_med", "enlil"]
BLEND_OF_GROUP = {"rec": ["rec27_eff"], "ts": ["ts_speed_med"], "tcn": ["tcn_pred"], "enlil": ["enlil"]}
ABLATE = ["omni", "ch", "rec", "trend", "enlil", "ts", "tcn"]
CACHE_VERSION = 3


# ----------------------------------------------------------------------------- compat with features.py versions
def get_sets(df):
    fs = F.feature_sets(df)
    if isinstance(fs, dict):                       # newer features.py: dict(base, lite, all)
        return fs
    base, allf = fs                                # older features.py: (base, all)
    return dict(base=base, lite=base, all=allf)


def get_groups(cols):
    if hasattr(F, "feature_groups"):
        return F.feature_groups(cols)
    cols = list(cols)
    omni = getattr(F, "OMNI_COLS", ["kp", "dst", "ap", "f107", "density", "bz", "dst_min_24h", "bz_min_24h"])
    g = {"omni": [c for c in cols if c in omni],
         "ch": [c for c in cols if c.startswith("ch_")],
         "rec": [c for c in cols if c.startswith("rec")],
         "trend": [c for c in cols if c.startswith(("d_obs_", "obs_max_", "obs_min_", "obs_std_", "persist"))],
         "enlil": [c for c in cols if c.startswith("enlil")],
         "ts": [c for c in cols if c.startswith("ts_speed")],
         "c2": [c for c in cols if c.startswith("c2_speed")],
         "tcn": [c for c in cols if c == "tcn_pred"]}
    derived = [c for c in ("ref", "base_spread") if c in cols]
    for k in ("enlil", "ts", "tcn"):
        g[k] = g[k] + derived
    return g


# ----------------------------------------------------------------------------- model backends
PRESETS = {
    # what the earlier ladder used
    "legacy": dict(depth=4, eta=0.05, mcw=20, lam=5.0, sub=0.8, col=0.8, bins=256, msl=200),
    # few, shallow, coarse trees; a leaf must span >= ~10 issue times (72 rows each)
    "reg": dict(depth=3, eta=0.03, mcw=720, lam=20.0, sub=0.5, col=0.5, bins=32, msl=700),
}


class XGBBackend:
    def __init__(self, preset):
        self.pr = PRESETS[preset]

    def fit(self, X, y, w, rounds):
        import xgboost as xgb
        pr = self.pr
        self.xgb, self.feats = xgb, list(X.columns)
        p = dict(objective="reg:squarederror", max_depth=pr["depth"], eta=pr["eta"], subsample=pr["sub"],
                 colsample_bytree=pr["col"], min_child_weight=pr["mcw"], reg_lambda=pr["lam"], max_bin=pr["bins"],
                 seed=42, tree_method="hist", base_score=float(np.average(y, weights=w)))
        d = xgb.DMatrix(X, label=y, weight=w, feature_names=self.feats)
        self.bst = xgb.train(p, d, num_boost_round=rounds)

    def predict_many(self, X, ns):                  # ns: rounds >= 1
        d = self.xgb.DMatrix(X[self.feats], feature_names=self.feats)
        return {n: self.bst.predict(d, iteration_range=(0, n)) for n in ns}

    def importance(self):
        return pd.Series(self.bst.get_score(importance_type="gain")).sort_values(ascending=False)


class HGBBackend:                                    # sklearn fallback (used for the synthetic tests); same interface
    def __init__(self, preset):
        self.pr = PRESETS[preset]

    def fit(self, X, y, w, rounds):
        from sklearn.ensemble import HistGradientBoostingRegressor
        pr = self.pr
        self.feats = list(X.columns)
        self.m = HistGradientBoostingRegressor(learning_rate=pr["eta"], max_iter=rounds, max_depth=pr["depth"],
                                               min_samples_leaf=pr["msl"], l2_regularization=pr["lam"],
                                               max_bins=min(pr["bins"], 255), max_features=pr["col"], random_state=0)
        self.m.fit(X, y, sample_weight=w)

    def predict_many(self, X, ns):
        want, out = set(ns), {}
        for i, p in enumerate(self.m.staged_predict(X[self.feats]), 1):
            if i in want:
                out[i] = p
        return out

    def importance(self):
        return None


BACKENDS = {"xgb": XGBBackend, "hgb": HGBBackend}


# ----------------------------------------------------------------------------- one variant = blend (optional) + trees (optional)
def mk(feats, loss="metric", base="blend", trees=True, preset="reg", blend_cols=BLEND_COLS, rounds=None):
    return dict(feats=list(feats), loss=loss, base=base, trees=trees, preset=preset,
                blend_cols=list(blend_cols), rounds=rounds)


class Stage:
    """base = 'none' (absolute target) | 'mean' (d['ref']) | 'blend' (per-lead ridge);  trees fit the residual."""

    def __init__(self, sp, backend, rounds):
        self.sp, self.backend, self.rounds = sp, backend, rounds
        self.blend = self.be = None

    def _ref(self, d):
        if self.blend is not None:
            return self.blend.predict(d)
        if self.sp["base"] == "mean":
            return d["ref"].to_numpy(float)
        return 0.0

    def fit(self, d):
        sp = self.sp
        self.feats = [f for f in sp["feats"] if f in d.columns]
        if sp["base"] == "blend":
            self.blend = Blend(sp["blend_cols"]).fit_two_pass(d)
        if sp["trees"] and self.rounds > 0:
            ref = self._ref(d)
            y = d["obs"].to_numpy(float) - ref
            if sp["loss"] == "heuristic":
                w = 1.0 + (d["obs"].to_numpy(float) / 400.0) ** 2            # = your old timse_objective
            else:
                w = metric_weights(d, ref if sp["base"] != "none" else d["ref"].to_numpy(float))
            self.be = BACKENDS[self.backend](sp["preset"])
            self.be.fit(d[self.feats], y, w, self.rounds)
        return self

    def predict_many(self, x, ns):
        base = self._ref(x)
        pos = [n for n in ns if n > 0]
        tp = self.be.predict_many(x[self.feats], pos) if (self.be is not None and pos) else {}
        return {n: (base + tp[n]) if n in tp else base + 0.0 for n in ns}


def _frame(d, pred):
    return pd.DataFrame({"t0": d["t0"].to_numpy(), "lead_time": d["lead_time"].to_numpy(),
                         "obs": d["obs"].to_numpy(), "persistence": d["persistence"].to_numpy(), "pred": pred})


def fit_select(sp, tr, va, backend, step, refit, fixed_n=None):
    """Choose the number of tree rounds on `va` (0 allowed for blend/mean bases), then refit on train+val.
    fixed_n: skip the choice and use this many rounds (used for ablations, so they differ from T only in features)."""
    rounds = sp["rounds"] or 0
    if not sp["trees"]:
        ns = [0]
    elif fixed_n is not None:
        ns, rounds = [fixed_n], fixed_n
    else:
        ns = ([0] if sp["base"] != "none" else []) + list(range(step, rounds + 1, step))
    st = Stage(sp, backend, rounds if sp["trees"] else 0).fit(tr)
    pv = st.predict_many(va, ns)
    curve = {n: score(_frame(va, pv[n]))["S_final"] for n in ns}
    best_n = min(curve, key=curve.get)                 # ties -> fewer rounds
    if refit:
        st = Stage(sp, backend, best_n if sp["trees"] else 0).fit(pd.concat([tr, va], ignore_index=True))
    return st, best_n, curve


def split(df, train_end, val_end, purge_h=72):
    p = pd.Timedelta(hours=purge_h)                    # purge so no training target reaches into the next block
    tr = df[df.t0 < pd.Timestamp(train_end) - p]
    va = df[(df.t0 >= pd.Timestamp(train_end)) & (df.t0 < pd.Timestamp(val_end) - p)]
    te = df[df.t0 >= pd.Timestamp(val_end)]
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


# ----------------------------------------------------------------------------- the ladder
DATES_CSV = None                                  # set by --dates-csv
CME_PATH = None                                   # set by --cme-path
CME_LAMBDAS = (0.5, 1.0)
STRIP_PATH = None                                 # set by --strip-path

CALS = [("R+shift", ((1, 72),), False),                       # 1 parameter
        ("R+shift+stretch", ((1, 72),), True),                # 2 parameters
        ("R+cal 3 groups", ((1, 24), (25, 48), (49, 72)), True)]   # 6 parameters


def run_ladder(df, backend="xgb", rounds=300, legacy_rounds=600, step=25, boot=100, refit=True, ablate=True,
               train_end="2025-10-01", val_end="2026-01-01"):
    tr, va, te = split(df, train_end, val_end)
    print(f"rows  train {len(tr):,}  val {len(va):,}  test {len(te):,}   "
          f"(t0 test {te.t0.min().date()} .. {te.t0.max().date()})")
    sets = get_sets(df)
    for d in (tr, va, te):
        d["ref"] = d["ref"].fillna(d["persistence"]).fillna(tr["obs"].mean())
        d["rec27_eff"] = d["rec27"].fillna(d["rec27_mean12"]) if "rec27_mean12" in d else d["rec27"]

    variants = {
        "A current (heuristic w)": mk(sets["base"], "heuristic", "none", True, "legacy", rounds=legacy_rounds),
        "D resid on mean-ref": mk(sets["all"], "metric", "mean", True, "legacy", rounds=legacy_rounds),
        "R per-lead blend": mk([], "metric", "blend", False),
        "R + clim27": mk([], "metric", "blend", False, blend_cols=BLEND_COLS + ["clim27"]),
        "T blend+trees (lite)": mk(sets["lite"], rounds=rounds),
        "T+ blend+trees (all)": mk(sets["all"], rounds=rounds),
    }
    has_c2 = "c2_speed_med" in df.columns
    if has_c2:                                              # the previous R (no Chronos-2) in the same run -> paired comparison
        variants["R -c2 (previous R)"] = mk([], "metric", "blend", False, blend_cols=[c for c in BLEND_COLS if c != "c2_speed_med"])
    if ablate:
        lite_groups = get_groups(sets["lite"])
        for g in ABLATE:
            drop = set(lite_groups.get(g, []))
            if drop:
                variants[f"T -{g}"] = mk([f for f in sets["lite"] if f not in drop], rounds=rounds,
                                         blend_cols=[c for c in BLEND_COLS if c not in BLEND_OF_GROUP.get(g, [])])

    res = te[["t0", "lead_time", "obs", "persistence"]].copy()
    fitted = {}
    for name, sp in variants.items():
        t = time.time()
        fixed = fitted["T blend+trees (lite)"][1] if name.startswith("T -") else None   # same #trees as T
        st, best_n, curve = fit_select(sp, tr, va, backend, step, refit, fixed)
        fitted[name] = (st, best_n)
        res[name] = st.predict_many(te, [best_n])[best_n]
        extra = f"  (val S with 0 trees {curve[0]:.1f})" if 0 in curve and best_n != 0 else ""
        print(f"  trained {name:<26} trees={best_n:<4} val S={curve[best_n]:7.2f}{extra}   ({time.time() - t:.0f}s)")

    # ---- tail calibration on top of R: R fitted on TRAIN only, calibration fitted on VAL, scored on TEST (no refit mismatch)
    stR0 = Stage(variants["R per-lead blend"], backend, 0).fit(tr)
    pv = stR0.predict_many(va, [0])[0]
    res["R (train-only)"] = stR0.predict_many(te, [0])[0]
    res["R+clim27 (train-only)"] = Stage(variants["R + clim27"], backend, 0).fit(tr).predict_many(te, [0])[0]
    cal_names = ["R (train-only)"]
    print("\nTail calibration on R  (R fitted on train only, calibration fitted on val, scored on test)")
    print(f"  val S before calibration: {score(_frame(va, pv))['S_final']:.2f}   (the calibrated val numbers below are in-sample)")
    for nm, groups, stretch in CALS:
        cal = TailCal(groups, stretch).fit(va, pv)
        res[nm] = cal.predict(te, res["R (train-only)"].to_numpy())
        cal_names.append(nm)
        print(f"  {nm:<18} val S {score(_frame(va, cal.predict(va, pv)))['S_final']:.2f}   {cal.describe()}")

    baselines = {"persistence": te["persistence"], "27-day recurrence": te["rec27"],
                 "enlil only": te["enlil"], "timestral only": te["ts_speed_med"], "tcn only": te["tcn_pred"],
                 **({"chronos only": te["c2_speed_med"]} if has_c2 else {}),
                 "climatology (train mean)": pd.Series(tr["obs"].mean(), index=te.index)}
    for k, v in baselines.items():
        res[k] = v.to_numpy()

    cme_names = []
    if CME_PATH is not None:
        k = (pd.read_csv(CME_PATH, index_col=0, parse_dates=True) if CME_PATH.endswith(".csv") else pd.read_parquet(CME_PATH))
        kl = k.stack().reset_index()
        kl.columns = ["t0", "h", "cme_k"]
        kl["lead_time"] = kl["h"].str[1:].astype(int)
        kl["t0"] = pd.to_datetime(kl["t0"]).astype("datetime64[ns]")
        res["t0"] = pd.to_datetime(res["t0"]).astype("datetime64[ns]")
        res["cme_k"] = res[["t0", "lead_time"]].merge(kl[["t0", "lead_time", "cme_k"]], on=["t0", "lead_time"], how="left")["cme_k"].fillna(0.0).to_numpy()
        print(f"\nCME kernel feature: nonzero on {(res['cme_k'].abs() > 5).mean() * 100:.1f}% of test rows, max {res['cme_k'].max():.0f} km/s")
        for lam in CME_LAMBDAS:
            for base in ("R per-lead blend", "R (train-only)"):
                nm = f"R + cme x{lam:g}" if base.startswith("R per") else f"R (train-only) + cme x{lam:g}"
                res[nm] = res[base] + lam * res["cme_k"]
                cme_names.append(nm)
    strip_names = []
    if STRIP_PATH is not None:
        sf = pd.read_parquet(STRIP_PATH)
        sf["t0"] = pd.to_datetime(sf["t0"]).astype("datetime64[ns]")
        res["t0"] = pd.to_datetime(res["t0"]).astype("datetime64[ns]")
        res["strip_adj"] = res[["t0", "lead_time"]].merge(sf[["t0", "lead_time", "strip_adj"]], on=["t0", "lead_time"], how="left")["strip_adj"].fillna(0.0).to_numpy()
        print(f"\nstrip adjustment: nonzero on {(res['strip_adj'] != 0).mean() * 100:.0f}% of test rows, std {res['strip_adj'].std():.1f} km/s")
        for lam in CME_LAMBDAS:
            for base in ("R per-lead blend", "R (train-only)"):
                nm = f"R + strip x{lam:g}" if base.startswith("R per") else f"R (train-only) + strip x{lam:g}"
                res[nm] = res[base] + lam * res["strip_adj"]
                strip_names.append(nm)
    names = list(baselines) + list(variants) + cal_names + ["R+clim27 (train-only)"] + cme_names + strip_names
    S = {n: score(res, n) for n in names}
    print("\n" + f"{'':<27}{'S_final':>8}{'RMSE':>8}{'TRMSE':>8}{'S1-24':>8}{'S25-48':>8}{'S49-72':>8}{'bias':>8}")
    for n in names:
        s = S[n]
        print(f"{n:<27}{s['S_final']:8.2f}{s['RMSE']:8.2f}{s['TRMSE']:8.2f}"
              f"{s['S_1_24']:8.2f}{s['S_25_48']:8.2f}{s['S_49_72']:8.2f}{s['bias']:8.1f}")

    t = time.time()
    B = bootstrap_matrix(res, names, n_boot=boot)
    idx = {n: i for i, n in enumerate(names)}

    def delta(a, b):
        d = B[:, idx[a]] - B[:, idx[b]]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return S[a]["S_final"] - S[b]["S_final"], lo, hi

    best_base = min(baselines, key=lambda k: S[k]["S_final"])
    print(f"\nPaired block bootstrap, S_final(new) - S_final(old), 95% CI  ({boot} resamples, 5-day blocks, "
          f"{time.time() - t:.0f}s)")
    for a, b in [("D resid on mean-ref", "A current (heuristic w)"), ("R per-lead blend", best_base),
                 ("T blend+trees (lite)", "R per-lead blend"), ("T blend+trees (lite)", best_base),
                 ("T+ blend+trees (all)", "T blend+trees (lite)")]:
        m, lo, hi = delta(a, b)
        flag = "  <- CI excludes 0" if (lo > 0 or hi < 0) else ""
        print(f"  {a:<24} vs {b:<26} {m:+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]{flag}")

    if has_c2:
        print("\nChronos-2 feed (paired block bootstrap, 95% CI; negative = Chronos-2 helps)")
        for a_, b_ in [("R per-lead blend", "R -c2 (previous R)"), ("chronos only", "timestral only"),
                       ("R per-lead blend", "timestral only")]:
            m, lo, hi = delta(a_, b_)
            flag = "  <- CI excludes 0" if (lo > 0 or hi < 0) else ""
            print(f"  {a_:<22} vs {b_:<22} {m:+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]{flag}")

    print("\nRegime anchor: trailing 27-day mean speed added to the blend  (paired block bootstrap, 95% CI)")
    for a_, b_ in [("R + clim27", "R per-lead blend"), ("R+clim27 (train-only)", "R (train-only)")]:
        m, lo, hi = delta(a_, b_)
        flag = "  <- CI excludes 0" if (lo > 0 or hi < 0) else ""
        print(f"  {a_:<22} vs {b_:<18} {m:+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]{flag}")
    print("  (refit-on-train+val minus train-only for the same blend: a positive number means adding the newest data HURT on the test)")
    for a_, b_ in [("R per-lead blend", "R (train-only)"), ("R + clim27", "R+clim27 (train-only)")]:
        m, lo, hi = delta(a_, b_)
        print(f"  {a_:<22} vs {b_:<24} {m:+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]")

    if cme_names:
        print("\nCME arrival kernel added to R  (paired block bootstrap, 95% CI; negative = the kernel helps)")
        for nm in cme_names:
            base = "R per-lead blend" if nm.startswith("R + ") else "R (train-only)"
            m_, lo, hi = delta(nm, base)
            flag = "  <- CI excludes 0" if (lo > 0 or hi < 0) else ""
            print(f"  {nm:<26} vs {base:<18} {m_:+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]{flag}")

    if strip_names:
        print("\nCoronal-hole strip adjustment added to R  (paired block bootstrap, 95% CI; negative = the images help)")
        for nm in strip_names:
            base = "R per-lead blend" if nm.startswith("R + ") else "R (train-only)"
            m_, lo, hi = delta(nm, base)
            flag = "  <- CI excludes 0" if (lo > 0 or hi < 0) else ""
            print(f"  {nm:<28} vs {base:<18} {m_:+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]{flag}")

    print("\nTail calibration vs the same R without calibration  (paired block bootstrap, 95% CI)")
    for cn in cal_names[1:]:
        m, lo, hi = delta(cn, "R (train-only)")
        flag = "  <- CI excludes 0" if (lo > 0 or hi < 0) else ""
        print(f"  {cn:<18} vs R (train-only)  {m:+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]{flag}")

    T = "T blend+trees (lite)"
    if ablate:
        stT, nT = fitted[T]
        gall = get_groups(te.columns)
        print("\nFeed ablation on T  (positive = the group HELPS; 'dead feed' = group set to NaN at inference, no retrain)")
        print(f"  {'group':<8}{'S without (retrained)':>24}{'  delta vs T  [95% CI]':>30}{'S if feed dead':>18}")
        for g in ABLATE:
            nm = f"T -{g}"
            if nm not in idx:
                continue
            m, lo, hi = delta(nm, T)
            masked = te.copy()
            cols = [c for c in gall.get(g, []) + BLEND_OF_GROUP.get(g, []) if c in masked.columns]
            masked[cols] = np.nan
            dead = score(_frame(masked, stT.predict_many(masked, [nT])[nT]))["S_final"]
            flag = "  <-" if (lo > 0 or hi < 0) else ""
            print(f"  {g:<8}{S[nm]['S_final']:24.2f}{m:+14.2f}  [{lo:+6.2f}, {hi:+6.2f}]{flag:<4}{dead:14.2f}")
        print(f"  (T itself: {S[T]['S_final']:.2f}, trees={nT}; every T -g uses the same number of trees)")
        print("  CIs cover test-sampling noise only, NOT training/seed noise: treat |delta| below ~2 km/s as noise.")

    if DATES_CSV is not None:
        print_date_report(res, "R per-lead blend", others=["tcn only", "timestral only"], csv=DATES_CSV)

    pl = {k: per_lead(res, k)["S"] for k in [best_base, "R per-lead blend", T] + cal_names + ["R+clim27 (train-only)"]}
    sel = [1, 6, 12, 24, 36, 48, 60, 72]
    print("\nS_tau at selected leads:")
    print(pd.DataFrame({k: v.loc[sel] for k, v in pl.items()}).round(1).T.to_string())

    stR = fitted["R per-lead blend"][0]
    if stR.blend is not None:
        print("\nBlend weights (R, refit on train+val) at a few lead groups:")
        wt = stR.blend.weights_table()
        print(wt.iloc[[0, 3, 7, 11]].round(2).to_string() if len(wt) >= 12 else wt.round(2).to_string())
    be = fitted[T][0].be
    if be is not None and be.importance() is not None:
        print("\nTop features by gain (T):")
        print(be.importance().head(15).round(1).to_string())
    return res


# ----------------------------------------------------------------------------- CLI
def _signature(cfg):
    files = {}
    for p in [q for q in (cfg.ace_path, cfg.omni_path, cfg.ch_path, cfg.enlil_path, cfg.ts_path, cfg.c2_path, cfg.tcn_path, F.__file__) if q]:
        files[p] = (os.path.getsize(p), int(os.path.getmtime(p))) if os.path.exists(p) else None
    return dict(version=CACHE_VERSION, cfg=dataclasses.asdict(cfg), files=files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="xgb", choices=list(BACKENDS))
    ap.add_argument("--rounds", type=int, default=300, help="max tree rounds for the regularised variants")
    ap.add_argument("--legacy-rounds", type=int, default=600)
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--no-refit", action="store_true")
    ap.add_argument("--no-ablate", action="store_true")
    ap.add_argument("--cache", default=None, help="pickle path; rebuilt automatically when config/data/code change")
    ap.add_argument("--enlil-path", default=None)
    ap.add_argument("--enlil-delay", type=int, default=None, help="hours between a run's nominal time and availability")
    ap.add_argument("--enlil-drop", type=float, default=None, help="share of ENLIL runs randomly dropped (stale runs)")
    ap.add_argument("--enlil-max-age", type=int, default=None)
    ap.add_argument("--obs-latency", type=int, default=None, help="hours of ACE latency at issue time (live: ~3)")
    ap.add_argument("--lookback-days", type=int, default=None, help="live ingest keeps 30 days")
    ap.add_argument("--tcn-path", default=None, help="parquet with columns h01..h72 (e.g. tcn_v2 net_predictions_v2.parquet)")
    ap.add_argument("--tcn-offset", type=int, default=None,
                    help="hours the file's index is AHEAD of t0: 72 for the old net_predictions file, 0 for tcn_v2 output")
    ap.add_argument("--tcn-stale", type=int, default=None,
                    help="emulate ACE latency for a TCN file whose inputs ended at t0 (old file: 3). NOT for tcn_v2 files")
    ap.add_argument("--dates-csv", default=None, help="print which dates drive the test score (R) and write the table here")
    ap.add_argument("--out-preds", default="results/ladder_test_preds.pkl",
                    help="where the test predictions are saved (read by event_report.py); give each setup its own file")
    ap.add_argument("--cme-path", default=None, help="data/cme_kernel.parquet from cme_kernel.py build")
    ap.add_argument("--c2-path", default=None, help="Chronos-2 feed parquet from chronos2_feed.py (adds c2_speed_med/p90 and a blend input)")
    ap.add_argument("--strip-path", default=None, help="data/strip_adj.parquet from strip_feature.py")
    ap.add_argument("--ts-path", default=None, help="Timestral speed feature parquet (t0, lead_time, timestral_med, timestral_spread); default data/timestral_speed_features.parquet")
    ap.add_argument("--t0-start", default="2025-01-01")
    a = ap.parse_args()
    global DATES_CSV, CME_PATH, STRIP_PATH
    DATES_CSV = a.dates_csv
    CME_PATH = a.cme_path
    STRIP_PATH = a.strip_path

    cfg = Cfg(t0_start=a.t0_start)
    over = dict(enlil_path=a.enlil_path, enlil_delay_h=a.enlil_delay, enlil_drop_prob=a.enlil_drop,
                enlil_max_age_h=a.enlil_max_age, obs_latency_h=a.obs_latency, lookback_days=a.lookback_days,
                tcn_path=a.tcn_path, tcn_index_offset_h=a.tcn_offset,
                tcn_stale_h=a.tcn_stale, c2_path=a.c2_path, ts_path=a.ts_path)
    for k, v in over.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise SystemExit(f"features.Cfg has no field '{k}' - your features.py is older than this runner")
        setattr(cfg, k, v)

    sig, df = _signature(cfg), None
    if a.cache and os.path.exists(a.cache):
        try:
            blob = pickle.load(open(a.cache, "rb"))
        except Exception:
            blob = None
        if isinstance(blob, dict) and blob.get("sig") == sig:
            df = blob["df"]
            print(f"loaded cached dataset {a.cache}: {len(df):,} rows")
        else:
            print("cache is stale (config, data files or features.py changed) -> rebuilding")
    if df is None:
        t = time.time()
        df = make_features(load_raw(cfg), cfg)
        print(f"built dataset: {len(df):,} rows, {df.t0.nunique():,} issue times ({time.time() - t:.0f}s)")
        if a.cache:
            pickle.dump(dict(sig=sig, df=df), open(a.cache, "wb"))

    print("\nFeatures with the most missing values (check these first):")
    print(coverage_report(df, get_sets(df)["all"]).head(12).round(3).to_string())
    if "c2_speed_med" in df.columns:                       # Chronos-2 feed present: it becomes a blend input and an ablation group
        if "c2_speed_med" not in BLEND_COLS:
            BLEND_COLS.append("c2_speed_med")
        BLEND_OF_GROUP["c2"] = ["c2_speed_med"]
        if "c2" not in ABLATE:
            ABLATE.append("c2")
    res = run_ladder(df, a.backend, a.rounds, a.legacy_rounds, a.step, a.boot, not a.no_refit, not a.no_ablate)
    os.makedirs(os.path.dirname(a.out_preds) or ".", exist_ok=True)
    pickle.dump(res, open(a.out_preds, "wb"))
    print(f"\ntest predictions saved to {a.out_preds}   (python event_report.py --pred {a.out_preds})")


if __name__ == "__main__":
    main()
