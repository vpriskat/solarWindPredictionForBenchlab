"""Do the new coronal-hole longitude strips predict solar-wind speed BEYOND what persistence + 27-day recurrence already know?
Long history (2010-2024), so the answer does not depend on the short 2025-26 stacker window.

  python segment_ch.py --start 2010-06-01 --end 2024-10-01 --hours 0000 --out data/ch_new        # hours of CPU; can be split by year
  python strip_test.py --strips data/ch_new \
      --obs ../benchlabSolarWindPrediction/data/omni_2020_2024_hourly.parquet:speed \
      --obs ../benchlabSolarWindPrediction/data/swpc_ace_historical/swpc_ace_swepam_hourly.parquet:speed_km_s

Per lead (24, 48, 72 h): baseline = ridge on [last speed, speed 27 d before the target, 27-day mean speed];
augmented = baseline + ridge on the strips of the images at t0, t0-1d, t0-2d, t0-3d (residual model, strong shrinkage).
Leave-one-year-out (training rows within 4 days of the held-out year are removed); S_tau of the pooled out-of-year predictions,
paired 5-day block bootstrap for the difference.  Negative delta = the images help.
"""
import argparse
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from score_tools import bootstrap_matrix, score

LEADS = (24, 48, 72)
LAGS_H = (0, 24, 48, 72)


def load_strips(folder):
    parts = [pd.read_csv(f) for f in sorted(glob.glob(f"{folder}/strips_*.csv"))]
    d = pd.concat(parts, ignore_index=True)
    d["t_obs"] = pd.to_datetime(d["t_obs"])
    d = d.sort_values("t_obs").drop_duplicates("t_obs").set_index("t_obs")
    cols = [c for c in d.columns if c.startswith(("chf_eq", "int_eq")) or c in ("chf_north", "chf_south")]
    return d[cols]


def build(obs, strips, t0s):
    """One row per (t0, lead): baseline predictors, target, image features of the images at t0 - lag."""
    obs = obs.resample("h").mean()
    p6 = obs.rolling(6, min_periods=1).mean()
    m27 = obs.rolling(648, min_periods=300).mean()
    feats = {}
    for lag in LAGS_H:
        q = pd.DataFrame({"t0": t0s - pd.Timedelta(hours=lag)}).sort_values("t0")
        j = pd.merge_asof(q, strips.rename_axis("ti").reset_index(), left_on="t0", right_on="ti",
                          direction="backward", tolerance=pd.Timedelta(hours=18))
        j.index = q.index
        feats[lag] = j.drop(columns=["t0", "ti"]).add_suffix(f"_L{lag}").reindex(range(len(t0s)))
    X_img = pd.concat([feats[l].reset_index(drop=True) for l in LAGS_H], axis=1)
    rows = []
    for h in LEADS:
        y = obs.reindex(t0s + pd.Timedelta(hours=h)).to_numpy()
        rec = obs.reindex(t0s + pd.Timedelta(hours=h - 648)).to_numpy()
        df = pd.DataFrame({"t0": t0s, "lead_time": h, "obs": y, "persistence": p6.reindex(t0s).to_numpy(),
                           "rec27": rec, "m27": m27.reindex(t0s).to_numpy()})
        df["rec27"] = df["rec27"].fillna(df["persistence"])
        rows.append(pd.concat([df, X_img], axis=1))
    D = pd.concat(rows, ignore_index=True)
    return D.dropna(subset=["obs", "persistence", "m27"]).reset_index(drop=True)


def cv(D, alphas=(100.0, 1000.0, 10000.0), years=range(2012, 2025), purge_days=4):
    img_cols = [c for c in D.columns if "_L" in c]
    out = D[["t0", "lead_time", "obs", "persistence"]].copy()
    out["base"] = np.nan
    for a in alphas:
        out[f"img_a{a:g}"] = np.nan
    yr = D["t0"].dt.year
    for h in LEADS:
        ih = D["lead_time"] == h
        for Y in years:
            te = ih & (yr == Y)
            if te.sum() < 50:
                continue
            lo, hi = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=purge_days), pd.Timestamp(f"{Y}-12-31") + pd.Timedelta(days=purge_days)
            tr = ih & ~((D["t0"] >= lo) & (D["t0"] <= hi))
            B = ["persistence", "rec27", "m27"]
            base = Ridge(alpha=1.0).fit(D.loc[tr, B], D.loc[tr, "obs"])
            pb_tr, pb_te = base.predict(D.loc[tr, B]), base.predict(D.loc[te, B])
            out.loc[te, "base"] = pb_te
            res = D.loc[tr, "obs"].to_numpy() - pb_tr
            sc = StandardScaler().fit(D.loc[tr, img_cols].fillna(0.0))
            Xtr = sc.transform(D.loc[tr, img_cols].fillna(0.0))
            Xte = sc.transform(D.loc[te, img_cols].fillna(0.0))
            for a in alphas:
                out.loc[te, f"img_a{a:g}"] = pb_te + Ridge(alpha=a).fit(Xtr, res).predict(Xte)
    return out.dropna(subset=["base"])


def report(out, boot=100):
    cols = [c for c in out.columns if c.startswith("img_a")]
    names = ["persistence", "base"] + cols
    B = bootstrap_matrix(out, names, n_boot=boot)
    ix = {n: i for i, n in enumerate(names)}
    print(f"\nleave-one-year-out, {len(out) // len(LEADS):,} issue times per lead, years {out['t0'].dt.year.min()}-{out['t0'].dt.year.max()}")
    print(f"{'':<14}{'S_final':>9}   per lead S ({', '.join(str(h) for h in LEADS)})        delta vs baseline [95% CI]")
    for n in names:
        s = score(out, n)
        per = [score(out[out.lead_time == h], n)["S_final"] for h in LEADS]
        if n in ("persistence", "base"):
            d = ""
        else:
            lo, hi = np.percentile(B[:, ix[n]] - B[:, ix["base"]], [2.5, 97.5])
            d = f"{s['S_final'] - score(out, 'base')['S_final']:+7.2f} [{lo:+6.2f},{hi:+6.2f}]" + ("  <- images help" if hi < 0 else "")
        print(f"{n:<14}{s['S_final']:9.2f}   " + "  ".join(f"{x:7.2f}" for x in per) + f"        {d}")
    print("(three ridge strengths = three looks at the same data; trust a result only if it is consistent across them)")


def main():
    from donki_history import load_obs
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", default="data/ch_new")
    ap.add_argument("--obs", action="append", required=True)
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--boot", type=int, default=100)
    a = ap.parse_args()
    strips = load_strips(a.strips)
    print(f"strips: {len(strips)} images, {strips.index.min()} .. {strips.index.max()}, {strips.shape[1]} columns")
    obs = load_obs(a.obs)
    t0s = pd.date_range(a.start, a.end, freq="6h")
    D = build(obs, strips, t0s)
    print(f"{len(D):,} rows after dropping missing targets/baselines")
    report(cv(D), a.boot)


if __name__ == "__main__":
    main()
