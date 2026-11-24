"""Turn the coronal-hole longitude strips into a forecast-adjustment feature for the stacker.

NOT the TCN: a plain ridge model trained ONLY on 2012 .. 2024 (before ENLIL/Timestral exist), so nothing from the
2025-26 test window is ever used.  For every lead h = 1..72:

    base_h   = Ridge(persistence, rec27, m27)                      -> speed
    strip_adj(t0,h) = Ridge_alpha(strips of the images at t0-delay, -1d, -2d, -3d)  fitted on  (obs - base_h)

The output is the *residual* the images explain, in km/s (0 where no image is available).  It is then tested as
R + lambda * strip_adj  (lambda fixed in advance: 0.5, 1) and as an extra stacker input.

    python strip_feature.py --strips data/ch_new \
        --obs ../benchlabSolarWindPrediction/data/omni_2020_2024_hourly.parquet:speed \
        --obs ../benchlabSolarWindPrediction/data/swpc_ace_historical/swpc_ace_swepam_hourly.parquet:speed_km_s \
        --train-end 2024-12-31 --apply-start 2025-01-01 --apply-end 2026-10-31 \
        --img-delay 0 --latency 0 --alpha 1000 --out data/strip_adj.parquet

--img-delay H : hours between an image's nominal time and its availability (set it from check_aia_live.py)
--latency   H : hours of ACE latency at issue time (persistence/m27 are read at t0 - H); keep equal to the ladder's --obs-latency
Also prints a hold-out check on the apply window (leads 24/48/72): S of base vs base + strip_adj vs base + 0.5*strip_adj.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

H = pd.Timedelta(hours=1)
LAGS_H = (0, 24, 48, 72)
LEADS = tuple(range(1, 73))
REC = 27 * 24


def image_matrix(strips, t0s, delay_h, tol_h=18):
    """Image features at (t0 - delay - lag), merge_asof backward with 18 h tolerance (same as strip_test.py)."""
    parts = []
    s = strips.rename_axis("ti").reset_index()
    for lag in LAGS_H:
        q = pd.DataFrame({"t0": t0s - pd.Timedelta(hours=lag + delay_h), "k": np.arange(len(t0s))}).sort_values("t0")
        j = pd.merge_asof(q, s, left_on="t0", right_on="ti", direction="backward", tolerance=pd.Timedelta(hours=tol_h))
        j = j.sort_values("k").drop(columns=["t0", "ti", "k"]).reset_index(drop=True)
        parts.append(j.add_suffix(f"_L{lag}"))
    return pd.concat(parts, axis=1)


def base_inputs(obs, t0s, h, latency_h):
    obs = obs.resample("h").mean()
    p6 = obs.rolling(6, min_periods=1).mean()
    m27 = obs.rolling(REC, min_periods=300).mean()
    tl = t0s - latency_h * H
    pers = p6.reindex(tl).to_numpy()
    rec = obs.reindex(t0s + pd.Timedelta(hours=h - REC)).to_numpy()
    rec = np.where(np.isnan(rec), pers, rec)
    return pd.DataFrame({"persistence": pers, "rec27": rec, "m27": m27.reindex(tl).to_numpy()}), obs


def fit_apply(strips, obs, train_t0, apply_t0, delay_h=0, latency_h=0, alpha=1000.0, verbose=True, apply_strips=None, tol_h=18):
    Xi_tr = image_matrix(strips, train_t0, delay_h, tol_h)
    Xi_ap = image_matrix(strips if apply_strips is None else apply_strips, apply_t0, delay_h, tol_h)
    has_ap = Xi_ap.notna().any(axis=1).to_numpy()
    has_tr = Xi_tr.notna().any(axis=1).to_numpy()
    cols = list(Xi_tr.columns)
    sc = StandardScaler().fit(Xi_tr.loc[has_tr].fillna(0.0))
    Z_tr = sc.transform(Xi_tr.fillna(0.0))
    Z_ap = sc.transform(Xi_ap.fillna(0.0))
    out = []
    for h in LEADS:
        B_tr, obs_h = base_inputs(obs, train_t0, h, latency_h)
        B_ap, _ = base_inputs(obs, apply_t0, h, latency_h)
        y = obs_h.reindex(train_t0 + pd.Timedelta(hours=h)).to_numpy()
        ok = B_tr.notna().all(axis=1).to_numpy() & ~np.isnan(y) & has_tr          # train only where an image exists
        base = Ridge(alpha=1.0).fit(B_tr[ok], y[ok])
        res = y[ok] - base.predict(B_tr[ok])
        rm = Ridge(alpha=alpha).fit(Z_tr[ok], res)
        okp = B_ap.notna().all(axis=1).to_numpy()
        pb = np.full(len(apply_t0), np.nan)
        if okp.any():
            pb[okp] = base.predict(B_ap[okp])
        adj = np.where(has_ap, rm.predict(Z_ap), 0.0)
        out.append(pd.DataFrame({"t0": apply_t0, "lead_time": h, "strip_adj": adj, "strip_base": pb}))
        if verbose and h in (24, 48, 72):
            print(f"  lead {h:2d}: {ok.sum():,} training rows, in-sample residual R2 "
                  f"{1 - np.var(res - rm.predict(Z_tr[ok])) / np.var(res):.3f}, "
                  f"strip_adj std on apply window {adj[has_ap].std():.1f} km/s")
    return pd.concat(out, ignore_index=True), has_ap


def holdout_check(feed, obs):
    from score_tools import score
    d = feed.copy()
    d["target_time"] = pd.DatetimeIndex(d["t0"]) + pd.to_timedelta(d["lead_time"], unit="h")
    d["obs"] = obs.resample("h").mean().reindex(d["target_time"]).to_numpy()
    d["persistence"] = d["strip_base"]
    d = d.dropna(subset=["obs", "strip_base"])
    d = d[d["lead_time"].isin((24, 48, 72))].copy()
    d["adj1"] = d["strip_base"] + d["strip_adj"]
    d["adj05"] = d["strip_base"] + 0.5 * d["strip_adj"]
    print(f"\nhold-out on the apply window, {len(d) // 3:,} issue times per lead (ridge base is only persistence+rec27+m27)")
    for n, c in (("base", "strip_base"), ("base + 0.5*adj", "adj05"), ("base + adj", "adj1")):
        s = score(d.assign(pred=d[c]), "pred")
        print(f"  {n:<16} S={s['S_final']:.2f}")
    corr = np.corrcoef(d["strip_adj"], d["obs"] - d["strip_base"])[0, 1]
    print(f"  corr(strip_adj, obs - base) = {corr:+.3f}   (positive = the images explain part of the base error)")


def main():
    from donki_history import load_obs
    from strip_test import load_strips
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", default="data/ch_new")
    ap.add_argument("--obs", action="append", required=True)
    ap.add_argument("--train-start", default="2012-07-01")
    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument("--apply-start", default="2025-01-01")
    ap.add_argument("--apply-end", default="2026-10-31")
    ap.add_argument("--apply-strips", default=None,
                    help="separate strips folder used ONLY for the apply window (e.g. data/ch_nrt = near-real-time images); the model is "
                         "still trained on --strips (definitive images)")
    ap.add_argument("--chf-only", action="store_true",
                    help="drop the intensity features (int_eq_*); keep only coronal-hole fractions (more robust between NRT and definitive)")
    ap.add_argument("--img-delay", type=int, default=0)
    ap.add_argument("--img-tol", type=int, default=18, help="max age (h) of the image used, after --img-delay (use 30 with delay >= 1 and daily images)")
    ap.add_argument("--latency", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1000.0)
    ap.add_argument("--out", default="data/strip_adj.parquet")
    a = ap.parse_args()
    strips = load_strips(a.strips)
    print(f"strips: {len(strips)} images {strips.index.min()} .. {strips.index.max()}")
    if a.chf_only:
        strips = strips[[c for c in strips.columns if not c.startswith("int_eq")]]
        print(f"--chf-only: {strips.shape[1]} columns kept")
    ap_strips = load_strips(a.apply_strips) if a.apply_strips else None
    if a.chf_only and ap_strips is not None:
        ap_strips = ap_strips[strips.columns]
    if ap_strips is not None:
        print(f"apply strips: {len(ap_strips)} images {ap_strips.index.min()} .. {ap_strips.index.max()}  ({a.apply_strips})")
    obs = load_obs(a.obs)
    train_t0 = pd.date_range(a.train_start, a.train_end, freq="6h")
    apply_t0 = pd.date_range(a.apply_start, a.apply_end, freq="6h")
    last = (ap_strips if ap_strips is not None else strips).index.max()
    if last < pd.Timestamp(a.apply_end) - pd.Timedelta(days=2):
        print(f"WARNING: apply strips end {last:%Y-%m-%d}; later issue times get strip_adj = 0")
    feed, has = fit_apply(strips, obs, train_t0, apply_t0, a.img_delay, a.latency, a.alpha, apply_strips=ap_strips, tol_h=a.img_tol)
    print(f"{has.mean() * 100:.0f}% of apply issue times have at least one image within tolerance")
    feed[["t0", "lead_time", "strip_adj"]].to_parquet(a.out)
    print(f"wrote {a.out}: {len(feed):,} rows")
    holdout_check(feed, obs)


if __name__ == "__main__":
    main()
