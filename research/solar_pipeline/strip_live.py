"""Production path for the coronal-hole strip adjustment: fit ONCE, then compute strip_adj for any issue time from the
near-real-time (NRT) AIA images.  Only the images are needed at prediction time (the adjustment does not use ACE data).

  # 1) fit on everything available (definitive strips + NRT strips for the days the definitive archive does not have yet)
  python strip_live.py fit --strips data/ch_new --extra-strips data/ch_nrt --chf-only \
      --obs ../benchlabSolarWindPrediction/data/omni_2020_2024_hourly.parquet:speed \
      --obs ../benchlabSolarWindPrediction/data/swpc_ace_historical/swpc_ace_swepam_hourly.parquet:speed_km_s \
      --train-end 2026-10-31 --img-delay 1 --img-tol 30 --model data/strip_model.pkl

  # 2) every issue time (00/06/12/18 UTC): downloads the NRT images it needs (cached), writes strip_adj for leads 1..72
  python strip_live.py predict --model data/strip_model.pkl --t0 2026-11-07T06:00 --out data/strip_adj_live.parquet
  python strip_live.py predict --model data/strip_model.pkl --t0 2026-11-07T00:00 --t0-end 2026-11-09T18:00   # a range (6-hourly)

  The pipeline then uses   final = R + LAMBDA * strip_adj   (LAMBDA = 0.5 chosen in the test; rows with NaN/missing -> 0).

Behaviour that matches the test (strip_feature.py): images are the daily 00:00 UTC slot (NRT image within 10 min), the image
used for t0 is the newest one at least --img-delay hours old and at most --img-tol hours old, at lags 0/24/48/72 h; a lag with no
image contributes zeros; an issue time with no image at all gets strip_adj = 0.
"""
import argparse
import os
import pickle
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import strip_feature as F
from strip_test import load_strips

CACHE_DEFAULT = "data/ch_live/strips_live.csv"


# --------------------------------------------------------------------------------------------- fit
def fit_model(strips, obs, train_t0, delay_h=1, tol_h=30, alpha=1000.0, latency_h=0, chf_only=True):
    if chf_only:
        strips = strips[[c for c in strips.columns if not c.startswith("int_eq")]]
    Xi = F.image_matrix(strips, train_t0, delay_h, tol_h)
    has = Xi.notna().any(axis=1).to_numpy()
    sc = StandardScaler().fit(Xi.loc[has].fillna(0.0))
    Z = sc.transform(Xi.fillna(0.0))
    coefs, icpts, n_rows = [], [], []
    for h in F.LEADS:
        B, obs_h = F.base_inputs(obs, train_t0, h, latency_h)
        y = obs_h.reindex(train_t0 + pd.Timedelta(hours=h)).to_numpy()
        ok = B.notna().all(axis=1).to_numpy() & ~np.isnan(y) & has
        base = Ridge(alpha=1.0).fit(B[ok], y[ok])
        rm = Ridge(alpha=alpha).fit(Z[ok], y[ok] - base.predict(B[ok]))
        coefs.append(rm.coef_)
        icpts.append(rm.intercept_)
        n_rows.append(int(ok.sum()))
    coef, icpt = np.stack(coefs), np.array(icpts)
    adj_tr = Z[has] @ coef.T + icpt                                  # training-set adjustments, rows with images (n, 72)
    # per-feature range seen in training (standardised units), to flag unusually extreme images later
    z_lo, z_hi = np.percentile(Z[has], [0.5, 99.5], axis=0)
    return dict(strip_cols=list(strips.columns), columns=list(Xi.columns), mean=sc.mean_, scale=sc.scale_,
                adj_min=adj_tr.min(axis=0), adj_max=adj_tr.max(axis=0),
                adj_p01=np.percentile(adj_tr, 0.5, axis=0), adj_p99=np.percentile(adj_tr, 99.5, axis=0),
                z_lo=z_lo, z_hi=z_hi,
                coef=coef, intercept=icpt, delay_h=delay_h, tol_h=tol_h, alpha=alpha,
                chf_only=chf_only, latency_h=latency_h, train_rows=n_rows,
                trained_on=f"{train_t0[0]:%Y-%m-%d} .. {train_t0[-1]:%Y-%m-%d}, {len(strips)} images, "
                           f"last image {strips.index.max():%Y-%m-%d}")


def adjust(model, strips, t0s, cap=True):
    """strip_adj(t0, lead) from a strips table; zeros where no image is available.
    cap=True clips each lead to the [min, max] adjustment seen in training (a linear model can extrapolate on unusually large
    coronal holes).  Returns (long DataFrame with strip_adj and strip_adj_raw, has-image mask, per-t0 diagnostics DataFrame)."""
    st = strips[model["strip_cols"]]
    Xi = F.image_matrix(st, pd.DatetimeIndex(t0s), model["delay_h"], model["tol_h"])
    assert list(Xi.columns) == model["columns"], "feature columns differ from the fitted model"
    has = Xi.notna().any(axis=1).to_numpy()
    Z = (Xi.fillna(0.0).to_numpy() - model["mean"]) / model["scale"]
    raw = Z @ model["coef"].T + model["intercept"]                  # (n_t0, 72)
    raw[~has] = 0.0
    adj = raw.copy()
    have_range = "adj_min" in model
    if cap and have_range:
        adj = np.clip(raw, model["adj_min"], model["adj_max"])
    # diagnostics: how extreme are the features / the adjustment compared with training?
    diag = pd.DataFrame({"t0": pd.DatetimeIndex(t0s), "has_image": has})
    if "z_lo" in model:
        out_hi = (Z > model["z_hi"]) & has[:, None]
        out_lo = (Z < model["z_lo"]) & has[:, None]
        diag["n_features_outside_train_range"] = (out_hi | out_lo).sum(axis=1)
        zmax = np.where(has[:, None], np.abs(Z), 0).max(axis=1)
        diag["max_abs_z"] = zmax
        diag["worst_feature"] = [model["columns"][i] if h else "" for i, h in zip(np.abs(Z).argmax(axis=1), has)]
    if have_range:
        diag["n_leads_capped"] = (np.abs(adj - raw) > 1e-9).sum(axis=1)
        diag["n_leads_beyond_p99.5"] = (((raw > model["adj_p99"]) | (raw < model["adj_p01"])) & has[:, None]).sum(axis=1)
    out = pd.DataFrame({"t0": np.repeat(pd.DatetimeIndex(t0s).to_numpy(), len(F.LEADS)),
                        "lead_time": np.tile(np.array(F.LEADS), len(t0s)), "strip_adj": adj.ravel(),
                        "strip_adj_raw": raw.ravel()})
    return out, has, diag


# --------------------------------------------------------------------------------------------- images (NRT, cached)
def needed_days(t0, delay_h, tol_h):
    """00:00 UTC slots that can be used by t0: from (t0 - delay - 72 h - tol) up to (t0 - delay)."""
    hi = t0 - pd.Timedelta(hours=delay_h)
    lo = t0 - pd.Timedelta(hours=delay_h + 72 + tol_h)
    return list(pd.date_range(lo.normalize(), hi.normalize(), freq="D"))


def load_cache(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    d = pd.read_csv(path)
    d["t_obs"] = pd.to_datetime(d["t_obs"])
    d["slot"] = pd.to_datetime(d["slot"])
    return d.drop_duplicates("slot", keep="last")


def fetch_missing(days, cache_path, now=None, wait_min=25):
    """Download + segment the NRT 00:00 image of each day not yet in the cache (only if it can already exist)."""
    import requests
    from astropy.io import fits
    import segment_ch as S
    S.SOURCE = "nrt"
    now = now or pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    have = set(load_cache(cache_path).get("slot", pd.Series(dtype="datetime64[ns]")))
    rows, sess = [], requests.Session()
    for day in days:
        if day in have or day + pd.Timedelta(minutes=wait_min) > now:
            continue
        try:
            f = S.fetch_nrt(day, "0000", sess)
            if f is None:
                print(f"  no NRT image near {day:%Y-%m-%d} 00:00 (gap) -> that lag contributes zeros")
                rows.append(dict(t_obs=pd.NaT, slot=day))                 # remember the gap so we do not retry forever
                continue
            with fits.open(f) as hd:
                hdu = next(x for x in hd if x.data is not None and x.data.ndim == 2)
                row, _ = S.process_image(np.asarray(hdu.data, float), S.safe_header(hdu.header))
            row["slot"] = day
            rows.append(row)
            print(f"  fetched {day:%Y-%m-%d}: image at {row['t_obs']}")
        except Exception as ex:
            print(f"  {day:%Y-%m-%d}: {repr(ex)[:100]} (not cached; will retry next time)")
    if rows:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        new = pd.DataFrame(rows)
        new.to_csv(cache_path, mode="a", header=not os.path.exists(cache_path), index=False)


def strips_from_cache(cache_path):
    d = load_cache(cache_path)
    if d.empty:
        return pd.DataFrame()
    d = d.dropna(subset=["t_obs"]).sort_values("t_obs").drop_duplicates("t_obs").set_index("t_obs")
    cols = [c for c in d.columns if c.startswith(("chf_eq", "int_eq")) or c in ("chf_north", "chf_south")]
    return d[cols]


# --------------------------------------------------------------------------------------------- CLI
def cmd_fit(a):
    from donki_history import load_obs
    strips = load_strips(a.strips)
    if a.extra_strips:
        extra = load_strips(a.extra_strips)
        extra = extra[extra.index > strips.index.max()]                 # NRT only for days beyond the definitive archive
        strips = pd.concat([strips, extra]).sort_index()
        print(f"definitive images to {strips.index[len(strips) - len(extra) - 1]:%Y-%m-%d}, + {len(extra)} NRT images")
    obs = load_obs(a.obs)
    train_t0 = pd.date_range(a.train_start, a.train_end, freq="6h")
    m = fit_model(strips, obs, train_t0, a.img_delay, a.img_tol, a.alpha, a.latency, a.chf_only)
    print(f"fitted: {m['trained_on']}; rows/lead {m['train_rows'][23]:,} (24 h), {m['train_rows'][71]:,} (72 h); "
          f"{len(m['columns'])} features")
    os.makedirs(os.path.dirname(a.model) or ".", exist_ok=True)
    pickle.dump(m, open(a.model, "wb"))
    print("saved", a.model)


def cmd_predict(a):
    m = pickle.load(open(a.model, "rb"))
    t0s = pd.date_range(pd.Timestamp(a.t0), pd.Timestamp(a.t0_end or a.t0), freq="6h")
    days = sorted({d for t0 in t0s for d in needed_days(t0, m["delay_h"], m["tol_h"])})
    if not a.no_fetch:
        fetch_missing(days, a.cache)
    strips = strips_from_cache(a.cache)
    if a.extra_strips:
        strips = pd.concat([strips, load_strips(a.extra_strips)]).sort_index()
        strips = strips[~strips.index.duplicated(keep="last")]
    if strips.empty:
        print("no images available -> strip_adj = 0 for every row")
        strips = pd.DataFrame(columns=m["strip_cols"], index=pd.DatetimeIndex([], name="t_obs"), dtype=float)
    out, has, diag = adjust(m, strips, t0s, cap=not a.no_cap)
    if "adj_min" not in m:
        print("NOTE: this model file predates the range check; refit with `fit` to enable the cap and the warnings")
    print(f"{len(t0s)} issue time(s), {has.sum()} with an image; strip_adj at leads 24/48/72 for the first t0: "
          + ", ".join(f"{out[(out.t0 == t0s[0]) & (out.lead_time == h)]['strip_adj'].iloc[0]:+.1f}" for h in (24, 48, 72)) + " km/s")
    for _, d in diag.iterrows():
        msg = []
        if d.get("n_leads_capped", 0):
            msg.append(f"CAPPED at the training maximum on {int(d['n_leads_capped'])} leads")
        if d.get("n_leads_beyond_p99.5", 0):
            msg.append(f"{int(d['n_leads_beyond_p99.5'])} leads beyond the training 99.5th percentile")
        if d.get("n_features_outside_train_range", 0) >= 5:
            msg.append(f"{int(d['n_features_outside_train_range'])} features outside the training range (max |z| {d['max_abs_z']:.1f}, {d['worst_feature']})")
        if msg:
            print(f"  WARNING t0 {d['t0']:%Y-%m-%d %H:%M}: " + "; ".join(msg))
    if a.out:
        out.to_parquet(a.out) if a.out.endswith(".parquet") else out.to_csv(a.out, index=False)
        print("wrote", a.out)
    return out


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--strips", default="data/ch_new")
    f.add_argument("--extra-strips", default=None, help="NRT strips folder, used for days after the definitive archive ends")
    f.add_argument("--obs", action="append", required=True)
    f.add_argument("--train-start", default="2012-07-01")
    f.add_argument("--train-end", default="2026-10-31")
    f.add_argument("--img-delay", type=int, default=1)
    f.add_argument("--img-tol", type=int, default=30)
    f.add_argument("--alpha", type=float, default=1000.0)
    f.add_argument("--latency", type=int, default=0)
    f.add_argument("--chf-only", action="store_true")
    f.add_argument("--model", default="data/strip_model.pkl")
    f.set_defaults(fn=cmd_fit)
    p = sp.add_parser("predict")
    p.add_argument("--model", default="data/strip_model.pkl")
    p.add_argument("--t0", required=True, help="issue time, e.g. 2026-11-07T06:00 (UTC)")
    p.add_argument("--t0-end", default=None)
    p.add_argument("--cache", default=CACHE_DEFAULT)
    p.add_argument("--extra-strips", default=None, help="also use an existing strips folder (e.g. data/ch_nrt) before downloading")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--no-cap", action="store_true", help="do not clip to the training range (the raw value is always kept in strip_adj_raw)")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_predict)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
