"""Vectorised feature pipeline for the BenchLab solar-wind stacker.

Design rules
  * One row per (t0, lead_time).  t0 is on the 6-hourly UTC grid (00/06/12/18), like the live schedule.
  * Every feature is computed from information at or before t0, EXCEPT forecast products
    (ENLIL, TCN, Timestral), which are looked up by the issue time they were produced for.
  * Features that depend on the lead time are *aligned to the target time* (t0 + tau):
      - 27/54-day recurrence of ACE speed       -> obs(t0 + tau - 27d)
      - coronal-hole features ~3-5 days earlier -> ch(t0 + tau - 72/96/120h)   (<= t0 for tau <= 72)
      - ENLIL at tau and at neighbouring leads  -> absorbs arrival-time errors
  * Missing values stay NaN (no interpolation / back-fill of ground truth, per the spec).

Items to VERIFY against your own data are marked  # VERIFY.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

H = pd.Timedelta(hours=1)

CH7 = ["ch_earth_facing_weighted", "ch_equatorial_area_frac", "ch_north_south_ratio",
       "ch_largest_dist_to_center", "ch_largest_area_frac", "ch_largest_complexity", "ch_num_regions"]
CH_LAG_COLS = ["ch_earth_facing_weighted", "ch_equatorial_area_frac",
               "ch_largest_area_frac", "ch_largest_dist_to_center"]

BASE_FEATURES = ["lead_time", "persistence", "rec27_now", "enlil", "tcn_pred", "ts_speed_med",
                 "ts_speed_p90", "c2_speed_med", "c2_speed_p90", "kp", "dst", "ap", "f107", "density", "bz"] + CH7
NON_FEATURES = ["t0", "target_time", "obs"]
OMNI_COLS = ["kp", "dst", "ap", "f107", "density", "bz", "dst_min_24h", "bz_min_24h"]
# small, physically motivated extra set (the full engineered set overfits ~9 months of training data)
ENG_LITE = ["rec27", "rec27_mean12", "rec27_max12", "d_obs_6h", "d_obs_24h", "obs_std_24h", "persist_age_h",
            "hour", "enlil_run_age", "enlil_bc", "enlil_min_pm12", "enlil_max_pm12", "base_spread",
            "ch_earth_facing_weighted_lag96", "ch_earth_facing_weighted_max72_120"]


@dataclass
class Cfg:
    ace_path: str = "../benchlabSolarWindPrediction/data/swpc_ace_historical/swpc_ace_swepam_hourly.parquet"
    omni_path: str = "../benchlabSolarWindPrediction/data/omni_features_data_2025_january_to_august_2026.parquet"
    ch_path: str = "data/ch_advanced_physics.parquet"
    # point this at the re-downloaded 0-96h ENLIL file
    enlil_path: str = "../benchlabSolarWindPrediction/data/curated/enlil_2025_2026_0_72h.parquet"
    ts_path: str = "data/timestral_speed_features.parquet"
    c2_path: str | None = None           # Chronos-2 feed (chronos2_feed.py): t0, lead_time, chronos_med, chronos_spread; None = not used
    tcn_path: str = "../benchlabSolarWindPrediction/results/net_predictions_2025_2026.parquet"
    t0_start: str = "2025-01-01"
    t0_end: str | None = None            # default: last ACE obs - 72h
    require_coverage: bool = True        # restrict t0 to where OMNI and CH exist (as your old loop did)
    enlil_delay_h: int = 0               # VERIFY: hours between a run's nominal time and its availability
    enlil_drop_prob: float = 0.0         # simulate missing/late runs: drop this share of runs (live: stale runs)
    enlil_max_age_h: int = 72            # a run older than this is treated as unavailable (NaN)
    enlil_seed: int = 0
    obs_latency_h: int = 0               # live SWPC ACE 1h feed lags ~2-3 h: use ACE-derived t0 features as of t0-L
    lookback_days: int = 30              # live ingest only keeps 30 days -> rec54 is dropped when < 54
    tcn_index_offset_h: int = 72         # VERIFY: your old code read net_df at t0 + 72h
    tcn_stale_h: int = 0                 # emulate latency for a TCN file whose inputs ended AT t0: at t0 use the forecast
                                         # issued stale_h earlier (row t0-L, column h_{k+L} -> lead k; leads > 72-L are NaN)
    ch_lags_h: tuple = (72, 96, 120)


# ----------------------------------------------------------------------------- loading
def _ns(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.as_unit("ns") if hasattr(idx, "as_unit") else idx


def _naive_utc(s) -> pd.DatetimeIndex:
    return _ns(pd.to_datetime(s, utc=True).dt.tz_localize(None))


def load_raw(cfg: Cfg) -> dict:
    """Mirrors the loading code of xgb_ensemble_9.py, minus any interpolation of ground truth."""
    # ACE ground truth (hourly, NaN where missing)
    ace = pd.read_parquet(cfg.ace_path)
    idx = _naive_utc(ace["timestamp"])
    obs = pd.Series(ace["speed_km_s"].to_numpy(float), index=idx).sort_index()
    obs = obs[~obs.index.duplicated(keep="last")]
    off = (obs.index != obs.index.floor("h")).sum()
    if off:
        print(f"WARNING: {off} ACE timestamps are not on the hour and will be dropped")
        obs = obs[obs.index == obs.index.floor("h")]
    obs = obs.reindex(pd.date_range(obs.index.min(), obs.index.max(), freq="h").as_unit("ns"))

    # OMNI (geomagnetic indices etc.)   # VERIFY: final OMNI values; live you only get quick-look versions
    o = pd.read_parquet(cfg.omni_path)
    stamp = pd.to_datetime(o["YEAR"].astype(str) + " " + o["DOY"].astype(str) + " " + o["HOUR"].astype(str),
                           format="%Y %j %H", utc=True).dt.tz_localize(None)
    o.index = _ns(stamp)
    o = o.sort_index()
    o = o[~o.index.duplicated(keep="last")]
    o = o[["PROTON_DENSITY", "BZ_GSM", "KP_INDEX", "DST_INDEX", "AP_INDEX", "F10_7_INDEX"]]
    o = o.replace([9999.9, 9999.0, 999.9, 99.9], np.nan).ffill()
    o.columns = ["density", "bz", "kp", "dst", "ap", "f107"]

    # coronal-hole features -> hourly (forward fill = causal)
    ch = pd.read_parquet(cfg.ch_path)
    ch["t0"] = _ns(pd.to_datetime(ch["t0"]))
    ch = ch.set_index("t0").sort_index()
    ch = ch[~ch.index.duplicated(keep="last")].resample("h").ffill()

    # ENLIL: Series[(run_date, lead_hours)] -> km/s
    e = pd.read_parquet(cfg.enlil_path)
    d = pd.DataFrame({"run": _naive_utc(e["run_date"]),
                      "lead": e["forecast_horizon_hours"].round().astype(int).to_numpy(),
                      "v": (e["earth_v1"] / 1000.0).to_numpy()})
    enlil = d.groupby(["run", "lead"])["v"].mean()

    # Timestral: DataFrame[(t0, lead_time)]
    t = pd.read_parquet(cfg.ts_path)
    t["t0"] = _ns(pd.to_datetime(t["t0"]))
    t = t.rename(columns={"timestral_med": "ts_speed_med", "timestral_spread": "ts_speed_p90"})
    t["lead_time"] = t["lead_time"].astype(int)
    ts = t.groupby(["t0", "lead_time"])[["ts_speed_med", "ts_speed_p90"]].mean()

    # Chronos-2 (optional): DataFrame[(t0, lead_time)] -> c2_speed_med / c2_speed_p90
    c2 = None
    if cfg.c2_path:
        c = pd.read_parquet(cfg.c2_path)
        c["t0"] = _ns(pd.to_datetime(c["t0"]))
        c = c.rename(columns={"chronos_med": "c2_speed_med", "chronos_spread": "c2_speed_p90"})
        c["lead_time"] = c["lead_time"].astype(int)
        c2 = c.groupby(["t0", "lead_time"])[["c2_speed_med", "c2_speed_p90"]].mean()

    # TCN: Series[(t0, lead)]  (index of the file is t0 + offset)   # VERIFY the offset
    n = pd.read_parquet(cfg.tcn_path)
    nidx = _ns(pd.to_datetime(n.index, utc=True).tz_localize(None)) - pd.Timedelta(hours=cfg.tcn_index_offset_h)
    L = cfg.tcn_stale_h
    parts = [pd.DataFrame({"t0": nidx + pd.Timedelta(hours=L), "lead": int(c[1:]) - L, "v": n[c].to_numpy(float)})
             for c in n.columns if c.startswith("h") and c[1:].isdigit() and int(c[1:]) - L >= 1]
    tcn = pd.concat(parts).groupby(["t0", "lead"])["v"].mean()

    return dict(obs=obs, omni=o, ch=ch, enlil=enlil, ts=ts, tcn=tcn, c2=c2)


# ----------------------------------------------------------------------------- features
def _pick_runs(times, runs_all, delay, drop_prob, max_age_h, seed):
    """For each time, the latest ENLIL run that is available (run <= time - delay), optionally after randomly
    dropping runs (simulates outages) and rejecting runs older than max_age_h.  Returns (run, ok)."""
    runs = runs_all
    if drop_prob > 0:
        keep = np.random.default_rng(seed).random(len(runs)) >= drop_prob
        if keep.any():
            runs = runs[keep]
    rv = runs.to_numpy()
    pos = np.searchsorted(rv, (times - delay).to_numpy(), side="right") - 1
    ok = pos >= 0
    run = rv[np.where(ok, pos, 0)]
    ok &= ((times.to_numpy() - run) / np.timedelta64(1, "h")) <= max_age_h
    return pd.DatetimeIndex(np.where(ok, run, np.datetime64("NaT"))).as_unit("ns"), ok


def make_features(raw: dict, cfg: Cfg, t0s=None, drop_nan_obs: bool = True) -> pd.DataFrame:
    obs, omni, ch, enlil = raw["obs"], raw["omni"], raw["ch"], raw["enlil"]
    delay = pd.Timedelta(hours=cfg.enlil_delay_h)

    if t0s is None:
        lo = pd.Timestamp(cfg.t0_start)
        hi = pd.Timestamp(cfg.t0_end) if cfg.t0_end else obs.index.max() - 72 * H
        if cfg.require_coverage:
            lo = max(lo, omni.index.min(), ch.index.min())
            hi = min(hi, omni.index.max(), ch.index.max())
        t0s = pd.date_range(lo.ceil("6h"), hi, freq="6h")
    t0s = _ns(t0s)
    n0 = len(t0s)

    # ---- features at t0 (ACE-derived ones are evaluated at t0a = t0 - latency) ----------------------
    t0a = _ns(t0s - pd.Timedelta(hours=cfg.obs_latency_h))
    obs_ff = obs.ffill()
    last_valid = pd.Series(np.where(obs.notna(), obs.index.to_numpy(), np.datetime64("NaT")), index=obs.index).ffill()
    f0 = pd.DataFrame(index=t0s)
    f0["persistence"] = obs_ff.reindex(t0a).to_numpy()        # latest valid ACE value (spec definition)
    f0["persist_age_h"] = ((t0s - pd.DatetimeIndex(last_valid.reindex(t0a).to_numpy())) / H).to_numpy()
    for h in (6, 12, 24, 48):
        f0[f"d_obs_{h}h"] = f0["persistence"].to_numpy() - obs_ff.shift(h).reindex(t0a).to_numpy()
    for w in (24, 72):
        r = obs.rolling(w, min_periods=w // 3)
        f0[f"obs_max_{w}h"] = r.max().reindex(t0a).to_numpy()
        f0[f"obs_min_{w}h"] = r.min().reindex(t0a).to_numpy()
        f0[f"obs_std_{w}h"] = r.std().reindex(t0a).to_numpy()
    f0["clim27"] = obs.rolling(27 * 24, min_periods=200).mean().reindex(t0a).to_numpy()   # trailing 27-day mean: regime level
    f0["rec27_now"] = obs.shift(27 * 24).reindex(t0a).to_numpy()   # recurrence value valid at t0a (= old `rotation`)
    f0["hour"] = t0s.hour.to_numpy()

    om = omni.reindex(t0s)
    for c in ["kp", "dst", "ap", "f107", "density", "bz"]:
        f0[c] = om[c].to_numpy()
    f0["dst_min_24h"] = omni["dst"].rolling(24, min_periods=1).min().reindex(t0s)
    f0["bz_min_24h"] = omni["bz"].rolling(24, min_periods=1).min().reindex(t0s)

    chc = ch.reindex(t0s)
    ch7 = [c for c in CH7 if c in ch.columns]
    for c in ch7:
        f0[c] = chc[c].to_numpy()

    # ---- ENLIL nowcast error series: how wrong was the latest available run, hour by hour -----
    runs_all = _ns(enlil.index.get_level_values(0).unique().sort_values())
    pol = (delay, cfg.enlil_drop_prob, cfg.enlil_max_age_h, cfg.enlil_seed)
    u = obs.index
    run_u, ok_u = _pick_runs(u, runs_all, *pol)
    lead_u = np.where(ok_u, np.rint(((u - run_u) / H).to_numpy()), -10**6).astype(np.int64)
    now = pd.Series(enlil.reindex(pd.MultiIndex.from_arrays([run_u, lead_u])).to_numpy(), index=u)
    err = obs - now
    f0["enlil_err_now"] = err.reindex(t0a).to_numpy()
    f0["enlil_bias24"] = err.rolling(24, min_periods=6).mean().reindex(t0a).to_numpy()
    f0["enlil_bias72"] = err.rolling(72, min_periods=18).mean().reindex(t0a).to_numpy()

    # ---- long frame: one row per (t0, tau) --------------------------------------------------------
    rep = np.repeat(np.arange(n0), 72)
    tau = np.tile(np.arange(1, 73), n0)
    t0_l = t0s[rep]
    tgt = _ns(t0_l + pd.to_timedelta(tau, unit="h"))
    df = f0.iloc[rep].reset_index(drop=True)
    df.insert(0, "lead_time", tau)
    df.insert(0, "target_time", tgt)
    df.insert(0, "t0", t0_l)
    df["obs"] = obs.reindex(tgt).to_numpy()

    def at(s, times):
        return s.reindex(times).to_numpy()

    # lead-aligned 27/54-day recurrence (the official 27-day baseline is exactly `rec27`)
    d27, d54 = pd.Timedelta(days=27), pd.Timedelta(days=54)
    df["rec27"] = at(obs, tgt - d27)
    if cfg.lookback_days >= 54:                                       # live ingest keeps only 30 days
        df["rec54"] = at(obs, tgt - d54)
    win = obs.rolling(25, center=True, min_periods=8)
    df["rec27_mean12"] = at(win.mean(), tgt - d27)
    df["rec27_max12"] = at(win.max(), tgt - d27)
    df["rec27_minus_pers"] = df["rec27"] - df["persistence"]

    # lead-aligned coronal-hole features (look 3-5 days before the target time; always <= t0)
    lag_cols = [c for c in CH_LAG_COLS if c in ch.columns]
    ch_max = ch[lag_cols].rolling("48h", closed="both").max()       # max over [x-48h, x]
    for c in lag_cols:
        for lag in cfg.ch_lags_h:
            df[f"{c}_lag{lag}"] = at(ch[c], tgt - lag * H)
        df[f"{c}_max72_120"] = at(ch_max[c], tgt - 72 * H)

    # ENLIL at tau and neighbouring leads, using the latest run available at t0
    run_l, ok_l = _pick_runs(t0_l, runs_all, *pol)
    lead_l = np.where(ok_l, np.rint(((tgt - run_l) / H).to_numpy()), -10**6).astype(np.int64)

    def enl(shift):
        return enlil.reindex(pd.MultiIndex.from_arrays([run_l, lead_l + shift])).to_numpy()

    df["enlil"] = enl(0)
    for s in (-24, -12, -6, 6, 12, 24):
        df[f"enlil_{'m' if s < 0 else 'p'}{abs(s)}"] = enl(s)
    nb = df[["enlil_m12", "enlil", "enlil_p12"]]
    df["enlil_min_pm12"], df["enlil_max_pm12"], df["enlil_mean_pm12"] = nb.min(axis=1), nb.max(axis=1), nb.mean(axis=1)
    df["enlil_bc"] = df["enlil"] + df["enlil_bias24"]               # bias-corrected ENLIL
    df["enlil_run_age"] = ((t0_l - run_l) / H).to_numpy()

    # Timestral and TCN (keyed by issue time)
    key = pd.MultiIndex.from_arrays([t0_l, tau])
    tsx = raw["ts"].reindex(key)
    df["ts_speed_med"] = tsx["ts_speed_med"].to_numpy()
    df["ts_speed_p90"] = tsx["ts_speed_p90"].to_numpy()
    df["tcn_pred"] = raw["tcn"].reindex(key).to_numpy()
    if raw.get("c2") is not None:
        c2x = raw["c2"].reindex(key)
        df["c2_speed_med"] = c2x["c2_speed_med"].to_numpy()
        df["c2_speed_p90"] = c2x["c2_speed_p90"].to_numpy()

    # agreement between base predictions + reference predictor for residual modelling / weights
    base = df[["enlil", "tcn_pred", "ts_speed_med", "persistence", "rec27"]]
    df["base_spread"] = base.max(axis=1) - base.min(axis=1)
    df["ref"] = df[["enlil", "tcn_pred", "ts_speed_med"]].mean(axis=1).fillna(df["persistence"])

    if drop_nan_obs:                                                # spec: pairs without ground truth are excluded
        df = df[df["obs"].notna()]
    num = df.select_dtypes("float64").columns.difference(["obs"])
    df[num] = df[num].astype("float32")
    return df.reset_index(drop=True)


def feature_sets(df: pd.DataFrame):
    """-> dict(base=..., lite=base+ENG_LITE, all=base+all engineered)"""
    base = [c for c in BASE_FEATURES if c in df.columns]
    eng = [c for c in df.columns if c not in base and c not in NON_FEATURES]
    return dict(base=base, lite=base + [c for c in ENG_LITE if c in df.columns], all=base + eng)


def feature_groups(cols) -> dict:
    """Groups for ablation (--drop / --ablate).  Dropping ts/tcn/enlil also drops the derived `ref`, `base_spread`."""
    cols = list(cols)
    g = {
        "omni": [c for c in cols if c in OMNI_COLS],
        "ch": [c for c in cols if c.startswith("ch_")],
        "rec": [c for c in cols if c.startswith("rec")],
        "trend": [c for c in cols if c.startswith(("d_obs_", "obs_max_", "obs_min_", "obs_std_", "persist_age"))],
        "enlil": [c for c in cols if c.startswith("enlil")],
        "ts": [c for c in cols if c.startswith("ts_speed")],
        "c2": [c for c in cols if c.startswith("c2_speed")],
        "tcn": [c for c in cols if c == "tcn_pred"],
    }
    derived = [c for c in ("ref", "base_spread") if c in cols]
    for k in ("enlil", "ts", "tcn"):
        g[k] = g[k] + derived
    return g


def coverage_report(df: pd.DataFrame, feats) -> pd.DataFrame:
    """Fraction of NaN per feature: check this before trusting any feature."""
    return (df[feats].isna().mean().sort_values(ascending=False).to_frame("nan_frac"))
