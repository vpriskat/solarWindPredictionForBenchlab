"""Self-tests on SYNTHETIC data in your source-file schema.  They check correctness and plumbing,
not real-world skill: run `python test_synthetic.py`  (add `--ladder` to also run the full ladder)."""
import dataclasses
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import lfilter

import check_enlil as CE
import features as F
import run_experiments as R
import score_tools as S
from blend import Blend


# ----------------------------------------------------------------------------- synthetic source files
def synth_sources(seed=0):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2024-06-01", "2026-06-30 23:00", freq="h")
    n = len(hours)
    t = np.arange(n)
    rot = rng.normal(0, 0.35, n // 648 + 2)[t // 648]                       # rotation-to-rotation change
    area = 0.10 + 0.10 * np.clip(np.sin(2 * np.pi * t / 648) + 0.4 * np.sin(2 * np.pi * t / 324 + 1.0), 0, None)
    area = np.clip(area * (1 + rot), 0.01, None)
    lag = np.r_[np.full(96, area[0]), area[:-96]]                           # wind responds ~4 days after the CH
    ar = lfilter([1], [1, -0.97], rng.normal(0, 8, n))
    clean = 360 + 700 * lag + ar
    for te in rng.integers(2000, n - 500, 14):                              # CME-like spikes
        clean[te:] += 250 * np.exp(-np.arange(n - te) / 12.0)
    clean = np.clip(clean, 250, 950)
    obs_clean = pd.Series(clean, index=hours)

    ace = pd.DataFrame({"timestamp": hours.tz_localize("UTC"), "speed_km_s": clean})
    ace.loc[rng.random(n) < 0.03, "speed_km_s"] = np.nan
    ace = ace.drop(ace.index[rng.random(n) < 0.01])
    ace = pd.concat([ace, ace.iloc[:5]])                                    # duplicated timestamps

    oh = pd.date_range("2025-01-01", "2026-05-31 23:00", freq="h")
    omni = pd.DataFrame({"YEAR": oh.year, "DOY": oh.dayofyear, "HOUR": oh.hour,
                         "PLASMA_SPEED": obs_clean.reindex(oh).to_numpy(),
                         "PROTON_DENSITY": rng.gamma(3, 2, len(oh)), "BZ_GSM": rng.normal(0, 3, len(oh)),
                         "KP_INDEX": rng.integers(0, 60, len(oh)).astype(float),
                         "DST_INDEX": rng.normal(-15, 20, len(oh)), "AP_INDEX": rng.integers(0, 40, len(oh)).astype(float),
                         "F10_7_INDEX": 150 + 30 * np.sin(t[:len(oh)] / 3000)})
    omni.loc[rng.random(len(oh)) < 0.01, "KP_INDEX"] = 9999.0

    cd = pd.date_range("2024-09-01", "2026-05-31", freq="D")
    a = pd.Series(area, index=hours).reindex(cd).to_numpy()
    ch = pd.DataFrame({"t0": cd, "ch_earth_facing_weighted": a + rng.normal(0, 0.01, len(cd)),
                       "ch_equatorial_area_frac": 0.6 * a, "ch_north_south_ratio": rng.uniform(0.3, 0.7, len(cd)),
                       "ch_largest_dist_to_center": rng.uniform(0, 0.8, len(cd)),
                       "ch_largest_area_frac": 0.5 * a + rng.normal(0, 0.005, len(cd)),
                       "ch_largest_complexity": rng.uniform(5, 20, len(cd)), "ch_num_regions": rng.integers(20, 70, len(cd)).astype(float)})

    runs = pd.date_range("2025-01-01", "2026-05-30", freq="D")
    leads = np.arange(0, 97)
    R_, L_ = np.meshgrid(np.arange(len(runs)), leads, indexing="ij")
    valid = runs[R_.ravel()] + pd.to_timedelta(L_.ravel(), unit="h")
    ev = 0.85 * obs_clean.reindex(valid + pd.Timedelta(hours=6)).to_numpy() + 40 + rng.normal(0, 40, len(valid))
    enlil = pd.DataFrame({"timestamp": valid.tz_localize("UTC"), "run_date": runs[R_.ravel()].tz_localize("UTC"),
                          "forecast_horizon_hours": L_.ravel() + rng.normal(0, 1e-7, len(valid)), "earth_v1": ev * 1000})
    enlil = pd.concat([enlil, enlil.sample(frac=0.05, random_state=1)])      # duplicate rows to be averaged

    t0g = pd.date_range("2025-01-01", "2026-05-28 18:00", freq="6h")
    p0 = obs_clean.reindex(t0g).to_numpy()
    tau = np.arange(1, 73)
    ts = pd.DataFrame({"t0": np.repeat(t0g.to_numpy(), 72), "lead_time": np.tile(tau, len(t0g))})
    dec = np.exp(-ts["lead_time"].to_numpy() / 30)
    ts["timestral_med"] = 360 + (np.repeat(p0, 72) - 360) * dec + rng.normal(0, 20, len(ts))
    ts["timestral_spread"] = 5 + 0.6 * ts["lead_time"].to_numpy()
    dec2 = np.exp(-tau[None, :] / 45)
    tcn = pd.DataFrame(360 + (p0[:, None] - 360) * dec2 + rng.normal(0, 30, (len(t0g), 72)),
                       index=(t0g + pd.Timedelta(hours=72)).tz_localize("UTC"), columns=[f"h{k:02d}" for k in tau])
    return {"ace": ace, "omni": omni, "ch": ch, "enlil": enlil, "ts": ts, "tcn": tcn}, obs_clean


def patch_read_parquet(src):
    pd.read_parquet = lambda path, *a, **k: src[path].copy()
    return F.Cfg(ace_path="ace", omni_path="omni", ch_path="ch", enlil_path="enlil", ts_path="ts", tcn_path="tcn")


# ----------------------------------------------------------------------------- tests
def test_tail_c_matches_spec():
    rng = np.random.default_rng(3)

    def timse_spec(v, p):
        q = np.quantile(v, np.arange(101) / 100)
        tot = 0.0
        for k in range(100):
            w = q[k + 1] - q[k]
            if w == 0:
                continue
            m = v >= q[k]
            tot += w * np.mean((v[m] - p[m]) ** 2)
        return tot / (v.max() - v.min())

    for rounding in (None, 5.0, 25.0):                                       # rounding creates tied observations
        v = 300 + rng.gamma(3, 60, 3000)
        v = v if rounding is None else np.round(v / rounding) * rounding
        p = 0.8 * v + 70 + rng.normal(0, 40, 3000)
        a, b = timse_spec(v, p), (S.tail_c(v) * (v - p) ** 2).sum()
        assert abs(a - b) < 1e-8 * a, (rounding, a, b)
    print("ok  tail_c reproduces the spec's TIMSE (incl. tied observations)")


def test_weight_gradient():
    rng = np.random.default_rng(4)
    v = 300 + rng.gamma(3, 60, 2000)
    p = 0.8 * v + 70 + rng.normal(0, 40, 2000)
    c = S.tail_c(v)

    def Sval(pp):
        e2 = (v - pp) ** 2
        return (e2.mean() * (c * e2).sum()) ** 0.25
    base = Sval(p)
    mse, timse = ((v - p) ** 2).mean(), (c * (v - p) ** 2).sum()
    for i in rng.integers(0, 2000, 5):
        eps = 5.0
        e = v[i] - p[i]
        q = p.copy()
        q[i] = v[i] - np.sign(e) * np.sqrt(e ** 2 + eps)
        num = (Sval(q) - base) / eps
        ana = base / 4 * (1 / (2000 * mse) + c[i] / timse)
        assert abs(num - ana) < 0.02 * abs(ana) + 1e-12, (num, ana)
    print("ok  metric weights equal dS/d(err_i^2) (finite-difference check)")


def test_loader_and_shapes(cfg):
    raw = F.load_raw(cfg)
    assert raw["obs"].index.is_unique and raw["obs"].index.freq is not None or len(raw["obs"]) > 0
    assert raw["enlil"].index.is_unique and raw["ts"].index.is_unique and raw["tcn"].index.is_unique
    assert str(raw["obs"].index.dtype) == "datetime64[ns]"
    df = F.make_features(raw, cfg)
    assert df["obs"].notna().all() and (df.groupby("t0").size() <= 72).all()
    assert (df["t0"].dt.hour % 6 == 0).all() and (df["t0"].dt.minute == 0).all()
    assert ((df["target_time"] - df["t0"]).dt.total_seconds() / 3600 == df["lead_time"]).all()
    print(f"ok  loader + features: {len(df):,} rows, {df['t0'].nunique():,} issue times, {df.shape[1]} columns")
    return raw, df


def test_alignment(raw, df):
    obs = raw["obs"]
    r = df.sample(2000, random_state=0)
    exp = obs.reindex(r["target_time"] - pd.Timedelta(days=27)).to_numpy()
    np.testing.assert_allclose(r["rec27"].to_numpy(float), exp, equal_nan=True)
    exp = raw["ch"]["ch_earth_facing_weighted"].reindex(r["target_time"] - pd.Timedelta(hours=96)).to_numpy()
    np.testing.assert_allclose(r["ch_earth_facing_weighted_lag96"].to_numpy(float), exp, rtol=1e-5, equal_nan=True)
    # ENLIL: run = same-day 00 UTC, lead = tau + hour
    row = r.iloc[0]
    run = row["t0"].floor("D")
    lead = int((row["target_time"] - run) / pd.Timedelta(hours=1))
    assert abs(raw["enlil"].get((run, lead), np.nan) - row["enlil"]) < 1e-3 or np.isnan(row["enlil"])
    print("ok  lead-aligned features look up the intended timestamps (rec27, CH lag, ENLIL run/lead)")


def test_no_lookahead(raw, cfg):
    rng = np.random.default_rng(5)
    for t_star in [pd.Timestamp("2025-11-20 12:00"), pd.Timestamp("2026-02-03 00:00"), pd.Timestamp("2026-04-14 18:00")]:
        a = F.make_features(raw, cfg, t0s=[t_star], drop_nan_obs=False)
        bad = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in raw.items()}
        for k in ("obs", "omni", "ch"):                                      # corrupt everything observed AFTER t0
            m = bad[k].index > t_star
            bad[k] = bad[k].astype(float)
            bad[k].loc[m] = rng.uniform(50, 2000, (m.sum(),) + bad[k].shape[1:]) if bad[k].ndim > 1 else rng.uniform(50, 2000, m.sum())
        e = bad["enlil"].copy()
        later = e.index.get_level_values(0) > t_star                           # ENLIL runs started after t0
        e[later] = rng.uniform(50, 2000, later.sum())
        bad["enlil"] = e
        b = F.make_features(bad, cfg, t0s=[t_star], drop_nan_obs=False)
        cols = [c for c in a.columns if c not in ("obs",) and a[c].dtype.kind == "f"]
        np.testing.assert_allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True, rtol=1e-6)
        assert not np.allclose(a["obs"], b["obs"], equal_nan=True), "corruption was ineffective"
    print("ok  no look-ahead: corrupting ACE/OMNI/CH after t0 and ENLIL runs after t0 changes no feature")


def test_scorer_baselines(df):
    d = df.copy()
    d["pred"] = d["rec27"]
    s = S.score(d, "pred")
    assert np.isfinite(s["S_final"]) and s["S_1_24"] > 0
    d["pred"] = np.nan                                                      # all NaN -> equals persistence
    assert abs(S.score(d, "pred")["S_final"] - S.score(d.assign(pred=d["persistence"]), "pred")["S_final"]) < 1e-9
    print("ok  scorer: NaN predictions fall back to persistence")


def test_blend():
    rng = np.random.default_rng(7)
    n = 60_000
    x1 = 400 + rng.normal(0, 80, n)
    x2 = 400 + rng.normal(0, 80, n) + 0.5 * (x1 - 400)
    d = pd.DataFrame(dict(lead_time=rng.integers(1, 73, n), x1=x1, x2=x2,
                          persistence=0.5 * x1 + 0.5 * x2 + rng.normal(0, 10, n),
                          obs=0.6 * x1 + 0.4 * x2 + 30 + rng.normal(0, 20, n)))
    b = Blend(["x1", "x2"], width=12, lam=1e-6).fit(d)
    for c, beta in b.params.values():
        assert abs(beta[0] - 0.6) < 0.03 and abs(beta[1] - 0.4) < 0.03 and abs(c - 30) < 8, (c, beta)
    d2 = d.copy()
    d2.loc[rng.random(n) < 0.1, "x2"] = np.nan                                # missing feed -> persistence
    assert np.isfinite(b.predict(d2)).all()
    b2 = Blend(["x1", "x2"], width=12, lam=1e6).fit(d)                        # huge ridge -> equal weights
    for c, beta in b2.params.values():
        assert abs(beta[0] - 0.5) < 0.02 and abs(beta[1] - 0.5) < 0.02, beta
    print("ok  blend recovers weights, imputes missing feeds with persistence, shrinks to equal weights")


def test_sfinal_and_bootstrap(df):
    d = df.copy()
    d["pred"] = d["rec27"].fillna(d["persistence"]).fillna(d["obs"].mean())
    a = S.score(d, "pred")["S_final"]
    b = S.sfinal_arrays(d["lead_time"].to_numpy(), d["obs"].to_numpy(float), d["pred"].to_numpy(float))
    assert abs(a - b) < 1e-6 * a, (a, b)
    d["p2"] = d["pred"] + 25.0
    m = S.bootstrap_matrix(d, ["pred", "p2"], n_boot=20)
    assert m.shape == (20, 2)
    full = S.score(d, "p2")["S_final"] - S.score(d, "pred")["S_final"]        # (+25 km/s can IMPROVE S: the metric
    assert abs((m[:, 1] - m[:, 0]).mean() - full) < 1.0, (full, (m[:, 1] - m[:, 0]).mean())   # rewards tail over-prediction)
    print("ok  sfinal_arrays matches score(); bootstrap_matrix paired differences match full-data differences")


def test_latency_no_lookahead(raw, cfg):
    if not hasattr(cfg, "obs_latency_h"):
        print("skip latency test (features.Cfg has no obs_latency_h)")
        return
    L = 3
    c3, c0 = dataclasses.replace(cfg, obs_latency_h=L), dataclasses.replace(cfg, obs_latency_h=0)
    rng = np.random.default_rng(6)
    for t_star in [pd.Timestamp("2025-11-20 12:00"), pd.Timestamp("2026-02-03 00:00"), pd.Timestamp("2026-04-14 18:00")]:
        o = raw["obs"].copy()
        m = o.index > t_star - pd.Timedelta(hours=L)                           # ACE not yet available at t0 (latency L)
        o[m] = rng.uniform(50, 2000, m.sum())
        bad = dict(raw, obs=o)
        a3 = F.make_features(raw, c3, t0s=[t_star], drop_nan_obs=False)
        b3 = F.make_features(bad, c3, t0s=[t_star], drop_nan_obs=False)
        cols = [c for c in a3.columns if c != "obs" and a3[c].dtype.kind == "f"]
        np.testing.assert_allclose(a3[cols].to_numpy(float), b3[cols].to_numpy(float), equal_nan=True, rtol=1e-6)
        a0 = F.make_features(raw, c0, t0s=[t_star], drop_nan_obs=False)          # sensitivity: without latency it leaks
        b0 = F.make_features(bad, c0, t0s=[t_star], drop_nan_obs=False)
        assert not np.allclose(a0[cols].to_numpy(float), b0[cols].to_numpy(float), equal_nan=True)
    print("ok  obs_latency_h=3: ACE data newer than t0-3h changes no feature (and would without the latency)")


def test_tcn_stale(raw, cfg):
    import copy
    c0, c3 = copy.deepcopy(cfg), copy.deepcopy(cfg)
    c0.tcn_stale_h, c3.tcn_stale_h = 0, 3
    a, b = F.load_raw(c0)["tcn"], F.load_raw(c3)["tcn"]
    t0 = a.index.get_level_values(0)[len(a) // 2]
    t3 = t0 + pd.Timedelta(hours=3)
    for k in (1, 24, 69):                                   # forecast issued at t0 for V(t0+k+3) == used at t0+3 for lead k
        assert np.isclose(b[(t3, k)], a[(t0, k + 3)])
    assert (t3, 70) not in b.index and b.index.get_level_values(1).max() == 69
    print("ok  tcn_stale_h shifts rows/columns consistently (row t0-L, column k+L -> lead k)")


def test_tailcal():
    from tailcal import TailCal
    rng = np.random.default_rng(5)
    n = 4000
    lead = np.tile(np.arange(1, 73), n // 72 + 1)[:n]
    v = 380 + rng.gamma(2.5, 70, n)
    p = 450 + 0.55 * (v - 450) + rng.normal(0, 40, n)                    # shrinks toward the mean, loses the fast tail
    d = pd.DataFrame(dict(lead_time=lead, obs=v, persistence=p))
    before = S.score(d.assign(pred=p))["S_final"]
    cal = TailCal(((1, 72),), True).fit(d, p)
    after = S.score(d.assign(pred=cal.predict(d, p)))["S_final"]
    assert cal.params[(1, 72)]["b"] > 1.2 and after < before - 3, (cal.params, before, after)
    good = 450 + (v - 450) + rng.normal(0, 40, n)                        # already calibrated: b should stay near 1
    c2 = TailCal(((1, 72),), True).fit(d, good)
    assert abs(c2.params[(1, 72)]["b"] - 1.0) < 0.15, c2.params
    nan = p.copy(); nan[:5] = np.nan
    assert np.isnan(cal.predict(d, nan)[:5]).all()                       # NaN stays NaN -> scorer's persistence fallback
    print(f"ok  tailcal: S {before:.1f} -> {after:.1f} (b={cal.params[(1, 72)]['b']:.2f}); calibrated input keeps b~1")


def test_date_attribution():
    from date_report import attribution, date_report
    rng = np.random.default_rng(2)
    t0 = pd.date_range("2026-01-01", periods=240, freq="6h")
    rows = []
    for t in t0:
        for tau in range(1, 73):
            rows.append((t, tau))
    d = pd.DataFrame(rows, columns=["t0", "lead_time"])
    base = 420 + 60 * np.sin(np.arange(len(d)) / 900.0)
    d["obs"] = base + rng.gamma(2, 25, len(d))
    d["persistence"] = d["obs"] + rng.normal(0, 60, len(d))
    d["pred"] = d["obs"] + rng.normal(0, 30, len(d))
    ev = (d["t0"] + pd.to_timedelta(d["lead_time"], unit="h")).dt.floor("D") == pd.Timestamp("2026-01-20")
    d.loc[ev, "pred"] -= 150                                              # a missed fast stream on one day
    a = attribution(d)
    assert abs(a["attr"].sum() - 1.0) < 1e-9
    out, s = date_report(d, "pred", ["persistence"])
    assert out.index[0] == pd.Timestamp("2026-01-20") and out["share_%"].iloc[0] > 3 * out["share_%"].iloc[5]
    base_S = S.score(d, "pred")["S_final"]                                # Euler check: scale the errors of the top day by (1+delta)
    delta = 0.01
    d2 = d.copy()
    err = d2.loc[ev, "pred"] - d2.loc[ev, "obs"]
    d2.loc[ev, "pred"] = d2.loc[ev, "obs"] + (1 + delta) * err
    dS = S.score(d2, "pred")["S_final"] - base_S
    expect = delta * base_S * out["share_%"].iloc[0] / 100.0
    assert abs(dS - expect) < 0.05 * abs(expect), (dS, expect)
    print(f"ok  date attribution: sums to 1, finds the injected event ({out['share_%'].iloc[0]:.1f}% of S), "
          f"first-order check dS={dS:.4f} vs {expect:.4f}")


def test_event_report():
    from event_report import _split, table_scores, anatomy, regimes
    rng = np.random.default_rng(5)
    t0 = pd.date_range("2026-01-01", periods=200, freq="6h")
    d = pd.DataFrame([(t, k) for t in t0 for k in range(1, 73)], columns=["t0", "lead_time"])
    d["obs"] = 420 + rng.gamma(2, 30, len(d))
    d["persistence"] = d["obs"] + rng.normal(0, 60, len(d))
    d["R per-lead blend"] = d["obs"] + rng.normal(0, 40, len(d))
    d["enlil only"] = d["obs"] + rng.normal(0, 70, len(d))
    r, ins = _split(d, "2026-01-20:2026-01-22")
    ev = ins.to_numpy()
    d.loc[ev, "R per-lead blend"] -= 300
    r, ins = _split(d, "2026-01-20:2026-01-22")
    t = table_scores(r, ins, ["persistence", "R per-lead blend"])
    assert t.loc["R per-lead blend", "event cost"] > 20 and abs(t.loc["persistence", "event cost"]) < 10
    a = anatomy(r, "2026-01-20T00:2026-01-21T00", ["persistence", "enlil only", "R per-lead blend"])
    assert "enlil max" in a.columns and len(a) >= 2
    g = regimes(r, ins, ["R per-lead blend"])
    assert set(g) == {"all", "w/o window"} and abs(g["all"]["share_%"].sum() - 100) < 1e-6
    print(f"ok  event report (event cost of the broken model {t.loc['R per-lead blend', 'event cost']:.1f} vs persistence "
          f"{t.loc['persistence', 'event cost']:.1f})")


def test_cme_kernel():
    from cme_kernel import estimate_kernel, feature
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-01", "2024-12-31", freq="h")
    v = pd.Series(420 + rng.normal(0, 20, len(idx)), index=idx)
    ev = []
    for k in range(60):
        T = pd.Timestamp("2018-02-01") + pd.Timedelta(days=40 * k)
        if T > pd.Timestamp("2024-11-30"):
            break
        real = (T + pd.Timedelta(hours=float(rng.normal(0, 6)))).round("h")
        if rng.random() < 0.7:
            t = pd.date_range(real, real + pd.Timedelta(hours=40), freq="h")
            v[t] += 200 * np.exp(-np.arange(len(t)) / 25)
        ev.append(dict(first_known=T - pd.Timedelta(hours=40), pred_first=T, last_known=T - pd.Timedelta(hours=20), pred_last=T))
    ev = pd.DataFrame(ev)
    K, n = estimate_kernel(ev, v)
    assert abs(K.loc[-24]) < 25 and 60 < K.loc[12] < 160
    t0s = pd.date_range(ev.pred_last[0] - pd.Timedelta(days=3), periods=12, freq="6h")
    f = feature(ev.iloc[:1], K, t0s)
    early = t0s < ev.first_known[0]
    assert f.shape == (12, 72) and f.values.max() > 50 and early.any() and f[early].abs().values.sum() == 0
    print(f"ok  cme kernel ({n} events, anomaly +{K.loc[12]:.0f} km/s 12 h after the predicted arrival, zero before it was known)")


def test_enlil_alignment(raw):
    tab = CE.enlil_alignment(raw["obs"], raw["enlil"])
    best = CE.best_shifts(tab)
    assert abs(best["0-23h"] - 6) <= 3, best               # synthetic ENLIL = ACE 6 h later, i.e. ENLIL early by 6 h
    print(f"ok  check_enlil recovers the built-in 6 h offset (best shifts per band: {best})")


def test_chronos_feed(src, cfg):
    """c2 feed (chronos_med/chronos_spread) is loaded, aligned on (t0, lead_time); absent => no c2 columns."""
    ts = src["ts"]
    src["c2"] = pd.DataFrame({"t0": ts["t0"], "lead_time": ts["lead_time"], "chronos_med": ts["timestral_med"] + 7.0,
                              "chronos_spread": ts["timestral_med"] + 70.0})
    cfg2 = dataclasses.replace(cfg, c2_path="c2")
    df0 = F.make_features(F.load_raw(cfg), cfg)
    assert not any(c.startswith("c2_") for c in df0.columns)
    df2 = F.make_features(F.load_raw(cfg2), cfg2)
    assert {"c2_speed_med", "c2_speed_p90"} <= set(df2.columns)
    d = df2["c2_speed_med"] - df2["ts_speed_med"]
    assert df2["c2_speed_med"].isna().equals(df2["ts_speed_med"].isna()) and np.allclose(d.dropna(), 7.0, atol=1e-3)
    assert "c2" in F.feature_groups(df2.columns) and "c2_speed_med" in F.feature_sets(df2)["lite"]
    print("ok  chronos-2 feed: loaded, aligned with Timestral rows, optional (absent -> no c2 columns)")
    return df2


if __name__ == "__main__":
    t = time.time()
    test_tail_c_matches_spec()
    test_weight_gradient()
    src, _ = synth_sources()
    cfg = patch_read_parquet(src)
    raw, df = test_loader_and_shapes(cfg)
    df_c2 = test_chronos_feed(src, cfg)          # before the tests that re-patch pd.read_parquet
    test_alignment(raw, df)
    test_no_lookahead(raw, cfg)
    test_scorer_baselines(df)
    test_blend()
    test_sfinal_and_bootstrap(df)
    test_latency_no_lookahead(raw, cfg)
    test_tcn_stale(raw, cfg)
    test_tailcal()
    test_date_attribution()
    test_event_report()
    test_cme_kernel()
    test_enlil_alignment(raw)
    print(f"all tests passed ({time.time() - t:.0f}s)")
    if "--ladder" in sys.argv:
        print("\n--- ladder on SYNTHETIC data (plumbing check only; numbers mean nothing for real data) ---")
        R.run_ladder(df, backend="hgb", rounds=100, legacy_rounds=100, step=25, boot=15, refit=True)
        print("\n--- ladder WITH a Chronos-2 feed (plumbing check) ---")
        R.BLEND_COLS.append("c2_speed_med"); R.BLEND_OF_GROUP["c2"] = ["c2_speed_med"]; R.ABLATE.append("c2")
        R.run_ladder(df_c2, backend="hgb", rounds=100, legacy_rounds=100, step=25, boot=15, refit=True)
