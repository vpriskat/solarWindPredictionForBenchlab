"""numpy-only tests for windows.py:   python test_windows.py"""
import numpy as np
import pandas as pd

from windows import (FEATS, Aug, Grid, Scaler, WCfg, build_sample, make_tail_lut, score_matrix, tail_c,
                     tail_weight, take, valid_t0)

CFG = WCfg(clip_z=1e12)        # identity scaler + unique V ramp: do not clip
H, R, S = CFG.horizon, CFG.recent, CFG.seg


def make_df(n=4000, seed=0, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(rng.normal(size=(n, 5)) * 3 + 10, index=idx, columns=FEATS)
    df["V"] = np.arange(n, dtype=float) + 400.0            # unique, so alignment is checkable
    df["N"] = rng.gamma(2, 3, n)
    if nan_frac:
        df = df.mask(rng.random((n, 5)) < nan_frac)
        df["V"] = df["V"].mask(rng.random(n) < nan_frac)
    return df


def ident_scaler():
    return Scaler(np.zeros(5), np.ones(5))


def unz(g, z):                                              # scaled V -> km/s (identity scaler)
    return z * g.scaler.std[0] + g.scaler.mean[0]


def test_take():
    a = np.arange(10.0)[:, None]
    assert take(a, -3, 5)[:, 0].tolist() == [0, 0, 0, 0, 1]
    assert take(a, 8, 5)[:, 0].tolist() == [8, 9, 0, 0, 0]
    assert take(a, 20, 3, fill=np.nan).shape == (3, 1) and np.isnan(take(a, 20, 3, fill=np.nan)).all()
    print("ok  take")


def test_alignment_and_latency():
    df = make_df()
    g = Grid(df, ident_scaler(), CFG)
    i0 = 2500
    s = build_sample(g, i0, CFG)
    V = g.v
    # target
    assert np.allclose(s["yv"], V[i0 + 1:i0 + 1 + H])
    # last recent hour is e = i0 - latency
    e = i0 - CFG.latency
    assert np.isclose(unz(g, s["xr"][-1, 0]), V[e])
    assert np.isclose(unz(g, s["base"]), V[e]) and np.isclose(s["scal"][2], CFG.latency / 72)
    # recurrence: position pad+h-1 is V(t0+h-P)
    for k, P in enumerate(CFG.periods):
        for h in (1, 24, 72):
            assert np.isclose(unz(g, s["xrec"][k, CFG.pad + h - 1, 0]), V[i0 + h - P]), (P, h)
            assert np.isclose(unz(g, s["fill"][k, h - 1]), V[i0 + h - P])
    # the whole recurrence segment is strictly older than the last available hour
    assert i0 - CFG.periods[0] + H + CFG.pad < e
    print("ok  alignment + latency")


def test_causality():
    """Replace EVERYTHING after the last available hour: inputs must not change, targets must."""
    df = make_df(nan_frac=0.05)
    i0 = 2600
    e = i0 - CFG.latency
    df2 = df.copy()
    rng = np.random.default_rng(1)
    df2.iloc[e + 1:] = rng.normal(size=df2.iloc[e + 1:].shape) * 100 + 500
    s1 = build_sample(Grid(df, ident_scaler(), CFG), i0, CFG)
    s2 = build_sample(Grid(df2, ident_scaler(), CFG), i0, CFG)
    for k in ("xr", "xrec", "scal", "base", "fill"):
        assert np.array_equal(s1[k], s2[k]), f"look-ahead through {k}"
    assert not np.array_equal(s1["yv"], s2["yv"])
    print("ok  no look-ahead (inputs identical when all data after t0-latency is replaced)")


def test_nan_handling():
    df = make_df()
    i0 = 2500
    e = i0 - CFG.latency
    df.iloc[e - 5:e + 1, 0] = np.nan                         # last 6 V hours missing
    df.iloc[e - 100:e - 90, 1] = np.nan                      # a B gap
    g = Grid(df, ident_scaler(), CFG)
    s = build_sample(g, i0, CFG)
    assert (s["xr"][-6:, 0] == 0).all() and (s["xr"][-6:, 5] == 0).all() and s["xr"][-7, 5] == 1
    assert np.isclose(unz(g, s["base"]), g.v[e - 6])        # persistence = last VALID hour
    assert np.isclose(s["scal"][2], (CFG.latency + 6) / 72)
    assert (s["xr"][R - 100 - 1:R - 90 - 1, 6] == 0).all()   # BX mask
    assert np.isfinite(s["xr"]).all() and np.isfinite(s["xrec"]).all()
    print("ok  NaN -> zero + mask, persistence = last valid")


def test_edges_and_live():
    df = make_df(n=3000)
    g = Grid(df, ident_scaler(), CFG)
    s = build_sample(g, 100, CFG)                            # not enough history: padded with masked zeros
    assert s["xrec"][1, :, 5:].sum() == 0 and s["ym"].sum() == H
    # live: grid ends at the newest obs; t0 is 3 h later, targets unknown
    t_last = df.index[-1]
    t0 = t_last + pd.Timedelta(hours=CFG.latency)
    gl = Grid(df, ident_scaler(), CFG, t_end=t0 + pd.Timedelta(hours=H))
    i0 = gl.index_of(t0)
    s = build_sample(gl, i0, CFG)
    assert s["ym"].sum() == 0 and np.isclose(unz(gl, s["xr"][-1, 0]), df["V"].iloc[-1])
    assert np.isclose(s["scal"][2], CFG.latency / 72)
    print("ok  edges + live extension")


def test_valid_t0():
    df = make_df(nan_frac=0.2)
    g = Grid(df, ident_scaler(), CFG)
    idx = valid_t0(g, CFG, step=6)
    assert len(idx) > 100 and idx.min() >= CFG.min_hist + CFG.latency
    assert idx.max() < len(g) - H
    assert (np.diff(idx) >= 6).all()
    print(f"ok  valid_t0 ({len(idx)} issue times)")


def test_augmentation():
    df = make_df()
    g = Grid(df, ident_scaler(), CFG)
    aug = Aug()
    rng = np.random.default_rng(3)
    plain = build_sample(g, 2500, CFG)
    fr, lat = [], set()
    for _ in range(200):
        s = build_sample(g, 2500, CFG, rng, aug)
        assert s["xr"].shape == (R, 10) and s["xrec"].shape == (2, S, 10) and s["fill"].shape == (2, H)
        assert (s["xr"][:, :5] * (1 - s["xr"][:, 5:]) == 0).all()      # masked => zero value
        fr.append(1 - s["xr"][:, 5].mean())
        lat.add(int(round(s["scal"][2] * 72)))                         # persist age = latency (+ gap)
        assert np.array_equal(s["yv"], plain["yv"])                    # targets never augmented
    nb = [build_sample(g, 2500, CFG, rng, aug) for _ in range(300)]
    b_off = np.mean([(s["xr"][:, 6:9].sum() == 0) and (s["xrec"][:, :, 6:9].sum() == 0) for s in nb])
    assert 0.08 < b_off < 0.25, b_off                                   # ~p_chan_b
    assert 0.05 < np.mean(fr) < 0.40 and {2, 3, 4} <= lat
    print(f"ok  augmentation (mean dropped fraction {np.mean(fr):.2f})")


def test_tail_weight_and_score():
    rng = np.random.default_rng(0)
    v = 380 + rng.gamma(3, 60, 20000)
    lut = make_tail_lut(v)
    w = tail_weight(np.array([350., 450., 600., 900., np.nan]), np.array([1, 1, 1, 1, 0.]), lut)
    assert w[0] <= w[1] <= w[2] <= w[3] and w[3] <= 6.0 and w[4] == 1.0 and w[0] >= 1.0
    c = tail_c(v)
    assert abs(c.sum() - 1.0) < 1e-6                                   # sum c = 1 (TIMSE is a weighted mean)
    obs = rng.normal(450, 80, (500, H))
    t, S_ = score_matrix(obs, obs + 10)
    assert np.allclose(t["RMSE"], 10) and np.isfinite(S_)
    print(f"ok  tail weights (range {w[:4].min():.2f}-{w[:4].max():.2f}), score_matrix")


def test_drop_feats():
    df = make_df(nan_frac=0.02)
    cfg = WCfg(clip_z=1e12, drop=("BX_GSE", "BY_GSM", "BZ_GSM"))
    g = Grid(df, ident_scaler(), cfg)
    s = build_sample(g, 2500, cfg)
    assert s["xr"][:, [1, 2, 3, 6, 7, 8]].sum() == 0 and s["xrec"][:, :, [1, 2, 3, 6, 7, 8]].sum() == 0
    assert s["xr"][:, 5].sum() > 0 and s["xr"][:, 0].std() > 0                     # V and N untouched
    assert WCfg.from_dict(cfg.to_dict()).drop == cfg.drop
    old = cfg.to_dict(); old.pop("drop")                                            # checkpoints from before this flag
    assert WCfg.from_dict(old).drop == ()
    print("ok  drop feature columns + checkpoint compat")


if __name__ == "__main__":
    for f in (test_take, test_alignment_and_latency, test_causality, test_nan_handling, test_edges_and_live,
              test_valid_t0, test_augmentation, test_tail_weight_and_score, test_drop_feats):
        f()
    print("\nall windows tests passed")
