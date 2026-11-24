"""Window builder shared by training AND inference (numpy only, no torch).

Everything the network sees is produced by `build_sample`.  The live fetch script must call exactly the same
function on the live hourly table, so train/serve input cannot drift.

Conventions
-----------
* Hourly grid, naive-UTC timestamps, NaN = missing (never interpolated).
* Issue time t0 sits at grid index i0.  Observation with grid index <= e = i0 - latency are "available"
  (ACE hourly arrives ~3 h late).  Nothing after e is ever read for the inputs.
* Targets are V(t0+h), h = 1..H.
* Recurrence segment for period P (27 d = 648 h, 54 d = 1296 h): grid indices i0 - P + 1 - pad ... i0 - P + H + pad.
  Segment position (pad + h - 1) is the value that "recurs" at lead h  (V(t0 + h - P)).
* Channels per hour: 5 scaled features (V, BX, BY, BZ, log1p(N)) + 5 mask channels (1 = observed).
"""
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

FEATS = ["V", "BX_GSE", "BY_GSM", "BZ_GSM", "N"]
N_SCAL = 6


@dataclass
class WCfg:
    horizon: int = 72
    latency: int = 3
    recent: int = 168                 # hours of recent history (7 d)
    pad: int = 48                     # margin around the recurrence window (+-2 d)
    periods: tuple = (648, 1296)      # 27 d, 54 d
    clip_z: float = 6.0
    drop: tuple = ()                  # feature names forced to "missing" everywhere (train, val, predict), e.g. B

    @property
    def seg(self):
        return self.horizon + 2 * self.pad

    @property
    def min_hist(self):
        return max(self.periods) + self.pad

    def to_dict(self):
        d = asdict(self)
        d["periods"] = list(self.periods)
        d["drop"] = list(self.drop)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["periods"] = tuple(d["periods"])
        d["drop"] = tuple(d.get("drop", ()))
        return WCfg(**d)


@dataclass
class Aug:
    p_block: float = 0.5      # prob. of adding 1-3 contiguous gaps (3-48 h) in the recent window
    p_iid: float = 0.05       # iid hour dropout in the recent window
    p_rec: float = 0.05       # iid hour dropout in the recurrence windows
    jitter: bool = True       # latency +-1 h
    p_chan_b: float = 0.15    # prob. that Bx,By,Bz are entirely missing in a sample (2025+ feed may lack them)
    p_chan_n: float = 0.10    # same for density


# ----------------------------------------------------------------------------------------------- scaling
def _transform(raw):
    x = np.array(raw, dtype=float, copy=True)
    x[:, 4] = np.log1p(np.clip(x[:, 4], 0, None))          # density is heavy-tailed
    return x


class Scaler:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, float)
        self.std = np.asarray(std, float)

    @staticmethod
    def fit(df):
        x = _transform(df[FEATS].to_numpy(float))
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        return Scaler(mean, np.where(std > 0, std, 1.0))

    def to_dict(self):
        return dict(mean=self.mean.tolist(), std=self.std.tolist())

    @staticmethod
    def from_dict(d):
        return Scaler(d["mean"], d["std"])


# -------------------------------------------------------------------------------------------------- grid
def take(a, start, length, fill=0.0):
    """a[start:start+length] with out-of-range rows replaced by `fill` (never wraps around)."""
    out = np.full((length,) + a.shape[1:], fill, dtype=a.dtype)
    lo, hi = max(start, 0), min(start + length, len(a))
    if hi > lo:
        out[lo - start:hi - start] = a[lo:hi]
    return out


class Grid:
    """Hourly table -> scaled arrays.  `t_end` extends the grid with empty hours (live: up to t0 + H)."""

    def __init__(self, df, scaler, cfg, t_end=None):
        idx = pd.date_range(df.index.min().floor("h"), max(df.index.max(), pd.Timestamp(t_end) if t_end is not None
                                                           else df.index.max()), freq="h")
        d = df.reindex(idx)[FEATS]
        self.t = idx
        self.start = idx[0]
        self.scaler = scaler
        self.v = d["V"].to_numpy(float)                                  # km/s, NaN allowed
        z = (_transform(d.to_numpy(float)) - scaler.mean) / scaler.std
        z = np.clip(z, -cfg.clip_z, cfg.clip_z)
        m = np.isfinite(z)
        self.z = np.where(m, z, 0.0).astype(np.float32)
        self.m = m.astype(np.float32)
        for name in cfg.drop:                                            # e.g. B is not available in the live feed
            assert name != "V", "V cannot be dropped"
            j = FEATS.index(name)
            self.z[:, j] = 0.0
            self.m[:, j] = 0.0
        self.cs_z = np.concatenate([[0.0], np.cumsum(self.z[:, 0].astype(float))])    # V column only
        self.cs_m = np.concatenate([[0.0], np.cumsum(self.m[:, 0].astype(float))])
        doy = idx.dayofyear.to_numpy() / 365.25 * 2 * np.pi
        self.sin, self.cos = np.sin(doy).astype(np.float32), np.cos(doy).astype(np.float32)

    def __len__(self):
        return len(self.t)

    def index_of(self, ts):
        i = int(self.t.get_loc(pd.Timestamp(ts)))
        return i

    def _win(self, a, b):                                   # sum over grid indices a..b inclusive (clamped)
        a = max(a, 0)
        b = min(b, len(self) - 1)
        if b < a:
            return 0.0, 0.0
        return self.cs_z[b + 1] - self.cs_z[a], self.cs_m[b + 1] - self.cs_m[a]


def valid_t0(g, cfg, start=None, end=None, step=1, min_targets=12, min_recent=24):
    """Grid indices usable as issue times (history long enough, some targets, some recent data)."""
    n = len(g)
    i = np.arange(cfg.min_hist + cfg.latency + 2, n - cfg.horizon)
    if start is not None:
        i = i[g.t[i] >= pd.Timestamp(start)]
    if end is not None:
        i = i[g.t[i] <= pd.Timestamp(end)]
    e = i - cfg.latency
    n_tgt = g.cs_m[i + cfg.horizon + 1] - g.cs_m[i + 1]
    n_rec = g.cs_m[e + 1] - g.cs_m[e - cfg.recent + 1]
    i = i[(n_tgt >= min_targets) & (n_rec >= min_recent)]
    return i[:: max(step, 1)]


# ----------------------------------------------------------------------------------------------- sample
def build_sample(g, i0, cfg, rng=None, aug=None):
    """All model inputs (+ targets) for issue index i0.  rng/aug given -> training-time degradation."""
    H, R, S = cfg.horizon, cfg.recent, cfg.seg
    train = rng is not None and aug is not None
    lat = cfg.latency + (int(rng.integers(-1, 2)) if train and aug.jitter else 0)
    e = i0 - lat                                            # last available hour

    z = take(g.z, e - R + 1, R)
    m = take(g.m, e - R + 1, R)
    if train:
        drop = np.zeros(R, bool)
        if rng.random() < aug.p_block:
            for _ in range(int(rng.integers(1, 4))):
                L = int(rng.integers(3, 49))
                s = int(rng.integers(0, R - L + 1))
                drop[s:s + L] = True
        drop |= rng.random(R) < aug.p_iid
        m = m * (~drop)[:, None]
        z = z * m
    xr = np.concatenate([z, m], axis=1)

    # persistence baseline = last valid V in the recent window (scaled), and its age
    vm = m[:, 0] > 0
    if vm.any():
        j = int(np.flatnonzero(vm)[-1])
        base = float(z[j, 0])
        age = lat + (R - 1 - j)
    else:
        base, age = 0.0, 72
    age = min(age, 72)

    xrec, fill = [], []
    for P in cfg.periods:
        s = i0 - P + 1 - cfg.pad
        zz, mm = take(g.z, s, S), take(g.m, s, S)
        if train:
            keep = rng.random(S) >= aug.p_rec
            mm = mm * keep[:, None]
            zz = zz * mm
        xrec.append(np.concatenate([zz, mm], axis=1))
        rv, rm = zz[cfg.pad:cfg.pad + H, 0], mm[cfg.pad:cfg.pad + H, 0]
        fill.append(np.where(rm > 0, rv, base))
    xrec = np.stack(xrec)
    fill = np.stack(fill)
    if train:                                               # whole feature groups missing (feed-level outages)
        nf = len(FEATS)
        cols = ([1, 2, 3] if rng.random() < aug.p_chan_b else []) + ([4] if rng.random() < aug.p_chan_n else [])
        for c in cols:
            xr[:, c] = 0.0
            xr[:, nf + c] = 0.0
            xrec[:, :, c] = 0.0
            xrec[:, :, nf + c] = 0.0

    s7, n7 = g._win(e - 168 + 1, e)
    s27, n27 = g._win(e - 648 + 1, e)
    scal = np.array([g.sin[i0], g.cos[i0], age / 72.0, vm.mean(),
                     s7 / n7 if n7 > 0 else 0.0, s27 / n27 if n27 > 0 else 0.0], dtype=np.float32)

    yv = take(g.v, i0 + 1, H, fill=np.nan)                  # raw km/s, NaN where missing / outside the grid
    ym = np.isfinite(yv)
    y = np.where(ym, (yv - g.scaler.mean[0]) / g.scaler.std[0], 0.0)

    f32 = lambda a: np.asarray(a, dtype=np.float32)
    return dict(xr=f32(xr), xrec=f32(xrec), scal=f32(scal), base=f32(base), fill=f32(fill),
                y=f32(y), ym=f32(ym), yv=f32(np.where(ym, yv, 0.0)))


# ------------------------------------------------------------------------------------- metric helpers
def tail_c(v):
    """Per-observation c_i with TIMSE = sum_i c_i e_i^2  (same as score_tools.tail_c)."""
    v = np.asarray(v, float)
    n = len(v)
    q = np.quantile(v, np.linspace(0, 1, 101))
    if q[-1] == q[0]:
        return np.full(n, 1.0 / n)
    w = np.diff(q) / (q[-1] - q[0])
    Nk = n - np.searchsorted(np.sort(v), q[:-1], side="left")
    cw = np.cumsum(w / Nk)
    K = np.minimum(np.searchsorted(q, v, side="right") - 1, 99)
    return cw[K]


def make_tail_lut(v_ref):
    """Lookup  speed -> N*c  (mean 1) from a reference speed distribution (use the recent regime)."""
    v = np.asarray(v_ref, float)
    v = np.sort(v[np.isfinite(v)])
    c = tail_c(v) * len(v)
    return v, c


def tail_weight(yv, ym, lut, gamma=0.25, clip=6.0):
    """Metric-aligned sample weight  1 + gamma*N*c(v)  (gradient of S wrt a squared error is
    proportional to 1/(N MSE) + c/TIMSE;  gamma ~ MSE/TIMSE ~ 0.2-0.3).  Missing targets get weight 1 (masked later)."""
    nc = np.interp(yv, lut[0], lut[1])
    w = np.minimum(1.0 + gamma * nc, clip)
    return np.where(ym > 0, w, 1.0).astype(np.float32)


def lead_scores(v, p):
    c = tail_c(v)
    e2 = (v - p) ** 2
    rm, tr = np.sqrt(e2.mean()), np.sqrt((c * e2).sum())
    return rm, tr, np.sqrt(rm * tr)


def score_matrix(obs, pred):
    """obs, pred: (n_t0, H) km/s, NaN obs dropped.  Returns per-lead DataFrame and S_final."""
    rows = []
    for h in range(obs.shape[1]):
        m = np.isfinite(obs[:, h]) & np.isfinite(pred[:, h])
        if m.sum() < 20:
            rows.append((h + 1, np.nan, np.nan, np.nan))
            continue
        rows.append((h + 1, *lead_scores(obs[m, h], pred[m, h])))
    t = pd.DataFrame(rows, columns=["lead", "RMSE", "TRMSE", "S"]).set_index("lead")
    return t, float(t["S"].mean())
