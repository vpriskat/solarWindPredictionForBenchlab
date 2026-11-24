"""Tail calibration: p' = p + a + (b - 1)*(p - mu)   (shift a, stretch b about mu; b = 1 means no stretch), fitted per lead group by minimising the competition metric itself.

The metric per lead is S_tau = (MSE * TIMSE)^(1/4), TIMSE = sum_i c_i e_i^2 (c_i from the observations only), so this is a
direct search over (a, b) of mean_tau S_tau.  A shift (b = 1) trades bias for tail error; a stretch (b > 1) un-shrinks
predictions that regress to the mean, which is what the tail term punishes.

Few parameters on purpose: the calibration block has only a few hundred issue times (a few dozen independent samples),
so per-lead or per-feature calibration would just memorise it.
"""
import numpy as np

from score_tools import tail_c


class TailCal:
    def __init__(self, groups=((1, 72),), stretch=True, a_grid=None, b_grid=None):
        self.groups = [tuple(g) for g in groups]
        self.stretch = stretch
        self.a_grid = np.arange(-60.0, 60.1, 5.0) if a_grid is None else np.asarray(a_grid, float)
        if b_grid is None:
            b_grid = np.arange(0.8, 1.601, 0.05) if stretch else np.array([1.0])
        self.b_grid = np.asarray(b_grid, float)
        self.params = {}

    def fit(self, d, p):
        lead = d["lead_time"].to_numpy().astype(int)
        v = d["obs"].to_numpy(float)
        p = np.asarray(p, float)
        p = np.where(np.isfinite(p), p, d["persistence"].to_numpy(float))
        for lo, hi in self.groups:
            leads = []
            for tau in range(lo, hi + 1):
                m = (lead == tau) & np.isfinite(v) & np.isfinite(p)
                if m.sum() >= 20:
                    leads.append((v[m], p[m], tail_c(v[m])))
            if not leads:
                continue
            mu = float(np.mean(np.concatenate([x[1] for x in leads])))
            best = (np.inf, 0.0, 1.0)
            for b in self.b_grid:
                for a in self.a_grid:
                    s = 0.0
                    for v_, p_, c_ in leads:
                        e2 = (v_ - (p_ + a + (b - 1.0) * (p_ - mu))) ** 2
                        s += (e2.mean() * (c_ * e2).sum()) ** 0.25
                    s /= len(leads)
                    if s < best[0]:
                        best = (s, float(a), float(b))
            self.params[(lo, hi)] = dict(a=best[1], b=best[2], mu=mu, S_fit=best[0])
        return self

    def predict(self, d, p):
        lead = d["lead_time"].to_numpy().astype(int)
        out = np.array(p, float, copy=True)
        for (lo, hi), pr in self.params.items():
            m = (lead >= lo) & (lead <= hi) & np.isfinite(out)
            out[m] = out[m] + pr["a"] + (pr["b"] - 1.0) * (out[m] - pr["mu"])
        return out

    def describe(self):
        return "  ".join(f"leads {lo}-{hi}: a={pr['a']:+.0f} b={pr['b']:.2f}" for (lo, hi), pr in self.params.items())
