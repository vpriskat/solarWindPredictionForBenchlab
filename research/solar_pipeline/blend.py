"""Per-lead-group linear blend of base predictors (persistence, recurrence, TCN, Timestral, ENLIL, ...)."""
import numpy as np

from score_tools import metric_weights


class Blend:
    """y_hat = c_g + sum_k beta_gk * x_k, one fit per group of `width` consecutive lead times.

    * Ridge shrinks beta toward the EQUAL-weight average (1/K each), not toward 0: with little data the blend
      degrades to 'average the predictors and fix the bias'.  `lam` is relative to the predictors' variance.
    * A missing x_k is replaced by `persistence` (the spec's own fallback), so a dead feed degrades gracefully.
    * `fit_two_pass` uses the competition-metric sample weights (evaluated at the first-pass predictions).
    """

    def __init__(self, cols, width=6, lam=0.1):
        self.cols, self.width, self.lam = list(cols), int(width), float(lam)
        self.params, self.fallback = {}, 400.0

    def _X(self, d):
        X = d[self.cols].to_numpy(float)
        pers = d["persistence"].to_numpy(float)
        pers = np.where(np.isfinite(pers), pers, self.fallback)
        return np.where(np.isfinite(X), X, pers[:, None])

    def _grp(self, d):
        return (d["lead_time"].to_numpy().astype(int) - 1) // self.width

    def fit(self, d, w=None):
        y = d["obs"].to_numpy(float)
        self.fallback = float(np.nanmean(y))
        w = np.ones(len(d)) if w is None else np.asarray(w, float)
        X, g = self._X(d), self._grp(d)
        K = X.shape[1]
        b0 = np.full(K, 1.0 / K)
        self.params = {}
        for gi in np.unique(g):
            m = g == gi
            Xg, yg, wg = X[m], y[m], w[m]
            sw = wg.sum()
            xm = wg @ Xg / sw
            r = yg - Xg @ b0
            rm = wg @ r / sw
            Xc = Xg - xm
            s2 = (wg @ (Xc ** 2) / sw).mean()
            A = (Xc * wg[:, None]).T @ Xc + self.lam * sw * s2 * np.eye(K)
            delta = np.linalg.solve(A, (Xc * wg[:, None]).T @ (r - rm))
            self.params[int(gi)] = (float(rm - xm @ delta), b0 + delta)
        return self

    def fit_two_pass(self, d):
        self.fit(d)
        return self.fit(d, metric_weights(d, self.predict(d)))

    def predict(self, d):
        X, g = self._X(d), self._grp(d)
        keys = np.array(sorted(self.params))
        out = np.empty(len(d))
        for gi in np.unique(g):
            m = g == gi
            c, beta = self.params[int(keys[np.argmin(np.abs(keys - gi))])]    # nearest fitted group if unseen
            out[m] = c + X[m] @ beta
        return out

    def weights_table(self):
        import pandas as pd
        return pd.DataFrame({f"leads {k * self.width + 1}-{(k + 1) * self.width}": dict(zip(["const"] + self.cols, [c, *b]))
                             for k, (c, b) in sorted(self.params.items())}).T
