"""Competition metric tools (BenchLab solar-wind spec) + metric-aligned training weights.

TIMSE = sum_i c_i * err_i^2  (c_i depends only on the observations of that lead time),
S_tau = (MSE_tau * TIMSE_tau)^(1/4),  S_final = mean_tau S_tau.
"""
import numpy as np
import pandas as pd


def tail_c(v):
    """Per-observation weights c_i so that TIMSE = sum_i c_i*(v_i-p_i)^2 (spec's velocity-space integral)."""
    v = np.asarray(v, float)
    n = len(v)
    q = np.quantile(v, np.linspace(0, 1, 101))
    if q[-1] == q[0]:
        return np.full(n, 1.0 / n)
    w = np.diff(q) / (q[-1] - q[0])                                   # k = 0..99 (zero-width -> 0)
    Nk = n - np.searchsorted(np.sort(v), q[:-1], side="left")         # #{v >= q_k}
    cw = np.cumsum(w / Nk)
    K = np.minimum(np.searchsorted(q, v, side="right") - 1, 99)       # last k with q_k <= v_i
    return cw[K]


def _lead_scores(v, p):
    c = tail_c(v)
    e2 = (v - p) ** 2
    rmse, trmse = np.sqrt(e2.mean()), np.sqrt((c * e2).sum())
    return rmse, trmse, np.sqrt(rmse * trmse)


def per_lead(df, pred_col="pred"):
    """df needs lead_time, obs, persistence, pred_col. NaN obs dropped; NaN preds -> persistence (spec)."""
    d = df[df["obs"].notna()]
    p = d[pred_col].where(d[pred_col].notna(), d["persistence"]).to_numpy(float)
    v = d["obs"].to_numpy(float)
    lead = d["lead_time"].to_numpy()
    rows = []
    for tau in np.unique(lead):
        m = (lead == tau) & np.isfinite(p) & np.isfinite(v)
        rm, tr, s = _lead_scores(v[m], p[m])
        rows.append((tau, rm, tr, s, (p[m] - v[m]).mean()))
    return pd.DataFrame(rows, columns=["lead", "RMSE", "TRMSE", "S", "bias"]).set_index("lead")


def score(df, pred_col="pred"):
    t = per_lead(df, pred_col)
    return dict(S_final=t.S.mean(), RMSE=t.RMSE.mean(), TRMSE=t.TRMSE.mean(), bias=t.bias.mean(),
                S_1_24=t.S.loc[1:24].mean(), S_25_48=t.S.loc[25:48].mean(), S_49_72=t.S.loc[49:72].mean())


def metric_weights(df, ref_pred, clip=15.0):
    """Sample weights = gradient of mean_tau S_tau wrt each sample's squared error, evaluated at a
    reference predictor (errors of `ref_pred` on df):  w_i = S_tau * (1/(N*MSE_tau) + c_i/TIMSE_tau).
    Normalised to median 1 and clipped (a few events carry very large weight)."""
    v = df["obs"].to_numpy(float)
    lead = df["lead_time"].to_numpy()
    r = np.asarray(ref_pred, float)
    w = np.empty(len(df))
    for tau in np.unique(lead):
        m = np.flatnonzero(lead == tau)
        c = tail_c(v[m])
        e2 = (v[m] - r[m]) ** 2
        mse, timse = e2.mean(), (c * e2).sum()
        S = (mse * timse) ** 0.25
        w[m] = S * (1.0 / (len(m) * mse) + c / timse)
    w = w / np.median(w)
    return np.minimum(w, clip)


def paired_block_bootstrap(df, col_a, col_b, n_boot=100, block_days=5, seed=0):
    """Resample blocks of consecutive days (keeps autocorrelation); same rows for both models.
    Returns mean and 95% CI of S_final(a) - S_final(b)  (negative = a is better)."""
    rng = np.random.default_rng(seed)
    d0 = df["t0"].dt.normalize()
    block = ((d0 - d0.min()).dt.days.to_numpy()) // block_days
    groups = {b: np.flatnonzero(block == b) for b in np.unique(block)}
    keys = np.array(list(groups))
    cols = ["t0", "lead_time", "obs", "persistence", col_a, col_b]
    base = df[cols]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[b] for b in rng.choice(keys, size=len(keys), replace=True)])
        s = base.iloc[idx]
        diffs.append(score(s, col_a)["S_final"] - score(s, col_b)["S_final"])
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5])


def sfinal_arrays(lead, v, p):
    """S_final from plain arrays (no NaNs): per-lead S, equal-weight mean over leads."""
    lead, v, p = np.asarray(lead), np.asarray(v, float), np.asarray(p, float)
    order = np.argsort(lead, kind="stable")
    lead, v, p = lead[order], v[order], p[order]
    b = np.r_[0, np.flatnonzero(np.diff(lead)) + 1, len(lead)]
    return float(np.mean([_lead_scores(v[a:c], p[a:c])[2] for a, c in zip(b[:-1], b[1:])]))


def bootstrap_matrix(df, cols, n_boot=100, block_days=5, seed=0):
    """S_final of EVERY column in `cols` on the SAME block-bootstrap resamples -> array (n_boot, len(cols)).
    Differences between columns are therefore paired.  Rows with a NaN observation are dropped; NaN predictions
    fall back to persistence; rows still non-finite in any column are dropped for all columns."""
    cols = list(cols)
    v = df["obs"].to_numpy(float)
    pers = df["persistence"].to_numpy(float)
    P = np.column_stack([np.where(np.isfinite(df[c].to_numpy(float)), df[c].to_numpy(float), pers) for c in cols])
    ok = np.isfinite(v) & np.isfinite(P).all(axis=1)
    lead, v, P = df["lead_time"].to_numpy()[ok], v[ok], P[ok]
    d0 = df["t0"].dt.normalize()
    block = (((d0 - d0.min()).dt.days.to_numpy()) // block_days)[ok]
    groups = {b: np.flatnonzero(block == b) for b in np.unique(block)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(cols)))
    for i in range(n_boot):
        idx = np.concatenate([groups[b] for b in rng.choice(keys, size=len(keys), replace=True)])
        for j in range(len(cols)):
            out[i, j] = sfinal_arrays(lead[idx], v[idx], P[idx, j])
    return out
