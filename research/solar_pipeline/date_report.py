"""Which dates drive the score?  Exact attribution of S_final to individual (t0, lead) rows, then grouped by date.

S_tau = (MSE_tau * TIMSE_tau)^(1/4) is homogeneous of degree 1/2 in the squared errors (c_i depend only on the observations),
so by Euler's theorem   S_tau = 2 * sum_i (dS_tau/de_i^2) * e_i^2   and row i's share of S_tau is
        share_i = 0.5 * ( e_i^2 / (N*MSE_tau)  +  c_i e_i^2 / TIMSE_tau )          (sums to 1 over the rows of a lead)
S_final = mean_tau S_tau, so  attr_i = S_tau * share_i / sum_tau S_tau  sums to 1.
Check: scaling the errors of a set of rows by (1+d) changes S_final by d * S_final * (sum of their attr)   (to first order).
"""
import numpy as np
import pandas as pd

from score_tools import tail_c


def attribution(df, pred_col="pred"):
    """df needs lead_time, obs, persistence, pred_col (NaN pred -> persistence, NaN obs dropped, as the spec)."""
    d = df[df["obs"].notna()].copy()
    p = d[pred_col].where(d[pred_col].notna(), d["persistence"]).to_numpy(float)
    v = d["obs"].to_numpy(float)
    ok = np.isfinite(p) & np.isfinite(v)
    d, p, v = d[ok].copy(), p[ok], v[ok]
    lead = d["lead_time"].to_numpy()
    a = np.zeros(len(d))
    S_sum = 0.0
    for tau in np.unique(lead):
        m = lead == tau
        c = tail_c(v[m])
        e2 = (v[m] - p[m]) ** 2
        mse, timse = e2.mean(), (c * e2).sum()
        S = (mse * timse) ** 0.25
        a[m] = S * 0.5 * (e2 / (m.sum() * mse) + c * e2 / timse)
        S_sum += S
    d["attr"] = a / S_sum
    d["pred_used"] = p
    d["err"] = p - v
    return d


def date_report(res, col, others=(), by="target", top=15, persistence_col="persistence"):
    """res: rows (t0, lead_time, obs, persistence, <model columns>).  by = 'target' (when the error happens) or 'issue' (t0)."""
    r = res.copy()
    r["t0"] = pd.to_datetime(r["t0"])
    r["target_time"] = r["t0"] + pd.to_timedelta(r["lead_time"], unit="h")
    key = "target_time" if by == "target" else "t0"
    d = attribution(r, col)
    d["day"] = d[key].dt.floor("D")
    g = d.groupby("day")
    out = pd.DataFrame({"share_%": 100 * g["attr"].sum(), "rows": g.size(),
                        "obs_mean": g["obs"].mean(), "obs_max": g["obs"].max(), "bias": g["err"].mean(),
                        "RMSE": g["err"].apply(lambda e: np.sqrt((e ** 2).mean()))})
    for nm in list(others) + [persistence_col]:
        if nm in r.columns:
            x = r.loc[d.index, nm].where(r.loc[d.index, nm].notna(), r.loc[d.index, persistence_col])
            out[f"RMSE_{nm[:10]}"] = (x - d["obs"]).pow(2).groupby(d["day"]).mean().pow(0.5)
    out = out.sort_values("share_%", ascending=False)
    cum = out["share_%"].cumsum()
    summary = dict(days=len(out), top5=float(out["share_%"].head(5).sum()), top10=float(out["share_%"].head(10).sum()),
                   top20=float(out["share_%"].head(20).sum()), days_for_50pct=int((cum < 50).sum() + 1),
                   uniform_share=float(100.0 / len(out)))
    return out, summary


def print_date_report(res, col, others=(), top=15, csv=None):
    for by, title in (("target", "TARGET date (when the error happens)"), ("issue", "ISSUE date t0 (when the forecast was made)")):
        out, s = date_report(res, col, others, by=by)
        print(f"\nShare of S_final by {title}  --  {col}")
        print(f"  {s['days']} days; top 5 days = {s['top5']:.0f}% of S, top 10 = {s['top10']:.0f}%, top 20 = {s['top20']:.0f}%; "
              f"{s['days_for_50pct']} days make up half of it (uniform would be {s['uniform_share']:.1f}% per day)")
        with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
            print(out.head(top).to_string())
        if csv and by == "target":
            out.to_csv(csv)
            print(f"  full table written to {csv}")
