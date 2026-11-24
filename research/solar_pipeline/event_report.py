"""How much of the test result depends on one event?   Reads results/ladder_test_preds.pkl (written by run_experiments.py),
so it needs no retraining:

    python event_report.py                                    # default window 2026-01-14:2026-01-22 (by target time)
    python event_report.py --window 2026-01-14:2026-01-22 --peak 2026-01-19T12:2026-01-21T00 --boot 200

1. every model scored on all rows vs without the window  (+ rank change)
2. paired block bootstrap of the key comparisons WITHOUT the window
3. anatomy of the event: what each model predicted for the peak hours, by issue day  (did ENLIL see it?)
4. error by observed regime (slow / normal / fast / extreme): bias and RMSE per model, with and without the window
"""
import argparse
import pickle

import numpy as np
import pandas as pd

from score_tools import bootstrap_matrix, score

KEY = ["persistence", "27-day recurrence", "enlil only", "timestral only", "tcn only", "A current (heuristic w)",
       "D resid on mean-ref", "R per-lead blend", "R + clim27", "T blend+trees (lite)", "T -tcn", "T -ts", "T -enlil",
       "R (train-only)", "R + cme x0.5", "R + cme x1",
       "chronos only", "R -c2 (previous R)", "R + strip x0.5", "R + strip x1", "R (train-only) + strip x0.5", "R (train-only) + strip x1"]
PAIRS = [("R per-lead blend", "R -c2 (previous R)"), ("chronos only", "timestral only"), ("R per-lead blend", "timestral only"), ("R per-lead blend", "tcn only"), ("R per-lead blend", "persistence"),
         ("T blend+trees (lite)", "R per-lead blend"), ("R + clim27", "R per-lead blend"),
         ("T -tcn", "T blend+trees (lite)"), ("T -ts", "T blend+trees (lite)"), ("T -enlil", "T blend+trees (lite)"),
         ("R (train-only)", "R per-lead blend"), ("R + cme x0.5", "R per-lead blend"), ("R + cme x1", "R per-lead blend"),
         ("R + strip x0.5", "R per-lead blend"), ("R + strip x1", "R per-lead blend"),
         ("R (train-only) + strip x0.5", "R (train-only)"), ("R (train-only) + strip x1", "R (train-only)")]
REGIMES = [("slow <400", -np.inf, 400), ("normal 400-550", 400, 550), ("fast 550-700", 550, 700), ("extreme >=700", 700, np.inf)]


def _split(res, window):
    a, b = [pd.Timestamp(x) for x in window.split(":")]
    r = res.copy()
    r["t0"] = pd.to_datetime(r["t0"])
    r["target"] = r["t0"] + pd.to_timedelta(r["lead_time"], unit="h")
    inside = (r["target"] >= a) & (r["target"] < b + pd.Timedelta(days=1))
    return r, inside


def table_scores(r, inside, models):
    rows = []
    for m in models:
        s_all, s_ex = score(r, m), score(r[~inside], m)
        rows.append((m, s_all["S_final"], s_ex["S_final"], s_all["bias"], s_ex["bias"]))
    t = pd.DataFrame(rows, columns=["model", "S all", "S w/o window", "bias all", "bias w/o"]).set_index("model")
    t["rank all"] = t["S all"].rank().astype(int)
    t["rank w/o"] = t["S w/o window"].rank().astype(int)
    t["event cost"] = t["S all"] - t["S w/o window"]
    return t


def table_boot(r, inside, boot):
    names = sorted({m for p in PAIRS for m in p if m in r.columns})
    out = {}
    for tag, d in (("all", r), ("w/o window", r[~inside])):
        B = bootstrap_matrix(d, names, n_boot=boot)
        ix = {n: i for i, n in enumerate(names)}
        S = {n: score(d, n)["S_final"] for n in names}
        for a, b in PAIRS:
            if a in ix and b in ix:
                lo, hi = np.percentile(B[:, ix[a]] - B[:, ix[b]], [2.5, 97.5])
                out.setdefault((a, b), {})[tag] = (S[a] - S[b], lo, hi)
    rows = []
    for (a, b), v in out.items():
        f = lambda t: f"{v[t][0]:+6.2f} [{v[t][1]:+6.2f},{v[t][2]:+6.2f}]" + ("*" if v[t][1] > 0 or v[t][2] < 0 else " ")
        rows.append((f"{a}  vs  {b}", f("all"), f("w/o window")))
    return pd.DataFrame(rows, columns=["new vs old (negative = new better)", "all rows", "without window   (* = CI excludes 0)"])


def anatomy(r, peak, models):
    a, b = [pd.Timestamp(x) for x in peak.split(":")]
    d = r[(r["target"] >= a) & (r["target"] <= b)].copy()
    d["issue_day"] = d["t0"].dt.floor("D")
    g = d.groupby("issue_day")
    out = pd.DataFrame({"n": g.size(), "obs mean": g["obs"].mean(), "obs max": g["obs"].max()})
    for m in models:
        if m in d.columns:
            p = d[m].where(d[m].notna(), d["persistence"])
            out[m[:12] + " mean"] = p.groupby(d["issue_day"]).mean()
    mx = {}
    if "enlil only" in d.columns:
        e = d["enlil only"]
        out["enlil max"] = e.groupby(d["issue_day"]).max()
        out["enlil NaN%"] = 100 * e.isna().groupby(d["issue_day"]).mean()
    return out


def regimes(r, inside, models):
    res = {}
    for tag, d in (("all", r), ("w/o window", r[~inside])):
        rows = []
        for name, lo, hi in REGIMES:
            m = (d["obs"] >= lo) & (d["obs"] < hi)
            row = {"regime": name, "share_%": 100 * m.mean()}
            for mod in models:
                if mod in d.columns:
                    p = d.loc[m, mod].where(d.loc[m, mod].notna(), d.loc[m, "persistence"])
                    e = p - d.loc[m, "obs"]
                    row[f"{mod[:12]} bias/rmse"] = f"{e.mean():+5.0f}/{np.sqrt((e ** 2).mean()):4.0f}"
            rows.append(row)
        res[tag] = pd.DataFrame(rows).set_index("regime")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/ladder_test_preds.pkl")
    ap.add_argument("--window", default="2026-01-14:2026-01-22", help="inclusive days (by TARGET time) removed from the test")
    ap.add_argument("--peak", default="2026-01-19T12:2026-01-21T00", help="target hours shown in the event anatomy")
    ap.add_argument("--boot", type=int, default=100)
    a = ap.parse_args()
    res = pickle.load(open(a.pred, "rb"))
    r, inside = _split(res, a.window)
    models = [m for m in KEY if m in r.columns]
    print(f"window {a.window}: {inside.mean() * 100:.1f}% of test rows removed ({inside.sum():,} of {len(r):,})")
    pd.set_option("display.width", 220)
    with pd.option_context("display.float_format", "{:.1f}".format):
        print("\n1) S_final with and without the window (event cost = how much of S the window adds)")
        print(table_scores(r, inside, models).to_string())
        print("\n2) paired block bootstrap, S_final(new) - S_final(old), 95% CI")
        print(table_boot(r, inside, a.boot).to_string(index=False))
        print(f"\n3) the event: predictions for target hours {a.peak}, by the day the forecast was issued")
        print(anatomy(r, a.peak, ["persistence", "enlil only", "timestral only", "tcn only", "R per-lead blend"]).to_string())
        reg = regimes(r, inside, ["persistence", "timestral only", "tcn only", "R per-lead blend"])
        for tag in ("all", "w/o window"):
            print(f"\n4) bias/RMSE by observed regime, {tag}")
            print(reg[tag].to_string())


if __name__ == "__main__":
    main()
