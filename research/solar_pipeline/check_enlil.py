"""Is ENLIL aligned with ACE, and does it carry any skill?

    python check_enlil.py --enlil-path <file>        (run it on the re-downloaded file BEFORE trusting any ENLIL feature)

For every ENLIL run R and lead L, e(R, L) is valid at V = R + L.  We correlate e(R, L) with ACE speed at V + s for
s in [-36, +36] h, per lead band, and report the best s:
    s* ~ 0           aligned (a few hours of offset is ordinary arrival-time error)
    s* > 0           ENLIL shows the structure s* hours BEFORE ACE sees it  (ENLIL early)
    s* < 0           ENLIL shows it late
    s* near +-24/48, or jumping between bands   ->  suspect a run_date / lead indexing bug (e.g. mis-filtered leads)
A low correlation at every shift means the run/lead join itself is wrong or ENLIL has no skill at that lead.
"""
import argparse

import numpy as np
import pandas as pd

from features import Cfg, load_raw


def enlil_alignment(obs, enlil, shifts=range(-36, 37, 3), bands=((0, 24), (24, 48), (48, 72), (72, 97)), min_n=200):
    e = enlil.rename("v").reset_index()
    e.columns = ["run", "lead", "v"]
    valid = e["run"] + pd.to_timedelta(e["lead"], unit="h")
    v = e["v"].to_numpy(float)
    rows = []
    for s in shifts:
        o = obs.reindex(valid + pd.Timedelta(hours=s)).to_numpy(float)
        for lo, hi in bands:
            m = ((e["lead"] >= lo) & (e["lead"] < hi)).to_numpy() & np.isfinite(o) & np.isfinite(v)
            if m.sum() < min_n:
                continue
            rows.append(dict(shift=s, band=f"{lo}-{hi - 1}h", n=int(m.sum()),
                             corr=float(np.corrcoef(v[m], o[m])[0, 1]), bias=float((o[m] - v[m]).mean())))
    return pd.DataFrame(rows)


def summarize(tab):
    out = []
    for band, g in tab.groupby("band", sort=False):
        g = g.set_index("shift")
        if 0 not in g.index:
            continue
        s = int(g["corr"].idxmax())
        out.append(dict(band=band, n_at_0=int(g.loc[0, "n"]), corr_at_0=g.loc[0, "corr"], best_shift_h=s,
                        corr_at_best=g.loc[s, "corr"], bias_at_0=g.loc[0, "bias"]))
    return pd.DataFrame(out).set_index("band")


def best_shifts(tab):
    return summarize(tab)["best_shift_h"].to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enlil-path", default=None)
    a = ap.parse_args()
    cfg = Cfg()
    if a.enlil_path:
        cfg.enlil_path = a.enlil_path
    raw = load_raw(cfg)
    e = raw["enlil"]
    runs = e.index.get_level_values(0)
    leads = e.index.get_level_values(1)
    print(f"ENLIL: {runs.nunique()} runs {runs.min()} .. {runs.max()}, leads {leads.min()}..{leads.max()} h")
    tab = enlil_alignment(raw["obs"], e)
    print("\ncorr(ENLIL at valid time V, ACE at V+shift) by lead band")
    print(tab.pivot(index="shift", columns="band", values="corr").round(3).to_string())
    print("\nSummary")
    print(summarize(tab).round(3).to_string())
    print("\nbias = mean(ACE - ENLIL) at shift 0 (km/s).  See the docstring for how to read best_shift_h.")


if __name__ == "__main__":
    main()
