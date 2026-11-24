#!/usr/bin/env python3
"""Replay past cycles through the real container image and score against the NOAA monthly ACE truth.

  python3 replay_eval.py --days 10 --end 2026-11-18T18:00 --image swp-local --jobs 3
  python3 replay_eval.py --t0 20261115T060000Z 20261116T060000Z          # explicit cycles

For each t0 it runs:  docker run --entrypoint python3 <image> /app/model.py <t0>   (no upload, stdout = forecast JSON)
Notes: AIA images are fetched live for the past days (if the NRT archive still has them); ENLIL can only be used when the
single run currently on the SWPC server qualifies for that t0 (started <= t0-12h, age <= 72h), so old cycles run without ENLIL.
Metrics: per-cycle RMSE, overall RMSE/MAE by lead group, vs persistence (last valid <= t0-3h) and 27-day recurrence.
"""
import argparse, json, os, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

MON = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/{ym}_ace_swepam_1h.txt"


def truth_series(t_lo, t_hi):
    rows = {}
    for ym in pd.period_range(t_lo, t_hi, freq="M").strftime("%Y%m"):
        txt = urllib.request.urlopen(urllib.request.Request(MON.format(ym=ym), headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode()
        for ln in txt.splitlines():
            p = ln.split()
            if len(p) >= 9 and p[0].isdigit() and int(p[6]) == 0 and 0 < float(p[8]) < 3000:
                rows[pd.Timestamp(int(p[0]), int(p[1]), int(p[2]), int(p[3][:2]), int(p[3][2:]))] = float(p[8])
    return pd.Series(rows).sort_index()


def run_cycle(a):
    t0, image, outdir = a
    stamp = t0.strftime("%Y%m%dT%H%M%SZ")
    f = os.path.join(outdir, stamp + ".json")
    if not os.path.exists(f):
        r = subprocess.run(["docker", "run", "--rm", "--entrypoint", "python3", image, "/app/model.py", stamp], capture_output=True, text=True)
        try:
            d = json.loads(r.stdout)
            assert len(d["forecast_speed_kms"]) == 72
        except Exception:
            print("FAILED", stamp, r.stderr[-300:], file=sys.stderr); return t0, None, ""
        json.dump(d, open(f, "w"))
        open(os.path.join(outdir, stamp + ".log"), "w").write(r.stderr)
    log = open(os.path.join(outdir, stamp + ".log")).read() if os.path.exists(os.path.join(outdir, stamp + ".log")) else ""
    return t0, np.array(json.load(open(f))["forecast_speed_kms"], float), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", nargs="*"); ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--end", default=None, help="last t0, UTC, default: latest 6-hourly cycle whose truth is >=72h old is NOT required")
    ap.add_argument("--image", default="swp-local"); ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default="replay_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.t0:
        t0s = [pd.Timestamp(pd.to_datetime(s, format="%Y%m%dT%H%M%SZ")) for s in a.t0]
    else:
        end = pd.Timestamp(a.end) if a.end else (pd.Timestamp.utcnow().tz_localize(None).floor("6h") - pd.Timedelta(hours=6))
        t0s = list(pd.date_range(end - pd.Timedelta(days=a.days), end, freq="6h"))
    truth = truth_series(min(t0s) - pd.Timedelta(days=60), max(t0s) + pd.Timedelta(hours=73))
    print(f"{len(t0s)} cycles; truth rows {len(truth)}, last label {truth.index.max()}")
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run_cycle, [(t, a.image, a.out) for t in t0s]))
    leads = np.arange(1, 73)
    recs = []
    for t0, f, log in res:
        if f is None:
            continue
        tgt = t0 + pd.to_timedelta(leads, unit="h")
        y = truth.reindex(tgt).to_numpy()
        hist = truth[truth.index <= t0 - pd.Timedelta(hours=3)]
        pers = hist.iloc[-1] if len(hist) else np.nan
        rec = truth.reindex(tgt - pd.Timedelta(days=27)).to_numpy()
        rec = np.where(np.isfinite(rec), rec, pers)
        m = np.isfinite(y)
        lvl = "full" if "'level': 'full'" in log else ("?" if not log else "fallback")
        recs.append(dict(t0=t0, n=int(m.sum()), level=lvl, enlil="n_enlil': 0" not in log,
                         rmse_model=np.sqrt(np.mean((f[m] - y[m]) ** 2)) if m.any() else np.nan,
                         rmse_pers=np.sqrt(np.mean((pers - y[m]) ** 2)) if m.any() else np.nan,
                         rmse_rec=np.sqrt(np.mean((rec[m] - y[m]) ** 2)) if m.any() else np.nan,
                         _f=f, _y=y, _p=np.full(72, pers), _r=rec))
    d = pd.DataFrame(recs)
    print(d.drop(columns=[c for c in d if c.startswith("_")]).round(1).to_string(index=False))
    F = np.vstack(d["_f"]); Y = np.vstack(d["_y"]); P = np.vstack(d["_p"]); R = np.vstack(d["_r"])
    print("\nRMSE by lead group (all cycles pooled)   model  persistence  recurrence")
    for g in range(0, 72, 12):
        s = slice(g, g + 12); m = np.isfinite(Y[:, s])
        r = lambda X: np.sqrt(np.mean((X[:, s][m] - Y[:, s][m]) ** 2))
        print(f"  leads {g+1:2d}-{g+12:2d}                      {r(F):6.1f} {r(P):10.1f} {r(R):11.1f}")
    m = np.isfinite(Y)
    rm = lambda X: np.sqrt(np.mean((X[m] - Y[m]) ** 2))
    print(f"  ALL                              {rm(F):6.1f} {rm(P):10.1f} {rm(R):11.1f}   (MAE model {np.mean(np.abs(F[m]-Y[m])):.1f})")
    d.drop(columns=[c for c in d if c.startswith("_")]).to_csv(os.path.join(a.out, "summary.csv"), index=False)


main()
