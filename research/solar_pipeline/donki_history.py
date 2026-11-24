"""DONKI CME + WSA-ENLIL arrival history: download it, then check how good the predicted arrivals really were.

  python donki_history.py fetch --start 2013-01-01 --end 2026-11-01 --key YOUR_NASA_KEY --out data/donki_cme.csv
  python donki_history.py evaluate --csv data/donki_cme.csv \
        --obs data/omni_2020_2024_hourly.parquet:speed  --obs data/swpc_ace_historical/swpc_ace_swepam_hourly.parquet:speed_km_s \
        [--obs <your 2010-2019 file>:<speed column>] --out data/donki_arrivals.csv

fetch   : 30-day chunks, resumable (skips chunks already in --out), retries on rate limits.  Get a free key at api.nasa.gov
          (DEMO_KEY allows only ~30 requests/hour; 2013-2026 is ~170 chunks).
evaluate: for every CME with a predicted Earth shock arrival (ENLIL run) compare with the observed speed:
          * hit rate: is there a speed jump >= --jump km/s within +-36 h of the predicted arrival?
          * timing error (observed - predicted) of the FIRST prediction and of the LAST one
          * how the size of the jump relates to the CME speed / Kp estimate
          This gives the numbers needed to design the arrival feature (probability, timing sigma, amplitude).
"""
import argparse
import os
import time
import urllib.error

import numpy as np
import pandas as pd

from check_new_feeds import DONKI, _get, donki_rows


# ------------------------------------------------------------------------------------------------ fetch
def fetch(start, end, key, out, chunk_days=30, sleep=1.0):
    have = set()
    if os.path.exists(out):
        old = pd.read_csv(out, parse_dates=["cme_start"])
        have = set(pd.to_datetime(old["chunk"]).dt.strftime("%Y-%m-%d"))
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    while s < e:
        t1 = min(s + pd.Timedelta(days=chunk_days), e)
        tag = s.strftime("%Y-%m-%d")
        if tag not in have:
            for attempt in range(5):
                try:
                    cm = _get(f"{DONKI}?startDate={tag}&endDate={t1.strftime('%Y-%m-%d')}&api_key={key}")
                    break
                except urllib.error.HTTPError as ex:
                    wait = 70 if ex.code == 429 else 5 * (attempt + 1)
                    print(f"  {tag}: HTTP {ex.code}, waiting {wait}s")
                    time.sleep(wait)
                except Exception as ex:                      # network hiccup
                    print(f"  {tag}: {ex!r}, retrying")
                    time.sleep(5 * (attempt + 1))
            else:
                raise SystemExit(f"giving up at {tag}; rerun to resume")
            d = donki_rows(cm) if cm else pd.DataFrame()
            if len(d) == 0:                                   # keep a marker so empty chunks are not refetched
                d = pd.DataFrame({"chunk": [tag]})
            d["chunk"] = tag
            d.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"  {tag}: {len(cm)} CMEs, {len(d)} rows")
            time.sleep(sleep)
        s = t1
    print("done ->", out)


# ------------------------------------------------------------------------------------------- evaluation
def load_obs(specs):
    parts = []
    for spec in specs:
        path, col = spec.rsplit(":", 1)
        d = pd.read_parquet(path)
        tcol = next((c for c in ("timestamp", "time", "datetime", "date", "t0") if c in d.columns), None)
        if tcol is not None:
            d.index = pd.to_datetime(d[tcol])
        elif "YEAR" in d.columns:
            d.index = pd.to_datetime(d["YEAR"].astype(str) + " " + d["DOY"].astype(str) + " " + d["HOUR"].astype(str), format="%Y %j %H")
        d.index = pd.DatetimeIndex(d.index).tz_localize(None) if getattr(d.index, "tz", None) is not None else pd.DatetimeIndex(d.index)
        s = d[col].astype(float).replace([9999.0, 99999.0, 999.9], np.nan)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        parts.append(s)
        print(f"  obs {path}:{col}  {s.index.min()} .. {s.index.max()}  ({s.notna().mean() * 100:.0f}% valid)")
    obs = pd.concat(parts)
    obs = obs[~obs.index.duplicated(keep="first")].sort_index()          # first file wins where they overlap
    return obs.resample("h").mean()


def predicted_arrivals(d):
    """One row per predicted Earth arrival, with the FIRST and LAST prediction known for it (arrivals within 6 h = same event)."""
    a = d.dropna(subset=["shock_arrival"]).copy()
    a = a[a["shock_arrival"] > a["known_from"]]
    if a.empty:
        return a
    a = a.sort_values("known_from")
    # arrivals within 6 h of each other are one event
    s = a["shock_arrival"].sort_values()
    grp = (s.diff().dt.total_seconds().fillna(1e9) > 6 * 3600).cumsum()
    a["event"] = grp.reindex(a.index)
    g = a.sort_values("known_from").groupby("event")
    out = pd.DataFrame({"first_known": g["known_from"].first(), "pred_first": g["shock_arrival"].first(),
                        "last_known": g["known_from"].last(), "pred_last": g["shock_arrival"].last(),
                        "cme_speed": g["speed"].max(), "half_angle": g["half_angle"].max(), "kp_max": g["kp_max"].max(),
                        "n_runs": g.size()})
    return out


def observed_jump(obs, t, win=36, k=3):
    """Largest rise of the k-hour mean speed in [t-win, t+win]; returns (arrival time, jump km/s, speed after, speed before)."""
    lo, hi = t - pd.Timedelta(hours=win), t + pd.Timedelta(hours=win)
    v = obs[lo - pd.Timedelta(hours=k):hi + pd.Timedelta(hours=k)]
    if v.notna().mean() < 0.6 or len(v) < 2 * k + 2:
        return None
    after = v.rolling(k, min_periods=2).mean().shift(-(k - 1))        # mean of v[t..t+k-1]
    before = v.rolling(k, min_periods=2).mean().shift(1)              # mean of v[t-k..t-1]
    j = (after - before)[lo:hi].dropna()
    if j.empty:
        return None
    ta = j.idxmax()
    return ta, float(j.loc[ta]), float(after.loc[ta]), float(before.loc[ta])


def evaluate(events, obs, jump=60.0, win=36, seed=0):
    """Every predicted arrival is scored twice with the SAME procedure: at the predicted time, and at a randomly shifted
    time (7-60 days away, either sign) = the null: how often a +-win window contains a jump by chance."""
    rng = np.random.default_rng(seed)
    rows = []
    for ev, r in events.iterrows():
        o = observed_jump(obs, r["pred_last"], win)
        shift = pd.Timedelta(days=float(rng.uniform(7, 60) * rng.choice([-1, 1])))
        n = observed_jump(obs, r["pred_last"] + shift, win)
        if o is None or n is None:
            continue
        ta, j, va, vb = o
        rows.append(dict(event=ev, **r.to_dict(), obs_arrival=ta, jump=j, v_after=va, v_before=vb, hit=j >= jump,
                         err_last_h=(ta - r["pred_last"]).total_seconds() / 3600,
                         err_first_h=(ta - r["pred_first"]).total_seconds() / 3600,
                         lead_first_h=(r["pred_first"] - r["first_known"]).total_seconds() / 3600,
                         null_jump=n[1], null_hit=n[1] >= jump,
                         null_err_h=(n[0] - (r["pred_last"] + shift)).total_seconds() / 3600))
    return pd.DataFrame(rows)


def summarize(m, jump, win=36):
    yrs = max((m["pred_last"].max() - m["pred_last"].min()).days / 365.25, 0.1)
    print(f"\n{len(m)} predicted arrivals with usable observations ({m['pred_last'].min():%Y-%m} .. {m['pred_last'].max():%Y-%m}, "
          f"{len(m) / yrs:.0f} per year)")
    print(f"\nHIT = speed jump >= {jump:.0f} km/s within +-{win} h.  'null' = the same test at a random date 7-60 days away.")
    print(f"  hit rate   real {100 * m['hit'].mean():.0f}%   null {100 * m['null_hit'].mean():.0f}%")
    for w in (6, 12, 24):
        real = (m["hit"] & (m["err_last_h"].abs() <= w)).mean()
        null = (m["null_hit"] & (m["null_err_h"].abs() <= w)).mean()
        print(f"  hit within +-{w:>2} h   real {100 * real:4.0f}%   null {100 * null:4.0f}%   (null is what pure chance gives)")
    edges = np.arange(-win, win + 1, 12)
    hr = np.histogram(m.loc[m["hit"], "err_last_h"], edges)[0] / len(m) * 100
    hn = np.histogram(m.loc[m["null_hit"], "null_err_h"], edges)[0] / len(m) * 100
    print("\n  timing of the largest jump relative to the predicted arrival, % of all predictions per 12 h bin:")
    print("  bin (h)      " + "".join(f"{f'{int(a_)}..{int(b_)}':>10}" for a_, b_ in zip(edges[:-1], edges[1:])))
    print("  real         " + "".join(f"{x:10.1f}" for x in hr))
    print("  null (chance)" + "".join(f"{x:10.1f}" for x in hn))
    h = m[m["hit"]]
    print(f"\n  median lead time of the first prediction: {m['lead_first_h'].median():.0f} h before the predicted arrival")
    print(f"  median jump of hits {h['jump'].median():.0f} km/s (null {m.loc[m['null_hit'], 'null_jump'].median():.0f}), "
          f"speed after {h['v_after'].median():.0f} km/s")
    for col, edges_, labels in (("cme_speed", [0, 600, 900, 1200, 5000], ["<600", "600-900", "900-1200", ">1200"]),
                                ("kp_max", [-1, 4.5, 6.5, 10], ["Kp<=4", "Kp5-6", "Kp>=7"]),
                                ("half_angle", [0, 25, 40, 90], ["<25", "25-40", ">40"])):
        g = m.assign(b=pd.cut(m[col], edges_, labels=labels)).groupby("b", observed=True)
        t = g.agg(n=("hit", "size"), hit=("hit", "mean"), null=("null_hit", "mean"), jump=("jump", "median"))
        t["hit"], t["null"] = (100 * t["hit"]).round(0), (100 * t["null"]).round(0)
        print(f"\nby {col} (hit and null in %):")
        print(t.round(0).to_string())
    print(f"\ncorrelation of CME speed with the observed jump (hits): {h['cme_speed'].corr(h['jump']):.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--start", default="2013-01-01")
    f.add_argument("--end", default=str(pd.Timestamp.now("UTC").date()))
    f.add_argument("--key", default="DEMO_KEY")
    f.add_argument("--out", default="data/donki_cme.csv")
    e = sub.add_parser("evaluate")
    e.add_argument("--csv", default="data/donki_cme.csv")
    e.add_argument("--obs", action="append", required=True, help="path:speed_column  (repeat; first file wins on overlap)")
    e.add_argument("--jump", type=float, default=60.0)
    e.add_argument("--win", type=int, default=36)
    e.add_argument("--query", default=None, help='pandas filter on the matched events, e.g. "kp_max >= 7 and half_angle > 40"')
    e.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "fetch":
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        fetch(a.start, a.end, a.key, a.out)
    else:
        d = pd.read_csv(a.csv)
        for c in ("cme_start", "submitted", "time21_5", "enlil_done", "shock_arrival", "known_from"):
            if c in d.columns:
                d[c] = pd.to_datetime(d[c], errors="coerce")
        ev = predicted_arrivals(d.dropna(subset=["cme_id"]))
        print(f"{len(ev)} distinct predicted arrivals in {a.csv}")
        m = evaluate(ev, load_obs(a.obs), a.jump, a.win)
        if a.query:
            m = m.query(a.query)
            print(f"subset: {a.query}  ({len(m)} events)")
        summarize(m, a.jump, a.win)
        if a.out:
            m.to_csv(a.out, index=False)
            print("wrote", a.out)


if __name__ == "__main__":
    main()
