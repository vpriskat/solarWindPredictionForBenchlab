"""Inspect two public feeds the organisers allow (anything public and available before t0):

  python check_new_feeds.py rtsw  [--ace ace_hourly.parquet] [--save rtsw_snapshot.parquet]
  python check_new_feeds.py donki --start 2026-01-10 --end 2026-01-21 [--key YOUR_NASA_KEY] [--csv donki_cme.csv] [--t0 2026-01-18T12:00]

rtsw : NOAA SWPC real-time solar wind (DSCOVR/ACE, 1-minute).  How fresh is the newest point, how many hours have data, which
       spacecraft is 'active', and (optional) how its hourly means compare with the ACE hourly file you use.
donki: NASA DONKI CME analyses (+ WSA-ENLIL runs and their estimated shock arrival).  One row per analysis / ENLIL run, with the
       times each record was submitted, so you can keep only what was known before an issue time t0 (--t0).
Run it on your machine (the sandbox cannot reach these hosts).  Field names are read defensively: unknown ones are printed.
"""
import argparse
import json
import sys
import urllib.request

import numpy as np
import pandas as pd

RTSW = "https://services.swpc.noaa.gov/json/rtsw/"
DONKI = "https://api.nasa.gov/DONKI/CME"


def _get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ------------------------------------------------------------------------------------------------ RTSW
def rtsw_frames():
    wind = pd.DataFrame(_get(RTSW + "rtsw_wind_1m.json"))
    mag = pd.DataFrame(_get(RTSW + "rtsw_mag_1m.json"))
    for d in (wind, mag):
        d["time_tag"] = pd.to_datetime(d["time_tag"], utc=True).dt.tz_localize(None)
    return wind, mag


def rtsw_report(wind, mag, now=None):
    now = pd.Timestamp.now("UTC").tz_localize(None) if now is None else now
    out = []
    for name, d, col in (("wind", wind, "proton_speed"), ("mag", mag, "bz_gsm")):
        if col not in d.columns:
            out.append(f"{name}: column {col} missing; columns are {list(d.columns)}")
            continue
        out.append(f"{name}: {len(d):,} rows, {d.time_tag.min()} .. {d.time_tag.max()}  "
                   f"(newest point is {(now - d.time_tag.max()).total_seconds() / 60:.0f} min old)")
        span = pd.date_range(d.time_tag.min().floor("h"), d.time_tag.max().floor("h"), freq="h")

        def cov(dd):                                         # share of hours with >=20 valid minutes
            v = dd.dropna(subset=[col])
            hrs = pd.Series(1, index=v.time_tag).resample("h").sum()
            return 100 * (hrs.reindex(span).fillna(0) >= 20).mean()

        act = d[d["active"].astype(bool)] if "active" in d.columns else d
        out.append(f"   all sources together : {cov(d):5.1f}% of hours   (rows per source: "
                   f"{d['source'].value_counts().to_dict() if 'source' in d.columns else '-'})")
        out.append(f"   the 'active' flag    : {cov(act):5.1f}% of hours   (rows per source: "
                   f"{act['source'].value_counts().to_dict() if 'source' in act.columns else '-'})")
        if "source" in d.columns:
            for sname, dd in d.groupby("source"):
                out.append(f"   source {str(sname):<8}     : {cov(dd):5.1f}% of hours  newest valid "
                           f"{dd.dropna(subset=[col]).time_tag.max()}")
    return "\n".join(out)


def rtsw_hourly(wind, source="ACE"):
    """Hourly mean speed of ONE spacecraft (default ACE = the scoring target's instrument)."""
    d = wind[wind["source"] == source] if source and "source" in wind.columns else wind
    return d.dropna(subset=["proton_speed"]).set_index("time_tag")["proton_speed"].resample("h").mean()


def compare_ace(hourly, ace_path):
    a = pd.read_parquet(ace_path)
    col = "speed_km_s" if "speed_km_s" in a.columns else a.columns[0]
    a = a[col]
    a.index = pd.to_datetime(a.index)
    j = pd.concat([hourly.rename("rtsw"), a.rename("ace")], axis=1, join="inner").dropna()
    if len(j) < 5:
        return f"no overlap between RTSW and {ace_path} ({len(j)} hours): fetch the live ACE file the same minute and rerun"
    e = j.rtsw - j.ace
    best = max(range(-3, 4), key=lambda s: j.rtsw.shift(s).corr(j.ace))
    return (f"RTSW vs ACE hourly on {len(j)} hours: bias {e.mean():+.1f} km/s, RMSE {np.sqrt((e ** 2).mean()):.1f}, "
            f"corr {j.rtsw.corr(j.ace):.3f}, best shift {best} h")


# ----------------------------------------------------------------------------------------------- DONKI
def donki_rows(cmes):
    rows = []
    for c in cmes:
        base = dict(cme_id=c.get("activityID"), cme_start=c.get("startTime"), cme_note=(c.get("note") or "")[:80])
        for a in (c.get("cmeAnalyses") or [{}]):
            ba = dict(base, submitted=a.get("submissionTime"), time21_5=a.get("time21_5"), speed=a.get("speed"),
                      half_angle=a.get("halfAngle"), lon=a.get("longitude"), lat=a.get("latitude"),
                      type=a.get("type"), most_accurate=a.get("isMostAccurate"))
            runs = a.get("enlilList") or [{}]
            for e in runs:
                rows.append(dict(ba, enlil_done=e.get("modelCompletionTime"), shock_arrival=e.get("estimatedShockArrivalTime"),
                                 duration_h=e.get("estimatedDuration"), earth_glancing=e.get("isEarthGB"),
                                 kp_max=max([e.get(k) or 0 for k in ("kp_18", "kp_90", "kp_135", "kp_180")] or [0])))
    d = pd.DataFrame(rows)
    for c in ("cme_start", "submitted", "time21_5", "enlil_done", "shock_arrival"):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce", utc=True).dt.tz_localize(None)
    # the moment this row's information existed: the later of the analysis and (if present) the ENLIL run completion
    d["known_from"] = d[["submitted", "enlil_done"]].max(axis=1)
    return d


def known_at(d, t0):
    """Rows usable for a forecast issued at t0: submitted (and, for the ENLIL columns, completed) before t0.
    ENLIL columns of rows whose run finished later are blanked, not the whole analysis."""
    t0 = pd.Timestamp(t0)
    k = d[d["submitted"] <= t0].copy()
    late = k["enlil_done"] > t0
    k.loc[late, ["shock_arrival", "duration_h", "earth_glancing", "kp_max"]] = np.nan
    return k


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rtsw")
    r.add_argument("--ace", default=None)
    r.add_argument("--save", default=None, help="write the 1-minute wind+mag snapshot here (start logging the live feed now)")
    d = sub.add_parser("donki")
    d.add_argument("--start", required=True)
    d.add_argument("--end", required=True)
    d.add_argument("--key", default="DEMO_KEY")
    d.add_argument("--csv", default=None)
    d.add_argument("--t0", default=None, help="show only what was known before this issue time (UTC)")
    a = ap.parse_args()
    if a.cmd == "rtsw":
        wind, mag = rtsw_frames()
        print(rtsw_report(wind, mag))
        if a.ace:
            for src in sorted(wind["source"].unique()) if "source" in wind.columns else [None]:
                print(f"[{src}] " + compare_ace(rtsw_hourly(wind, src), a.ace))
        if a.save:
            pd.concat({"wind": wind, "mag": mag}, names=["feed"]).to_parquet(a.save)
            print("saved", a.save)
    else:
        cm = _get(f"{DONKI}?startDate={a.start}&endDate={a.end}&api_key={a.key}")
        print(f"{len(cm)} CMEs; top-level fields: {sorted(cm[0].keys()) if cm else '-'}")
        d = donki_rows(cm)
        if a.t0:
            d = known_at(d, a.t0)
            print(f"restricted to what was known before {a.t0}")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(d.drop(columns=["cme_note"]).sort_values("known_from").to_string(index=False))
        if a.csv:
            d.to_csv(a.csv, index=False)
            print("wrote", a.csv)


if __name__ == "__main__":
    main()
