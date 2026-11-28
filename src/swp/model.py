#!/usr/bin/env python3
"""Solar-wind speed forecast, t0+1h..t0+72h.   Usage:  python3 model.py 20261111T060000Z   -> JSON on stdout (logs on stderr).

Model: per-lead-group ridge blend of  [persistence, 27-day recurrence, WSA-ENLIL]  (fitted on 2025-01..2025-09 only)
       + 0.75 * coronal-hole adjustment from near-real-time AIA 193 images (ridge fitted on <= 2024).
Information cut-off: ACE measurements labelled <= t0 - 3 h only; ENLIL runs started <= t0 - 12 h only; AIA images <= t0 - 1 h.
Every layer has a fallback; the output always has 72 finite values.
"""
import json, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import urllib.request
from swp import diagnostics
import numpy as np
import pandas as pd
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")   # blend / strip model files
H = 72
LAG_H = 3                  # ACE hourly labels admitted: <= t0 - LAG_H
ENLIL_DELAY_H = 12         # an ENLIL run is admitted only if its start time <= t0 - 12 h
ENLIL_MAX_AGE_H = 72       # ... and not older than 72 h
ENLIL_MAX_LEAD = 72        # training archive covers leads 0..72 h only
STRIP_LAMBDA = 0.75
CLIM = 440.0               # last-resort value if there is no ACE data at all
SWPC = "https://services.swpc.noaa.gov"
ACE_MONTHLY = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/{ym}_ace_swepam_1h.txt"
ACE_JSON = SWPC + "/json/ace/swepam/ace_swepam_1h.json"
ENLIL_JSON = SWPC + "/json/enlil_time_series.json"
T_START = time.time()


def log(*a):
    print(f"[{time.time() - T_START:6.1f}s]", *a, file=sys.stderr, flush=True)


def http_get(url, timeout=30, retries=2):
    last = None
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (swp-submission)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:                      # noqa
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


# ------------------------------------------------------------------------------------------------ ACE history
def parse_monthly(text):
    rows = {}
    for ln in text.splitlines():
        p = ln.split()
        if len(p) < 9 or not p[0].isdigit():
            continue
        try:
            if int(p[6]) != 0:
                continue
            v = float(p[8])
            if v <= 0 or v > 3000:
                continue
            rows[pd.Timestamp(int(p[0]), int(p[1]), int(p[2]), int(p[3][:2]), int(p[3][2:]))] = v
        except ValueError:
            continue
    return rows

def fetch_ace(t0, days=36):
    """Valid (status 0) hourly ACE speed with label <= t0 - LAG_H, as an hourly Series (NaN where missing).
    Raises RuntimeError if nothing could be fetched because every source failed; returns (None, src) if the
    sources answered but no usable data is left."""
    cut = t0 - pd.Timedelta(hours=LAG_H)
    rows, src, errors = {}, [], []
    for ym in pd.period_range(t0 - pd.Timedelta(days=days), t0, freq="M").strftime("%Y%m"):
        try:
            rows.update(parse_monthly(http_get(ACE_MONTHLY.format(ym=ym), 40).decode("ascii", "replace")))
            src.append(f"monthly:{ym}")
        except Exception as e:
            log("ACE monthly", ym, "failed:", repr(e)[:80])
            errors.append(f"monthly:{ym}: {e!r}"[:100])
    if not rows or max(rows) < cut - pd.Timedelta(days=3):          # monthly files missing/stale -> SWPC json (dsflag 0)
        try:
            for r in json.loads(http_get(ACE_JSON, 40)):
                if r.get("dsflag") == 0 and r.get("speed") is not None and 0 < r["speed"] < 3000:
                    rows[pd.Timestamp(r["time_tag"]).tz_localize(None) if pd.Timestamp(r["time_tag"]).tzinfo else pd.Timestamp(r["time_tag"])] = float(r["speed"])
            src.append("json")
        except Exception as e:
            log("ACE json failed:", repr(e)[:80])
            errors.append(f"json: {e!r}"[:100])
    if not rows:
        if errors:                                   # nothing fetched and something went wrong: a failure, not "empty"
            raise RuntimeError("ACE unavailable: " + "; ".join(errors)[:300])
        return None, src
    s = pd.Series(rows).sort_index()
    s = s[s.index <= cut]
    s = s[~s.index.duplicated(keep="last")]
    s = s[s.index == s.index.floor("h")]
    if s.empty:
        return None, src
    return s.reindex(pd.date_range(s.index.min().floor("D"), cut, freq="h")), src

def load_ace_rows(t_ref, days=36):
    """All valid ACE speeds (status 0), without the competition lag cut: (dict timestamp -> km/s, sources, errors)."""
    cut = t_ref - pd.Timedelta(hours=LAG_H)
    rows, src, errors = {}, [], []
    for ym in pd.period_range(t_ref - pd.Timedelta(days=days), t_ref, freq="M").strftime("%Y%m"):
        try:
            rows.update(parse_monthly(http_get(ACE_MONTHLY.format(ym=ym), 40).decode("ascii", "replace")))
            src.append(f"monthly:{ym}")
        except Exception as e:
            log("ACE monthly", ym, "failed:", repr(e)[:80])
            errors.append(f"monthly:{ym}: {e!r}"[:100])
    if not rows or max(rows) < cut - pd.Timedelta(days=3):          # monthly files missing/stale -> SWPC json (dsflag 0)
        try:
            for r in json.loads(http_get(ACE_JSON, 40)):
                if r.get("dsflag") == 0 and r.get("speed") is not None and 0 < r["speed"] < 3000:
                    rows[pd.Timestamp(r["time_tag"]).tz_localize(None) if pd.Timestamp(r["time_tag"]).tzinfo else pd.Timestamp(r["time_tag"])] = float(r["speed"])
            src.append("json")
        except Exception as e:
            log("ACE json failed:", repr(e)[:80])
            errors.append(f"json: {e!r}"[:100])
    return rows, src, errors


# ------------------------------------------------------------------------------------------------ ENLIL (latest run)
def fetch_enlil():
    """(run_start, Series of Earth radial speed on the hourly grid run_start.., km/s) or None."""
    payload = json.loads(http_get(ENLIL_JSON, 60))
    if isinstance(payload, dict):
        payload = next((v for v in payload.values() if isinstance(v, list)), [])
    recs = [(r["time_tag"], r["v_r"]) for r in payload if isinstance(r.get("time_tag"), str) and r.get("v_r") is not None]
    if len(recs) < 10:
        return None
    t = pd.to_datetime([a for a, _ in recs], utc=True).tz_localize(None)
    v = np.array([b for _, b in recs], float)
    o = np.argsort(t)
    t, v = t[o], v[o]
    run = t.min().round("h")
    grid = pd.date_range(run, t.max().floor("h"), freq="h")
    ep = pd.Timestamp("1970-01-01")
    x = np.interp(((grid - ep) / pd.Timedelta(seconds=1)).to_numpy(float),
                  ((t - ep) / pd.Timedelta(seconds=1)).to_numpy(float), v)
    return run, pd.Series(x, index=grid)

def _clean(v):
    return [None if not np.isfinite(x) else round(float(x), 1) for x in v]
# ------------------------------------------------------------------------------------------------ core (pure function)
def enlil_for(t0, enlil):
    """72 values (NaN where unavailable): latest run, admitted only if started <= t0-12h and age <= 72h, lead <= 72h."""
    out = np.full(H, np.nan)
    if enlil is None:
        return out, None
    run, ser = enlil
    if run > t0 - pd.Timedelta(hours=ENLIL_DELAY_H) or (t0 - run) / pd.Timedelta(hours=1) > ENLIL_MAX_AGE_H:
        return out, None
    tgt = t0 + pd.to_timedelta(np.arange(1, H + 1), unit="h")
    lead = np.rint((tgt - run) / pd.Timedelta(hours=1)).astype(int)
    vals = ser.reindex(tgt).to_numpy(float).copy()
    vals[(lead < 0) | (lead > ENLIL_MAX_LEAD)] = np.nan
    return vals, run


def core(t0, obs, enlil, strip_adj, blend, lam=STRIP_LAMBDA):
    """obs: hourly Series (labels <= t0-3h, NaN gaps); enlil: (run, Series) or None; strip_adj: 72 array or None."""
    tgt = t0 + pd.to_timedelta(np.arange(1, H + 1), unit="h")
    if obs is None or obs.dropna().empty:
        pers = np.nan
        rec = np.full(H, np.nan)
    else:
        pers = float(obs.ffill().iloc[-1])                                       # latest valid value (label <= t0-3h)
        v27 = obs.reindex(tgt - pd.Timedelta(days=27)).to_numpy(float).copy()           # the value 27 days before the target
        m12 = obs.rolling(25, center=True, min_periods=8).mean().reindex(tgt - pd.Timedelta(days=27)).to_numpy(float)
        rec = np.where(np.isfinite(v27), v27, m12)                               # rec27_eff
    enl, run = enlil_for(t0, enlil)
    X = np.stack([np.full(H, pers), rec, enl], axis=1)                           # [persistence, rec27_eff, enlil]
    base = np.where(np.isfinite(pers), pers, blend["fallback"])
    X = np.where(np.isfinite(X), X, base)                                        # a missing input -> persistence
    pred = np.empty(H)
    keys = np.array(sorted(int(k) for k in blend["params"]))
    for i in range(H):
        gi = i // blend["width"]
        g = int(keys[np.argmin(np.abs(keys - gi))])
        p = blend["params"][str(g)]
        pred[i] = p["const"] + X[i] @ np.array(p["beta"])
    adj = np.zeros(H) if strip_adj is None else np.nan_to_num(np.asarray(strip_adj, float))
    comps = dict(rec27=rec, enlil=enl, blend=pred, adj=lam * adj)
    return pred + lam * adj, dict(persistence=pers, enlil_run=None if run is None else str(run),
                                  n_enlil=int(np.isfinite(enl).sum()), n_rec=int(np.isfinite(rec).sum()),
                                  strip_used=strip_adj is not None, _components=comps)
# ------------------------------------------------------------------------------------------------ coronal-hole (AIA NRT) adjustment
def strip_adjustment(t0, deadline_s=240):
    """(72-array or None, info dict) from NRT AIA 193 images (daily 00:00 UTC slots).  Any problem -> (None, info)."""
    from swp import segment_ch as S
    import requests
    started = time.time()
    meta = json.load(open(os.path.join(HERE, "strip_model_meta.json")))
    z = np.load(os.path.join(HERE, "strip_model.npz"))
    delay, tol = meta["delay_h"], meta["tol_h"]
    hi = t0 - pd.Timedelta(hours=delay)
    lo = t0 - pd.Timedelta(hours=delay + 72 + tol)
    days = [d for d in pd.date_range(lo.normalize(), hi.normalize(), freq="D") if d + pd.Timedelta(minutes=25) <= pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)]
    S.SOURCE = "nrt"
    t_end = time.time() + deadline_s
    fails = []
    from astropy.io import fits

    def one(day):
        tag = f"{day:%Y-%m-%d}"
        try:
            base = S.NRT_DIR.format(y=f"{day:%Y}", m=f"{day:%m}", d=f"{day:%d}", hh="00")
            r = requests.get(base, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
            if r.status_code != 200:
                fails.append(f"{tag}:index_http_{r.status_code}")
                return None
            name = S.nrt_pick(r.text, day, "0000")
            if name is None:
                fails.append(f"{tag}:no_file")
                return None
            if time.time() > t_end:
                fails.append(f"{tag}:deadline")
                return None
            r = requests.get(base + name, headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
            if r.status_code != 200 or b"SIMPLE" not in r.content[:100]:
                fails.append(f"{tag}:fits_http_{r.status_code}")
                return None
            from io import BytesIO
            with fits.open(BytesIO(r.content)) as hd:
                hdu = next(x for x in hd if x.data is not None and x.data.ndim == 2)
                row, _ = S.process_image(np.asarray(hdu.data, float), S.safe_header(hdu.header))
            row["slot"] = day
            return row
        except Exception as e:
            log("AIA", tag, repr(e)[:90])
            fails.append(f"{tag}:{e!r}"[:70])
            return None

    with ThreadPoolExecutor(max_workers=5) as ex:
        rows = [r for r in ex.map(one, days) if r is not None]
    log(f"AIA images: {len(rows)}/{len(days)}")
    sinfo = {"strip_n_days": len(days), "strip_n_images": len(rows), "strip_fail_reasons": sorted(fails)[:6],
             "strip_fetch_s": round(time.time() - started, 1)}
    if not rows:
        return None, sinfo
    strips = pd.DataFrame(rows).dropna(subset=["t_obs"]).set_index("t_obs").sort_index()[meta["strip_cols"]]
    newest = strips.index.max()
    sinfo["strip_newest_obs"] = newest.strftime("%Y-%m-%dT%H:%M:%SZ")
    sinfo["strip_newest_age_h"] = round((t0 - newest) / pd.Timedelta(hours=1), 2)
    adj, used = strip_from_images(t0, strips, meta, z)
    sinfo["strip_image_times"] = used
    sinfo["strip_lags_matched"] = sum(u is not None for u in used)
    return adj, sinfo


def strip_from_images(t0, strips, meta, z):
    """(72-array or None, [matched image time or None for lags 0/24/48/72])."""
    s = strips.rename_axis("ti").reset_index()
    parts, used = [], []
    for lag in (0, 24, 48, 72):
        q = pd.DataFrame({"t0": [t0 - pd.Timedelta(hours=lag + meta["delay_h"])], "k": [0]})
        j = pd.merge_asof(q, s, left_on="t0", right_on="ti", direction="backward", tolerance=pd.Timedelta(hours=meta["tol_h"]))
        ti = j["ti"].iloc[0]
        used.append(None if pd.isna(ti) else ti.strftime("%Y-%m-%dT%H:%M:%SZ"))
        parts.append(j.drop(columns=["t0", "ti", "k"]).reset_index(drop=True).add_suffix(f"_L{lag}"))
    Xi = pd.concat(parts, axis=1)
    assert list(Xi.columns) == meta["columns"], "image feature columns differ from the fitted model"
    if not Xi.notna().any(axis=1).iloc[0]:
        return None, used
    Z = (Xi.fillna(0.0).to_numpy() - z["mean"]) / z["scale"]
    raw = (Z @ z["coef"].T + z["intercept"])[0]
    return np.clip(raw, z["adj_min"], z["adj_max"]), used


def finalize(t0, obs, src, enlil, adj, info = None):
    info = dict(info or {}, level="full", ace_sources=src)
    if obs is not None:
        info["ace_last_label"] = str(obs.dropna().index.max())
        info["ace_cut"] = str(t0 - pd.Timedelta(hours=LAG_H))
    blend = json.load(open(os.path.join(HERE, "final_blend.json")))
    pred, meta = core(t0, obs, enlil, adj, blend)
    comps = meta.pop("_components")
    info.update(meta)
    if obs is None or not np.isfinite(meta["persistence"]):
        info["level"] = "no_ace"
        base = enlil_for(t0, enlil)[0]
        pred = np.where(np.isfinite(base), base, CLIM)
    elif adj is None:
        info["level"] = "no_strip"
    if not np.all(np.isfinite(pred)):
        info["level"] = "persistence"
        p = meta["persistence"]
        pred = np.full(H, p if np.isfinite(p) else CLIM)
    final = np.clip(pred, 180.0, 1200.0)
    info.update(diagnostics.describe_cycle(t0, obs, enlil, adj, final))
    info["components"] = {k: _clean(v) for k, v in {**comps, "final": final}.items()}
    return final, info


# ------------------------------------------------------------------------------------------------ orchestration
def predict(t0):
    info = {}
    obs, src = None, []
    try:
        obs, src = fetch_ace(t0)
    except Exception as e:
        log("ACE failed:", repr(e)[:100]); info["ace_error"] = repr(e)[:100]
    enlil = None
    try:
        enlil = fetch_enlil()
    except Exception as e:
        log("ENLIL failed:", repr(e)[:100]); info["enlil_error"] = repr(e)[:100]
    adj = None
    if os.environ.get("SWP_NO_STRIP") != "1":
        try:
            dj, sinfo = strip_adjustment(t0)
        except Exception as e:
            log("strip failed:", repr(e)[:120]); info["strip_error"] = repr(e)[:120]
    return finalize(t0, obs, src, enlil, adj, info)


def main():
    if len(sys.argv) != 2:
        print("usage: model.py <YYYYMMDDTHHMMSSZ>", file=sys.stderr); return 2
    t0 = pd.Timestamp(datetime.strptime(sys.argv[1], "%Y%m%dT%H%M%SZ"))
    try:
        pred, info = predict(t0)
    except Exception:                                  # last-resort net: never exit without a valid forecast
        traceback.print_exc(file=sys.stderr)
        pred, info = np.full(H, CLIM), dict(level="crash_fallback")
    info["retrieved_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log("info:", info)
    log("forecast_kms:", json.dumps([round(float(v), 1) for v in pred]))
    json.dump({"t0": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "valid_from": (t0 + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "forecast_speed_kms": [float(v) for v in pred]}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())