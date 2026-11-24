"""Which hour did a prediction file see last?  (needs only the parquet and the ACE file, no model code)

A lead-1 forecast is almost the last observed value, so corr(h01 at row time T, ACE at T+s) peaks at s = (last input hour) - T.
Report that relative to the issue time t0 = T - index_offset:
    0   inputs end at t0 (no latency: NOT reproducible live, where ACE is ~3 h old)
   -3   inputs end at t0-3 h (live-like)

    python check_tcn_alignment.py --tcn results/net_predictions_2025_2026.parquet --index-offset 72 --ace data/swpc_ace_historical/swpc_ace_swepam_hourly.parquet
    python check_tcn_alignment.py --tcn runs/v2_dropB_small/net_predictions_v2.parquet --index-offset 0 --ace ...
    python check_tcn_alignment.py --selftest
"""
import argparse

import numpy as np
import pandas as pd


def load_obs(path):
    a = pd.read_parquet(path)
    ts = pd.to_datetime(a["timestamp"], utc=True).dt.tz_localize(None).dt.floor("h")
    s = pd.Series(a["speed_km_s"].to_numpy(float), index=ts)
    return s[~s.index.duplicated(keep="last")].sort_index()


def corr_by_shift(pred_df, obs, col, shifts=range(-240, 25)):
    idx = pd.to_datetime(pred_df.index, utc=True).tz_localize(None)
    p = pred_df[col].to_numpy(float)
    out = {}
    for s in shifts:
        o = obs.reindex(idx + pd.Timedelta(hours=s)).to_numpy(float)
        m = np.isfinite(p) & np.isfinite(o)
        out[s] = np.corrcoef(p[m], o[m])[0, 1] if m.sum() > 50 else np.nan
    return pd.Series(out)


def report(pred_df, obs, offset, extra=()):
    idx = pd.to_datetime(pred_df.index, utc=True).tz_localize(None)
    print(f"rows {len(pred_df)}  index {idx.min()} .. {idx.max()}  median step {pd.Series(idx).diff().median()}  "
          f"columns {pred_df.shape[1]}  NaN share {np.isnan(pred_df.to_numpy(float)).mean():.3f}")
    res = {}
    for col in ("h01", "h02", "h03"):
        c = corr_by_shift(pred_df, obs, col)
        s = int(c.idxmax())
        res[col] = s
        near = "  ".join(f"{k:+d}:{c.get(s + k, np.nan):.3f}" for k in (-3, -1, 0, 1, 3))
        print(f"  {col}: best shift {s:+d} h (corr {c.max():.3f})   corr around the peak [{near}]")
    for col in extra:
        if col not in pred_df.columns:
            print(f"  {col}: not in file")
            continue
        c = corr_by_shift(pred_df, obs, col)
        print(f"  {col}: best shift {int(c.idxmax()):+d} h (corr {c.max():.3f})   corr at shift 0: {c.get(0, np.nan):.3f}   "
              f"at -{offset}: {c.get(-offset, np.nan):.3f}")
    rel = res["h01"] + offset
    print(f"\n=> the last input hour is about t0 {rel:+d} h  (t0 = row time - {offset} h)")
    print("   0 or later: the file did not see the ACE latency; -3: live-like.  (assumes h01 is close to the last observed V)")
    return rel


def selftest():
    rng = np.random.default_rng(0)
    n = 24 * 400
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    v = pd.Series(450 + lfilter_ar(rng.normal(0, 25, n)), index=idx)
    t0 = idx[1000:n - 100:6]
    for lat, off in ((0, 72), (3, 72), (3, 0)):
        pred = pd.DataFrame({f"h{k:02d}": v.reindex(t0 - pd.Timedelta(hours=lat)).to_numpy() + rng.normal(0, 3, len(t0))
                             for k in (1, 2, 3)}, index=(t0 + pd.Timedelta(hours=off)).tz_localize("UTC"))
        obs = v.mask(rng.random(n) < 0.3)                                # gappy like the real feed
        rel = report(pred, obs, off)
        assert rel == -lat, (rel, lat, off)
    print("\nselftest passed")


def lfilter_ar(x, phi=0.97):
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = phi * y[i - 1] + x[i]
    return y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcn")
    ap.add_argument("--ace")
    ap.add_argument("--index-offset", type=int, default=0, help="hours the file's index is ahead of t0 (old file: 72)")
    ap.add_argument("--extra", default="", help="more columns to inspect, e.g. obs,net,h24,h48,h60,h72")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        report(pd.read_parquet(a.tcn), load_obs(a.ace), a.index_offset, [c for c in a.extra.split(',') if c])
