"""
Rebuild of data/timestral_speed_features.parquet (columns: t0, lead_time, timestral_med, timestral_spread).
Timestral-123M, univariate on hourly ACE/OMNI bulk speed, 56-day context, 72-step forecast.
  median = forecast median ; spread = p90 - median.

Also importable:  speed_features(forecaster, speed_series, t0, latency_h=0)  -> DataFrame(72 rows)

Usage (validate first against your stored file):
  python timestral_speed.py --speed data/swpc_ace_historical/swpc_ace_swepam_hourly.parquet \
      --col speed --t0-start 2025-01-01 --t0-end 2025-01-15 --step-h 6 --latency 0 \
      --out ts_speed_test.parquet --compare data/timestral_speed_features.parquet
Try --latency 0 and 3 (and --step-h 24 if your stored t0 are daily) and keep the one that matches.
"""
import argparse
import numpy as np
import pandas as pd

CTX = 56 * 24
H = 72


def load_speed(path, col="speed"):
    d = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    for tc in ("time", "timestamp", "datetime", "time_tag", "date", "t"):
        if tc in d.columns:
            d = d.set_index(pd.to_datetime(d[tc])); break
    d.index = pd.to_datetime(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_convert("UTC").tz_localize(None)
    if col not in d.columns:
        cand = [c for c in d.columns if any(k in c.lower() for k in ("speed", "vel", "flow", "bulk"))]
        raise SystemExit(f"column {col!r} not in file. columns: {list(d.columns)}\n  speed-like: {cand}  -> pass --col <name>")
    s = d[col].astype(float)
    s = s[(s > 0) & (s < 3000)]
    return s.resample("1h").mean()


def _last(x):
    x = np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, float)
    if x.ndim == 3:                              # (B, T, H)
        x = x[0, -1, :]
    elif x.ndim == 2:                            # (T, H) or (B, H)
        x = x[-1, :]
    x = x.reshape(-1)
    assert x.size == H, f"unexpected forecast shape, got {x.size} values"
    return x


def speed_features(forecaster, s, t0, latency_h=0):
    end = pd.Timestamp(t0) - pd.Timedelta(hours=latency_h)
    idx = pd.date_range(end - pd.Timedelta(hours=CTX - 1), end, freq="1h")
    ctx = s.reindex(idx).interpolate(limit_direction="both")
    if ctx.isna().any() or ctx.notna().sum() < CTX * 0.8:
        return None
    arr = ctx.values.astype(np.float32)[None, None, :]          # (B=1, V=1, T): the model wants variates before time
    fc = forecaster.forecast(arr, prediction_length=H)
    med = _last(fc.median)                      # median is (B, T_context, H): the forecast issued at the LAST context step
    p90 = _last(fc.quantile(0.9))
    return pd.DataFrame({"t0": pd.Timestamp(t0), "lead_time": np.arange(1, H + 1),
                         "timestral_med": med.astype(float), "timestral_spread": (p90 - med).astype(float)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", required=True); ap.add_argument("--col", default="speed")
    ap.add_argument("--t0-start", required=True); ap.add_argument("--t0-end", required=True)
    ap.add_argument("--step-h", type=int, default=6); ap.add_argument("--latency", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--compare")
    a = ap.parse_args()
    import torch
    from timestral import TimestralForecaster
    torch.manual_seed(0)
    fcst = TimestralForecaster.from_pretrained("mistralai/Timestral-123M", device="cpu")
    s = load_speed(a.speed, a.col)
    out = []
    for t0 in pd.date_range(a.t0_start, a.t0_end, freq=f"{a.step_h}h"):
        r = speed_features(fcst, s, t0, a.latency)
        if r is not None: out.append(r)
    df = pd.concat(out, ignore_index=True); df.to_parquet(a.out); print("saved", a.out, len(df))
    if a.compare:
        ref = pd.read_parquet(a.compare)
        m = df.merge(ref, on=["t0", "lead_time"], suffixes=("", "_ref"))
        print(f"matched rows {len(m)}")
        for c in ("timestral_med", "timestral_spread"):
            print(c, "MAE", (m[c] - m[c + "_ref"]).abs().mean(), " corr", m[c].corr(m[c + "_ref"]))
