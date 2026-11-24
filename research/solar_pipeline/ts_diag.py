"""Find which setup reproduces data/timestral_speed_features.parquet.  Run in benchlabSolarWindPrediction:
python ts_diag.py --speed data/swpc_ace_historical/swpc_ace_swepam_hourly.parquet --col speed_km_s \
    --stored ../hss_prediction/data/timestral_speed_features.parquet
"""
import argparse, itertools
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster

ap = argparse.ArgumentParser(); ap.add_argument("--speed", required=True); ap.add_argument("--col", default="speed_km_s")
ap.add_argument("--stored", required=True); ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()
d = pd.read_parquet(a.speed)
raw = d.set_index(pd.to_datetime(d.timestamp).dt.tz_convert("UTC").dt.tz_localize(None))[a.col].astype(float)
valid = raw[(raw > 0) & (raw < 3000)]
series = {"clean-interp": valid.resample("1h").mean(), "raw-unfiltered": raw.resample("1h").mean()}
st = pd.read_parquet(a.stored); st["t0"] = pd.to_datetime(st.t0)
f = TimestralForecaster.from_pretrained("mistralai/Timestral-123M", device="cpu")
t0s = sorted(st.t0.unique())[: a.n]
print("quantile_levels:", None)
CTX = 1344
rows = []
for t0 in t0s:
    ref = st[st.t0 == t0].sort_values("lead_time")
    rm, rs = ref.timestral_med.values, ref.timestral_spread.values
    for sname, s in series.items():
        for lat in (0, 1, 3, 6, 24, 72):
            end = pd.Timestamp(t0) - pd.Timedelta(hours=lat)
            idx = pd.date_range(end - pd.Timedelta(hours=CTX - 1), end, freq="1h")
            ctx = s.reindex(idx).interpolate(limit_direction="both").ffill().bfill().values.astype(np.float32)
            fc = f.forecast(ctx[None, None, :], prediction_length=72)
            if not rows:
                print("quantile_levels:", np.round(np.asarray(fc.quantile_levels), 3).tolist())
            med = np.asarray(fc.median[0, -1, :], float)
            Q = np.asarray(fc.quantiles[0, -1, :, :], float)            # (72, nq)
            lv = np.asarray(fc.quantile_levels, float)
            def q(p): return Q[:, int(np.argmin(abs(lv - p)))]
            for shift in (-1, 0, 1):
                m = np.roll(med, -shift)[2:-2]; r = rm[2:-2]
                rows.append(dict(t0=t0, series=sname, lat=lat, shift=shift, kind="med", mae=np.abs(m - r).mean()))
            for nm, sp in (("q90-med", q(.9) - med), ("q95-med", q(.95) - med), ("q90-q10", q(.9) - q(.1)), ("q99-med", q(.99) - med),
                           ("q75-q25", q(.75) - q(.25)), ("q90-q50(nearest)", q(.9) - q(.5))):
                rows.append(dict(t0=t0, series=sname, lat=lat, shift=0, kind=nm, mae=np.abs(sp - rs).mean()))
R = pd.DataFrame(rows)
print("\nMEDIAN: MAE vs stored (km/s), averaged over t0")
print(R[R.kind == "med"].groupby(["series", "lat", "shift"]).mae.mean().sort_values().head(8).round(2).to_string())
print("\nSPREAD: MAE vs stored, averaged over t0 (the stored mean spread is %.1f)" % st.timestral_spread.mean())
print(R[R.kind != "med"].groupby(["kind", "series", "lat"]).mae.mean().sort_values().head(8).round(2).to_string())
