"""CME arrival kernel: turn DONKI/ENLIL predicted shock arrivals (strong CMEs only) into one number per (t0, lead).

  python cme_kernel.py build --arrivals data/donki_arrivals.csv \
      --obs ../benchlabSolarWindPrediction/data/omni_2020_2024_hourly.parquet:speed \
      --obs ../benchlabSolarWindPrediction/data/swpc_ace_historical/swpc_ace_swepam_hourly.parquet:speed_km_s \
      --train-end 2024-12-31 --t0-start 2025-01-01 --t0-end 2026-10-31 --out data/cme_kernel.parquet

Idea: for strong CMEs (default Kp>=7 and half angle >40, the subset where the predicted arrival beats chance) the average
observed speed anomaly around the predicted arrival time, K(tau) = mean_e [ v_e(T_e + tau) - baseline_e ], is estimated on the
TRAINING years only (the baseline is the mean speed 48-84 h before the predicted arrival = what a persistence forecast knows).
The kernel already contains the timing error of the predictions (it is centred on the PREDICTED time) and the misses.
For every issue time t0 we use only predictions known before t0 (the later ENLIL run if it exists), and the feature for lead h is
sum_e K(t0 + h - T_e)   [km/s].  The blend adds  lambda * feature  to R; lambda is 0 / 0.5 / 1 and is judged on the test, never fitted there.
"""
import argparse

import numpy as np
import pandas as pd

from donki_history import load_obs

TAU = np.arange(-60, 97)                           # hours relative to the predicted arrival


def event_table(arrivals_csv, query="kp_max >= 7 and half_angle > 40"):
    m = pd.read_csv(arrivals_csv)
    for c in ("first_known", "pred_first", "last_known", "pred_last"):
        m[c] = pd.to_datetime(m[c])
    return m.query(query).sort_values("pred_last").reset_index(drop=True) if query else m


def estimate_kernel(events, obs, smooth=5, min_valid=0.7, verbose=False):
    """Mean speed anomaly vs hours since the predicted arrival, over events with enough valid observations."""
    rows = []
    for _, e in events.iterrows():
        T = pd.Timestamp(e["pred_last"]).round("h")        # obs are hourly: an arrival at hh:23 would match nothing
        win = obs.reindex(pd.date_range(T - pd.Timedelta(hours=84), T + pd.Timedelta(hours=96), freq="h"))
        if verbose:
            print(f"   event {pd.Timestamp(T):%Y-%m-%d %H}  valid {win.notna().mean():.2f}  {'used' if win.notna().mean() >= min_valid else 'DROPPED'}")
        if win.notna().mean() < min_valid:
            continue
        base = win[:T - pd.Timedelta(hours=48)].mean()
        if not np.isfinite(base):
            continue
        a = (win - base).reindex(T + pd.to_timedelta(TAU, unit="h")).to_numpy()
        rows.append(a)
    A = np.array(rows)
    K = np.nanmean(A, axis=0)
    K = pd.Series(K, index=TAU).rolling(smooth, center=True, min_periods=1).mean()
    return K, len(A)


def feature(events, K, t0s, horizon=72):
    """DataFrame (t0 x lead 1..H): sum over events known at t0 of K(t0 + h - T_used)."""
    t0s = pd.DatetimeIndex(t0s)
    kv = K.to_numpy()
    out = np.zeros((len(t0s), horizon))
    h = np.arange(1, horizon + 1)
    for _, e in events.iterrows():
        for i, t0 in enumerate(t0s):
            if t0 < e["first_known"]:
                continue
            T = e["pred_last"] if t0 >= e["last_known"] else e["pred_first"]
            tau = (t0 + pd.to_timedelta(h, unit="h") - T) / pd.Timedelta(hours=1)
            tau = np.round(tau).astype(int)
            ok = (tau >= TAU[0]) & (tau <= TAU[-1])
            out[i, ok] += kv[tau[ok] - TAU[0]]
    return pd.DataFrame(out, index=t0s, columns=[f"h{x:02d}" for x in h])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--arrivals", default="data/donki_arrivals.csv")
    b.add_argument("--obs", action="append", required=True)
    b.add_argument("--query", default="kp_max >= 7 and half_angle > 40")
    b.add_argument("--train-end", default="2024-12-31")
    b.add_argument("--t0-start", default="2025-01-01")
    b.add_argument("--t0-end", default="2026-10-31")
    b.add_argument("--out", default="data/cme_kernel.parquet")
    b.add_argument("--min-valid", type=float, default=0.7, help="min share of valid obs in an event window")
    b.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    ev = event_table(a.arrivals, a.query)
    obs = load_obs(a.obs)
    tr = ev[ev["pred_last"] <= pd.Timestamp(a.train_end)]
    K, n = estimate_kernel(tr, obs, min_valid=a.min_valid, verbose=a.verbose)
    print(f"\nkernel from {n} training events (of {len(tr)}; {len(ev)} in total, {len(ev) - len(tr)} after {a.train_end})")
    print("hours since predicted arrival:   " + "".join(f"{t:>7d}" for t in (-48, -36, -24, -12, 0, 12, 24, 36, 48, 72)))
    print("mean speed anomaly (km/s):       " + "".join(f"{K.loc[t]:7.0f}" for t in (-48, -36, -24, -12, 0, 12, 24, 36, 48, 72)))
    t0s = pd.date_range(a.t0_start, a.t0_end, freq="6h")
    f = feature(ev, K, t0s)
    print(f"feature: {len(t0s)} issue times, {(f.abs().max(axis=1) > 5).mean() * 100:.1f}% of them have an active CME term, max {f.values.max():.0f} km/s")
    (f.to_csv if a.out.endswith(".csv") else f.to_parquet)(a.out)
    K.rename("anomaly_kms").to_csv(a.out.rsplit(".", 1)[0] + "_profile.csv")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
