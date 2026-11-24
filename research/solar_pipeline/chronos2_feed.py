"""Chronos-2 zero-shot forecast feed (speed, leads 1..72) for the stacker.  Run on a Kaggle GPU notebook.

    pip install -q "chronos-forecasting>=2.0" "pandas[pyarrow]"
    python chronos2_feed.py --ace swpc_ace_swepam_hourly.parquet --out chronos2_speed_features.parquet \
        --t0-start 2025-01-01 --latency 3 --context 1536 --enlil enlil_2025_2026_0_72h.parquet \
        --omni omni_features_data_2025_january_to_august_2026.parquet --eval

Output parquet (same layout as the Timestral feed): t0, lead_time, chronos_med, chronos_spread
  chronos_med    = median forecast
  chronos_spread = 0.9 quantile  (read as c2_speed_med / c2_speed_p90 by features.py after the patch below)

Causality: the context for issue time t0 ends at t0 - latency (the live ACE feed lags ~3 h); the *known-future*
covariates are only things available at t0:
  rec27  : ACE speed 27 days earlier (always known for the next 72 h)
  enlil  : latest ENLIL run with run_date <= t0 (optional; NaN where missing -- Chronos-2 accepts NaN covariates)
past-only covariates (context only): density, bz from OMNI (optional; quick-look values live!)

Quick test without a GPU/model:  python chronos2_feed.py --ace ... --mock --max-t0 20
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

H = pd.Timedelta(hours=1)
Q = [0.1, 0.5, 0.9]
REC = 27 * 24


# ----------------------------------------------------------------------------- data
def load_obs(path):
    a = pd.read_parquet(path)
    ts = pd.to_datetime(a["timestamp"], utc=True).dt.tz_localize(None)
    s = pd.Series(a["speed_km_s"].to_numpy(float), index=pd.DatetimeIndex(ts)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s[s.index == s.index.floor("h")]
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))


def load_omni(path):
    o = pd.read_parquet(path)
    st = pd.to_datetime(o["YEAR"].astype(str) + " " + o["DOY"].astype(str) + " " + o["HOUR"].astype(str),
                        format="%Y %j %H", utc=True).dt.tz_localize(None)
    o.index = pd.DatetimeIndex(st)
    o = o[~o.index.duplicated(keep="last")].sort_index()[["PROTON_DENSITY", "BZ_GSM"]]
    o = o.replace([9999.9, 9999.0, 999.9, 99.9], np.nan)
    o.columns = ["density", "bz"]
    return o


def load_enlil(path):
    e = pd.read_parquet(path)
    run = pd.to_datetime(e["run_date"], utc=True).dt.tz_localize(None)
    d = pd.DataFrame({"run": run.to_numpy(), "lead": e["forecast_horizon_hours"].round().astype(int).to_numpy(),
                      "v": (e["earth_v1"] / 1000.0).to_numpy()})
    d["valid"] = pd.DatetimeIndex(d["run"]) + pd.to_timedelta(d["lead"], unit="h")
    return d.groupby(["run", "valid"])["v"].mean(), np.sort(d["run"].unique())


def enlil_for(t0, times, ser, runs, max_age_h=72):
    """ENLIL speed at `times` from the latest run <= t0 (NaN if none / too old / hour not covered)."""
    k = np.searchsorted(runs, np.datetime64(t0), side="right") - 1
    if k < 0 or (t0 - pd.Timestamp(runs[k])) > pd.Timedelta(hours=max_age_h):
        return np.full(len(times), np.nan)
    r = runs[k]
    try:
        sub = ser.loc[r]
    except KeyError:
        return np.full(len(times), np.nan)
    return sub.reindex(times).to_numpy(float)


# ----------------------------------------------------------------------------- batches
def build_batch(t0s, obs, latency, ctx, omni=None, enlil=None):
    """One series per t0.  Returns context_df, future_df, and {id: (t0, first_future_time)}."""
    rows_c, rows_f, meta = [], [], {}
    for t0 in t0s:
        end = t0 - latency * H                                   # last context hour
        ctimes = pd.date_range(end - (ctx - 1) * H, end, freq="h")
        ftimes = pd.date_range(end + H, t0 + 72 * H, freq="h")   # includes the latency gap, so leads align
        sid = t0.strftime("%Y%m%dT%H")
        meta[sid] = (t0, ftimes[0])

        def rec(ts):
            return obs.reindex(ts - pd.Timedelta(hours=REC)).to_numpy(float)

        c = pd.DataFrame({"id": sid, "timestamp": ctimes, "target": obs.reindex(ctimes).to_numpy(float),
                          "rec27": rec(ctimes)})
        f = pd.DataFrame({"id": sid, "timestamp": ftimes, "rec27": rec(ftimes)})
        if enlil is not None:
            c["enlil"] = enlil_for(t0, ctimes, *enlil)
            f["enlil"] = enlil_for(t0, ftimes, *enlil)
        if omni is not None:
            for col in omni.columns:
                c[col] = omni[col].reindex(ctimes).to_numpy(float)
        rows_c.append(c)
        rows_f.append(f)
    return pd.concat(rows_c, ignore_index=True), pd.concat(rows_f, ignore_index=True), meta


def _qcol(df, q):
    for name in (str(q), q, f"{q:.1f}", f"q{q}"):
        if name in df.columns:
            return name
    raise KeyError(f"quantile {q} not in columns {list(df.columns)}")


def to_feed(pred, meta):
    """Chronos output (id, timestamp, quantile columns) -> rows (t0, lead_time, med, q90) with leads 1..72."""
    med_c = _qcol(pred, 0.5) if any(c in pred.columns for c in ("0.5", 0.5)) else "predictions"
    hi_c = _qcol(pred, 0.9)
    out = pd.DataFrame({"id": pred["id"].to_numpy(), "ts": pd.to_datetime(pred["timestamp"]).to_numpy(),
                        "med": pred[med_c].to_numpy(float), "hi": pred[hi_c].to_numpy(float)})
    t0 = out["id"].map(lambda i: meta[i][0])
    out["t0"] = t0.to_numpy()
    out["lead_time"] = ((out["ts"] - pd.DatetimeIndex(out["t0"])) / H).round().astype(int)
    out = out[(out["lead_time"] >= 1) & (out["lead_time"] <= 72)]
    return out.rename(columns={"med": "chronos_med", "hi": "chronos_spread"})[
        ["t0", "lead_time", "chronos_med", "chronos_spread"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- forecasters
class Chronos2:
    def __init__(self, name="amazon/chronos-2", device="cuda"):
        from chronos import Chronos2Pipeline
        self.p = Chronos2Pipeline.from_pretrained(name, device_map=device)

    def __call__(self, ctx, fut, pred_len):
        return self.p.predict_df(ctx, future_df=fut, prediction_length=pred_len, quantile_levels=Q,
                                 id_column="id", timestamp_column="timestamp", target="target")


class Mock:
    """No model: median = last context value (persistence); only for plumbing tests."""
    def __call__(self, ctx, fut, pred_len):
        rows = []
        for sid, g in ctx.groupby("id"):
            last = g["target"].ffill().iloc[-1]
            for t in fut.loc[fut["id"] == sid, "timestamp"].iloc[:pred_len]:
                rows.append({"id": sid, "timestamp": t, "0.1": last * 0.9, "0.5": last, "0.9": last * 1.1})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- run
def run(a):
    obs = load_obs(a.ace)
    omni = load_omni(a.omni) if a.omni else None
    enlil = load_enlil(a.enlil) if a.enlil else None
    t_end = pd.Timestamp(a.t0_end) if a.t0_end else obs.dropna().index.max() - 72 * H
    t0s = pd.date_range(pd.Timestamp(a.t0_start), t_end, freq=f"{a.step}h")
    if a.max_t0:
        t0s = t0s[: a.max_t0]
    print(f"{len(t0s)} issue times {t0s[0]} .. {t0s[-1]}, context {a.context} h, latency {a.latency} h, "
          f"enlil={'yes' if enlil else 'no'}, omni={'yes' if omni is not None else 'no'}")

    model = Mock() if a.mock else Chronos2(a.model, a.device)
    done = pd.read_parquet(a.out) if os.path.exists(a.out) and a.resume else None
    if done is not None:
        t0s = t0s[~t0s.isin(pd.DatetimeIndex(done["t0"]))]
        print(f"resuming: {len(t0s)} issue times left")
    parts = [done] if done is not None else []
    for i in range(0, len(t0s), a.batch):
        chunk = t0s[i:i + a.batch]
        ctx, fut, meta = build_batch(chunk, obs, a.latency, a.context, omni, enlil)
        # series that are all-NaN in the context cannot be forecast -> skip them
        ok = ctx.groupby("id")["target"].apply(lambda s: s.notna().sum() >= a.min_obs)
        keep = ok[ok].index
        ctx, fut = ctx[ctx["id"].isin(keep)], fut[fut["id"].isin(keep)]
        if ctx.empty:
            continue
        pred = model(ctx, fut, 72 + a.latency)
        parts.append(to_feed(pred, meta))
        pd.concat(parts, ignore_index=True).to_parquet(a.out)          # checkpoint each batch
        print(f"  {min(i + a.batch, len(t0s))}/{len(t0s)}", flush=True)
    feed = pd.concat(parts, ignore_index=True).drop_duplicates(["t0", "lead_time"], keep="last")
    feed.to_parquet(a.out)
    print(f"wrote {a.out}: {len(feed)} rows")
    return feed, obs


def evaluate(feed, obs, a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import score_tools as S
    df = feed.copy()
    df["target_time"] = pd.DatetimeIndex(df["t0"]) + pd.to_timedelta(df["lead_time"], unit="h")
    df["obs"] = obs.reindex(df["target_time"]).to_numpy(float)
    last = obs.ffill().reindex(pd.DatetimeIndex(df["t0"]) - a.latency * H).to_numpy(float)
    df["persistence"] = last
    r27 = obs.reindex(df["target_time"] - pd.Timedelta(hours=REC)).to_numpy(float)
    df["rec27"] = r27
    cols = {"persistence": "persistence", "chronos median": "chronos_med", "rec27": "rec27"}
    if a.compare_ts and os.path.exists(a.compare_ts):
        t = pd.read_parquet(a.compare_ts)
        t["t0"] = pd.to_datetime(t["t0"])
        t = t.groupby(["t0", "lead_time"])["timestral_med"].mean().rename("timestral")
        df = df.merge(t, left_on=["t0", "lead_time"], right_index=True, how="left")
        cols["timestral"] = "timestral"
    df["chronos+persist 50/50"] = 0.5 * df["chronos_med"] + 0.5 * df["persistence"]
    cols["chronos+persist 50/50"] = "chronos+persist 50/50"
    df = df.dropna(subset=["obs"])
    # NaN preds -> persistence, as in the official scoring
    print("\nS_final (lower is better), common rows:", len(df))
    for name, c in cols.items():
        d = df.copy()
        d["pred"] = d[c].fillna(d["persistence"])
        r = S.score(d, 'pred')
        print(f"  {name:24s} S={r['S_final']:.1f}  RMSE={r['RMSE']:.1f}  TRMSE={r['TRMSE']:.1f}  "
              f"S1-24={r['S_1_24']:.1f}  S25-48={r['S_25_48']:.1f}  S49-72={r['S_49_72']:.1f}")
    ev = df[(df["t0"] < "2026-01-14") | (df["t0"] > "2026-01-22")]
    print("\nwithout the Jan-14..22 2026 window:", len(ev))
    for name, c in cols.items():
        d = ev.copy()
        d["pred"] = d[c].fillna(d["persistence"])
        r = S.score(d, 'pred')
        print(f"  {name:24s} S={r['S_final']:.1f}  RMSE={r['RMSE']:.1f}  TRMSE={r['TRMSE']:.1f}  "
              f"S1-24={r['S_1_24']:.1f}  S25-48={r['S_25_48']:.1f}  S49-72={r['S_49_72']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ace", required=True)
    ap.add_argument("--out", default="chronos2_speed_features.parquet")
    ap.add_argument("--omni", default=None, help="adds density, bz as past-only covariates")
    ap.add_argument("--enlil", default=None, help="adds ENLIL speed as a known-future covariate")
    ap.add_argument("--t0-start", default="2025-01-01")
    ap.add_argument("--t0-end", default=None)
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--latency", type=int, default=3)
    ap.add_argument("--context", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-obs", type=int, default=200)
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-t0", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--compare-ts", default=None, help="Timestral feed parquet to compare against")
    a = ap.parse_args()
    feed, obs = run(a)
    if a.eval:
        evaluate(feed, obs, a)


if __name__ == "__main__":
    main()
