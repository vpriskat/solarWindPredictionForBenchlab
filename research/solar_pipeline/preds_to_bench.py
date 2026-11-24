"""Convert the long test-prediction table written by run_experiments.py (--out-preds *.pkl) into the benchmark folder contract.

  python preds_to_bench.py --pred results/ladder_test_preds.pkl --list                      # show the model columns
  python preds_to_bench.py --pred results/ladder_test_preds.pkl --col "R (train-only) + strip x0.5" \
        --name mine --out results/bench_runs                                              # -> results/bench_runs/mine/<window>/<stamp>.json
  python bench_score.py --kit <kit>/scoring_demo --live --cache-dir data/ace_monthly --runs results/bench_runs/mine ...

Input table: columns t0, lead_time (1..72), and one column per model (forecast km/s for t0+lead_time h); other columns (obs, persistence, ...) are ignored.
Each window folder gets only the cycles of its dates (00/06/12/18 UTC).  Missing/NaN values are written as null (the platform scores them as persistence).
Pass several --col to write several models at once (--name then gives the folder prefix: <name>__<column>).
"""
import argparse, json, pickle, re
from pathlib import Path
import numpy as np
import pandas as pd

WINDOWS = {"jan-apr": ("2026-01-01", "2026-04-30"), "mar-jun": ("2026-03-01", "2026-06-30"),
           "may-aug": ("2026-05-01", "2026-08-31"), "jan-aug": ("2026-01-01", "2026-08-31")}
META = {"t0", "lead_time", "obs", "persistence"}


def load_table(path):
    obj = pickle.load(open(path, "rb")) if path.endswith((".pkl", ".pickle")) else (
        pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path))
    if isinstance(obj, dict):
        frames = [v for v in obj.values() if isinstance(v, pd.DataFrame) and {"t0", "lead_time"} <= set(v.columns)]
        if not frames:
            raise SystemExit("no DataFrame with t0/lead_time columns inside that pickle")
        obj = max(frames, key=len)
    obj = obj.copy()
    obj["t0"] = pd.to_datetime(obj["t0"])
    if obj["t0"].dt.tz is not None:
        obj["t0"] = obj["t0"].dt.tz_convert("UTC").dt.tz_localize(None)
    return obj


def slug(s):
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", s).strip("_")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True); ap.add_argument("--col", action="append", default=[])
    ap.add_argument("--name", default=None, help="model folder name (default: the column name)")
    ap.add_argument("--out", default="results/bench_runs"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--windows", nargs="*", default=[f"{k}:{a}:{b}" for k, (a, b) in WINDOWS.items()], help="folder:start:end")
    a = ap.parse_args()
    d = load_table(a.pred)
    cols = [c for c in d.columns if c not in META and pd.api.types.is_numeric_dtype(d[c])]
    t = d.t0.drop_duplicates().sort_values()
    print(f"{len(d):,} rows, {t.nunique()} cycles {t.min()} .. {t.max()}; lead_time {d.lead_time.min()}..{d.lead_time.max()}")
    off = t[~t.dt.hour.isin([0, 6, 12, 18]) | (t.dt.minute != 0)]
    if len(off):
        print(f"  WARNING: {len(off)} t0 are not on the 00/06/12/18 UTC grid (they are dropped), e.g. {off.iloc[0]}")
    if a.list or not a.col:
        print("model columns:"); [print("  ", c) for c in cols]; return
    for c in a.col:
        if c not in d.columns:
            raise SystemExit(f"column {c!r} not found; use --list")
    for c in a.col:
        name = (a.name if len(a.col) == 1 and a.name else f"{a.name + '__' if a.name else ''}{slug(c)}")
        w = d.pivot_table(index="t0", columns="lead_time", values=c, aggfunc="first").reindex(columns=range(1, 73))
        w = w[w.index.hour.isin([0, 6, 12, 18]) & (w.index.minute == 0)]
        root = Path(a.out) / name
        for spec in a.windows:
            lab, s, e = spec.split(":")
            sub = w[(w.index >= pd.Timestamp(s)) & (w.index < pd.Timestamp(e) + pd.Timedelta(days=1))]
            full = pd.date_range(s, pd.Timestamp(e) + pd.Timedelta(hours=18), freq="6h")
            full = full[full < pd.Timestamp(e) + pd.Timedelta(days=1)]
            (root / lab).mkdir(parents=True, exist_ok=True)
            for f in (root / lab).glob("*.json"):
                f.unlink()
            for t0, row in sub.iterrows():
                v = [None if not np.isfinite(x) else float(x) for x in row.values]
                (root / lab / f"{t0:%Y%m%dT%H%M%SZ}.json").write_text(json.dumps(
                    {"t0": f"{t0:%Y-%m-%dT%H:%M:%SZ}", "valid_from": f"{t0 + pd.Timedelta(hours=1):%Y-%m-%dT%H:%M:%SZ}", "forecast_speed_kms": v}))
            nnull = int(sub.isna().sum().sum())
            print(f"  {name}/{lab}: {len(sub)}/{len(full)} cycles written" + (f", {nnull} null values" if nnull else "")
                  + (f"  <-- {len(full) - len(sub)} cycles MISSING (scored as persistence)" if len(sub) < len(full) else ""))
        (root / "manifest.json").write_text(json.dumps({"name": name, "source": str(a.pred), "column": c}))
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
