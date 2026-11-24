"""Build the candidate final models from a ladder pickle:  R without Timestral and TCN (train-only) + lambda * strip_adj.
python make_final_cols.py --pred results/ladder_lat3_enl12.pkl --out results/ladder_final_enl12.pkl
then:  python preds_to_bench.py --pred results/ladder_final_enl12.pkl --out results/bench_final --col "FINAL strip x0.5" --col "FINAL strip x0.75" --col "FINAL strip x1" --col "FINAL no strip"
"""
import argparse, pickle
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--base", default="R -ts+tcn (train-only)"); ap.add_argument("--lams", type=float, nargs="*", default=[0.5, 0.75, 1.0])
a = ap.parse_args()
d = pickle.load(open(a.pred, "rb"))
d = d if isinstance(d, pd.DataFrame) else max([v for v in d.values() if isinstance(v, pd.DataFrame)], key=len)
for c in (a.base, "strip_adj"):
    assert c in d.columns, f"{c!r} not in the pickle (run the ladder with --strip-path and the ablation patches)"
d = d.copy()
d["FINAL no strip"] = d[a.base]
for lam in a.lams:
    d[f"FINAL strip x{lam:g}"] = d[a.base] + lam * d["strip_adj"]
pickle.dump(d, open(a.out, "wb")); print("saved", a.out, [c for c in d.columns if c.startswith("FINAL")])
