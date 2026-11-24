"""Export the final blend (persistence + rec27_eff + enlil, per-lead-group ridge, TRAIN-ONLY) to JSON for the container,
and check that it reproduces the ladder's column 'R -ts+tcn (train-only)' on the test rows.
Run in hss_prediction (needs blend.py, score_tools.py, run_experiments.py next to it):
  python export_blend.py --cache cache_lat3_enl12.pkl --pred results/ladder_lat3_enl12.pkl --out final_blend.json
"""
import argparse, json, pickle
import numpy as np, pandas as pd
from blend import Blend
import run_experiments as RE

ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); ap.add_argument("--pred", required=True)
ap.add_argument("--out", default="final_blend.json"); ap.add_argument("--cols", nargs="*", default=["persistence", "rec27_eff", "enlil"])
ap.add_argument("--train-end", default="2025-10-01"); ap.add_argument("--val-end", default="2026-01-01")
a = ap.parse_args()
df = pickle.load(open(a.cache, "rb"))["df"]
tr, va, te = RE.split(df, a.train_end, a.val_end)
for d in (tr, va, te):
    d["ref"] = d["ref"].fillna(d["persistence"]).fillna(tr["obs"].mean())
    d["rec27_eff"] = d["rec27"].fillna(d["rec27_mean12"]) if "rec27_mean12" in d else d["rec27"]
b = Blend(a.cols).fit_two_pass(tr)
pred = b.predict(te)
ref = pickle.load(open(a.pred, "rb")); ref = ref if isinstance(ref, pd.DataFrame) else max([v for v in ref.values() if isinstance(v, pd.DataFrame)], key=len)
m = te[["t0", "lead_time"]].assign(mine=pred).merge(ref[["t0", "lead_time", "R -ts+tcn (train-only)"]], on=["t0", "lead_time"])
print(f"reproduction check on {len(m):,} test rows: max |diff| = {np.abs(m.mine - m['R -ts+tcn (train-only)']).max():.6f} km/s")
out = dict(cols=a.cols, width=b.width, lam=b.lam, fallback=b.fallback, train_end=a.train_end,
           params={str(k): dict(const=c, beta=list(map(float, beta))) for k, (c, beta) in b.params.items()})
json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out); print(b.weights_table().round(3).to_string())
