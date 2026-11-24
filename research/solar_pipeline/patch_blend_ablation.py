"""Adds 'R -ts / -tcn / -enlil / -rec (train-only)' ablations + paired bootstrap to run_experiments.py (in place).
python patch_blend_ablation.py run_experiments.py"""
import sys
p = sys.argv[1]; s = open(p).read()
if "drop_names" in s:
    raise SystemExit("already patched")
a1 = '    res["R+clim27 (train-only)"] = Stage(variants["R + clim27"], backend, 0).fit(tr).predict_many(te, [0])[0]\n'
a2 = "+ cme_names + strip_names\n"
a3 = "    if has_c2:\n        print(\"\\nChronos-2 feed (paired"
for a in (a1, a2, a3):
    assert s.count(a) == 1, f"anchor not found exactly once: {a[:50]!r}"
s = s.replace(a1, a1 + '''    drop_names = []                                   # which blend input matters? refit R without it (train-only, like R (train-only))
    for g in ("ts", "tcn", "enlil", "rec"):
        cols = [c for c in BLEND_COLS if c not in BLEND_OF_GROUP[g]]
        nm = f"R -{g} (train-only)"
        res[nm] = Stage(mk([], "metric", "blend", False, blend_cols=cols), backend, 0).fit(tr).predict_many(te, [0])[0]
        drop_names.append(nm)
''')
s = s.replace(a2, "+ cme_names + strip_names + drop_names\n")
s = s.replace(a3, '''    print("\\nBlend ablation on R (train-only), paired block bootstrap, 95% CI  (positive = dropping the input HURTS, i.e. it helps)")
    for g in ("ts", "tcn", "enlil", "rec"):
        m, lo, hi = delta(f"R -{g} (train-only)", "R (train-only)")
        flag = "  <- CI excludes 0" if (lo > 0 or hi < 0) else ""
        print(f"  R without {g:<6} vs R (train-only)   {m:+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]{flag}")

''' + a3)
open(p, "w").write(s); print("patched", p)
