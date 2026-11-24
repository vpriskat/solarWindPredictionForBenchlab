"""Adds the joint ablation 'R -ts+tcn (train-only)' (run after patch_blend_ablation.py).   python patch_blend_ablation2.py run_experiments.py"""
import sys
p = sys.argv[1]; s = open(p).read()
if "ts+tcn" in s:
    raise SystemExit("already patched")
o1 = '    for g in ("ts", "tcn", "enlil", "rec"):\n        cols = [c for c in BLEND_COLS if c not in BLEND_OF_GROUP[g]]\n'
o2 = '    for g in ("ts", "tcn", "enlil", "rec"):\n        m, lo, hi = delta('
assert s.count(o1) == 1 and s.count(o2) == 1, "anchors not found (is patch_blend_ablation.py applied?)"
s = s.replace(o1, '    for g in ("ts", "tcn", "enlil", "rec", "ts+tcn"):\n        cols = [c for c in BLEND_COLS if not any(c in BLEND_OF_GROUP[x] for x in g.split("+"))]\n')
s = s.replace(o2, '    for g in ("ts", "tcn", "enlil", "rec", "ts+tcn"):\n        m, lo, hi = delta(')
open(p, "w").write(s); print("patched", p)
