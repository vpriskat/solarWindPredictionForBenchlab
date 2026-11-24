"""Do near-real-time (NRT) images give the same strips as the definitive ones?  (decides whether the ridge model trained on
definitive images may be applied to NRT images in production)

    python segment_ch.py --source nrt --start 2025-01-01 --end 2026-11-03 --hours 0000 --out data/ch_nrt
    python compare_strips.py --a data/ch_new --b data/ch_nrt

Matches images of the two folders by nominal time (<= 30 min apart) and prints, per feature group: correlation, mean difference,
and std of the difference relative to the feature's own std.  Correlation >~0.95 and small relative difference = interchangeable.
"""
import argparse

import numpy as np
import pandas as pd

from strip_test import load_strips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/ch_new", help="definitive strips")
    ap.add_argument("--b", default="data/ch_nrt", help="NRT strips")
    a = ap.parse_args()
    A, B = load_strips(a.a), load_strips(a.b)
    A, B = A.rename_axis("t").reset_index(), B.rename_axis("t").reset_index()
    m = pd.merge_asof(A.sort_values("t"), B.sort_values("t"), on="t", direction="nearest",
                      tolerance=pd.Timedelta(minutes=30), suffixes=("_a", "_b")).dropna()
    print(f"definitive {len(A)} images, NRT {len(B)} images, matched {len(m)}  ({m['t'].min()} .. {m['t'].max()})")
    cols = [c[:-2] for c in m.columns if c.endswith("_a")]
    rows = []
    for c in cols:
        x, y = m[c + "_a"].to_numpy(float), m[c + "_b"].to_numpy(float)
        sd = np.nanstd(x)
        rows.append((c, np.corrcoef(x, y)[0, 1] if sd > 0 and np.nanstd(y) > 0 else np.nan, np.nanmean(y - x),
                     np.nanstd(y - x) / sd if sd > 0 else np.nan))
    r = pd.DataFrame(rows, columns=["feature", "corr", "mean_diff", "sd_diff/sd"]).set_index("feature")
    g = r.groupby(r.index.str.replace(r"\d+$", "", regex=True)).agg({"corr": ["min", "median"], "mean_diff": "median", "sd_diff/sd": "median"})
    print("\nper feature family (corr min/median, median mean_diff, median sd(diff)/sd(feature)):")
    print(g.round(3).to_string())
    print(f"\noverall median corr {r['corr'].median():.3f}, worst {r['corr'].min():.3f} ({r['corr'].idxmin()})")
    chf = [c for c in cols if c.startswith("chf_eq")]
    if chf:
        print(f"mean CH fraction (equatorial bins): definitive {m[[c + '_a' for c in chf]].to_numpy().mean():.4f}, NRT {m[[c + '_b' for c in chf]].to_numpy().mean():.4f}")
    print("\nverdict:", "interchangeable" if r["corr"].median() > 0.95 and r["sd_diff/sd"].median() < 0.35 else
          "DIFFERENT enough that the ridge model should be retrained on NRT strips (or the threshold re-tuned)")


if __name__ == "__main__":
    main()
