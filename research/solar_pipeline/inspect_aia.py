"""Print what an AIA synoptic file really contains (header keys, value ranges, radial profile) so the disk geometry can be checked.
   python inspect_aia.py --date 2024-03-01 --hour 0000        (downloads one file from JSOC)
   python inspect_aia.py --file path/to/AIA20240301_0000_0193.fits
Paste the whole output."""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2024-03-01")
    ap.add_argument("--hour", default="0000")
    ap.add_argument("--file", default=None)
    a = ap.parse_args()
    from astropy.io import fits
    if a.file:
        hd = fits.open(a.file)
    else:
        import pandas as pd
        from segment_ch import fetch
        f = fetch(pd.Timestamp(a.date), a.hour)
        if f is None:
            sys.exit("not found on JSOC")
        hd = fits.open(f)
    for i, h in enumerate(hd):
        shp = None if h.data is None else h.data.shape
        print(f"HDU {i}: {type(h).__name__} shape {shp} dtype {None if h.data is None else h.data.dtype}")
    h = next(x for x in hd if x.data is not None and x.data.ndim == 2)
    hdr = h.header
    keys = ["NAXIS1", "NAXIS2", "BITPIX", "BSCALE", "BZERO", "CDELT1", "CDELT2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "R_SUN", "RSUN_OBS",
            "RSUN_REF", "CRLT_OBS", "HGLT_OBS", "CRLN_OBS", "DSUN_OBS", "T_OBS", "DATE-OBS", "T_REC", "EXPTIME", "WAVELNTH", "LVL_NUM",
            "QUALITY", "INSTRUME", "ORIGIN", "PIXLUNIT", "BUNIT", "CROTA2", "SOLAR_P0", "SOLAR_B0"]
    print("\nheader:")
    for k in keys:
        if k in hdr:
            print(f"  {k:<9} {hdr[k]}")
    d = np.asarray(h.data, float)
    print(f"\ndata: shape {d.shape}, NaN {np.isnan(d).sum()}, min {np.nanmin(d):.3g}, p1 {np.nanpercentile(d, 1):.3g}, "
          f"median {np.nanmedian(d):.3g}, p99 {np.nanpercentile(d, 99):.3g}, max {np.nanmax(d):.3g}")
    n = d.shape[0]
    k = n // 20
    print("corner medians:", [round(float(np.nanmedian(c)), 3) for c in (d[:k, :k], d[:k, -k:], d[-k:, :k], d[-k:, -k:])])
    yy, xx = np.indices(d.shape)
    r = np.hypot(xx - (n / 2 - 0.5), yy - (n / 2 - 0.5)).astype(int).ravel()
    prof = np.bincount(r, np.log1p(np.clip(np.nan_to_num(d), 0, None)).ravel()) / np.maximum(np.bincount(r), 1)
    step = max(n // 40, 1)
    print("\nradial mean of log1p(intensity) around the image centre (radius px : value):")
    print("  " + "  ".join(f"{i}:{prof[i]:.2f}" for i in range(0, min(len(prof), int(0.75 * n)), step)))
    from segment_ch import geometry
    cx, cy, R, b0, t = geometry(d, dict(hdr))
    print(f"\nheader-based disk (R = RSUN_OBS / CDELT1): centre ({cx:.1f}, {cy:.1f}), radius {R:.1f} px, B0 {b0:.2f}, T_OBS {t}")

if __name__ == "__main__":
    main()
