"""Coronal-hole segmentation + longitude strips from JSOC AIA 193 synoptic images  (replaces the TEST_MODE script).

Fixes compared with the old script:
  * disk radius and centre come from the FITS header (RSUN_OBS / CDELT1, CRPIX), not a fixed 450 / 420 px.  (R_SUN in the rebinned
    1k files is still in full-resolution pixels: 1612 instead of ~404.)
  * NO k-means on raw intensity.  The 193 disk is not uniform (mean ln I dips at mid-radius, brightens at the limb), so
    'darkest of 3 clusters' labelled ~40% of the disk as CH.  Here ln I is divided by its median at the same radius and CH =
    darker than CH_RATIO (0.55) of that quiet-Sun level, then cleaned.  The ratio is a choice: use --png to check masks by eye.
  * every image is stamped with its own header time (T_OBS / DATE-OBS), not the midnight of the day it was looked up for
    (the old loop fell back to 12:00, 06:00 or 18:00 images but stamped them 00:00 = up to 12 h of look-ahead).
  * all 4 synoptic images per day are used (00, 06, 12, 18 UTC) instead of the first one found.
  * no TEST_MODE; resumable per month; --start/--end on the command line.
  * besides a 256x256 uint8 CH mask it stores what a model can actually use: for 16 Stonyhurst longitude bins (-80..80 deg)
    x 2 latitude bands the area-weighted CH fraction and the normalised mean log intensity, plus B0 and the CH fraction per
    hemisphere.  (A CH seen at west longitude W reaches Earth after roughly W / 13.2 deg-per-day, so longitude carries timing.)

  python segment_ch.py --start 2010-06-01 --end 2024-10-01 --hours 0000 1200 --out data/ch_new
  python segment_ch.py --selftest
"""
import argparse
import os
import warnings
import pickle
import re
import time
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from skimage.morphology import binary_closing, disk, remove_small_objects
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, uniform_filter1d

LON_EDGES = np.arange(-80, 81, 10)             # 16 longitude bins
LAT_BANDS = {"eq": (-30, 30), "hi": (30, 60)}   # |lat| <= 30 and 30 < |lat| <= 60 (both hemispheres together)
FIT_FRAC, MASK_FRAC = 0.95, 0.98                # clustering disk / mask cut-off, in solar radii
URL = "http://jsoc.stanford.edu/data/aia/synoptic/{y}/{m}/{d}/H{h}/AIA{y}{m}{d}_{h}_0193.fits"
NRT_DIR = "https://jsoc1.stanford.edu/data/aia/synoptic/nrt/{y}/{m}/{d}/H{hh}00/"      # near-real-time stream, ~15-20 min latency
SOURCE = "definitive"                                                                  # or "nrt" (--source)
NRT_MAX_OFFSET_MIN = 10                                                                # accept an NRT image within 10 min of the slot


# ------------------------------------------------------------------------------------------------ geometry
def geometry(data, hdr):
    """(cx, cy, R_px, B0_deg, t_obs) from the header.  R = RSUN_OBS / CDELT1 (arcsec per pixel of THIS image).
    Do not use R_SUN: in the rebinned 1k synoptic files it is still in full-resolution pixels (1612 instead of ~404)."""
    ny, nx = data.shape
    scale = nx / float(hdr.get("NAXIS1", nx))                    # 0.25 if a 4k image was block-averaged to 1k
    cdelt = float(hdr.get("CDELT1", 2.4)) / scale                # arcsec per pixel of the image we actually have
    R = float(hdr.get("RSUN_OBS", 960.0)) / cdelt
    cx = (float(hdr["CRPIX1"]) - 1) * scale if "CRPIX1" in hdr else nx / 2 - 0.5
    cy = (float(hdr["CRPIX2"]) - 1) * scale if "CRPIX2" in hdr else ny / 2 - 0.5
    if not (0.30 * nx <= R <= 0.47 * nx):
        raise ValueError(f"implausible solar radius {R:.0f}px on a {nx}px image (RSUN_OBS={hdr.get('RSUN_OBS')}, "
                         f"CDELT1={hdr.get('CDELT1')}, NAXIS1={hdr.get('NAXIS1')}); skipped")
    t = hdr.get("T_OBS") or hdr.get("DATE-OBS") or hdr.get("DATE_OBS")
    t = pd.Timestamp(str(t).replace("Z", "")) if t else pd.NaT
    if "CRLT_OBS" in hdr:
        b0 = float(hdr["CRLT_OBS"])
    elif "HGLT_OBS" in hdr:
        b0 = float(hdr["HGLT_OBS"])
    else:
        doy = t.dayofyear if pd.notna(t) else 1
        b0 = 7.25 * np.sin(2 * np.pi * (doy - 157) / 365.25)
    return cx, cy, R, b0, t


def disk_coords(shape, cx, cy, R, b0_deg):
    """Per pixel: r/R, heliographic Stonyhurst lon/lat (deg) for on-disk pixels, z = cos(angle from disk centre)."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    x, y = (xx - cx) / R, (cy - yy) / R                           # x = west, y = north (north-up image)
    rr = np.hypot(x, y)
    z = np.sqrt(np.clip(1 - rr ** 2, 0, None))
    b0 = np.deg2rad(b0_deg)
    lat = np.arcsin(np.clip(y * np.cos(b0) + z * np.sin(b0), -1, 1))
    lon = np.arctan2(x, z * np.cos(b0) - y * np.sin(b0))
    return rr, np.rad2deg(lon), np.rad2deg(lat), z


# ------------------------------------------------------------------------------------------- segmentation
CH_RATIO = 0.55                                    # CH = pixels darker than this fraction of the quiet-Sun level at the same radius


def clv_correct(data, rr):
    """ln(intensity) minus its median over the ring of equal radius (centre-to-limb variation removed).  The AIA 193 disk is NOT
    uniform: the mean log intensity dips at mid-radius and brightens at the limb, so an absolute-intensity threshold or a
    k-means on raw intensities labels whole annuli as 'coronal hole'."""
    d = np.log(np.clip(np.nan_to_num(data, nan=0.0), 1.0, None))
    sel = rr <= MASK_FRAC
    ri = (rr[sel] * 1000).astype(int)                              # ring index: 0.001 R wide (~0.4 px)
    prof = pd.Series(d[sel]).groupby(ri).median()
    prof = prof.reindex(range(int(ri.max()) + 1)).interpolate(limit_direction="both")
    prof = pd.Series(uniform_filter1d(prof.to_numpy(), 9), index=prof.index)
    corr = np.zeros_like(d)
    corr[sel] = d[sel] - prof.reindex(ri).to_numpy()
    return corr


def segment(data, rr, ratio=None):
    """CH mask = smoothed CLV-corrected ln-intensity below ln(ratio), cleaned.  Returns (mask, corrected ln ratio map)."""
    ratio = CH_RATIO if ratio is None else ratio
    corr = gaussian_filter(clv_correct(data, rr), 1.5)
    m = (corr < np.log(ratio)) & (rr <= MASK_FRAC)
    m = remove_small_objects(m, min_size=40)
    m = binary_closing(m, disk(2)) & (rr <= MASK_FRAC)
    return m, corr


def strips(mask, logI, rr, lon, lat, z):
    """Area-weighted (1/z) CH fraction and mean CLV-corrected ln intensity per longitude bin x latitude band."""
    on = rr <= FIT_FRAC
    zi = logI                                                      # ln(intensity / quiet Sun at that radius)
    out = {}
    w = 1.0 / np.clip(z, 0.25, None)
    for name, (lo, hi) in LAT_BANDS.items():
        sel = on & ((lat >= lo) & (lat <= hi) if name == "eq" else (np.abs(lat) > 30) & (np.abs(lat) <= 60))
        b = np.digitize(lon, LON_EDGES) - 1
        for i in range(len(LON_EDGES) - 1):
            s = sel & (b == i)
            ws = w[s].sum()
            out[f"chf_{name}_{i:02d}"] = float((w[s] * mask[s]).sum() / ws) if ws > 0 else np.nan
            out[f"int_{name}_{i:02d}"] = float((w[s] * zi[s]).sum() / ws) if ws > 0 else np.nan
    n, s_ = on & (lat > 0), on & (lat <= 0)
    out["chf_north"] = float((w[n] * mask[n]).sum() / w[n].sum())
    out["chf_south"] = float((w[s_] * mask[s_]).sum() / w[s_].sum())
    return out


PREVIEW = []                                       # filled with output paths when --png N is given


def preview(data, mask, corr, rr, row, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(np.log(np.clip(data, 1, None)), origin="lower", cmap="gray")
    ax[0].set_title(f"AIA 193 ln(I)   {row['t_obs']}")
    ax[1].imshow(corr, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax[1].set_title("ln(I / quiet Sun at this radius)")
    ax[2].imshow(np.log(np.clip(data, 1, None)), origin="lower", cmap="gray")
    ax[2].contour(mask.astype(float), [0.5], colors="cyan", linewidths=0.8)
    ax[2].set_title(f"CH mask (ratio<{CH_RATIO}), {100 * row['ch_frac_disk']:.1f}% of the disk")
    for a_ in ax:
        a_.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def safe_header(header):
    """Header -> dict, skipping cards astropy cannot parse (older AIA files have a malformed OSCNMEAN card;
    dict(header) raises VerifyError on it).  We only need a handful of geometry keys, so drop what fails."""
    out = {}
    for card in header.cards:
        try:
            out[card.keyword] = card.value
        except Exception:
            pass
    return out


def process_image(data, hdr):
    f = data.shape[0] // 1024
    if f > 1 and data.shape[0] % 1024 == 0:                        # 4096 -> 1024 (mean of 4x4): 16x faster, same science
        data = data.reshape(1024, f, 1024, f).mean(axis=(1, 3))
    cx, cy, R, b0, t = geometry(data, hdr)
    rr, lon, lat, z = disk_coords(data.shape, cx, cy, R, b0)
    rr_n = rr
    mask, logI = segment(data, rr_n)
    row = dict(t_obs=t, b0=b0, R_px=R, ch_frac_disk=float(mask[rr <= FIT_FRAC].mean()))
    row.update(strips(mask, logI, rr_n, lon, lat, z))
    if PREVIEW:
        preview(data, mask, logI, rr, row, PREVIEW.pop(0))
    small = resize(mask.astype(float), (256, 256), anti_aliasing=True, preserve_range=True)
    return row, (small * 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------- download
def nrt_pick(listing, day, hour, max_offset_min=NRT_MAX_OFFSET_MIN):
    """Name of the NRT file (AIAyyyymmdd_HHMMSS_0193.fits) closest to the slot day+hour, or None."""
    slot = pd.Timestamp(day).normalize() + pd.Timedelta(hours=int(hour[:2]), minutes=int(hour[2:]))
    best, best_d = None, None
    for m in re.finditer(r"AIA(\d{8})_(\d{6})_0193\.fits", listing):
        t = pd.Timestamp(f"{m.group(1)} {m.group(2)}")
        d = abs((t - slot).total_seconds()) / 60
        if best_d is None or d < best_d:
            best, best_d = m.group(0), d
    return best if best is not None and best_d <= max_offset_min else None


def fetch_nrt(day, hour, session=None):
    import requests
    base = NRT_DIR.format(y=f"{day:%Y}", m=f"{day:%m}", d=f"{day:%d}", hh=hour[:2])
    get = (session or requests).get
    r = get(base, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    if r.status_code != 200:
        return None
    name = nrt_pick(r.text, day, hour)
    if name is None:
        return None
    r = get(base + name, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    return BytesIO(r.content) if r.status_code == 200 and b"SIMPLE" in r.content[:100] else None


def fetch(day, hour, session=None):
    import requests
    if SOURCE == "nrt":
        return fetch_nrt(day, hour, session)
    url = URL.format(y=f"{day:%Y}", m=f"{day:%m}", d=f"{day:%d}", h=hour)
    r = (session or requests).get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    if r.status_code == 200 and b"SIMPLE" in r.content[:100]:
        return BytesIO(r.content)
    return None


def run(start, end, hours, out, sleep=0.5, n_png=0, every=1):
    from astropy.io import fits
    import requests
    os.makedirs(out, exist_ok=True)
    PREVIEW[:] = [f"{out}/preview_{i:02d}.png" for i in range(n_png)]
    sess = requests.Session()
    month = pd.Timestamp(start).replace(day=1)
    while month < pd.Timestamp(end):
        nxt = month + pd.offsets.MonthBegin(1)
        f_csv = f"{out}/strips_{month:%Y-%m}.csv"
        if os.path.exists(f_csv):
            print("skip", month.strftime("%Y-%m"))
            month = nxt
            continue
        rows, masks = [], []
        day = max(month, pd.Timestamp(start))
        while day < min(nxt, pd.Timestamp(end)):
            for h in hours:
                try:
                    f = fetch(day, h, sess)
                    if f is None:
                        continue
                    with fits.open(f) as hd:
                        hdu = next(x for x in hd if x.data is not None and x.data.ndim == 2)
                        row, small = process_image(np.asarray(hdu.data, float), safe_header(hdu.header))
                    row["requested"] = day + pd.Timedelta(hours=int(h[:2]))
                    rows.append(row)
                    masks.append(small)
                except Exception as ex:                                # keep going; the month file records what worked
                    print(f"  {day:%Y-%m-%d} {h}: {ex!r}")
                time.sleep(sleep)
            day += pd.Timedelta(days=every)
        if rows:
            pd.DataFrame(rows).to_csv(f_csv, index=False)
            pickle.dump(np.stack(masks), open(f"{out}/masks256_{month:%Y-%m}.pickle", "wb"))
        print(f"{month:%Y-%m}: {len(rows)} images")
        month = nxt


# ---------------------------------------------------------------------------------------------- self test
def selftest():
    n, R = 1024, 400.0
    hdr = dict(R_SUN=R, CRPIX1=512.5, CRPIX2=512.5, CDELT1=2.4, T_OBS="2024-03-01T12:00:00.0Z", CRLT_OBS=-6.0)
    hdr = dict(RSUN_OBS=960.0, CDELT1=2.4, NAXIS1=1024, R_SUN=1612.0, CRPIX1=512.5, CRPIX2=512.5,
               T_OBS="2024-03-01T12:00:00.0Z", CRLT_OBS=-6.0)                  # R_SUN is stale (full-res px) like in your files
    cx_, cy_, R_, b0, t = geometry(np.zeros((n, n)), hdr)
    assert abs(R_ - 400) < 1e-9 and abs(cx_ - 511.5) < 1e-9 and b0 == -6.0 and t == pd.Timestamp("2024-03-01 12:00")
    h4 = dict(hdr, NAXIS1=4096, CDELT1=0.6, CRPIX1=2048.5, CRPIX2=2048.5)                  # 4k image block-averaged to 1k
    assert abs(geometry(np.zeros((n, n)), h4)[2] - 400) < 1e-9
    try:
        geometry(np.zeros((n, n)), dict(hdr, RSUN_OBS=3000.0))
        raise SystemExit("implausible radius should be rejected")
    except ValueError:
        pass
    rr, lon, lat, z = disk_coords((n, n), cx_, cy_, R_, b0)
    # centre pixel sits at lon 0, lat = B0 (as seen from the observer): lat = -6
    assert abs(lon[512, 512]) < 1.0 and abs(lat[512, 512] - (-6.0)) < 1.0
    # a point on the west limb along the equator-ish: lon ~ +90
    assert 80 < lon[512 - 6, 512 + 395] < 95
    rng = np.random.default_rng(1)
    # the measured 2024-03-01 radial profile of ln(I): 6.0 at the centre, a dip to ~5.3 at mid-radius, 6.15 at the limb
    rp = np.array([0, 100, 200, 300, 350, 400, 425, 450, 500, 600, 700]) / 400.0
    lp = np.array([6.0, 5.95, 5.75, 5.40, 5.45, 6.1, 6.1, 5.3, 4.0, 2.2, 0.2])
    base = np.interp(rr, rp, lp)
    img0 = np.exp(base + rng.normal(0, 0.12, (n, n)))                                        # granulation-like noise
    m0, c0 = segment(img0, rr)
    frac0 = m0[rr <= FIT_FRAC].mean()
    assert frac0 < 0.02, f"a CH-free disk with centre-to-limb variation must give ~0 CH, got {frac0:.3f}"
    ch = (np.hypot(lon - 35, lat - 0) < 18) & (rr <= 1.0)                                      # CH centred at lon +35, lat 0
    img = img0.copy()
    img[ch] *= 0.3
    mask, logI = segment(img, rr)
    assert mask[rr > MASK_FRAC].sum() == 0, "no CH pixels outside the limb"
    iou = (mask & ch & (rr <= FIT_FRAC)).sum() / ((mask | ch) & (rr <= FIT_FRAC)).sum()
    assert iou > 0.75, iou
    row = strips(mask, logI, rr, lon, lat, z)
    chf = np.array([row[f"chf_eq_{i:02d}"] for i in range(16)])
    centre = LON_EDGES[:-1] + 5
    assert abs(centre[np.nanargmax(chf)] - 35) <= 10, (centre[np.nanargmax(chf)], chf.round(2))
    assert chf[np.argmin(abs(centre + 45))] < 0.1
    row2, small = process_image(img, hdr)
    assert small.shape == (256, 256) and small.dtype == np.uint8 and pd.notna(row2["t_obs"])
    print(f"ok  segmentation: CH-free disk {100 * frac0:.2f}% CH; patch IoU {iou:.2f}, peak CH bin at lon {centre[np.nanargmax(chf)]:+.0f} (truth +35)")


def main():
    global CH_RATIO, SOURCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-06-01")
    ap.add_argument("--end", default="2024-10-01")
    ap.add_argument("--hours", nargs="+", default=["0000", "1200"], help="synoptic hours to use (0000 0600 1200 1800)")
    ap.add_argument("--out", default="data/ch_new")
    ap.add_argument("--png", type=int, default=0, help="save preview pictures of the first N images (check the masks by eye!)")
    ap.add_argument("--ch-ratio", type=float, default=CH_RATIO, help="CH = darker than this fraction of the quiet-Sun level")
    ap.add_argument("--source", choices=["definitive", "nrt"], default="definitive",
                    help="definitive synoptic archive (appears ~7 days after observation) or the near-real-time stream (~15-20 min)")
    ap.add_argument("--every", type=int, default=1, help="use every Nth day (sampling for quick comparisons)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    CH_RATIO = a.ch_ratio
    SOURCE = a.source
    selftest() if a.selftest else run(a.start, a.end, a.hours, a.out, n_png=a.png, every=a.every)


if __name__ == "__main__":
    main()
