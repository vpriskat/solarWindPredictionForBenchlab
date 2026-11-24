"""Probe every candidate source of FRESH solar EUV images from YOUR machine and report how old the newest image is.

    python probe_aia_sources.py                 # all sources
    python probe_aia_sources.py nrt sdo         # a subset: nrt | mostrecent | sdo | helio | suvi

Sources
  nrt         JSOC near-real-time synoptic FITS (1k x 1k, DN, ~15-20 min latency per the JSOC README): crawls the directory
              tree to the NEWEST and the OLDEST day (= how long NRT data is kept), lists the newest files, prints the header
              keys that segment_ch.py needs if astropy is installed
  mostrecent  JSOC 'mostrecent' directory (fixed file names, newest image per filter)
  sdo         SDO browse JPEGs + the AIA 193 RSS feed (quick-look, 8-bit colour)
  helio       Helioviewer API getClosestImage for AIA 193 (JPEG2000 archive, history back to 2010)
  suvi        NOAA SWPC SUVI 195 A images (GOES; different instrument)
Everything is wrapped in try/except: a failing source prints why and the others continue.  Send me the whole output.
"""
import re
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

import requests

NOW = datetime.now(timezone.utc)
UA = {"User-Agent": "solar-wind-probe/1.0"}


def get(url, **kw):
    return requests.get(url, timeout=40, headers=UA, **kw)


def links(url):
    r = get(url)
    r.raise_for_status()
    hrefs = re.findall(r'href="([^"?#]+)"', r.text)
    return [h for h in hrefs if not h.startswith(("/", "http", "..", "?"))], r


def age(dt):
    return (NOW - dt).total_seconds() / 3600


def crawl(base, pick=max, depth=6):
    """Descend through sub-directories always taking the max (newest) or min (oldest) name; returns the path chain."""
    url, chain = base, []
    for _ in range(depth):
        hs, _r = links(url)
        dirs = sorted(h for h in hs if h.endswith("/"))
        if not dirs:
            return url, hs
        d = pick(dirs)
        chain.append(d)
        url = urljoin(url, d)
    return url, links(url)[0]


def head_time(url):
    r = requests.head(url, timeout=30, headers=UA, allow_redirects=True)
    lm = r.headers.get("Last-Modified")
    return r.status_code, (parsedate_to_datetime(lm).astimezone(timezone.utc) if lm else None), r.headers.get("Content-Length")


def probe_nrt():
    base = "https://jsoc1.stanford.edu/data/aia/synoptic/nrt/"
    top, r = links(base)
    print(f"[nrt] {base}: {len(top)} entries, first/last: {top[:3]} .. {top[-3:]}")
    for label, pick in (("NEWEST", max), ("OLDEST", min)):
        url, files = crawl(base, pick)
        print(f"[nrt] {label} leaf directory: {url}")
        f193 = [f for f in files if "193" in f]
        print(f"      {len(files)} files, 193 files: {f193[:3]} .. {f193[-3:]}")
        if label == "NEWEST" and f193:
            code, lm, size = head_time(urljoin(url, f193[-1]))
            print(f"      newest 193 file HTTP {code}, size {size}, Last-Modified {lm}  (age {age(lm):.1f} h)" if lm else f"      HTTP {code}")
            try:
                from astropy.io import fits
                import io
                data = get(urljoin(url, f193[-1])).content
                with fits.open(io.BytesIO(data)) as hd:
                    h = next(x for x in hd if x.data is not None and x.data.ndim == 2)
                    hdr = {c.keyword: c.value for c in h.header.cards if c.keyword in
                           ("T_OBS", "DATE-OBS", "RSUN_OBS", "CDELT1", "CRPIX1", "CRPIX2", "CRLT_OBS", "HGLT_OBS", "NAXIS1", "WAVELNTH", "LVL_NUM", "QUALITY")}
                    print(f"      header: {hdr}; image {h.data.shape}, median {float(__import__('numpy').nanmedian(h.data)):.1f} DN")
            except Exception as ex:
                print(f"      (header not read: {repr(ex)[:80]})")
        if label == "OLDEST":
            print(f"      => NRT retention: oldest directory above; compare with today ({NOW:%Y-%m-%d}) to see how many days are kept")


def probe_mostrecent():
    base = "https://jsoc1.stanford.edu/data/aia/synoptic/mostrecent/"
    hs, r = links(base)
    print(f"[mostrecent] {base}: {hs[:12]}")
    for h in [x for x in hs if "193" in x][:3] + [x for x in hs if "time" in x.lower()][:2]:
        code, lm, size = head_time(urljoin(base, h))
        print(f"      {h}: HTTP {code}, size {size}, Last-Modified {lm}" + (f"  (age {age(lm):.1f} h)" if lm else ""))
        if size and size.isdigit() and int(size) < 500:
            print("      content:", get(urljoin(base, h)).text.strip()[:200])


def probe_sdo():
    d = NOW
    url = f"https://sdo.gsfc.nasa.gov/assets/img/browse/{d:%Y}/{d:%m}/{d:%d}/"
    hs, _ = links(url)
    jp = sorted(h for h in hs if "_0193" in h and h.lower().endswith((".jpg", ".png")))
    print(f"[sdo] {url}: {len(hs)} entries, {len(jp)} AIA193 images; last: {jp[-3:]}")
    if jp:
        m = re.match(r"(\d{8})_(\d{6})", jp[-1])
        if m:
            t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            print(f"      newest {jp[-1]}  -> age {age(t):.2f} h")
    try:
        r = get("https://sdo.gsfc.nasa.gov/feeds/aia_0193.rss")
        pub = re.findall(r"<pubDate>([^<]+)</pubDate>", r.text)
        print(f"[sdo] RSS aia_0193: HTTP {r.status_code}, newest pubDate {pub[:1]}" +
              (f" (age {age(parsedate_to_datetime(pub[0]).astimezone(timezone.utc)):.2f} h)" if pub else ""))
    except Exception as ex:
        print("[sdo] rss failed", repr(ex)[:80])


def probe_helio():
    api = "https://api.helioviewer.org/v2"
    r = get(f"{api}/getDataSources/", params={"verbose": "false"})
    print(f"[helio] getDataSources HTTP {r.status_code}")
    sid = None
    try:
        js = r.json()
        def walk(o, path=()):
            nonlocal sid
            if isinstance(o, dict):
                if "sourceId" in o and "193" in "/".join(path) and "AIA" in "/".join(path):
                    sid = o["sourceId"]
                for k, v in o.items():
                    walk(v, path + (str(k),))
        walk(js)
    except Exception as ex:
        print("      could not parse sources", repr(ex)[:80])
    sid = sid or 13
    print(f"      AIA 193 sourceId = {sid}")
    r = get(f"{api}/getClosestImage/", params={"date": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "sourceId": sid})
    print(f"[helio] getClosestImage: HTTP {r.status_code} {r.text[:300]}")
    try:
        t = datetime.strptime(r.json()["date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        print(f"      newest image date {t}  -> age {age(t):.2f} h")
    except Exception:
        pass
    old = get(f"{api}/getClosestImage/", params={"date": "2013-06-01T00:00:00Z", "sourceId": sid})
    print(f"[helio] history check 2013-06-01: HTTP {old.status_code} {old.text[:150]}")


def probe_suvi():
    for url in ("https://services.swpc.noaa.gov/images/animations/suvi/primary/195/",
                "https://services.swpc.noaa.gov/products/solar-ultraviolet-imager/"):
        try:
            hs, r = links(url)
            print(f"[suvi] {url}: HTTP {r.status_code}, {len(hs)} entries, last: {sorted(hs)[-3:]}")
            names = [h for h in hs if re.search(r"\d{8}T\d{6}Z", h)]
            if names:
                m = re.search(r"(\d{8}T\d{6})Z", sorted(names)[-1])
                t = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                print(f"      newest {sorted(names)[-1]} -> age {age(t):.2f} h")
        except Exception as ex:
            print(f"[suvi] {url}: failed {repr(ex)[:100]}")


PROBES = {"nrt": probe_nrt, "mostrecent": probe_mostrecent, "sdo": probe_sdo, "helio": probe_helio, "suvi": probe_suvi}


def main():
    sel = sys.argv[1:] or list(PROBES)
    print(f"now (UTC) {NOW:%Y-%m-%d %H:%M}\n")
    for k in sel:
        print(f"==== {k} ====")
        try:
            PROBES[k]()
        except Exception as ex:
            print(f"[{k}] FAILED: {repr(ex)[:200]}")
        print()


if __name__ == "__main__":
    main()
