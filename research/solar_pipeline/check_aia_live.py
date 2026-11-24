"""How fresh are the AIA 193 synoptic images, really?  (decides --img-delay for strip_feature.py and the live pipeline)

    python check_aia_live.py                       # last 3 days, slots 00/06/12/18 UTC
    python check_aia_live.py --days 5 --hours 0000 0600 1200 1800 --log data/aia_live_log.csv
    python check_aia_live.py --listing             # also list the newest files in today's directory

For every nominal slot it HEADs  http://jsoc.stanford.edu/data/aia/synoptic/YYYY/MM/DD/Hhhmm/AIAyyyymmdd_hhmm_0193.fits
and reports: exists?, Last-Modified (UTC) and  delay = Last-Modified - slot time.   Run it a few times a day (or loop it
with --log appending one line per slot per run) -- the largest delay you see is what --img-delay must cover.
Also shows the newest slot that already exists ("newest image age" now): the issue time t0 may only use images older than that.
"""
import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

URL = "http://jsoc.stanford.edu/data/aia/synoptic/{y}/{m}/{d}/H{h}/AIA{y}{m}{d}_{h}_0193.fits"
DIR = "http://jsoc.stanford.edu/data/aia/synoptic/{y}/{m}/{d}/H{hh}00/"


def head(url, timeout=30):
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code in (405, 501) or ("Last-Modified" not in r.headers and r.status_code == 200):
            r = requests.get(url, timeout=timeout, stream=True, headers={"Range": "bytes=0-0"})
        return r.status_code, r.headers.get("Last-Modified"), r.headers.get("Content-Length")
    except Exception as ex:
        return None, None, repr(ex)[:60]


def scan(now, n):
    base = "http://jsoc.stanford.edu/data/aia/synoptic/{y}/{m}/{d}/"
    print(f"now (UTC) {now:%Y-%m-%d %H:%M}; day directories on JSOC for the last {n} days")
    for k in range(n, -1, -1):
        day = now - timedelta(days=k)
        url = base.format(y=f"{day:%Y}", m=f"{day:%m}", d=f"{day:%d}")
        try:
            r = requests.get(url, timeout=30)
        except Exception as ex:
            print(f"  {day:%Y-%m-%d}  request failed: {repr(ex)[:70]}")
            continue
        hdirs = sorted(set(re.findall(r'href="(H\d{4})/?"', r.text)))
        line = f"  {day:%Y-%m-%d}  HTTP {r.status_code}  hour dirs: {len(hdirs)}"
        if hdirs:
            r2 = requests.get(url + hdirs[-1] + "/", timeout=30)
            files = re.findall(r'href="(AIA[0-9_]+_0193\.fits)"', r2.text)
            line += f"  last dir {hdirs[-1]}: {len(files)} files, e.g. {files[:1]} .. {files[-1:]}"
        print(line)
    print("\nIf the day directories stop at some date, the archive lags by that much (set --img-delay accordingly or the images cannot be used live).")
    print("If directories exist but the exact slot names (AIAyyyymmdd_hhmm_0193.fits) differ, send me the 'e.g.' names above.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--hours", nargs="+", default=["0000", "0600", "1200", "1800"])
    ap.add_argument("--log", default=None, help="append one CSV line per slot (checked_at, slot, status, last_modified, delay_h)")
    ap.add_argument("--listing", action="store_true")
    ap.add_argument("--scan", type=int, default=0, metavar="N",
                    help="look at the last N days: does the day directory exist, which hour directories, newest file name and its Last-Modified")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    if a.scan:
        scan(now, a.scan)
        return
    print(f"now (UTC) {now:%Y-%m-%d %H:%M}")
    rows, newest = [], None
    for k in range(a.days, -1, -1):
        day = (now - timedelta(days=k)).replace(hour=0, minute=0, second=0, microsecond=0)
        for h in a.hours:
            slot = day + timedelta(hours=int(h[:2]), minutes=int(h[2:]))
            if slot > now:
                continue
            code, lm, size = head(URL.format(y=f"{slot:%Y}", m=f"{slot:%m}", d=f"{slot:%d}", h=h))
            lmt = parsedate_to_datetime(lm).astimezone(timezone.utc) if lm else None
            delay = (lmt - slot).total_seconds() / 3600 if lmt else None
            rows.append((slot, code, lmt, delay))
            if code == 200:
                newest = slot
            print(f"  slot {slot:%Y-%m-%d %H:%M}  HTTP {code}  size {size}  last-modified "
                  f"{lmt:%Y-%m-%d %H:%M}  delay {delay:5.1f} h" if lmt else
                  f"  slot {slot:%Y-%m-%d %H:%M}  HTTP {code}  {size if code is None else ''}")
    ok = [r for r in rows if r[1] == 200 and r[3] is not None]
    missing = [r for r in rows if r[1] != 200]
    if ok:
        d = sorted(r[3] for r in ok)
        print(f"\ndelay hours over {len(ok)} slots: min {d[0]:.1f}  median {d[len(d) // 2]:.1f}  max {d[-1]:.1f}")
    print(f"missing slots: {len(missing)} of {len(rows)}" + ("  (" + ", ".join(f"{r[0]:%m-%d %H}" for r in missing[-6:]) + ")" if missing else ""))
    if newest:
        print(f"newest existing slot: {newest:%Y-%m-%d %H:%M} UTC, i.e. {(now - newest).total_seconds() / 3600:.1f} h old right now")
    if a.log:
        new = not __import__("os").path.exists(a.log)
        with open(a.log, "a") as f:
            if new:
                f.write("checked_at,slot,status,last_modified,delay_h\n")
            for slot, code, lmt, delay in rows:
                f.write(f"{now:%Y-%m-%dT%H:%M},{slot:%Y-%m-%dT%H:%M},{code},{lmt:%Y-%m-%dT%H:%M}" if lmt else
                        f"{now:%Y-%m-%dT%H:%M},{slot:%Y-%m-%dT%H:%M},{code},")
                f.write(f",{delay:.2f}\n" if delay is not None else ",\n")
    if a.listing:
        t = now
        r = requests.get(DIR.format(y=f"{t:%Y}", m=f"{t:%m}", d=f"{t:%d}", hh=f"{t:%H}"), timeout=30)
        names = re.findall(r'href="(AIA[0-9_]+_0193\.fits)"', r.text)
        print(f"\nlisting {r.url}: HTTP {r.status_code}, {len(names)} files, newest: {names[-3:]}")
    print("\nrule: issue time t0 may use an image whose nominal time + delay <= t0.  Use  --img-delay <max delay rounded up>  in strip_feature.py "
          "(and 24 if the daily image of the current day is not there by t0).")


if __name__ == "__main__":
    main()
