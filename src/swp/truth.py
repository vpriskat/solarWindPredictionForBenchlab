"""Truth series: hourly ACE speed, one JSON per month, merged idempotently."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from swp import model

FMT = "%Y-%m-%dT%H:%M:%SZ"


def series_to_dict(s):
    s = s.dropna()
    return {ts.strftime(FMT): round(float(v), 1) for ts, v in s.items()}


def merge_truth(old, new):
    """New non-missing values win; a missing value never overwrites an existing one."""
    ok = {k: v for k, v in new.items() if v is not None and np.isfinite(v)}
    return dict(sorted({**old, **ok}.items()))


def month_key(ts_str):
    return ts_str[:4] + ts_str[5:7]                      # "2026-11-28T..." -> "202611"


def fetch_truth(now, days=7):
    rows, src, errors = model.load_ace_rows(now, days)
    if not rows:
        raise RuntimeError("truth unavailable: " + ("; ".join(errors) or "no rows")[:300])
    s = pd.Series(rows).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s[s.index == s.index.floor("h")]
    return series_to_dict(s), src


def update_truth(store, values):
    """Merge hourly values into truth/YYYYMM.json. Returns {month: number of values now stored} for changed months."""
    by_month = {}
    for k, v in values.items():
        by_month.setdefault(month_key(k), {})[k] = v
    written = {}
    for ym, new in by_month.items():
        key = f"truth/{ym}.json"
        old = json.loads(store.get(key))["values"] if store.exists(key) else {}
        merged = merge_truth(old, new)
        if merged == old:
            continue
        doc = {"month": ym, "updated_at": datetime.now(timezone.utc).strftime(FMT), "n_valid": len(merged), "values": merged}
        store.put(key, json.dumps(doc).encode())
        written[ym] = len(merged)
    return written