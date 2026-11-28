"""Website-ready JSON built from the raw store (forecasts/, runs/, truth/)."""
import json
from datetime import datetime

import pandas as pd

FMT = "%Y-%m-%dT%H:%M:%SZ"
STAMP = "%Y%m%dT%H%M%SZ"
STATUS_KEYS = ("level", "ace_status", "enlil_status", "strip_status")
LATEST_KEYS = STATUS_KEYS + ("ace_last_lag_h", "n_enlil", "enlil_run", "enlil_run_age_h", "strip_n_images",
                             "strip_lags_matched", "strip_newest_age_h", "fc_max_step_kms", "fc_max_step_lead",
                             "git_sha", "run_url", "retrieved_at_utc")


def _load(store, key):
    return json.loads(store.get(key)) if store.exists(key) else None


def _forecast_values(doc):
    """Accepts a bare list or a dict holding the 72 values."""
    if isinstance(doc, list):
        return [round(float(v), 1) for v in doc]
    for k in ("forecast_speed_kms", "forecast", "values", "pred", "prediction"):
        if k in doc:
            return [round(float(v), 1) for v in doc[k]]
    raise KeyError(f"no forecast values in keys {list(doc)[:6]}")


def _t0(stamp):
    return pd.Timestamp(datetime.strptime(stamp, STAMP))


def list_cycles(store):
    """Sorted t0 stamps (YYYYMMDDTHHMMSSZ) that have a forecast."""
    names = [k.rsplit("/", 1)[-1].removesuffix(".json") for k in store.list_prefix("forecasts/")]
    return sorted(n for n in names if len(n) == 16 and n.endswith("Z"))


def build_truth(store, now, days):
    start = (now - pd.Timedelta(days=days)).floor("h")
    values = {}
    for ym in pd.period_range(start, now, freq="M").strftime("%Y%m"):
        doc = _load(store, f"truth/{ym}.json")
        if doc:
            values.update(doc["values"])
    arr = [values.get(t.strftime(FMT)) for t in pd.date_range(start, now.floor("h"), freq="h")]
    last = max((i for i, v in enumerate(arr) if v is not None), default=-1)
    return {"start": start.strftime(FMT), "step_h": 1, "values": arr[: last + 1]}


def build_recent(store, now, days=7, truth_days=10):
    cutoff = now - pd.Timedelta(days=days)
    cycles = []
    for stamp in list_cycles(store):
        t0 = _t0(stamp)
        if t0 < cutoff or t0 > now:
            continue
        run = _load(store, f"runs/{stamp}.json") or {}
        cycles.append({"t0": t0.strftime(FMT), **{k: run.get(k) for k in STATUS_KEYS},
                       "values": _forecast_values(_load(store, f"forecasts/{stamp}.json"))})
    return {"generated_at": now.strftime(FMT), "cycles": cycles, "truth": build_truth(store, now, truth_days)}


def build_latest(store, now):
    stamps = [s for s in list_cycles(store) if _t0(s) <= now]
    if not stamps:
        return None
    stamp = stamps[-1]
    t0 = _t0(stamp)
    run = _load(store, f"runs/{stamp}.json") or {}
    return {"generated_at": now.strftime(FMT), "t0": t0.strftime(FMT), "valid_from": (t0 + pd.Timedelta(hours=1)).strftime(FMT),
            "values": _forecast_values(_load(store, f"forecasts/{stamp}.json")),
            "components": run.get("components"), "info": {k: run.get(k) for k in LATEST_KEYS}}


def _put(store, key, doc):
    store.put(key, json.dumps(doc, separators=(",", ":")).encode())


def publish(store, now, days=7):
    written = []
    latest = build_latest(store, now)
    if latest is not None:
        _put(store, "web/latest.json", latest)
        written.append("web/latest.json")
    _put(store, "web/recent.json", build_recent(store, now, days))
    written.append("web/recent.json")
    return written