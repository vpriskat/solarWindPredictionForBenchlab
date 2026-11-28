from prefect import flow, task, get_run_logger
from swp import model
from prefect.futures import wait
import pandas as pd
import json, os, sys, time, traceback
from datetime import datetime, timedelta, timezone
import numpy as np
from swp.storage import CycleRepository
from swp.storage.factory import create_byte_store
from swp import diagnostics
T_START = time.time()


def log(*a):
    print(f"[{time.time() - T_START:6.1f}s]", *a, file=sys.stderr, flush=True)

@task
def get_ace(t0):
    return model.fetch_ace(t0)

@task
def get_strip(t0):
    return model.strip_adjustment(t0)

@task
def get_enlil():
    return model.fetch_enlil()


def resolve_future(fut, data_source: str, info: dict, logger, default=None):
    if fut is None:
        info[f"{data_source}_status"] = "skipped"
        return default

    if fut.state.is_failed():
        exc = fut.state.result(raise_on_failure=False)
        msg = repr(exc)[:100]
        logger.error("%s task failed: %r", data_source, exc)
        info[f"{data_source}_status"] = "failed"
        info[f"{data_source}_error"] = msg
        return default
    else:
        result = fut.result()
        is_empty = result is None or (isinstance(result, tuple) and result[0] is None)
        info[f"{data_source}_status"] = "empty" if is_empty else "success"
        return default if result is None else result





@flow
def live_cycle(t0):
    logger = get_run_logger()
    info = {}
    ace_fut = get_ace.submit(t0)
    enlil_fut = get_enlil.submit()
    strip_fut = get_strip.submit(t0) if os.environ.get("SWP_NO_STRIP") != "1" else None

    futures = [f for f in (ace_fut, enlil_fut, strip_fut) if f is not None]
    wait(futures)

    result_ace = resolve_future(ace_fut, data_source="ace", info=info, logger=logger, default=(None, []))
    obs, src = result_ace

    result_enlil = resolve_future(enlil_fut, data_source="enlil", info=info, logger=logger, default=None)
    enlil = result_enlil

    result_strip = resolve_future(strip_fut, data_source="strip", info=info, logger=logger, default=(None, {}))
    adj, strip_info = result_strip
    info.update(strip_info)

    return model.finalize(t0, obs, src, enlil, adj, info)


def latest_cycle_t0(now=None):
    """Most recent 6-hour cycle boundary (00/06/12/18 UTC), as a naive timestamp."""
    now = now or datetime.now(timezone.utc)
    return now.replace(hour=now.hour // 6 * 6, minute=0, second=0, microsecond=0, tzinfo=None)


def main(argv):
    if len(argv) > 1:
        print("usage: live_cycle.py [YYYYMMDDTHHMMSSZ]  (default: latest 6h cycle)", file=sys.stderr)
        return 1
    if len(argv) == 1:
        t0 = pd.Timestamp(datetime.strptime(argv[0], "%Y%m%dT%H%M%SZ"))
    else:
        t0 = pd.Timestamp(latest_cycle_t0())
    t0String = t0.strftime("%Y%m%dT%H%M%SZ")

    pred, info = live_cycle(t0)

    info.update(diagnostics.provenance(model.HERE))
    info["retrieved_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log("info:", {k: v for k, v in info.items() if k != "components"})
    log("forecast_kms:", json.dumps([round(float(v), 1) for v in pred]))
    json.dump({"t0": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "valid_from": (t0 + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "forecast_speed_kms": [float(v) for v in pred]}, sys.stdout)

    store = CycleRepository(create_byte_store())
    store.save_forecast(t0String, pred)
    store.save_run_record(t0String, info)
    return 0