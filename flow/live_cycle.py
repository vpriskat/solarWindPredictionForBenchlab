from prefect import flow, task, get_run_logger
from swp import model
from prefect.futures import wait
import pandas as pd
import json, os, sys, time, traceback
from datetime import datetime, timedelta, timezone
import numpy as np
from swp.storage import CycleRepository
from swp.storage.factory import create_byte_store

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
        is_empty = result is None or (
                data_source == "ace" and isinstance(result, tuple) and result[0] is None
        )
        if is_empty:
            info[f"{data_source}_status"] = "empty"
            return default
        info[f"{data_source}_status"] = "success"
        return result




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

    result_strip = resolve_future(strip_fut, data_source="strip", info=info, logger=logger, default=None)
    adj = result_strip

    return model.finalize(t0, obs, src, enlil, adj, info)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: model.py <YYYYMMDDTHHMMSSZ>", file=sys.stderr)
        sys.exit(1)

    t0String = sys.argv[1]
    t0 = pd.Timestamp(datetime.strptime(t0String, "%Y%m%dT%H%M%SZ"))
    pred, info  = live_cycle(t0)

    info["retrieved_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log("info:", info)
    log("forecast_kms:", json.dumps([round(float(v), 1) for v in pred]))
    json.dump({"t0": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "valid_from": (t0 + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "forecast_speed_kms": [float(v) for v in pred]}, sys.stdout)

    store = CycleRepository( create_byte_store() )
    store.save_forecast(t0String, pred)
    store.save_run_record(t0String, info)