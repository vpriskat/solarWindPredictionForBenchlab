import argparse
import sys
from datetime import datetime, timezone

import pandas as pd
from prefect import flow, task, get_run_logger

from swp import truth
from swp.storage.factory import create_byte_store


def utc_now():
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)


@task
def get_truth_values(days):
    return truth.fetch_truth(utc_now(), days)


@flow
def update_truth(days=7):
    logger = get_run_logger()
    values, src = get_truth_values(days)
    written = truth.update_truth(create_byte_store(), values)
    logger.info("truth: %d hourly values from %s; months written: %s", len(values), src, written)
    return written


def main(argv):
    ap = argparse.ArgumentParser(prog="truth")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--backfill-from", metavar="YYYYMM")
    a = ap.parse_args(argv)
    days = a.days
    if a.backfill_from:
        days = (utc_now() - pd.Timestamp(datetime.strptime(a.backfill_from, "%Y%m"))).days + 1
    update_truth(days)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))