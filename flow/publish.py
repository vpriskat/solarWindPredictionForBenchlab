import argparse
import sys

from prefect import flow, get_run_logger

from swp import web
from swp.storage.factory import create_byte_store
from update_truth import utc_now


@flow
def publish(days=7):
    logger = get_run_logger()
    written = web.publish(create_byte_store(), utc_now(), days)
    logger.info("published: %s", written)
    return written


def main(argv):
    ap = argparse.ArgumentParser(prog="publish")
    ap.add_argument("--days", type=int, default=7)
    publish(ap.parse_args(argv).days)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))