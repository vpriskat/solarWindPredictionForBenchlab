"""Dispatcher: python3 flow/cli.py <command> [args]"""
import sys

COMMANDS = {
    "cycle": ("live_cycle", "forecast cycle: cycle [YYYYMMDDTHHMMSSZ]"),
    "truth": ("update_truth", "update the truth series: truth [--days N] [--backfill-from YYYYMM]"),
    "publish": ("publish", "write website JSON: publish [--days N]"),
}


def main(argv):
    if not argv or argv[0] not in COMMANDS:
        print("usage: cli.py <command> [args]\n" + "\n".join(f"  {k}: {v[1]}" for k, v in COMMANDS.items()), file=sys.stderr)
        return 2
    module = __import__(COMMANDS[argv[0]][0])
    return module.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))