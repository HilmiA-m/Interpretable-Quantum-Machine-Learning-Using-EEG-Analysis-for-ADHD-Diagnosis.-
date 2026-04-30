import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _parse():
    p = argparse.ArgumentParser(
        description="BALLADEER — Quantum ML for ADHD diagnosis")
    p.add_argument(
        "--data", metavar="PATH",
        help="Path to the BALLADEER ADHD DATASET folder "
             "(overrides BALLADEER_DATA env var and the hardcoded default)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.data:
        os.environ["BALLADEER_DATA"] = args.data

    from src.pipeline import run
    run()
