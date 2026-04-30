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
    p.add_argument(
        "--skip-vqc", action="store_true",
        help="Skip VQC training and reload weights from "
             "RESULTS/best_quantum_params_v40.json  (saves ~2 hours)")
    p.add_argument(
        "--skip-qsvm", action="store_true",
        help="Skip QSVM kernel computation (replaces with 0.5 dummy predictions)")
    p.add_argument(
        "--skip-qcnn", action="store_true",
        help="Skip QCNN training (replaces with 0.5 dummy predictions)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.data:
        os.environ["BALLADEER_DATA"] = args.data

    from src.pipeline import run
    run(
        skip_vqc  = args.skip_vqc,
        skip_qsvm = args.skip_qsvm,
        skip_qcnn = args.skip_qcnn,
    )
