"""
Feature extraction script — run once before training.

Reads raw EEG, behavioural, game, and wearable files, extracts all features,
imputes missing values, and saves a compact processed dataset to data/processed/.
After running this script the raw data folder can be safely deleted.

Usage
-----
    # Basic — uses data/ folder by default
    python preprocess.py

    # Point at a different raw data location
    python preprocess.py --data "C:/path/to/BALLADEER ADHD DATASET"

    # Extract features AND delete raw data when done
    python preprocess.py --delete-raw

    # Full example
    python preprocess.py --data "C:/path/to/dataset" --delete-raw
"""

import os
import sys
import json
import argparse
import shutil
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_PROC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")


def _parse():
    p = argparse.ArgumentParser(
        description="Extract features from raw BALLADEER dataset and save to data/processed/")
    p.add_argument("--data", metavar="PATH",
                   help="Path to raw dataset folder (default: data/)")
    p.add_argument("--delete-raw", action="store_true",
                   help="Delete the raw dataset folder after successful extraction")
    return p.parse_args()


def main():
    args = _parse()

    if args.data:
        os.environ["BALLADEER_DATA"] = args.data

    from src import config

    print("=" * 68)
    print("BALLADEER — Feature Extraction")
    print(f"Source : {config.DATA_ROOT}")
    print(f"Output : {_PROC_DIR}")
    print("=" * 68)

    if not os.path.exists(config.DATA_ROOT):
        sys.exit(
            f"\n[FATAL] Dataset not found at: {config.DATA_ROOT}\n"
            f"        Use --data to specify the correct path.\n"
        )

    # ── Load raw modalities ───────────────────────────────────────────────────
    print("\n[1] Loading demographics...")
    from src.data.demographics import load_demographics
    user_meta = load_demographics()

    print("\n[2] Loading Embrace+ wearable data...")
    from src.data.embrace import load_embrace
    embrace_lookup, embrace_cols = load_embrace()

    # ── Build feature matrix ──────────────────────────────────────────────────
    print("\n[3] Extracting features for all subjects...")
    print("    (EEG: MNE pipeline + CAR + artifact rejection)")
    print("    (Behavioural: RT stats + ex-Gaussian τ)")
    print("    (Game: velocity, omissions, commissions)")
    from src.dataset import build_dataset
    X_raw, y, uid_list, feat_names = build_dataset(
        user_meta, embrace_lookup, embrace_cols)

    # ── Save processed dataset ────────────────────────────────────────────────
    os.makedirs(_PROC_DIR, exist_ok=True)

    np.save(os.path.join(_PROC_DIR, "X.npy"), X_raw)
    np.save(os.path.join(_PROC_DIR, "y.npy"), y)

    with open(os.path.join(_PROC_DIR, "feat_names.json"), "w") as f:
        json.dump(feat_names, f, indent=2)

    with open(os.path.join(_PROC_DIR, "uid_list.json"), "w") as f:
        json.dump(uid_list, f, indent=2)

    meta = {
        "extracted_at":  datetime.now().isoformat(timespec="seconds"),
        "n_subjects":    int(len(y)),
        "n_adhd":        int(y.sum()),
        "n_control":     int((y == 0).sum()),
        "n_features":    int(X_raw.shape[1]),
        "embrace_cols":  embrace_cols,
        "feat_names":    feat_names,
    }
    with open(os.path.join(_PROC_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = (X_raw.nbytes + y.nbytes) / 1e6
    print(f"\n[OK] Saved to {_PROC_DIR}")
    print(f"     X.npy        : {X_raw.shape}  ({size_mb:.1f} MB)")
    print(f"     y.npy        : {y.shape}")
    print(f"     feat_names   : {len(feat_names)} features")
    print(f"     uid_list     : {len(uid_list)} subjects")
    print(f"     meta.json    : dataset metadata")

    # ── Optionally delete raw data ────────────────────────────────────────────
    if args.delete_raw:
        raw_path = config.DATA_ROOT
        print(f"\n[delete-raw] Removing {raw_path} ...")
        if os.path.abspath(raw_path) == os.path.abspath(_PROC_DIR):
            print("  Skipped — raw and processed paths are the same folder.")
        else:
            shutil.rmtree(raw_path)
            print(f"  [OK] Deleted {raw_path}")
            # Recreate the empty data/ folder so the project structure stays intact
            empty_data = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data")
            os.makedirs(empty_data, exist_ok=True)
            gitkeep = os.path.join(empty_data, ".gitkeep")
            if not os.path.exists(gitkeep):
                open(gitkeep, "w").close()
            print(f"  [OK] Recreated empty data/ with .gitkeep")

    print("\nDone. Run  python main.py  to train all models.")


if __name__ == "__main__":
    main()
