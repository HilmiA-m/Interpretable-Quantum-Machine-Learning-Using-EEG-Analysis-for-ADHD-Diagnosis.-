import os
import numpy as np
from src import config


def load_embrace():
    try:
        import pandas as pd
        ep_df = pd.read_csv(
            os.path.join(config.DATA_ROOT, "balladeer_embraceplus_data.csv"), sep=";")
        cols = [c for c in ep_df.columns
                if c != "username"
                and "wearing" not in c
                and any(kw in c.lower() for kw in ("eda_mean", "hr_mean", "mean_bpm"))]
        if not cols:
            cols = [c for c in ep_df.columns
                    if c != "username"
                    and any(c.endswith(s) for s in ("_mean", "_mean_bpm"))
                    and "wearing" not in c][:3]
        lookup = {r["username"]: r for _, r in ep_df.iterrows()}
        print(f"[4] Embrace+: {len(cols)} selected cols ({cols}), {len(lookup)} subjects")
        return lookup, cols
    except Exception as e:
        print(f"[4] Embrace+ skipped: {e}")
        return {}, []


def get_embrace(uid, lookup, cols):
    row = lookup.get(uid)
    if row is None:
        return np.full(len(cols), np.nan)
    return np.array([float(row.get(c, np.nan)) for c in cols], dtype=float)
