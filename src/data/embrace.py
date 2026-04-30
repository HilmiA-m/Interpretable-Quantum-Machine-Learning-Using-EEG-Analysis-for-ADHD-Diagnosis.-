import os
import numpy as np
from src import config

# Keywords that identify useful Embrace+ columns.
# Expanded beyond original (eda_mean, hr_mean) to include:
#   eda_std  — tonic/phasic EDA variability (autonomic arousal marker)
#   rmssd / sdnn — HRV metrics correlated with inhibitory control in ADHD
_KEEP_KW  = ("eda_mean", "eda_std", "hr_mean", "mean_bpm", "hr_std", "rmssd", "sdnn")
_SKIP_KW  = ("wearing",)


def load_embrace():
    try:
        import pandas as pd
        ep_df = pd.read_csv(
            os.path.join(config.DATA_ROOT, "balladeer_embraceplus_data.csv"), sep=";")
        cols = [
            c for c in ep_df.columns
            if c != "username"
            and not any(sk in c.lower() for sk in _SKIP_KW)
            and any(kw in c.lower() for kw in _KEEP_KW)
        ]
        if not cols:
            # Broad fallback: any _mean / _std column
            cols = [
                c for c in ep_df.columns
                if c != "username"
                and not any(sk in c.lower() for sk in _SKIP_KW)
                and any(c.endswith(s) for s in ("_mean", "_std", "_mean_bpm"))
            ][:6]
        lookup = {r["username"]: r for _, r in ep_df.iterrows()}
        print(f"[4] Embrace+: {len(cols)} cols ({cols}), {len(lookup)} subjects")
        return lookup, cols
    except Exception as e:
        print(f"[4] Embrace+ skipped: {e}")
        return {}, []


def get_embrace(uid, lookup, cols):
    row = lookup.get(uid)
    if row is None:
        return np.full(len(cols), np.nan)
    return np.array([float(row.get(c, np.nan)) for c in cols], dtype=float)
