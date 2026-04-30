import glob
import json
import numpy as np
from src import config


def _fit_exgaussian_tau(rts):
    """Return the exponential tail τ of an ex-Gaussian fit (µ of the exp component)."""
    if len(rts) < 8:
        return 0.0
    try:
        from scipy.stats import exponnorm
        K, _loc, scale = exponnorm.fit(rts, floc=0)
        return float(np.clip(K * scale, 0.0, 10.0))
    except Exception:
        return 0.0


def extract_tags_features(uid):
    import pandas as pd
    from scipy.stats import skew as _skew, kurtosis as _kurt

    all_rts, omissions_all, per_lvl_rts = [], [], []

    for lvl in config.SLACKLINE_LEVELS:
        tfiles = glob.glob(f"{config.DATA_ROOT}/{uid}/{lvl}/*/*_TAGS_*.csv")
        if not tfiles:
            per_lvl_rts.append([])
            continue
        df = pd.read_csv(tfiles[0])
        rts, omissions = [], 0
        for _, row in df.iterrows():
            try:
                v = row["value"]
                if isinstance(v, str):
                    v = (v.replace("'", '"')
                         .replace("True", "true").replace("False", "false"))
                    r = json.loads(v)["reactionOrOmission"][0]
                else:
                    continue
                reacted = str(r.get("reacted", "false")).lower() == "true"
                if reacted:
                    rt = float(r.get("reactionTime", 0))
                    if 0 < rt < 10:
                        rts.append(rt)
                else:
                    omissions += 1
            except Exception:
                continue
        n_total = max(len(rts) + omissions, 1)
        per_lvl_rts.append(rts)
        all_rts.extend(rts)
        omissions_all.append(omissions / n_total)

    mean_rt = float(np.mean(all_rts))  if all_rts else 0.0
    std_rt  = float(np.std(all_rts))   if all_rts else 0.0
    cv_rt   = std_rt / (mean_rt + 1e-6) if mean_rt > 0 else 0.0
    omis_r  = float(np.mean(omissions_all)) if omissions_all else 0.0

    # Distribution shape — discriminates ADHD via a heavier exponential tail
    rt_skew = float(_skew(all_rts))     if len(all_rts) >= 4 else 0.0
    rt_kurt = float(_kurt(all_rts))     if len(all_rts) >= 4 else 0.0
    rt_tau  = _fit_exgaussian_tau(all_rts)

    # Intra-block (per-level) RT variability — captures vigilance decrement
    cv_per_block = []
    for lvl_rts in per_lvl_rts:
        if len(lvl_rts) >= 3:
            cv_per_block.append(np.std(lvl_rts) / (np.mean(lvl_rts) + 1e-6))
    rt_cv_pb = float(np.mean(cv_per_block)) if cv_per_block else 0.0

    # Order must match base_names in dataset.py:
    # ["mean_RT", "RT_CV", "RT_skew", "RT_kurt", "RT_tau", "RT_cv_per_block", "omission_rate"]
    return np.array([mean_rt, cv_rt, rt_skew, rt_kurt, rt_tau, rt_cv_pb, omis_r], dtype=float)


def extract_game_features(uid):
    import pandas as pd
    gfiles = glob.glob(
        f"{config.DATA_ROOT}/{uid}/AttentionRobotsDesktop/*/*_GAME_DATA*.csv")
    if not gfiles:
        return np.zeros(3)
    try:
        df  = pd.read_csv(gfiles[0])
        row = df.iloc[0]
        vel  = [float(row.get(f"velocidadTrabajoBloque{i}", 0)) for i in range(1, 5)]
        omis = [float(row.get(f"omisionBloque{i}",          0)) for i in range(1, 5)]
        coms = [float(row.get(f"comisionBloque{i}",         0)) for i in range(1, 5)]
        return np.array([float(np.mean(vel)), sum(omis), sum(coms)], dtype=float)
    except Exception:
        return np.zeros(3)
