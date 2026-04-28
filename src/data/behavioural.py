import glob
import json
import numpy as np
from src import config


def extract_tags_features(uid):
    import pandas as pd
    rts_all, omissions_all = [], []
    for lvl in config.SLACKLINE_LEVELS:
        tfiles = glob.glob(f"{config.DATA_ROOT}/{uid}/{lvl}/*/*_TAGS_*.csv")
        if not tfiles:
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
        if rts:
            rts_all.extend(rts)
        omissions_all.append(omissions / n_total)
    mean_rt = float(np.mean(rts_all)) if rts_all else 0.0
    std_rt  = float(np.std(rts_all))  if rts_all else 0.0
    cv_rt   = std_rt / (mean_rt + 1e-6) if mean_rt > 0 else 0.0
    omis_r  = float(np.mean(omissions_all)) if omissions_all else 0.0
    return np.array([mean_rt, cv_rt, omis_r], dtype=float)


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
