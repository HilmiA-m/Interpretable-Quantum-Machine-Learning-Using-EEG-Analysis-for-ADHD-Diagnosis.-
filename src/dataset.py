import json
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src import config
from src.data.eeg import extract_eeg_features, find_eeg_file
from src.data.behavioural import extract_tags_features, extract_game_features
from src.data.embrace import get_embrace

_PROC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed")


def processed_data_exists():
    return os.path.exists(os.path.join(_PROC_DIR, "X.npy"))


def load_processed():
    """Load pre-extracted feature matrix saved by preprocess.py."""
    X_raw = np.load(os.path.join(_PROC_DIR, "X.npy"))
    y     = np.load(os.path.join(_PROC_DIR, "y.npy"))
    with open(os.path.join(_PROC_DIR, "feat_names.json")) as f:
        feat_names = json.load(f)
    with open(os.path.join(_PROC_DIR, "uid_list.json")) as f:
        uid_list = json.load(f)
    with open(os.path.join(_PROC_DIR, "meta.json")) as f:
        meta = json.load(f)
    print(f"  [processed] Loaded from {_PROC_DIR}")
    print(f"  Subjects={meta['n_subjects']}  "
          f"ADHD={meta['n_adhd']}  Control={meta['n_control']}  "
          f"Features={meta['n_features']}")
    print(f"  Extracted at: {meta.get('extracted_at', 'unknown')}")
    return X_raw, y, uid_list, feat_names

# 22 EEG features — must match the return order of extract_eeg_features()
EEG_FEAT_NAMES = [
    "TBR_mean", "TBR_std", "frontal_TBR", "intra_TBR_std",   # 0-3
    "alpha_asym", "theta_asym",                                # 4-5
    "occ_alpha",                                               # 6
    "theta_entropy", "beta_entropy",                           # 7-8
    "hjorth_mob_f", "hjorth_comp_f",                          # 9-10
    "hjorth_mob_o", "hjorth_comp_o",                          # 11-12
    "hjorth_mob_c", "hjorth_comp_c",                          # 13-14
    "TAR", "gamma_theta_ratio",                                # 15-16
    "delta_pow", "theta_pow", "alpha_pow", "beta_pow", "gamma_pow",  # 17-21
]

# Index lookup used for cross-level gradient features
_IDX = {name: i for i, name in enumerate(EEG_FEAT_NAMES)}


def build_dataset(user_meta, embrace_lookup, embrace_cols):
    X_rows, y_rows, uid_list = [], [], []

    for uid, meta in user_meta.items():
        eeg_parts = []
        for lvl in config.SLACKLINE_LEVELS:
            p = find_eeg_file(uid, lvl)
            if p:
                feats = extract_eeg_features(p)
                if feats is not None:
                    eeg_parts.append(feats)
        if not eeg_parts:
            continue
        eeg_avg = np.mean(eeg_parts, axis=0)

        # ── Cross-level gradient features (neural load response) ─────────────
        if len(eeg_parts) >= 2:
            def _delta(key): return eeg_parts[-1][_IDX[key]] - eeg_parts[0][_IDX[key]]
            tbr_delta   = _delta("TBR_mean")
            asym_delta  = _delta("alpha_asym")
            occ_a_delta = _delta("occ_alpha")
            theta_delta = _delta("theta_pow")
            hjmf_delta  = _delta("hjorth_mob_f")
        else:
            tbr_delta = asym_delta = occ_a_delta = theta_delta = hjmf_delta = 0.0

        # ── Missingness indicators ────────────────────────────────────────────
        eeg_n_levels    = float(len(eeg_parts))
        embrace_present = 1.0 if embrace_lookup.get(uid) is not None else 0.0

        tags_feats   = extract_tags_features(uid)
        tags_present = 1.0 if tags_feats[0] > 0 else 0.0   # mean_RT > 0 ↔ data found

        embrace_feats = get_embrace(uid, embrace_lookup, embrace_cols)

        row = np.concatenate([
            np.array([meta["age"], meta["gender"]]),
            eeg_avg,
            np.array([tbr_delta, asym_delta, occ_a_delta, theta_delta, hjmf_delta]),
            np.array([eeg_n_levels, embrace_present, tags_present]),
            embrace_feats,
            tags_feats,
            extract_game_features(uid),
        ])
        X_rows.append(row)
        y_rows.append(meta["label"])
        uid_list.append(uid)

    X_raw = np.array(X_rows, dtype=float)
    y     = np.array(y_rows)

    # ── NaN imputation (median per feature, on all data) ────────────────────
    for col in range(X_raw.shape[1]):
        m = np.isnan(X_raw[:, col])
        if m.any():
            med = np.nanmedian(X_raw[~m, col]) if (~m).any() else 0.0
            X_raw[m, col] = med

    feat_names = (
        ["age", "gender"]
        + EEG_FEAT_NAMES
        + ["TBR_delta", "alpha_asym_delta", "occ_alpha_delta",
           "theta_pow_delta", "hjorth_mob_f_delta"]
        + ["eeg_n_levels", "embrace_present", "tags_present"]
        + [f"embrace_{c}" for c in embrace_cols]
        + ["mean_RT", "RT_CV", "RT_skew", "RT_kurt", "RT_tau",
           "RT_cv_per_block", "omission_rate"]
        + ["game_velocity", "game_omissions", "game_commissions"]
    )

    print(f"  Final dataset : n={len(y)}, ADHD={int(y.sum())}, Ctrl={int((y == 0).sum())}")
    print(f"  Feature matrix: {X_raw.shape}")
    return X_raw, y, uid_list, feat_names


def preprocess(X_raw, y, feat_names):
    idx_all = np.arange(len(y))
    idx_tr, idx_te = train_test_split(
        idx_all, test_size=config.TEST_SIZE, random_state=42, stratify=y)
    y_tr = y[idx_tr]; y_te = y[idx_te]
    X_tr_raw = X_raw[idx_tr].copy()
    X_te_raw = X_raw[idx_te].copy()

    # ── Remove zero-variance features (computed on train only) ───────────────
    var_mask    = X_tr_raw.std(axis=0) > 1e-10
    X_tr_raw    = X_tr_raw[:, var_mask]
    X_te_raw    = X_te_raw[:, var_mask]
    feat_names  = [fn for fn, keep in zip(feat_names, var_mask) if keep]
    n_dropped   = int((~var_mask).sum())
    if n_dropped:
        print(f"  Dropped {n_dropped} zero-variance features (train-only check)")

    # ── Winsorize on training statistics to cap extreme outliers ────────────
    # 5th/95th percentile bounds computed on train, applied to both splits.
    lower = np.percentile(X_tr_raw, 5,  axis=0)
    upper = np.percentile(X_tr_raw, 95, axis=0)
    X_tr_raw = np.clip(X_tr_raw, lower, upper)
    X_te_raw = np.clip(X_te_raw, lower, upper)

    # ── Quantum pipeline: StdScale → PCA-N → MinMax [0, π] ─────────────────
    ss_q       = StandardScaler().fit(X_tr_raw)
    X_tr_q_std = ss_q.transform(X_tr_raw)
    X_te_q_std = ss_q.transform(X_te_raw)
    pca        = PCA(n_components=config.N_PCA, random_state=42).fit(X_tr_q_std)
    X_tr_pca   = pca.transform(X_tr_q_std)
    X_te_pca   = pca.transform(X_te_q_std)
    print(f"  Q PCA var: {np.cumsum(pca.explained_variance_ratio_).round(3).tolist()}")
    mm     = MinMaxScaler((0, np.pi)).fit(X_tr_pca)
    X_tr_q = mm.transform(X_tr_pca)
    X_te_q = mm.transform(X_te_pca)

    # ── Classical pipeline: StdScale only ───────────────────────────────────
    ss_cls   = StandardScaler().fit(X_tr_raw)
    X_tr_cls = ss_cls.transform(X_tr_raw)
    X_te_cls = ss_cls.transform(X_te_raw)

    print(f"  Train: {len(y_tr)} | Test: {len(y_te)}")
    print(f"  Train ADHD={int(y_tr.sum())}  Ctrl={int((y_tr == 0).sum())}")
    print(f"  Test  ADHD={int(y_te.sum())}  Ctrl={int((y_te == 0).sum())}")
    print("  Winsorization: [5%, 95%] bounds from train applied to both splits.")
    print("  *** No SMOTE — class_weight='balanced' instead ***")

    return dict(
        X_tr_raw=X_tr_raw, X_te_raw=X_te_raw,
        X_tr_q=X_tr_q,     X_te_q=X_te_q,
        X_tr_cls=X_tr_cls, X_te_cls=X_te_cls,
        y_tr=y_tr,         y_te=y_te,
        ss_q=ss_q,         pca=pca,  mm=mm,
        lower=lower,        upper=upper,
        feat_names=feat_names,
    )
