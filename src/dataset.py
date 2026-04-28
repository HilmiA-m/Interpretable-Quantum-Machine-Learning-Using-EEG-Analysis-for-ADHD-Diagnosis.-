import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src import config
from src.data.eeg import extract_eeg_features, find_eeg_file
from src.data.behavioural import extract_tags_features, extract_game_features
from src.data.embrace import get_embrace

EEG_FEAT_NAMES = [
    "TBR_mean", "TBR_std", "alpha_asym", "theta_asym",
    "occ_alpha", "theta_entropy", "beta_entropy",
    "hjorth_mob", "hjorth_comp", "TAR",
    "delta_pow", "theta_pow", "alpha_pow", "beta_pow", "gamma_pow",
]


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
        if len(eeg_parts) >= 2:
            tbr_delta  = eeg_parts[-1][0] - eeg_parts[0][0]
            asym_delta = eeg_parts[-1][2] - eeg_parts[0][2]
        else:
            tbr_delta, asym_delta = 0.0, 0.0
        row = np.concatenate([
            np.array([meta["age"], meta["gender"]]),
            eeg_avg,
            np.array([tbr_delta, asym_delta]),
            get_embrace(uid, embrace_lookup, embrace_cols),
            extract_tags_features(uid),
            extract_game_features(uid),
        ])
        X_rows.append(row)
        y_rows.append(meta["label"])
        uid_list.append(uid)

    X_raw = np.array(X_rows, dtype=float)
    y     = np.array(y_rows)

    for col in range(X_raw.shape[1]):
        m = np.isnan(X_raw[:, col])
        if m.any():
            med = np.nanmedian(X_raw[~m, col]) if (~m).any() else 0.0
            X_raw[m, col] = med

    var_mask = X_raw.std(axis=0) > 1e-10
    X_raw    = X_raw[:, var_mask]

    base_names = (
        ["age", "gender"]
        + EEG_FEAT_NAMES
        + ["TBR_delta", "alpha_asym_delta"]
        + [f"embrace_{c}" for c in embrace_cols]
        + ["mean_RT", "RT_CV", "omission_rate"]
        + ["game_velocity", "game_omissions", "game_commissions"]
    )
    feat_names = [base_names[i] for i in range(len(base_names))
                  if i < len(var_mask) and var_mask[i]]

    print(f"  Final dataset: n={len(y)}, ADHD={int(y.sum())}, Ctrl={int((y == 0).sum())}")
    print(f"  Feature matrix: {X_raw.shape} (same for quantum and classical)")
    return X_raw, y, uid_list, feat_names


def preprocess(X_raw, y):
    idx_all = np.arange(len(y))
    idx_tr, idx_te = train_test_split(
        idx_all, test_size=config.TEST_SIZE, random_state=42, stratify=y)
    y_tr = y[idx_tr]; y_te = y[idx_te]
    X_tr_raw = X_raw[idx_tr]; X_te_raw = X_raw[idx_te]

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

    ss_cls   = StandardScaler().fit(X_tr_raw)
    X_tr_cls = ss_cls.transform(X_tr_raw)
    X_te_cls = ss_cls.transform(X_te_raw)

    print(f"  Train: {len(y_tr)} subjects | Test: {len(y_te)} subjects")
    print(f"  Quantum input : {X_tr_q.shape[1]}d (PCA)")
    print(f"  Classical input: {X_tr_cls.shape[1]}d")
    print(f"  Train classes  : ADHD={int(y_tr.sum())}  Ctrl={int((y_tr == 0).sum())}")
    print(f"  Test  classes  : ADHD={int(y_te.sum())}  Ctrl={int((y_te == 0).sum())}")
    print("  *** No SMOTE — using class_weight='balanced' instead ***")

    return dict(
        X_tr_raw=X_tr_raw, X_te_raw=X_te_raw,
        X_tr_q=X_tr_q,     X_te_q=X_te_q,
        X_tr_cls=X_tr_cls, X_te_cls=X_te_cls,
        y_tr=y_tr,         y_te=y_te,
        ss_q=ss_q,         pca=pca,  mm=mm,
    )
