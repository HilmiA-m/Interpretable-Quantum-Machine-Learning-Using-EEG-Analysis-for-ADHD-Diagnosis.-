# ======================================================================
# BALLADEER Quantum v40 — curated features + no SMOTE
# ======================================================================
# Key changes vs v39:
#
# [Features]
#   - CURATED feature set: 15 domain-relevant EEG features per level
#     instead of 50 (dropped 29 per-channel TBRs + redundant band stds).
#   - AVERAGE across 3 slackline levels instead of concatenating.
#     Plus 2 delta features (TBR change, alpha-asym change Lvl1→Lvl11).
#   - Behavioural features averaged across levels (3 features, not 15).
#   - Game features reduced to 3 key metrics.
#   - Embrace+ reduced to EDA + HR means only.
#   - Total: ~25 features (same for quantum AND classical paths).
#     This matches the sample size (127 subjects) far better than
#     261/361 features.
#
# [Pipeline]
#   - SMOTE REMOVED — class_weight="balanced" used instead.
#     SMOTE on ~50 ADHD subjects was creating non-physiological
#     synthetic samples that hurt generalization.
#   - MI selection REMOVED — not needed with curated features.
#   - Both quantum and classical models use the SAME feature set.
#
# [Models unchanged]
#   - VQC:  8q × 5L, full CZ, 249 params, subprocess restarts
#   - QSVM: ZZ/IQP 2-rep kernel, C grid search, Nyström option
#   - QCNN: 8q, 70 params
#   - SVM / RF / XGBoost baselines (class_weight=balanced)
# ======================================================================

import os, sys, json, glob, time, warnings, random, subprocess
os.environ["TF_METAL_DISABLE"]     = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import mne
import pennylane as qml
from pennylane import numpy as pnp

from sklearn.preprocessing     import StandardScaler, MinMaxScaler
from sklearn.decomposition     import PCA
from sklearn.model_selection   import (StratifiedKFold, train_test_split,
                                       GridSearchCV, cross_val_predict)
from sklearn.svm               import SVC
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (accuracy_score, roc_auc_score,
                                       average_precision_score,
                                       precision_score, recall_score,
                                       f1_score, fbeta_score,
                                       confusion_matrix, roc_curve,
                                       precision_recall_curve)
from xgboost                   import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42); random.seed(42)

# ─────────────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = "/Users/mah/Desktop/BALLADEER_Quantum/BALLADEER ADHD DATASET"
DEMO_JSON  = os.path.join(DATA_ROOT, "users_demographics.json")
RUNNER     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vqc_subprocess_runner.py")

N_QUBITS  = 8
N_LAYERS  = 5
N_PCA     = N_QUBITS

# VQC
N_RESTARTS       = 12
EPOCHS           = 400
LR_INIT          = 0.04
LR_MIN           = 0.001
WARMUP           = 20
BATCH_SIZE        = 32
FOCAL_ALPHA      = 0.70
FOCAL_GAMMA      = 2.0
L2_REG           = 0.001
PATIENCE         = 10
MIN_DELTA        = 0.004
VAL_EVERY        = 15
ENSEMBLE_K       = 4
MAX_RESTART_SECS = 600

# QCNN
N_QCNN_QUBITS  = 8
N_QCNN_PARAMS  = 70
QCNN_RESTARTS  = 16
QCNN_TOP_K     = 4
QCNN_EPOCHS    = 300
QCNN_LR        = 0.03
QCNN_LR_MIN    = 0.003
QCNN_VAL_EVERY = 20
QCNN_PATIENCE  = 8
QCNN_MIN_DELTA = 0.002

# QSVM
QSVM_C_GRID         = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1,
                       1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
QSVM_KERNEL_CENTER  = True       # subtract mean — usually helps SVC
NYSTROEM_TRIGGER    = 120        # use full kernel if n_tr ≤ 120, else Nyström
NYSTROEM_M          = 40         # # of landmark points

TEST_SIZE = 0.20
FBETA     = 1.0
BOOT_N    = 1000

SLACKLINE_LEVELS = ["SlacklineLvl1", "SlacklineLvl6", "SlacklineLvl11"]
BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,45)}
CGX_CH = ["AF7","Fpz","F7","Fz","T7","FC6","Fp1","F4","C4","Oz",
          "CP6","Cz","PO8","CP5","O2","O1","P3","P4","P7","P8",
          "Pz","PO7","T8","C3","Fp2","F3","F8","FC5","AF8"]
N_CGX = len(CGX_CH)
LEFT_FRONTAL  = [CGX_CH.index(c) for c in ["AF7","F7","Fp1","F3"]]
RIGHT_FRONTAL = [CGX_CH.index(c) for c in ["AF8","F8","Fp2","F4"]]
OCCIPITAL     = [CGX_CH.index(c) for c in ["Oz","O1","O2","PO7","PO8"]]

print("=" * 72)
print("BALLADEER — Quantum ML for ADHD  (v40: curated features + no SMOTE)")
print("=" * 72)

# ─── runner sanity check ──────────────────────────────────────────────
if not os.path.exists(RUNNER):
    sys.exit(
        f"\n[FATAL] VQC runner not found: {RUNNER}\n"
        f"        Without it every VQC restart will return random params.\n"
        f"        Make sure _vqc_subprocess_runner.py is in DATA_ROOT.\n"
    )
print(f"[runner] {RUNNER} OK")

# ─────────────────────────────────────────────────────────────────────
# 1. DEMOGRAPHICS
# ─────────────────────────────────────────────────────────────────────
with open(DEMO_JSON) as f:
    demo = json.load(f)
user_meta = {}
for e in demo:
    diag = e.get("diagnosed", "")
    if diag not in ("yes", "no"):
        continue
    user_meta[e["user"]] = {"label": 1 if diag == "yes" else 0,
                            "age":    float(e.get("age", 12)),
                            "gender": float(e.get("gender", 1))}
n_adhd = sum(v["label"] for v in user_meta.values())
print(f"\n[1] Users: {len(user_meta)}  ADHD={n_adhd}  Control={len(user_meta)-n_adhd}")

# ─────────────────────────────────────────────────────────────────────
# 2. EEG FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────
def _infer_fs(df):
    for col in ("timestamps", "Timestamp"):
        if col in df.columns:
            ts = df[col].values.astype(float)
            d  = np.diff(ts); d = d[(d > 0) & (d < 1.0)]
            if len(d):
                fs = int(round(1.0 / np.median(d)))
                if 64 <= fs <= 2000:
                    return fs
    return 500


def _hjorth(sig):
    var0 = np.var(sig) + 1e-12
    d1   = np.diff(sig); var1 = np.var(d1) + 1e-12
    d2   = np.diff(d1);  var2 = np.var(d2) + 1e-12
    mob  = np.sqrt(var1 / var0)
    return mob, np.sqrt(var2 / var1) / (mob + 1e-12)


def extract_eeg_features(csv_path):
    """Curated EEG feature extractor — 15 domain-relevant features per level.

    Returns: np.array of shape (15,) or None on failure.

    Features (each is a single scalar, not per-channel):
      0: TBR mean       — frontal theta/beta ratio (classic ADHD biomarker)
      1: TBR std        — variability across channels
      2: alpha_asym     — log(right/left frontal alpha power)
      3: theta_asym     — log(right/left frontal theta power)
      4: occ_alpha      — occipital alpha mean power
      5: theta_entropy  — spectral entropy in theta band
      6: beta_entropy   — spectral entropy in beta band
      7: hjorth_mob     — frontal Hjorth mobility
      8: hjorth_comp    — frontal Hjorth complexity
      9: tar            — theta/alpha ratio
     10: delta_power    — mean delta band power
     11: theta_power    — mean theta band power
     12: alpha_power    — mean alpha band power
     13: beta_power     — mean beta band power
     14: gamma_power    — mean gamma band power
    """
    try:
        df = pd.read_csv(csv_path)
        SKIP    = {"timestamps","Counter","Interpolated","Battery","Marker",
                   "Packet","TRIGGER","Packet Counter"}
        SKIP_KW = ("CQ.","EQ.","MOT.","PM.","POW.","ACC","ExG","A1","A2")
        eeg_cols = [c for c in df.columns
                    if c not in SKIP and not any(k in c for k in SKIP_KW)
                    and df[c].dtype in (float, "float64", "int64", "float32")]
        if not eeg_cols:
            return None
        data  = df[eeg_cols].values.T.astype(float)
        valid = np.array([data[i].std() > 1e-6 for i in range(data.shape[0])])
        if valid.sum() == 0:
            return None
        data_v = data[valid]; fs = _infer_fs(df)
        info = mne.create_info(data_v.shape[0], fs, ch_types="eeg")
        raw  = mne.io.RawArray(data_v, info, verbose=False)
        raw.filter(1, 45, fir_design="firwin", verbose=False)
        n_fft = max(64, min(fs * 4, data_v.shape[1] // 2))
        psd_o = raw.compute_psd(method="welch", fmin=1, fmax=45,
                                n_fft=n_fft, verbose=False)
        freqs = psd_o.freqs; P = psd_o.get_data()
        bp    = {b: P[:, (freqs >= lo) & (freqs < hi)].mean(axis=1)
                 for b, (lo, hi) in BANDS.items()}

        def pad29(a):
            a = a[:N_CGX] if len(a) >= N_CGX else np.pad(a, (0, N_CGX - len(a)),
                                                         constant_values=np.nan)
            return a.astype(float)

        def safe_r(a29, idx):
            v = a29[idx]
            return float(np.nanmean(v)) if not np.all(np.isnan(v)) else 0.0

        # TBR across channels
        tbr_ch = pad29(bp["theta"] / (bp["beta"] + 1e-12))
        tbr_v  = tbr_ch[~np.isnan(tbr_ch)]
        tbr_mean = float(tbr_v.mean()) if len(tbr_v) else 0.0
        tbr_std  = float(tbr_v.std())  if len(tbr_v) else 0.0

        # Frontal asymmetries
        a29 = pad29(bp["alpha"]); t29 = pad29(bp["theta"])
        lfa = safe_r(a29, LEFT_FRONTAL);  rfa = safe_r(a29, RIGHT_FRONTAL)
        lft = safe_r(t29, LEFT_FRONTAL);  rft = safe_r(t29, RIGHT_FRONTAL)
        alpha_asym = float(np.log(rfa / (lfa + 1e-12) + 1e-12))
        theta_asym = float(np.log(rft / (lft + 1e-12) + 1e-12))

        # Occipital alpha
        occ_alpha = safe_r(a29, OCCIPITAL)

        # Spectral entropy
        def sp_ent(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            p = P[:, mask].mean(axis=0)
            p = p / (p.sum() + 1e-12)
            return float(-np.sum(p * np.log(p + 1e-12)))

        # Hjorth parameters (frontal channels)
        ra = raw.get_data()
        fi = [i for i in LEFT_FRONTAL + RIGHT_FRONTAL if i < ra.shape[0]]
        mobs, comps = [], []
        for i in fi:
            if i < ra.shape[0]:
                m, c = _hjorth(ra[i])
                mobs.append(m); comps.append(c)
        hm = float(np.mean(mobs)) if mobs else 0.0
        hc = float(np.mean(comps)) if comps else 0.0

        # Theta/alpha ratio
        tar = float((bp["theta"] / (bp["alpha"] + 1e-12)).mean())

        feats = np.array([
            tbr_mean, tbr_std,
            alpha_asym, theta_asym,
            occ_alpha,
            sp_ent(4, 8), sp_ent(13, 30),
            hm, hc, tar,
            bp["delta"].mean(), bp["theta"].mean(),
            bp["alpha"].mean(), bp["beta"].mean(),
            bp["gamma"].mean(),
        ], dtype=float)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None


def find_eeg_file(uid, lvl):
    for pat in [f"{DATA_ROOT}/{uid}/{lvl}/*/{uid}_EEG_CGX_*.csv",
                f"{DATA_ROOT}/{uid}/{lvl}*/*/{uid}_EEG_CGX_*.csv"]:
        f = glob.glob(pat)
        if f:
            return f[0]
    return None


_probe = None
for uid in list(user_meta)[:10]:
    p = find_eeg_file(uid, "SlacklineLvl1")
    if p:
        _probe = extract_eeg_features(p)
        if _probe is not None:
            break
N_FEATS_PER_ACT = len(_probe) if _probe is not None else 15
print(f"\n[2] EEG: {N_FEATS_PER_ACT} curated feats/activity (averaged across 3 levels)")

# ─────────────────────────────────────────────────────────────────────
# 3. BEHAVIOURAL FEATURES
# ─────────────────────────────────────────────────────────────────────
print("[3] Extracting behavioural features (TAGS + GAME_DATA)...")

def extract_tags_features(uid):
    """Behavioural TAGS: mean RT, RT variability (CV), omission rate.
    Averaged across the 3 slackline levels → 3 features."""
    rts_all, omissions_all, n_totals = [], [], []
    for lvl in SLACKLINE_LEVELS:
        tfiles = glob.glob(f"{DATA_ROOT}/{uid}/{lvl}/*/*_TAGS_*.csv")
        if not tfiles:
            continue
        df = pd.read_csv(tfiles[0])
        rts, omissions, commissions = [], 0, 0
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
                correct = str(r.get("correct", "false")).lower() == "true"
                if reacted:
                    rt = float(r.get("reactionTime", 0))
                    if 0 < rt < 10:
                        rts.append(rt)
                    if not correct:
                        commissions += 1
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
    """Game data: mean work velocity, total omissions, total commissions → 3 features."""
    gfiles = glob.glob(f"{DATA_ROOT}/{uid}/AttentionRobotsDesktop/*/*_GAME_DATA*.csv")
    if not gfiles:
        return np.zeros(3)
    try:
        df = pd.read_csv(gfiles[0]); row = df.iloc[0]
        vel  = [float(row.get(f"velocidadTrabajoBloque{i}", 0)) for i in range(1, 5)]
        omis = [float(row.get(f"omisionBloque{i}", 0))         for i in range(1, 5)]
        coms = [float(row.get(f"comisionBloque{i}", 0))        for i in range(1, 5)]
        return np.array([float(np.mean(vel)), sum(omis), sum(coms)], dtype=float)
    except Exception:
        return np.zeros(3)

# ─────────────────────────────────────────────────────────────────────
# 4. EMBRACE+ FEATURES (reduced: EDA + HR only)
# ─────────────────────────────────────────────────────────────────────
embrace_lookup = {}; EMBRACE_COLS = []
try:
    ep_df = pd.read_csv(os.path.join(DATA_ROOT, "balladeer_embraceplus_data.csv"),
                        sep=";")
    # Select only EDA and heart rate means — the most ADHD-relevant autonomic signals
    EMBRACE_COLS = [c for c in ep_df.columns if c != "username"
                    and "wearing" not in c
                    and any(kw in c.lower() for kw in ("eda_mean", "hr_mean",
                                                       "mean_bpm"))]
    # Fallback: if no specific match, take up to 3 mean columns
    if not EMBRACE_COLS:
        EMBRACE_COLS = [c for c in ep_df.columns if c != "username"
                        and any(c.endswith(s) for s in ("_mean", "_mean_bpm"))
                        and "wearing" not in c][:3]
    embrace_lookup = {r["username"]: r for _, r in ep_df.iterrows()}
    print(f"[4] Embrace+: {len(EMBRACE_COLS)} selected cols ({EMBRACE_COLS}), "
          f"{len(embrace_lookup)} subjects")
except Exception as e:
    print(f"[4] Embrace+ skipped: {e}")


def get_embrace(uid):
    row = embrace_lookup.get(uid)
    if row is None:
        return np.full(len(EMBRACE_COLS), np.nan)
    return np.array([float(row.get(c, np.nan)) for c in EMBRACE_COLS], dtype=float)

# ─────────────────────────────────────────────────────────────────────
# 5. BUILD FEATURE MATRICES (curated — averaged across levels)
# ─────────────────────────────────────────────────────────────────────
print("\n[5] Building curated feature matrices (avg across levels)...")
X_rows, y_rows, uid_list = [], [], []

# Feature names for interpretability
EEG_FEAT_NAMES = ["TBR_mean", "TBR_std", "alpha_asym", "theta_asym",
                  "occ_alpha", "theta_entropy", "beta_entropy",
                  "hjorth_mob", "hjorth_comp", "TAR",
                  "delta_pow", "theta_pow", "alpha_pow", "beta_pow", "gamma_pow"]

for uid, meta in user_meta.items():
    eeg_parts = []
    for lvl in SLACKLINE_LEVELS:
        p = find_eeg_file(uid, lvl)
        feats = extract_eeg_features(p) if p else None
        if feats is not None:
            eeg_parts.append(feats)
    if len(eeg_parts) == 0:
        continue

    # AVERAGE EEG features across available levels (not concatenate)
    eeg_avg = np.mean(eeg_parts, axis=0)

    # Delta features: change from first to last available level
    if len(eeg_parts) >= 2:
        tbr_delta   = eeg_parts[-1][0] - eeg_parts[0][0]   # TBR mean change
        asym_delta  = eeg_parts[-1][2] - eeg_parts[0][2]    # alpha asym change
    else:
        tbr_delta, asym_delta = 0.0, 0.0

    row = np.concatenate([
        np.array([meta["age"], meta["gender"]]),
        eeg_avg,
        np.array([tbr_delta, asym_delta]),
        get_embrace(uid),
        extract_tags_features(uid),
        extract_game_features(uid),
    ])
    X_rows.append(row)
    y_rows.append(meta["label"]); uid_list.append(uid)


def _fill_nan(X):
    for col in range(X.shape[1]):
        m = np.isnan(X[:, col])
        if m.any():
            med = np.nanmedian(X[~m, col]) if (~m).any() else 0.0
            X[m, col] = med
    return X


X_raw = _fill_nan(np.array(X_rows, dtype=float))
y     = np.array(y_rows)

# Remove zero-variance features (uninformative)
var_mask = X_raw.std(axis=0) > 1e-10
X_raw    = X_raw[:, var_mask]

print(f"  Final dataset: n={len(y)}, ADHD={int(y.sum())}, Ctrl={int((y==0).sum())}")
print(f"  Feature matrix: {X_raw.shape} (same for quantum and classical)")

# Build feature name list for interpretability
FEAT_NAMES = (["age", "gender"]
              + EEG_FEAT_NAMES
              + ["TBR_delta", "alpha_asym_delta"]
              + [f"embrace_{c}" for c in EMBRACE_COLS]
              + ["mean_RT", "RT_CV", "omission_rate"]
              + ["game_velocity", "game_omissions", "game_commissions"])
FEAT_NAMES = [FEAT_NAMES[i] for i in range(len(FEAT_NAMES)) if i < len(var_mask) and var_mask[i]]

# ─────────────────────────────────────────────────────────────────────
# 6. PREPROCESSING — split → scale → PCA (quantum path only)
#    No SMOTE. No MI selection. Features are already curated.
# ─────────────────────────────────────────────────────────────────────
print(f"\n[6] Split → StandardScale → PCA-{N_PCA} (quantum) / raw (classical)")

idx_all = np.arange(len(y))
idx_tr, idx_te = train_test_split(idx_all, test_size=TEST_SIZE,
                                  random_state=42, stratify=y)
y_tr = y[idx_tr]; y_te = y[idx_te]

# Train/test split (same raw features for both paths)
X_tr_raw = X_raw[idx_tr]
X_te_raw = X_raw[idx_te]

# === Quantum path: StandardScale → PCA → MinMaxScale [0, π] ===
ss_q = StandardScaler().fit(X_tr_raw)
X_tr_q_std = ss_q.transform(X_tr_raw)
X_te_q_std = ss_q.transform(X_te_raw)
pca = PCA(n_components=N_PCA, random_state=42).fit(X_tr_q_std)
X_tr_pca = pca.transform(X_tr_q_std)
X_te_pca = pca.transform(X_te_q_std)
print(f"  Q PCA var: {np.cumsum(pca.explained_variance_ratio_).round(3).tolist()}")
mm = MinMaxScaler((0, np.pi)).fit(X_tr_pca)
X_tr_q = mm.transform(X_tr_pca)
X_te_q = mm.transform(X_te_pca)

# === Classical path: StandardScale (no PCA) ===
ss_cls = StandardScaler().fit(X_tr_raw)
X_tr_cls = ss_cls.transform(X_tr_raw)
X_te_cls = ss_cls.transform(X_te_raw)

print(f"  Train: {X_tr_q.shape[0]} subjects | Test: {X_te_q.shape[0]} subjects")
print(f"  Quantum input : {X_tr_q.shape[1]}d (PCA)")
print(f"  Classical input: {X_tr_cls.shape[1]}d")
print(f"  Train classes  : ADHD={int(y_tr.sum())}  Ctrl={int((y_tr==0).sum())}")
print(f"  Test  classes  : ADHD={int(y_te.sum())}  Ctrl={int((y_te==0).sum())}")
print(f"  *** No SMOTE — using class_weight='balanced' instead ***")

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def best_f1_threshold(y_true, y_proba, beta=FBETA):
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        s = fbeta_score(y_true, (y_proba >= t).astype(int),
                        beta=beta, zero_division=0)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


def bootstrap_ci(y_true, y_pred, y_proba, n=BOOT_N, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y_true))
    acc_b, roc_b, pr_b, prec_b, rec_b, f1_b = [], [], [], [], [], []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        yt, yp, ypr = y_true[s], y_pred[s], y_proba[s]
        if len(np.unique(yt)) < 2:
            continue
        acc_b.append(accuracy_score(yt, yp))
        try: roc_b.append(roc_auc_score(yt, ypr))
        except: pass
        try: pr_b.append(average_precision_score(yt, ypr))
        except: pass
        prec_b.append(precision_score(yt, yp, zero_division=0))
        rec_b.append(recall_score(yt, yp, zero_division=0))
        f1_b.append(f1_score(yt, yp, zero_division=0))
    def ci(v):
        return ((float(np.mean(v)),
                 float(np.percentile(v, 2.5)),
                 float(np.percentile(v, 97.5)))
                if v else (0.0, 0.0, 0.0))
    return {k: ci(v) for k, v in
            zip(("acc", "roc_auc", "pr_auc", "prec", "rec", "f1"),
                (acc_b, roc_b, pr_b, prec_b, rec_b, f1_b))}


def report(name, y_true, y_proba, y_oof, p_oof):
    t      = best_f1_threshold(y_oof, p_oof)
    y_pred = (y_proba >= t).astype(int)
    ci     = bootstrap_ci(y_true, y_pred, y_proba)
    try:    pr_auc_raw = float(average_precision_score(y_true, y_proba))
    except: pr_auc_raw = float("nan")
    print(f"\n=== {name}  (F1 thr={t:.3f}  OOF-cal n={len(y_oof)}) ===")
    for k, (m, lo, hi) in ci.items():
        print(f"  {k.upper():8s}: {m:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  PR_AUC (raw): {pr_auc_raw:.3f}")
    return t, y_pred, ci

# ─────────────────────────────────────────────────────────────────────
# 7. VQC — subprocess per restart
# ─────────────────────────────────────────────────────────────────────
print(f"\n[7] VQC: {N_QUBITS}q × {N_LAYERS}L, full CZ, "
      f"{N_RESTARTS}R × {EPOCHS}ep")
try:
    dev_vqc = qml.device("lightning.qubit", wires=N_QUBITS)
    VQC_DIFF = "adjoint"
    print("    lightning.qubit (adjoint)")
except Exception:
    dev_vqc = qml.device("default.qubit", wires=N_QUBITS)
    VQC_DIFF = "backprop"
    print("    default.qubit (backprop)")

N_ENC    = N_LAYERS * N_QUBITS * 3
N_ROT    = N_ENC
N_LIN    = N_QUBITS
N_BIAS   = 1
N_PARAMS = N_ENC + N_ROT + N_LIN + N_BIAS
CZ_PAIRS = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)]


@qml.qnode(dev_vqc, interface="autograd", diff_method=VQC_DIFF)
def vqc_circuit(params, x):
    enc_s = params[:N_ENC].reshape(N_LAYERS, N_QUBITS, 3)
    rot_a = params[N_ENC:N_ENC + N_ROT].reshape(N_LAYERS, N_QUBITS, 3)
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(enc_s[l, q, 0] * x[q], wires=q)
            qml.RY(enc_s[l, q, 1] * x[q], wires=q)
            qml.RZ(enc_s[l, q, 2] * x[q], wires=q)
        for q in range(N_QUBITS):
            qml.Rot(rot_a[l, q, 0], rot_a[l, q, 1], rot_a[l, q, 2], wires=q)
        for (a, b) in CZ_PAIRS:
            qml.CZ(wires=[a, b])
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


def vqc_forward(params, x):
    w = params[N_ENC + N_ROT: N_ENC + N_ROT + N_LIN]
    b = params[-1]
    evs = vqc_circuit(params, x)
    return 1.0 / (1.0 + pnp.exp(-pnp.clip(b + pnp.sum(w * pnp.stack(evs)),
                                          -15, 15)))


def vqc_proba(params, X):
    return np.array([float(vqc_forward(params, x)) for x in X])


def init_params(seed, strat):
    pnp.random.seed(seed)
    if strat == "near_identity":
        enc = pnp.ones(N_ENC) + 0.05 * pnp.random.randn(N_ENC)
        rot = 0.1 * pnp.random.randn(N_ROT)
    elif strat == "small_enc":
        enc = 0.5 * pnp.ones(N_ENC) + 0.1 * pnp.random.randn(N_ENC)
        rot = 0.2 * pnp.random.randn(N_ROT)
    elif strat == "large_enc":
        enc = 1.5 * pnp.ones(N_ENC) + 0.1 * pnp.random.randn(N_ENC)
        rot = 0.1 * pnp.random.randn(N_ROT)
    elif strat == "random_rot":
        enc = pnp.ones(N_ENC) + 0.05 * pnp.random.randn(N_ENC)
        rot = 0.5 * pnp.random.randn(N_ROT)
    elif strat == "uniform":
        enc = pnp.array(pnp.random.uniform(0.5, 1.5, N_ENC))
        rot = pnp.array(pnp.random.uniform(-np.pi / 4, np.pi / 4, N_ROT))
    elif strat == "decay":
        enc = pnp.linspace(1.5, 0.5, N_ENC) + 0.05 * pnp.random.randn(N_ENC)
        rot = 0.15 * pnp.random.randn(N_ROT)
    elif strat == "warm":
        enc = 1.2 * pnp.ones(N_ENC) + 0.08 * pnp.random.randn(N_ENC)
        rot = 0.08 * pnp.random.randn(N_ROT)
    elif strat == "small_enc_v2":
        enc = 0.3 * pnp.ones(N_ENC) + 0.08 * pnp.random.randn(N_ENC)
        rot = 0.15 * pnp.random.randn(N_ROT)
    elif strat == "uniform_narrow":
        enc = pnp.array(pnp.random.uniform(0.7, 1.3, N_ENC))
        rot = pnp.array(pnp.random.uniform(-np.pi / 6, np.pi / 6, N_ROT))
    else:
        enc = pnp.ones(N_ENC) + 0.1 * pnp.random.randn(N_ENC)
        rot = 0.1 * pnp.random.randn(N_ROT)
    return pnp.array(
        pnp.concatenate([enc, rot,
                         0.1 * pnp.random.randn(N_LIN),
                         pnp.array([0.0])]),
        requires_grad=True,
    )


RESTART_PLAN = [
    (542,  "decay"),        (142,  "small_enc"),    (442,  "uniform"),
    (502,  "decay"),        (522,  "decay"),        (562,  "decay"),
    (582,  "decay"),        (602,  "decay"),        (642,  "decay"),
    (742,  "decay"),        (842,  "warm"),         (42,   "near_identity"),
    (242,  "large_enc"),    (342,  "random_rot"),   (942,  "small_enc_v2"),
    (1042, "uniform_narrow"), (1142, "warm"),       (1242, "small_enc"),
]

# VQC inner split (no SMOTE — use class_weight in the loss via FOCAL_ALPHA)
X_tr_in, X_val, y_tr_in, y_val = train_test_split(
    X_tr_raw, y_tr, test_size=0.25, random_state=42, stratify=y_tr)
X_tr_is_q = mm.transform(pca.transform(ss_q.transform(X_tr_in)))
X_val_q   = mm.transform(pca.transform(ss_q.transform(X_val)))

restart_records = []
HYPERPARAMS_VQC = dict(
    N_QUBITS=N_QUBITS, N_LAYERS=N_LAYERS, EPOCHS=EPOCHS,
    LR_INIT=LR_INIT, LR_MIN=LR_MIN, WARMUP=WARMUP, BATCH_SIZE=BATCH_SIZE,
    PATIENCE=PATIENCE, MIN_DELTA=MIN_DELTA, VAL_EVERY=VAL_EVERY,
    FOCAL_ALPHA=FOCAL_ALPHA, FOCAL_GAMMA=FOCAL_GAMMA, L2_REG=L2_REG,
)

print(f"    {N_RESTARTS} restarts × {EPOCHS} ep  "
      f"(subprocess hard-kill after {MAX_RESTART_SECS}s)")
for r in range(N_RESTARTS):
    seed, strat = RESTART_PLAN[r % len(RESTART_PLAN)]
    args_path   = f"/tmp/_vqc_args_{r}.npy"
    result_path = f"/tmp/_vqc_result_{r}.npy"
    args_dict   = dict(seed=seed, strat=strat,
                       X_tr=X_tr_is_q, y_tr=y_tr_in,
                       X_val=X_val_q, y_val=y_val, **HYPERPARAMS_VQC)
    np.save(args_path, args_dict)
    t0 = time.time()
    try:
        subprocess.run([sys.executable, RUNNER, args_path, result_path],
                       timeout=MAX_RESTART_SECS,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       cwd=DATA_ROOT)
        if os.path.exists(result_path):
            result = np.load(result_path, allow_pickle=True).item()
            best_p = pnp.array(result["params"], requires_grad=True)
            restart_records.append({"params": best_p,
                                    "val_auc": result["val_auc"],
                                    "threshold": result["threshold"],
                                    "n_epochs": result.get("n_epochs", -1)})
            print(f"    r{r+1:2d} [{strat:14s}]: val_AUC={result['val_auc']:.3f}  "
                  f"thr={result['threshold']:.3f}  ep={result.get('n_epochs',-1):3d}  "
                  f"{time.time()-t0:.0f}s")
        else:
            print(f"    r{r+1:2d} [{strat:14s}]: failed (no result)  "
                  f"{time.time()-t0:.0f}s")
            restart_records.append({"params": init_params(seed, strat),
                                    "val_auc": -1.0, "threshold": 0.5,
                                    "n_epochs": 0})
    except subprocess.TimeoutExpired:
        print(f"    r{r+1:2d} [{strat:14s}]: TIMEOUT {MAX_RESTART_SECS}s — killed")
        restart_records.append({"params": init_params(seed, strat),
                                "val_auc": -1.0, "threshold": 0.5,
                                "n_epochs": 0})
    except Exception as e:
        print(f"    r{r+1:2d} [{strat:14s}]: ERROR {e}")
        restart_records.append({"params": init_params(seed, strat),
                                "val_auc": -1.0, "threshold": 0.5,
                                "n_epochs": 0})
    for p in [args_path, result_path]:
        try: os.remove(p)
        except: pass

restart_records.sort(key=lambda d: d["val_auc"], reverse=True)
good_k = [r for r in restart_records if r["val_auc"] > 0.0]
top_k  = (good_k[:ENSEMBLE_K] if len(good_k) >= ENSEMBLE_K
          else (good_k if good_k else restart_records[:1]))
if good_k:
    aucs = [r["val_auc"] for r in good_k]
    print(f"    {len(good_k)}/{N_RESTARTS} succeeded — "
          f"val_AUC mean={np.mean(aucs):.3f}  max={max(aucs):.3f}  "
          f"min={min(aucs):.3f}")
if len(top_k) < ENSEMBLE_K:
    print(f"    WARNING: only {len(top_k)}/{ENSEMBLE_K} restarts succeeded")
w_k = np.exp(np.array([r["val_auc"] for r in top_k]) / 0.5)
w_k /= w_k.sum()

p_vqc_val = np.average(
    np.stack([vqc_proba(r["params"], X_val_q) for r in top_k]),
    axis=0, weights=w_k)
p_vqc = np.average(
    np.stack([vqc_proba(r["params"], X_te_q) for r in top_k]),
    axis=0, weights=w_k)
np.save(os.path.join(DATA_ROOT, "best_quantum_params_v40.npy"),
        np.array(top_k[0]["params"]))
t_vqc, y_pred_vqc, _ = report("VQC", y_te, p_vqc, y_val, p_vqc_val)

# ─────────────────────────────────────────────────────────────────────
# 8. QSVM — ZZ kernel with symmetry exploitation + centering + Nyström
# ─────────────────────────────────────────────────────────────────────
print(f"\n[8] QSVM: ZZ feature map (IQP, 2 reps, {N_QUBITS} qubits)")
dev_k = qml.device("default.qubit", wires=N_QUBITS)


def _zz_map(x, n_reps=2):
    for _ in range(n_reps):
        for q in range(N_QUBITS):
            qml.Hadamard(wires=q)
        for q in range(N_QUBITS):
            qml.RZ(2.0 * x[q], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
            qml.RZ(2.0 * (np.pi - x[q]) * (np.pi - x[(q + 1) % N_QUBITS]),
                   wires=(q + 1) % N_QUBITS)
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])


@qml.qnode(dev_k)
def zz_kernel_circuit(x1, x2):
    _zz_map(x1)
    qml.adjoint(_zz_map)(x2)
    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(A, B, label="kernel"):
    """Compute K[i,j] = |<φ(B[j])|φ(A[i])>|² with symmetry exploitation."""
    n_a, n_b = len(A), len(B)
    K = np.zeros((n_a, n_b))
    is_sym = (A is B)
    total = (n_a * (n_a + 1)) // 2 if is_sym else n_a * n_b
    done = 0
    t0 = time.time()
    for i in range(n_a):
        j_start = i if is_sym else 0
        for j in range(j_start, n_b):
            v = float(zz_kernel_circuit(A[i], B[j])[0])
            K[i, j] = v
            if is_sym and i != j:
                K[j, i] = v
            done += 1
            if done % 200 == 0:
                pct = 100 * done / total
                eta = (time.time() - t0) * (total - done) / max(done, 1)
                print(f"      {label}: {done}/{total} ({pct:4.1f}%)  "
                      f"ETA {eta:.0f}s", end="\r", flush=True)
    print(f"      {label}: {total}/{total} (100.0%) done in "
          f"{time.time()-t0:.0f}s            ")
    return K


def center_kernel(K_tr, K_te=None):
    """Center kernel matrix (subtract row/col/total means)."""
    col_means  = K_tr.mean(axis=0)
    row_means  = K_tr.mean(axis=1)
    total_mean = K_tr.mean()
    K_tr_c = (K_tr
              - col_means[np.newaxis, :]
              - row_means[:, np.newaxis]
              + total_mean)
    if K_te is None:
        return K_tr_c
    te_row_means = K_te.mean(axis=1)
    K_te_c = (K_te
              - te_row_means[:, np.newaxis]
              - col_means[np.newaxis, :]
              + total_mean)
    return K_tr_c, K_te_c


# Use same untransformed-then-quantum-encoded training points (no SMOTE)
X_tr_nosmote_q = mm.transform(pca.transform(ss_q.transform(X_tr_raw)))
n_tr_q = len(X_tr_nosmote_q)

if n_tr_q <= NYSTROEM_TRIGGER:
    print(f"    n_tr={n_tr_q} ≤ {NYSTROEM_TRIGGER} → full quantum kernel")
    K_tr = quantum_kernel(X_tr_nosmote_q, X_tr_nosmote_q, label="K_tr")
    K_te = quantum_kernel(X_te_q, X_tr_nosmote_q, label="K_te")
    use_nystroem = False
else:
    print(f"    n_tr={n_tr_q} > {NYSTROEM_TRIGGER} → Nyström (m={NYSTROEM_M})")
    rng = np.random.default_rng(42)
    landmark_idx = rng.choice(n_tr_q, NYSTROEM_M, replace=False)
    L = X_tr_nosmote_q[landmark_idx]
    K_mm = quantum_kernel(L, L, label="K_mm")
    K_mm += 1e-6 * np.eye(NYSTROEM_M)
    U, S, Vt = np.linalg.svd(K_mm)
    S_inv_sqrt = np.where(S > 1e-12, 1.0 / np.sqrt(S), 0.0)
    norm_mat = (U * S_inv_sqrt) @ Vt
    K_tr_nm = quantum_kernel(X_tr_nosmote_q, L, label="K_tr_nm")
    K_te_nm = quantum_kernel(X_te_q, L, label="K_te_nm")
    Phi_tr = K_tr_nm @ norm_mat.T
    Phi_te = K_te_nm @ norm_mat.T
    K_tr = Phi_tr @ Phi_tr.T
    K_te = Phi_te @ Phi_tr.T
    use_nystroem = True

if QSVM_KERNEL_CENTER:
    K_tr, K_te = center_kernel(K_tr, K_te)
    print("    kernel centered")

# CV C-search using precomputed kernel
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_C, best_cv = 1.0, -1.0
for C in QSVM_C_GRID:
    scores = []
    for tri, vai in skf.split(K_tr, y_tr):
        m = SVC(kernel="precomputed", C=C, probability=True, random_state=42)
        m.fit(K_tr[np.ix_(tri, tri)], y_tr[tri])
        pp = m.predict_proba(K_tr[np.ix_(vai, tri)])[:, 1]
        try:
            scores.append(roc_auc_score(y_tr[vai], pp))
        except Exception:
            pass
    mv = float(np.mean(scores)) if scores else -1.0
    if mv > best_cv:
        best_cv, best_C = mv, C
print(f"    Best C={best_C}  (CV AUC={best_cv:.3f})  "
      f"{'Nyström' if use_nystroem else 'full'}{' centered' if QSVM_KERNEL_CENTER else ''}")

qsvm = SVC(kernel="precomputed", C=best_C, probability=True, random_state=42)
qsvm.fit(K_tr, y_tr)
p_qsvm = qsvm.predict_proba(K_te)[:, 1]

print("    Computing QSVM OOF threshold calibration...")
qsvm_oof = np.zeros(len(y_tr))
for tri, vai in skf.split(K_tr, y_tr):
    m = SVC(kernel="precomputed", C=best_C, probability=True, random_state=42)
    m.fit(K_tr[np.ix_(tri, tri)], y_tr[tri])
    qsvm_oof[vai] = m.predict_proba(K_tr[np.ix_(vai, tri)])[:, 1]
t_qsvm, y_pred_qsvm, _ = report("QSVM-ZZ", y_te, p_qsvm, y_tr, qsvm_oof)

# ─────────────────────────────────────────────────────────────────────
# 9. QCNN-8
# ─────────────────────────────────────────────────────────────────────
print(f"\n[9] QCNN-8: {N_QCNN_QUBITS}q, {N_QCNN_PARAMS}p, "
      f"{QCNN_RESTARTS}R×{QCNN_EPOCHS}ep, top-{QCNN_TOP_K}")
dev_qcnn = qml.device("default.qubit", wires=N_QCNN_QUBITS)


def _conv(p, w):
    qml.U3(p[0], p[1], p[2], wires=w[0])
    qml.U3(p[3], p[4], p[5], wires=w[1])
    qml.CNOT(wires=w)
    qml.RY(p[6], wires=w[0])
    qml.RZ(p[7], wires=w[1])
    qml.CNOT(wires=[w[1], w[0]])


def _pool(p, w):
    qml.CRZ(p[0], wires=w)
    qml.PauliX(wires=w[0])
    qml.CRX(p[1], wires=w)
    qml.PauliX(wires=w[0])


@qml.qnode(dev_qcnn, interface="autograd", diff_method="backprop")
def qcnn_circuit(params, x):
    for q in range(N_QCNN_QUBITS):
        qml.RY(x[q], wires=q)
    _conv(params[0:8], [0, 1]);  _conv(params[8:16], [2, 3])
    _conv(params[16:24], [4, 5]); _conv(params[24:32], [6, 7])
    _pool(params[32:34], [0, 1]); _pool(params[34:36], [2, 3])
    _pool(params[36:38], [4, 5]); _pool(params[38:40], [6, 7])
    _conv(params[40:48], [0, 2]); _conv(params[48:56], [4, 6])
    _pool(params[56:58], [0, 2]); _pool(params[58:60], [4, 6])
    _conv(params[60:68], [0, 4])
    _pool(params[68:70], [0, 4])
    return qml.expval(qml.PauliZ(0))


def qcnn_proba(p, X):
    return (np.array([float(qcnn_circuit(p, x)) for x in X]) + 1) / 2


def qcnn_bce(params, Xb, yb):
    n_pos = max(sum(1 for y_ in yb if y_ == 1), 1)
    n_neg = max(len(yb) - n_pos, 1)
    w_pos = n_neg / n_pos
    tot = pnp.array(0.0, requires_grad=True)
    for xi, yi in zip(Xb, yb):
        raw = qcnn_circuit(params, xi)
        p   = pnp.clip((raw + 1) / 2, 1e-7, 1 - 1e-7)
        pt  = p if yi == 1 else (1 - p)
        w   = w_pos if yi == 1 else 1.0
        tot = tot - w * pnp.log(pt)
    return tot / len(Xb) + 0.0005 * pnp.sum(params ** 2)


X_qcnn_in_raw, X_qcnn_val_raw, y_qcnn_in, y_qcnn_val = train_test_split(
    X_tr_raw, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
X_qcnn_in  = mm.transform(pca.transform(ss_q.transform(X_qcnn_in_raw)))
X_qcnn_val = mm.transform(pca.transform(ss_q.transform(X_qcnn_val_raw)))


def qcnn_lr(ep):
    t = ep / max(1, QCNN_EPOCHS - 1)
    return QCNN_LR_MIN + 0.5 * (QCNN_LR - QCNN_LR_MIN) * (1 + np.cos(np.pi * t))


qcnn_records = []
for r in range(QCNN_RESTARTS):
    pnp.random.seed(300 + r * 77)
    p = pnp.array(0.1 * pnp.random.randn(N_QCNN_PARAMS), requires_grad=True)
    opt = qml.AdamOptimizer(QCNN_LR)
    best_p_r, best_auc_r, patience_r = p.copy(), -1.0, 0
    for ep in range(QCNN_EPOCHS):
        opt.stepsize = float(qcnn_lr(ep))
        idx = np.random.choice(len(X_qcnn_in),
                               min(BATCH_SIZE, len(X_qcnn_in)), replace=False)
        Xb, yb = X_qcnn_in[idx], y_qcnn_in[idx]
        p, _ = opt.step_and_cost(
            lambda pp, _X=Xb, _y=yb: qcnn_bce(pp, _X, _y), p)
        if (ep + 1) % QCNN_VAL_EVERY == 0:
            vp = qcnn_proba(p, X_qcnn_val)
            try:    v_auc = roc_auc_score(y_qcnn_val, vp)
            except: v_auc = 0.5
            if v_auc > best_auc_r + QCNN_MIN_DELTA:
                best_auc_r, best_p_r, patience_r = v_auc, p.copy(), 0
            else:
                patience_r += 1
                if patience_r >= QCNN_PATIENCE:
                    break
    qcnn_records.append({"params": best_p_r, "val_auc": best_auc_r})
    print(f"    r{r+1:2d}/{QCNN_RESTARTS} val_AUC={best_auc_r:.3f}")

qcnn_records.sort(key=lambda d: d["val_auc"], reverse=True)
top_qcnn = qcnn_records[:QCNN_TOP_K]
w_qcnn = np.exp(np.array([r["val_auc"] for r in top_qcnn]) / 0.5)
w_qcnn /= w_qcnn.sum()
p_qcnn = np.average(
    np.stack([qcnn_proba(r["params"], X_te_q) for r in top_qcnn]),
    axis=0, weights=w_qcnn)
p_qcnn_val = np.average(
    np.stack([qcnn_proba(r["params"], X_qcnn_val) for r in top_qcnn]),
    axis=0, weights=w_qcnn)
t_qcnn, y_pred_qcnn, _ = report("QCNN-8", y_te, p_qcnn, y_qcnn_val, p_qcnn_val)

# ─────────────────────────────────────────────────────────────────────
# 10. CLASSICAL BASELINES (no SMOTE — class_weight balanced)
# ─────────────────────────────────────────────────────────────────────
print("\n[10] Classical baselines (5-fold CV, roc_auc, class_weight=balanced)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# SVM
svm_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            "kernel": ["rbf", "poly"]}
svm = GridSearchCV(
    SVC(probability=True, class_weight="balanced", random_state=42),
    svm_grid, cv=cv, n_jobs=-1, scoring="roc_auc"
).fit(X_tr_cls, y_tr)
p_svm = svm.predict_proba(X_te_cls)[:, 1]
print(f"  SVM: {svm.best_params_}  CV={svm.best_score_:.3f}")
svm_oof_clf = SVC(probability=True, class_weight="balanced",
                  random_state=42, **svm.best_params_)
p_svm_oof = cross_val_predict(svm_oof_clf, X_tr_cls, y_tr,
                              cv=5, method="predict_proba")[:, 1]
t_svm, y_pred_svm, _ = report("SVM", y_te, p_svm, y_tr, p_svm_oof)

# RF (same features — no separate MI selection needed)
rf_grid = {"n_estimators": [500, 700, 1000, 1500],
           "max_depth": [None, 10, 15, 20],
           "min_samples_leaf": [1, 2],
           "max_features": ["sqrt", "log2"]}
rf = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    rf_grid, cv=cv, n_jobs=-1, scoring="roc_auc"
).fit(X_tr_cls, y_tr)
p_rf = rf.predict_proba(X_te_cls)[:, 1]
print(f"  RF:  {rf.best_params_}  CV={rf.best_score_:.3f}")
rf_oof_clf = RandomForestClassifier(class_weight="balanced",
                                    random_state=42, **rf.best_params_)
p_rf_oof = cross_val_predict(rf_oof_clf, X_tr_cls, y_tr,
                             cv=5, method="predict_proba")[:, 1]
t_rf, y_pred_rf, _ = report("RF", y_te, p_rf, y_tr, p_rf_oof)

# XGBoost
n_neg_tr = int((y_tr == 0).sum()); n_pos_tr = int(y_tr.sum())
spw = n_neg_tr / n_pos_tr
xgb_grid = {"n_estimators": [500, 700, 1000], "max_depth": [3, 4, 5],
            "learning_rate": [0.02, 0.05, 0.1], "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.8], "min_child_weight": [1, 3],
            "gamma": [0.1, 0.5], "reg_alpha": [0.1, 1.0], "reg_lambda": [1.0]}
xgb = GridSearchCV(
    XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                  random_state=42, n_jobs=1),
    xgb_grid, cv=cv, n_jobs=-1, scoring="roc_auc"
).fit(X_tr_cls, y_tr)
p_xgb = xgb.predict_proba(X_te_cls)[:, 1]
print(f"  XGB: {xgb.best_params_}  CV={xgb.best_score_:.3f}")
xgb_oof_clf = XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                            random_state=42, n_jobs=1, **xgb.best_params_)
p_xgb_oof = cross_val_predict(xgb_oof_clf, X_tr_cls, y_tr,
                              cv=5, method="predict_proba")[:, 1]
t_xgb, y_pred_xgb, _ = report("XGBoost", y_te, p_xgb, y_tr, p_xgb_oof)

# ─── Feature importance (XGBoost) for interpretability ─────────────
print("\n[10b] Feature importance (XGBoost)...")
xgb_final = XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                           random_state=42, n_jobs=1, **xgb.best_params_)
xgb_final.fit(X_tr_cls, y_tr)
importances = xgb_final.feature_importances_
n_feats_plot = min(len(FEAT_NAMES), len(importances))
feat_imp = sorted(zip(FEAT_NAMES[:n_feats_plot], importances[:n_feats_plot]),
                  key=lambda x: x[1], reverse=True)
print("  Top features:")
for fn, fi in feat_imp[:10]:
    print(f"    {fn:25s} {fi:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
names_sorted = [x[0] for x in feat_imp[:15]]
vals_sorted  = [x[1] for x in feat_imp[:15]]
ax.barh(range(len(names_sorted)), vals_sorted, color="steelblue", alpha=0.85)
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(names_sorted, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("XGBoost Feature Importance", fontsize=11)
ax.set_title("Top 15 Features — XGBoost (v40 curated)", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(DATA_ROOT, "feature_importance_v40.png"), dpi=150)
plt.close(fig)
print("  [OK] feature_importance_v40.png")

# ─────────────────────────────────────────────────────────────────────
# 11. RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────
def mrow(yt, yp, ypr):
    try:    roc = roc_auc_score(yt, ypr)
    except: roc = float("nan")
    try:    pr = average_precision_score(yt, ypr)
    except: pr = float("nan")
    return dict(acc=accuracy_score(yt, yp),
                prec=precision_score(yt, yp, zero_division=0),
                rec=recall_score(yt, yp, zero_division=0),
                f1=f1_score(yt, yp, zero_division=0),
                roc_auc=roc, pr_auc=pr)


results = {"VQC":     mrow(y_te, y_pred_vqc,  p_vqc),
           "QSVM-ZZ": mrow(y_te, y_pred_qsvm, p_qsvm),
           "QCNN-8":  mrow(y_te, y_pred_qcnn, p_qcnn),
           "SVM":     mrow(y_te, y_pred_svm,  p_svm),
           "RF":      mrow(y_te, y_pred_rf,   p_rf),
           "XGBoost": mrow(y_te, y_pred_xgb,  p_xgb)}

print("\n" + "=" * 90)
print(f"  {'Model':<14}  {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} "
      f"{'ROC-AUC':>9} {'PR-AUC':>8}")
print("  " + "─" * 78)
for name, m in results.items():
    flag = " ◄" if m['acc'] >= 0.80 else ""
    print(f"  {name:<14}  {m['acc']:>7.3f} {m['prec']:>7.3f} {m['rec']:>7.3f} "
          f"{m['f1']:>7.3f} {m['roc_auc']:>9.3f} {m['pr_auc']:>8.3f}{flag}")
print("=" * 90)
best_roc_m = max(results, key=lambda k: results[k]["roc_auc"])
best_pr_m  = max(results, key=lambda k: results[k]["pr_auc"])
best_acc_m = max(results, key=lambda k: results[k]["acc"])
best_f1_m  = max(results, key=lambda k: results[k]["f1"])
print(f"\n  Best Acc    : {best_acc_m} = {results[best_acc_m]['acc']:.3f}")
print(f"  Best F1     : {best_f1_m}  = {results[best_f1_m]['f1']:.3f}")
print(f"  Best ROC-AUC: {best_roc_m} = {results[best_roc_m]['roc_auc']:.3f}")
print(f"  Best PR-AUC : {best_pr_m}  = {results[best_pr_m]['pr_auc']:.3f}")

# ─────────────────────────────────────────────────────────────────────
# 12. FIGURES
# ─────────────────────────────────────────────────────────────────────
COLORS = {"VQC":"royalblue","QSVM-ZZ":"indigo","QCNN-8":"mediumslateblue",
          "SVM":"darkorange","RF":"forestgreen","XGBoost":"crimson"}
PROBAS = {"VQC":p_vqc,"QSVM-ZZ":p_qsvm,"QCNN-8":p_qcnn,
          "SVM":p_svm,"RF":p_rf,"XGBoost":p_xgb}

fig, ax = plt.subplots(figsize=(9, 7))
for name, proba in PROBAS.items():
    fpr, tpr, _ = roc_curve(y_te, proba)
    try:    auc = roc_auc_score(y_te, proba)
    except: auc = float("nan")
    ls = "--" if "Q" in name else "-"
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
            color=COLORS[name], lw=1.8, ls=ls)
ax.plot([0, 1], [0, 1], "k--", alpha=.3)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC — BALLADEER ADHD v40")
ax.legend(fontsize=8.5); fig.tight_layout()
fig.savefig(os.path.join(DATA_ROOT, "roc_curves_v40.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 7))
for name, proba in PROBAS.items():
    prec_c, rec_c, _ = precision_recall_curve(y_te, proba)
    try:    ap = average_precision_score(y_te, proba)
    except: ap = float("nan")
    ls = "--" if "Q" in name else "-"
    ax.plot(rec_c, prec_c, label=f"{name} (AP={ap:.3f})",
            color=COLORS[name], lw=1.8, ls=ls)
baseline = y_te.mean()
ax.axhline(baseline, color="k", ls="--", alpha=.3, label=f"Baseline={baseline:.2f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision–Recall — BALLADEER ADHD v40")
ax.legend(fontsize=8.5); fig.tight_layout()
fig.savefig(os.path.join(DATA_ROOT, "pr_curves_v40.png"), dpi=150)
plt.close(fig)

PRED = {"VQC":(y_pred_vqc,p_vqc,t_vqc),"QSVM-ZZ":(y_pred_qsvm,p_qsvm,t_qsvm),
        "QCNN-8":(y_pred_qcnn,p_qcnn,t_qcnn),"SVM":(y_pred_svm,p_svm,t_svm),
        "RF":(y_pred_rf,p_rf,t_rf),"XGB":(y_pred_xgb,p_xgb,t_xgb)}
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, (yp, proba, thr)) in zip(axes.flatten(), PRED.items()):
    cm = confusion_matrix(y_te, yp)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Ctrl", "ADHD"], yticklabels=["Ctrl", "ADHD"])
    try:    auc = roc_auc_score(y_te, proba)
    except: auc = float("nan")
    ax.set_title(f"{name}  AUC={auc:.3f}  thr={thr:.2f}")
fig.suptitle("Confusion Matrices — BALLADEER ADHD v40", y=1.01, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(DATA_ROOT, "confusion_matrices_v40.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────
# 13. SUMMARY
# ─────────────────────────────────────────────────────────────────────
print(f"""
{'='*72}
BALLADEER v40 — COMPLETE (curated features, no SMOTE)
{'='*72}
Subjects    : {len(y)} | ADHD={int(y.sum())} Control={int((y==0).sum())}
Features    : {X_raw.shape[1]} curated (same for quantum + classical)
Q pipeline  : {X_raw.shape[1]} → PCA-{N_PCA} → [0,π]
CLS pipeline: {X_raw.shape[1]} → StdScale
Threshold   : F1-OOF (calibrated on val/OOF, never on test)
SMOTE       : REMOVED (class_weight=balanced instead)

VQC     : {N_QUBITS}q×{N_LAYERS}L, full CZ, {N_PARAMS}p, {N_RESTARTS}R×{EPOCHS}ep, top-{ENSEMBLE_K}
          runner: {RUNNER}
          {len(good_k)}/{N_RESTARTS} restarts succeeded
QSVM-ZZ : ZZ/IQP 2 reps, C={best_C}{' Nyström m='+str(NYSTROEM_M) if use_nystroem else ' full'}
          {'centered' if QSVM_KERNEL_CENTER else 'raw'} kernel
QCNN-8  : {N_QCNN_QUBITS}q, {N_QCNN_PARAMS}p, {QCNN_RESTARTS}R×{QCNN_EPOCHS}ep, top-{QCNN_TOP_K}
SVM     : class_weight=balanced, {svm.best_params_}
RF      : class_weight=balanced, {rf.best_params_}
XGBoost : scale_pos_weight={spw:.2f}, {xgb.best_params_}

Best Acc    : {best_acc_m} = {results[best_acc_m]['acc']:.3f}
Best F1     : {best_f1_m}  = {results[best_f1_m]['f1']:.3f}
Best ROC-AUC: {best_roc_m} = {results[best_roc_m]['roc_auc']:.3f}
Best PR-AUC : {best_pr_m}  = {results[best_pr_m]['pr_auc']:.3f}

Outputs: roc_curves_v40.png  pr_curves_v40.png
         confusion_matrices_v40.png  feature_importance_v40.png
         best_quantum_params_v40.npy
{'='*72}
""")
