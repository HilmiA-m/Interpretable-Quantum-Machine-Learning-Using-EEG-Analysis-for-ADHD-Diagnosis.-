import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, fbeta_score,
)
from src import config


def best_f1_threshold(y_true, y_proba, beta=config.FBETA):
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        s = fbeta_score(y_true, (y_proba >= t).astype(int), beta=beta, zero_division=0)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


def bootstrap_ci(y_true, y_pred, y_proba, n=config.BOOT_N, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    acc_b, roc_b, pr_b, prec_b, rec_b, f1_b = [], [], [], [], [], []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        yt, yp, ypr = y_true[s], y_pred[s], y_proba[s]
        if len(np.unique(yt)) < 2:
            continue
        acc_b.append(accuracy_score(yt, yp))
        try:    roc_b.append(roc_auc_score(yt, ypr))
        except: pass
        try:    pr_b.append(average_precision_score(yt, ypr))
        except: pass
        prec_b.append(precision_score(yt, yp, zero_division=0))
        rec_b.append(recall_score(yt, yp, zero_division=0))
        f1_b.append(f1_score(yt, yp, zero_division=0))

    def ci(v):
        return ((float(np.mean(v)),
                 float(np.percentile(v, 2.5)),
                 float(np.percentile(v, 97.5)))
                if v else (0.0, 0.0, 0.0))

    return {k: ci(v) for k, v in zip(
        ("acc", "roc_auc", "pr_auc", "prec", "rec", "f1"),
        (acc_b, roc_b, pr_b, prec_b, rec_b, f1_b),
    )}


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


def model_row(y_true, y_pred, y_proba):
    try:    roc = roc_auc_score(y_true, y_proba)
    except: roc = float("nan")
    try:    pr  = average_precision_score(y_true, y_proba)
    except: pr  = float("nan")
    return dict(
        acc=accuracy_score(y_true, y_pred),
        prec=precision_score(y_true, y_pred, zero_division=0),
        rec=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc,
        pr_auc=pr,
    )
