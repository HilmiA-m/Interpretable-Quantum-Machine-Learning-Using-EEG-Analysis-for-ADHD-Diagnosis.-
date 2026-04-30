import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)

from src import config

_COLORS = {
    "VQC":     "royalblue",
    "QSVM-ZZ": "indigo",
    "QCNN-8":  "mediumslateblue",
    "LR":      "peru",
    "SVM":     "darkorange",
    "RF":      "forestgreen",
    "XGBoost": "crimson",
}


def save_figures(y_te, probas, preds, thresholds, feat_imp, run_id="v40"):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    _save_roc(y_te, probas, run_id)
    _save_pr(y_te, probas, run_id)
    _save_confusion(y_te, preds, probas, thresholds, run_id)
    _save_feature_importance(feat_imp, run_id)


def _save_roc(y_te, probas, run_id):
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_te, proba)
        try:    auc = roc_auc_score(y_te, proba)
        except: auc = float("nan")
        ls = "--" if "Q" in name else "-"
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=_COLORS.get(name, "gray"), lw=1.8, ls=ls)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC — BALLADEER ADHD ({run_id})")
    ax.legend(fontsize=8.5); fig.tight_layout()
    fname = f"roc_curves_{run_id}.png"
    fig.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  [OK] RESULTS/{fname}")


def _save_pr(y_te, probas, run_id):
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, proba in probas.items():
        prec_c, rec_c, _ = precision_recall_curve(y_te, proba)
        try:    ap = average_precision_score(y_te, proba)
        except: ap = float("nan")
        ls = "--" if "Q" in name else "-"
        ax.plot(rec_c, prec_c, label=f"{name} (AP={ap:.3f})",
                color=_COLORS.get(name, "gray"), lw=1.8, ls=ls)
    baseline = y_te.mean()
    ax.axhline(baseline, color="k", ls="--", alpha=0.3, label=f"Baseline={baseline:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall — BALLADEER ADHD ({run_id})")
    ax.legend(fontsize=8.5); fig.tight_layout()
    fname = f"pr_curves_{run_id}.png"
    fig.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  [OK] RESULTS/{fname}")


def _save_confusion(y_te, preds, probas, thresholds, run_id):
    n = len(preds)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    ax_flat = np.array(axes).flatten()
    for ax, (name, yp) in zip(ax_flat, preds.items()):
        cm = confusion_matrix(y_te, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Ctrl", "ADHD"], yticklabels=["Ctrl", "ADHD"])
        try:    auc = roc_auc_score(y_te, probas[name])
        except: auc = float("nan")
        ax.set_title(f"{name}  AUC={auc:.3f}  thr={thresholds[name]:.2f}")
    for ax in ax_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Confusion Matrices — BALLADEER ADHD ({run_id})", y=1.01, fontsize=12)
    fig.tight_layout()
    fname = f"confusion_matrices_{run_id}.png"
    fig.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] RESULTS/{fname}")


def _save_feature_importance(feat_imp, run_id):
    names_sorted = [x[0] for x in feat_imp[:15]]
    vals_sorted  = [x[1] for x in feat_imp[:15]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(names_sorted)), vals_sorted, color="steelblue", alpha=0.85)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("XGBoost Feature Importance", fontsize=11)
    ax.set_title(f"Top 15 Features — XGBoost ({run_id})", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fname = f"feature_importance_{run_id}.png"
    fig.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  [OK] RESULTS/{fname}")
