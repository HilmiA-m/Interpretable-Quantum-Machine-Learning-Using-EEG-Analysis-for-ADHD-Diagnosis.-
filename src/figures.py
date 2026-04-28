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
    "SVM":     "darkorange",
    "RF":      "forestgreen",
    "XGBoost": "crimson",
}


def save_figures(y_te, probas, preds, thresholds, feat_imp):
    """
    probas:     dict  name → probability array
    preds:      dict  name → binary prediction array
    thresholds: dict  name → float threshold
    feat_imp:   list of (feature_name, importance) pairs
    """
    _save_roc(y_te, probas)
    _save_pr(y_te, probas)
    _save_confusion(y_te, preds, probas, thresholds)
    _save_feature_importance(feat_imp)


def _save_roc(y_te, probas):
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_te, proba)
        try:    auc = roc_auc_score(y_te, proba)
        except: auc = float("nan")
        ls = "--" if "Q" in name else "-"
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=_COLORS[name], lw=1.8, ls=ls)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC — BALLADEER ADHD v40")
    ax.legend(fontsize=8.5); fig.tight_layout()
    fig.savefig(os.path.join(config.DATA_ROOT, "roc_curves_v40.png"), dpi=150)
    plt.close(fig)
    print("  [OK] roc_curves_v40.png")


def _save_pr(y_te, probas):
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, proba in probas.items():
        prec_c, rec_c, _ = precision_recall_curve(y_te, proba)
        try:    ap = average_precision_score(y_te, proba)
        except: ap = float("nan")
        ls = "--" if "Q" in name else "-"
        ax.plot(rec_c, prec_c, label=f"{name} (AP={ap:.3f})",
                color=_COLORS[name], lw=1.8, ls=ls)
    baseline = y_te.mean()
    ax.axhline(baseline, color="k", ls="--", alpha=0.3, label=f"Baseline={baseline:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall — BALLADEER ADHD v40")
    ax.legend(fontsize=8.5); fig.tight_layout()
    fig.savefig(os.path.join(config.DATA_ROOT, "pr_curves_v40.png"), dpi=150)
    plt.close(fig)
    print("  [OK] pr_curves_v40.png")


def _save_confusion(y_te, preds, probas, thresholds):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (name, yp) in zip(axes.flatten(), preds.items()):
        cm = confusion_matrix(y_te, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Ctrl", "ADHD"], yticklabels=["Ctrl", "ADHD"])
        try:    auc = roc_auc_score(y_te, probas[name])
        except: auc = float("nan")
        ax.set_title(f"{name}  AUC={auc:.3f}  thr={thresholds[name]:.2f}")
    fig.suptitle("Confusion Matrices — BALLADEER ADHD v40", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(config.DATA_ROOT, "confusion_matrices_v40.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] confusion_matrices_v40.png")


def _save_feature_importance(feat_imp):
    names_sorted = [x[0] for x in feat_imp[:15]]
    vals_sorted  = [x[1] for x in feat_imp[:15]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(names_sorted)), vals_sorted, color="steelblue", alpha=0.85)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("XGBoost Feature Importance", fontsize=11)
    ax.set_title("Top 15 Features — XGBoost (v40 curated)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.DATA_ROOT, "feature_importance_v40.png"), dpi=150)
    plt.close(fig)
    print("  [OK] feature_importance_v40.png")
