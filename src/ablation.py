"""
Modality ablation study.

Seven feature subsets × three classical models (SVM, RF, XGBoost)
run under 5-fold stratified CV using only training data.
Answers the reviewer question: "does multi-modal fusion actually help?"
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score
from xgboost import XGBClassifier

from src import config


# ── feature-group name patterns ───────────────────────────────────────────────

_DEMO_NAMES   = {"age", "gender"}
_EEG_PREFIXES = (
    "TBR", "frontal_TBR", "intra_TBR", "alpha_asym", "theta_asym",
    "occ_alpha", "theta_entropy", "beta_entropy", "hjorth_", "TAR",
    "gamma_theta", "delta_pow", "theta_pow", "alpha_pow", "beta_pow",
    "gamma_pow", "TBR_delta", "alpha_asym_delta", "occ_alpha_delta",
    "theta_pow_delta", "hjorth_mob_f_delta", "eeg_n_levels",
)
_BEHAV_PREFIXES = (
    "mean_RT", "RT_CV", "RT_skew", "RT_kurt", "RT_tau",
    "RT_cv_per_block", "omission_rate", "game_velocity",
    "game_omissions", "game_commissions", "tags_present",
)
_PHYSIO_PREFIXES = ("embrace_", "embrace_present")


def _feature_mask(feat_names, include_groups):
    """Boolean mask selecting features whose names belong to any of the groups."""
    mask = np.zeros(len(feat_names), dtype=bool)
    for i, fn in enumerate(feat_names):
        for grp in include_groups:
            if grp == "demo" and fn in _DEMO_NAMES:
                mask[i] = True
            elif grp == "eeg" and any(fn.startswith(p) for p in _EEG_PREFIXES):
                mask[i] = True
            elif grp == "behav" and any(fn.startswith(p) for p in _BEHAV_PREFIXES):
                mask[i] = True
            elif grp == "physio" and any(fn.startswith(p) for p in _PHYSIO_PREFIXES):
                mask[i] = True
    return mask


_SUBSETS = {
    "Demo only":       ["demo"],
    "EEG only":        ["eeg"],
    "Behavioural only":["behav"],
    "Physiological":   ["physio"],
    "EEG + Behav":     ["eeg", "behav"],
    "EEG + Physio":    ["eeg", "physio"],
    "All modalities":  ["demo", "eeg", "behav", "physio"],
}


def run_ablation(X_tr_raw, y_tr, feat_names, run_id):
    """
    Modality ablation: 7 subsets × 3 classical models, 5-fold stratified CV.
    X_tr_raw should be winsorized but *un-scaled* (Pipeline handles scaling).
    """
    print("\n" + "=" * 72)
    print("[ablation] Modality ablation study (5-fold CV, ROC-AUC)")
    print("=" * 72)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    n_neg = int((y_tr == 0).sum())
    n_pos = int(y_tr.sum())
    spw   = n_neg / max(n_pos, 1)
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def _make_models():
        return {
            "SVM": Pipeline([
                ("sc", StandardScaler()),
                ("m",  SVC(probability=True, class_weight="balanced", random_state=42)),
            ]),
            "RF": Pipeline([
                ("sc", StandardScaler()),
                ("m",  RandomForestClassifier(
                    n_estimators=500, class_weight="balanced", random_state=42)),
            ]),
            "XGBoost": Pipeline([
                ("sc", StandardScaler()),
                ("m",  XGBClassifier(
                    n_estimators=300, eval_metric="logloss",
                    scale_pos_weight=spw, random_state=42, n_jobs=1,
                    device="cpu")),
            ]),
        }

    results = {}   # subset_name → {model_name: {"mean": float, "std": float}}
    for subset_name, groups in _SUBSETS.items():
        mask = _feature_mask(feat_names, groups)
        if not mask.any():
            print(f"  {subset_name:<22s}: no features matched — skipped")
            results[subset_name] = {}
            continue
        X_sub = X_tr_raw[:, mask]
        n_sel = int(mask.sum())
        results[subset_name] = {}
        _scoring = {
            "roc_auc": "roc_auc",
            "f1":      make_scorer(f1_score, zero_division=0),
        }
        for model_name, model in _make_models().items():
            cv_out = cross_validate(
                model, X_sub, y_tr, cv=cv, scoring=_scoring, n_jobs=-1)
            auc_mu = float(cv_out["test_roc_auc"].mean())
            auc_sd = float(cv_out["test_roc_auc"].std())
            f1_mu  = float(cv_out["test_f1"].mean())
            f1_sd  = float(cv_out["test_f1"].std())
            results[subset_name][model_name] = {
                "auc_mean": auc_mu, "auc_std": auc_sd,
                "f1_mean":  f1_mu,  "f1_std":  f1_sd,
                "n_features": n_sel,
            }
            print(f"  {subset_name:<22s} | {model_name:<8s}: "
                  f"AUC={auc_mu:.3f}±{auc_sd:.3f}  "
                  f"F1={f1_mu:.3f}±{f1_sd:.3f}  (n={n_sel})")

    # ── JSON ──────────────────────────────────────────────────────────────────
    payload = {"run_id": run_id, "cv": "5-fold StratifiedKFold",
               "scoring": "roc_auc", "subsets": results}
    jpath = os.path.join(config.RESULTS_DIR, f"ablation_{run_id}.json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [OK] RESULTS/ablation_{run_id}.json")

    # ── Bar chart: AUC (left) + F1 (right) ───────────────────────────────────
    subset_names = list(_SUBSETS.keys())
    model_names  = ["SVM", "RF", "XGBoost"]
    colors       = ["steelblue", "forestgreen", "crimson"]
    x     = np.arange(len(subset_names))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=False)
    for metric, ax, ylabel, chance in [
        ("auc", axes[0], "ROC-AUC (5-fold CV)", 0.5),
        ("f1",  axes[1], "F1 Score (5-fold CV)", None),
    ]:
        for k, (mname, col) in enumerate(zip(model_names, colors)):
            means = np.array([
                results.get(s, {}).get(mname, {}).get(f"{metric}_mean", 0.0)
                for s in subset_names])
            stds  = np.array([
                results.get(s, {}).get(mname, {}).get(f"{metric}_std", 0.0)
                for s in subset_names])
            bars = ax.bar(x + (k - 1) * width, means, width,
                          yerr=stds, capsize=3, label=mname,
                          color=col, alpha=0.82, error_kw={"elinewidth": 1.2})
            for bar, mu in zip(bars, means):
                if mu > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{mu:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(subset_names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, 1.05)
        if chance is not None:
            ax.axhline(chance, color="black", lw=0.8, ls="--", alpha=0.4, label="Chance")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Modality Ablation Study ({run_id})  —  Error bars = ±1 SD (5-fold CV)",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = f"ablation_{run_id}.png"
    fig.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] RESULTS/{fname}")

    return payload
