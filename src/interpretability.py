"""
Interpretability analyses for BALLADEER.

Five complementary lenses:
  1. SHAP beeswarm        — per-sample, per-feature signed attribution (XGBoost)
  2. Permutation importance — model-agnostic importance cross-check (SVM)
  3. PCA loadings heatmap  — which original features drive each quantum input dim
  4. VQC encoding analysis — which PCA components the circuit amplifies most,
                             composed back to original feature space
  5. Calibration curves    — reliability of predicted probabilities for all models
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src import config

try:
    import shap as _shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False
    print("[interpretability] shap not installed — SHAP analysis skipped. "
          "Install with: pip install shap")


# ── helpers ──────────────────────────────────────────────────────────────────

def _savefig(fig, name, run_id):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    fname = f"{name}_{run_id}.png"
    fig.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] RESULTS/{fname}")


# ── 1. SHAP ──────────────────────────────────────────────────────────────────

def shap_analysis(xgb_model, X_tr_cls, X_te_cls, feat_names, run_id):
    if not _SHAP_OK:
        return
    print("\n[interp] SHAP analysis (XGBoost)...")
    explainer  = _shap.TreeExplainer(xgb_model)
    shap_vals  = explainer.shap_values(X_te_cls)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]   # positive class for binary classifiers

    # Beeswarm (summary) plot
    fig, ax = plt.subplots(figsize=(10, 8))
    _shap.summary_plot(
        shap_vals, X_te_cls,
        feature_names=feat_names,
        max_display=20,
        show=False,
        plot_type="dot",
    )
    fig = plt.gcf()
    fig.suptitle(f"SHAP Feature Attribution — XGBoost ({run_id})",
                 fontsize=11, fontweight="bold", y=1.01)
    _savefig(fig, "shap_beeswarm", run_id)

    # Bar plot of mean |SHAP|
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:20]
    fig, ax  = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(order)), mean_abs[order][::-1], color="steelblue", alpha=0.85)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order[::-1]], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"SHAP Mean Importance — XGBoost ({run_id})",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "shap_bar", run_id)
    print(f"  Top SHAP features: "
          + ", ".join(feat_names[i] for i in order[:5]))


# ── 2. Permutation importance ─────────────────────────────────────────────────

def permutation_importance(svm_estimator, X_te_cls, y_te, feat_names, run_id):
    print("\n[interp] Permutation importance (SVM)...")
    from sklearn.inspection import permutation_importance as _pi
    result = _pi(
        svm_estimator, X_te_cls, y_te,
        n_repeats=30, scoring="roc_auc", random_state=42, n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    means = result.importances_mean[order][::-1]
    stds  = result.importances_std[order][::-1]
    ax.barh(range(len(order)), means, xerr=stds,
            color="darkorange", alpha=0.80, capsize=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order[::-1]], fontsize=9)
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Mean decrease in ROC-AUC (30 repeats)", fontsize=11)
    ax.set_title(f"Permutation Importance — SVM ({run_id})",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "permutation_importance", run_id)
    print(f"  Top permutation features: "
          + ", ".join(feat_names[i] for i in order[:5]))


# ── 3. PCA loadings heatmap ──────────────────────────────────────────────────

def pca_loadings(pca, feat_names, run_id):
    print("\n[interp] PCA loadings heatmap...")
    comps = pca.components_                   # (N_PCA, N_features)
    n_pc, n_feat = comps.shape

    # Show top 25 features by max absolute loading across all PCs
    max_abs = np.abs(comps).max(axis=0)
    top_idx = np.argsort(max_abs)[::-1][:min(25, n_feat)]
    sub_comps  = comps[:, top_idx]
    sub_names  = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in top_idx]
    var_exp    = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(max(10, len(top_idx) * 0.55), n_pc * 0.7 + 1.5))
    im = ax.imshow(sub_comps, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(sub_comps).max(), vmax=np.abs(sub_comps).max())
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_pc))
    ax.set_yticklabels(
        [f"PC{i+1} ({var_exp[i]*100:.1f}%)" for i in range(n_pc)], fontsize=9)
    plt.colorbar(im, ax=ax, label="Loading weight")
    ax.set_title(f"PCA Loadings — quantum input dimensions ({run_id})\n"
                 f"(top {len(top_idx)} features by max |loading|)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "pca_loadings", run_id)


# ── 4. VQC encoding scale analysis ───────────────────────────────────────────

def vqc_encoding(top_params, pca, feat_names, run_id):
    print("\n[interp] VQC encoding analysis...")
    _N_ENC = config.N_LAYERS * config.N_QUBITS * 3
    if len(top_params) < _N_ENC:
        print("  Skipped — params array shorter than expected.")
        return

    enc_s = np.abs(top_params[:_N_ENC]).reshape(
        config.N_LAYERS, config.N_QUBITS, 3)

    # Per-qubit mean encoding strength (averaged over layers and rotation axes)
    enc_strength = enc_s.mean(axis=(0, 2))  # shape: (N_QUBITS,)

    # ── Panel A: encoding strength per qubit / PCA dimension ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(range(config.N_QUBITS), enc_strength, color="royalblue", alpha=0.85)
    ax.set_xticks(range(config.N_QUBITS))
    ax.set_xticklabels([f"PC{i+1}" for i in range(config.N_QUBITS)], fontsize=9)
    ax.set_xlabel("PCA component / qubit", fontsize=10)
    ax.set_ylabel("Mean |encoding scale|", fontsize=10)
    ax.set_title("VQC: encoding strength per PCA dimension", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel B: attribution back to original features ────────────────────────
    # For each original feature f:
    #   attribution(f) = Σ_q  enc_strength[q] × |pca.components_[q, f]|
    comps    = pca.components_          # (N_PCA, N_features)
    n_feat   = comps.shape[1]
    attr_raw = np.abs(comps).T @ enc_strength   # (N_features,)
    top_idx  = np.argsort(attr_raw)[::-1][:20]
    top_names = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in top_idx]

    ax2 = axes[1]
    ax2.barh(range(len(top_idx)), attr_raw[top_idx][::-1],
             color="mediumslateblue", alpha=0.85)
    ax2.set_yticks(range(len(top_idx)))
    ax2.set_yticklabels(top_names[::-1], fontsize=9)
    ax2.set_xlabel("VQC attribution score\n(enc_strength × |PCA loading|)", fontsize=10)
    ax2.set_title("VQC: attribution to original features", fontsize=11)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(f"VQC Encoding Interpretability ({run_id})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "vqc_encoding", run_id)
    print(f"  Top VQC-attributed features: "
          + ", ".join(top_names[:5]))


# ── 5. Calibration curves ────────────────────────────────────────────────────

def calibration_curves(y_te, probas, run_id):
    print("\n[interp] Calibration curves...")
    _COLORS = {
        "VQC":     "royalblue",    "QSVM-ZZ": "indigo",
        "QCNN-8":  "mediumslateblue", "LR":   "peru",
        "SVM":     "darkorange",   "RF":      "forestgreen",
        "XGBoost": "crimson",
    }
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
    for name, proba in probas.items():
        try:
            frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=5,
                                                    strategy="quantile")
            ls = "--" if "Q" in name else "-"
            ax.plot(mean_pred, frac_pos, marker="o", lw=1.8, ls=ls,
                    color=_COLORS.get(name, "gray"), label=name)
        except Exception:
            pass
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives (ADHD)", fontsize=11)
    ax.set_title(f"Calibration Curves — BALLADEER ({run_id})", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "calibration", run_id)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_all(xgb_model, svm_estimator, pca, vqc_top_params,
            X_tr_cls, X_te_cls, y_te, feat_names, run_id, probas):
    print("\n" + "=" * 72)
    print("[14] Interpretability analyses")
    print("=" * 72)
    shap_analysis(xgb_model, X_tr_cls, X_te_cls, feat_names, run_id)
    permutation_importance(svm_estimator, X_te_cls, y_te, feat_names, run_id)
    pca_loadings(pca, feat_names, run_id)
    vqc_encoding(vqc_top_params, pca, feat_names, run_id)
    calibration_curves(y_te, probas, run_id)
