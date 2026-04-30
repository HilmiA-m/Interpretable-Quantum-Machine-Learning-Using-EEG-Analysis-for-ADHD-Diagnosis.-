"""
Quantum-specific analyses:

  A. Kernel Target Alignment (KTA)
     — compares ZZ quantum kernel alignment with labels vs classical kernels.
     High KTA → kernel geometry fits class structure → QSVM should work well.

  B. VQC Parameter Noise Robustness
     — Gaussian noise at σ ∈ {0, 0.01, 0.03, 0.05, 0.10} applied to trained
       VQC weights, 20 trials each; plots AUC degradation curve.
     Demonstrates that the learned circuit is stable under small perturbations.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from src import config


# ── A. Kernel Target Alignment ────────────────────────────────────────────────

def _kta(K, y):
    """Normalized KTA: <K_y, K>_F / (||K_y||_F * ||K||_F).

    K_y[i,j] = y_i * y_j  where y ∈ {-1, +1}.
    """
    y_pm = 2.0 * y - 1.0          # {0,1} → {-1,+1}
    K_y  = np.outer(y_pm, y_pm)
    num  = float(np.sum(K * K_y))
    den  = float(np.sqrt(np.sum(K ** 2)) * np.sqrt(np.sum(K_y ** 2)))
    return num / den if den > 1e-12 else 0.0


def _rbf_kernel(X, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    sq_dists = (
        np.sum(X ** 2, axis=1, keepdims=True)
        + np.sum(X ** 2, axis=1)
        - 2.0 * X @ X.T
    )
    return np.exp(-gamma * np.clip(sq_dists, 0, None))


def _poly_kernel(X, degree=3, coef0=1.0):
    return (X @ X.T + coef0) ** degree


def _linear_kernel(X):
    return X @ X.T


def run_kta(K_tr_quantum, X_tr_q, y_tr, run_id):
    """
    Compare KTA of ZZ quantum kernel vs classical kernels on the training set.
    X_tr_q: quantum-pipeline features (PCA + MinMax scaled).
    K_tr_quantum: precomputed ZZ kernel matrix for training set.
    """
    print("\n[quantum_analysis] Kernel Target Alignment (KTA)")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    kernels = {
        "ZZ Quantum": K_tr_quantum,
        "RBF":        _rbf_kernel(X_tr_q),
        "RBF (γ=0.1)":_rbf_kernel(X_tr_q, gamma=0.1),
        "Linear":     _linear_kernel(X_tr_q),
        "Polynomial": _poly_kernel(X_tr_q),
    }

    kta_scores = {}
    for name, K in kernels.items():
        score = _kta(K, y_tr)
        kta_scores[name] = round(score, 5)
        print(f"  KTA [{name:<14s}] = {score:.5f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = list(kta_scores.keys())
    vals   = [kta_scores[n] for n in names]
    colors = ["royalblue" if "Quantum" in n else "steelblue" for n in names]
    bars   = ax.bar(names, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Kernel Target Alignment", fontsize=11)
    ax.set_title(
        f"Kernel Target Alignment — quantum vs classical ({run_id})\n"
        f"Higher KTA → kernel geometry better separates ADHD / Control",
        fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.20 + 0.01)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    fig.tight_layout()
    fname = f"kta_{run_id}.png"
    fig.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] RESULTS/{fname}")

    payload = {"run_id": run_id, "kta_scores": kta_scores}
    jpath = os.path.join(config.RESULTS_DIR, f"kta_{run_id}.json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [OK] RESULTS/kta_{run_id}.json")
    return payload


# ── B. VQC Parameter Noise Robustness ─────────────────────────────────────────

def run_vqc_noise_robustness(top_params, X_te_q, y_te, run_id,
                             sigmas=(0.0, 0.01, 0.03, 0.05, 0.10),
                             n_trials=20, seed=42):
    """
    Perturb trained VQC parameters with Gaussian noise and measure AUC degradation.
    top_params: 1-D numpy array of trained circuit parameters.
    X_te_q:     test set, quantum pipeline features.
    """
    print("\n[quantum_analysis] VQC noise robustness")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Import VQC inference from the trained model
    from src.models.vqc import vqc_proba
    import pennylane.numpy as pnp

    rng = np.random.default_rng(seed)
    auc_results = {}   # sigma → list of AUC values

    base_auc = float(roc_auc_score(y_te, vqc_proba(
        pnp.array(top_params, requires_grad=False), X_te_q)))
    print(f"  Baseline (σ=0): AUC = {base_auc:.4f}")

    for sigma in sigmas:
        aucs = []
        for _ in range(n_trials):
            if sigma == 0.0:
                noisy = pnp.array(top_params.copy(), requires_grad=False)
            else:
                noise = rng.normal(0, sigma, size=top_params.shape)
                noisy = pnp.array(top_params + noise, requires_grad=False)
            try:
                p = vqc_proba(noisy, X_te_q)
                if len(np.unique(y_te)) < 2:
                    aucs.append(base_auc)
                else:
                    aucs.append(float(roc_auc_score(y_te, p)))
            except Exception:
                aucs.append(0.5)
        auc_results[float(sigma)] = {
            "mean": round(float(np.mean(aucs)), 5),
            "std":  round(float(np.std(aucs)),  5),
            "all":  [round(a, 5) for a in aucs],
        }
        print(f"  σ={sigma:.2f}  AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Line plot with error band
    sigma_vals = [float(s) for s in sigmas]
    means = np.array([auc_results[s]["mean"] for s in sigma_vals])
    stds  = np.array([auc_results[s]["std"]  for s in sigma_vals])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigma_vals, means, "o-", color="royalblue", lw=2, ms=7, label="Mean AUC")
    ax.fill_between(sigma_vals, means - stds, means + stds,
                    alpha=0.20, color="royalblue", label="±1 SD")
    ax.axhline(0.5, color="gray", lw=0.9, ls="--", label="Chance (0.5)")
    ax.axhline(base_auc, color="green", lw=1.2, ls=":", label=f"Baseline ({base_auc:.3f})")
    ax.set_xlabel("Gaussian noise σ applied to VQC parameters", fontsize=11)
    ax.set_ylabel("ROC-AUC on test set", fontsize=11)
    ax.set_title(
        f"VQC Parameter Noise Robustness ({run_id})\n"
        f"({n_trials} trials per noise level)",
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"vqc_noise_{run_id}.png"
    fig.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] RESULTS/{fname}")

    payload = {
        "run_id":    run_id,
        "n_trials":  n_trials,
        "base_auc":  round(base_auc, 5),
        "sigma_results": {str(k): v for k, v in auc_results.items()},
    }
    jpath = os.path.join(config.RESULTS_DIR, f"vqc_noise_{run_id}.json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [OK] RESULTS/vqc_noise_{run_id}.json")
    return payload


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_all(K_tr_quantum, X_tr_q, X_te_q, y_tr, y_te, vqc_top_params, run_id):
    print("\n" + "=" * 72)
    print("[quantum_analysis] Quantum-specific analyses")
    print("=" * 72)
    kta_payload   = run_kta(K_tr_quantum, X_tr_q, y_tr, run_id)
    noise_payload = run_vqc_noise_robustness(vqc_top_params, X_te_q, y_te, run_id)
    return {"kta": kta_payload, "noise": noise_payload}
