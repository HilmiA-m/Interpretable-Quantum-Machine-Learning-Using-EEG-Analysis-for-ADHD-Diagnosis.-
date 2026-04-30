"""
Pairwise statistical significance tests for all model pairs.

  1. McNemar's test   — compares binary error rates (exact when n<25, χ² otherwise)
  2. Bootstrap AUC CI — 95% CI for (AUC_A − AUC_B); excludes 0 → significant

Both are standard requirements for medical AI publications.
Bonferroni-corrected threshold is also reported for the 15 pairwise comparisons.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src import config


# ── statistical primitives ────────────────────────────────────────────────────

def _mcnemar_pvalue(n01: int, n10: int) -> float:
    """Two-sided McNemar's test p-value.

    Exact binomial when n01+n10 < 25, continuity-corrected chi-squared otherwise.
    """
    n = n01 + n10
    if n == 0:
        return 1.0
    if n < 25:
        from scipy.stats import binom
        b = min(n01, n10)
        return float(min(1.0, 2.0 * binom.cdf(b, n, 0.5)))
    chi2 = (abs(n01 - n10) - 1.0) ** 2 / n
    from scipy.stats import chi2 as chi2_dist
    return float(chi2_dist.sf(chi2, df=1))


def _bootstrap_auc_diff(y_true, proba_a, proba_b, n_boot=1000, seed=42):
    """Bootstrap 95% CI for (AUC_A − AUC_B).

    Returns (mean_diff, ci_lo, ci_hi, approx_p_value).
    p-value approximated as 2 × fraction of bootstrap samples with wrong sign.
    """
    rng   = np.random.default_rng(seed)
    idx   = np.arange(len(y_true))
    diffs = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        try:
            d = (roc_auc_score(y_true[s], proba_a[s])
                 - roc_auc_score(y_true[s], proba_b[s]))
            diffs.append(d)
        except Exception:
            pass
    if not diffs:
        return 0.0, -1.0, 1.0, 1.0
    diffs  = np.array(diffs)
    mean_d = float(diffs.mean())
    lo     = float(np.percentile(diffs, 2.5))
    hi     = float(np.percentile(diffs, 97.5))
    p_val  = float(np.mean(diffs <= 0) if mean_d > 0 else np.mean(diffs >= 0))
    return mean_d, lo, hi, float(min(1.0, 2.0 * p_val))


# ── main entry point ──────────────────────────────────────────────────────────

def run_significance_tests(y_te, preds, probas, run_id):
    """
    Compute pairwise McNemar and bootstrap AUC tests for every model pair.
    Saves two heatmaps and a JSON to RESULTS/.
    """
    print("\n[stats] Pairwise significance tests (McNemar + bootstrap AUC)...")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    names = list(preds.keys())
    n     = len(names)
    n_pairs = n * (n - 1) // 2
    bonf_alpha = 0.05 / n_pairs   # Bonferroni-corrected threshold

    mcnemar_p    = np.ones((n, n))
    auc_diff_mu  = np.zeros((n, n))
    auc_diff_sig = np.zeros((n, n), dtype=bool)   # CI excludes 0

    detail = {}
    for i in range(n):
        for j in range(i + 1, n):
            na, nb = names[i], names[j]
            ca = (preds[na] == y_te)
            cb = (preds[nb] == y_te)
            n01 = int(np.sum(~ca &  cb))
            n10 = int(np.sum( ca & ~cb))
            p_mc = _mcnemar_pvalue(n01, n10)
            mcnemar_p[i, j] = mcnemar_p[j, i] = p_mc

            mu, lo, hi, p_auc = _bootstrap_auc_diff(y_te, probas[na], probas[nb])
            auc_diff_mu[i, j]  =  mu
            auc_diff_mu[j, i]  = -mu
            sig = not (lo <= 0.0 <= hi)
            auc_diff_sig[i, j] = auc_diff_sig[j, i] = sig

            detail[f"{na}_vs_{nb}"] = {
                "mcnemar_p":          round(p_mc,  4),
                "mcnemar_sig_0.05":   bool(p_mc < 0.05),
                "mcnemar_sig_bonf":   bool(p_mc < bonf_alpha),
                "auc_diff_mean":      round(mu,    4),
                "auc_diff_ci95_lo":   round(lo,    4),
                "auc_diff_ci95_hi":   round(hi,    4),
                "auc_diff_sig":       bool(sig),
                "n01": n01, "n10": n10,
            }
            sig_str = "**" if p_mc < bonf_alpha else ("*" if p_mc < 0.05 else "")
            print(f"  {na:10s} vs {nb:10s}  "
                  f"McNemar p={p_mc:.4f}{sig_str:2s}  "
                  f"ΔAUC={mu:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}]"
                  + (" *" if sig else ""))

    print(f"  Bonferroni threshold: p < {bonf_alpha:.4f}  "
          f"(**=Bonferroni, *=uncorrected)")

    # ── JSON ──────────────────────────────────────────────────────────────────
    payload = {
        "models": names,
        "n_pairs": n_pairs,
        "bonferroni_alpha": round(bonf_alpha, 6),
        "pairs": detail,
    }
    jpath = os.path.join(config.RESULTS_DIR, f"significance_tests_{run_id}.json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [OK] RESULTS/significance_tests_{run_id}.json")

    # ── McNemar p-value heatmap ───────────────────────────────────────────────
    log_p = -np.log10(np.clip(mcnemar_p + np.eye(n), 1e-10, 1))
    np.fill_diagonal(log_p, 0)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(log_p, cmap="Reds", vmin=0, vmax=max(2, log_p.max()))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            txt = f"{mcnemar_p[i,j]:.3f}"
            col = "white" if log_p[i, j] > 1.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=col)
    plt.colorbar(im, ax=ax, label="−log₁₀(p-value)  [redder = more significant]")
    ax.set_title(
        f"McNemar's Test — pairwise error-rate comparison ({run_id})\n"
        f"Bonferroni α = {bonf_alpha:.4f}  (n_pairs={n_pairs})",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    f1 = f"mcnemar_heatmap_{run_id}.png"
    fig.savefig(os.path.join(config.RESULTS_DIR, f1), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] RESULTS/{f1}")

    # ── AUC difference matrix ─────────────────────────────────────────────────
    vmax = max(np.abs(auc_diff_mu).max(), 0.05)
    fig, ax = plt.subplots(figsize=(9, 7))
    im2 = ax.imshow(auc_diff_mu, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            marker = " *" if auc_diff_sig[i, j] else ""
            txt = f"{auc_diff_mu[i,j]:+.3f}{marker}"
            col = "white" if abs(auc_diff_mu[i, j]) > vmax * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=col)
    plt.colorbar(im2, ax=ax, label="ΔAUC (row − col)  [* = 95% CI excludes 0]")
    ax.set_title(
        f"AUC Difference Matrix — bootstrap 95% CI ({run_id})\n"
        f"(* = statistically significant difference)",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    f2 = f"auc_diff_matrix_{run_id}.png"
    fig.savefig(os.path.join(config.RESULTS_DIR, f2), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] RESULTS/{f2}")

    return payload
