# Interpretable Quantum Machine Learning for ADHD Diagnosis Using EEG

> A multimodal quantum-classical hybrid framework that combines EEG neuroimaging, behavioural response-time modelling, and physiological wearable data to diagnose ADHD — with full interpretability at every layer of the pipeline.

---

## Overview

ADHD affects ~10% of children and ~5% of adults worldwide, yet diagnosis remains subjective and clinician-dependent. This project introduces **BALLADEER** — a reproducible, interpretable, and statistically rigorous ML pipeline that treats ADHD diagnosis as a binary classification problem grounded in neuroscience.

Three quantum models (VQC, QSVM, QCNN) are benchmarked against four classical baselines (Logistic Regression, SVM, Random Forest, XGBoost) under identical preprocessing, threshold calibration, and evaluation conditions. Every design decision is motivated by peer-reviewed cognitive neuroscience.

---

## What Makes This Different

| Aspect | This Project |
|---|---|
| **Quantum models** | Three distinct quantum architectures, not just one |
| **Interpretability** | 5 complementary lenses — SHAP, permutation importance, PCA loadings, VQC encoding attribution, calibration curves |
| **Statistical rigour** | Pairwise McNemar + bootstrap AUC with Bonferroni correction across all model pairs |
| **Ablation study** | 7 feature subsets × 4 models — proves multi-modal fusion actually helps |
| **Quantum validation** | Kernel Target Alignment and VQC noise robustness analysis |
| **Leakage-free** | Winsorisation, zero-variance removal, and threshold calibration all computed on training data only |
| **Reproducible** | Every run produces a timestamped JSON + full set of figures |

---

## Models

### Quantum
| Model | Description |
|---|---|
| **VQC** | 8-qubit × 5-layer variational circuit with full CZ entanglement, 249 parameters, 12-restart ensemble with focal loss and cosine LR decay |
| **QSVM-ZZ** | ZZ/IQP feature map (2 reps), quantum kernel SVM with Nyström approximation for scalability, kernel centering |
| **QCNN-8** | Quantum convolutional network with U3+CNOT conv blocks, CRZ/CRX pooling, learned 8-qubit linear readout |

### Classical Baselines
| Model | Description |
|---|---|
| **Logistic Regression** | Linear baseline, class-balanced, grid search over C and solver |
| **SVM** | RBF/polynomial kernel, class-balanced, grid search over 36 hyperparameter combinations |
| **Random Forest** | 500–1500 trees, class-balanced, grid search over depth and feature sampling |
| **XGBoost** | `scale_pos_weight` for class imbalance, large regularised grid search, GPU-accelerated if available |

---

## Data Modalities

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Modalities                        │
├──────────────┬──────────────────┬──────────────┬───────────┤
│ Demographics │      EEG         │ Behavioural  │ Wearable  │
│   age, sex   │  22 features     │  7 features  │ Embrace+  │
│              │  29-ch CGX EEG   │  TAGS task   │ EDA, HR,  │
│              │  3 difficulty    │  ex-Gaussian │ HRV       │
│              │  levels          │  RT model    │           │
└──────────────┴──────────────────┴──────────────┴───────────┘
```

### EEG Features (22)
Computed per session, averaged across 3 slackline difficulty levels:
- **TBR** (theta/beta ratio) — the primary ADHD biomarker, global and frontal-specific
- **Frontal alpha asymmetry** — left/right log-ratio, correlates with emotional dysregulation
- **Spectral entropy** — theta and beta band signal complexity
- **Hjorth parameters** — mobility and complexity for frontal, occipital, and central regions
- **Band power** — delta, theta, alpha, beta, gamma
- **Cross-level gradients** — how features change from easy to hard slackline tasks (neural load response)

### Behavioural Features (7)
From the TAGS reaction-time task:
- Mean RT, RT coefficient of variation, RT skewness, RT kurtosis
- **Ex-Gaussian τ** — exponential tail of RT distribution, the gold-standard ADHD cognitive marker
- Per-block RT variability (vigilance decrement)
- Omission rate

---

## Pipeline

```
Raw Data
    │
    ├── EEG (.csv)  → MNE: 1–45 Hz filter → Common Average Reference
    │                   → Epoch artifact rejection (150 µV threshold)
    │                   → 22 features per session
    │
    ├── TAGS (.csv) → RT parsing → ex-Gaussian fit → 7 features
    │
    ├── GAME_DATA   → Velocity, omissions, commissions
    │
    └── Embrace+    → EDA, HR, HRV metrics
             │
             ▼
    Feature matrix (all modalities concatenated)
             │
    NaN imputation (median, training only)
    Zero-variance removal (training only)
    Winsorisation [5th–95th pct, training only]
             │
    ┌────────┴────────┐
    │                 │
  Quantum          Classical
  pipeline         pipeline
  StdScale →       StdScale
  PCA-8 →
  MinMax [0,π]
    │                 │
  VQC  QSVM  QCNN   LR  SVM  RF  XGB
             │
    OOF threshold calibration (F1-optimal, never on test)
             │
    Evaluation: Acc, Prec, Recall, F1, ROC-AUC, PR-AUC
    + Bootstrap 95% CI for all metrics
```

---

## Interpretability

Five complementary lenses, all saved as figures:

1. **SHAP beeswarm + bar** — per-sample signed feature attributions (XGBoost)
2. **Permutation importance** — model-agnostic importance via AUC drop (SVM, 30 repeats)
3. **PCA loadings heatmap** — which original features drive each quantum input dimension
4. **VQC encoding attribution** — traces circuit encoding strength back to original EEG features via the PCA chain
5. **Calibration curves** — reliability of predicted probabilities for all 7 models

---

## Statistical Validation

- **McNemar's test** — exact binomial (n < 25) or continuity-corrected χ² for all model pairs
- **Bootstrap AUC 95% CI** — 2000 samples, CI excludes 0 → significant
- **Bonferroni correction** — threshold adjusted for all pairwise comparisons
- **Repeated stratified K-fold** — 5-fold × 3 repeats on full dataset for classical models
- **Modality ablation** — 7 feature subsets, ROC-AUC + F1, proves multi-modal fusion outperforms any single modality

---

## Quantum-Specific Analyses

- **Kernel Target Alignment (KTA)** — compares ZZ quantum kernel alignment with class labels vs RBF, linear, and polynomial classical kernels
- **VQC noise robustness** — Gaussian parameter perturbation at σ ∈ {0, 0.01, 0.03, 0.05, 0.10}, 20 trials each, demonstrates circuit stability

---

## Project Structure

```
.
├── main.py                          # Entry point
├── src/
│   ├── config.py                    # All hyperparameters in one place
│   ├── pipeline.py                  # Orchestration (16 steps)
│   ├── dataset.py                   # Feature assembly & preprocessing
│   ├── metrics.py                   # Bootstrap CI, threshold calibration
│   ├── figures.py                   # ROC, PR, confusion matrix plots
│   ├── results.py                   # Per-run JSON + BEST_RESULTS tracking
│   ├── stats.py                     # McNemar + bootstrap AUC tests
│   ├── ablation.py                  # Modality ablation study
│   ├── quantum_analysis.py          # KTA + VQC noise robustness
│   ├── interpretability.py          # SHAP, permutation, PCA, VQC encoding
│   ├── data/
│   │   ├── eeg.py                   # MNE EEG feature extraction (22 features)
│   │   ├── behavioural.py           # RT + ex-Gaussian modelling (7 features)
│   │   ├── embrace.py               # Wearable physiological features
│   │   └── demographics.py          # Age, sex, ADHD label loading
│   └── models/
│       ├── vqc.py                   # Variational Quantum Classifier
│       ├── qsvm.py                  # Quantum SVM (ZZ kernel)
│       ├── qcnn.py                  # Quantum CNN
│       └── classical.py             # LR, SVM, RF, XGBoost + repeated CV
├── workers/
│   └── vqc_subprocess_runner.py     # Isolated VQC training (hard-kill timeout)
└── RESULTS/                         # All outputs (auto-created)
    ├── run_YYYYMMDD_HHMMSS.json
    ├── BEST_RESULTS.json
    ├── roc_curves_*.png
    ├── confusion_matrices_*.png
    ├── shap_beeswarm_*.png
    ├── ablation_*.png
    ├── kta_*.png
    ├── vqc_noise_*.png
    └── ...
```

---

## Setup

### Requirements
```bash
pip install pennylane pennylane-lightning mne scipy scikit-learn xgboost shap seaborn matplotlib pandas
```

For GPU acceleration:
```bash
pip install pennylane-lightning-gpu   # NVIDIA only
```

### Dataset

Set the path to the BALLADEER ADHD DATASET folder before running:

```bash
# Option A — environment variable (persists across runs)
export BALLADEER_DATA="/path/to/BALLADEER ADHD DATASET"

# Option B — command-line flag (one-off)
python main.py --data "/path/to/BALLADEER ADHD DATASET"
```

### Run

```bash
python main.py
```

All outputs are written to `RESULTS/` with a unique timestamp per run. The best metrics across all runs are tracked in `RESULTS/BEST_RESULTS.json`.

---

## Outputs per Run

| File | Contents |
|---|---|
| `run_*.json` | Full metrics, config snapshot, dataset info |
| `BEST_RESULTS.json` | All-time best Acc / F1 / ROC-AUC / PR-AUC across runs |
| `roc_curves_*.png` | ROC curves for all 7 models |
| `pr_curves_*.png` | Precision-recall curves |
| `confusion_matrices_*.png` | 7-panel confusion matrix grid |
| `shap_beeswarm_*.png` | SHAP feature attribution (XGBoost) |
| `shap_bar_*.png` | Mean SHAP importance bar chart |
| `permutation_importance_*.png` | Permutation importance (SVM) |
| `pca_loadings_*.png` | PCA component loadings heatmap |
| `vqc_encoding_*.png` | VQC attribution traced to original features |
| `calibration_*.png` | Calibration curves for all models |
| `mcnemar_heatmap_*.png` | Pairwise McNemar p-value matrix |
| `auc_diff_matrix_*.png` | Bootstrap AUC difference matrix |
| `significance_tests_*.json` | Full pairwise test results |
| `ablation_*.png` | Modality ablation AUC + F1 bar chart |
| `kta_*.png` | Kernel Target Alignment comparison |
| `vqc_noise_*.png` | VQC parameter noise robustness curve |
| `best_quantum_params_v40.json` | Best VQC trained parameters |

---

## Key Design Decisions

**Why quantum models?**
Quantum kernels can implicitly map data into exponentially large Hilbert spaces. For EEG data — where the meaningful signal is buried in cross-frequency interactions — this expressibility may capture structure that classical kernels miss.

**Why ex-Gaussian RT modelling?**
The exponential component τ of the ex-Gaussian distribution specifically measures response-time variability caused by attentional lapses, not motor speed. It is the most discriminative single cognitive marker of ADHD in the literature.

**Why multiple restarts + ensemble?**
Variational quantum circuits are highly non-convex. A single random initialisation will almost always converge to a suboptimal solution. Running 12 diverse initialisation strategies and ensembling the top 4 by validation AUC is the standard mitigation.

**Why OOF threshold calibration?**
Default 0.5 threshold is arbitrary on imbalanced data. Calibrating on out-of-fold predictions picks the F1-optimal decision boundary without ever touching the test set.

---
