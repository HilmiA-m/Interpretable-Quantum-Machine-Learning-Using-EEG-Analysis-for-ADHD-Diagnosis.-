import os
import sys
import warnings
import random
import numpy as np

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)


def run():
    from src import config

    print("=" * 72)
    print("BALLADEER — Quantum ML for ADHD  (v40: curated features + no SMOTE)")
    print("=" * 72)

    if not os.path.exists(config.WORKER_SCRIPT):
        sys.exit(
            f"\n[FATAL] VQC runner not found: {config.WORKER_SCRIPT}\n"
            f"        Ensure workers/vqc_subprocess_runner.py is at the project root.\n"
        )
    print(f"[runner] {config.WORKER_SCRIPT} OK")

    # ── 1. Demographics ──────────────────────────────────────────────────────
    from src.data.demographics import load_demographics
    user_meta = load_demographics()

    # ── 2. EEG feature probe ─────────────────────────────────────────────────
    from src.data.eeg import extract_eeg_features, find_eeg_file
    _probe = None
    for uid in list(user_meta)[:10]:
        p = find_eeg_file(uid, "SlacklineLvl1")
        if p:
            _probe = extract_eeg_features(p)
            if _probe is not None:
                break
    n_eeg_feats = len(_probe) if _probe is not None else 15
    print(f"\n[2] EEG: {n_eeg_feats} curated feats/activity (averaged across 3 levels)")

    # ── 3. Behavioural note (extraction happens inside build_dataset) ─────────
    print("[3] Extracting behavioural features (TAGS + GAME_DATA)...")

    # ── 4. Embrace+ ──────────────────────────────────────────────────────────
    from src.data.embrace import load_embrace
    embrace_lookup, embrace_cols = load_embrace()

    # ── 5. Build feature matrix ──────────────────────────────────────────────
    print("\n[5] Building curated feature matrices (avg across levels)...")
    from src.dataset import build_dataset, preprocess
    X_raw, y, _uid_list, feat_names = build_dataset(
        user_meta, embrace_lookup, embrace_cols)

    # ── 6. Preprocess ────────────────────────────────────────────────────────
    print(f"\n[6] Split → StandardScale → PCA-{config.N_PCA} (quantum) / raw (classical)")
    data = preprocess(X_raw, y)

    y_tr     = data["y_tr"]
    y_te     = data["y_te"]
    X_tr_raw = data["X_tr_raw"]
    X_te_q   = data["X_te_q"]
    ss_q, pca, mm = data["ss_q"], data["pca"], data["mm"]

    # ── 7. VQC ───────────────────────────────────────────────────────────────
    from src.models.vqc import run_vqc
    vqc = run_vqc(X_tr_raw, y_tr, X_te_q, y_te, ss_q, pca, mm)

    # ── 8. QSVM ──────────────────────────────────────────────────────────────
    from src.models.qsvm import run_qsvm
    qsvm = run_qsvm(X_tr_raw, y_tr, X_te_q, y_te, ss_q, pca, mm)

    # ── 9. QCNN ──────────────────────────────────────────────────────────────
    from src.models.qcnn import run_qcnn
    qcnn = run_qcnn(X_tr_raw, y_tr, X_te_q, y_te, ss_q, pca, mm)

    # ── 10. Classical baselines ──────────────────────────────────────────────
    from src.models.classical import run_classical_baselines
    cls = run_classical_baselines(
        data["X_tr_cls"], y_tr, data["X_te_cls"], y_te, feat_names)

    # ── 11. Results table ────────────────────────────────────────────────────
    from src.metrics import model_row
    results = {
        "VQC":     model_row(y_te, vqc["y_pred"],        vqc["p"]),
        "QSVM-ZZ": model_row(y_te, qsvm["y_pred"],       qsvm["p"]),
        "QCNN-8":  model_row(y_te, qcnn["y_pred"],       qcnn["p"]),
        "SVM":     model_row(y_te, cls["svm"]["y_pred"],  cls["svm"]["p"]),
        "RF":      model_row(y_te, cls["rf"]["y_pred"],   cls["rf"]["p"]),
        "XGBoost": model_row(y_te, cls["xgb"]["y_pred"],  cls["xgb"]["p"]),
    }

    print("\n" + "=" * 90)
    print(f"  {'Model':<14}  {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} "
          f"{'ROC-AUC':>9} {'PR-AUC':>8}")
    print("  " + "─" * 78)
    for name, m in results.items():
        flag = " ◄" if m["acc"] >= 0.80 else ""
        print(f"  {name:<14}  {m['acc']:>7.3f} {m['prec']:>7.3f} {m['rec']:>7.3f} "
              f"{m['f1']:>7.3f} {m['roc_auc']:>9.3f} {m['pr_auc']:>8.3f}{flag}")
    print("=" * 90)
    best_roc = max(results, key=lambda k: results[k]["roc_auc"])
    best_pr  = max(results, key=lambda k: results[k]["pr_auc"])
    best_acc = max(results, key=lambda k: results[k]["acc"])
    best_f1  = max(results, key=lambda k: results[k]["f1"])
    print(f"\n  Best Acc    : {best_acc} = {results[best_acc]['acc']:.3f}")
    print(f"  Best F1     : {best_f1}  = {results[best_f1]['f1']:.3f}")
    print(f"  Best ROC-AUC: {best_roc} = {results[best_roc]['roc_auc']:.3f}")
    print(f"  Best PR-AUC : {best_pr}  = {results[best_pr]['pr_auc']:.3f}")

    # ── 12. Figures ──────────────────────────────────────────────────────────
    from src.figures import save_figures
    probas     = {"VQC": vqc["p"],       "QSVM-ZZ": qsvm["p"],      "QCNN-8": qcnn["p"],
                  "SVM": cls["svm"]["p"], "RF": cls["rf"]["p"],      "XGBoost": cls["xgb"]["p"]}
    preds      = {"VQC": vqc["y_pred"],  "QSVM-ZZ": qsvm["y_pred"], "QCNN-8": qcnn["y_pred"],
                  "SVM": cls["svm"]["y_pred"], "RF": cls["rf"]["y_pred"],
                  "XGBoost": cls["xgb"]["y_pred"]}
    thresholds = {"VQC": vqc["t"],       "QSVM-ZZ": qsvm["t"],      "QCNN-8": qcnn["t"],
                  "SVM": cls["svm"]["t"], "RF": cls["rf"]["t"],      "XGBoost": cls["xgb"]["t"]}
    save_figures(y_te, probas, preds, thresholds, cls["feat_imp"])

    # ── 13. Summary ──────────────────────────────────────────────────────────
    N_PARAMS_VQC = _N_ENC = config.N_LAYERS * config.N_QUBITS * 3
    N_PARAMS_VQC = N_PARAMS_VQC * 2 + config.N_QUBITS + 1
    nystr_tag = f" Nyström m={config.NYSTROEM_M}" if qsvm["use_nystroem"] else " full"
    ctr_tag   = " centered" if config.QSVM_KERNEL_CENTER else ""
    print(f"""
{'='*72}
BALLADEER v40 — COMPLETE (curated features, no SMOTE)
{'='*72}
Subjects    : {len(y)} | ADHD={int(y.sum())} Control={int((y == 0).sum())}
Features    : {X_raw.shape[1]} curated (same for quantum + classical)
Q pipeline  : {X_raw.shape[1]} → PCA-{config.N_PCA} → [0,π]
CLS pipeline: {X_raw.shape[1]} → StdScale
Threshold   : F1-OOF (calibrated on val/OOF, never on test)
SMOTE       : REMOVED (class_weight=balanced instead)

VQC     : {config.N_QUBITS}q×{config.N_LAYERS}L, full CZ, {N_PARAMS_VQC}p, {config.N_RESTARTS}R×{config.EPOCHS}ep, top-{config.ENSEMBLE_K}
          runner: {config.WORKER_SCRIPT}
          {vqc['n_good']}/{config.N_RESTARTS} restarts succeeded
QSVM-ZZ : ZZ/IQP 2 reps, C={qsvm['best_C']}{nystr_tag}{ctr_tag}
QCNN-8  : {config.N_QCNN_QUBITS}q, {config.N_QCNN_PARAMS}p, {config.QCNN_RESTARTS}R×{config.QCNN_EPOCHS}ep, top-{config.QCNN_TOP_K}
SVM     : class_weight=balanced, {cls['svm']['params']}
RF      : class_weight=balanced, {cls['rf']['params']}
XGBoost : scale_pos_weight={cls['spw']:.2f}, {cls['xgb']['params']}

Best Acc    : {best_acc} = {results[best_acc]['acc']:.3f}
Best F1     : {best_f1}  = {results[best_f1]['f1']:.3f}
Best ROC-AUC: {best_roc} = {results[best_roc]['roc_auc']:.3f}
Best PR-AUC : {best_pr}  = {results[best_pr]['pr_auc']:.3f}

Outputs: roc_curves_v40.png  pr_curves_v40.png
         confusion_matrices_v40.png  feature_importance_v40.png
         best_quantum_params_v40.npy
{'='*72}
""")
