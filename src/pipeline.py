import os
import sys
import warnings
import random
from datetime import datetime
import numpy as np

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)


def run(skip_vqc=False, skip_qsvm=False, skip_qcnn=False):
    from src import config

    _ts     = datetime.now()
    _run_id = f"run_{_ts.strftime('%Y%m%d_%H%M%S')}"
    _ts_str = _ts.isoformat(timespec="seconds")

    print("=" * 72)
    print(f"BALLADEER — Quantum ML for ADHD  (v40: curated features + no SMOTE)")
    print(f"Run ID: {_run_id}")
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
    _dataset_info = {
        "n_subjects": int(len(y)),
        "n_adhd":     int(y.sum()),
        "n_control":  int((y == 0).sum()),
        "n_features": int(X_raw.shape[1]),
    }

    # ── 6. Preprocess ────────────────────────────────────────────────────────
    print(f"\n[6] Split → StandardScale → PCA-{config.N_PCA} (quantum) / raw (classical)")
    data = preprocess(X_raw, y, feat_names)
    feat_names = data["feat_names"]   # updated after zero-variance removal

    y_tr     = data["y_tr"]
    y_te     = data["y_te"]
    X_tr_raw = data["X_tr_raw"]
    X_te_q   = data["X_te_q"]
    ss_q, pca, mm = data["ss_q"], data["pca"], data["mm"]

    # ── 7. VQC ───────────────────────────────────────────────────────────────
    from src.models.vqc import run_vqc
    if skip_vqc:
        import json as _json
        import pennylane.numpy as pnp
        from src.metrics import report as _report
        _vp = os.path.join(config.RESULTS_DIR, "best_quantum_params_v40.json")
        if not os.path.exists(_vp):
            sys.exit(f"\n[FATAL] --skip-vqc requested but {_vp} not found.\n")
        with open(_vp) as _f:
            _vd = _json.load(_f)
        _params = pnp.array(_vd["params"], requires_grad=False)
        from src.models.vqc import vqc_proba
        p_vqc = vqc_proba(_params, X_te_q)
        _oof_half = np.full(len(y_tr), 0.5)
        t_vqc, y_pred_vqc, _ = _report("VQC (loaded)", y_te, p_vqc, y_tr, _oof_half)
        vqc = {"p": p_vqc, "y_pred": y_pred_vqc, "t": t_vqc,
               "n_good": 1, "top_params": np.array(_vd["params"])}
        print(f"    [skip-vqc] Loaded params from {_vp}  (val_AUC={_vd['val_auc']:.3f})")
    else:
        vqc = run_vqc(X_tr_raw, y_tr, X_te_q, y_te, ss_q, pca, mm)

    # ── 8. QSVM ──────────────────────────────────────────────────────────────
    from src.models.qsvm import run_qsvm
    if skip_qsvm:
        from src.metrics import report as _report
        p_qs = np.full(len(y_te), 0.5)
        t_qs, y_pred_qs, _ = _report("QSVM-ZZ (skipped)", y_te, p_qs, y_tr,
                                      np.full(len(y_tr), 0.5))
        K_dummy = np.eye(len(y_tr))
        X_tr_q_dummy = mm.transform(pca.transform(ss_q.transform(X_tr_raw)))
        qsvm = {"p": p_qs, "y_pred": y_pred_qs, "t": t_qs,
                "best_C": None, "use_nystroem": False,
                "K_tr": K_dummy, "X_tr_q": X_tr_q_dummy}
        print("    [skip-qsvm] QSVM skipped — dummy 0.5 predictions used.")
    else:
        qsvm = run_qsvm(X_tr_raw, y_tr, X_te_q, y_te, ss_q, pca, mm)

    # ── 9. QCNN ──────────────────────────────────────────────────────────────
    from src.models.qcnn import run_qcnn
    if skip_qcnn:
        from src.metrics import report as _report
        p_qc = np.full(len(y_te), 0.5)
        t_qc, y_pred_qc, _ = _report("QCNN-8 (skipped)", y_te, p_qc, y_tr,
                                      np.full(len(y_tr), 0.5))
        qcnn = {"p": p_qc, "y_pred": y_pred_qc, "t": t_qc}
        print("    [skip-qcnn] QCNN skipped — dummy 0.5 predictions used.")
    else:
        qcnn = run_qcnn(X_tr_raw, y_tr, X_te_q, y_te, ss_q, pca, mm)

    # ── 10. Classical baselines ──────────────────────────────────────────────
    from src.models.classical import run_classical_baselines, run_repeated_cv
    cls = run_classical_baselines(
        data["X_tr_cls"], y_tr, data["X_te_cls"], y_te, feat_names)

    # ── 10c. Repeated CV robustness check (classical only) ───────────────────
    # Full-dataset repeated K-fold gives variance-aware estimates independent
    # of the single 80/20 split. Quantum models are excluded (too slow to CV).
    from sklearn.preprocessing import StandardScaler as _SS
    _X_cls_all = _SS().fit_transform(X_raw)
    _cv_results = run_repeated_cv(_X_cls_all, y)

    # ── 10d. Modality ablation study ─────────────────────────────────────────
    from src.ablation import run_ablation
    _ablation = run_ablation(data["X_tr_raw"], y_tr, feat_names, _run_id)

    # ── 11. Results table ────────────────────────────────────────────────────
    from src.metrics import model_row
    results = {
        "VQC":     model_row(y_te, vqc["y_pred"],        vqc["p"]),
        "QSVM-ZZ": model_row(y_te, qsvm["y_pred"],       qsvm["p"]),
        "QCNN-8":  model_row(y_te, qcnn["y_pred"],       qcnn["p"]),
        "LR":      model_row(y_te, cls["lr"]["y_pred"],   cls["lr"]["p"]),
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

    # ── 11b. Pairwise statistical significance tests ──────────────────────────
    probas     = {"VQC": vqc["p"],        "QSVM-ZZ": qsvm["p"],      "QCNN-8": qcnn["p"],
                  "LR":  cls["lr"]["p"],  "SVM": cls["svm"]["p"],    "RF": cls["rf"]["p"],
                  "XGBoost": cls["xgb"]["p"]}
    preds      = {"VQC": vqc["y_pred"],   "QSVM-ZZ": qsvm["y_pred"], "QCNN-8": qcnn["y_pred"],
                  "LR":  cls["lr"]["y_pred"], "SVM": cls["svm"]["y_pred"],
                  "RF":  cls["rf"]["y_pred"], "XGBoost": cls["xgb"]["y_pred"]}
    thresholds = {"VQC": vqc["t"],        "QSVM-ZZ": qsvm["t"],      "QCNN-8": qcnn["t"],
                  "LR":  cls["lr"]["t"],  "SVM": cls["svm"]["t"],    "RF": cls["rf"]["t"],
                  "XGBoost": cls["xgb"]["t"]}
    from src.stats import run_significance_tests
    run_significance_tests(y_te, preds, probas, _run_id)

    # ── 12. Figures ──────────────────────────────────────────────────────────
    from src.figures import save_figures
    save_figures(y_te, probas, preds, thresholds, cls["feat_imp"], run_id=_run_id)

    # ── 13. Save JSON results ────────────────────────────────────────────────
    from src.results import save_run, update_best
    print(f"\n[13] Saving results → RESULTS/{_run_id}.json  +  BEST_RESULTS.json")
    _config_snapshot = {
        "n_qubits": config.N_QUBITS, "n_layers": config.N_LAYERS,
        "n_pca":    config.N_PCA,    "n_restarts": config.N_RESTARTS,
        "epochs":   config.EPOCHS,   "ensemble_k": config.ENSEMBLE_K,
        "test_size": config.TEST_SIZE,
    }
    _payload = save_run(_run_id, _ts_str, results, _dataset_info, _config_snapshot)
    update_best(_payload)

    # ── 14. Interpretability ─────────────────────────────────────────────────
    from src.interpretability import run_all as _run_interp
    _run_interp(
        xgb_model     = cls["xgb_model"],
        svm_estimator = cls["svm_estimator"],
        pca           = pca,
        vqc_top_params= vqc["top_params"],
        X_tr_cls      = data["X_tr_cls"],
        X_te_cls      = data["X_te_cls"],
        y_te          = y_te,
        feat_names    = feat_names,
        run_id        = _run_id,
        probas        = probas,
    )

    # ── 15. Quantum analyses (KTA + VQC noise robustness) ────────────────────
    from src.quantum_analysis import run_all as _run_quantum
    _run_quantum(
        K_tr_quantum   = qsvm["K_tr"],
        X_tr_q         = qsvm["X_tr_q"],
        X_te_q         = data["X_te_q"],
        y_tr           = y_tr,
        y_te           = y_te,
        vqc_top_params = vqc["top_params"],
        run_id         = _run_id,
    )

    # ── 16. Summary ──────────────────────────────────────────────────────────
    N_PARAMS_VQC = _N_ENC = config.N_LAYERS * config.N_QUBITS * 3
    N_PARAMS_VQC = N_PARAMS_VQC * 2 + config.N_QUBITS + 1
    nystr_tag = f" Nyström m={config.NYSTROEM_M}" if qsvm["use_nystroem"] else " full"
    ctr_tag   = " centered" if config.QSVM_KERNEL_CENTER else ""
    print(f"""
{'='*72}
BALLADEER v40 — COMPLETE (curated features, no SMOTE)
{'='*72}
Run ID      : {_run_id}
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
QCNN-8  : {config.N_QCNN_QUBITS}q, {config.N_QCNN_PARAMS}p (70 circ+8 readout+1), {config.QCNN_RESTARTS}R×{config.QCNN_EPOCHS}ep, top-{config.QCNN_TOP_K}
LR      : class_weight=balanced, {cls['lr']['params']}
SVM     : class_weight=balanced, {cls['svm']['params']}
RF      : class_weight=balanced, {cls['rf']['params']}
XGBoost : scale_pos_weight={cls['spw']:.2f}, {cls['xgb']['params']}

Best Acc    : {best_acc} = {results[best_acc]['acc']:.3f}
Best F1     : {best_f1}  = {results[best_f1]['f1']:.3f}
Best ROC-AUC: {best_roc} = {results[best_roc]['roc_auc']:.3f}
Best PR-AUC : {best_pr}  = {results[best_pr]['pr_auc']:.3f}

Repeated CV (classical, 5-fold × 3):
  SVM     : {_cv_results['SVM']['mean']:.3f} ± {_cv_results['SVM']['std']:.3f}
  RF      : {_cv_results['RF']['mean']:.3f} ± {_cv_results['RF']['std']:.3f}
  XGBoost : {_cv_results['XGBoost']['mean']:.3f} ± {_cv_results['XGBoost']['std']:.3f}

Outputs (RESULTS/):
  {_run_id}.json                    ← full metrics + config snapshot
  BEST_RESULTS.json                 ← all-time bests across runs
  roc_curves_{_run_id}.png
  pr_curves_{_run_id}.png
  confusion_matrices_{_run_id}.png
  feature_importance_{_run_id}.png
  shap_beeswarm_{_run_id}.png
  shap_bar_{_run_id}.png
  permutation_importance_{_run_id}.png
  pca_loadings_{_run_id}.png
  vqc_encoding_{_run_id}.png
  calibration_{_run_id}.png
  mcnemar_heatmap_{_run_id}.png     ← pairwise McNemar p-values
  auc_diff_matrix_{_run_id}.png     ← bootstrap AUC differences
  significance_tests_{_run_id}.json
  ablation_{_run_id}.png            ← modality ablation bar chart
  ablation_{_run_id}.json
  kta_{_run_id}.png                 ← kernel target alignment
  kta_{_run_id}.json
  vqc_noise_{_run_id}.png           ← VQC noise robustness
  vqc_noise_{_run_id}.json
  best_quantum_params_v40.json
{'='*72}
""")
