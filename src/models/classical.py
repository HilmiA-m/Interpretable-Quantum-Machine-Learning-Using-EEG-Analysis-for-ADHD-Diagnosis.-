import subprocess
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_predict,
    RepeatedStratifiedKFold, cross_val_score,
)
from xgboost import XGBClassifier

from src.metrics import report


def _cuda_available():
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=3).returncode == 0
    except Exception:
        return False

_XGB_DEVICE = "cuda" if _cuda_available() else "cpu"


def run_classical_baselines(X_tr_cls, y_tr, X_te_cls, y_te, feat_names):
    print("\n[10] Classical baselines (5-fold CV, roc_auc, class_weight=balanced)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── SVM ──────────────────────────────────────────────────────────────────
    svm_grid = {
        "C":      [0.01, 0.1, 1, 10, 100, 1000],
        "gamma":  ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
        "kernel": ["rbf", "poly"],
    }
    svm = GridSearchCV(
        SVC(probability=True, class_weight="balanced", random_state=42),
        svm_grid, cv=cv, n_jobs=-1, scoring="roc_auc",
    ).fit(X_tr_cls, y_tr)
    p_svm = svm.predict_proba(X_te_cls)[:, 1]
    print(f"  SVM: {svm.best_params_}  CV={svm.best_score_:.3f}")
    p_svm_oof = cross_val_predict(
        SVC(probability=True, class_weight="balanced",
            random_state=42, **svm.best_params_),
        X_tr_cls, y_tr, cv=5, method="predict_proba")[:, 1]
    t_svm, y_pred_svm, _ = report("SVM", y_te, p_svm, y_tr, p_svm_oof)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_grid = {
        "n_estimators":    [500, 700, 1000, 1500],
        "max_depth":       [None, 10, 15, 20],
        "min_samples_leaf":[1, 2],
        "max_features":    ["sqrt", "log2"],
    }
    rf = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        rf_grid, cv=cv, n_jobs=-1, scoring="roc_auc",
    ).fit(X_tr_cls, y_tr)
    p_rf = rf.predict_proba(X_te_cls)[:, 1]
    print(f"  RF:  {rf.best_params_}  CV={rf.best_score_:.3f}")
    p_rf_oof = cross_val_predict(
        RandomForestClassifier(class_weight="balanced",
                               random_state=42, **rf.best_params_),
        X_tr_cls, y_tr, cv=5, method="predict_proba")[:, 1]
    t_rf, y_pred_rf, _ = report("RF", y_te, p_rf, y_tr, p_rf_oof)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    n_neg = int((y_tr == 0).sum()); n_pos = int(y_tr.sum())
    spw   = n_neg / n_pos
    xgb_grid = {
        "n_estimators":    [500, 700, 1000],
        "max_depth":       [3, 4, 5],
        "learning_rate":   [0.02, 0.05, 0.1],
        "subsample":       [0.7, 0.8, 0.9],
        "colsample_bytree":[0.6, 0.8],
        "min_child_weight":[1, 3],
        "gamma":           [0.1, 0.5],
        "reg_alpha":       [0.1, 1.0],
        "reg_lambda":      [1.0],
    }
    xgb = GridSearchCV(
        XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                      device=_XGB_DEVICE, random_state=42, n_jobs=1),
        xgb_grid, cv=cv, n_jobs=-1, scoring="roc_auc",
    ).fit(X_tr_cls, y_tr)
    p_xgb = xgb.predict_proba(X_te_cls)[:, 1]
    print(f"  XGB: {xgb.best_params_}  CV={xgb.best_score_:.3f}")
    p_xgb_oof = cross_val_predict(
        XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                      device=_XGB_DEVICE, random_state=42, n_jobs=1, **xgb.best_params_),
        X_tr_cls, y_tr, cv=5, method="predict_proba")[:, 1]
    t_xgb, y_pred_xgb, _ = report("XGBoost", y_te, p_xgb, y_tr, p_xgb_oof)

    # ── Feature importance (XGBoost on full train set) ────────────────────────
    print("\n[10b] Feature importance (XGBoost)...")
    xgb_final = XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                               device=_XGB_DEVICE, random_state=42, n_jobs=1,
                               **xgb.best_params_)
    xgb_final.fit(X_tr_cls, y_tr)
    imps     = xgb_final.feature_importances_
    n_plot   = min(len(feat_names), len(imps))
    feat_imp = sorted(zip(feat_names[:n_plot], imps[:n_plot]),
                      key=lambda x: x[1], reverse=True)
    print("  Top features:")
    for fn, fi in feat_imp[:10]:
        print(f"    {fn:30s} {fi:.4f}")

    return {
        "svm":           {"p": p_svm,  "y_pred": y_pred_svm,  "t": t_svm,
                          "params": svm.best_params_},
        "rf":            {"p": p_rf,   "y_pred": y_pred_rf,   "t": t_rf,
                          "params": rf.best_params_},
        "xgb":           {"p": p_xgb,  "y_pred": y_pred_xgb,  "t": t_xgb,
                          "params": xgb.best_params_},
        "feat_imp":      feat_imp,
        "spw":           spw,
        "xgb_model":     xgb_final,    # fitted model for SHAP
        "svm_estimator": svm,          # GridSearchCV → .best_estimator_ for permutation importance
    }


def run_repeated_cv(X_cls_all, y_all, seed=42):
    """Repeated 5-fold CV (3 repeats) on the full dataset — classical models only.

    Provides variance-aware performance estimates that are independent of the
    single 80/20 train/test split used for the main evaluation.
    """
    print("\n[10c] Repeated stratified 5-fold CV (3 repeats) — robustness check")
    rskf  = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    n_neg = int((y_all == 0).sum()); n_pos = int(y_all.sum())
    spw   = n_neg / max(n_pos, 1)

    models = {
        "SVM": Pipeline([
            ("sc", StandardScaler()),
            ("m",  SVC(probability=True, class_weight="balanced", random_state=seed)),
        ]),
        "RF": Pipeline([
            ("sc", StandardScaler()),
            ("m",  RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                          random_state=seed)),
        ]),
        "XGBoost": Pipeline([
            ("sc", StandardScaler()),
            ("m",  XGBClassifier(n_estimators=300, eval_metric="logloss",
                                 scale_pos_weight=spw, device=_XGB_DEVICE,
                                 random_state=seed, n_jobs=1)),
        ]),
    }

    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_cls_all, y_all,
                                 cv=rskf, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = {
            "mean": float(scores.mean()),
            "std":  float(scores.std()),
            "all":  scores.tolist(),
        }
        print(f"  {name:<10}: ROC-AUC = {scores.mean():.3f} ± {scores.std():.3f}"
              f"  (n={len(scores)} folds)")
    return cv_results
