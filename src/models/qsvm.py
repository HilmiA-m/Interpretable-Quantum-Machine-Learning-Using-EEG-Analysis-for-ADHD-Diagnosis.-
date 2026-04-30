import time
import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src import config
from src.metrics import report

def _pick_device(n_wires, label=""):
    for name, _ in [("lightning.gpu", "adjoint"),
                    ("lightning.qubit", "adjoint"),
                    ("default.qubit", "backprop")]:
        try:
            d = qml.device(name, wires=n_wires)
            print(f"    [{label}] {name}")
            return d
        except Exception:
            continue
    raise RuntimeError("No PennyLane device available")

_dev_k = _pick_device(config.N_QUBITS, "QSVM")


def _zz_map(x, n_reps=2):
    for _ in range(n_reps):
        for q in range(config.N_QUBITS):
            qml.Hadamard(wires=q)
        for q in range(config.N_QUBITS):
            qml.RZ(2.0 * x[q], wires=q)
        for q in range(config.N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % config.N_QUBITS])
            qml.RZ(2.0 * (np.pi - x[q]) * (np.pi - x[(q + 1) % config.N_QUBITS]),
                   wires=(q + 1) % config.N_QUBITS)
            qml.CNOT(wires=[q, (q + 1) % config.N_QUBITS])


@qml.qnode(_dev_k)
def _zz_kernel_circuit(x1, x2):
    _zz_map(x1)
    qml.adjoint(_zz_map)(x2)
    return qml.probs(wires=range(config.N_QUBITS))


def _quantum_kernel(A, B, label="kernel"):
    n_a, n_b = len(A), len(B)
    K        = np.zeros((n_a, n_b))
    is_sym   = A is B
    total    = (n_a * (n_a + 1)) // 2 if is_sym else n_a * n_b
    done     = 0
    t0       = time.time()
    for i in range(n_a):
        j_start = i if is_sym else 0
        for j in range(j_start, n_b):
            v = float(_zz_kernel_circuit(A[i], B[j])[0])
            K[i, j] = v
            if is_sym and i != j:
                K[j, i] = v
            done += 1
            if done % 200 == 0:
                pct = 100 * done / total
                eta = (time.time() - t0) * (total - done) / max(done, 1)
                print(f"      {label}: {done}/{total} ({pct:4.1f}%)  ETA {eta:.0f}s",
                      end="\r", flush=True)
    print(f"      {label}: {total}/{total} (100.0%) done in {time.time()-t0:.0f}s            ")
    return K


def _center_kernel(K_tr, K_te=None):
    col_means  = K_tr.mean(axis=0)
    row_means  = K_tr.mean(axis=1)
    total_mean = K_tr.mean()
    K_tr_c = K_tr - col_means[np.newaxis, :] - row_means[:, np.newaxis] + total_mean
    if K_te is None:
        return K_tr_c
    te_row_means = K_te.mean(axis=1)
    K_te_c = K_te - te_row_means[:, np.newaxis] - col_means[np.newaxis, :] + total_mean
    return K_tr_c, K_te_c


def run_qsvm(X_tr_raw, y_tr, X_te_q, y_te, ss_q, pca, mm):
    print(f"\n[8] QSVM: ZZ feature map (IQP, 2 reps, {config.N_QUBITS} qubits)")
    X_tr_q = mm.transform(pca.transform(ss_q.transform(X_tr_raw)))
    n_tr   = len(X_tr_q)

    if n_tr <= config.NYSTROEM_TRIGGER:
        print(f"    n_tr={n_tr} ≤ {config.NYSTROEM_TRIGGER} → full quantum kernel")
        K_tr = _quantum_kernel(X_tr_q, X_tr_q, label="K_tr")
        K_te = _quantum_kernel(X_te_q, X_tr_q, label="K_te")
        use_nystroem = False
    else:
        print(f"    n_tr={n_tr} > {config.NYSTROEM_TRIGGER} → Nyström (m={config.NYSTROEM_M})")
        rng          = np.random.default_rng(42)
        landmark_idx = rng.choice(n_tr, config.NYSTROEM_M, replace=False)
        L    = X_tr_q[landmark_idx]
        K_mm = _quantum_kernel(L, L, label="K_mm") + 1e-6 * np.eye(config.NYSTROEM_M)
        U, S, Vt   = np.linalg.svd(K_mm)
        S_inv_sqrt = np.where(S > 1e-12, 1.0 / np.sqrt(S), 0.0)
        norm_mat   = (U * S_inv_sqrt) @ Vt
        K_tr_nm = _quantum_kernel(X_tr_q, L, label="K_tr_nm")
        K_te_nm = _quantum_kernel(X_te_q, L, label="K_te_nm")
        Phi_tr  = K_tr_nm @ norm_mat.T
        Phi_te  = K_te_nm @ norm_mat.T
        K_tr    = Phi_tr @ Phi_tr.T
        K_te    = Phi_te @ Phi_tr.T
        use_nystroem = True

    if config.QSVM_KERNEL_CENTER:
        K_tr, K_te = _center_kernel(K_tr, K_te)
        print("    kernel centered")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_C, best_cv = 1.0, -1.0
    for C in config.QSVM_C_GRID:
        scores = []
        for tri, vai in skf.split(K_tr, y_tr):
            m = SVC(kernel="precomputed", C=C, probability=True, random_state=42)
            m.fit(K_tr[np.ix_(tri, tri)], y_tr[tri])
            pp = m.predict_proba(K_tr[np.ix_(vai, tri)])[:, 1]
            try:    scores.append(roc_auc_score(y_tr[vai], pp))
            except: pass
        mv = float(np.mean(scores)) if scores else -1.0
        if mv > best_cv:
            best_cv, best_C = mv, C
    nystr_tag = f" Nyström m={config.NYSTROEM_M}" if use_nystroem else " full"
    ctr_tag   = " centered" if config.QSVM_KERNEL_CENTER else ""
    print(f"    Best C={best_C}  (CV AUC={best_cv:.3f}) {nystr_tag}{ctr_tag}")

    qsvm = SVC(kernel="precomputed", C=best_C, probability=True, random_state=42)
    qsvm.fit(K_tr, y_tr)
    p_qsvm = qsvm.predict_proba(K_te)[:, 1]

    print("    Computing QSVM OOF threshold calibration...")
    qsvm_oof = np.zeros(len(y_tr))
    for tri, vai in skf.split(K_tr, y_tr):
        m = SVC(kernel="precomputed", C=best_C, probability=True, random_state=42)
        m.fit(K_tr[np.ix_(tri, tri)], y_tr[tri])
        qsvm_oof[vai] = m.predict_proba(K_tr[np.ix_(vai, tri)])[:, 1]

    t_qsvm, y_pred_qsvm, _ = report("QSVM-ZZ", y_te, p_qsvm, y_tr, qsvm_oof)
    return {"p": p_qsvm, "y_pred": y_pred_qsvm, "t": t_qsvm,
            "best_C": best_C, "use_nystroem": use_nystroem,
            "K_tr": K_tr,     # raw (possibly centered) training kernel matrix for KTA
            "X_tr_q": X_tr_q} # quantum-pipeline train features for KTA classical comparison
