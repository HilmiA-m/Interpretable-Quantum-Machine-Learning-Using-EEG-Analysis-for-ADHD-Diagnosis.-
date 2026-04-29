import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.model_selection import train_test_split
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

_dev_qcnn = _pick_device(config.N_QCNN_QUBITS, "QCNN")


def _conv(p, w):
    qml.U3(p[0], p[1], p[2], wires=w[0])
    qml.U3(p[3], p[4], p[5], wires=w[1])
    qml.CNOT(wires=w)
    qml.RY(p[6], wires=w[0])
    qml.RZ(p[7], wires=w[1])
    qml.CNOT(wires=[w[1], w[0]])


def _pool(p, w):
    qml.CRZ(p[0], wires=w)
    qml.PauliX(wires=w[0])
    qml.CRX(p[1], wires=w)
    qml.PauliX(wires=w[0])


@qml.qnode(_dev_qcnn, interface="autograd", diff_method="backprop")
def _qcnn_circuit(params, x):
    for q in range(config.N_QCNN_QUBITS):
        qml.RY(x[q], wires=q)
    _conv(params[0:8],   [0, 1]); _conv(params[8:16],  [2, 3])
    _conv(params[16:24], [4, 5]); _conv(params[24:32], [6, 7])
    _pool(params[32:34], [0, 1]); _pool(params[34:36], [2, 3])
    _pool(params[36:38], [4, 5]); _pool(params[38:40], [6, 7])
    _conv(params[40:48], [0, 2]); _conv(params[48:56], [4, 6])
    _pool(params[56:58], [0, 2]); _pool(params[58:60], [4, 6])
    _conv(params[60:68], [0, 4])
    _pool(params[68:70], [0, 4])
    return qml.expval(qml.PauliZ(0))


def _qcnn_proba(p, X):
    return (np.array([float(_qcnn_circuit(p, x)) for x in X]) + 1) / 2


def _qcnn_bce(params, Xb, yb):
    n_pos = max(sum(1 for yi in yb if yi == 1), 1)
    n_neg = max(len(yb) - n_pos, 1)
    w_pos = n_neg / n_pos
    tot   = pnp.array(0.0, requires_grad=True)
    for xi, yi in zip(Xb, yb):
        raw = _qcnn_circuit(params, xi)
        p   = pnp.clip((raw + 1) / 2, 1e-7, 1 - 1e-7)
        pt  = p if yi == 1 else (1 - p)
        w   = w_pos if yi == 1 else 1.0
        tot = tot - w * pnp.log(pt)
    return tot / len(Xb) + 0.0005 * pnp.sum(params ** 2)


def _qcnn_lr(ep):
    t = ep / max(1, config.QCNN_EPOCHS - 1)
    return config.QCNN_LR_MIN + 0.5 * (config.QCNN_LR - config.QCNN_LR_MIN) * (1 + np.cos(np.pi * t))


def run_qcnn(X_tr_raw, y_tr, X_te_q, y_te, ss_q, pca, mm):
    print(f"\n[9] QCNN-8: {config.N_QCNN_QUBITS}q, {config.N_QCNN_PARAMS}p, "
          f"{config.QCNN_RESTARTS}R×{config.QCNN_EPOCHS}ep, top-{config.QCNN_TOP_K}")

    X_in_raw, X_val_raw, y_in, y_val = train_test_split(
        X_tr_raw, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
    X_in  = mm.transform(pca.transform(ss_q.transform(X_in_raw)))
    X_val = mm.transform(pca.transform(ss_q.transform(X_val_raw)))

    records = []
    for r in range(config.QCNN_RESTARTS):
        pnp.random.seed(300 + r * 77)
        p   = pnp.array(0.1 * pnp.random.randn(config.N_QCNN_PARAMS), requires_grad=True)
        opt = qml.AdamOptimizer(config.QCNN_LR)
        best_p_r, best_auc_r, patience_r = p.copy(), -1.0, 0
        for ep in range(config.QCNN_EPOCHS):
            opt.stepsize = float(_qcnn_lr(ep))
            idx = np.random.choice(len(X_in), min(config.BATCH_SIZE, len(X_in)), replace=False)
            Xb, yb = X_in[idx], y_in[idx]
            p, _ = opt.step_and_cost(
                lambda pp, _X=Xb, _y=yb: _qcnn_bce(pp, _X, _y), p)
            if (ep + 1) % config.QCNN_VAL_EVERY == 0:
                vp = _qcnn_proba(p, X_val)
                try:    v_auc = roc_auc_score(y_val, vp)
                except: v_auc = 0.5
                if v_auc > best_auc_r + config.QCNN_MIN_DELTA:
                    best_auc_r, best_p_r, patience_r = v_auc, p.copy(), 0
                else:
                    patience_r += 1
                    if patience_r >= config.QCNN_PATIENCE:
                        break
        records.append({"params": best_p_r, "val_auc": best_auc_r})
        print(f"    r{r+1:2d}/{config.QCNN_RESTARTS} val_AUC={best_auc_r:.3f}")

    records.sort(key=lambda d: d["val_auc"], reverse=True)
    top_k  = records[:config.QCNN_TOP_K]
    w_k    = np.exp(np.array([rec["val_auc"] for rec in top_k]) / 0.5)
    w_k   /= w_k.sum()
    p_qcnn = np.average(
        np.stack([_qcnn_proba(rec["params"], X_te_q) for rec in top_k]),
        axis=0, weights=w_k)
    p_val  = np.average(
        np.stack([_qcnn_proba(rec["params"], X_val) for rec in top_k]),
        axis=0, weights=w_k)

    t_qcnn, y_pred_qcnn, _ = report("QCNN-8", y_te, p_qcnn, y_val, p_val)
    return {"p": p_qcnn, "y_pred": y_pred_qcnn, "t": t_qcnn}
