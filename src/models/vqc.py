import os
import sys
import time
import subprocess
import tempfile
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.model_selection import train_test_split

from src import config
from src.metrics import report

try:
    _dev  = qml.device("lightning.qubit", wires=config.N_QUBITS)
    _DIFF = "adjoint"
    print("    [VQC] lightning.qubit (adjoint)")
except Exception:
    _dev  = qml.device("default.qubit", wires=config.N_QUBITS)
    _DIFF = "backprop"
    print("    [VQC] default.qubit (backprop)")

_N_ENC    = config.N_LAYERS * config.N_QUBITS * 3
_N_ROT    = _N_ENC
_N_LIN    = config.N_QUBITS
_N_PARAMS = _N_ENC + _N_ROT + _N_LIN + 1
_CZ_PAIRS = [(i, j) for i in range(config.N_QUBITS) for j in range(i + 1, config.N_QUBITS)]

_RESTART_PLAN = [
    (542,  "decay"),          (142,  "small_enc"),     (442,  "uniform"),
    (502,  "decay"),          (522,  "decay"),          (562,  "decay"),
    (582,  "decay"),          (602,  "decay"),          (642,  "decay"),
    (742,  "decay"),          (842,  "warm"),           (42,   "near_identity"),
    (242,  "large_enc"),      (342,  "random_rot"),     (942,  "small_enc_v2"),
    (1042, "uniform_narrow"), (1142, "warm"),           (1242, "small_enc"),
]


@qml.qnode(_dev, interface="autograd", diff_method=_DIFF)
def vqc_circuit(params, x):
    enc_s = params[:_N_ENC].reshape(config.N_LAYERS, config.N_QUBITS, 3)
    rot_a = params[_N_ENC:_N_ENC + _N_ROT].reshape(config.N_LAYERS, config.N_QUBITS, 3)
    for l in range(config.N_LAYERS):
        for q in range(config.N_QUBITS):
            qml.RX(enc_s[l, q, 0] * x[q], wires=q)
            qml.RY(enc_s[l, q, 1] * x[q], wires=q)
            qml.RZ(enc_s[l, q, 2] * x[q], wires=q)
        for q in range(config.N_QUBITS):
            qml.Rot(rot_a[l, q, 0], rot_a[l, q, 1], rot_a[l, q, 2], wires=q)
        for (a, b) in _CZ_PAIRS:
            qml.CZ(wires=[a, b])
    return [qml.expval(qml.PauliZ(q)) for q in range(config.N_QUBITS)]


def vqc_forward(params, x):
    w   = params[_N_ENC + _N_ROT: _N_ENC + _N_ROT + _N_LIN]
    b   = params[-1]
    evs = vqc_circuit(params, x)
    return 1.0 / (1.0 + pnp.exp(-pnp.clip(b + pnp.sum(w * pnp.stack(evs)), -15, 15)))


def vqc_proba(params, X):
    return np.array([float(vqc_forward(params, x)) for x in X])


def init_params(seed, strat):
    pnp.random.seed(seed)
    if strat == "near_identity":
        enc = pnp.ones(_N_ENC) + 0.05 * pnp.random.randn(_N_ENC)
        rot = 0.1 * pnp.random.randn(_N_ROT)
    elif strat == "small_enc":
        enc = 0.5 * pnp.ones(_N_ENC) + 0.1 * pnp.random.randn(_N_ENC)
        rot = 0.2 * pnp.random.randn(_N_ROT)
    elif strat == "large_enc":
        enc = 1.5 * pnp.ones(_N_ENC) + 0.1 * pnp.random.randn(_N_ENC)
        rot = 0.1 * pnp.random.randn(_N_ROT)
    elif strat == "random_rot":
        enc = pnp.ones(_N_ENC) + 0.05 * pnp.random.randn(_N_ENC)
        rot = 0.5 * pnp.random.randn(_N_ROT)
    elif strat == "uniform":
        enc = pnp.array(pnp.random.uniform(0.5, 1.5, _N_ENC))
        rot = pnp.array(pnp.random.uniform(-np.pi / 4, np.pi / 4, _N_ROT))
    elif strat == "decay":
        enc = pnp.linspace(1.5, 0.5, _N_ENC) + 0.05 * pnp.random.randn(_N_ENC)
        rot = 0.15 * pnp.random.randn(_N_ROT)
    elif strat == "warm":
        enc = 1.2 * pnp.ones(_N_ENC) + 0.08 * pnp.random.randn(_N_ENC)
        rot = 0.08 * pnp.random.randn(_N_ROT)
    elif strat == "small_enc_v2":
        enc = 0.3 * pnp.ones(_N_ENC) + 0.08 * pnp.random.randn(_N_ENC)
        rot = 0.15 * pnp.random.randn(_N_ROT)
    elif strat == "uniform_narrow":
        enc = pnp.array(pnp.random.uniform(0.7, 1.3, _N_ENC))
        rot = pnp.array(pnp.random.uniform(-np.pi / 6, np.pi / 6, _N_ROT))
    else:
        enc = pnp.ones(_N_ENC) + 0.1 * pnp.random.randn(_N_ENC)
        rot = 0.1 * pnp.random.randn(_N_ROT)
    return pnp.array(
        pnp.concatenate([enc, rot, 0.1 * pnp.random.randn(_N_LIN), pnp.array([0.0])]),
        requires_grad=True,
    )


def run_vqc(X_tr_raw, y_tr, X_te_q, y_te, ss_q, pca, mm):
    print(f"\n[7] VQC: {config.N_QUBITS}q × {config.N_LAYERS}L, full CZ, "
          f"{config.N_RESTARTS}R × {config.EPOCHS}ep")

    X_tr_in, X_val, y_tr_in, y_val = train_test_split(
        X_tr_raw, y_tr, test_size=0.25, random_state=42, stratify=y_tr)
    X_tr_is_q = mm.transform(pca.transform(ss_q.transform(X_tr_in)))
    X_val_q   = mm.transform(pca.transform(ss_q.transform(X_val)))

    hyperparams = dict(
        N_QUBITS=config.N_QUBITS, N_LAYERS=config.N_LAYERS, EPOCHS=config.EPOCHS,
        LR_INIT=config.LR_INIT,   LR_MIN=config.LR_MIN,     WARMUP=config.WARMUP,
        BATCH_SIZE=config.BATCH_SIZE, PATIENCE=config.PATIENCE, MIN_DELTA=config.MIN_DELTA,
        VAL_EVERY=config.VAL_EVERY, FOCAL_ALPHA=config.FOCAL_ALPHA,
        FOCAL_GAMMA=config.FOCAL_GAMMA, L2_REG=config.L2_REG,
    )
    tmp = tempfile.gettempdir()
    restart_records = []

    print(f"    {config.N_RESTARTS} restarts × {config.EPOCHS} ep  "
          f"(subprocess hard-kill after {config.MAX_RESTART_SECS}s)")
    for r in range(config.N_RESTARTS):
        seed, strat = _RESTART_PLAN[r % len(_RESTART_PLAN)]
        args_path   = os.path.join(tmp, f"_vqc_args_{r}.npy")
        result_path = os.path.join(tmp, f"_vqc_result_{r}.npy")
        np.save(args_path, dict(seed=seed, strat=strat,
                                X_tr=X_tr_is_q, y_tr=y_tr_in,
                                X_val=X_val_q,  y_val=y_val, **hyperparams))
        t0 = time.time()
        try:
            subprocess.run(
                [sys.executable, config.WORKER_SCRIPT, args_path, result_path],
                timeout=config.MAX_RESTART_SECS,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if os.path.exists(result_path):
                res    = np.load(result_path, allow_pickle=True).item()
                best_p = pnp.array(res["params"], requires_grad=True)
                restart_records.append({
                    "params":    best_p,
                    "val_auc":   res["val_auc"],
                    "threshold": res["threshold"],
                    "n_epochs":  res.get("n_epochs", -1),
                })
                print(f"    r{r+1:2d} [{strat:14s}]: val_AUC={res['val_auc']:.3f}  "
                      f"thr={res['threshold']:.3f}  ep={res.get('n_epochs',-1):3d}  "
                      f"{time.time()-t0:.0f}s")
            else:
                print(f"    r{r+1:2d} [{strat:14s}]: failed (no result)  {time.time()-t0:.0f}s")
                restart_records.append({"params": init_params(seed, strat),
                                        "val_auc": -1.0, "threshold": 0.5, "n_epochs": 0})
        except subprocess.TimeoutExpired:
            print(f"    r{r+1:2d} [{strat:14s}]: TIMEOUT {config.MAX_RESTART_SECS}s — killed")
            restart_records.append({"params": init_params(seed, strat),
                                    "val_auc": -1.0, "threshold": 0.5, "n_epochs": 0})
        except Exception as e:
            print(f"    r{r+1:2d} [{strat:14s}]: ERROR {e}")
            restart_records.append({"params": init_params(seed, strat),
                                    "val_auc": -1.0, "threshold": 0.5, "n_epochs": 0})
        for p in (args_path, result_path):
            try: os.remove(p)
            except: pass

    restart_records.sort(key=lambda d: d["val_auc"], reverse=True)
    good_k = [rec for rec in restart_records if rec["val_auc"] > 0.0]
    top_k  = (good_k[:config.ENSEMBLE_K] if len(good_k) >= config.ENSEMBLE_K
              else (good_k if good_k else restart_records[:1]))
    if good_k:
        aucs = [rec["val_auc"] for rec in good_k]
        print(f"    {len(good_k)}/{config.N_RESTARTS} succeeded — "
              f"val_AUC mean={np.mean(aucs):.3f}  max={max(aucs):.3f}  min={min(aucs):.3f}")
    if len(top_k) < config.ENSEMBLE_K:
        print(f"    WARNING: only {len(top_k)}/{config.ENSEMBLE_K} restarts succeeded")

    w_k = np.exp(np.array([rec["val_auc"] for rec in top_k]) / 0.5)
    w_k /= w_k.sum()
    p_vqc_val = np.average(
        np.stack([vqc_proba(rec["params"], X_val_q) for rec in top_k]),
        axis=0, weights=w_k)
    p_vqc = np.average(
        np.stack([vqc_proba(rec["params"], X_te_q) for rec in top_k]),
        axis=0, weights=w_k)
    np.save(os.path.join(config.DATA_ROOT, "best_quantum_params_v40.npy"),
            np.array(top_k[0]["params"]))

    t_vqc, y_pred_vqc, _ = report("VQC", y_te, p_vqc, y_val, p_vqc_val)
    return {"p": p_vqc, "y_pred": y_pred_vqc, "t": t_vqc, "n_good": len(good_k)}
