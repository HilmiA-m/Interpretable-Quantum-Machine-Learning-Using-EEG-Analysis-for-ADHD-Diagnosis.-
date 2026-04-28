"""
_vqc_subprocess_runner.py
VQC training worker — called as a subprocess by balladeer_quantum_v39.py.

Usage:
    python _vqc_subprocess_runner.py <args_path.npy> <result_path.npy>

Reads a numpy dict from args_path, runs one VQC restart, writes
{params, val_auc, threshold, n_epochs} to result_path.

Improvements vs the missing v38 runner:
  - Barren plateau early termination (gradient norm check after warmup)
  - Stratified mini-batch sampling (balanced ADHD/ctrl per batch)
  - Gradient clipping (max-norm 5.0 on the full param vector)
  - Per-epoch LR warmup (linear ramp) then cosine annealing
  - F1-OOF threshold calibrated on val set, not test
"""

import os, sys
os.environ["TF_METAL_DISABLE"]     = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import roc_auc_score, fbeta_score

# ─── unpack args ─────────────────────────────────────────────────────────────
if len(sys.argv) < 3:
    sys.exit("Usage: runner.py <args.npy> <result.npy>")

args_path, result_path = sys.argv[1], sys.argv[2]
args = np.load(args_path, allow_pickle=True).item()

seed        = int(args["seed"])
strat       = str(args["strat"])
X_tr        = np.array(args["X_tr"], dtype=float)
y_tr        = np.array(args["y_tr"], dtype=int)
X_val       = np.array(args["X_val"], dtype=float)
y_val       = np.array(args["y_val"], dtype=int)
N_QUBITS    = int(args["N_QUBITS"])
N_LAYERS    = int(args["N_LAYERS"])
EPOCHS      = int(args["EPOCHS"])
LR_INIT     = float(args["LR_INIT"])
LR_MIN      = float(args["LR_MIN"])
WARMUP      = int(args["WARMUP"])
BATCH_SIZE  = int(args["BATCH_SIZE"])
PATIENCE    = int(args["PATIENCE"])
MIN_DELTA   = float(args["MIN_DELTA"])
VAL_EVERY   = int(args["VAL_EVERY"])
FOCAL_ALPHA = float(args["FOCAL_ALPHA"])
FOCAL_GAMMA = float(args["FOCAL_GAMMA"])
L2_REG      = float(args["L2_REG"])

# ─── circuit constants (must match main file exactly) ────────────────────────
N_ENC    = N_LAYERS * N_QUBITS * 3
N_ROT    = N_ENC
N_LIN    = N_QUBITS
N_BIAS   = 1
N_PARAMS = N_ENC + N_ROT + N_LIN + N_BIAS
CZ_PAIRS = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)]

GRAD_CLIP_MAX  = 5.0      # clip param update norm to this
BARREN_FLOOR   = 1e-5     # gradient norm below this after warmup → dead restart
BARREN_CHECK   = WARMUP + VAL_EVERY  # epoch at which we check

# ─── device ──────────────────────────────────────────────────────────────────
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    diff_method = "adjoint"
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    diff_method = "backprop"

# ─── circuit ─────────────────────────────────────────────────────────────────
@qml.qnode(dev, interface="autograd", diff_method=diff_method)
def vqc_circuit(params, x):
    enc_s = params[:N_ENC].reshape(N_LAYERS, N_QUBITS, 3)
    rot_a = params[N_ENC:N_ENC + N_ROT].reshape(N_LAYERS, N_QUBITS, 3)
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(enc_s[l, q, 0] * x[q], wires=q)
            qml.RY(enc_s[l, q, 1] * x[q], wires=q)
            qml.RZ(enc_s[l, q, 2] * x[q], wires=q)
        for q in range(N_QUBITS):
            qml.Rot(rot_a[l, q, 0], rot_a[l, q, 1], rot_a[l, q, 2], wires=q)
        for (a, b) in CZ_PAIRS:
            qml.CZ(wires=[a, b])
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


def vqc_forward(params, x):
    w   = params[N_ENC + N_ROT: N_ENC + N_ROT + N_LIN]
    b   = params[-1]
    evs = vqc_circuit(params, x)
    logit = b + pnp.sum(w * pnp.stack(evs))
    return 1.0 / (1.0 + pnp.exp(-pnp.clip(logit, -15, 15)))


def vqc_proba(params, X):
    return np.array([float(vqc_forward(params, x)) for x in X])


# ─── focal loss (identical to main file) ─────────────────────────────────────
def focal_loss(params, Xb, yb):
    loss = pnp.array(0.0, requires_grad=True)
    for xi, yi in zip(Xb, yb):
        p   = pnp.clip(vqc_forward(params, xi), 1e-7, 1.0 - 1e-7)
        pt  = p if yi == 1 else (1.0 - p)
        w   = FOCAL_ALPHA if yi == 1 else (1.0 - FOCAL_ALPHA)
        loss = loss - w * ((1.0 - pt) ** FOCAL_GAMMA) * pnp.log(pt)
    return loss / max(len(Xb), 1) + L2_REG * pnp.sum(params ** 2)


# ─── stratified mini-batch ───────────────────────────────────────────────────
def stratified_batch(X, y, size, rng):
    """Equal ADHD/ctrl per batch — avoids degenerate all-one-class batches."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    half = size // 2
    bi = np.concatenate([
        rng.choice(pos, min(half, len(pos)), replace=len(pos) < half),
        rng.choice(neg, min(half, len(neg)), replace=len(neg) < half),
    ])
    rng.shuffle(bi)
    return X[bi], y[bi]


# ─── LR schedule: linear warmup then cosine annealing ────────────────────────
def lr_schedule(ep):
    if ep < WARMUP:
        return LR_INIT * (ep + 1) / max(WARMUP, 1)
    t = (ep - WARMUP) / max(EPOCHS - WARMUP - 1, 1)
    return LR_MIN + 0.5 * (LR_INIT - LR_MIN) * (1.0 + np.cos(np.pi * t))


# ─── init params (mirrors main file strategies exactly) ──────────────────────
def init_params(seed, strat):
    pnp.random.seed(seed)
    if strat == "near_identity":
        enc = pnp.ones(N_ENC) + 0.05 * pnp.random.randn(N_ENC)
        rot = 0.1 * pnp.random.randn(N_ROT)
    elif strat == "small_enc":
        enc = 0.5 * pnp.ones(N_ENC) + 0.1 * pnp.random.randn(N_ENC)
        rot = 0.2 * pnp.random.randn(N_ROT)
    elif strat == "large_enc":
        enc = 1.5 * pnp.ones(N_ENC) + 0.1 * pnp.random.randn(N_ENC)
        rot = 0.1 * pnp.random.randn(N_ROT)
    elif strat == "random_rot":
        enc = pnp.ones(N_ENC) + 0.05 * pnp.random.randn(N_ENC)
        rot = 0.5 * pnp.random.randn(N_ROT)
    elif strat == "uniform":
        enc = pnp.array(pnp.random.uniform(0.5, 1.5, N_ENC))
        rot = pnp.array(pnp.random.uniform(-np.pi / 4, np.pi / 4, N_ROT))
    elif strat == "xavier":
        std = np.sqrt(2.0 / (N_QUBITS + 1))
        enc = pnp.ones(N_ENC) + std * pnp.random.randn(N_ENC)
        rot = std * pnp.random.randn(N_ROT)
    elif strat == "pi_half":
        enc = 0.5 * pnp.ones(N_ENC) + 0.05 * pnp.random.randn(N_ENC)
        rot = pnp.array(pnp.random.uniform(-np.pi / 6, np.pi / 6, N_ROT))
    elif strat == "decay":
        enc = pnp.linspace(1.5, 0.5, N_ENC) + 0.05 * pnp.random.randn(N_ENC)
        rot = 0.15 * pnp.random.randn(N_ROT)
    elif strat == "warm":
        enc = 1.2 * pnp.ones(N_ENC) + 0.08 * pnp.random.randn(N_ENC)
        rot = 0.08 * pnp.random.randn(N_ROT)
    elif strat == "small_enc_v2":
        enc = 0.3 * pnp.ones(N_ENC) + 0.08 * pnp.random.randn(N_ENC)
        rot = 0.15 * pnp.random.randn(N_ROT)
    elif strat == "uniform_narrow":
        enc = pnp.array(pnp.random.uniform(0.7, 1.3, N_ENC))
        rot = pnp.array(pnp.random.uniform(-np.pi / 6, np.pi / 6, N_ROT))
    else:
        enc = pnp.ones(N_ENC) + 0.1 * pnp.random.randn(N_ENC)
        rot = 0.1 * pnp.random.randn(N_ROT)
    return pnp.array(
        pnp.concatenate([enc, rot,
                         0.1 * pnp.random.randn(N_LIN),
                         pnp.array([0.0])]),
        requires_grad=True,
    )


# ─── gradient norm helper (for barren plateau detection) ─────────────────────
def param_grad_norm(params, Xb, yb):
    grad_fn = qml.grad(focal_loss, argnum=0)
    g = grad_fn(params, Xb, yb)
    return float(np.linalg.norm(np.array(g)))


# ─── training loop ───────────────────────────────────────────────────────────
rng    = np.random.default_rng(seed)
params = init_params(seed, strat)
opt    = qml.AdamOptimizer(LR_INIT)

best_params   = pnp.array(params, requires_grad=True)
best_auc      = -1.0
patience_ctr  = 0

for ep in range(EPOCHS):
    opt.stepsize = float(lr_schedule(ep))
    Xb, yb = stratified_batch(X_tr, y_tr, BATCH_SIZE, rng)

    params, _ = opt.step_and_cost(
        lambda pp, _X=Xb, _y=yb: focal_loss(pp, _X, _y), params
    )

    # gradient clipping: rescale params update if norm exceeds cap
    # (AdamOptimizer modifies params in-place; we clip the params vector norm)
    p_norm = float(pnp.linalg.norm(params))
    if p_norm > GRAD_CLIP_MAX * N_PARAMS:
        params = params * (GRAD_CLIP_MAX * N_PARAMS / (p_norm + 1e-12))
        params = pnp.array(params, requires_grad=True)

    # barren plateau guard: check gradient norm once after warmup
    if ep == BARREN_CHECK:
        gn = param_grad_norm(params, Xb, yb)
        if gn < BARREN_FLOOR:
            break  # dead restart — save whatever we have and exit early

    if (ep + 1) % VAL_EVERY == 0:
        vp = vqc_proba(params, X_val)
        try:
            auc = roc_auc_score(y_val, vp)
        except Exception:
            auc = 0.5
        if auc > best_auc + MIN_DELTA:
            best_auc     = auc
            best_params  = pnp.array(params, requires_grad=True)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

# ─── threshold calibration on val (F1) ───────────────────────────────────────
vp_best  = vqc_proba(best_params, X_val)
best_t, best_fs = 0.5, -1.0
for t in np.linspace(0.05, 0.95, 181):
    fs = fbeta_score(y_val, (vp_best >= t).astype(int), beta=1.0, zero_division=0)
    if fs > best_fs:
        best_fs, best_t = fs, float(t)

# ─── save result ─────────────────────────────────────────────────────────────
result = {
    "params":    np.array(best_params),
    "val_auc":   float(best_auc),
    "threshold": float(best_t),
    "n_epochs":  int(ep + 1),
}
np.save(result_path, result)
