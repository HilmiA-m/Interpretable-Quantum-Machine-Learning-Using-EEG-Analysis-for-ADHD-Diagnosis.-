import os

DATA_ROOT = "/Users/mah/Desktop/BALLADEER_Quantum/BALLADEER ADHD DATASET"
DEMO_JSON = os.path.join(DATA_ROOT, "users_demographics.json")

N_QUBITS = 8
N_LAYERS = 5
N_PCA    = N_QUBITS

# VQC
N_RESTARTS       = 12
EPOCHS           = 400
LR_INIT          = 0.04
LR_MIN           = 0.001
WARMUP           = 20
BATCH_SIZE       = 32
FOCAL_ALPHA      = 0.70
FOCAL_GAMMA      = 2.0
L2_REG           = 0.001
PATIENCE         = 10
MIN_DELTA        = 0.004
VAL_EVERY        = 15
ENSEMBLE_K       = 4
MAX_RESTART_SECS = 600

# QCNN
N_QCNN_QUBITS  = 8
N_QCNN_PARAMS  = 70
QCNN_RESTARTS  = 16
QCNN_TOP_K     = 4
QCNN_EPOCHS    = 300
QCNN_LR        = 0.03
QCNN_LR_MIN    = 0.003
QCNN_VAL_EVERY = 20
QCNN_PATIENCE  = 8
QCNN_MIN_DELTA = 0.002

# QSVM
QSVM_C_GRID        = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
QSVM_KERNEL_CENTER = True
NYSTROEM_TRIGGER   = 120
NYSTROEM_M         = 40

TEST_SIZE = 0.20
FBETA     = 1.0
BOOT_N    = 1000

SLACKLINE_LEVELS = ["SlacklineLvl1", "SlacklineLvl6", "SlacklineLvl11"]
BANDS = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "gamma": (30, 45),
}

CGX_CH = [
    "AF7", "Fpz", "F7", "Fz", "T7", "FC6", "Fp1", "F4", "C4", "Oz",
    "CP6", "Cz", "PO8", "CP5", "O2", "O1", "P3", "P4", "P7", "P8",
    "Pz", "PO7", "T8", "C3", "Fp2", "F3", "F8", "FC5", "AF8",
]
N_CGX         = len(CGX_CH)
LEFT_FRONTAL  = [CGX_CH.index(c) for c in ["AF7", "F7", "Fp1", "F3"]]
RIGHT_FRONTAL = [CGX_CH.index(c) for c in ["AF8", "F8", "Fp2", "F4"]]
FRONTAL       = LEFT_FRONTAL + RIGHT_FRONTAL
OCCIPITAL     = [CGX_CH.index(c) for c in ["Oz", "O1", "O2", "PO7", "PO8"]]
CENTRAL       = [CGX_CH.index(c) for c in ["Cz", "C3", "C4"]]

_src_dir      = os.path.dirname(os.path.abspath(__file__))
_project_dir  = os.path.dirname(_src_dir)
WORKER_SCRIPT = os.path.join(_project_dir, "workers", "vqc_subprocess_runner.py")
RESULTS_DIR   = os.path.join(_project_dir, "RESULTS")
