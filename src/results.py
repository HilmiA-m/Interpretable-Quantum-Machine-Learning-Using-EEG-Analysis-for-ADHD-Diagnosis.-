import json
import os

from src import config


def _ensure_dir():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)


def save_run(run_id, timestamp_str, results, dataset_info, config_snapshot):
    _ensure_dir()

    best = {
        metric: max(results, key=lambda m: results[m][metric])
        for metric in ("acc", "f1", "roc_auc", "pr_auc")
    }

    payload = {
        "run_id":    run_id,
        "timestamp": timestamp_str,
        "dataset":   dataset_info,
        "config":    config_snapshot,
        "models": {
            name: {k: round(float(v), 6) for k, v in m.items()}
            for name, m in results.items()
        },
        "best": {
            metric: {
                "model": best[metric],
                "value": round(float(results[best[metric]][metric]), 6),
            }
            for metric in ("acc", "f1", "roc_auc", "pr_auc")
        },
        "figures": [
            f"roc_curves_{run_id}.png",
            f"pr_curves_{run_id}.png",
            f"confusion_matrices_{run_id}.png",
            f"feature_importance_{run_id}.png",
        ],
    }

    path = os.path.join(config.RESULTS_DIR, f"{run_id}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [OK] {path}")
    return payload


def update_best(payload):
    _ensure_dir()
    best_path = os.path.join(config.RESULTS_DIR, "BEST_RESULTS.json")

    if os.path.exists(best_path):
        with open(best_path) as f:
            best = json.load(f)
    else:
        best = {"last_updated": None, "best_overall": {}, "runs": []}

    for metric in ("acc", "f1", "roc_auc", "pr_auc"):
        new_val  = payload["best"][metric]["value"]
        current  = best["best_overall"].get(metric, {})
        if new_val > current.get("value", -1.0):
            best["best_overall"][metric] = {
                "model":     payload["best"][metric]["model"],
                "value":     new_val,
                "run_id":    payload["run_id"],
                "timestamp": payload["timestamp"],
            }

    best["last_updated"] = payload["timestamp"]
    best["runs"].append({
        "run_id":       payload["run_id"],
        "timestamp":    payload["timestamp"],
        "n_subjects":   payload["dataset"]["n_subjects"],
        "best_roc_auc": payload["best"]["roc_auc"]["value"],
        "best_f1":      payload["best"]["f1"]["value"],
        "best_acc":     payload["best"]["acc"]["value"],
        "best_model":   payload["best"]["roc_auc"]["model"],
        "all_roc_auc":  {name: m["roc_auc"] for name, m in payload["models"].items()},
    })

    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"  [OK] {best_path}")
