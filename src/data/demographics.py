import json
from src import config


def load_demographics():
    with open(config.DEMO_JSON) as f:
        demo = json.load(f)
    user_meta = {}
    for e in demo:
        diag = e.get("diagnosed", "")
        if diag not in ("yes", "no"):
            continue
        user_meta[e["user"]] = {
            "label":  1 if diag == "yes" else 0,
            "age":    float(e.get("age", 12)),
            "gender": float(e.get("gender", 1)),
        }
    n_adhd = sum(v["label"] for v in user_meta.values())
    print(f"\n[1] Users: {len(user_meta)}  ADHD={n_adhd}  Control={len(user_meta) - n_adhd}")
    return user_meta
