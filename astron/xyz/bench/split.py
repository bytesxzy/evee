"""Deterministic dev / held-out split of the bundled ARC corpora.

Architecture iteration and any policy fitting use DEV only.  HOLD is not read
by any training or tuning code path; it exists so the final number is an
honest estimate on tasks the architecture was never shaped against.

The split is a pure function of the task id, so it is reproducible and cannot
drift as files are added or reordered.
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "arc")
SALT = "astra-arc-split-v1"
DEV_FRACTION = 0.40


def bucket(task_id):
    h = hashlib.sha256((SALT + "|" + task_id).encode()).hexdigest()
    return "dev" if (int(h[:8], 16) / 0xFFFFFFFF) < DEV_FRACTION else "hold"


def ids(data=DATA):
    return sorted(f[:-5] for f in os.listdir(data) if f.endswith(".json"))


def partition(data=DATA):
    dev, hold = [], []
    for t in ids(data):
        (dev if bucket(t) == "dev" else hold).append(t)
    return dev, hold


if __name__ == "__main__":
    import json
    import sys
    dev, hold = partition()
    if len(sys.argv) > 1 and sys.argv[1] == "--write":
        for name, group in (("dev", dev), ("hold", hold)):
            d = os.path.join(ROOT, "data", "split_" + name)
            os.makedirs(d, exist_ok=True)
            for t in group:
                src = os.path.join(DATA, t + ".json")
                dst = os.path.join(d, t + ".json")
                if not os.path.exists(dst):
                    with open(src, "rb") as a, open(dst, "wb") as b:
                        b.write(a.read())
        print("wrote data/split_dev (%d) and data/split_hold (%d)" % (len(dev), len(hold)))
    print(json.dumps({
        "dev": len(dev), "hold": len(hold),
        "dev_arc1": sum(1 for t in dev if t.startswith("arc1")),
        "dev_arc2": sum(1 for t in dev if t.startswith("arc2")),
        "hold_arc1": sum(1 for t in hold if t.startswith("arc1")),
        "hold_arc2": sum(1 for t in hold if t.startswith("arc2")),
        "sha256_of_dev_ids": hashlib.sha256("\n".join(dev).encode()).hexdigest()[:16],
        "sha256_of_hold_ids": hashlib.sha256("\n".join(hold).encode()).hexdigest()[:16],
    }, indent=1))
