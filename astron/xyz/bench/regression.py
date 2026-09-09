"""Reproducible, isolated comparison of two engine checkouts; never trains.

Example (run from the updated checkout)::

    python bench/regression.py --root ../original/astra --out evidence/before.json
    python bench/regression.py --manifest evidence/before.json --out evidence/after.json
    python bench/regression.py --compare evidence/before.json evidence/after.json

Each task runs in a fresh process with PYTHONHASHSEED=0. Only demonstrations
and test inputs are sent to it; answers remain in the scoring process. The
process timeout is the solver budget plus a recorded startup/cleanup grace.
Selection is deterministic and balanced across the filename prefixes (ARC1,
ARC2). Timed search can still vary with machine load, so small differences
require replication and are not statistical evidence of general improvement.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKER = r'''
import json, sys, time
sys.path.insert(0, sys.argv[1])
from engine import learn, portfolio
learn.activate(None)
request = json.load(sys.stdin)
t0 = time.monotonic()
result = portfolio.solve(request["train"], request["test_inputs"],
                         time_budget=request["budget"], k=request["k"],
                         loo=request["loo"])
print(json.dumps({"predictions": result.predictions,
                  "solver": result.solver, "chosen": result.chosen,
                  "n_hyps": result.n_hyps, "n_fit": result.n_fit,
                  "solver_seconds": time.monotonic() - t0}))
'''


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_tasks(files, limit=40, seed="astra-regression-v1"):
    """Round robin by collection, hashed within each collection; no labels."""
    groups = {}
    for name in sorted(files):
        groups.setdefault(name.split("_", 1)[0], []).append(name)
    for names in groups.values():
        names.sort(key=lambda n: (hashlib.sha256((seed + n).encode()).hexdigest(), n))
    selected = []
    while groups and (not limit or len(selected) < limit):
        for group in sorted(list(groups)):
            selected.append(groups[group].pop(0))
            if not groups[group]:
                del groups[group]
            if limit and len(selected) == limit:
                break
    return selected


def solver_request(task, budget, k, loo):
    """Construct the complete worker payload without serializing test answers."""
    return {"train": [[p["input"], p["output"]] for p in task["train"]],
            "test_inputs": [p["input"] for p in task["test"]],
            "budget": budget, "k": k, "loo": loo}


def score(predictions, answers):
    # Length checks prevent vacuous all()/zip() successes on empty/truncated output.
    if not answers or len(predictions) != len(answers):
        return 0, 0
    top1 = all(p and p[0] == answer for p, answer in zip(predictions, answers))
    top2 = all(any(g == answer for g in p[:2])
               for p, answer in zip(predictions, answers))
    return int(top1), int(top2)


def run_task(root, task, budget, k, loo, grace):
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    result = {}
    error = None
    try:
        proc = subprocess.run(
            [sys.executable, "-c", WORKER, str(root)],
            input=json.dumps(solver_request(task, budget, k, loo)),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(root), env=env, timeout=budget + grace, check=False)
        if proc.returncode:
            error = "worker exit %s: %s" % (proc.returncode, proc.stderr[-1200:])
        else:
            result = json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        error = "process timeout (budget + grace = %.3fs)" % (budget + grace)
    except (ValueError, OSError) as exc:
        error = repr(exc)
    predictions = result.pop("predictions", [])
    top1, top2 = score(predictions, [p["output"] for p in task["test"]])
    result.update(solved=top1, solved2=top2, error=error,
                  time=round(time.monotonic() - started, 6), n_test=len(task["test"]))
    return result


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted((Path(root) / "engine").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compare(before, after):
    for key in ("budget", "k", "loo", "grace", "hash_seed", "policy"):
        if before[key] != after[key]:
            raise ValueError("comparison setting differs: " + key)
    left = {r["id"]: r for r in before["per_task"]}
    right = {r["id"]: r for r in after["per_task"]}
    if set(left) != set(right):
        raise ValueError("task IDs differ")
    for tid in left:
        if left[tid]["sha256"] != right[tid]["sha256"]:
            raise ValueError("task content differs: " + tid)
    result = {"n": len(left), "before_engine": before["engine_sha256"],
              "after_engine": after["engine_sha256"]}
    for metric in ("solved", "solved2"):
        wins = sorted(t for t in left if right[t][metric] > left[t][metric])
        losses = sorted(t for t in left if right[t][metric] < left[t][metric])
        result[metric] = {"before": sum(r[metric] for r in left.values()),
                          "after": sum(r[metric] for r in right.values()),
                          "wins": wins, "losses": losses}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--tasks", type=Path, default=ROOT / "data" / "arc")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--manifest", type=Path, help="Use exactly these prior task IDs and content")
    ap.add_argument("--compare", nargs=2, type=Path)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--grace", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--seed", default="astra-regression-v1")
    args = ap.parse_args()
    if args.compare:
        print(json.dumps(compare(*map(read_json, args.compare)), indent=2))
        return
    if (not args.out or args.limit < 0 or args.k < 1 or
            not math.isfinite(args.budget) or args.budget <= 0 or
            not math.isfinite(args.grace) or args.grace < 0):
        ap.error("--out required; budget > 0, grace >= 0, limit >= 0, k >= 1")
    root = args.root.resolve()
    files = [p.name for p in args.tasks.glob("*.json")]
    manifest = read_json(args.manifest) if args.manifest else None
    expected = {r["id"]: r["sha256"] for r in manifest["per_task"]} if manifest else {}
    names = [r["id"] + ".json" for r in manifest["per_task"]] if manifest else select_tasks(files, args.limit, args.seed)
    if not names:
        ap.error("no tasks selected")
    engine_hash = source_hash(root)
    started = time.monotonic()
    rows = []
    for index, name in enumerate(names):
        if name not in files:
            ap.error("manifest task missing: " + name)
        raw = (args.tasks / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        tid = Path(name).stem
        if expected and digest != expected[tid]:
            ap.error("manifest task content changed: " + tid)
        task = json.loads(raw)
        if not task.get("test") or any("output" not in p for p in task["test"]):
            ap.error("scoring requires nonempty labeled tests: " + tid)
        row = run_task(root, task, args.budget, args.k, not args.no_loo, args.grace)
        row.update(id=tid, sha256=digest)
        rows.append(row)
        print("%d/%d %s top1=%d top2=%d %.2fs %s" %
              (index + 1, len(names), tid, row["solved"], row["solved2"],
               row["time"], row["error"] or ""), flush=True)
    if source_hash(root) != engine_hash:
        ap.error("engine sources changed during the run; rerun for consistent evidence")
    report = {"n": len(rows), "solved": sum(r["solved"] for r in rows),
              "solved_top2": sum(r["solved2"] for r in rows),
              "errors": sum(bool(r["error"]) for r in rows),
              "budget": args.budget, "k": args.k, "loo": not args.no_loo,
              "grace": args.grace, "hash_seed": 0, "policy": None,
              "seed": args.seed, "python": platform.python_version(),
              "platform": platform.platform(), "engine_sha256": engine_hash,
              "wall": round(time.monotonic() - started, 6), "per_task": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
