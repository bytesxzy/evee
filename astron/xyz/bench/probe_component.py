"""Run one candidate module through the existing answer-isolated scoring harness.

Exploration is restricted to the development split. These module-only runs are
diagnostics, not full-engine scores. No test output is handled in this module.
"""
import argparse
import importlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench import split
from bench.run_arc import run_one
from engine import portfolio


def one(job):
    module, path, budget = job
    portfolio._REGISTRY[:] = [importlib.import_module("engine.solvers." + module)]
    portfolio._ARCH_MATH_TRIED = True
    portfolio._ARCH_MATH = None
    return run_one((path, budget, 2, False, None, ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--budget", type=float, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    folder = os.path.join(ROOT, "data", "split_dev")
    files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
    assert all(split.bucket(f[:-5]) == "dev" for f in files)
    jobs = [(a.module, os.path.join(folder, f), a.budget) for f in files]
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futures = [ex.submit(one, j) for j in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda r: r["id"])
    result = dict(n=len(rows), budget=a.budget, module_only=a.module,
                  wall=round(time.time()-t0, 3),
                  cpu_sum=round(sum(r["time"] for r in rows), 3), per_task=rows)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}))
    print("solved", sum(r["solved"] for r in rows),
          "fitted", sum(bool(r["hyps"]) for r in rows),
          "errors", sum(bool(r["error"]) for r in rows))


if __name__ == "__main__":
    main()
