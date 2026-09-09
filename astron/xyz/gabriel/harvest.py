"""Grow the corpus: solve a rotating slice of tasks and keep the programs.

``bench/evolve.py`` holds its run records in memory and writes only a lineage
summary, so a cron of evolve rounds alone never adds a single program to the
language model's training data.  This adds them, by doing the one thing that
produces programs: running the engine.

Each invocation takes the next ``--limit`` tasks from the fit split, runs them
with GABRIEL bound, and writes one evidence file in the format
``bench/run_arc.py`` already uses -- which is the format :mod:`gabriel.corpus`
already reads.  The offset advances and wraps, so consecutive cron ticks cover
different tasks and the whole fit split is swept over an hour or so.

Old harvest files are pruned to a fixed count.  Unbounded log and evidence
growth on a small VPS is how a training loop turns into a full disk at 3am.

Only fit-split tasks are ever harvested.  The evolve holdout stays untouched by
anything that feeds back into the engine.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench import run_arc                        # noqa: E402

from gabriel import bench as gbench              # noqa: E402
from gabriel import bind                         # noqa: E402

STATE = os.path.join(ROOT, "policy", "gabriel_harvest.state")
PREFIX = "gabriel-harvest-"


def _state():
    try:
        with open(STATE) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"offset": 0, "sweeps": 0}


def _save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(st, fh)
    os.replace(tmp, STATE)


def prune(evidence_dir, keep):
    files = sorted(f for f in os.listdir(evidence_dir) if f.startswith(PREFIX))
    dropped = []
    for f in files[:max(0, len(files) - keep)]:
        try:
            os.remove(os.path.join(evidence_dir, f))
            dropped.append(f)
        except OSError:
            pass
    return dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default=run_arc.DATA)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--keep", type=int, default=40,
                    help="how many harvest files to retain")
    ap.add_argument("--policy", default=bind.POLICY_PATH)
    ap.add_argument("--lm", default=bind.LM_PATH)
    ap.add_argument("--stock", action="store_true",
                    help="harvest with the unmodified engine")
    a = ap.parse_args(argv)

    files = gbench.select(a.tasks, "fit")
    if not files:
        print(json.dumps({"error": "no fit tasks", "tasks": a.tasks}))
        return 1
    st = _state()
    off = int(st.get("offset", 0)) % len(files)
    slice_ = [files[(off + i) % len(files)] for i in range(min(a.limit, len(files)))]

    t0 = time.time()
    policy = a.policy if os.path.exists(a.policy) else ""
    res = gbench.run(slice_, a.tasks, a.budget, a.jobs, policy=policy,
                     lm=a.lm, stock=a.stock)
    evidence = os.path.join(ROOT, "evidence")
    os.makedirs(evidence, exist_ok=True)
    out = os.path.join(evidence, "%s%d.json" % (PREFIX, int(time.time())))
    payload = {"n": len(res), "budget": a.budget, "offset": off,
               "policy": policy or None, "stock": bool(a.stock),
               "wall": round(time.time() - t0, 1),
               "per_task": sorted(res.values(), key=lambda r: r["id"])}
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)

    st["offset"] = (off + len(slice_)) % len(files)
    st["sweeps"] = int(st.get("sweeps", 0)) + (1 if st["offset"] <= off else 0)
    st["last"] = int(time.time())
    _save_state(st)
    dropped = prune(evidence, a.keep)
    print(json.dumps({"harvested": len(res),
                      "solved": sum(r["solved"] for r in res.values()),
                      "offset": off, "next_offset": st["offset"],
                      "sweeps": st["sweeps"], "pruned": len(dropped),
                      "file": os.path.basename(out),
                      "wall": payload["wall"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
