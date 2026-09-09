"""Measure GABRIEL against the stock engine, on identical tasks, with the
harness's own scoring.

Scoring is not reimplemented here: ``run_one`` and ``score_predictions`` are
imported from ``bench/run_arc.py``, so a task counts as solved under exactly
the same rule -- every test pair exact, first guess only -- and a comparison
against the numbers in ``RESULTS.md`` stays honest.

    python3 -m gabriel.bench --mode both --split fit --limit 40 --budget 10

``--mode both`` runs the same tasks twice, once stock and once bound, and
reports the paired sign test ``bench/evolve.py`` uses to decide adoption.  A
change that merely reshuffles which tasks are solved does not count as better
here either.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench import run_arc                        # noqa: E402
from bench.evolve import sign_test, split        # noqa: E402

from gabriel import bind                         # noqa: E402


def _init(policy_path, lm_path, proposals):
    """Bind GABRIEL in this worker, then tell run_arc a policy is already up.

    ``run_arc._ensure_policy`` would otherwise activate a plain policy on the
    first task and silently unbind the model.  Marking its cache is how the
    binding survives without editing the harness.
    """
    bind.bind(policy_path, lm_path, proposals)
    run_arc._POLICY_LOADED.clear()
    run_arc._POLICY_LOADED[policy_path] = True


def select(tasks_dir, which="fit", limit=0, filt="", hold_frac=0.4):
    files = sorted(f for f in os.listdir(tasks_dir) if f.endswith(".json"))
    if filt:
        files = [f for f in files if filt in f]
    fit, hold = split(files, hold_frac)
    files = {"fit": fit, "hold": hold, "all": files}[which]
    return files[:limit] if limit else files


def run(files, tasks_dir, budget, jobs, k=2, loo=True, policy="", lm="",
        proposals=True, stock=False):
    args = [(os.path.join(tasks_dir, f), budget, k, loo, None, policy)
            for f in files]
    out = []
    if jobs <= 1:
        if not stock:
            _init(policy, lm, proposals)
        for a in args:
            out.append(run_arc.run_one(a))
    else:
        init = (None, ()) if stock else (_init, (policy, lm, proposals))
        with ProcessPoolExecutor(max_workers=jobs, initializer=init[0],
                                 initargs=init[1]) as ex:
            for r in ex.map(run_arc.run_one, args):
                out.append(r)
    return {r["id"]: r for r in out}


def summarise(res):
    n = len(res)
    s1 = sum(r["solved"] for r in res.values())
    s2 = sum(r["solved2"] for r in res.values())
    return {"n": n, "solved": s1, "solved_top2": s2,
            "rate": round(s1 / float(n), 4) if n else 0.0,
            "cpu_sum": round(sum(r["time"] for r in res.values()), 1),
            "by_solver": _by_solver(res)}


def _by_solver(res):
    out = {}
    for r in res.values():
        if r["solved"] and r.get("solver"):
            out[r["solver"]] = out.get(r["solver"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default=run_arc.DATA)
    ap.add_argument("--split", choices=("fit", "hold", "all"), default="fit")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter", default="")
    ap.add_argument("--budget", type=float, default=14.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--policy", default=bind.POLICY_PATH)
    ap.add_argument("--lm", default=bind.LM_PATH)
    ap.add_argument("--no-proposals", action="store_true")
    ap.add_argument("--mode", choices=("gabriel", "stock", "both"),
                    default="gabriel")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    policy = a.policy if os.path.exists(a.policy) else ""
    files = select(a.tasks, a.split, a.limit, a.filter)
    t0 = time.time()
    summary = {"n": len(files), "split": a.split, "budget": a.budget,
               "policy": policy or None, "lm": a.lm,
               "proposals": not a.no_proposals}
    res = {}

    if a.mode in ("stock", "both"):
        stock = run(files, a.tasks, a.budget, a.jobs, a.k, not a.no_loo,
                    policy="", stock=True)
        summary["stock"] = summarise(stock)
        print("stock   " + json.dumps(summary["stock"]), flush=True)
        res = stock
    if a.mode in ("gabriel", "both"):
        gab = run(files, a.tasks, a.budget, a.jobs, a.k, not a.no_loo,
                  policy=policy, lm=a.lm, proposals=not a.no_proposals)
        summary["gabriel"] = summarise(gab)
        print("gabriel " + json.dumps(summary["gabriel"]), flush=True)
        res = gab
    if a.mode == "both":
        wins = sorted(t for t in gab if gab[t]["solved"] and not stock.get(t, {}).get("solved"))
        losses = sorted(t for t in gab if stock.get(t, {}).get("solved") and not gab[t]["solved"])
        summary["paired"] = {"wins": len(wins), "losses": len(losses),
                             "p": round(sign_test(len(wins), len(losses)), 4),
                             "win_ids": wins[:20], "loss_ids": losses[:20]}
        print("paired  " + json.dumps(summary["paired"]), flush=True)
    summary["wall"] = round(time.time() - t0, 1)

    if a.out:
        path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = dict(summary)
        payload["per_task"] = sorted(res.values(), key=lambda r: r["id"])
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=1)
        print("wrote " + path, flush=True)
    else:
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
