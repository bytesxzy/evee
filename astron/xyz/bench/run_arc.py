"""Measurement harness.

Guarantees, by construction:

* a solver is handed train pairs and test *inputs* only -- the answers stay in
  this process and are compared after the fact;
* a task counts as solved only when **every** test pair of that task is exact;
* the orchestrator cooperatively checks a per-task deadline. A pathological
  solver can still overrun it; the process pool is not a hard timeout. Use
  bench/regression.py for isolated tasks with an external process timeout.

Usage::

    python3 bench/run_arc.py --out evidence/run.json --budget 30 --jobs 4
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import grid as G  # noqa: E402
from engine import learn, planner, portfolio  # noqa: E402
from engine.task import Ctx  # noqa: E402

_POLICY_LOADED = {}


def _ensure_policy(path, planner_path="", drop=()):
    """Load and activate a policy and planner once per worker process."""
    key = (path, planner_path, tuple(drop))
    if key in _POLICY_LOADED:
        return
    _POLICY_LOADED.clear()
    _POLICY_LOADED[key] = True
    learn.activate(learn.Policy.load(path) if path else None)
    planner.activate(planner.Planner.load(planner_path) if planner_path else None)
    from engine import hypcache
    hypcache.ENABLED = "hypcache" not in drop
    if "objops" in drop:
        # not a registered generator: it is the object vocabulary the general
        # enumerator composes over, so it is switched off rather than removed
        from engine import objops
        objops.ENABLED = False
    if drop:
        # ablation support: remove named generators from the registry so the
        # measured effect of a component is the effect of its absence
        portfolio._load_default()
        keep = [m for m in portfolio._REGISTRY
                if portfolio._module_key(m) not in drop]
        del portfolio._REGISTRY[:]
        portfolio._REGISTRY.extend(keep)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "arc")


def load(path):
    with open(path) as fh:
        return json.load(fh)


def score_predictions(predictions, answers):
    """Require every test pair, and count only the first two ranked guesses."""
    if not answers or len(predictions) != len(answers):
        return False, False
    top1 = all(p and p[0] == a for p, a in zip(predictions, answers))
    top2 = all(any(g == a for g in p[:2]) for p, a in zip(predictions, answers))
    return top1, top2


def run_one(args):
    # the planner and ablation fields are optional so that callers written
    # against the original six-field job tuple keep working
    (path, budget, k, loo, priors, policy_path), rest = args[:6], args[6:]
    planner_path = rest[0] if len(rest) > 0 else ""
    drop = rest[1] if len(rest) > 1 else ()
    tid = os.path.basename(path)[:-5]
    _ensure_policy(policy_path, planner_path, drop)
    d = load(path)
    train = [(p["input"], p["output"]) for p in d["train"]]
    tests = d["test"]
    test_inputs = [p["input"] for p in tests]
    answers = [G.from_list(p["output"]) for p in tests]
    if priors:
        portfolio.SOLVER_PRIOR.update(priors)
    t0 = time.time()
    try:
        res = portfolio.solve(train, test_inputs, time_budget=budget, k=k, loo=loo)
        preds = res.predictions
        hyps = res.hyps
        solver = res.solver
        chosen = res.chosen
        err = None
    except Exception as exc:                       # bounded solver failure
        preds = [[] for _ in test_inputs]
        hyps, solver, chosen, err = [], None, [], repr(exc)[:200]
    el = time.time() - t0
    try:
        sigs = list(learn.signatures(Ctx(train, test_inputs)))
    except Exception:
        sigs = []
    # every synthesised program that fit the demonstrations, for abstraction
    # mining -- not just the one that happened to win the vote
    programs = [n.split(":", 1)[1] for n, _s in hyps
                if n.startswith(("compose:", "enumerate:"))]
    program = None
    if chosen and chosen[0]:
        solver = chosen[0][0]
        program = chosen[0][1]
    elif hyps:
        program = hyps[0][0].split(":", 1)[-1]

    top1, top2 = score_predictions(preds, answers)
    # partial credit: mean cell agreement of the top-1 guess (diagnostic only)
    part = 0.0
    for p, a in zip(preds, answers):
        if p and G.dims(p[0]) == G.dims(a):
            n = G.area(a)
            m = sum(1 for ra, rb in zip(p[0], a) for x, y in zip(ra, rb) if x == y)
            part += m / float(n)
    part /= max(1, len(answers))
    return {
        "id": tid, "solved": int(top1), "solved2": int(top2),
        "partial": round(part, 4), "time": round(el, 3),
        "solver": solver, "hyps": hyps[:3], "error": err,
        "n_test": len(tests), "sigs": sigs, "program": program,
        "programs": programs,
        "cache": res.diagnostics.get("hypothesis_cache", {}) if err is None else {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter", default="")
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--tasks", default=DATA)
    ap.add_argument("--priors", default="")
    ap.add_argument("--policy", default="")
    ap.add_argument("--planner", default="")
    ap.add_argument("--drop", default="",
                    help="comma-separated generator modules to disable")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.tasks) if f.endswith(".json"))
    if a.filter:
        files = [f for f in files if a.filter in f]
    if a.limit:
        files = files[:a.limit]
    priors = json.loads(a.priors) if a.priors else None
    drop = tuple(x for x in a.drop.split(",") if x)
    jobs = [(os.path.join(a.tasks, f), a.budget, a.k, not a.no_loo, priors,
             a.policy, a.planner, drop) for f in files]

    t0 = time.time()
    out = []
    if a.jobs <= 1:
        for j in jobs:
            out.append(run_one(j))
            if not a.quiet:
                r = out[-1]
                print("%-18s %d %.1fs %s" % (r["id"], r["solved"], r["time"],
                                             r["solver"] or ""), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = {ex.submit(run_one, j): j for j in jobs}
            done = 0
            for fu in as_completed(futs):
                try:
                    out.append(fu.result())
                except Exception as exc:
                    out.append({"id": os.path.basename(futs[fu][0])[:-5],
                                "solved": 0, "solved2": 0, "partial": 0.0,
                                "time": 0.0, "solver": None, "hyps": [],
                                "error": repr(exc)[:200], "n_test": 1})
                done += 1
                if not a.quiet and done % 25 == 0:
                    s = sum(r["solved"] for r in out)
                    print("  %d/%d  solved=%d  (%.0fs)" %
                          (done, len(jobs), s, time.time() - t0), flush=True)
    out.sort(key=lambda r: r["id"])
    n = len(out)
    s1 = sum(r["solved"] for r in out)
    s2 = sum(r["solved2"] for r in out)
    a1 = sum(r["solved"] for r in out if r["id"].startswith("arc1"))
    a2 = sum(r["solved"] for r in out if r["id"].startswith("arc2"))
    n1 = sum(1 for r in out if r["id"].startswith("arc1"))
    n2 = sum(1 for r in out if r["id"].startswith("arc2"))
    summary = {
        "n": n, "solved": s1, "solved_top2": s2,
        "rate": round(s1 / float(n), 4) if n else 0,
        "rate_top2": round(s2 / float(n), 4) if n else 0,
        "arc1": {"n": n1, "solved": a1,
                 "rate": round(a1 / float(n1), 4) if n1 else 0},
        "arc2": {"n": n2, "solved": a2,
                 "rate": round(a2 / float(n2), 4) if n2 else 0},
        "partial_mean": round(sum(r["partial"] for r in out) / float(n), 4) if n else 0,
        "errors": sum(1 for r in out if r["error"]),
        "wall": round(time.time() - t0, 1),
        "cpu_sum": round(sum(r["time"] for r in out), 1),
        "budget": a.budget, "k": a.k, "loo": not a.no_loo,
        "policy": a.policy or None,
        "planner": a.planner or None,
        "dropped": list(drop),
        "tasks_dir": os.path.basename(os.path.normpath(a.tasks)),
        "per_task": out,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"},
                     indent=2))
    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(summary, fh, indent=1)
        print("wrote", p)


if __name__ == "__main__":
    main()
