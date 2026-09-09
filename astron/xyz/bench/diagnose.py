"""Failure-mode census.

For every task the engine gets wrong we want to know *which* stage lost it, so
that architecture effort goes where the losses are rather than where intuition
says they are.  The census separates two fundamentally different failures:

* the correct grid was somewhere in the candidate set and the ranking put
  something else first (a scoring problem), versus
* the correct grid was never produced by any hypothesis (a generation problem).

The second is split again by whether *any* hypothesis explained the training
pairs at all.  A task with fitted hypotheses but no correct one means the
program space contains a wrong-but-consistent rule and lacks the right one;
a task with zero fitted hypotheses means the space does not contain any
explanation of the demonstrations.

Structural tags (symmetry, counting, motion, panels, ...) are computed from the
train pairs only and are descriptive: they say what kind of task is being lost,
not what the fix is.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import grid as G          # noqa: E402
from engine import portfolio          # noqa: E402
from engine.task import Ctx           # noqa: E402
from engine import objects as O       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def tags(train, test_inputs):
    """Descriptive structure tags, from demonstrations only."""
    t = []
    ctx = Ctx(train, test_inputs)
    same = ctx.same_shape
    t.append("shape:same" if same else "shape:diff")
    if not same:
        if ctx.shape_ratio:
            t.append("kind:upscale")
        elif ctx.inv_shape_ratio:
            t.append("kind:downscale")
        elif all(G.area(b) < G.area(a) for a, b in train):
            t.append("kind:extract")
        else:
            t.append("kind:reshape")
    outs = [b for _, b in train]
    if len({G.dims(b) for b in outs}) == 1 and all(G.area(b) <= 9 for b in outs):
        t.append("kind:tiny_out")
    if same and all(G.symmetries(b) for b in outs):
        t.append("struct:sym_out")
    try:
        for seg in ("c4", "c8"):
            hit = 0
            for a, b in train:
                n = len(O.segment(a, seg, G.background(a)))
                if G.area(b) == n:
                    hit += 1
            if hit == len(train):
                t.append("struct:counting")
                break
    except Exception:
        pass
    if same and all(G.histogram(a) == G.histogram(b) for a, b in train):
        t.append("struct:permutation")
    if all(G.palette(b) - G.palette(a) for a, b in train):
        t.append("struct:new_colors")
    try:
        ns = [len(O.segment(a, "c8", G.background(a))) for a, _ in train]
        m = sum(ns) / float(len(ns))
        t.append("objs:%s" % ("few" if m <= 4 else ("some" if m <= 12 else "many")))
    except Exception:
        pass
    try:
        from engine.solvers.partition import _sep_color_candidates
        if _sep_color_candidates(ctx):
            t.append("struct:panels")
    except Exception:
        pass
    t.append("ntrain:%d" % len(train))
    return t


def one(args):
    path, budget = args
    tid = os.path.basename(path)[:-5]
    with open(path) as fh:
        d = json.load(fh)
    train = [(p["input"], p["output"]) for p in d["train"]]
    tests = d["test"]
    test_inputs = [p["input"] for p in tests]
    answers = [G.from_list(p["output"]) for p in tests]
    t0 = time.time()
    try:
        res = portfolio.solve(train, test_inputs, time_budget=budget, k=2,
                              loo=True, collect_all=True)
        err = None
    except Exception as exc:
        res, err = None, repr(exc)[:200]
    el = time.time() - t0
    if res is None:
        return {"id": tid, "class": "ERROR", "error": err, "time": el,
                "tags": [], "n_fit": 0, "n_hyps": 0, "rank": None}
    preds = res.predictions
    top1 = all(p and p[0] == a for p, a in zip(preds, answers))
    top2 = all(any(g == a for g in p[:2]) for p, a in zip(preds, answers))
    ranks = []
    for p, a in zip(preds, answers):
        ranks.append(p.index(a) if a in p else None)
    present = all(r is not None for r in ranks)
    worst = max(ranks) if present else None
    if top1:
        klass = "SOLVED"
    elif top2:
        klass = "SOLVED_TOP2"
    elif present:
        klass = "MISRANKED"
    elif res.n_fit > 0:
        klass = "FIT_BUT_WRONG"
    elif res.n_hyps > 0:
        klass = "NO_FIT"
    else:
        klass = "NO_HYPOTHESIS"
    return {"id": tid, "class": klass, "time": round(el, 2),
            "tags": tags(train, test_inputs), "n_fit": res.n_fit,
            "n_hyps": res.n_hyps, "rank": worst,
            "n_distinct": sum(len(p) for p in preds) / max(1, len(preds)),
            "solver": res.solver, "error": None,
            "top_program": res.hyps[0][0] if res.hyps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(ROOT, "data", "split_dev"))
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.tasks) if f.endswith(".json"))
    if a.limit:
        files = files[:a.limit]
    jobs = [(os.path.join(a.tasks, f), a.budget) for f in files]
    out, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs)):
            try:
                out.append(fu.result())
            except Exception as exc:
                out.append({"id": "?", "class": "ERROR", "error": repr(exc)[:120],
                            "tags": [], "n_fit": 0, "n_hyps": 0, "rank": None})
            if (i + 1) % 25 == 0:
                print("  %d/%d (%.0fs)" % (i + 1, len(jobs), time.time() - t0),
                      flush=True)
    out.sort(key=lambda r: r["id"])
    census = Counter(r["class"] for r in out)
    all_tags, bad_tags = Counter(), Counter()
    for r in out:
        for t in r["tags"]:
            all_tags[t] += 1
            if r["class"] != "SOLVED":
                bad_tags[t] += 1
    lift = sorted(((bad_tags[t] / float(all_tags[t]), all_tags[t], t)
                   for t in all_tags if all_tags[t] >= 8), reverse=True)
    summary = {
        "n": len(out), "census": dict(census),
        "solved": census["SOLVED"],
        "rate": round(census["SOLVED"] / float(len(out)), 4) if out else 0,
        "tag_failure_rate": [{"tag": t, "n": n, "fail_rate": round(f, 3)}
                             for f, n, t in lift],
        "per_task": out,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"},
                     indent=1))
    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(summary, fh, indent=1)
        print("wrote", p)


if __name__ == "__main__":
    main()
