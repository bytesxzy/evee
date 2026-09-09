"""Fit the reasoning planner from successful symbolic searches.

Supervision comes from the engine's own successes, never from the answer key.
For each development task we hide its *last training pair* and ask every
hypothesis family, independently, to explain the remaining demonstrations.  A
family that then predicts the hidden pair correctly is a family whose search
generalised on this task, and that is the action the planner should have chosen
in that state.  Test outputs are not read at any point in this file; the
evaluation corpus is not read at all.

From each such (task, winning family) pair we manufacture the state/action
trajectory the planner will meet at run time:

    S0 (nothing tried)                    -> winner
    S1 (some other families already ran)  -> winner
    ...

so the model learns not only "which family suits this kind of task" but "which
family suits it *given that these have already come back empty*".

The fitted model is scored on a task-disjoint slice of the development split
before being refitted on all of it, so the accuracy reported here is an
out-of-sample number rather than a training-set number.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import grid as G          # noqa: E402
from engine import planner as PL      # noqa: E402
from engine import portfolio          # noqa: E402
from engine.learn import signatures   # noqa: E402
from engine.task import Ctx           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# deep searchers are capped harder: they would otherwise eat the whole probe
_SLOW = {"compose", "cascade", "enumerate_dsl", "objwise", "rewrite", "analogy"}


def probe_task(args):
    """Which families generalise to a held-out demonstration of this task."""
    path, budget = args
    tid = os.path.basename(path)[:-5]
    with open(path) as fh:
        d = json.load(fh)
    train = [(p["input"], p["output"]) for p in d["train"]]
    if len(train) < 3:
        return None
    portfolio._load_default()
    full_sigs = list(signatures(Ctx(train, [p["input"] for p in d["test"]])))
    sub, (probe_in, probe_out) = train[:-1], train[-1]
    probe_out = G.as_grid(probe_out)
    winners = []
    t0 = time.time()
    for mod in portfolio._REGISTRY:
        key = portfolio._module_key(mod)
        share = budget * (2.0 if key in _SLOW else 1.0)
        ctx = Ctx(sub, [probe_in], deadline=time.time() + share)
        try:
            hyps = mod.generate(ctx)
        except Exception:
            continue
        for hp in hyps:
            try:
                if hp.fits(ctx.train) and hp.apply(G.as_grid(probe_in)) == probe_out:
                    winners.append(key)
                    break
            except Exception:
                continue
    return {"id": tid, "sigs": full_sigs, "winners": winners,
            "time": round(time.time() - t0, 2)}


def build_examples(records, planner, rng, n_history=3):
    ex = []
    fams = list(PL.FAMILIES)
    for r in records:
        if not r["winners"]:
            continue
        sigs = tuple(r["sigs"])
        losers = [f for f in fams if f not in r["winners"]]
        for w in r["winners"]:
            feats = planner.features(sigs, (), 0, 1.0, 0, exclude=r["id"])
            ex.append((feats, w, frozenset(fams)))
            for _ in range(n_history):
                k = rng.randint(1, min(9, max(1, len(losers))))
                ran = tuple(rng.sample(losers, k))
                left = max(0.05, 1.0 - 0.09 * k)
                feats = planner.features(sigs, ran, 0, left, k, exclude=r["id"])
                avail = frozenset(f for f in fams if f not in ran)
                ex.append((feats, w, avail))
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(ROOT, "data", "split_dev"))
    ap.add_argument("--budget", type=float, default=0.55,
                    help="per-family probe budget in seconds")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--records", default=os.path.join(ROOT, "evidence",
                                                      "planner_records.json"))
    ap.add_argument("--out", default=PL.PLANNER_PATH)
    ap.add_argument("--reuse-records", action="store_true")
    a = ap.parse_args()

    if a.reuse_records and os.path.exists(a.records):
        with open(a.records) as fh:
            records = json.load(fh)
        print("reusing %d probe records" % len(records))
    else:
        files = sorted(f for f in os.listdir(a.tasks) if f.endswith(".json"))
        if a.limit:
            files = files[:a.limit]
        jobs = [(os.path.join(a.tasks, f), a.budget) for f in files]
        records, t0 = [], time.time()
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(probe_task, j) for j in jobs]
            for i, fu in enumerate(as_completed(futs)):
                try:
                    r = fu.result()
                except Exception:
                    r = None
                if r:
                    records.append(r)
                if (i + 1) % 25 == 0:
                    print("  probed %d/%d (%.0fs)" % (i + 1, len(jobs),
                                                      time.time() - t0),
                          flush=True)
        records.sort(key=lambda r: r["id"])
        os.makedirs(os.path.dirname(a.records), exist_ok=True)
        with open(a.records, "w") as fh:
            json.dump(records, fh, indent=1)
        print("wrote", a.records)

    with_w = [r for r in records if r["winners"]]
    print("tasks probed: %d, with a generalising family: %d" %
          (len(records), len(with_w)))
    print("family frequency:",
          Counter(w for r in with_w for w in r["winners"]).most_common())

    # task-disjoint validation slice, so the accuracy below is out of sample
    def side(tid):
        h = int(hashlib.sha256(("plnr|" + tid).encode()).hexdigest()[:8], 16)
        return "valid" if (h % 100) < 25 else "train"

    tr = [r for r in with_w if side(r["id"]) == "train"]
    va = [r for r in with_w if side(r["id"]) == "valid"]

    rng = random.Random(7)
    scratch = PL.Planner()
    for r in tr:
        for w in set(r["winners"]):
            scratch.memory.add(r["sigs"], w, r["id"])
    ex_tr = build_examples(tr, scratch, rng)
    ex_va = build_examples(va, scratch, random.Random(8))
    scratch.fit(ex_tr)
    hon = scratch.accuracy(ex_va)
    uniform = PL.Planner()
    print(json.dumps({"planner_out_of_sample": hon,
                      "n_train_examples": len(ex_tr),
                      "n_valid_examples": len(ex_va),
                      "chance_top1": round(1.0 / len(PL.FAMILIES), 4)},
                     indent=1))

    # refit on the whole development split for deployment
    final = PL.Planner()
    for r in with_w:
        for w in set(r["winners"]):
            final.memory.add(r["sigs"], w, r["id"])
    ex_all = build_examples(with_w, final, random.Random(9))
    final.fit(ex_all)
    final.meta = {"trained_on": os.path.basename(a.tasks),
                  "n_tasks": len(with_w), "n_examples": len(ex_all),
                  "out_of_sample": hon,
                  "probe": "leave-one-training-pair-out",
                  "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    final.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
