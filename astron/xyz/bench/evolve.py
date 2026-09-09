"""Gated self-improvement loop.

Each round the engine (a) runs on the *fit* split with the policy currently in
force, (b) fits a candidate policy from the accumulated experience, (c) re-runs
the fit split under the candidate, and (d) adopts the candidate only if a
paired comparison on identical tasks says it is better.

The gate is a one-sided sign test over discordant tasks::

    p = sum_{i>=w} C(w+l, i) / 2^(w+l)

where ``w`` is the number of tasks the candidate solves and the incumbent does
not, and ``l`` the reverse.  A candidate that merely reshuffles which tasks are
solved does not pass.  This is what stops the loop from drifting: a change has
to *earn* adoption against the version it replaces.

The *holdout* split is never used to fit anything and never gates anything.  It
is measured once, at the end, to report whether the learned policy transfers to
tasks it was not fitted on.  Splitting is by MD5 of the task id, so it is
reproducible and independent of solve order.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from bench.run_arc import DATA, run_one  # noqa: E402
from engine import learn  # noqa: E402

POLICY_DIR = os.path.join(ROOT, "policy")
LINEAGE = os.path.join(POLICY_DIR, "lineage.jsonl")


def split(files, hold_frac=0.4, seed="astra-v3"):
    fit, hold = [], []
    for f in files:
        d = hashlib.md5((seed + f).encode()).digest()[0] / 255.0
        (hold if d < hold_frac else fit).append(f)
    return fit, hold


def run_split(files, policy_path, budget, jobs, k=2):
    jobs_ = [(os.path.join(DATA, f), budget, k, True, None, policy_path)
             for f in files]
    out = []
    if jobs <= 1:
        for j in jobs_:
            out.append(run_one(j))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(run_one, j) for j in jobs_]
            for fu in as_completed(futs):
                try:
                    out.append(fu.result())
                except Exception as exc:
                    out.append({"id": "?", "solved": 0, "solved2": 0,
                                "partial": 0.0, "time": 0.0, "solver": None,
                                "hyps": [], "error": repr(exc)[:120],
                                "n_test": 1, "sigs": [], "program": None})
    return {r["id"]: r for r in out}


def sign_test(w, l):
    """One-sided p-value that the candidate is no better than the incumbent."""
    n = w + l
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(w, n + 1)) / float(2 ** n)


def compare(base, cand):
    wins = sorted(t for t in cand if cand[t]["solved"] and not base.get(t, {}).get("solved"))
    losses = sorted(t for t in cand if base.get(t, {}).get("solved") and not cand[t]["solved"])
    return wins, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--budget", type=float, default=12.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--hold-frac", type=float, default=0.4)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--final-holdout", action="store_true")
    a = ap.parse_args()

    os.makedirs(POLICY_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".json"))
    fit_files, hold_files = split(files, a.hold_frac)
    print("fit=%d hold=%d" % (len(fit_files), len(hold_files)), flush=True)

    cur_path = ""                      # "" means: no policy (base engine)
    history = []
    lineage = []
    t_start = time.time()

    base = run_split(fit_files, cur_path, a.budget, a.jobs)
    n0 = sum(r["solved"] for r in base.values())
    print("round 0 (no policy): fit solved %d/%d" % (n0, len(fit_files)), flush=True)
    history.extend(base.values())

    accepted = 0
    for rnd in range(1, a.rounds + 1):
        prev = learn.Policy.load(cur_path) if cur_path else None
        cand = learn.fit(list(history), prev=prev)
        cand_path = os.path.join(POLICY_DIR, "candidate_r%d.json" % rnd)
        cand.save(cand_path)
        res = run_split(fit_files, cand_path, a.budget, a.jobs)
        n1 = sum(r["solved"] for r in res.values())
        wins, losses = compare(base, res)
        p = sign_test(len(wins), len(losses))
        ok = (len(wins) > len(losses)) and p <= a.alpha
        rec = {"round": rnd, "fit_before": n0, "fit_after": n1,
               "wins": len(wins), "losses": len(losses), "p": round(p, 4),
               "accepted": bool(ok), "abstractions": len(cand.abstractions),
               "signatures": cand.meta.get("n_signatures", 0),
               "win_ids": wins[:20], "loss_ids": losses[:20],
               "elapsed": round(time.time() - t_start, 1)}
        lineage.append(rec)
        print(json.dumps(rec), flush=True)
        history.extend(res.values())
        if ok:
            accepted += 1
            active = os.path.join(POLICY_DIR, "policy.json")
            cand.save(active)
            cur_path = active
            base = res
            n0 = n1
        else:
            print("  rejected: keeping previous policy", flush=True)

    with open(LINEAGE, "w") as fh:
        for r in lineage:
            fh.write(json.dumps(r) + "\n")

    summary = {"fit_n": len(fit_files), "hold_n": len(hold_files),
               "rounds": a.rounds, "accepted": accepted,
               "final_fit_solved": n0, "lineage": lineage,
               "policy": cur_path or None}

    if a.final_holdout:
        print("measuring holdout ...", flush=True)
        h0 = run_split(hold_files, "", a.budget, a.jobs)
        h1 = run_split(hold_files, cur_path, a.budget, a.jobs) if cur_path else h0
        b = sum(r["solved"] for r in h0.values())
        c = sum(r["solved"] for r in h1.values())
        w, l = compare(h0, h1)
        summary["holdout"] = {"base": b, "policy": c, "n": len(hold_files),
                              "wins": len(w), "losses": len(l),
                              "p": round(sign_test(len(w), len(l)), 4)}
        print(json.dumps(summary["holdout"], indent=2))

    with open(os.path.join(ROOT, "evidence", "evolution.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("done in %.0fs" % (time.time() - t_start))


if __name__ == "__main__":
    main()
