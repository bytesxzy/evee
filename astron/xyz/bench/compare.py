"""Head-to-head comparison against the previous engine's recorded run.

The prior system (CELL4 Astra V2) is written in Lua and needs LuaJIT, which is
not present in this environment, so its numbers are taken from the per-task
results it shipped in ``evidence/baseline-v2-arc.json`` rather than re-run
here.  The comparison is still exact and paired: the same 550 task files, the
same identifiers, and a per-task win/loss table -- not two aggregate rates.

Usage::  python3 bench/compare.py evidence/astra-final.json
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def sign_test(w, l):
    n = w + l
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(w, n + 1)) / float(2 ** n)


def main():
    cand_path = sys.argv[1] if len(sys.argv) > 1 else "evidence/astra-final.json"
    base_path = os.path.join(ROOT, "evidence", "baseline-v2-arc.json")
    with open(os.path.join(ROOT, cand_path) if not os.path.isabs(cand_path)
              else cand_path) as fh:
        cand = json.load(fh)
    with open(base_path) as fh:
        base = json.load(fh)

    b = {t["id"]: int(t["solved"]) for t in base["per_task"]}
    c = {t["id"]: int(t["solved"]) for t in cand["per_task"]}
    ids = sorted(set(b) & set(c))
    assert len(ids) == len(b) == len(c), (
        "task sets differ: %d vs %d vs %d" % (len(ids), len(b), len(c)))

    wins = [t for t in ids if c[t] and not b[t]]
    losses = [t for t in ids if b[t] and not c[t]]
    both = [t for t in ids if b[t] and c[t]]
    p = sign_test(len(wins), len(losses))
    nb, nc = sum(b.values()), sum(c.values())

    def fam(pref):
        f = [t for t in ids if t.startswith(pref)]
        return len(f), sum(b[t] for t in f), sum(c[t] for t in f)

    out = {
        "n_tasks": len(ids),
        "baseline_engine": "cell4 astra v2 (lua, recorded run)",
        "baseline_solved": nb,
        "candidate_solved": nc,
        "candidate_solved_top2": cand.get("solved_top2"),
        "absolute_gain": nc - nb,
        "relative_gain_pct": round(100.0 * (nc - nb) / nb, 2) if nb else None,
        "accuracy_points": round(100.0 * (nc - nb) / len(ids), 2),
        "paired_wins": len(wins),
        "paired_losses": len(losses),
        "both_solved": len(both),
        "sign_test_p": p,
        "arc1": dict(zip(("n", "baseline", "candidate"), fam("arc1"))),
        "arc2": dict(zip(("n", "baseline", "candidate"), fam("arc2"))),
        "lost_tasks": losses,
        "candidate_budget_s": cand.get("budget"),
        "candidate_cpu_seconds": cand.get("cpu_sum"),
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(ROOT, "evidence", "comparison.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
