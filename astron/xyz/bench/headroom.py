"""How much of the remaining gap is ranking rather than search?

For tasks the engine failed, ask whether the correct answer was produced by
*some* fitting hypothesis and merely ranked below the winner.  That number is
the ceiling a better ranking rule could reach without any new solver at all.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import grid as G, portfolio  # noqa: E402

DATA = os.path.join(ROOT, "data", "arc")


def one(args):
    tid, budget = args
    with open(os.path.join(DATA, tid + ".json")) as fh:
        d = json.load(fh)
    train = [(p["input"], p["output"]) for p in d["train"]]
    tins = [p["input"] for p in d["test"]]
    ans = [G.from_list(p["output"]) for p in d["test"]]
    try:
        r = portfolio.solve(train, tins, time_budget=budget, k=99,
                            collect_all=True)
    except Exception:
        return tid, -1, 0
    ranks = []
    for preds, a in zip(r.predictions, ans):
        ranks.append(preds.index(a) if a in preds else -1)
    if any(x < 0 for x in ranks):
        return tid, -1, len(r.predictions[0]) if r.predictions else 0
    return tid, max(ranks), len(r.predictions[0])


def main():
    run = json.load(open(os.path.join(ROOT, sys.argv[1])))
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ids = [r["id"] for r in run["per_task"] if not r["solved"] and r["hyps"]]
    if limit:
        ids = ids[:limit]
    res = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(one, (t, budget)) for t in ids]
        for f in as_completed(futs):
            res.append(f.result())
    reach = [t for t, k, _n in res if k >= 0]
    print(json.dumps({
        "checked": len(res),
        "answer_present_but_misranked": len(reach),
        "would_be_solved_at_top2": sum(1 for _t, k, _n in res if 0 <= k < 2),
        "would_be_solved_at_top5": sum(1 for _t, k, _n in res if 0 <= k < 5),
        "mean_candidates": round(sum(n for _t, _k, n in res) / max(1, len(res)), 1),
        "examples": sorted(reach)[:20],
    }, indent=2))


if __name__ == "__main__":
    main()
