"""Is the general synthesiser limited by time, by depth, or by beam width?

Three different fixes follow from the three answers, so it is worth measuring
rather than assuming.  For a set of tasks the engine currently fails, this runs
``enum_core.search`` at several (depth, beam) settings with a generous
deadline and reports how many produce a program that fits every training pair,
and how many of those also predict the held-out answer.

The answer decides where effort goes: if widening the beam finds programs that
the default misses, the search is the bottleneck; if it does not, the program
space is, and only new primitives can help.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import enum_core, grid as G   # noqa: E402
from engine.task import Ctx               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETTINGS = [("default", 4, 1500, 12.0), ("wide", 4, 6000, 40.0),
            ("deep", 6, 1500, 40.0), ("wide+deep", 6, 6000, 60.0)]


def one(args):
    path, name, depth, states, budget = args
    with open(path) as fh:
        d = json.load(fh)
    train = [(p["input"], p["output"]) for p in d["train"]]
    tests = d["test"]
    ctx = Ctx(train, [p["input"] for p in tests])
    answers = [G.from_list(p["output"]) for p in tests]
    ctx.deadline = time.time() + budget
    t0 = time.time()
    try:
        found = enum_core.search(ctx, depth=depth, max_states=states,
                                 deadline=ctx.deadline, level="full",
                                 use_binary=True)
    except Exception:
        found = []
    ok = 0
    for _n, _c, fn in found:
        try:
            if all(fn(a) == b for a, b in ctx.train):
                ok += 1
        except Exception:
            pass
    right = 0
    for _n, _c, fn in found:
        try:
            if all(fn(a) == b for a, b in ctx.train) and \
                    all(fn(t) == ans for t, ans in zip(ctx.test_inputs, answers)):
                right = 1
                break
        except Exception:
            pass
    return {"id": os.path.basename(path)[:-5], "setting": name,
            "fitted": ok, "correct": right, "time": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "evidence",
                                                  "DEV_v6.json"))
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with open(a.run) as fh:
        rows = json.load(fh)["per_task"]
    unsolved = [r["id"] for r in rows if not r["solved"]][:a.limit]
    data = os.path.join(ROOT, "data", "arc")
    jobs = [(os.path.join(data, t + ".json"), n, d, s, b)
            for t in unsolved for n, d, s, b in SETTINGS]
    out = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for fu in as_completed(futs):
            try:
                out.append(fu.result())
            except Exception:
                pass
    agg = {}
    for r in out:
        s = agg.setdefault(r["setting"], {"n": 0, "any_fit": 0, "correct": 0,
                                          "time": 0.0})
        s["n"] += 1
        s["any_fit"] += int(r["fitted"] > 0)
        s["correct"] += r["correct"]
        s["time"] += r["time"]
    for k, v in agg.items():
        v["time"] = round(v["time"] / max(1, v["n"]), 2)
    print(json.dumps({"tasks": len(unsolved), "settings": agg}, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"per": out, "agg": agg}, fh, indent=1)


if __name__ == "__main__":
    main()
