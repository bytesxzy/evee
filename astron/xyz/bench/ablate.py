"""Turn each new component off in turn and measure what it was worth.

Complexity is easy to add and hard to justify.  Every configuration below runs
the same corpus, the same budget and the same scoring as the headline run, with
exactly one part of the architecture removed, so the difference is attributable.

Ablations run on the held-out split.  The development split shaped the
architecture, so a component's contribution measured there would be flattered by
the same selection that put it in.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (label, extra arguments) -- everything else is held fixed
CONFIGS = [
    ("full", []),
    ("no_relational_process", ["--drop", "relproc"]),
    ("no_hypothesis_cache", ["--drop", "hypcache"]),
    ("no_planner", ["--planner", ""]),
    ("no_object_ops", ["--drop", "objops"]),
    ("no_object_induction", ["--drop", "objproc,objchain"]),
    ("no_deep_composition", ["--drop", "refine,cascade,conditional,objchain"]),
    ("no_new_structure_families",
     ["--drop", "paneltable,selfstamp,extend,tally,locate,assemble"]),
    ("no_counterfactual_loo", ["--no-loo"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(ROOT, "data", "split_hold"))
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--planner", default=os.path.join(ROOT, "policy",
                                                      "arc_planner.json"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "evidence",
                                                     "ablation"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = []
    for label, extra in CONFIGS:
        if a.only and label not in a.only.split(","):
            continue
        out = os.path.join(a.outdir, label + ".json")
        cmd = [sys.executable, os.path.join(HERE, "run_arc.py"),
               "--tasks", a.tasks, "--budget", str(a.budget),
               "--jobs", str(a.jobs), "--out", out, "--quiet"]
        if "--planner" not in extra:
            cmd += ["--planner", a.planner]
        cmd += extra
        t0 = time.time()
        print("=== %s ===" % label, flush=True)
        subprocess.run(cmd, check=False)
        with open(out) as fh:
            d = json.load(fh)
        rows.append({"config": label, "n": d["n"], "solved": d["solved"],
                     "rate": d["rate"], "top2": d["solved_top2"],
                     "cpu_sum": d["cpu_sum"], "wall": round(time.time() - t0, 1),
                     "no_hypothesis": sum(1 for r in d["per_task"]
                                          if not r["solved"] and not r["hyps"])})
        print(json.dumps(rows[-1], indent=1), flush=True)
    base = next((r for r in rows if r["config"] == "full"), None)
    if base:
        for r in rows:
            r["delta_vs_full"] = r["solved"] - base["solved"]
    print(json.dumps(rows, indent=1))
    with open(os.path.join(a.outdir, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
