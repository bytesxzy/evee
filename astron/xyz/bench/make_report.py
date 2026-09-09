"""Render the results table into RESULTS.md from measured evidence.

Numbers in the write-up come from the JSON the harness wrote, never from
transcription, so the document cannot drift from the run.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MARK = "<!--RESULTS-TABLE-->"


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "-"


def main():
    run = json.load(open(os.path.join(ROOT, sys.argv[1])))
    cmp_ = json.load(open(os.path.join(ROOT, "evidence", "comparison.json")))
    evo = None
    p = os.path.join(ROOT, "evidence", "evolution.json")
    if os.path.exists(p):
        evo = json.load(open(p))

    n = run["n"]
    b = cmp_["baseline_solved"]
    lines = []
    a = lines.append
    a("## Headline")
    a("")
    a("| engine | tasks solved (1 attempt) | rate | 2 attempts | rate |")
    a("|---|---:|---:|---:|---:|")
    a("| CELL4 Astra V2 (previous, Lua) | %d / %d | %s | - | - |"
      % (b, n, pct(b, n)))
    a("| **ASTRA** (this engine) | **%d / %d** | **%s** | %d / %d | %s |"
      % (run["solved"], n, pct(run["solved"], n),
         run["solved_top2"], n, pct(run["solved_top2"], n)))
    a("")
    a("**%+d tasks, %+.0f%% relative, %+.1f accuracy points.** "
      "Paired per task: **%d wins, %d losses**, sign test p = %.2e."
      % (cmp_["absolute_gain"], cmp_["relative_gain_pct"],
         cmp_["accuracy_points"], cmp_["paired_wins"], cmp_["paired_losses"],
         cmp_["sign_test_p"]))
    a("")
    a("## By corpus")
    a("")
    a("| corpus | tasks | previous | ASTRA | ASTRA rate |")
    a("|---|---:|---:|---:|---:|")
    for key, label in (("arc1", "ARC-AGI-1 files"), ("arc2", "ARC-AGI-2 files")):
        c = cmp_[key]
        a("| %s | %d | %d | %d | %s |"
          % (label, c["n"], c["baseline"], c["candidate"],
             pct(c["candidate"], c["n"])))
    a("")
    a("## Run settings")
    a("")
    a("| | |")
    a("|---|---|")
    a("| per-task budget | %.0f s wall clock |" % run["budget"])
    a("| attempts scored | top-1 and top-2 |")
    a("| total CPU | %.0f s over %d tasks (%.1f s/task mean) |"
      % (run["cpu_sum"], n, run["cpu_sum"] / n))
    a("| wall clock | %.0f s at %d-way parallelism |" % (run["wall"], 4))
    a("| bounded solver errors | %d (counted as unsolved) |" % run["errors"])
    a("| learned policy | %s |" % (run.get("policy") or "none (stock engine)"))
    a("")
    if cmp_["lost_tasks"]:
        a("Tasks the previous engine solved and this one does not: %s."
          % ", ".join("`%s`" % t for t in cmp_["lost_tasks"]))
    else:
        a("**No regressions**: every task the previous engine solved, this one "
          "also solves.")
    a("")
    if evo:
        a("## Self-improvement loop")
        a("")
        a("| round | fit split before | after | wins | losses | p | adopted |")
        a("|---:|---:|---:|---:|---:|---:|---|")
        for r in evo["lineage"]:
            a("| %d | %d | %d | %d | %d | %.3f | %s |"
              % (r["round"], r["fit_before"], r["fit_after"], r["wins"],
                 r["losses"], r["p"], "yes" if r["accepted"] else "no"))
        a("")
        h = evo.get("holdout")
        if h:
            a("Holdout (%d tasks, never fitted, never gated on): stock engine "
              "**%d**, learned policy **%d** (%d wins, %d losses, p = %.3f)."
              % (h["n"], h["base"], h["policy"], h["wins"], h["losses"], h["p"]))
            a("")

    body = "\n".join(lines)
    path = os.path.join(ROOT, "RESULTS.md")
    text = open(path).read()
    if MARK in text:
        text = text.replace(MARK, body)
    else:                       # already rendered once: replace the block
        head, _, rest = text.partition("## Headline")
        tail = rest.partition("## How the comparison is made")[2]
        text = head + body + "\n## How the comparison is made" + tail
    open(path, "w").write(text)
    print(body)


if __name__ == "__main__":
    main()
