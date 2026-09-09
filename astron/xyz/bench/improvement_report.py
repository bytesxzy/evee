"""Build aggregate before/after evidence without exposing held-out task details."""
import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import report, split


def metrics(rows):
    n = len(rows)
    return dict(n=n, solved=sum(r["solved"] for r in rows),
                solved_top2=sum(r["solved2"] for r in rows),
                no_hypothesis=sum(not r["hyps"] for r in rows),
                fitted_but_wrong=sum(bool(r["hyps"]) and not r["solved"] for r in rows),
                errors=sum(bool(r["error"]) for r in rows),
                mean_seconds=round(sum(r["time"] for r in rows)/max(1, n), 3))


def compare(before, after):
    a = {r["id"]: r for r in before}
    b = {r["id"]: r for r in after}
    if a.keys() != b.keys():
        raise ValueError("comparisons require identical task sets")
    return dict(before=metrics(before), after=metrics(after),
                newly_solved=sum(not a[k]["solved"] and b[k]["solved"] for k in a),
                regressed=sum(a[k]["solved"] and not b[k]["solved"] for k in a),
                relative_multiplier=round(sum(r["solved"] for r in after) /
                                          max(1, sum(r["solved"] for r in before)), 5))


def groups(before, after):
    out = {}
    for name in ("all", "dev", "hold", "arc1", "arc2",
                 "dev_arc1", "dev_arc2", "hold_arc1", "hold_arc2"):
        def select(r):
            return (("dev" not in name or split.bucket(r["id"]) == "dev") and
                    ("hold" not in name or split.bucket(r["id"]) == "hold") and
                    ("arc1" not in name or r["id"].startswith("arc1")) and
                    ("arc2" not in name or r["id"].startswith("arc2")))
        out[name] = compare([r for r in before if select(r)],
                            [r for r in after if select(r)])
    return out


def failures(rows):
    failed = [r for r in rows if not r["solved"]]
    signatures = Counter(s for r in failed for s in r.get("sigs", []))
    return dict(
        no_hypothesis=sum(not r["hyps"] for r in failed),
        solved_on_second_attempt=sum(r["solved2"] for r in failed),
        fitted_without_top2_solve=sum(bool(r["hyps"]) and not r["solved2"] for r in failed),
        signature_counts=dict(signatures.most_common()))


def source_fingerprint():
    files = {}
    for folder in ("engine", "bench", "tests"):
        for p in sorted((ROOT / folder).rglob("*.py")):
            files[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    files["policy/arc_planner.json"] = hashlib.sha256(
        (ROOT/"policy/arc_planner.json").read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, files=files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", nargs="+", required=True)
    ap.add_argument("--ablation", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    baseline = json.loads(Path(a.before).read_text())["per_task"]
    after = []
    run_reports = []
    for path in a.after:
        rep, rows = report.score(path)
        run_reports.append(rep)
        after.extend(rows)
    if len({r["id"] for r in after}) != len(after):
        raise ValueError("after runs overlap")
    summary = dict(comparison=groups(baseline, after),
                   before_seal=report.seal(baseline),
                   after_seal=report.seal(after),
                   after_runs=run_reports,
                   remaining_failures={s: failures([r for r in after if split.bucket(r["id"]) == s])
                                       for s in ("dev", "hold")},
                   source=source_fingerprint(), ablations=[])
    for path in a.ablation:
        rep, rows = report.score(path)
        ids = {r["id"] for r in rows}
        full = [r for r in after if r["id"] in ids]
        summary["ablations"].append(dict(file=path, seal=rep["seal"],
            removed_vs_full=compare(rows, full)))
    Path(a.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["comparison"], indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

