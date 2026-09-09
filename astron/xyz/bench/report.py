"""Score a run file by corpus and by split, and seal the evidence.

The seal is a SHA-256 over the task inputs actually scored plus the per-task
outcome, so a later claim about a run can be checked against the run itself
rather than against a summary someone typed out.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import split as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rate(rows, pred=lambda r: True, key="solved"):
    sel = [r for r in rows if pred(r)]
    n = len(sel)
    s = sum(r.get(key, 0) for r in sel)
    return {"n": n, "solved": s, "rate": round(s / float(n), 4) if n else 0.0}


def seal(rows, data=None):
    data = data or os.path.join(ROOT, "data", "arc")
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["id"]):
        p = os.path.join(data, r["id"] + ".json")
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(hashlib.sha256(fh.read()).digest())
        h.update(("%s:%d:%d\n" % (r["id"], r.get("solved", 0),
                                  r.get("solved2", 0))).encode())
    return h.hexdigest()


def score(path):
    with open(path) as fh:
        d = json.load(fh)
    rows = d["per_task"]
    for r in rows:
        r["_split"] = S.bucket(r["id"])
    out = {
        "file": os.path.basename(path),
        "budget": d.get("budget"), "wall": d.get("wall"),
        "cpu_sum": d.get("cpu_sum"),
        "all": _rate(rows),
        "all_top2": _rate(rows, key="solved2"),
        "arc1": _rate(rows, lambda r: r["id"].startswith("arc1")),
        "arc2": _rate(rows, lambda r: r["id"].startswith("arc2")),
        "arc1_top2": _rate(rows, lambda r: r["id"].startswith("arc1"), "solved2"),
        "arc2_top2": _rate(rows, lambda r: r["id"].startswith("arc2"), "solved2"),
        "dev": _rate(rows, lambda r: r["_split"] == "dev"),
        "dev_arc1": _rate(rows, lambda r: r["_split"] == "dev" and r["id"].startswith("arc1")),
        "dev_arc2": _rate(rows, lambda r: r["_split"] == "dev" and r["id"].startswith("arc2")),
        "hold": _rate(rows, lambda r: r["_split"] == "hold"),
        "hold_arc1": _rate(rows, lambda r: r["_split"] == "hold" and r["id"].startswith("arc1")),
        "hold_arc2": _rate(rows, lambda r: r["_split"] == "hold" and r["id"].startswith("arc2")),
        "no_hypothesis": sum(1 for r in rows if not r.get("solved") and not r.get("hyps")),
        "fitted_but_wrong": sum(1 for r in rows if not r.get("solved") and r.get("hyps")),
        "errors": sum(1 for r in rows if r.get("error")),
        "median_time": sorted(r.get("time", 0.0) for r in rows)[len(rows) // 2],
        "p95_time": sorted(r.get("time", 0.0) for r in rows)[int(len(rows) * 0.95)],
        "mean_time": round(sum(r.get("time", 0.0) for r in rows) / float(len(rows)), 3),
        "seal": seal(rows),
    }
    return out, rows


def compare(a, b):
    ra = {r["id"]: r.get("solved", 0) for r in a}
    rb = {r["id"]: r.get("solved", 0) for r in b}
    ids = sorted(set(ra) & set(rb))
    gained = [i for i in ids if not ra[i] and rb[i]]
    lost = [i for i in ids if ra[i] and not rb[i]]
    return {"common": len(ids), "newly_solved": len(gained),
            "regressed": len(lost), "gained_ids": gained, "lost_ids": lost}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    reports, rowsets = [], []
    for p in a.runs:
        rep, rows = score(p)
        reports.append(rep)
        rowsets.append(rows)
        print(json.dumps(rep, indent=1))
    result = {"reports": reports}
    if len(rowsets) == 2:
        result["delta"] = compare(rowsets[0], rowsets[1])
        d = dict(result["delta"])
        d["gained_ids"] = d["gained_ids"][:80]
        d["lost_ids"] = d["lost_ids"][:80]
        print(json.dumps(d, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
