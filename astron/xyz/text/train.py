"""One gated fitting pass for the text model.

Same discipline as ``gabriel/train.py``: fit, measure on a dev slice the fit
never saw, and **adopt only if the gate improved**.  The incumbent artifact is
left alone otherwise, and every round -- adopted or rejected -- is appended to
``policy/text_lineage.jsonl``.

This is seconds of arithmetic, not a training run.  The corpus is a few
thousand categorical signature/family pairs and the model is sparse and linear;
a full pass is well under a second on one core.  That is deliberate: the ARC
side is where compute belongs, and a router that needed a GPU would be the
wrong router.

Usage::

    python3 -m text.train                  # fit, gate, write policy/ if adopted
    python3 -m text.train --dry-run        # fit and report, write nothing
    python3 -m text.train --force          # write regardless of the gate
"""

import argparse
import json
import os
import time

from . import corpus as C
from . import lm as LM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POLICY = os.path.join(ROOT, "policy")
MODEL_PATH = os.path.join(POLICY, "text_gabriel_lm.json")
LINEAGE_PATH = os.path.join(POLICY, "text_lineage.jsonl")
EVIDENCE = os.path.join(ROOT, "evidence", "text_lm.json")


def fit(verbose=False):
    """Fit both heads from scratch.  Returns ``(model, report)``."""
    t0 = time.time()
    ftr, fdev, fhold, fstats = C.build_family_corpus()
    ctr, cdev, chold, cstats = C.build_candidate_corpus()

    model = LM.TextLM()
    fam_stats = model.train_families(ftr, dev=fdev, verbose=verbose)
    cand_stats = model.train_candidates(ctr, dev=cdev, verbose=verbose)

    report = {
        "family": dict(fam_stats),
        "candidate": dict(cand_stats),
        "corpus": {"family": fstats, "candidate": cstats},
        "gate": {
            "family_dev_accuracy": round(model.family_accuracy(fdev), 4),
            "family_dev_loss": round(model.family_loss(fdev), 4),
            "candidate_dev_loss": round(model.candidate_loss(cdev), 4),
        },
        "holdout": {
            # measured once, gates nothing -- the transfer estimate
            "family_accuracy": round(model.family_accuracy(fhold), 4),
            "family_loss": round(model.family_loss(fhold), 4),
            "candidate_loss": round(model.candidate_loss(chold), 4),
            "candidate_top1": round(_candidate_top1(model, chold), 4),
        },
        "baseline": {
            "family_dev_accuracy_uniform":
                round(_uniform_family_accuracy(fdev), 4),
            "family_holdout_accuracy_uniform":
                round(_uniform_family_accuracy(fhold), 4),
        },
        "seconds": round(time.time() - t0, 2),
    }
    model.meta = {
        "trained_at": int(time.time()),
        "family_features": len(model.family_w),
        "candidate_features": len(model.cand_w),
        "family_dev_accuracy": report["gate"]["family_dev_accuracy"],
        "family_holdout_accuracy": report["holdout"]["family_accuracy"],
    }
    return model, report


def _uniform_family_accuracy(examples):
    """What a model that only knows the grammar would score.

    The grammar already excludes most families, so this is the number the fitted
    model has to beat for the fit to have been worth anything.
    """
    if not examples:
        return 0.0
    hit = 0
    for ex in examples:
        allowed = sorted(ex["allowed"])
        if allowed and allowed[0] == ex["family"]:
            hit += 1
    return hit / float(len(examples))


def _candidate_top1(model, examples):
    """Fraction of scenarios whose intended candidate scores highest."""
    groups = {}
    for ex in examples:
        groups.setdefault(ex["text"], []).append(ex)
    hit, n = 0, 0
    for _text, rows in groups.items():
        if not any(r["label"] for r in rows):
            continue
        best = max(rows, key=lambda r: model.candidate_score(r["qsigs"], r["csigs"]))
        hit += 1 if best["label"] else 0
        n += 1
    return hit / float(max(1, n))


def adopt(model, report, path=MODEL_PATH, force=False):
    """Write the model only if it beat the incumbent on the gate."""
    incumbent = LM.TextLM.load(path)
    gate_new = report["gate"]["family_dev_loss"]
    gate_cand_new = report["gate"]["candidate_dev_loss"]
    decision = {"t": int(time.time()), "adopted": False,
                "family_dev_loss": gate_new,
                "candidate_dev_loss": gate_cand_new,
                "family_dev_accuracy": report["gate"]["family_dev_accuracy"],
                "holdout_family_accuracy": report["holdout"]["family_accuracy"]}
    if incumbent is None or force:
        decision["adopted"] = True
        decision["reason"] = "no incumbent" if incumbent is None else "forced"
    else:
        _tr, fdev, _fh, _s = C.build_family_corpus()
        _ct, cdev, _ch, _s2 = C.build_candidate_corpus()
        old_fam = incumbent.family_loss(fdev)
        old_cand = incumbent.candidate_loss(cdev)
        decision["incumbent_family_dev_loss"] = round(old_fam, 4)
        decision["incumbent_candidate_dev_loss"] = round(old_cand, 4)
        better = (gate_new <= old_fam + 1e-9) and (gate_cand_new <= old_cand + 1e-9)
        decision["adopted"] = bool(better)
        decision["reason"] = "dev loss fell" if better else "dev loss did not fall"
    if decision["adopted"]:
        model.save(path)
    os.makedirs(POLICY, exist_ok=True)
    with open(LINEAGE_PATH, "a") as fh:
        fh.write(json.dumps(decision, sort_keys=True) + "\n")
    return decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=MODEL_PATH)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    model, report = fit(verbose=a.verbose)
    if a.dry_run:
        report["adoption"] = {"adopted": False, "reason": "dry run"}
    else:
        report["adoption"] = adopt(model, report, path=a.out, force=a.force)
        os.makedirs(os.path.dirname(EVIDENCE), exist_ok=True)
        with open(EVIDENCE, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
