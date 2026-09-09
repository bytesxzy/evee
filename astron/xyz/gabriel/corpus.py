"""Turn the engine's own run records into language-model training data.

The corpus is every program the engine has ever synthesised that fitted all of
a task's training pairs, paired with that task's signatures.  It is read out of
``evidence/*.json`` -- the files ``bench/run_arc.py`` already writes -- so
training data accumulates as a side effect of benchmarking, with nothing to
collect separately.

Two split rules, both deliberate:

* Only tasks in the **fit** split of ``bench/evolve.py`` are read at all.  The
  evolve holdout is never trained on and never gates anything, here either.
* Inside the fit split, a further slice is held out as the language model's own
  dev set.  Adoption is gated on perplexity there, so the gate and the training
  data never overlap -- the same discipline the sign test enforces for the
  symbolic policy.

Nothing here reads a record's ``solved`` flag.  A program enters the corpus
because it reproduced every training pair, which is the only thing the engine
checked before proposing it; whether it then turned out to be right on the
task's test pair does not affect its weight, its presence, or anything else.
The language model therefore learns what programs for a task of this shape
*look like*, and learns it from information a solver is allowed to have.
"""

import glob
import hashlib
import json
import os

from . import tokens as T

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVIDENCE = os.path.join(ROOT, "evidence")
DEV_SEED = "gabriel-dev-v1"


def _fit_split(ids, hold_frac=0.4):
    """The evolve fit/hold split, imported so the two can never diverge."""
    from bench.evolve import split
    fit, _hold = split(sorted(i + ".json" for i in ids), hold_frac)
    return {f[:-5] for f in fit}


def _is_dev(task_id, dev_frac):
    d = hashlib.md5((DEV_SEED + task_id).encode()).digest()[0] / 255.0
    return d < dev_frac


def read_records(evidence_dir=EVIDENCE, extra=()):
    """Every per-task record in the evidence directory, newest file last."""
    out = []
    paths = sorted(glob.glob(os.path.join(evidence_dir, "*.json"))) + list(extra)
    for path in paths:
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        rows = d.get("per_task") if isinstance(d, dict) else None
        if not isinstance(rows, list):
            continue
        for r in rows:
            if isinstance(r, dict) and r.get("id") and r.get("sigs"):
                out.append(r)
    return out


def programs_of(record):
    """(family, program) pairs a record contributes, deduplicated."""
    out, seen = [], set()
    fam = record.get("solver") or ""
    for prog in [record.get("program")] + list(record.get("programs") or []):
        if not prog or not isinstance(prog, str):
            continue
        key = (fam, prog)
        if key in seen:
            continue
        seen.add(key)
        out.append((fam, prog))
    return out


def build(evidence_dir=EVIDENCE, dev_frac=0.15, hold_frac=0.4,
          min_count=1, extra=()):
    """Return ``(train_examples, dev_examples, stats)``.

    An example is ``{"sigs": [...], "body": [tokens], "weight": w, "id": ...}``.
    Identical (task, family, program) triples are collapsed and their
    multiplicity becomes a log weight, so a program that eight benchmark runs
    all rediscovered counts for more than one that appeared once -- but not
    eight times more.
    """
    import math
    records = read_records(evidence_dir, extra)
    ids = {r["id"] for r in records}
    fit_ids = _fit_split(ids, hold_frac)
    counts, sigs_of = {}, {}
    for r in records:
        tid = r["id"]
        if tid not in fit_ids:
            continue
        sigs_of[tid] = list(r.get("sigs") or [])
        for fam, prog in programs_of(r):
            counts[(tid, fam, prog)] = counts.get((tid, fam, prog), 0) + 1
    train, dev = [], []
    n_dropped = 0
    for (tid, fam, prog), c in sorted(counts.items()):
        if c < min_count:
            continue
        body = T.tokenize(prog)
        if not body or len(body) >= T.MAX_TOKENS:
            n_dropped += 1
            continue
        ex = {"id": tid, "sigs": sigs_of.get(tid, []), "family": fam,
              "program": prog, "weight": round(1.0 + math.log(c), 3),
              "body": ([T.fam_token(fam)] if fam else []) + body + [T.EOS]}
        (dev if _is_dev(tid, dev_frac) else train).append(ex)
    stats = {"records": len(records), "tasks": len(ids),
             "fit_tasks": len(fit_ids), "examples": len(train) + len(dev),
             "train": len(train), "dev": len(dev), "dropped": n_dropped,
             "evidence_files": len(glob.glob(os.path.join(evidence_dir, "*.json")))}
    return train, dev, stats
