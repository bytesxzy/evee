"""Train the GABRIEL language model, and adopt it only if it got better.

    python3 -m gabriel.train --max-seconds 120

One invocation is one gated training pass:

1. build the corpus from ``evidence/`` (fit split only);
2. warm-start from the model currently in force;
3. run SGD for a bounded number of seconds;
4. measure both models' perplexity on the same held-out dev slice, over the
   same vocabulary, so the comparison is paired;
5. write the candidate to ``policy/gabriel_lm.json`` only if it wins.

Step 5 is the whole point.  ``bench/evolve.py`` refuses to adopt a symbolic
policy that cannot beat its incumbent on a sign test; a language model that
quietly got worse overnight would be exactly the same failure, so it is refused
in the same way.  A rejected candidate is left in ``policy/`` under its own
name for inspection and the model in force is untouched.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gabriel import corpus                      # noqa: E402
from gabriel.lm import GabrielLM                # noqa: E402

POLICY_DIR = os.path.join(ROOT, "policy")
LM_PATH = os.path.join(POLICY_DIR, "gabriel_lm.json")
LINEAGE = os.path.join(POLICY_DIR, "gabriel_lineage.jsonl")
REPORT = os.path.join(ROOT, "evidence", "gabriel_lm.json")


def on_support(model, vocab):
    """A copy of ``model`` normalised over ``vocab``, still knowing only its own.

    Perplexities are only comparable over a common support, so the incumbent is
    re-normalised onto the candidate's vocabulary before the two are compared.
    What must *not* change is which tokens the incumbent knows: a token it was
    never trained on has no weights, and scoring that as a bare zero would rank
    it above every token the incumbent learned to distrust.  Its perplexity
    would blow up, and any candidate at all would look like an improvement.
    Keeping ``unigram`` means those tokens are scored as ``<unk>`` -- which is
    exactly what they are to this model -- and the comparison stays honest.
    """
    out = GabrielLM(model.to_dict())
    out.vocab = list(vocab)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lm", default=LM_PATH)
    ap.add_argument("--evidence", default=corpus.EVIDENCE)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--early-stop-dev", type=int, default=64,
                    help="dev examples used for per-epoch early stopping")
    ap.add_argument("--warm", action="store_true",
                    help="continue from the model in force instead of refitting")
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument("--lr", type=float, default=0.35)
    ap.add_argument("--negatives", type=int, default=12)
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--min-gain", type=float, default=0.005,
                    help="relative perplexity gain required to adopt")
    ap.add_argument("--force", action="store_true",
                    help="adopt the candidate even if it did not win")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    train, dev, stats = corpus.build(a.evidence, dev_frac=a.dev_frac)
    if not train:
        print(json.dumps({"error": "empty corpus", "stats": stats}))
        return 1
    if not a.quiet:
        print("corpus " + json.dumps(stats), flush=True)

    incumbent = GabrielLM.load(a.lm)
    cand = GabrielLM(incumbent.to_dict() if (incumbent and a.warm) else None)
    cand.build_vocab(train + dev)
    # A slice of dev drives early stopping; the whole of it decides adoption.
    # Both come from the same held-out tasks, and neither is ever trained on.
    fit = cand.train(train, epochs=a.epochs, lr=a.lr, negatives=a.negatives,
                     seed=a.seed, max_seconds=a.max_seconds,
                     verbose=not a.quiet, patience=a.patience,
                     dev=dev[:a.early_stop_dev] if dev else None)

    cand_ppl = cand.perplexity(dev) if dev else float("inf")
    base_ppl = None
    if incumbent is not None and incumbent.is_trained() and dev:
        base_ppl = on_support(incumbent, cand.vocab).perplexity(dev)

    if base_ppl is None:
        adopt, why = True, "no incumbent"
    elif cand_ppl < base_ppl * (1.0 - a.min_gain):
        adopt, why = True, "perplexity improved"
    elif a.force:
        adopt, why = True, "forced"
    else:
        adopt, why = False, "no improvement"

    os.makedirs(POLICY_DIR, exist_ok=True)
    cand.meta.update({"dev_perplexity": round(cand_ppl, 4),
                      "trained_at": int(time.time()),
                      "corpus": stats})
    if adopt:
        cand.save(a.lm)
    else:
        cand.save(os.path.join(POLICY_DIR, "gabriel_lm.rejected.json"))

    rec = {"t": int(time.time()), "adopted": bool(adopt), "why": why,
           "dev_perplexity": round(cand_ppl, 4),
           "incumbent_perplexity": round(base_ppl, 4) if base_ppl else None,
           "dev_examples": len(dev), "train_examples": len(train),
           "vocab": fit["vocab"], "features": fit["features"],
           "best_epoch": fit["best_epoch"], "warm": bool(a.warm),
           "tokens": fit["tokens"], "stopped": fit["stopped"],
           "loss_per_token": fit["loss_per_token"],
           "seconds": round(time.time() - t0, 1)}
    with open(LINEAGE, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as fh:
        json.dump({"latest": rec, "corpus": stats, "model": cand.meta}, fh, indent=1)
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
