"""Run inference with the model strapped on::

    python3 -m gabriel data/arc/arc1_007bbfb7.json --budget 20 --out preds.json

Identical output to ``python -m engine`` -- it calls the engine's own
``solve_task``, so the boundary between demonstrations and test inputs is the
same one the engine enforces -- with two differences that are the point of the
package: the fitted policy is actually loaded (``python -m engine`` never loads
one), and the language model is bound to the ranking prior, the enumerator's
operator order, and the proposal stream.

``--stock`` runs the unmodified engine through the same code path, which is how
you compare the two on one task without editing anything.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from engine.__main__ import solve_task

from . import NAME, VERSION, bind


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", type=Path)
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--policy", default=bind.POLICY_PATH)
    ap.add_argument("--lm", default=bind.LM_PATH)
    ap.add_argument("--no-proposals", action="store_true",
                    help="bind the model to ranking only, not to generation")
    ap.add_argument("--stock", action="store_true",
                    help="run the unmodified engine instead")
    a = ap.parse_args(argv)
    if not math.isfinite(a.budget) or a.budget <= 0 or a.k <= 0:
        ap.error("budget must be finite and positive; k must be positive")

    lm = None
    if not a.stock:
        _pol, lm = bind.bind(a.policy, a.lm, proposals=not a.no_proposals)
    try:
        task = json.loads(a.task.read_text(encoding="utf-8-sig"))
        result = solve_task(task, a.budget, a.k, not a.no_loo)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ap.error(str(exc))
        return 2
    result["gabriel"] = {
        "version": "%s %s" % (NAME, VERSION),
        "mode": "stock engine" if a.stock else "language model bound",
        "language_model": (None if lm is None else
                           {"vocab": len(lm.vocab),
                            "features": len(lm.W),
                            "dev_perplexity": lm.meta.get("dev_perplexity"),
                            "trained_at": lm.meta.get("trained_at")}),
        "proposals": bool(lm is not None and not a.no_proposals),
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
        print("wrote %s" % a.out)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
