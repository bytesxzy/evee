"""Run inference on an ARC JSON file: python -m engine task.json --budget 20."""
import argparse
import json
import math
import sys
from pathlib import Path

from . import portfolio


def solve_task(task, budget=20.0, k=2, loo=True):
    """Separate demonstration evidence from test inputs at the API boundary."""
    if not isinstance(task, dict) or not task.get("train") or not task.get("test"):
        raise ValueError("task requires nonempty train and test arrays")
    train = [(pair["input"], pair["output"]) for pair in task["train"]]
    inputs = [pair["input"] for pair in task["test"]]
    result = portfolio.solve(train, inputs, time_budget=budget, k=k, loo=loo)
    return {
        "predictions": [
            {"attempt_%d" % (i + 1): [list(row) for row in g]
             for i, g in enumerate(guesses)} for guesses in result.predictions
        ],
        "chosen": result.chosen,
        "hypotheses": result.hyps,
        "elapsed_seconds": result.elapsed,
        "generated": result.n_hyps,
        "fitted": result.n_fit,
        "diagnostics": getattr(result, "diagnostics", {}),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--no-loo", action="store_true")
    parser.add_argument("--out", type=Path, help="Write JSON to a file instead of stdout")
    args = parser.parse_args(argv)
    if not math.isfinite(args.budget) or args.budget <= 0 or args.k <= 0:
        parser.error("budget must be finite and positive; k must be positive")
    try:
        task = json.loads(args.task.read_text(encoding="utf-8-sig"))
        result = solve_task(task, args.budget, args.k, not args.no_loo)
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
