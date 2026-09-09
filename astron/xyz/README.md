# ASTRA — ARC reasoning engine, improved edition

A pure Python symbolic ARC solver. It infers programs from demonstration pairs,
validates each proposed rule against every demonstration, and ranks distinct
predictions. No neural model, third-party dependencies, or retraining required.
The existing data and historical evidence from the supplied archive are retained.

## Run inference

From this directory, using Python 3.8 or newer:

```sh
python -m engine data/arc/arc1_6150a2bd.json --budget 20 --out predictions.json
```

Any ARC task JSON with `train` input/output pairs and `test` inputs works. Test
outputs are optional and never passed to the solver. Output JSON contains up to
two attempts per input, the selected rule, alternative hypotheses, and diagnostic
information. An empty attempt object means no valid explanation was found.
Use `--k 1` for one attempt or `--no-loo` to disable demonstration refits.

The budget is cooperative: individual Python operations can finish after their
deadline. Use the isolated comparison runner below when a process timeout matters.

## What changed

- **Compositional reasoning:** cost/depth-aware equivalence pruning, a beam
  retaining both cheap and promising programs, and color mappings inferred from
  demonstrations after a structural transform. Binary search has its own budget
  share and checks output dimensions before constructing impossible grids.
- **Evidence-based ranking:** leave-one-out evidence follows the specific rule
  and its test predictions; incomplete refits remain neutral. Equivalent behaviors
  cannot crowd alternatives out of voting.
- **Relative object motion:** align objects to uniquely identified anchors,
  recomputing movement for each grid. Shared displacement constraints resolve
  identical copies that greedy matching mishandled.
- **Reliable boundaries:** reject malformed grids and nonfinite scores, contain
  generator failures, retain late inexpensive hypotheses, and expose diagnostics.
- **Honest measurement:** an isolated before/after runner, source and task hashes,
  fixed selection, and exact scoring of every test pair.

These are inference and architecture changes. No model training, policy update,
or evolution run was performed. See [IMPROVEMENTS.md](IMPROVEMENTS.md) for validation
and limitations. [RESULTS.md](RESULTS.md) contains the supplied edition's historical
550-task results, which must not be attributed to this revision without a rerun.

## Verify and compare

```sh
python -m unittest discover -s tests -v

# Repeat the bundled 40-task smoke comparison on this checkout:
python bench/regression.py --manifest evidence/regression_before.json --budget 1 --out evidence/recheck.json
python bench/regression.py --compare evidence/regression_before.json evidence/recheck.json

# Optional full public development-set evaluation (no training):
python bench/run_arc.py --budget 20 --jobs 4 --out evidence/full_run.json
```

Short wall-clock searches vary with hardware and load. A smoke comparison tests
regressions; it is not an official ARC score or evidence about hidden tasks.

## Layout

| Path | Purpose |
|---|---|
| `engine/task.py`, `engine/grid.py` | Evidence boundary and immutable grid algebra |
| `engine/objects.py` | Cached object segmentations and features |
| `engine/enum_core.py` | Bounded program synthesis |
| `engine/portfolio.py` | Candidate validation, refits, ranking, diagnostics |
| `engine/solvers/` | Specialist and compositional reasoning families |
| `engine/__main__.py` | JSON inference command |
| `bench/regression.py` | Isolated paired evaluation, without training |
| `tests/` | Behavioral and regression tests |
| `data/arc/`, `evidence/` | Supplied public tasks and measured evidence |
| `engine/learn.py`, `bench/evolve.py` | Existing optional learning code; not required |

The original architecture discussion is retained in
[ARCHITECTURE.md](ARCHITECTURE.md), with a note distinguishing the updated design.

## Current ARC results

See [`ARC-IMPROVEMENTS.md`](ARC-IMPROVEMENTS.md) for the measurements that
shaped this revision, what was added, and the before/after numbers:
ARC-AGI-1 43.75% -> 49.25%, ARC-AGI-2 30.00% -> 33.33% on the bundled corpora.
