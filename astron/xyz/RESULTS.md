# Measured results

> **Superseded.** The numbers on this page describe an earlier revision. The
> current engine, the diagnostics that shaped it and its before/after
> measurement are in [`ARC-IMPROVEMENTS.md`](ARC-IMPROVEMENTS.md). In short, on
> the same 550 public task files at the same 20-second budget:
> ARC-AGI-1 **175 -> 197 (43.75% -> 49.25%)**, ARC-AGI-2 **45 -> 50 (30.00% ->
> 33.33%)**, 28 newly solved, 1 regressed. On the held-out split that no
> development decision was allowed to see, 134 -> 140.


> Historical report from the supplied archive. The numbers below describe that
> earlier code, not the improved edition. Current revision checks and the paired
> 40-task smoke evaluation are recorded in `IMPROVEMENTS.md` and
> `evidence/regression_*.json`. The full 550-task benchmark has not been rerun
> for this edition.

All numbers on this page were produced by the code in this directory, on the
550 public ARC task files in `data/arc/`, in this environment. Nothing is
estimated, extrapolated, or carried over from another machine.

## Headline

| engine | tasks solved (1 attempt) | rate | 2 attempts | rate |
|---|---:|---:|---:|---:|
| CELL4 Astra V2 (previous, Lua) | 64 / 550 | 11.6% | - | - |
| **ASTRA** (this engine) | **209 / 550** | **38.0%** | 222 / 550 | 40.4% |

**+145 tasks, +227% relative, +26.4 accuracy points.** Paired per task: **147 wins, 2 losses**, sign test p = 1.57e-41.

## By corpus

| corpus | tasks | previous | ASTRA | ASTRA rate |
|---|---:|---:|---:|---:|
| ARC-AGI-1 files | 400 | 52 | 168 | 42.0% |
| ARC-AGI-2 files | 150 | 12 | 41 | 27.3% |

## Run settings

| | |
|---|---|
| per-task budget | 20 s wall clock |
| attempts scored | top-1 and top-2 |
| total CPU | 2020 s over 550 tasks (3.7 s/task mean) |
| wall clock | 508 s at 4-way parallelism |
| bounded solver errors | 0 (counted as unsolved) |
| learned policy | none (stock engine) |

Tasks the previous engine solved and this one does not: `arc1_46f33fce`, `arc1_8eb1be9a`.


## How the comparison is made

The previous engine (CELL4 Astra V2) is written in Lua and needs LuaJIT, which
is not available in this container, so its column is **its own recorded
per-task run**, shipped in `evidence/baseline-v2-arc.json` — not a re-run and
not an aggregate rate. The comparison is paired: identical task files,
identical identifiers, a per-task win/loss table, and a one-sided sign test
over the discordant tasks. `bench/compare.py` asserts that the two task-id sets
are identical before it counts anything.

## Scoring rules

- A task counts as solved only when **every** test pair of that task is exact.
  Tasks with two test inputs must get both.
- **top-1** is one attempt per test input. **top-2** is the two attempts
  official ARC-AGI allows. Both are reported; top-1 is the headline.
- Mean cell agreement is recorded per task but is **never** scored — it is a
  diagnostic for finding near misses.
- Solver exceptions are caught, counted, and left in the denominator as
  unsolved. They are not excluded.
- The solver receives train pairs and test *inputs*. Test outputs stay in the
  harness; `engine/task.py`'s `Ctx` has no field that could hold them, and
  `tests/test_engine.py` asserts it.

## Reproducing

```sh
python3 -m unittest discover -s tests -v
python3 bench/run_arc.py --budget 20 --jobs 4 --out evidence/run.json
python3 bench/compare.py evidence/run.json
python3 bench/evolve.py --rounds 3 --budget 14 --jobs 4 --final-holdout
```

Timings vary with the machine; the solved set does not, except where a task
sits near its per-task deadline.

## What the measurements say about the remaining gap

Two diagnostics were run rather than assumed, and both point the same way.

**More compute buys nothing.** The identical engine measured at a 45 s per-task
budget solves exactly the same 209 tasks (and the same 222 at two attempts) as
at 20 s. The search saturates well inside the budget; the bottleneck is the
hypothesis space, not the clock.

**Better ranking buys little.** `bench/headroom.py` re-runs every failed task
collecting *all* distinct predictions any fitting hypothesis produced, and asks
whether the right answer was in there and merely out-ranked. On a 70-task
sample of the failures it was present 10 times -- about 14%. The engine is not
mostly making the wrong choice among answers it found; on the tasks it misses
it usually never generates the right answer at all.

That is why the last several rounds of work went into new hypothesis families
rather than into tuning, and it is the honest statement of where the ceiling
currently sits.

## Honest scope

These are public development sets. Public ARC training tasks informed which
solver families were worth writing, and the fit split participates in policy
selection. The holdout figure is the transfer estimate. Nothing here is an
official ARC-AGI hidden-test result and it should not be read as one.
