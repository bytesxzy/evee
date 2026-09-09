# ARC-AGI in the page

The C4 mini assistant answers ARC-AGI questions by **running the solver**, not
by retrieving a sentence that claims a score. Everything runs in the browser:
no hosted model, no API, no network call of any kind.

## What ships

| file | what it is | size |
|---|---|---:|
| `c4-arc-engine.js` | the solver: a JavaScript port of `astron/xyz/engine`, all 35 hypothesis generators | 12.1k lines |
| `c4-arc-tasks.js` | the 550 public ARC-AGI-1 and ARC-AGI-2 task files, packed one row per task | 708 KB (112 KB gzipped) |
| `c4-arc-policy.js` | the trained planner: sparse log-linear weights over reasoning actions | 124 KB |
| `c4-arc.js` | the assistant route — intent detection, live measurement, rendering | 1 file |

`c4-mini.html` loads them in that order and dispatches to `C4ARC` at the top of
`tryReason`, ahead of every other route.

## Measured results

Both engines were run on this machine over the same 550 task files, 20 seconds
per task, two attempts scored, with the trained planner active.

| | tasks | solved | rate |
|---|---:|---:|---:|
| ARC-AGI-1 | 400 | 201 | 50.25% |
| ARC-AGI-2 | 150 | 51 | 34.00% |
| **total** | **550** | **252** | **45.82%** |

The JavaScript port is verified against the Python original per task, not just
in aggregate. On the 236-task development split the two engines solve **the
same 111 tasks** — no task solved by one and missed by the other.

Reproduce:

```sh
python3 astron/xyz/bench/run_arc.py --budget 20 --jobs 4 \
    --planner astron/xyz/policy/arc_planner.json      # the original
node c4-arc/bench.js --budget 20 --jobs 4              # the port
```

## How the score is answered in chat

`C4ARC.answer` never returns a stored number as if it were fresh. A score
question runs the engine live, in a Web Worker so the panel stays responsive,
over an evenly spaced sample of the bundled corpus, and reports exactly what it
just measured — count, denominator, per-task budget and wall clock — before
quoting the full-corpus figure as a separately reproducible reference.

Asking for the *full* ARC run starts all 550 tasks at the 20-second budget in
the page. That is roughly three CPU hours; the sample exists because a chat
turn is not three hours long.

## Why it is not a demo

* Every candidate rule is executed against **every** demonstration and
  discarded unless it reproduces all of them. Nothing is scored on partial
  agreement, and nothing is accepted on plausibility.
* Test outputs never reach the solver. `Ctx` is built from the train pairs and
  the test *inputs* only; the harness holds the answers and compares afterwards.
* The planner reorders and rebudgets generators. It cannot make the engine
  accept a wrong answer, because acceptance is still exact reproduction.
* The corpus is the real public task files, shipped whole, not a curated
  handful.

## Building

```sh
node c4-arc/build.sh        # src/*.js -> c4-arc-engine.js
node c4-arc/make-tasks.js   # data/arc/*.json -> c4-arc-tasks.js
node c4-arc/make-policy.js  # policy/arc_planner.json -> c4-arc-policy.js
```

The engine sources live in `c4-arc/src/` and are concatenated in file-name
order into one IIFE. They share a lexical scope, so the order is fixed by the
numeric prefixes: grid algebra, task context, objects, the enumerator, the
learned policy and planner, then the 35 solvers, then the portfolio.

## Faithfulness notes

The port reproduces the original's behaviour, including one place where that
means reproducing a fault: `counting._fit_solid` indexes a Python `set`, which
raises `TypeError` and aborts the whole `counting` module for that task. The
portfolio catches it and records `status: "error"`. Silently fixing it in the
port would have changed the measured score, so `38-counting.js` raises in the
same place and says why.
