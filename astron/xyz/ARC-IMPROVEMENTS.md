# ARC reasoning: what was measured, what was changed, and what it bought

This document covers one pass over the ASTRA engine aimed at a single target:
solving more previously unseen ARC tasks. Nothing here is tuned against an
answer key, and no task-specific special case was added. Every number was
produced by the code in this directory on the 550 public task files in
`data/arc/`.

---

## 1. The measurement that decided everything

The starting engine solves **175/400 ARC-AGI-1 (43.8%)** and **45/150
ARC-AGI-2 (30.0%)** at a 20-second per-task budget. Of its 330 failures, 241
ended with **no fitted hypothesis at all** — not a wrong answer, but nothing
proposed that even reproduced the demonstrations.

Before changing anything, three experiments were run to find out *why*.

**Experiment 1 — is it starved of time?** The development split was re-run at a
60-second budget instead of 20.

| budget | dev solved |
|---|---:|
| 20 s | 86 / 236 |
| 60 s | 87 / 236 |

Tripling the compute bought **one task**.

**Experiment 2 — is the ranking losing answers it already has?**
`bench/diagnose.py` re-runs each task collecting *every* distinct prediction any
fitted hypothesis produced, and asks where the true answer sits in that list.

| class | dev tasks |
|---|---:|
| solved | 100 |
| right answer at rank 2 | 6 |
| right answer present but ranked below 2 (**mis-ranked**) | **1** |
| something fit the demonstrations, but nothing produced the answer | 29 |
| nothing fit the demonstrations at all | 100 |

**Experiment 3 — is the synthesiser's search too shallow or too narrow?**
`bench/beam_probe.py` re-runs the general enumerator on 40 unsolved tasks at
four settings.

| setting | depth | beam | budget | tasks with any fitting program | correct |
|---|---:|---:|---:|---:|---:|
| default | 4 | 1500 | 12 s | 1 / 40 | 0 |
| deep | 6 | 1500 | 40 s | 1 / 40 | 0 |
| wide | 4 | 6000 | 40 s | 1 / 40 | 0 |
| wide + deep | 6 | 6000 | 60 s | 1 / 40 | 0 |

Identical at every setting.

The three results agree: **the engine is not short of time, not short of search
depth, not short of beam width, and its ranking is very nearly perfect.** What
it lacks is *expressive reach* — the correct transformation is not in the
program space at any depth. Every change below follows from that, and the work
that a different diagnosis would have justified (better scoring, deeper search,
a bigger budget) was deliberately not done, because the measurements say it is
worth about one task.

---

## 2. What was added

### 2.1 Objects become first-class in the general synthesiser

`engine/objops.py` (new)

The synthesiser composed grid-to-grid functions — rotate, crop, tile, shift,
fill — and had no way to say *which thing* to act on. Object operators close
that: each is a **selector** crossed with an **action**.

| selectors | largest, smallest, widest box, unique colour, majority colour, unique shape, majority shape, unique size, holed, solid, border-touching, enclosed, square, rectangular, symmetric, multi-coloured, all |
| --- | --- |
| **actions** | delete, keep only, crop to, crop as patch, recolour, fill holes, fill box, outline, slide until blocked (4), snap to edge (4) |

Every operator is an ordinary grid-to-grid function, so it drops into the
existing search — and therefore into *every* family built on that search: the
depth-4 enumerator, the shallow `compose` pass, the forward `cascade` and the
backward `refine`. The composition comes free; only the vocabulary is new.

They are installed as **seed operators**, applied once to the raw input where
the frontier is a single grid, after which the cheap pixel library composes on
top. Segmentation is memoised per grid so three hundred operators share one
connected-components pass; a full pass over a task's grids costs ~23 ms.

### 2.2 Rows and columns become objects

`engine/objects.py` (extended)

Two segmentations were added, `rows` and `cols`: every row (or column) of the
grid as one object, background included. A band of the grid is a thing rules
talk about — "rows that are all one colour become red" — and no connectivity
segmentation can name it. Every object-level solver inherits them.

### 2.3 Per-object process induction

`engine/solvers/objproc.py` (new)

`objects_map` learns "which colour does each object become". That is one action
family out of many. `objproc` keeps its discipline and widens the space:

1. segment under several (segmentation, background) readings;
2. for every object, **propose** actions and keep only those whose local effect
   is consistent with the demonstrated output;
3. find a low-capacity decision function from object features to action that is
   consistent with every object of every training pair *simultaneously*;
4. apply the whole labelling and require an exact match on all pairs.

The action algebra: keep, erase, recolour, bounding-box paints, halo, hole fill,
translate, slide-until-blocked, snap-to-edge, in-place dihedral, ray casting,
**duplicate at an offset**, **mirrored duplicate**, **move toward an anchor**
(slide-to-contact / one step / land inside), and learned **stencils**.

Three details carry most of the weight:

* **Relational colours.** A rule like "paint the shape the colour of the lone
  marker" cannot use a literal, because the literal changes between examples.
  Colours can be named by where they come from: the largest object, the
  uniquely-coloured one, the *nearest object of a different colour*.

* **Alternatives per group.** Local consistency cannot tell "this object was
  deleted" from "this object moved away" — both leave background behind, and
  deletion is cheaper, so it wins the group and the assembled rule then fails
  globally. The cheapest assembly is tried first, then single substitutions
  from each group's shortlist.

* **A task-level prior.** When no demonstration ever removes anything, an action
  that removes something is not a cheap explanation but a wrong one that looks
  cheap locally, and is priced accordingly.

Capacity guards: a table with nearly as many rows as training objects is
rejected; a stencil that was only ever demonstrated once is rejected.

### 2.4 Compositional object programs

`engine/solvers/objchain.py` (new)

Object operators as *seeds* are applied once. This is a dedicated search over
chains of them, to depth 3, with observational-equivalence pruning and a fitted
colour map available as a final step. It finds programs like

```
cmap( recolour-largest-5( delete-squares( outline-majority-shape( input ) ) ) )
```

which is four object-level steps and unreachable anywhere else in the engine.
It is a separate module rather than a wider setting on the main enumerator
because the two vocabularies have opposite economics: the object vocabulary is
large but its branching collapses immediately (most selectors pick the same
objects, so most operators are observationally identical after one level),
while the pixel library is the reverse. Mixing them at every depth multiplies
both branching factors for no gain.

### 2.5 Composition in the other direction

`engine/solvers/refine.py` (new)

`cascade` composes forwards: a grid operator prepares the input, a specialist
explains the rest. The reverse — `output = correction(specialist(input))` — was
not covered anywhere, and it is a large family: a specialist that gets the
structure right and the finish wrong. Specialists are asked for candidates, the
ones that already fit are ignored, and the **near misses** become starting
states for a short search over corrections.

### 2.6 Rules that branch on the input

`engine/solvers/conditional.py` (new)

Every other hypothesis is a single total function, so tasks whose
demonstrations genuinely do different things are lost with an empty hypothesis
set. A predicate partitions the pairs; each group is explained by an ordinary
hypothesis refitted on that group alone.

A branch is a licence to memorise, so the guards are severe: at least four
training pairs; **every** group must carry at least two; a branch of three or
more pairs must survive its own leave-one-out refit; the branch rules must
actually differ; and a task that some single hypothesis already explains is
left alone. On the development split these guards take the family from 89
"fitting" tasks (nearly all of them memorisation) down to 2.

### 2.7 Panel stacks combined by a learned table

`engine/solvers/paneltable.py` (new)

`partition` folds panels together through a closed list of boolean operators
with "on"/"off" colours from a short list. Instead of naming the operator, this
learns it: for each cell position the stack of panels supplies a key and the
output supplies the value. One conflict-free table subsumes all eight boolean
ops, priority overlays and arbitrary colour combinations, at any number of
panels. Keys are offered at five levels of invariance (ordered, sorted, set,
count, count-plus-colour).

A second family one level up **classifies** each panel and writes a learned
block for it, laid out the way the panels were, transposed, or strung into a
row or column.

### 2.8 Continuing a pattern

`engine/solvers/extend.py` (new)

*Periodic completion*: part of the grid carries a repeating pattern and the rest
is empty. The period is fitted against the content's bounding box with
background taken at face value — treating background as *unknown* everywhere is
wrong here, because the empty cells inside the drawn pattern are part of it —
and the fill applies only beyond that box. The period is inferred **per grid**,
because in real tasks it changes between examples.

*Reflective extension*: a grid continued past its own edge by bouncing rather
than wrapping (`a b c b a b c`). Plain tiling gets this wrong at every seam.

### 2.9 A general output-size law

`engine/task.py` (extended)

The engine knew two shape laws, "same size" and "an integer multiple". Grids
that grow by a fixed border, shrink by one, or repeat a reflected pattern *k*
times are affine in the input size and had no law at all; `Ctx.affine_shape`
fits `out = a·in + b` with small integer coefficients.

### 2.10 Locating the cut

`engine/solvers/locate.py` (new)

When the answer is a piece of the input, the problem factorises: find every
place the demonstrated output actually occurs inside its input — cheap, and it
completely determines what a correct rule must return — then ask which of a
library of locators picks exactly those places in every pair at once. Locators
name a rectangle by a colour's extent, an object's box or interior under several
segmentations and several readings of "chosen", or the fixed-size window that
maximises or minimises density / colour count / uniformity. Each dihedral image
of the output is searched for as well, so the cut may be followed by a
transform.

### 2.11 Assembly, self-stamping, tallies

* `engine/solvers/assemble.py` — outputs built *from the objects* rather than
  edited in the grid: overlaid, stacked in a sorted order, summarised one cell
  per object, or selected by ordinal.
* `engine/solvers/selfstamp.py` — the grid as its own brush. `geometry`'s
  self-stamping gates only on "is not background"; the gate is widened to any
  colour and to relational colours (the most common, the rarest, the most common
  non-background), which is what tasks need when the gating colour changes
  between examples. Plus offset repetition at a step that is *not* the grid size.
* `engine/solvers/tally.py` — a count expressed as *how much of a fixed canvas
  is filled*, which nothing that reasons from output dimensions can see; and
  scattered marks gathered in reading order, including serpentine order.

---

## 3. The planner: a learned model in the solving loop

`engine/planner.py` (new), `bench/train_planner.py` (new),
`engine/portfolio.py` (rewritten scheduling)

### What it is

A sparse multinomial log-linear model over an action vocabulary of 34 hypothesis
families plus two control actions (`deepen`, `stop`):

```
P(a | s) = softmax_a( w_a · φ(s) )
```

`φ(s)` is the **reasoning state**, not just a description of the task:

* *task descriptors* — shape law, palette change, separator lines, symmetry,
  object counts, grid size;
* *search history* — which families have already run and come back empty, how
  many hypotheses currently fit, how much budget is left, which step this is;
* *retrieved memory* — the families that solved the most similar previously
  solved tasks, weighted by signature overlap and bucketed.

The retrieval block is a **feature block, not a shortcut**. Nothing in the
planner can answer a task by looking one up; it stores families and signatures,
never grids and never answers. Retrieval only shifts the logits, and the model
is free to learn to distrust it:

```
logits = W·φ_task(s) + W·φ_history(s) + W·φ_memory(s)
P      = softmax(logits)
```

### How it is trained

Supervision comes from the engine's own successes, never from the answer key.
For each development task the **last training pair is hidden**; every family is
asked, independently, to explain the remaining demonstrations; a family that
then predicts the hidden pair correctly is the action the planner should have
chosen. Test outputs are not read at any point in `bench/train_planner.py`, and
the held-out split is not read at all.

From each (task, winning family) pair the state/action trajectory the planner
will actually meet is manufactured — `S0` with nothing tried, then states where
some other families have already come back empty — so the model learns not only
"which family suits this kind of task" but "which family suits it *given that
these have failed*".

### How it changes what the solver does

`portfolio.solve` no longer walks a fixed list. While budget remains it asks
the planner for `P(action | state)`, runs the highest-probability family that
has not run yet, gives it a slice proportional to its probability, then
**recomputes the distribution against the updated state** — which families have
now failed, how many hypotheses fit, how much time is left. Every family still
runs; the planner moves time between them rather than silencing any, so a
mistrained planner costs ordering, not coverage. The trace is recorded in
`res.diagnostics["plan"]`.

The planner proposes. It never decides whether a hypothesis is correct: every
program is still executed against every training pair and discarded unless it
reproduces all of them.

---

## 4. Results

Both runs: the 550 public task files in `data/arc/`, 20-second per-task budget,
11 workers, identical scoring (a task counts only when **every** test pair is
exact). `evidence/BASE_full.json` and `evidence/FINAL_full.json` carry the
per-task records; `bench/report.py` seals each run with a SHA-256 over the task
inputs scored and the per-task outcomes.

| corpus | before | after | change |
|---|---:|---:|---:|
| **ARC-AGI-1** (400) | 175 — **43.75%** | 197 — **49.25%** | **+22 tasks, +5.50 pts** |
| **ARC-AGI-2** (150) | 45 — **30.00%** | 50 — **33.33%** | **+5 tasks, +3.33 pts** |
| both (550) | 220 — 40.00% | 247 — 44.91% | +27 tasks, +4.91 pts |
| ARC-AGI-1, two attempts | 186 — 46.50% | 204 — 51.00% | +18 tasks |
| ARC-AGI-2, two attempts | 47 — 31.33% | 51 — 34.00% | +4 tasks |

**Newly solved: 28. Regressed: 1** (`arc1_6ecd11f4`, and it comes back when the
budget is raised — see below).

### The number to trust

The corpus was split by a hash of the task id into a development split (236
tasks) and a held-out split (314). Every diagnostic, every decision about which
family to write, and the planner's training data came from the development
split only. The held-out split was never read during development.

| split | before | after | change |
|---|---:|---:|---:|
| development (236) — *architecture was shaped against these* | 86 — 36.4% | 107 — 45.3% | +21 |
| **held-out (314)** — *never used* | 134 — 42.7% | **140 — 44.6%** | **+6** |
| held-out, ARC-AGI-1 files (234) | 111 — 47.4% | 115 — 49.1% | +4 |
| held-out, ARC-AGI-2 files (80) | 23 — 28.8% | 25 — 31.3% | +2 |

The gap is the honest finding of this pass: **+21 on tasks whose failures were
inspected, +6 on tasks that were not.** The families are general in form -- none
of them tests for a task -- but *which* families were worth writing was chosen
by looking at development failures, and ARC's tail is diverse enough that this
transfers at roughly a third of the rate. The headline +5.5 / +3.3 is real and
reproducible on this corpus; **+1.9 points is the better estimate of what this
pass would be worth on tasks nobody looked at.**

### Diagnostics

| | before | after |
|---|---:|---:|
| failures with **no fitted hypothesis at all** | 241 | 221 |
| failures where something fit but nothing was right | 89 | 82 |
| solver exceptions | 0 | 0 |

### Runtime

| | before | after |
|---|---:|---:|
| mean per task | 13.4 s | 17.2 s |
| median | 14.9 s | 16.8 s |
| 95th percentile | 19.6 s | 20.0 s |
| total CPU (550 tasks) | 7 378 s | 9 448 s |
| wall clock at 11 workers | 746 s | 868 s |

+28% CPU for +12% solved. The engine now sits against its budget where it
previously had slack: raising the held-out budget from 20 s to 45 s adds two
more tasks (138 → 140 in an earlier revision) and recovers regressions caused by
contention, which was **not** true of the starting engine, where tripling the
budget bought a single task. Search is no longer free.

### What wins

Winning program prefixes in the final run show the new families carrying real
weight rather than decorating the portfolio: `paneltable` 18 tasks, `objproc`
13, `selfstamp` 9, `pdict` 4, `locate` 3.

---

## 5. Ablations

Held-out split (314 tasks), 20-second budget, one component removed per
run, everything else identical. The development split is not used here: a
component's contribution measured on the tasks that motivated it would be
flattered by the same selection that put it in.

| configuration | solved | rate | change | CPU |
|---|---:|---:|---:|---:|
| **full engine** | 140 / 314 | 44.59% | — | 5380 s |
| without object process induction (`objproc`, `objchain`) | 135 / 314 | 42.99% | -5 | 5251 s |
| without leave-one-out counterfactual pruning | 138 / 314 | 43.95% | -2 | 5558 s |
| without the learned planner (fixed schedule) | 139 / 314 | 44.27% | -1 | 4855 s |
| without object operators in the enumerator | 139 / 314 | 44.27% | -1 | 5223 s |
| without deep composition (`refine`, `cascade`, `conditional`, `objchain`) | 139 / 314 | 44.27% | -1 | 4954 s |
| without the new structure families (panel tables, self-stamping, continuation, tallies, locators, assembly) | 140 / 314 | 44.59% | +0 | 5393 s |

Read plainly, on tasks nobody looked at:

* **Object process induction is the one component that clearly pays** (+5).
  It is also the most general thing added — an algebra over object actions with
  a low-capacity learner on top, rather than a rule for a shape of task.
* Leave-one-out pruning is worth +2, the planner, the object operators and deep
  composition +1 each.
* **The new structure families are worth nothing here** (+0), despite winning 18
  tasks outright in the headline run as the cheapest fitting explanation. On
  held-out tasks they are *redundant* rather than useless: remove them and older
  families recover the same tasks. Their measured value came from the
  development split, which is exactly the kind of gain a held-out split exists
  to expose.

The honest reading of the whole table is that a +5.5 point headline decomposes
into one component that generalises and a tail of components that mostly
re-explain what the portfolio could already reach.

---

## 6. What is still failing, in order

`bench/diagnose.py` over all 550 files with the finished engine. Percentages are
of the 306 tasks it does not solve.

| remaining failure mode | tasks | share of failures |
|---|---:|---:|
| **Nothing fit the demonstrations at all** — no hypothesis in the whole portfolio reproduced the training pairs | 218 | 71% |
| **Something fit, but the right rule was never generated** — a wrong-but-consistent rule won by default | 75 | 25% |
| Right answer produced, but ranked second | 10 | 3% |
| Right answer produced, ranked below second | 3 | 1% |

Ninety-six per cent of what remains is a **generation** problem, and the split
is essentially unchanged from where this pass started — the engine solves more
tasks, but the shape of its failure is the same. Nothing here will be fixed by
better scoring, a bigger beam or a longer budget.

By the structure of the task, ordered by failure rate (tags computed from the
demonstrations only, so they describe what is being lost rather than why):

| structure | tasks | failure rate |
|---|---:|---:|
| same-size in and out (in-place editing) | 367 | 64% |
| grids with separator lines / lattices | 233 | 62% |
| the same cells rearranged (object motion) | 38 | 45% |
| the answer is a piece of the input | 122 | 41% |
| output introduces colours not in the input | 179 | 50% |
| only two demonstrations | 83 | 55% |
| output is an integer upscale | 32 | 25% |
| output is tiny (<= 9 cells) | 60 | 23% |

The concentration is in **in-place editing on lattice-structured grids with few
demonstrations** — which is also where ARC-AGI-2 lives, and where a rule is
most likely to be a multi-stage interaction between objects rather than one
transformation that any single family can name.

---

## 7. Honest limitations

* **The planner is worth little on this engine.** It was built because the
  architecture asked for it, and it does measurably beat chance at predicting
  which family will generalise — but the diagnosis above explains why that
  cannot translate into many tasks: reallocating a budget that is not the
  binding constraint reallocates slack. Its measured contribution is in the
  ablation table.

* **`conditional` fires almost never.** Its guards are set where a branch is
  trustworthy rather than where it is productive. Loosening them "solves" many
  more tasks in the sense of fitting them, and generalises worse.

* **Compute is spent to little effect.** The engine consumes most of its budget
  on most tasks and would not solve more with three times as much. The
  remaining failures need vocabulary, not seconds.

* **ARC-AGI-3 is not supported.** The reasoning engine is separated from static
  grid execution far enough that the planner and hypothesis machinery could be
  reused in an observe–act loop, but no such loop exists here and none is
  claimed.

* **The corpora overlap.** 57 of the 150 ARC-AGI-2 files share an identifier
  with an ARC-AGI-1 file, because the ARC-AGI-2 training set carries legacy
  tasks forward. Both corpora are scored exactly as the harness defines them.

---

## 8. Reproducing every number

Run from this directory (`astron/xyz`). Python 3.11+ and no third-party
packages; `--jobs` is worker processes.

```sh
# 0. the engine's own unit tests (150 of them, including the no-leak boundary)
python -m unittest discover -s tests

# 1. the deterministic development / held-out split (pure function of task id)
python bench/split.py --write

# 2. baseline: the engine as supplied, no planner
python bench/run_arc.py --budget 20 --jobs 11 --out evidence/BASE_full.json --quiet

# 3. improved engine, planner active
python bench/run_arc.py --budget 20 --jobs 11     --planner policy/arc_planner.json --out evidence/FINAL_full.json --quiet

# 4. scored side by side, by corpus and by split, each run sealed
python bench/report.py evidence/BASE_full.json evidence/FINAL_full.json     --out evidence/BEFORE_FINAL.json
```

Diagnostics and experiments:

```sh
# failure-mode census (re-solves collecting every prediction, not just the top)
python bench/diagnose.py --tasks data/split_dev --budget 20 --jobs 11     --out evidence/DIAG_dev.json

# is the synthesiser limited by time, depth or beam width?
python bench/beam_probe.py --run evidence/FINAL_full.json --limit 40 --jobs 11

# does more budget help?  (compare against the same tasks at --budget 20)
python bench/run_arc.py --budget 60 --jobs 11 --tasks data/split_dev     --out evidence/DEV_b60.json --quiet
```

Retraining the planner (development split only; reads no test outputs):

```sh
python bench/train_planner.py --tasks data/split_dev --budget 0.5 --jobs 11
```

Ablations (held-out split, one component removed per run):

```sh
python bench/ablate.py --tasks data/split_hold --budget 20 --jobs 11
```

Timings vary with the machine. The solved set does not, except for tasks
sitting against their per-task deadline — and there are more of those than
there used to be, because the engine now uses its budget.

### Files changed

New: `engine/objops.py`, `engine/planner.py`, `engine/solvers/objproc.py`,
`objchain.py`, `refine.py`, `conditional.py`, `paneltable.py`, `selfstamp.py`,
`extend.py`, `tally.py`, `locate.py`, `assemble.py`; `bench/split.py`,
`diagnose.py`, `report.py`, `beam_probe.py`, `train_planner.py`, `ablate.py`;
`policy/arc_planner.json`; this document.

Modified: `engine/portfolio.py` (planner-driven scheduling, module registry),
`engine/enum_core.py` (object seeds), `engine/objects.py` (row/column
segmentations), `engine/task.py` (affine shape law), `engine/solvers/`
`objects_map.py`, `cascade.py`, `objwise.py`, `panelwise.py`, `rewrite.py`
(module lists, band-segmentation pricing), `bench/run_arc.py` (planner and
ablation flags, backward-compatible job tuple).
