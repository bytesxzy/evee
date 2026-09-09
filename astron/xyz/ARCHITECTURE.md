# ASTRA — architecture

> This is the original design document, retained for context. The improved
> edition changes synthesis, ranking, movement reasoning, and validation as
> described in `IMPROVEMENTS.md`. In particular, observational pruning now
> retains a cost/depth Pareto frontier; leave-one-out credit follows a concrete
> rule and its predictions; compatible shape laws remain alternatives. Any
> numerical performance claims below refer to the supplied revision.

ASTRA is an ARC-AGI reasoning engine. It reads a task's demonstration pairs,
searches a space of *explanations*, ranks them by how much they compress the
evidence, and applies the winner to the held-out input. It contains no neural
network and calls no external model. Every number in `RESULTS.md` was produced
by the code in this directory, on this machine.

This document is about *why* the engine is shaped this way. The short version:
ARC is not one problem, and a single search space large enough to contain every
ARC rule is too large to search. The engine's answer is a portfolio of narrow,
independently-derived hypothesis generators, unified by one honest ranking rule
and one general synthesiser that catches what the specialists miss — plus a
learning layer that reallocates search effort and grows the DSL from the tasks
it has already solved.

---

## 1. The central problem: a search space you cannot afford to enumerate

A DSL rich enough to express ARC solutions is combinatorially explosive. If the
library has `k` operators and solutions are `d` operators deep, naive
enumeration is `k^d`. With `k ≈ 90` and `d = 4` that is 65 million programs, and
`d = 4` is *not* deep enough for most of ARC.

There are only three ways out, and the engine uses all three.

**(a) Collapse behaviourally identical programs.** The enumerator
(`engine/enum_core.py`) never compares programs; it compares the *tuple of grids
a program produces on the task's own inputs*. Two syntactically different
programs with the same behaviour on the evidence are the same hypothesis, and
only one survives. `rot180` and `flip_h ∘ flip_v` are one node, not two. On real
tasks this prunes each level by roughly an order of magnitude, and it is what
makes depth 4 tractable in pure Python.

**(b) Replace search with fitting wherever a family is closed-form.** If a task
is "recolour every object according to some property of that object", you do not
have to *search* for the recolouring — you can read it off the training pairs
and check it for consistency. Most of the portfolio works this way: it fits, it
does not enumerate. `objects_map`, `blocks`, `substitute`, `cellwise` and
`symmetry` all solve a fitting problem in time linear in the data, over a space
that enumeration would need exponential time to cover.

**(c) Let the space grow only where it has paid off before.** The learning layer
(§5) mines abstractions from solved programs and installs them as single
operators, so effective depth grows with experience instead of with the budget.

## 2. Representation: the object vocabulary is the real bottleneck

Whatever a solver can express is bounded by how the grid was carved up. ARC's
notion of "object" is not fixed — sometimes it is a 4-connected monochrome blob,
sometimes an 8-connected multicoloured assembly, sometimes every cell of one
colour wherever it appears, sometimes a *dashed* structure whose parts are not
adjacent at all. Committing to one segmentation loses tasks that need another,
so `engine/objects.py` exposes seven and the portfolio tries each:

| name | what it means |
|---|---|
| `c4`, `c8` | connected, single colour, 4- or 8-neighbourhood |
| `m4`, `m8` | connected, any colours (multi-coloured assemblies) |
| `color` | one object per colour, connected or not |
| `g2`, `g2m` | connected up to a *gap* of 2 cells — dashed lines, scattered marks |
| `cells` | every cell its own object |

A second, subtler choice: **what counts as background**. Segmenting against the
task background finds the objects; segmenting against a *wall* colour instead
finds the regions those walls enclose. "Colour each pen by how big it is" is a
region task, not an object task, and it is invisible unless you are willing to
call the walls background. `objects_map` therefore tries the top few colours as
background, not just the inferred one.

Grids are immutable tuples of tuples. That is load-bearing: it makes them
hashable, which is what allows observational-equivalence pruning, the
segmentation cache, the feature cache and the window-scan cache to exist at all.
Those caches are not micro-optimisations — before they were added, a single task
recomputed the same connected-components analysis several hundred times and the
engine spent most of its budget on it.

## 3. The portfolio: eighteen ways to be right

Each module in `engine/solvers/` exports one function, `generate(ctx) -> [Hyp]`.
A `Hyp` is a *total function from an input grid to an output grid* plus a
structural cost. Modules do not decide anything and never see a test answer;
they propose.

| module | the family it explains |
|---|---|
| `geometry` | dihedral maps, crops, halves, quadrants, tilings, fractal self-stamping, gravity, hole filling, union with a mirror image |
| `colormap` | recolourings that generalise: by frequency *rank*, by role, by relation — not by literal colour |
| `partition` | separator lines and even splits into panels; select a panel relationally, or combine all panels cellwise (the AND/OR/XOR family) |
| `blocks` | learned dictionaries: cell → patch, patch → cell, panel → panel |
| `symmetry` | occlusion repair by symmetry-group closure (§4) |
| `tiling` | periodic motif extraction, periodic and *diagonal* fill, extension |
| `select` / `regions` | the answer is one object or one region: window scanning, frames, marked rectangles, colour answers |
| `cellwise` | local-rule induction over 24 context families (§4) |
| `objects_map` | per-object edits as a decision function over object features |
| `substitute` | shape → stencil dictionaries ("each seed grows into this motif") |
| `sequence` / `paint` | motion and drawing: gravity with collision, rays, connections, line extension, rectangle completion, reflection across a separator, symmetrisation |
| `analogy` | one complete exemplar in the grid, fragments elsewhere; complete them |
| `compose` | shallow enumeration (depth 2) so short programs are found early and ranked cheaply |
| `rewrite` | re-pose the task (crop it, compress it, downscale the outputs) and hand it back to the specialists |
| `cascade` | two-stage solving (§6) |
| `enumerate` | depth-4 bottom-up synthesis with a binary combination layer — the general fallback |

The portfolio is not a pile of special cases; it is a set of *different
inductive biases*. Their independence is what makes the ensemble vote in §7
meaningful.

## 4. Two mechanisms worth explaining in full

### Symmetry repair by group closure

The task: a structured grid has a region blanked out, and the answer is the
repaired grid (or just the missing patch).

The naive approach — test the eight dihedral symmetries — fails immediately,
because real ARC symmetries are about *arbitrary axes*, not the grid centre, and
are often translational or diagonal rather than reflective.

So `engine/solvers/symmetry.py` collects a large candidate set of cell
permutations: mirrors about every half-integer horizontal and vertical axis,
translations by every row and column period, translations along both diagonals,
and (on square grids) the two diagonal reflections and a quarter turn. A
permutation is kept only if it never contradicts an *observed* pair of cells and
is supported by enough overlap to be more than a coincidence.

The surviving permutations are then fed to a union-find over cells. This is the
key step: union-find closes under composition, so unioning under a set of
generators recovers the *entire group they generate* without ever enumerating
it. If the grid is invariant under a horizontal mirror and a vertical
translation, the engine gets every glide reflection for free. Each orbit that
contains at least one observed cell is then determined, and any orbit that is
entirely unobserved makes the strict variant decline to answer rather than
guess.

Adding diagonal translations to that generator set was worth real tasks on its
own: a grid whose colour depends on `(r+c) mod k` has no row period and no
column period, so without them it is not merely hard to repair — it is
invisible.

### Local-rule induction, and why it needs a capacity guard

For same-shape tasks, `cellwise` tries to explain each output cell as a function
of a bounded local context of the input: the cell's colour; its 4- or
8-neighbourhood, ordered or as a multiset; a bitmask of which neighbours match
it; counts of non-background neighbours; row/column parity and modular position;
distance to the border; the nearest non-background colour in each of the four
directions; the size, dimensions or shape of the object the cell belongs to; the
frequency rank of its colour; whether its row or column is uniform. Twenty-four
families in all. Each is a lookup table, fitted in one pass and rejected on the
first contradiction.

This single module covers denoising, outlining, ray drawing, colour swaps,
parity stripes, and a long tail of "each pixel becomes…" rules.

It is also the most dangerous module in the engine, because a rich enough
context *always* fits. A 3×3-neighbourhood table with as many entries as there
are cells has memorised the training data and predicts nothing. Two guards:

1. a hard capacity bound — a table with more than a third as many entries as
   observations is rejected outright;
2. **internal leave-one-out**: refit the table with one training pair withheld
   and check whether it still predicts that pair. A rule that survives is
   discounted; a rule that fails is *penalised*, not merely un-rewarded.

The second guard is the one that matters. Before it existed, this module was
producing the top-ranked hypothesis on dozens of tasks it then got wrong.

## 5. Self-improvement, and what would make it dishonest

The engine improves itself along three axes, all offline, all local. The design
constraint throughout: **a self-improving system that grades its own homework
will drift**, so nothing is adopted without a paired test against the version it
replaces.

**Signature-conditioned priors.** Each task is reduced to a handful of
categorical signatures — does the shape change, and how; does the palette grow
or shrink; is there a separator colour; how large is the grid; how many colours;
are the inputs symmetric. From the record of which family produced the accepted
program, `engine/learn.py` fits a bias per (signature, family) as a log-odds
against a uniform prior. At solve time the biases of the task's own signatures
are summed into the ranking prior. This changes *both* what is believed first
and, through module ordering, where the time goes. The bias is clamped to ±3, so
experience can reorder families but can never silence one.

**Library learning.** Accepted enumerator programs are parsed back into operator
chains; frequent contiguous sub-chains are compiled into single named operators
and installed into the DSL. This is the mechanism that makes the search space
grow with experience rather than stay fixed: a depth-4 search over a library
containing 3-operator abstractions reaches 12-operator programs. The abstraction
is a closure over base operators only, so abstractions cannot nest into each
other uncontrollably.

**Operator bias.** Operators that appear in accepted programs sort earlier in
enumeration. This matters more than it sounds: the enumerator is truncated by a
state cap, so what is explored first is what is explored *at all*.

**The gate.** `bench/evolve.py` runs the loop. Each round: run the fit split
under the policy in force; fit a candidate from accumulated experience; re-run
the fit split under the candidate; compare *per task*. Adoption requires a
one-sided sign test over discordant tasks,

```
p = Σ_{i≥w} C(w+l, i) / 2^(w+l)
```

where `w` is tasks the candidate solves and the incumbent does not and `l` the
reverse. A candidate that merely reshuffles which tasks it solves does not pass.
The holdout split — chosen by MD5 of the task id, so it is reproducible and
independent of solve order — never fits anything and never gates anything; it is
measured once, at the end, to report whether the policy transfers.

This is a real but bounded claim. It is policy learning over a fixed hypothesis
space plus a compression-based extension of that space. It is not open-ended
self-modification, and a single sign test does not control error over repeated
adaptive experiments — which is exactly why the holdout exists and why the
lineage of accepted and *rejected* rounds is recorded in `policy/lineage.jsonl`.

## 6. Two-stage solving: using the near misses

Every solver in the portfolio is all-or-nothing — a hypothesis either reproduces
every training output or it is discarded. That throws away the most informative
signal available: the transform that gets the grid *almost* right.

"Crop to the frame, then recolour by size" is invisible to every single-stage
family and trivial once the crop has been applied. So `cascade` runs a shallow
search scored by **cell agreement** instead of exact match, keeps the transforms
that make the most progress (preferring those that get the *shape* right, since
a second stage can repaint cells but cannot resize the grid), re-poses each as a
fresh task `(stage1(input), output)`, and hands it to the specialists. The
composition is validated end-to-end like anything else, so a first stage that
only looked promising costs nothing when the second stage fails to close the gap.

`rewrite` is the same idea applied in the other direction: rewrite the task
itself — crop it, strip its scaffolding lines, downscale its outputs — and reuse
the entire specialist portfolio on the rewritten version, inverting the rewrite
on the way out.

## 7. Ranking: a description-length argument, then a vote

Fitting the training pairs is necessary and nowhere near sufficient. On a
typical task the portfolio produces several hypotheses that all reproduce the
demonstrations and disagree about the answer. Choosing between them *is* the
problem.

The score is

```
score = family_prior + structural_cost + capacity_penalty − LOO_evidence
```

`structural_cost` is the size of the program or table the hypothesis carries.
`capacity_penalty` grows with the number of free parameters it fitted — a lookup
table pays for every row. `LOO_evidence` refits the *generating family* with one
training pair withheld and asks whether it still predicts that pair; surviving
earns up to −3, failing costs +1.5, and too few pairs to judge costs nothing.
This is Occam with the arithmetic made explicit: the shortest rule that still
explains a pair it never saw is the one to trust.

Then the vote. Independent solver families that arrive at the *same answer* are
much stronger evidence than one family agreeing with itself, so each answer is
scored by `Σ_families exp(−best_score_in_that_family / 2)` — each family
contributes once, at its best score, no matter how many variants it emitted.
This stops a module that generates nine hundred hypotheses from outvoting one
that generates three.

## 8. Measurement, and the guarantees behind it

A solver is constructed from the training pairs and the test *inputs*. Test
outputs never enter the solver's view — they stay in the harness and are
compared afterwards. This is structural, not a convention: `engine/task.py`'s
`Ctx` has no field to put them in.

A task counts as solved only when **every** test pair of that task is exact.
Partial cell agreement is recorded but is diagnostic only and is never scored.
Both top-1 (one attempt) and top-2 (the two attempts official ARC-AGI allows)
are reported. Per-task wall-clock is bounded by the orchestrator's deadline;
solver exceptions are caught and counted as unsolved rather than excluded.

`bench/compare.py` compares against the previous engine per task — the same 550
files, the same identifiers, a win/loss table and a sign test — not two aggregate
rates.

## 9. Where the budget goes

Time is allocated in two phases. The specialists are cheap and either fire or do
not, so they share the first 45% of the budget; whatever they leave unspent
flows to the search modules — `compose`, `rewrite`, `cascade`, `enumerate` —
which can always use more. Within phase 2 each module takes a share of what
actually remains, computed as it starts rather than fixed in advance.

## 10. What this engine cannot do

Stated plainly, because the honest limits are part of the result.

- It has **no notion of physics or agents**. Tasks whose rule is "simulate this
  until it stops" are outside every family here. *Counting is now partially
  covered*: `engine/solvers/counting.py` fits the family where the answer's
  size or content is a function of a count read off the input -- an `n x n`
  block, a bar of length `n`, the input tiled or scaled `n` times. It fits, it
  does not search, and a counter that is constant across the training inputs is
  penalised because it cannot be told apart from a constant. Tasks whose rule is
  "however many steps it takes" remain out of range.
- Its compositional reach is **two stages**, plus whatever depth the enumerator
  and learned abstractions provide. Genuinely three- and four-stage ARC-2 tasks
  are largely out of range, which is most of why the ARC-2 rate is roughly half
  the ARC-1 rate.
- The learning layer improves *allocation and reach* within a hypothesis space
  that a person designed. It does not invent a new kind of explanation. It
  cannot, and the architecture does not pretend otherwise.
- These are public development sets. Public ARC training tasks informed which
  families were worth writing, and the fit split participates in policy
  selection. The holdout number is the transfer estimate; nothing here is an
  official hidden-test result, and it should not be read as one.
