# GABRIEL

**GABRIEL** is the system: the ASTRA symbolic ARC engine with a language model
strapped to the places where it has to make a choice.  ASTRA is the engine
inside it -- the package names, imports and benchmark scripts keep that name so
nothing breaks; GABRIEL is the thing as a whole.

The model is trained here, from scratch, in pure Python, on programs this
engine wrote.  **No external model is consulted at any point.**  There is no API
key, no network call, no vendor SDK, and no dependency outside the standard
library.  That is not a stylistic choice: an ARC result produced with help from
an outside model is not an ARC result.

```bash
python3 -m gabriel data/arc/arc1_007bbfb7.json --budget 20 --out preds.json
python3 -m gabriel.train --max-seconds 120          # one gated training pass
python3 -m gabriel.bench --mode both --limit 32     # paired A/B against stock
```

## The model

A sparse log-linear (maximum-entropy) autoregressive language model over the
engine's own program language.

The engine names every hypothesis it builds, and those names are a small
regular language: `crop(rot90($))`, `T@fractal#0`, `sep#5.and->2`,
`paint#1($)>>c4@0.by_size_rank`.  `gabriel/tokens.py` scans that into tokens.
A training sequence is a task *prompt* followed by a program:

```
prompt:  <sig>shape:diff  <sig>size:up  <sig>ratio:3x3 ...
program: <fam>geometry  fractal  #  0  <eos>
```

The signatures come from `engine.learn.signatures`, computed from training
pairs only.  The solver family is emitted as the first token, which is why
`P(family | task)` is something the model produces directly rather than a
second head bolted on the side.

```
P(t | h) = softmax_t  sum_{f in F(h)} w[f][t]

F(h) = { b,  1|prev,  2|prev2|prev,  s|sig (one per signature),
         f|family,  p|position }
```

Trained by SGD with AdaGrad on cross-entropy, with a sampled softmax
(negatives from the unigram raised to 0.75) so a pass fits inside a cron tick,
and early-stopped on a held-out slice.  Perplexity is always reported with the
**exact** softmax, never the sampled approximation.

Measured on this checkout:

| | |
| --- | --- |
| corpus | 921 training / 201 dev examples, from 316 fit-split tasks |
| vocabulary | 282 tokens |
| parameters | 960 feature rows, ~118k non-zero weights (2.5 MB of JSON) |
| dev perplexity | **12.9** |
| unigram baseline, same dev, same vocabulary | 52.0 |
| uniform baseline | 282.0 |
| training time | ~7 s for a full pass on 4 vCPU |

## Where it is strapped in

All three points use extension points `engine/learn.py` already defines.  **No
file under `engine/`, `bench/`, or `engine/solvers/` was modified.**

**1. Which family to believe.**  `gabriel/bind.py` subclasses `learn.Policy`;
`bias_for(sigs)` adds `-w log(P(family | task) * n)` to the fitted prior, in the
same units and with the same `[-3, 3]` clamp the fitted policy uses.  The
portfolio ranks and orders solver modules with it.

**2. Which operator to try next.**  `enum_core.search` subtracts an operator
bias when it sorts its library and adds it to each node's beam bonus -- in a
search truncated by a state cap, what is explored first is what is explored at
all.  `GabrielPolicy.op_bias` is a property: `portfolio.solve` calls
`bias_for(sigs)` and then reads `op_bias`, in that order, so the property
answers for *that* task.  Values are squashed into `[0, 1.2]`, measured against
the median operator, so the model can promote what it believes in and can never
silence anything.  If that call order ever changed, the property falls back to
the fitted static bias -- degraded, never wrong.

**3. Which programs to write.**  `gabriel/proposer.py` is an ordinary solver
module.  It decodes operator chains from the model under a grammar built from
`enum_core.unary_ops(ctx)` -- so it can only propose programs that exist for
this task, learned abstractions included -- and hands them to the portfolio as
hypotheses.  The portfolio then does what it always does: runs each one on
every training pair and discards it unless every pair matches.  The model
proposes; the verifier disposes.  A confident model gets no benefit of the
doubt.

## What it measures

Paired A/B, 32 fit-split tasks, `--budget 8 --jobs 4`, scored by
`bench/run_arc.py`'s own `score_predictions` (every test pair exact, top-1):

| | solved | CPU seconds |
| --- | --- | --- |
| stock engine | 16 / 32 | 204.4 |
| GABRIEL bound | 16 / 32 | 167.3 |
| paired | 0 wins, 0 losses, sign test p = 1.0 | **-18%** |

Read that honestly: on this sample **binding the model is accuracy-neutral and
about 18% cheaper in CPU**.  It solves the same tasks sooner, because the
family prior gets the right module to fire earlier -- it does not yet solve
tasks the engine could not.  The chain corpus is the reason: only about 1,350
of the ~21,000 program tokens in `evidence/` are enumerator chains, which is
thin data for the proposer.  `gabriel/harvest.py` exists to grow exactly that,
one cron tick at a time.

Nothing here changes the numbers in `RESULTS.md`.  Those are the stock engine
on the bundled public files: ARC-1 168/400, ARC-2 41/150, 209/550 top-1.  They
are not official and not a hidden-set result.

## Discipline

The reason to trust a number from this loop is that adoption is gated the same
way in both halves:

* a candidate **policy** is adopted only if a one-sided sign test over
  discordant tasks says it beat the incumbent (`bench/evolve.py`, untouched);
* a candidate **language model** is adopted only if its perplexity fell on a
  held-out dev slice it was never trained on (`gabriel/train.py`), and within
  training, only the best-generalising epoch's weights are kept.  The two
  models are compared over a common support, but each keeps its own idea of
  which tokens it knows -- scoring an incumbent's unseen token as a bare zero
  would rank it above every token that incumbent had learned to distrust, and
  every candidate would look like an improvement.

Two further rules keep the corpus clean:

* only fit-split tasks are ever read or harvested.  The evolve holdout is
  never trained on and never gates anything, here either;
* nothing reads a record's `solved` flag.  A program enters the corpus because
  it reproduced every *training* pair -- whether it then turned out to be right
  on the task's test pair changes nothing.  The model learns from information a
  solver is allowed to have.

`tests/test_gabriel.py` covers all of it -- 26 tests, including that decoding
cannot leave the operator set, that a proposed program still has to fit, that
the holdout never enters the corpus, and that the solve context still has no
test outputs on it.

```bash
python3 -m unittest tests.test_gabriel -v
```

## Files

```
gabriel/tokens.py     the token language over program names
gabriel/lm.py         the model: features, training, perplexity, decoding
gabriel/corpus.py     evidence/*.json -> training examples (fit split only)
gabriel/train.py      one gated training pass
gabriel/bind.py       GabrielPolicy: the model as an engine policy
gabriel/proposer.py   the model as a solver module
gabriel/harvest.py    grow the corpus, one rotating slice per tick
gabriel/bench.py      paired measurement, using run_arc's own scoring
gabriel/__main__.py   inference CLI
```

Operations -- the VPS, the cron, the Namecheap terms -- are in
[README.OPS.md](README.OPS.md).

## The text sibling

`text/` is a separate package that applies this architecture -- signatures, a
conditional log-linear scorer, competing hypothesis families, constrained
decoding, verification before acceptance, a held-out gate -- to natural-language
queries instead of ARC programs.  It shares no vocabulary with GABRIEL and
imports nothing from `gabriel/` or `engine/`; an ARC program vocabulary cannot
score English and pretending it can would be a fake integration.  It exports two
small JSON artifacts that `robots.html` loads same-origin, which is the whole of
the connection -- ASTRON runs no daemon, so there is no API to call.

Nothing in this file changed to accommodate it.  See [text/README.md](text/README.md).
