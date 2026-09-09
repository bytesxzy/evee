"""The model writes programs; the engine decides whether they are true.

This is an ordinary solver module -- ``SOLVER``, ``PHASE``, ``generate(ctx)``,
yielding :class:`engine.task.Hyp` -- so the portfolio treats what the language
model writes exactly as it treats what a hand-written solver writes: it runs
the program on every training pair and throws it away unless every one matches.
A confident model gets no benefit of the doubt.

What the model contributes that the enumerator cannot: direction.  The
enumerator is breadth-first under a state cap, so its reach is bounded by
branching factor -- depth four over a large library is already most of its
budget.  A beam decoded from the model is narrow and deep, so chains of six
operators are reachable when the model has a strong opinion about which six.
The two are complementary, which is why this module is registered alongside the
enumerator rather than in place of it.

Every operator offered to the decoder comes from ``enum_core.unary_ops(ctx)``
-- including abstractions the policy has installed -- so the model can only
propose programs that exist for this task, and can compose learned
abstractions it has never seen used together.
"""

import time

from engine import enum_core, learn
from engine.task import Hyp

from . import tokens as T

SOLVER = "gabriel"
PHASE = 2

LM = None                 # set by gabriel.bind.bind()
MAX_LEN = 6               # deeper than the enumerator reaches unaided
BEAM = 10
LIMIT = 40
DECODE_SHARE = 0.25       # fraction of the module's slice spent decoding
DECODE_CAP = 2.0          # seconds; decoding must never eat the search
LM_WEIGHT = 0.5           # how much surprisal costs, in engine cost units


def _chain_fn(fns):
    def run(g):
        for f in fns:
            g = f(g)
            if g is None:
                return None
        return g
    return run


def generate(ctx):
    lm = LM
    if lm is None or not getattr(lm, "is_trained", lambda: False)():
        return []
    now = time.time()
    budget = DECODE_CAP
    if ctx.deadline is not None:
        budget = min(DECODE_CAP, max(0.05, (ctx.deadline - now) * DECODE_SHARE))
    ops = {}
    for name, cost, fn in enum_core.unary_ops(ctx, "full"):
        if name not in ops:
            ops[name] = (cost, fn)
    if not ops:
        return []
    try:
        sigs = learn.signatures(ctx)
    except Exception:
        sigs = ()
    chains = lm.beam_chains(sigs, list(ops), max_len=MAX_LEN, beam=BEAM,
                            limit=LIMIT, deadline=now + budget)
    out = []
    for chain, logprob in chains:
        if ctx.timed_out():
            break
        try:
            fns = [ops[o][1] for o in chain]
            cost = sum(ops[o][0] for o in chain)
        except KeyError:
            continue
        surprisal = -logprob / float(len(chain))
        out.append(Hyp(T.chain_name(chain), _chain_fn(fns),
                       1.5 + cost + LM_WEIGHT * surprisal, SOLVER))
    return out
