"""Rules that branch on a property of the input.

Every hypothesis elsewhere in the engine is a single total function, and the
portfolio only keeps the ones that reproduce *every* demonstration.  That is the
right discipline, but it makes one whole class of ARC rule unstateable: the ones
where the demonstrations genuinely do different things, and what selects between
them is a property of the input.  "If the marker is red, mirror; if it is blue,
rotate."  "Square grids get tiled, tall grids get stacked."  No primitive fits
all the pairs, so nothing is proposed and the task is lost with an empty
hypothesis set rather than a wrong answer.

The construction here is deliberately conservative, because a branch is a
licence to memorise:

* a predicate partitions the training pairs into groups;
* every group must be explained by a single ordinary hypothesis, refitted on
  that group alone -- so each branch is a real rule, not a stored answer;
* **every** group must carry at least two pairs.  A branch fitted to one
  demonstration is that demonstration, written down; the guard is what stops
  this family from "solving" any task with enough training pairs to cut up;
* where a branch has three or more pairs it must survive its own leave-one-out
  refit -- drop one, refit on the rest, predict the one dropped.  A two-pair
  branch cannot be cross-validated at all, and is admitted only on the strength
  of the other guards;
* a task that some single hypothesis already explains is left alone -- branching
  there would add cost and nothing else;
* the assembled program is then verified on all pairs like anything else, and
  it costs more than any single-branch rule, so it only wins when nothing
  simpler is available.

At test time an input whose predicate value was never demonstrated is a genuine
gap in the evidence.  Two readings are offered -- refuse, or fall back to the
best-supported branch -- and ranked in that order.
"""

import time
from collections import Counter

from .. import grid as G
from .. import objects as O
from .. import hypcache
from ..task import Ctx, Hyp

SOLVER = "compose"
PHASE = 2

_MAX_BRANCHES = 4


def _modules():
    from ..solvers import (blocks, cellwise, colormap, geometry, objects_map,
                           objproc, partition, select, symmetry, tiling)
    return (geometry, colormap, tiling, symmetry, select, partition, blocks,
            objects_map, objproc, cellwise)


# ---------------------------------------------------------------------------
# predicates: input grid -> a hashable, test-computable key
# ---------------------------------------------------------------------------

def _predicates(ctx):
    pal = sorted(ctx.in_palette)
    preds = [
        ("shape", lambda g: G.dims(g)),
        ("h", lambda g: len(g)),
        ("w", lambda g: len(g[0])),
        ("square", lambda g: len(g) == len(g[0])),
        ("tall", lambda g: len(g) > len(g[0])),
        ("ncol", lambda g: len(G.palette(g))),
        ("bg", lambda g: G.background(g)),
        ("top", lambda g: G.most_common_color(g)),
        ("rare", lambda g: G.least_common_color(g)),
        ("symh", lambda g: g == G.flip_h(g)),
        ("symv", lambda g: g == G.flip_v(g)),
        ("sym", lambda g: bool(G.symmetries(g))),
        ("corner", lambda g: g[0][0]),
    ]
    for c in pal[:6]:
        preds.append(("has%d" % c, (lambda c: lambda g: c in G.palette(g))(c)))
        preds.append(("cnt%d" % c, (lambda c: lambda g: G.count_color(g, c))(c)))
    for seg in ("c4", "c8"):
        preds.append(("n_" + seg, (lambda seg: lambda g: min(
            40, len(O.segment(g, seg, G.background(g)))))(seg)))
        preds.append(("np_" + seg, (lambda seg: lambda g: len(
            O.segment(g, seg, G.background(g))) % 2)(seg)))
        preds.append(("big_" + seg, (lambda seg: lambda g: max(
            [o.color for o in O.segment(g, seg, G.background(g))
             if o.size == max(x.size for x in O.segment(
                 g, seg, G.background(g)))] or [-1]))(seg)))
    return preds


def _keys(pred, grids):
    out = []
    for g in grids:
        try:
            k = pred(g)
        except Exception:
            return None
        try:
            hash(k)
        except TypeError:
            return None
        out.append(k)
    return out


# ---------------------------------------------------------------------------

class _Branching:
    __slots__ = ("pred", "table", "default")

    def __init__(self, pred, table, default):
        self.pred = pred
        self.table = table
        self.default = default

    def __call__(self, g):
        try:
            k = self.pred(g)
        except Exception:
            return None
        fn = self.table.get(k, self.default)
        if fn is None:
            return None
        try:
            return fn(g)
        except Exception:
            return None


def _branch_hyps(ctx, pairs, test_inputs, deadline):
    """Ordinary hypotheses refitted on one group of demonstrations."""
    sub = Ctx([(G.to_list(a), G.to_list(b)) for a, b in pairs],
              [G.to_list(t) for t in test_inputs], deadline=deadline)
    out = []
    for mod in _modules():
        if time.time() > deadline:
            break
        try:
            hyps = list(hypcache.generate(mod, sub))
        except Exception:
            continue
        best = None
        for hp in hyps:
            if hp.fits(sub.train) and (best is None or hp.cost < best.cost):
                best = hp
        if best is not None:
            out.append(best)
    out.sort(key=lambda h: h.cost)
    return out[:3]


def _loo_ok(ctx, idxs, deadline):
    """Refit this branch with one pair withheld and check it back.

    A two-pair branch has nothing to withhold -- refitting on a single
    demonstration proves nothing either way -- so it is passed through and
    carries the extra cost instead.
    """
    if len(idxs) < 3:
        return True
    rest = [ctx.train[i] for i in idxs[:-1]]
    held_in, held_out = ctx.train[idxs[-1]]
    hyps = _branch_hyps(ctx, rest, [held_in], deadline)
    return any(h.apply(held_in) == held_out for h in hyps)


def _single_fit_exists(ctx, deadline):
    """Whether an ordinary, unbranched rule already explains the task."""
    for mod in _modules():
        if time.time() > deadline:
            return False
        try:
            hyps = list(hypcache.generate(mod, ctx))
        except Exception:
            continue
        for hp in hyps:
            if hp.fits(ctx.train):
                return True
    return False


def generate(ctx):
    n = len(ctx.train)
    if n < 4:
        return []                     # every branch needs two pairs of its own
    deadline = ctx.deadline or (time.time() + 5.0)
    if time.time() > deadline:
        return []
    inputs = ctx.inputs
    if _single_fit_exists(ctx, min(deadline, time.time() + 1.2)):
        return []
    res, seen = [], set()
    tried = 0
    for name, pred in _predicates(ctx):
        if time.time() > deadline or len(res) >= 4:
            break
        keys = _keys(pred, inputs)
        if keys is None:
            continue
        groups = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        if len(groups) < 2 or len(groups) > min(_MAX_BRANCHES, n // 2):
            continue
        if min(len(v) for v in groups.values()) < 2:
            continue
        sig = tuple(sorted(tuple(v) for v in groups.values()))
        if sig in seen:
            continue                  # another predicate already cut this way
        seen.add(sig)
        tried += 1
        if tried > 5:
            break
        share = max(0.25, (deadline - time.time()) / 4.0)
        table, ok = {}, True
        support = {}
        for k, idxs in groups.items():
            pairs = [ctx.train[i] for i in idxs]
            hyps = _branch_hyps(ctx, pairs, ctx.test_inputs,
                                min(deadline, time.time() + share))
            if not hyps:
                ok = False
                break
            table[k] = hyps[0].fn
            support[k] = (len(idxs), hyps[0].cost, hyps[0].name)
        if not ok:
            continue
        if len({v[2] for v in support.values()}) < 2:
            continue                  # one rule everywhere is not a branch
        if any(not _loo_ok(ctx, groups[k], min(deadline, time.time() + share))
               for k in groups):
            continue
        cost = (4.2 + 0.8 * len(table)
                + sum(v[1] for v in support.values()) / float(len(support)))
        label = "if[%s]{%s}" % (name, ",".join(
            "%s:%s" % (k, support[k][2]) for k in sorted(support, key=repr))[:90])
        modal = max(support, key=lambda k: (support[k][0], -support[k][1]))
        for default, extra, tag in ((None, 0.0, ""),
                                    (table[modal], 0.5, "+fallback")):
            hp = Hyp(label + tag, _Branching(pred, table, default),
                     cost + extra, SOLVER)
            if hp.fits(ctx.train):
                res.append(hp)
                break
    return res
