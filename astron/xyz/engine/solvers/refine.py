"""Composition in the other direction: specialist first, correction after.

``cascade`` composes forwards -- a grid operator prepares the input, then a
specialist explains the rest.  The reverse composition is not covered anywhere,
and it is a large family: a specialist that gets the *structure* right and the
finish wrong.  "Take the odd object out, then rotate it", "combine the panels,
then recolour", "repair the symmetry, then crop to the repaired region" all have
the shape

    output = correction( specialist( input ) )

and none of them is reachable by a single specialist or by a forward cascade.

The mechanics matter for cost.  Specialists are asked for their candidates and
the ones that already fit are ignored -- the portfolio will find those anyway.
What is kept is the *near misses*: candidate transforms that are wrong but
close, measured by cell agreement and shape agreement against the training
outputs.  Each near miss becomes a starting state for a short operator search,
which is a search over corrections rather than over whole programs, so it stays
shallow even when the composed program is not.

States are deduplicated by the tuple of grids they produce, so two specialists
that happen to agree cost one search, not two.
"""

import time

from .. import enum_core
from .. import grid as G
from .. import hypcache
from ..task import Hyp

SOLVER = "compose"
PHASE = 2

_MAX_SEEDS = 34
_MAX_PER_MODULE = 26


def _modules():
    from ..solvers import (assemble, blocks, colormap, extend, geometry,
                           objects_map, objproc, panelabs, paneltable,
                           panelwise, partition, patterns, regions, select,
                           selfstamp, symmetry, tiling)
    return (geometry, select, partition, symmetry, tiling, blocks, regions,
            colormap, objects_map, panelabs, paneltable, panelwise,
            patterns, assemble, selfstamp, extend)


def _agreement(a, b):
    if a is None or G.dims(a) != G.dims(b):
        return 0.0
    n = m = 0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            n += 1
            if x == y:
                m += 1
    return m / float(n) if n else 0.0


def _score(state, target):
    n = len(target)
    agree = sum(_agreement(g, t) for g, t in zip(state, target)) / n
    shape = sum(1.0 for g, t in zip(state, target)
                if g is not None and G.dims(g) == G.dims(t)) / n
    return shape * 1.5 + agree


def _seeds(ctx, deadline):
    """Near-miss states produced by the specialists, best first."""
    n_tr = len(ctx.train)
    grids = ctx.inputs + ctx.test_inputs
    target = ctx.outputs
    seen = {grids}
    out = []
    for mod in _modules():
        if time.time() > deadline:
            break
        try:
            hyps = list(hypcache.generate(mod, ctx))
        except Exception:
            continue
        taken = 0
        for hp in hyps:
            if taken >= _MAX_PER_MODULE or time.time() > deadline:
                break
            try:
                st = tuple(hp.apply(g) for g in grids)
            except Exception:
                continue
            if any(g is None for g in st):
                continue
            if st in seen:
                continue
            seen.add(st)
            taken += 1
            if st[:n_tr] == target:
                continue                 # exact already; not our business
            s = _score(st[:n_tr], target)
            if s <= 0.05:
                continue
            out.append((s, hp.cost, hp, st))
    out.sort(key=lambda r: (-r[0], r[1]))
    return out[:_MAX_SEEDS]


def _correct(state, ops, target, n_tr, depth, deadline, seen):
    """Short operator search from one state; returns (name, cost, chain)."""
    frontier = [((), 0.0, "", state)]
    results = []
    for _d in range(depth):
        nxt = []
        for chain, cost, name, st in frontier:
            if time.time() > deadline:
                return results
            for oname, ocost, f in ops:
                new = []
                ok = True
                for g in st:
                    try:
                        r = f(g)
                    except Exception:
                        ok = False
                        break
                    if r is None or not isinstance(r, tuple) or not G.valid(r):
                        ok = False
                        break
                    new.append(r)
                if not ok:
                    continue
                new = tuple(new)
                if new in seen:
                    continue
                seen.add(new)
                rec = (chain + (f,), cost + ocost, "%s(%s)" % (oname, name), new)
                if new[:n_tr] == target:
                    results.append(rec)
                    if len(results) >= 3:
                        return results
                nxt.append((_score(new[:n_tr], target), rec))
        if not nxt or time.time() > deadline:
            break
        nxt.sort(key=lambda x: -x[0])
        frontier = [r for _s, r in nxt[:14]]
    return results


def _compose(hp, chain):
    run = enum_core._apply_chain(chain)

    def go(g):
        t = hp.fn(g)
        if t is None:
            return None
        return run(t)
    return go


def generate(ctx):
    deadline = ctx.deadline or (time.time() + 6.0)
    if time.time() > deadline:
        return []
    n_tr = len(ctx.train)
    target = ctx.outputs
    seed_dl = min(deadline, time.time() + max(0.6, (deadline - time.time()) * 0.45))
    try:
        seeds = _seeds(ctx, seed_dl)
    except Exception:
        return []
    if not seeds:
        return []
    ops = enum_core.unary_ops(ctx, "full")
    res, seen = [], set()
    for i, (_s, cost, hp, st) in enumerate(seeds):
        now = time.time()
        if now > deadline or len(res) >= 24:
            break
        left = deadline - now
        share = max(0.08, left / max(1, len(seeds) - i))
        depth = 2 if i < 8 else 1
        try:
            found = _correct(st, ops, target, n_tr, depth,
                             min(deadline, now + share), seen)
        except Exception:
            continue
        for chain, ccost, cname, _new in found:
            res.append(Hyp("refine:%s>>%s" % (hp.name, cname),
                           _compose(hp, chain), 3.6 + cost + ccost, SOLVER))
    return res
