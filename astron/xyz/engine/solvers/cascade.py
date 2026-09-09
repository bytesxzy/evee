"""Two-stage solving: get close, then solve the residual.

Most solvers are all-or-nothing -- a rule either reproduces every training
output or it is discarded.  That throws away the most useful signal in the
whole portfolio: the transform that gets the grid *almost* right.  A task like
"crop to the frame, then recolour by size" is invisible to every single-stage
family, yet trivial once the crop has been applied.

So: run a shallow search scored by cell agreement instead of exact match, keep
the transforms that make the most progress, re-pose each as a fresh task
``(stage1(input), output)``, and hand it to the specialists.  The composition
is then validated end-to-end on the training pairs like anything else, so a
first stage that only looked promising costs nothing when the second stage
fails to close the gap.
"""

import time

from .. import enum_core
from .. import grid as G
from .. import hypcache
from ..task import Ctx, Hyp

SOLVER = "compose"
PHASE = 2


def _agreement(a, b):
    """Mean cell agreement, 0 when shapes differ."""
    if a is None or G.dims(a) != G.dims(b):
        return 0.0
    n = 0
    m = 0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            n += 1
            if x == y:
                m += 1
    return m / float(n)


def _shape_match(a, b):
    return 1.0 if G.dims(a) == G.dims(b) else 0.0


def _near_states(ctx, depth, keep, deadline):
    """Shallow BFS ranked by how close each state gets to the outputs."""
    n_tr = len(ctx.train)
    grids = ctx.inputs + ctx.test_inputs
    target = ctx.outputs
    ops = enum_core.unary_ops(ctx, "full") + enum_core.seed_ops(ctx)
    start_score = sum(_agreement(g, t) for g, t in zip(grids[:n_tr], target)) / n_tr
    seen = {grids}
    frontier = [(grids, (), "$", 0.0)]
    scored = []
    for _d in range(depth):
        nxt = []
        for state, chain, name, cost in frontier:
            if time.time() > deadline:
                break
            for oname, ocost, f in ops:
                st = []
                ok = True
                for g in state:
                    try:
                        r = f(g)
                    except Exception:
                        ok = False
                        break
                    if r is None or not isinstance(r, tuple) or not G.valid(r):
                        ok = False
                        break
                    st.append(r)
                if not ok:
                    continue
                st = tuple(st)
                if st in seen:
                    continue
                seen.add(st)
                sc = sum(_agreement(g, t) for g, t in zip(st[:n_tr], target)) / n_tr
                shp = sum(_shape_match(g, t) for g, t in zip(st[:n_tr], target)) / n_tr
                rec = (st, chain + (f,), "%s(%s)" % (oname, name), cost + ocost)
                # rank shape-correct states first: a second stage can repaint
                # cells but cannot resize the grid
                nxt.append((shp * 2.0 + sc - 0.02 * (cost + ocost), rec))
            if time.time() > deadline:
                break
        if not nxt:
            break
        nxt.sort(key=lambda x: -x[0])
        scored.extend(nxt[:keep])
        frontier = [r for _s, r in nxt[:keep]]
    scored.sort(key=lambda x: -x[0])
    out = []
    seen_states = set()
    for s, rec in scored:
        if rec[0] in seen_states:
            continue
        seen_states.add(rec[0])
        # a stage that destroys the grid outright is not worth continuing
        if s <= 0.0 and start_score > 0.0:
            continue
        out.append(rec)
        if len(out) >= keep:
            break
    return out


def _modules():
    from ..solvers import (blocks, cellwise, colormap, extend, geometry,
                           objects_map, objproc, paneltable, partition, regions,
                           select, selfstamp, sequence, substitute, symmetry,
                           tally, tiling)
    return (geometry, colormap, symmetry, partition, tiling, blocks, regions,
            select, cellwise, objects_map, substitute, sequence,
            paneltable, selfstamp, extend, tally)


def generate(ctx):
    deadline = ctx.deadline or (time.time() + 6.0)
    if time.time() > deadline:
        return []
    stage_dl = min(deadline, time.time() + max(1.0, (deadline - time.time()) * 0.4))
    try:
        states = _near_states(ctx, depth=2, keep=8, deadline=stage_dl)
    except Exception:
        return []
    if not states:
        return []
    res = []
    mods = _modules()
    n_tr = len(ctx.train)
    for state, chain, name, cost in states:
        if time.time() > deadline or len(res) > 40:
            break
        pairs = [(G.to_list(state[i]), G.to_list(ctx.train[i][1]))
                 for i in range(n_tr)]
        tins = [G.to_list(g) for g in state[n_tr:]]
        sub = Ctx(pairs, tins)
        share = max(0.4, (deadline - time.time()) / max(1, len(states)))
        sub.deadline = min(deadline, time.time() + share)
        run1 = enum_core._apply_chain(chain)
        for mod in mods:
            if time.time() > sub.deadline:
                break
            try:
                hyps = list(hypcache.generate(mod, sub))
            except Exception:
                continue
            for hp in hyps:
                if hp.fits(sub.train):
                    res.append(Hyp("%s>>%s" % (name, hp.name),
                                   _compose(run1, hp), 4.0 + cost + hp.cost,
                                   SOLVER))
                    break        # one per module per stage-1 is plenty
    return res


def _compose(run1, hp):
    def run(g):
        t = run1(g)
        if t is None:
            return None
        return hp.fn(t)
    return run
