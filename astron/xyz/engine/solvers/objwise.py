"""Solve one object, apply it to all of them.

The object analogue of ``panelwise``. When a task says "do this to every
shape", each object of each training grid is another demonstration of the same
small rule, and the sub-task -- bounding-box patch in, bounding-box patch out --
is usually far easier than the whole grid.

The multiplier on evidence is the point: a task with three grids and five
objects each gives the sub-task fifteen pairs instead of three, which is often
the difference between a rule that looks like memorisation and one that clearly
is not.
"""

import time

from .. import grid as G
from .. import objects as O
from .. import hypcache
from ..task import Ctx, Hyp

SOLVER = "objects"
PHASE = 2

_SEGS = ("c8", "m8", "c4", "m4")


class _ObjwiseRule:
    __slots__ = ("seg", "bg", "hyp")

    def __init__(self, seg, bg, hyp):
        self.seg = seg
        self.bg = bg
        self.hyp = hyp

    def __call__(self, g):
        bg = G.bg_or(g, self.bg)
        objs = O.segment(g, self.seg, bg)
        if not objs or len(objs) > 40:
            return None
        out = [list(r) for r in g]
        for o in sorted(objs, key=lambda x: (x.r0, x.c0)):
            patch = G.subgrid(g, o.r0, o.c0, o.r1, o.c1)
            if patch is None:
                return None
            q = self.hyp.fn(patch)
            if q is None or G.dims(q) != G.dims(patch):
                return None
            for r in range(o.height):
                qrow = q[r]
                orow = out[o.r0 + r]
                for c in range(o.width):
                    orow[o.c0 + c] = qrow[c]
        return tuple(tuple(r) for r in out)


def _modules():
    from ..solvers import (cellwise, colormap, extend, geometry, objects_map,
                           objproc, selfstamp, symmetry, tiling)
    return (geometry, colormap, symmetry, tiling, cellwise, objects_map,
            selfstamp, extend)


def generate(ctx):
    if not ctx.same_shape:
        return []
    deadline = ctx.deadline or (time.time() + 3.0)
    res = []
    for bg in ([ctx.bg, None] if ctx.bg_varies else [ctx.bg]):
        for seg in _SEGS:
            if time.time() > deadline or len(res) > 10:
                break
            res.extend(_one(ctx, seg, bg, deadline))
    return res


def _one(ctx, seg, bg, deadline):
    res = []
    try:
        pairs = []
        changed = False
        for a, b in ctx.train:
            abg = G.bg_or(a, bg)
            objs = O.segment(a, seg, abg)
            if not objs or len(objs) > 40:
                return res
            for o in objs:
                pa = G.subgrid(a, o.r0, o.c0, o.r1, o.c1)
                pb = G.subgrid(b, o.r0, o.c0, o.r1, o.c1)
                if pa is None or pb is None:
                    return res
                if pa != pb:
                    changed = True
                pairs.append((G.to_list(pa), G.to_list(pb)))
        if not changed or len(pairs) < 3 or len(pairs) > 120:
            return res
        tins = []
        for t in ctx.test_inputs:
            tbg = G.bg_or(t, bg)
            for o in O.segment(t, seg, tbg):
                p = G.subgrid(t, o.r0, o.c0, o.r1, o.c1)
                if p is None:
                    return res
                tins.append(G.to_list(p))
        if not tins:
            return res
    except Exception:
        return res

    sub = Ctx(pairs, tins[:8])
    sub.deadline = min(deadline, time.time() + 1.0)
    for mod in _modules():
        if time.time() > sub.deadline:
            break
        try:
            hyps = list(hypcache.generate(mod, sub))
        except Exception:
            continue
        for hp in hyps:
            if hp.fits(sub.train):
                res.append(Hyp("objwise[%s]>>%s" % (seg, hp.name),
                               _ObjwiseRule(seg, bg, hp),
                               3.5 + hp.cost, SOLVER))
                break
        if len(res) > 10:
            break
    return res
