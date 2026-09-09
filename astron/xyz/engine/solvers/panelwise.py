"""Solve one panel, apply it to all of them.

Complementary to ``panelabs``: that one asks what happens *between* panels,
this one asks what happens *inside* each. When a grid is a lattice, the rule is
frequently applied independently to every cell of the lattice -- complete the
motif, recolour the contents, mirror it -- and every panel of every training
grid is then another demonstration of the same small rule.

That is a large multiplier on evidence: a task with three training grids and a
3x3 lattice supplies twenty-seven training pairs to the sub-task instead of
three, which is exactly the regime where the capacity guards elsewhere in the
engine stop rejecting things.
"""

import time

from .. import grid as G
from .. import hypcache
from ..task import Ctx, Hyp
from .panelabs import _positions
from .partition import _decompositions, sep_color_of

SOLVER = "partition"
PHASE = 2


def _norm_map(g, k):
    """Swap this grid's separator colour with a canonical slot.

    A lattice task often draws the same rule in a different colour in every
    grid, which makes the per-panel rule look inconsistent across examples.
    Swapping the separator colour into a fixed slot makes them the same rule
    again; the swap is an involution, so undoing it after rendering is the
    same operation.
    """
    sep = sep_color_of(g)
    if sep is None or sep == k:
        return None
    return {sep: k, k: sep}


class _PanelwiseRule:
    __slots__ = ("dec", "hyp", "norm")

    def __init__(self, dec, hyp, norm=None):
        self.dec = dec
        self.hyp = hyp
        self.norm = norm

    def __call__(self, g):
        cmap = None
        if self.norm is not None:
            cmap = _norm_map(g, self.norm)
            if cmap is None:
                return None
            g = G.apply_cmap(g, cmap)
        mat = self.dec(g)
        if not mat:
            return None
        pos = _positions(g, None, self.dec)
        if pos is None:
            return None
        rows, cols, ph, pw = pos
        out = [list(r) for r in g]
        for i, r0 in enumerate(rows):
            for j, c0 in enumerate(cols):
                p = mat[i][j]
                if p is None or G.dims(p) != (ph, pw):
                    return None
                q = self.hyp.fn(p)
                if q is None or G.dims(q) != (ph, pw):
                    return None
                for rr in range(ph):
                    qrow = q[rr]
                    orow = out[r0 + rr]
                    for cc in range(pw):
                        orow[c0 + cc] = qrow[cc]
        res = tuple(tuple(r) for r in out)
        return G.apply_cmap(res, cmap) if cmap else res


def _modules():
    from ..solvers import (cellwise, colormap, extend, geometry, objects_map,
                           objproc, selfstamp, sequence, substitute, symmetry,
                           tiling)
    return (geometry, colormap, symmetry, tiling, cellwise, objects_map,
            substitute, sequence, selfstamp, extend)


def _canonical_slot(ctx):
    used = set(ctx.in_palette) | set(ctx.out_palette)
    for k in range(9, -1, -1):
        if k not in used:
            return k
    return 9


def generate(ctx):
    if not ctx.same_shape:
        return []
    deadline = ctx.deadline or (time.time() + 4.0)
    res = []
    slot = _canonical_slot(ctx)
    for dname, dcost, dec in _decompositions(ctx)[:4]:
        for norm in (None, slot):
            if time.time() > deadline or len(res) > 12:
                break
            res.extend(_one(ctx, dname, dcost, dec, norm, deadline))
    return res


def _one(ctx, dname, dcost, dec, norm, deadline):
    res = []
    try:
        pairs = []
        changed = False
        shape = None
        for a, b in ctx.train:
            if norm is not None:
                cm = _norm_map(a, norm)
                if cm is None:
                    return res
                a, b = G.apply_cmap(a, cm), G.apply_cmap(b, cm)
            ma, mb = dec(a), dec(b)
            if not ma or not mb or len(ma) != len(mb) or len(ma[0]) != len(mb[0]):
                return res
            for ra, rb in zip(ma, mb):
                for pa, pb in zip(ra, rb):
                    if pa is None or pb is None or G.dims(pa) != G.dims(pb):
                        return res
                    if shape is None:
                        shape = G.dims(pa)
                    elif G.dims(pa) != shape:
                        return res
                    if pa != pb:
                        changed = True
                    pairs.append((G.to_list(pa), G.to_list(pb)))
        if not changed or not pairs or len(pairs) > 80:
            return res
        tins = []
        for t in ctx.test_inputs:
            if norm is not None:
                cm = _norm_map(t, norm)
                if cm is None:
                    return res
                t = G.apply_cmap(t, cm)
            m = dec(t)
            if not m:
                return res
            for row in m:
                for p in row:
                    if p is None or G.dims(p) != shape:
                        return res
                    tins.append(G.to_list(p))
    except Exception:
        return res

    sub = Ctx(pairs, tins)
    sub.deadline = min(deadline, time.time() + 1.2)
    tag = dname if norm is None else "%s~%d" % (dname, norm)
    for mod in _modules():
        if time.time() > sub.deadline:
            break
        try:
            hyps = list(hypcache.generate(mod, sub))
        except Exception:
            continue
        for hp in hyps:
            if hp.fits(sub.train):
                res.append(Hyp("panelwise[%s]>>%s" % (tag, hp.name),
                               _PanelwiseRule(dec, hp, norm),
                               dcost + 1.5 + hp.cost, SOLVER))
                break
        if len(res) > 12:
            break
    return res
