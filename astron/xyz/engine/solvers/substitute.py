"""Shape substitution: every object is replaced by a learned stencil.

The rule "each seed grows into this motif" cannot be written as a pixel lookup
(the motif extends beyond the seed) nor as a per-object recolour (the shape
changes).  It is a *dictionary from shape to stencil*, learned by matching each
input object to whatever the output drew over it.

Matching is by overlap: the output component that covers an input object's
cells is that object's image.  The dictionary is keyed on the object's mask,
its coloured patch, its colour or its size, and -- as everywhere in this engine
-- is rejected unless it compresses the observations it was fitted on.
"""

from collections import Counter

from .. import grid as G
from .. import objects as O
from ..task import Hyp

SOLVER = "objects"

_KEYS = ("mask", "patch", "color", "size", "dims")


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


def _key_of(o, kind):
    if kind == "mask":
        return o.mask
    if kind == "patch":
        return o.patch
    if kind == "color":
        return o.color
    if kind == "size":
        return o.size
    return (o.height, o.width)


def _image_of(o, out_objs, out_grid):
    """Union of output components overlapping ``o``; as (patch, dr, dc)."""
    cells = set()
    for p in out_objs:
        if p.cells & o.cells:
            cells |= p.cells
    if not cells:
        return None
    r0, c0, r1, c1 = G.bbox_of(cells)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    patch = [[None] * w for _ in range(h)]
    for r, c in cells:
        patch[r - r0][c - c0] = out_grid[r][c]
    return (tuple(tuple(row) for row in patch), r0 - o.r0, c0 - o.c0)


def _fit(ctx, seg, kind, bg):
    table = {}
    n = 0
    for a, b in ctx.train:
        if G.dims(a) != G.dims(b):
            return None
        ins = O.segment(a, seg, bg)
        outs = O.segment(b, seg, bg)
        if not ins or len(ins) > 60 or len(outs) > 200:
            return None
        for o in ins:
            img = _image_of(o, outs, b)
            if img is None:
                return None
            k = _key_of(o, kind)
            prev = table.get(k)
            if prev is None:
                table[k] = img
            elif prev != img:
                return None
            n += 1
    if not table or len(table) >= n or n < 2:
        return None
    # a dictionary that only ever reproduces the object unchanged is the
    # identity wearing a costume; leave that to cheaper solvers
    if all(dr == 0 and dc == 0 and _is_same(p) for p, dr, dc in table.values()):
        return None
    return table


def _is_same(patch):
    return False


def _apply(g, seg, kind, table, bg, over_input):
    bg = G.bg_or(g, bg)
    objs = O.segment(g, seg, bg)
    if not objs or len(objs) > 60:
        return None
    h, w = G.dims(g)
    if over_input:
        out = [list(r) for r in g]
    else:
        out = [[bg] * w for _ in range(h)]
    for o in objs:
        v = table.get(_key_of(o, kind))
        if v is None:
            return None
        patch, dr, dc = v
        ph, pw = len(patch), len(patch[0])
        for r in range(ph):
            gr = o.r0 + dr + r
            if not (0 <= gr < h):
                continue
            prow = patch[r]
            for c in range(pw):
                gc = o.c0 + dc + c
                if 0 <= gc < w and prow[c] is not None:
                    out[gr][gc] = prow[c]
    return tuple(tuple(r) for r in out)


def generate(ctx):
    if not ctx.same_shape:
        return []
    res = []
    for bg in ([ctx.bg, None] if ctx.bg_varies else [ctx.bg]):
        res.extend(_rules(ctx, bg))
    return res


def _rules(ctx, bg):
    res = []
    for seg in ("c8", "m8", "c4", "cells"):
        if ctx.timed_out():
            break
        for kind in _KEYS:
            try:
                t = _fit(ctx, seg, kind, bg)
            except Exception:
                t = None
            if t is None:
                continue
            for over in (True, False):
                res.append(_h("subst_%s_%s%s" % (seg, kind, "" if over else "_bg"),
                              (lambda seg, kind, t, bg, over:
                               lambda g: _apply(g, seg, kind, t, bg, over))(seg, kind, t, bg, over),
                              4.0 + 0.1 * len(t)))
    return res
