"""Answer = one object, chosen relationally.

A large ARC family is "find the odd one out / the biggest / the one that is
different and show it".  The search space factorises cleanly into
(segmentation) x (selector) x (rendering), so we enumerate that product rather
than writing one rule per task.
"""

from collections import Counter

from .. import grid as G
from .. import objects as O
from ..task import Hyp

SOLVER = "select"

_SEGS = ("c4", "c8", "m4", "m8", "color")


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


def _render_patch(grid, o, bg):
    return o.filled(bg)


def _render_crop(grid, o, bg):
    return G.subgrid(grid, o.r0, o.c0, o.r1, o.c1)


def _render_mask(grid, o, bg):
    return tuple(tuple(o.color if v else bg for v in row) for row in o.mask)


def _render_solid(grid, o, bg):
    return G.const_grid(o.height, o.width, o.color)


def _render_cell(grid, o, bg):
    return ((o.color,),)


def _render_isolate(grid, o, bg):
    """Object kept in place, everything else cleared."""
    h, w = G.dims(grid)
    out = [[bg] * w for _ in range(h)]
    for r, c in o.cells:
        out[r][c] = grid[r][c]
    return tuple(tuple(r) for r in out)


def _render_delete(grid, o, bg):
    out = [list(r) for r in grid]
    for r, c in o.cells:
        out[r][c] = bg
    return tuple(tuple(r) for r in out)


RENDERERS = (
    ("patch", _render_patch, 0.0),
    ("crop", _render_crop, 0.2),
    ("mask", _render_mask, 1.0),
    ("solid", _render_solid, 1.5),
    ("cell", _render_cell, 1.5),
    ("isolate", _render_isolate, 1.0),
    ("delete", _render_delete, 1.0),
)


def _pick(grid, seg, sel, bg):
    objs = O.segment(grid, seg, bg)
    if not objs or len(objs) > 200:
        return None
    return sel(objs)


_OVERLAY_OPS = ("or", "and", "xor", "mode", "first", "last")


def _overlay_objects(g, seg, op, bg, use_mask):
    """Superimpose every object of equal bounding-box size.

    "Stack the shapes and report what they have in common" is a whole ARC
    family, and it is invisible to the panel-logic solver because the pieces
    are objects scattered on the canvas rather than panels cut from a lattice.
    """
    bg = G.bg_or(g, bg)
    objs = O.segment(g, seg, bg)
    if len(objs) < 2 or len(objs) > 30:
        return None
    dims = {(o.height, o.width) for o in objs}
    if len(dims) != 1:
        return None
    h, w = dims.pop()
    if h * w > 400:
        return None
    patches = [o.filled(bg) for o in objs]
    if use_mask:
        patches = [tuple(tuple(1 if v != bg else bg for v in r) for r in p)
                   for p in patches]
    out = []
    for r in range(h):
        row = []
        for c in range(w):
            vals = [p[r][c] for p in patches]
            live = [v for v in vals if v != bg]
            if op == "or":
                row.append(live[0] if live else bg)
            elif op == "and":
                row.append(live[0] if len(live) == len(vals) else bg)
            elif op == "xor":
                row.append(live[0] if len(live) == 1 else bg)
            elif op == "mode":
                row.append(Counter(vals).most_common(1)[0][0])
            elif op == "first":
                row.append(vals[0])
            else:
                row.append(vals[-1])
        out.append(tuple(row))
    return tuple(out)


def generate(ctx):
    res = []
    for bg in ([ctx.bg, None] if ctx.bg_varies else [ctx.bg]):
        res.extend(_rules(ctx, bg))
    return res


def _rules(ctx, bg):
    res = []
    for seg in ("c8", "m8", "c4"):
        for op in _OVERLAY_OPS:
            for um in (False, True):
                res.append(_h("overlay_%s_%s%s" % (seg, op, "_m" if um else ""),
                              (lambda s, o, bg, m:
                               lambda g: _overlay_objects(g, s, o, bg, m))(seg, op, bg, um),
                              4.5))
    for seg in _SEGS:
        for sname, sel in O.SELECTORS:
            for rname, rend, rcost in RENDERERS:
                res.append(_h(
                    "%s.%s.%s" % (seg, sname, rname),
                    (lambda seg, sel, rend, bg:
                     lambda g: _apply(g, seg, sel, rend, bg))(seg, sel, rend, bg),
                    3.0 + rcost))
    # objects sorted by a key, take the n-th
    for seg in ("c4", "c8", "m8"):
        for key in ("size", "bbox_area", "top", "left"):
            for idx in (0, 1, -1):
                res.append(_h(
                    "%s.nth_%s%+d" % (seg, key, idx),
                    (lambda seg, key, idx, bg:
                     lambda g: _nth(g, seg, key, idx, bg))(seg, key, idx, bg),
                    5.0))
    return res


def _apply(g, seg, sel, rend, bg):
    bg = G.bg_or(g, bg)
    o = _pick(g, seg, sel, bg)
    if o is None:
        return None
    return rend(g, o, bg)


def _nth(g, seg, key, idx, bg):
    bg = G.bg_or(g, bg)
    objs = O.segment(g, seg, bg)
    if not objs or len(objs) > 200:
        return None
    f = O.OBJ_RANKERS[key]
    objs = sorted(objs, key=f, reverse=True)
    try:
        o = objs[idx]
    except IndexError:
        return None
    return o.filled(bg)
