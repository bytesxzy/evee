"""Outputs assembled *from the objects* rather than edited in the grid.

``select`` answers "which one object is it"; ``partition`` combines panels cut
out by separator lines.  Neither covers the family where the answer is built
out of several objects that were never laid out on a grid to begin with:
stack the shapes in size order, overlay them, or report one cell per object.

Every rule here is fitted and then verified on all demonstrations, so a
plausible-looking arrangement that does not reproduce the training pairs is
discarded like any other hypothesis.
"""

from collections import Counter

from .. import grid as G
from .. import objects as O
from ..task import Hyp

SOLVER = "select"

_SEGS = ("c4", "c8", "m4", "m8", "color")

_ORDERS = {
    "size": lambda o: (-o.size, o.r0, o.c0),
    "size_asc": lambda o: (o.size, o.r0, o.c0),
    "pos": lambda o: (o.r0, o.c0),
    "col": lambda o: (o.c0, o.r0),
    "color": lambda o: (o.color, o.r0, o.c0),
    "holes": lambda o: (-o.holes_count(), o.r0, o.c0),
    "bbox": lambda o: (-o.bbox_area, o.r0, o.c0),
}


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


def _objs(g, seg, bg, cap=40):
    b = G.bg_or(g, bg)
    objs = O.segment(g, seg, b)
    if not objs or len(objs) > cap:
        return None, b
    return objs, b


# --- overlay ---------------------------------------------------------------

def _overlay(objs, bg, mode, order):
    """Stack every object's patch, all normalised to a shared size."""
    dims = {(o.height, o.width) for o in objs}
    if len(dims) != 1:
        return None
    h, w = dims.pop()
    if h * w > 900:
        return None
    seq = sorted(objs, key=_ORDERS[order])
    out = [[bg] * w for _ in range(h)]
    if mode == "first":
        for o in seq:
            p = o.patch
            for r in range(h):
                for c in range(w):
                    if p[r][c] is not None and out[r][c] == bg:
                        out[r][c] = p[r][c]
    elif mode == "last":
        for o in seq:
            p = o.patch
            for r in range(h):
                for c in range(w):
                    if p[r][c] is not None:
                        out[r][c] = p[r][c]
    elif mode == "count":
        cnt = [[0] * w for _ in range(h)]
        for o in seq:
            p = o.patch
            for r in range(h):
                for c in range(w):
                    if p[r][c] is not None:
                        cnt[r][c] += 1
        n = len(seq)
        for r in range(h):
            for c in range(w):
                if cnt[r][c] == n:
                    out[r][c] = seq[0].patch[r][c] if seq[0].patch[r][c] is not None else bg
    elif mode == "odd":
        cnt = [[0] * w for _ in range(h)]
        vals = [[bg] * w for _ in range(h)]
        for o in seq:
            p = o.patch
            for r in range(h):
                for c in range(w):
                    if p[r][c] is not None:
                        cnt[r][c] += 1
                        vals[r][c] = p[r][c]
        for r in range(h):
            for c in range(w):
                if cnt[r][c] % 2 == 1:
                    out[r][c] = vals[r][c]
    else:
        return None
    return tuple(tuple(r) for r in out)


# --- stacking --------------------------------------------------------------

def _stack(objs, bg, order, axis, dedup, render):
    seq = sorted(objs, key=_ORDERS[order])
    parts = []
    seen = set()
    for o in seq:
        if render == "patch":
            p = o.filled(bg)
        elif render == "mask":
            p = tuple(tuple(o.color if v else bg for v in row) for row in o.mask)
        else:
            p = G.const_grid(o.height, o.width, o.color)
        if dedup:
            k = p
            if k in seen:
                continue
            seen.add(k)
        parts.append(p)
    if len(parts) < 2:
        return None
    if axis == "v":
        w = {G.dims(p)[1] for p in parts}
        if len(w) != 1:
            return None
        out = parts[0]
        for p in parts[1:]:
            out = G.vconcat(out, p)
    else:
        h = {G.dims(p)[0] for p in parts}
        if len(h) != 1:
            return None
        out = parts[0]
        for p in parts[1:]:
            out = G.hconcat(out, p)
    return out if out is not None and G.area(out) <= 2500 else None


# --- one cell per object ---------------------------------------------------

def _summary(objs, bg, order, shape):
    seq = sorted(objs, key=_ORDERS[order])
    n = len(seq)
    if n < 2 or n > 40:
        return None
    cols = [o.color for o in seq]
    if shape == "row":
        return (tuple(cols),)
    if shape == "col":
        return tuple((c,) for c in cols)
    if shape == "square":
        k = int(round(n ** 0.5))
        if k * k != n:
            return None
        return tuple(tuple(cols[r * k:(r + 1) * k]) for r in range(k))
    return None


def _grid_of_objects(objs, bg):
    """Objects laid out on a lattice -> one cell each, keeping their layout."""
    rows = sorted({o.r0 for o in objs})
    cols = sorted({o.c0 for o in objs})
    if len(rows) * len(cols) != len(objs) or len(rows) > 12 or len(cols) > 12:
        return None
    ri = {v: i for i, v in enumerate(rows)}
    ci = {v: i for i, v in enumerate(cols)}
    out = [[bg] * len(cols) for _ in range(len(rows))]
    for o in objs:
        out[ri[o.r0]][ci[o.c0]] = o.color
    return tuple(tuple(r) for r in out)


# --- ordinal selection -----------------------------------------------------

def _nth(objs, bg, order, idx, render):
    seq = sorted(objs, key=_ORDERS[order])
    if not (-len(seq) <= idx < len(seq)):
        return None
    o = seq[idx]
    if render == "patch":
        return o.filled(bg)
    if render == "mask":
        return tuple(tuple(o.color if v else bg for v in row) for row in o.mask)
    return G.const_grid(o.height, o.width, o.color)


# --- driver ----------------------------------------------------------------

def _rule(seg, bg, fn):
    def run(g):
        objs, b = _objs(g, seg, bg)
        if objs is None:
            return None
        try:
            return fn(objs, b)
        except Exception:
            return None
    return run


def _bgs(ctx):
    out = [ctx.bg]
    if ctx.bg_varies:
        out.append(None)
    return out


def generate(ctx):
    if ctx.same_shape:
        return []
    res, seen = [], set()

    def offer(name, fn, cost):
        if len(res) >= 60:
            return
        # this family is newer and less corroborated than ``select``; where both
        # explain the demonstrations, the established one should win
        h = _h(name, fn, cost + 0.7)
        if not h.fits(ctx.train):
            return
        try:
            sig = tuple(h.apply(g) for g in ctx.test_inputs)
        except Exception:
            return
        if sig in seen:
            return
        seen.add(sig)
        res.append(h)

    for bg in _bgs(ctx):
        for seg in _SEGS:
            if ctx.timed_out() or len(res) >= 60:
                return res
            for order in _ORDERS:
                for mode in ("first", "last", "count", "odd"):
                    offer("ov[%s/%s/%s/%s]" % (seg, bg, mode, order),
                          _rule(seg, bg,
                                lambda o, b, m=mode, r=order: _overlay(o, b, m, r)),
                          3.0)
                if ctx.timed_out():
                    return res
            for order in _ORDERS:
                for axis in ("v", "h"):
                    for dedup in (False, True):
                        for render in ("patch", "mask", "solid"):
                            offer("st[%s/%s/%s%s/%s]" % (seg, order, axis,
                                                         "d" if dedup else "",
                                                         render),
                                  _rule(seg, bg,
                                        lambda o, b, r=order, a=axis, d=dedup,
                                        q=render: _stack(o, b, r, a, d, q)),
                                  3.4)
                if ctx.timed_out():
                    return res
            for order in _ORDERS:
                for shape in ("row", "col", "square"):
                    offer("sum[%s/%s/%s]" % (seg, order, shape),
                          _rule(seg, bg,
                                lambda o, b, r=order, s=shape: _summary(o, b, r, s)),
                          3.2)
                for idx in (0, 1, 2, -1, -2):
                    for render in ("patch", "mask", "solid"):
                        offer("nth[%s/%s/%d/%s]" % (seg, order, idx, render),
                              _rule(seg, bg,
                                    lambda o, b, r=order, i=idx,
                                    q=render: _nth(o, b, r, i, q)),
                              3.0 + 0.2 * abs(idx))
            offer("lattice[%s]" % seg, _rule(seg, bg, _grid_of_objects), 3.0)
    return res
