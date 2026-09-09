"""Answers that are a count, drawn onto a fixed canvas.

``counting`` already covers the case where the *size* of the output is the
number: an ``n x n`` block, a bar of length ``n``, the input repeated ``n``
times.  A second, equally common shape of the same idea is invisible to it --
the canvas has a fixed size and the number is expressed by *how much of it is
filled*.  "Three shapes, so three cells of the three-by-three are red."  The
grid law there is constant, so nothing that reasons from output dimensions can
see the count at all.

The same fixed canvas serves a neighbouring family: scattered marks gathered up
in reading order and laid into a small grid, padded where they run out.

Both are fitted, not searched.  A counter is read off every training input, a
render law is proposed, and any law that misses one pair is dropped.  The guard
``counting`` needed applies here too: **a counter that never varies across the
training inputs proves nothing**, because "n = the number of objects" and
"n = 3" are then indistinguishable, and the constant would generalise wrongly.
"""

from collections import Counter

from .. import grid as G
from .. import objects as O
from ..task import Hyp

SOLVER = "counting"

_SEGS = ("c4", "c8", "m4", "m8", "color")


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


# --- counters --------------------------------------------------------------

def _counters(ctx):
    bg = ctx.bg
    out = []
    for seg in _SEGS:
        out.append(("n" + seg, 0.4,
                    (lambda seg: lambda g, b=bg: len(
                        O.segment(g, seg, G.bg_or(g, b))))(seg)))
        out.append(("big" + seg, 0.8,
                    (lambda seg: lambda g, b=bg: max(
                        [o.size for o in O.segment(g, seg, G.bg_or(g, b))]
                        or [0]))(seg)))
        out.append(("shp" + seg, 0.9,
                    (lambda seg: lambda g, b=bg: len(
                        {o.norm_key() for o in O.segment(g, seg, G.bg_or(g, b))}
                    ))(seg)))
    out.append(("ncol", 0.5, lambda g, b=bg: len(G.palette(g) - {G.bg_or(g, b)})))
    out.append(("ncell", 0.6, lambda g, b=bg: sum(
        1 for row in g for v in row if v != G.bg_or(g, b))))
    for c in sorted(ctx.in_palette)[:8]:
        out.append(("cnt%d" % c, 0.9,
                    (lambda c: lambda g: G.count_color(g, c))(c)))
    return out


# --- renders ---------------------------------------------------------------

def _fill_n(shape, n, on, off, order):
    h, w = shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    if order == "col":
        cells = [(r, c) for c in range(w) for r in range(h)]
    elif order == "rrow":
        cells = cells[::-1]
    elif order == "rcol":
        cells = [(r, c) for c in range(w) for r in range(h)][::-1]
    elif order == "up":
        cells = [(r, c) for r in range(h - 1, -1, -1) for c in range(w)]
    if n < 0 or n > h * w:
        return None
    out = [[off] * w for _ in range(h)]
    for r, c in cells[:n]:
        out[r][c] = on
    return tuple(tuple(r) for r in out)


def _fill_rows(shape, n, on, off, axis):
    h, w = shape
    if n < 0 or n > (h if axis == "row" else w):
        return None
    out = [[off] * w for _ in range(h)]
    if axis == "row":
        for r in range(n):
            out[r] = [on] * w
    else:
        for r in range(h):
            for c in range(n):
                out[r][c] = on
    return tuple(tuple(r) for r in out)


# --- gather ----------------------------------------------------------------

def _place(cells, h, w, bg, snake):
    """Reading order for the destination; serpentine is a common ARC idiom."""
    out = [[bg] * w for _ in range(h)]
    for i, v in enumerate(cells):
        r, c = i // w, i % w
        if snake and r % 2:
            c = w - 1 - c
        out[r][c] = v
    return tuple(tuple(r) for r in out)


def _gather(g, shape, bg, order, mode, snake=False):
    h, w = shape
    cells = []
    gh, gw = G.dims(g)
    idx = [(r, c) for r in range(gh) for c in range(gw)]
    if order == "col":
        idx = [(r, c) for c in range(gw) for r in range(gh)]
    for r, c in idx:
        if g[r][c] != bg:
            cells.append(g[r][c])
    if mode == "uniq":
        seen, keep = set(), []
        for v in cells:
            if v not in seen:
                seen.add(v)
                keep.append(v)
        cells = keep
    elif mode == "freq":
        cnt = Counter(cells)
        cells = [v for v, _n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]
    if len(cells) > h * w:
        return None
    return _place(cells, h, w, bg, snake)


# --- driver ----------------------------------------------------------------

def generate(ctx):
    shape = ctx.const_out_shape
    if shape is None or shape[0] * shape[1] > 120:
        return []
    res, seen = [], set()

    def offer(name, fn, cost):
        if len(res) >= 20:
            return
        hp = _h(name, fn, cost)
        if not hp.fits(ctx.train):
            return
        try:
            sig = tuple(hp.apply(t) for t in ctx.test_inputs)
        except Exception:
            return
        if sig in seen:
            return
        seen.add(sig)
        res.append(hp)

    pal = sorted(ctx.out_palette)
    if len(pal) <= 3:
        for cname, ccost, counter in _counters(ctx):
            if ctx.timed_out() or len(res) >= 20:
                break
            try:
                vals = [counter(a) for a in ctx.inputs]
            except Exception:
                continue
            if len(set(vals)) < 2:
                continue          # a constant counter is not evidence of counting
            for on in pal:
                for off in pal:
                    if on == off:
                        continue
                    for order in ("row", "col", "rrow", "rcol", "up"):
                        offer("tally[%s/%s/%d>%d]" % (cname, order, on, off),
                              (lambda f, o, a, b: lambda g: _fill_n(
                                  shape, f(g), a, b, o))(counter, order, on, off),
                              2.8 + ccost)
                    for axis in ("row", "col"):
                        offer("bars[%s/%s/%d>%d]" % (cname, axis, on, off),
                              (lambda f, x, a, b: lambda g: _fill_rows(
                                  shape, f(g), a, b, x))(counter, axis, on, off),
                              3.0 + ccost)

    bg = ctx.bg
    for order in ("row", "col"):
        for mode in ("all", "uniq", "freq"):
            for snake in (False, True):
                offer("gather[%s/%s%s]" % (order, mode, "/snake" if snake else ""),
                      (lambda o, m, s: lambda g: _gather(
                          g, shape, G.bg_or(g, bg), o, m, s))(order, mode, snake),
                      3.0 + (0.3 if snake else 0.0))
    return res
