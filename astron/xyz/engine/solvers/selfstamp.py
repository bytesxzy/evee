"""The grid used as its own brush.

Two closely related families, both about a grid printing copies of itself.

**Gated self-stamping.**  ``out[i*h+r][j*w+c] = g[r][c]`` wherever cell
``(i, j)`` passes a test.  ``geometry`` already offers this, but only with the
test "is not background" (or its negation).  That is one gate out of many, and
the interesting tasks pick a *particular* colour -- and often a colour that is
not the same one in every example, so it has to be named relationally: the most
common non-background colour, the rarest one, the colour of the corner.  Widening
the gate is the whole content of this family; the stamping itself is unchanged.
The stamp may also be a dihedral image of the grid rather than the grid.

**Offset repetition.**  A copy of the grid's content laid down again and again
at a fixed step -- most often diagonally -- on a canvas larger than the input.
Tiling covers whole-grid-period repeats; this covers steps that are *not* the
grid size, where the copies overlap each other and the last ones run off the
edge.  The step is not searched blindly: candidate steps are bounded by the
input's own dimensions and every one of them is checked against every
demonstration, so a step that only explains the first pair is discarded.
"""

from collections import Counter

from .. import grid as G
from ..task import Hyp

SOLVER = "tiling"

_MAX_OUT = 1400


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


# --- gate selectors --------------------------------------------------------

def _gates(ctx):
    pal = sorted(ctx.in_palette)
    bg = ctx.bg
    gates = [("nonbg", 0.0, lambda g, b=bg: (lambda v: v != b)),
             ("isbg", 0.3, lambda g, b=bg: (lambda v: v == b))]
    for c in pal:
        gates.append(("is%d" % c, 0.6,
                      (lambda c: lambda g: (lambda v: v == c))(c)))
        gates.append(("not%d" % c, 0.9,
                      (lambda c: lambda g: (lambda v: v != c))(c)))
    gates.append(("top", 0.5, lambda g: (lambda v, m=G.most_common_color(g): v == m)))
    gates.append(("nottop", 0.7, lambda g: (lambda v, m=G.most_common_color(g): v != m)))
    gates.append(("rare", 0.5, lambda g: (lambda v, m=G.least_common_color(g): v == m)))
    gates.append(("topnb", 0.6, lambda g: (lambda v, m=_top_nonbg(g, ctx.bg): v == m)))
    return gates


def _top_nonbg(g, bg):
    h = G.histogram(g)
    h.pop(bg, None)
    return max(h.items(), key=lambda kv: (kv[1], -kv[0]))[0] if h else -1


def _stamp(g, gatefn, fill, xform):
    h, w = G.dims(g)
    if h * w > 40 or (h * h) * (w * w) > _MAX_OUT:
        return None
    src = xform(g)
    if G.dims(src) != (h, w):
        return None
    out = [[fill] * (w * w) for _ in range(h * h)]
    test = gatefn(g)
    for i in range(h):
        for j in range(w):
            if test(g[i][j]):
                for r in range(h):
                    orow = out[i * h + r]
                    srow = src[r]
                    for c in range(w):
                        orow[j * w + c] = srow[c]
    return tuple(tuple(r) for r in out)


def _mask_only(g, gatefn, on, off):
    h, w = G.dims(g)
    test = gatefn(g)
    return tuple(tuple(on if test(v) else off for v in row) for row in g)


# --- offset repetition -----------------------------------------------------

def _out_shape(ctx, g):
    h, w = G.dims(g)
    if ctx.const_out_shape:
        return ctx.const_out_shape
    if ctx.shape_ratio:
        ky, kx = ctx.shape_ratio
        return (h * ky, w * kx)
    return None


def _repeat(ctx, g, dr, dc, fill, back, xform):
    d = _out_shape(ctx, g)
    if d is None:
        return None
    H, W = d
    if H * W > _MAX_OUT:
        return None
    h, w = G.dims(g)
    src = xform(g)
    if G.dims(src) != (h, w):
        return None
    out = [[fill] * W for _ in range(H)]
    ks = range(-back, max(H, W) + 1) if back else range(0, max(H, W) + 1)
    for k in ks:
        r0, c0 = dr * k, dc * k
        if r0 >= H or c0 >= W or r0 + h <= 0 or c0 + w <= 0:
            continue
        for r in range(h):
            rr = r0 + r
            if not (0 <= rr < H):
                continue
            srow = src[r]
            orow = out[rr]
            for c in range(w):
                cc = c0 + c
                if 0 <= cc < W and srow[c] != fill:
                    orow[cc] = srow[c]
    return tuple(tuple(r) for r in out)


def generate(ctx):
    res, seen = [], set()

    def offer(name, fn, cost):
        if len(res) >= 24:
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

    xforms = [("id", 0.0, lambda g: g)] + [(n, 0.5, f) for n, f in G.DIHEDRAL[1:]]
    fills = sorted({ctx.bg, 0} | (ctx.out_palette - ctx.in_palette))[:3]

    ratio = ctx.shape_ratio
    inv = ctx.inv_shape_ratio
    square_law = all(G.dims(b)[0] == G.dims(a)[0] ** 2 and
                     G.dims(b)[1] == G.dims(a)[1] ** 2
                     for a, b in ctx.train)
    if square_law:
        for gname, gcost, gate in _gates(ctx):
            if ctx.timed_out():
                return res
            for xname, xcost, xf in xforms:
                for fill in fills:
                    offer("stamp[%s/%s/%d]" % (gname, xname, fill),
                          (lambda gt, f, xf: lambda g: _stamp(g, gt, f, xf))(
                              gate, fill, xf),
                          3.2 + gcost + xcost)
                if len(res) >= 24:
                    return res

    if ctx.same_shape or inv:
        # the gate read as a picture in its own right
        pal = sorted(ctx.out_palette)[:4]
        for gname, gcost, gate in _gates(ctx):
            if ctx.timed_out():
                return res
            for on in pal:
                for off in pal:
                    if on == off:
                        continue
                    offer("gate[%s->%d/%d]" % (gname, on, off),
                          (lambda gt, a, b: lambda g: _mask_only(g, gt, a, b))(
                              gate, on, off),
                          3.4 + gcost)

    if ratio or ctx.const_out_shape:
        hs = max(len(a) for a in ctx.all_inputs)
        ws = max(len(a[0]) for a in ctx.all_inputs)
        if hs <= 14 and ws <= 14:
            for dr in range(-hs, hs + 1):
                if ctx.timed_out():
                    return res
                for dc in range(-ws, ws + 1):
                    if dr == 0 and dc == 0:
                        continue
                    for fill in fills[:2]:
                        for back in (0, 1):
                            offer("rep[%+d%+d/%d%s]" % (dr, dc, fill,
                                                        "b" if back else ""),
                                  (lambda dr, dc, f, b: lambda g: _repeat(
                                      ctx, g, dr, dc, f, b * 40,
                                      lambda x: x))(dr, dc, fill, back),
                                  3.6 + 0.05 * (abs(dr) + abs(dc)))
    return res
