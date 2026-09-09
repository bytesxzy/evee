"""Continuing a pattern that the grid only started.

Two idioms, both about periodicity, both common and both currently unreachable.

**Periodic completion.**  Part of the grid carries a repeating pattern and the
rest is empty; the answer continues the pattern into the empty part.  This is
not the same as tiling a motif: the period has to be *inferred from the cells
that are present*, agreeing across every residue class, and it has to be the
smallest such period, because a larger one would also fit the evidence and
would then copy the emptiness along with the pattern.  A period is accepted only
when every pair of cells in the same class that are both non-empty agree.

**Reflective extension.**  A grid extended past its own edge by bouncing rather
than wrapping: rows ``a b c`` continue ``b a b c b a``.  Plain tiling gets this
wrong at every seam, which is why the family looks unrelated to it from the
inside even though both are index arithmetic.  The index map is
``i -> h-1-|(i mod 2h-2) - (h-1)|`` for the shared-edge reading, and the
wrapping reading is offered alongside it.

The target size is taken from the task's own shape law -- a constant output
shape or an integer ratio -- so nothing here has to guess how far to continue.
"""

from .. import grid as G
from ..task import Hyp

SOLVER = "tiling"

_MAX = 2500


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


# --- periodic completion ---------------------------------------------------

def _consistent_period(g, bg, py, px):
    """Fill values per residue class, or ``None`` if the period contradicts."""
    h, w = G.dims(g)
    table = {}
    for r in range(h):
        row = g[r]
        for c in range(w):
            v = row[c]
            if v == bg:
                continue
            k = (r % py, c % px)
            if table.setdefault(k, v) != v:
                return None
    return table


def _content_box(g, bg):
    rs = [r for r, row in enumerate(g) for v in row if v != bg]
    cs = [c for row in g for c, v in enumerate(row) if v != bg]
    if not rs:
        return None
    return min(rs), min(cs), max(rs), max(cs)


def _continue_outward(g, bg, axis):
    """Read the period off the drawn region, then continue it into the empty one.

    Treating background as *unknown* everywhere is wrong for this family: the
    empty cells inside the drawn pattern are part of the pattern, and only the
    ones outside it are missing.  So the period is fitted against the content's
    bounding box with background taken at face value, and the fill is applied
    only beyond that box.
    """
    box = _content_box(g, bg)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    h, w = G.dims(g)
    bh, bw = r1 - r0 + 1, c1 - c0 + 1
    cands = []
    ys = range(1, bh) if axis in ("both", "y") else [max(bh, 1)]
    xs = range(1, bw) if axis in ("both", "x") else [max(bw, 1)]
    for py in ys:
        for px in xs:
            cands.append((py * px, py, px))
    cands.sort()
    for _a, py, px in cands:
        table = {}
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                k = ((r - r0) % py, (c - c0) % px)
                v = g[r][c]
                if table.setdefault(k, v) != v:
                    ok = False
                    break
            if not ok:
                break
        if not ok or len(table) < py * px:
            continue
        out, changed = [], False
        for r in range(h):
            row = []
            in_band = r0 <= r <= r1
            for c in range(w):
                inside = in_band and c0 <= c <= c1
                # a horizontal continuation says nothing about rows the pattern
                # never occupied, and a vertical one says nothing about columns
                off_axis = ((axis == "x" and not in_band)
                            or (axis == "y" and not (c0 <= c <= c1)))
                if inside or off_axis:
                    row.append(g[r][c])
                    continue
                v = table[((r - r0) % py, (c - c0) % px)]
                if v != g[r][c]:
                    changed = True
                row.append(v)
            out.append(tuple(row))
        if changed:
            return tuple(out)
    return None


def _periodic_fill_auto(g, bg, axis, only_bg):
    got = _best_period(g, bg, axis)
    if got is None:
        return None
    return _periodic_fill(g, bg, got[0], got[1], only_bg)


def _periodic_fill(g, bg, py, px, only_bg):
    table = _consistent_period(g, bg, py, px)
    if table is None:
        return None
    h, w = G.dims(g)
    out = []
    for r in range(h):
        row = []
        for c in range(w):
            v = g[r][c]
            if only_bg and v != bg:
                row.append(v)
            else:
                row.append(table.get((r % py, c % px), v))
        out.append(tuple(row))
    return tuple(out)


def _best_period(g, bg, axis):
    """Smallest period consistent with the cells that are actually there.

    Inferred per grid, not fixed by the task: the point of a periodic rule is
    that the period is a property of the grid in front of you, and in real
    tasks it changes from one example to the next.
    """
    h, w = G.dims(g)
    cands = []
    ys = range(1, h + 1) if axis in ("both", "y") else [h]
    xs = range(1, w + 1) if axis in ("both", "x") else [w]
    for py in ys:
        for px in xs:
            if py == h and px == w:
                continue
            cands.append((py * px, py, px))
    cands.sort()
    for _a, py, px in cands:
        table = _consistent_period(g, bg, py, px)
        if table is None:
            continue
        if len(table) < py * px:
            continue          # the period itself has gaps: nothing to copy from
        return py, px
    return None


# --- reflective / wrapping extension ---------------------------------------

def _mirror_index(i, n):
    if n <= 1:
        return 0
    period = 2 * n - 2
    j = i % period
    return j if j < n else period - j


def _extend(g, H, W, mode_y, mode_x):
    h, w = G.dims(g)
    if H * W > _MAX or H <= 0 or W <= 0:
        return None
    rows = []
    for i in range(H):
        r = _mirror_index(i, h) if mode_y == "mirror" else i % h
        src = g[r]
        rows.append(tuple(src[_mirror_index(j, w) if mode_x == "mirror" else j % w]
                          for j in range(W)))
    return tuple(rows)


def _target(ctx, g):
    h, w = G.dims(g)
    if ctx.const_out_shape:
        return ctx.const_out_shape
    if ctx.shape_ratio:
        ky, kx = ctx.shape_ratio
        return (h * ky, w * kx)
    law = ctx.affine_shape
    if law:
        (ay, by), (ax, bx) = law
        H, W = ay * h + by, ax * w + bx
        if H > 0 and W > 0:
            return (H, W)
    return None


def generate(ctx):
    res, seen = [], set()

    def offer(name, fn, cost):
        if len(res) >= 18:
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

    bg = ctx.bg
    if ctx.same_shape:
        for axis in ("both", "x", "y"):
            offer("continue[%s]" % axis,
                  (lambda ax: lambda g: _continue_outward(
                      g, G.bg_or(g, bg), ax))(axis),
                  2.5 + (0.2 if axis == "both" else 0.0))
        for axis in ("both", "x", "y"):
            for only_bg in (True, False):
                offer("period[%s%s]" % (axis, "/bg" if only_bg else ""),
                      (lambda ax, ob: lambda g: _periodic_fill_auto(
                          g, G.bg_or(g, bg), ax, ob))(axis, only_bg),
                      2.6 + (0.2 if axis == "both" else 0.0))

    if not ctx.same_shape:
        for my in ("mirror", "wrap"):
            for mx in ("mirror", "wrap"):
                offer("ext[%s/%s]" % (my, mx),
                      (lambda my, mx: lambda g: (
                          lambda d: _extend(g, d[0], d[1], my, mx)
                          if d else None)(_target(ctx, g)))(my, mx),
                      2.8 + (0.0 if my == mx else 0.3))
    return res
