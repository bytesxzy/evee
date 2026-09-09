"""Learned block dictionaries: cell <-> patch and panel <-> panel.

Three closely related rules, all learned as *compressing* lookup tables:

``blockmap``   every input cell expands into a k x m patch chosen by its colour;
``blockfold``  every k x m block of the input collapses to one output cell;
``panelmap``   the grid decomposes into panels and each panel is rewritten by
               a table keyed on the panel's own content.

A table is only accepted if it *compresses* -- at least two observations per
entry on average.  "Fewer entries than observations" is far too weak a bar: a
row-level table over thirty-row grids passes it while still being a transcript
of the training data, and it then outranks the family that actually knows the
rule.  Where the evidence is thin the table must also survive refitting with
one training pair withheld.
"""

from collections import Counter

from .. import grid as G
from ..task import Hyp
from .partition import _decompositions, panels_by_separator

SOLVER = "tiling"


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


# --------------------------------------------------------------------------
# cell -> patch
# --------------------------------------------------------------------------

def _fit_blockmap(ctx, ky, kx, keyed_on_shape):
    table = {}
    n_obs = 0
    for a, b in ctx.train:
        ah, aw = G.dims(a)
        if G.dims(b) != (ah * ky, aw * kx):
            return None
        for r in range(ah):
            for c in range(aw):
                patch = tuple(row[c * kx:(c + 1) * kx]
                              for row in b[r * ky:(r + 1) * ky])
                k = a[r][c] if not keyed_on_shape else (a[r][c], r % 2, c % 2)
                prev = table.get(k)
                if prev is None:
                    table[k] = patch
                elif prev != patch:
                    return None
                n_obs += 1
    if not table or len(table) * 2 > n_obs:
        return None

    def run(g, t=table, ky=ky, kx=kx, ks=keyed_on_shape):
        h, w = G.dims(g)
        if h * ky > 60 or w * kx > 60:
            return None
        out = [[None] * (w * kx) for _ in range(h * ky)]
        for r in range(h):
            for c in range(w):
                p = t.get(g[r][c] if not ks else (g[r][c], r % 2, c % 2))
                if p is None:
                    return None
                for i in range(ky):
                    orow = out[r * ky + i]
                    prow = p[i]
                    for j in range(kx):
                        orow[c * kx + j] = prow[j]
        return tuple(tuple(row) for row in out)
    return run


# --------------------------------------------------------------------------
# patch -> cell
# --------------------------------------------------------------------------

def _fit_blockfold(ctx, ky, kx):
    table = {}
    n_obs = 0
    for a, b in ctx.train:
        ah, aw = G.dims(a)
        bh, bw = G.dims(b)
        if ah != bh * ky or aw != bw * kx:
            return None
        for r in range(bh):
            for c in range(bw):
                patch = tuple(row[c * kx:(c + 1) * kx]
                              for row in a[r * ky:(r + 1) * ky])
                prev = table.get(patch)
                if prev is None:
                    table[patch] = b[r][c]
                elif prev != b[r][c]:
                    return None
                n_obs += 1
    if not table or len(table) * 2 > n_obs:
        return None

    def run(g, t=table, ky=ky, kx=kx):
        h, w = G.dims(g)
        if h % ky or w % kx:
            return None
        out = []
        for r in range(h // ky):
            row = []
            for c in range(w // kx):
                patch = tuple(rr[c * kx:(c + 1) * kx]
                              for rr in g[r * ky:(r + 1) * ky])
                v = t.get(patch)
                if v is None:
                    return None
                row.append(v)
            out.append(tuple(row))
        return tuple(out)
    return run


# --------------------------------------------------------------------------
# panel -> panel / panel -> cell
# --------------------------------------------------------------------------

def _fit_panelmap(ctx, dec, to_cell):
    table = {}
    n_obs = 0
    for a, b in ctx.train:
        mat = dec(a)
        if not mat:
            return None
        nr, nc = len(mat), len(mat[0])
        if to_cell:
            if G.dims(b) != (nr, nc):
                return None
            for i in range(nr):
                for j in range(nc):
                    p = mat[i][j]
                    if p is None:
                        return None
                    prev = table.get(p)
                    if prev is None:
                        table[p] = b[i][j]
                    elif prev != b[i][j]:
                        return None
                    n_obs += 1
        else:
            omat = dec(b)
            if not omat or len(omat) != nr or len(omat[0]) != nc:
                return None
            for i in range(nr):
                for j in range(nc):
                    p, q = mat[i][j], omat[i][j]
                    if p is None or q is None:
                        return None
                    prev = table.get(p)
                    if prev is None:
                        table[p] = q
                    elif prev != q:
                        return None
                    n_obs += 1
    if not table or len(table) * 2 > n_obs:
        return None

    def run(g, t=table, dec=dec, to_cell=to_cell):
        mat = dec(g)
        if not mat:
            return None
        if to_cell:
            out = []
            for row in mat:
                orow = []
                for p in row:
                    v = t.get(p)
                    if v is None:
                        return None
                    orow.append(v)
                out.append(tuple(orow))
            return tuple(out)
        rows = []
        for row in mat:
            band = None
            for p in row:
                q = t.get(p)
                if q is None:
                    return None
                band = q if band is None else G.hconcat(band, q)
            if band is None:
                return None
            rows.append(band)
        res = rows[0]
        for r in rows[1:]:
            res = G.vconcat(res, r)
        return res
    return run


# --------------------------------------------------------------------------
# row / column dictionaries
# --------------------------------------------------------------------------

def _linemap_table(train, axis):
    table = {}
    n = 0
    for a, b in train:
        if G.dims(a) != G.dims(b):
            return None, 0
        ra = a if axis == 0 else G.transpose(a)
        rb = b if axis == 0 else G.transpose(b)
        for x, y in zip(ra, rb):
            prev = table.get(x)
            if prev is None:
                table[x] = y
            elif prev != y:
                return None, 0
            n += 1
    return (table or None), n


def _fit_linemap(ctx, axis):
    """Learn a lookup from a whole row (or column) to its replacement.

    Rows are a natural unit for a surprising number of tasks -- striping,
    row-wise recolouring, sorting -- and a row-level table compresses far
    harder than a cell-level one, so the capacity guard passes on evidence a
    per-cell rule would have to memorise.
    """
    table, n = _linemap_table(ctx.train, axis)
    if table is None or len(table) * 2 > n:
        return None
    # A row dictionary generalises only if rows recur across grids; refitting
    # without a pair is the direct test of that.
    if len(ctx.train) >= 3:
        ok = 0
        for i in range(len(ctx.train)):
            sub = ctx.train[:i] + ctx.train[i + 1:]
            t2, _n2 = _linemap_table(sub, axis)
            if t2 is None:
                continue
            a, b = ctx.train[i]
            ra = a if axis == 0 else G.transpose(a)
            rb = b if axis == 0 else G.transpose(b)
            if all(t2.get(x) == y for x, y in zip(ra, rb)):
                ok += 1
        if ok * 2 < len(ctx.train):
            return None

    def run(g, t=table, axis=axis):
        rows = g if axis == 0 else G.transpose(g)
        out = []
        for x in rows:
            y = t.get(x)
            if y is None:
                return None
            out.append(y)
        res = tuple(out)
        return res if axis == 0 else G.transpose(res)
    return run


def _drop_lines(g, axis, mode, bg):
    """Delete rows (or columns) that are empty, or uniform, or duplicated."""
    bg = G.bg_or(g, bg)
    rows = g if axis == 0 else G.transpose(g)
    keep = []
    seen = set()
    for r in rows:
        if mode == "empty" and all(v == bg for v in r):
            continue
        if mode == "uniform" and len(set(r)) == 1:
            continue
        if mode == "dup":
            if r in seen:
                continue
            seen.add(r)
        keep.append(r)
    if not keep or len(keep) == len(rows):
        return None
    res = tuple(keep)
    return res if axis == 0 else G.transpose(res)


# --------------------------------------------------------------------------

def generate(ctx):
    res = []
    r = ctx.shape_ratio
    if r and r != (1, 1) and r[0] * r[1] <= 36:
        for keyed in (False, True):
            f = _fit_blockmap(ctx, r[0], r[1], keyed)
            if f is not None:
                res.append(_h("blockmap%dx%d%s" % (r[0], r[1],
                                                   "_p" if keyed else ""),
                              f, 3.0))
    ir = ctx.inv_shape_ratio
    if ir and ir != (1, 1) and ir[0] * ir[1] <= 64:
        f = _fit_blockfold(ctx, ir[0], ir[1])
        if f is not None:
            res.append(_h("blockfold%dx%d" % ir, f, 3.0))

    for axis in (0, 1):
        f = _fit_linemap(ctx, axis)
        if f is not None:
            res.append(_h("linemap%d" % axis, f, 5.0))
        for mode in ("empty", "uniform", "dup"):
            res.append(_h("drop_%s%d" % (mode, axis),
                          (lambda a, m, bg: lambda g: _drop_lines(g, a, m, bg))(axis, mode, ctx.bg),
                          3.2))
    for dname, dcost, dec in _decompositions(ctx)[:6]:
        for to_cell in (True, False):
            try:
                f = _fit_panelmap(ctx, dec, to_cell)
            except Exception:
                f = None
            if f is not None:
                res.append(_h("panelmap_%s_%s" % (dname, "cell" if to_cell else "grid"),
                              f, dcost + 1.5))
    return res
