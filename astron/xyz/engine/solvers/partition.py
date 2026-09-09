"""Grid decomposition into panels, and rules defined over the panel set.

Two decompositions are supported:

* **separator** -- full-width/height uniform lines of one colour cut the grid
  into a matrix of panels (the classic "two boards divided by a blue line");
* **even split** -- no separator, but the grid divides exactly into k x m
  congruent panels.

Given panels, the rules are: pick one (by a relational selector), combine them
all cellwise (the logic-op family), or reduce each panel to a single cell.
"""

from collections import Counter

from .. import grid as G
from ..task import Hyp

SOLVER = "partition"


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


# --------------------------------------------------------------------------
# decomposition
# --------------------------------------------------------------------------

def _sep_lines(g, color):
    h, w = G.dims(g)
    rows = [r for r in range(h) if all(v == color for v in g[r])]
    cols = [c for c in range(w) if all(g[r][c] == color for r in range(h))]
    return rows, cols


def _runs(marks, n):
    """Maximal index ranges not covered by ``marks``."""
    s = set(marks)
    out = []
    cur = None
    for i in range(n):
        if i in s:
            if cur is not None:
                out.append((cur, i - 1))
                cur = None
        else:
            if cur is None:
                cur = i
    if cur is not None:
        out.append((cur, n - 1))
    return out


def panels_by_separator(g, color):
    h, w = G.dims(g)
    rows, cols = _sep_lines(g, color)
    if not rows and not cols:
        return None
    rr = _runs(rows, h)
    cc = _runs(cols, w)
    if not rr or not cc or (len(rr) == 1 and len(cc) == 1):
        return None
    if len(rr) * len(cc) > 64:
        return None
    return [[G.subgrid(g, a, c, b, d) for c, d in cc] for a, b in rr]


def panels_even(g, ky, kx):
    h, w = G.dims(g)
    if h % ky or w % kx or (ky == 1 and kx == 1):
        return None
    ph, pw = h // ky, w // kx
    return [[G.subgrid(g, i * ph, j * pw, (i + 1) * ph - 1, (j + 1) * pw - 1)
             for j in range(kx)] for i in range(ky)]


def sep_color_of(g):
    """The colour ``panels_auto`` would use to cut this grid."""
    best = None
    for c in G.palette(g):
        rows, cols = _sep_lines(g, c)
        if not rows and not cols:
            continue
        if not panels_by_separator(g, c):
            continue
        score = (len(rows) + len(cols), -c)
        if best is None or score > best[0]:
            best = (score, c)
    return best[1] if best else None


def panels_auto(g):
    """Decompose using whichever colour rules this grid, chosen per grid.

    The separator colour is not always shared across a task's examples -- the
    same lattice can be drawn in a different colour in every grid -- so a
    task-wide separator colour finds nothing at all on those tasks.
    """
    best = None
    for c in G.palette(g):
        rows, cols = _sep_lines(g, c)
        if not rows and not cols:
            continue
        mat = panels_by_separator(g, c)
        if not mat:
            continue
        score = (len(rows) + len(cols), -c)
        if best is None or score > best[0]:
            best = (score, mat)
    return best[1] if best else None


def _flat(mat):
    return [p for row in mat for p in row if p is not None]


def _sep_color_candidates(ctx):
    """Colours that cut *every* training input into >1 panel."""
    cands = []
    pal = set(G.palette(ctx.train[0][0]))
    for a in ctx.all_inputs:
        pal &= G.palette(a)
    for c in sorted(pal):
        if all(panels_by_separator(a, c) for a in ctx.all_inputs):
            cands.append(c)
    return cands


def _decompositions(ctx):
    """List of (name, cost, fn(grid) -> panel matrix)."""
    out = []
    for c in _sep_color_candidates(ctx):
        out.append(("sep#%d" % c, 3.0,
                    (lambda c: lambda g: panels_by_separator(g, c))(c)))
    if all(panels_auto(a) for a in ctx.all_inputs):
        out.append(("sep_auto", 3.2, panels_auto))
    shapes = {G.dims(a) for a in ctx.all_inputs}
    for ky in (1, 2, 3, 4):
        for kx in (1, 2, 3, 4):
            if ky == 1 and kx == 1:
                continue
            if all(h % ky == 0 and w % kx == 0 for h, w in shapes):
                out.append(("even%dx%d" % (ky, kx), 3.5,
                            (lambda ky, kx: lambda g: panels_even(g, ky, kx))(ky, kx)))
    return out


# --------------------------------------------------------------------------
# panel selectors
# --------------------------------------------------------------------------

def _panel_scores(ps, bg):
    return {
        "ncolors": [len(G.palette(p)) for p in ps],
        "nnz": [sum(1 for r in p for v in r if v != bg) for p in ps],
        "nzero": [sum(1 for r in p for v in r if v == bg) for p in ps],
        "nobj": [len(G.flood_regions(p, bg, True, True)) for p in ps],
        "nsym": [len(G.symmetries(p)) for p in ps],
    }


def _select(ps, how, bg):
    if not ps:
        return None
    if how == "unique":
        cnt = Counter(ps)
        hits = [p for p in ps if cnt[p] == 1]
        return hits[0] if len(hits) == 1 else None
    if how == "majority":
        cnt = Counter(ps)
        k, n = cnt.most_common(1)[0]
        return k if n > 1 else None
    if how == "unique_shapewise":
        norm = [G.dedup(p) for p in ps]
        cnt = Counter(norm)
        hits = [p for p, nm in zip(ps, norm) if cnt[nm] == 1]
        return hits[0] if len(hits) == 1 else None
    sc = _panel_scores(ps, bg)
    for key, vals in sc.items():
        if how == "max_" + key or how == "min_" + key:
            tgt = max(vals) if how.startswith("max") else min(vals)
            hits = [p for p, v in zip(ps, vals) if v == tgt]
            return hits[0] if len(hits) == 1 else None
    return None


_SELECTORS = (["unique", "majority", "unique_shapewise"] +
              ["%s_%s" % (d, k) for k in
               ("ncolors", "nnz", "nzero", "nobj", "nsym") for d in ("max", "min")])


# --------------------------------------------------------------------------
# cellwise panel combination
# --------------------------------------------------------------------------

_LOGIC = {
    "and": lambda n, t: n == t,
    "or": lambda n, t: n > 0,
    "xor": lambda n, t: n == 1,
    "nand": lambda n, t: n < t,
    "nor": lambda n, t: n == 0,
    "xnor": lambda n, t: n != 1,
    "majority": lambda n, t: 2 * n > t,
    "exact2": lambda n, t: n == 2,
}


def _combine(ps, bg, op, on, off):
    if len(ps) < 2:
        return None
    d = G.dims(ps[0])
    for p in ps[1:]:
        if G.dims(p) != d:
            return None
    h, w = d
    f = _LOGIC[op]
    t = len(ps)
    out = []
    for r in range(h):
        row = []
        for c in range(w):
            n = sum(1 for p in ps if p[r][c] != bg)
            row.append(on if f(n, t) else off)
        out.append(tuple(row))
    return tuple(out)


def _overlay(ps, bg, order):
    if len(ps) < 2:
        return None
    d = G.dims(ps[0])
    if any(G.dims(p) != d for p in ps):
        return None
    seq = ps if order else ps[::-1]
    base = seq[0]
    for p in seq[1:]:
        base = G.paste_masked(base, p, 0, 0, bg)
    return base


def _reduce_cells(mat, bg, mode):
    out = []
    for row in mat:
        orow = []
        for p in row:
            if p is None:
                return None
            hist = G.histogram(p)
            if mode == "mode":
                orow.append(max(hist.items(), key=lambda kv: (kv[1], -kv[0]))[0])
            elif mode == "nonbg":
                nb = {k: v for k, v in hist.items() if k != bg}
                if not nb:
                    orow.append(bg)
                elif len(nb) == 1:
                    orow.append(next(iter(nb)))
                else:
                    orow.append(max(nb.items(), key=lambda kv: (kv[1], -kv[0]))[0])
            elif mode == "any":
                orow.append(1 if any(k != bg for k in hist) else 0)
            else:
                return None
        out.append(tuple(orow))
    return tuple(out)


# --------------------------------------------------------------------------

def generate(ctx):
    res = []
    bg = ctx.bg
    decs = _decompositions(ctx)
    if not decs:
        return res
    colors = sorted(ctx.out_palette | {bg, 0})
    for dname, dcost, dec in decs[:10]:
        # (a) fixed index
        try:
            m0 = dec(ctx.train[0][0])
        except Exception:
            continue
        if not m0:
            continue
        nr, nc = len(m0), len(m0[0])
        if nr * nc <= 12:
            for i in range(nr):
                for j in range(nc):
                    res.append(_h("%s.at%d,%d" % (dname, i, j),
                                  (lambda dec, i, j: lambda g: _at(dec(g), i, j))(dec, i, j),
                                  dcost + 2.0))
        # (b) relational selection
        for how in _SELECTORS:
            res.append(_h("%s.sel_%s" % (dname, how),
                          (lambda dec, how, bg: lambda g: _select(_flat(dec(g)), how, bg))(dec, how, bg),
                          dcost + 3.0))
        # (c) cellwise logic
        for op in _LOGIC:
            for on in colors:
                res.append(_h("%s.%s->%d" % (dname, op, on),
                              (lambda dec, op, on, bg: lambda g: _combine(_flat(dec(g)), bg, op, on, bg))(dec, op, on, bg),
                              dcost + 2.5))
        for order in (True, False):
            res.append(_h("%s.overlay%d" % (dname, order),
                          (lambda dec, o, bg: lambda g: _overlay(_flat(dec(g)), bg, o))(dec, order, bg),
                          dcost + 3.0))
        # (d) panel -> cell reduction
        for mode in ("mode", "nonbg", "any"):
            res.append(_h("%s.reduce_%s" % (dname, mode),
                          (lambda dec, m, bg: lambda g: _reduce_cells(dec(g), bg, m))(dec, mode, bg),
                          dcost + 3.5))
        # (e) panel matrix -> per-panel recolour by identity of panel
        res.append(_h("%s.panel_id" % dname,
                      (lambda dec, bg: lambda g: _reduce_cells(dec(g), bg, "nonbg"))(dec, bg),
                      dcost + 4.5))
    return res


def _at(mat, i, j):
    if not mat or i >= len(mat) or j >= len(mat[0]):
        return None
    return mat[i][j]
