"""Panel abstraction: solve the *grid of panels* as a task in its own right.

When a grid is a lattice of equal panels, the rule almost never operates on
pixels -- it operates on panels. "Every panel in the same row as a filled one
gets filled", "empty panels become the marker shape", "the odd panel out is
cleared". At pixel resolution those are long-range rules that no local family
can see; at panel resolution they are three-by-three grid transformations that
the portfolio already solves easily.

So: abstract each panel to a single cell, hand the resulting small task back to
the portfolio, and render the answer by stamping the learned patch for each
predicted panel code. The abstraction is lossy by design; the render table is
what puts the detail back, and it is learned from the training *outputs*, so
the stamped content can differ from anything in the input.
"""

import time

from collections import Counter

from .. import grid as G
from .. import hypcache
from ..task import Ctx, Hyp
from .partition import _decompositions, sep_color_of

SOLVER = "partition"
PHASE = 2


def _code(panel, bg):
    """One colour standing for a whole panel: its dominant non-background."""
    hist = G.histogram(panel)
    hist.pop(bg, None)
    if not hist:
        return bg
    return max(hist.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _abstract(mat, bg, role=False):
    """Panel matrix -> small grid.

    ``role`` collapses every non-empty panel to 1, which is what tasks about
    *which* panels are occupied need; the raw form keeps the panel's colour,
    which is what tasks about panel identity need.  Neither subsumes the other.
    """
    out = []
    for row in mat:
        r = []
        for p in row:
            if p is None:
                return None
            k = _code(p, bg)
            r.append((0 if k == bg else 1) if role else k)
        out.append(tuple(r))
    return tuple(out)


def _fill_color(g, bg, mode):
    if mode == "sep":
        return sep_color_of(g)
    if mode == "motif":
        hist = G.histogram(g)
        hist.pop(bg, None)
        sep = sep_color_of(g)
        hist.pop(sep, None)
        if not hist:
            return None
        return min(hist.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return mode


def _render_table(ctx, dec, bg):
    """code -> panel patch, learned from the training outputs."""
    table = {}
    shape = None
    for _a, b in ctx.train:
        mat = dec(b)
        if not mat:
            return None
        for row in mat:
            for p in row:
                if p is None:
                    return None
                if shape is None:
                    shape = G.dims(p)
                elif G.dims(p) != shape:
                    return None
                k = _code(p, bg)
                prev = table.get(k)
                if prev is None:
                    table[k] = p
                elif prev != p:
                    return None
    return table if table else None


def _render_shape(ctx, dec, bg):
    """A single stencil shared by every non-empty output panel.

    A literal code -> patch table cannot survive a task that draws the same
    motif in a different colour in each grid: the code changes, so the table
    fragments.  Learning the *shape* once and painting it in whatever colour
    the abstract answer names is what makes the render transfer.
    """
    mask = None
    shape = None
    for _a, b in ctx.train:
        mat = dec(b)
        if not mat:
            return None
        for row in mat:
            for p in row:
                if p is None:
                    return None
                if shape is None:
                    shape = G.dims(p)
                elif G.dims(p) != shape:
                    return None
                cols = G.palette(p) - {bg}
                if not cols:
                    continue
                if len(cols) > 1:
                    return None
                m = tuple(tuple(1 if v != bg else 0 for v in r) for r in p)
                if mask is None:
                    mask = m
                elif mask != m:
                    return None
    return mask


def _positions(g, dec_name, dec):
    """Top-left corner of each panel, recovered by matching the decomposition."""
    mat = dec(g)
    if not mat:
        return None
    h, w = G.dims(g)
    ph, pw = G.dims(mat[0][0]) if mat[0][0] else (0, 0)
    if ph == 0 or pw == 0:
        return None
    nr, nc = len(mat), len(mat[0])
    rows, cols = [], []
    # locate each panel by scanning for the first row/col offset that matches
    r = 0
    for i in range(nr):
        while r + ph <= h and tuple(x[:pw] for x in g[r:r + ph]) != mat[i][0]:
            r += 1
        if r + ph > h:
            return None
        rows.append(r)
        r += ph
    c = 0
    for j in range(nc):
        while c + pw <= w and tuple(x[c:c + pw] for x in g[rows[0]:rows[0] + ph]) != mat[0][j]:
            c += 1
        if c + pw > w:
            return None
        cols.append(c)
        c += pw
    return rows, cols, ph, pw


class _PanelRule:
    __slots__ = ("dec", "bg", "hyp", "table", "mask", "role", "cmode")

    def __init__(self, dec, bg, hyp, table, mask=None, role=False, cmode=None):
        self.dec = dec
        self.bg = bg
        self.hyp = hyp
        self.table = table
        self.mask = mask
        self.role = role
        self.cmode = cmode

    def _patch(self, code, ph, pw, fill=None):
        if self.role:
            if self.mask is None or len(self.mask) != ph or len(self.mask[0]) != pw:
                return None
            if not code:
                return G.const_grid(ph, pw, self.bg)
            if fill is None:
                return None
            return tuple(tuple(fill if v else self.bg for v in r)
                         for r in self.mask)
        if self.mask is not None:
            if len(self.mask) != ph or len(self.mask[0]) != pw:
                return None
            if code == self.bg:
                return G.const_grid(ph, pw, self.bg)
            return tuple(tuple(code if v else self.bg for v in r)
                         for r in self.mask)
        return self.table.get(code)

    def __call__(self, g):
        mat = self.dec(g)
        if not mat:
            return None
        small = _abstract(mat, self.bg, self.role)
        if small is None:
            return None
        pred = self.hyp.fn(small)
        if pred is None or not G.valid(pred):
            return None
        if G.dims(pred) != (len(mat), len(mat[0])):
            return None
        pos = _positions(g, None, self.dec)
        if pos is None:
            return None
        rows, cols, ph, pw = pos
        fill = _fill_color(g, self.bg, self.cmode) if self.role else None
        if self.role and fill is None:
            return None
        out = [list(r) for r in g]
        for i, r0 in enumerate(rows):
            for j, c0 in enumerate(cols):
                patch = self._patch(pred[i][j], ph, pw, fill)
                if patch is None:
                    return None
                if G.dims(patch) != (ph, pw):
                    return None
                for rr in range(ph):
                    prow = patch[rr]
                    orow = out[r0 + rr]
                    for cc in range(pw):
                        orow[c0 + cc] = prow[cc]
        return tuple(tuple(r) for r in out)


def _modules():
    from ..solvers import (cellwise, colormap, geometry, objects_map, partition,
                           sequence, symmetry, tiling)
    return (geometry, colormap, cellwise, symmetry, tiling, partition,
            objects_map, sequence)


def generate(ctx):
    if not ctx.same_shape:
        return []
    deadline = ctx.deadline or (time.time() + 4.0)
    bg = ctx.bg
    res = []
    decs = _decompositions(ctx)[:4]
    for dname, dcost, dec in decs:
        for role in (False, True):
            if time.time() > deadline or len(res) > 30:
                break
            res.extend(_one(ctx, dname, dcost, dec, bg, role, deadline))
    return res


def _one(ctx, dname, dcost, dec, bg, role, deadline):
    res = []
    try:
        pairs = []
        for a, b in ctx.train:
            ma, mb = dec(a), dec(b)
            if not ma or not mb or len(ma) != len(mb) or len(ma[0]) != len(mb[0]):
                return res
            sa, sb = _abstract(ma, bg, role), _abstract(mb, bg, role)
            if sa is None or sb is None or sa == sb:
                return res
            pairs.append((G.to_list(sa), G.to_list(sb)))
        if not pairs:
            return res
        table = None if role else _render_table(ctx, dec, bg)
        mask = _render_shape(ctx, dec, bg)
        if table is None and mask is None:
            return res
        tins = []
        for t in ctx.test_inputs:
            m = dec(t)
            if not m:
                return res
            s = _abstract(m, bg, role)
            if s is None:
                return res
            tins.append(G.to_list(s))
    except Exception:
        return res

    sub = Ctx(pairs, tins)
    sub.deadline = min(deadline, time.time() + 1.2)
    variants = []
    if role:
        if mask is not None:
            for cm in ("sep", "motif") + tuple(sorted(ctx.out_palette))[:4]:
                variants.append(("role:%s" % cm, None, mask, True, cm))
    else:
        if table is not None:
            variants.append(("tbl", table, None, False, None))
        if mask is not None:
            variants.append(("shp", None, mask, False, None))
    if not variants:
        return res
    for mod in _modules():
        if time.time() > sub.deadline:
            break
        try:
            hyps = list(hypcache.generate(mod, sub))
        except Exception:
            continue
        hit = None
        for hp in hyps:
            if hp.fits(sub.train):
                hit = hp
                break
        if hit is None:
            continue
        for rname, tb, mk, rl, cm in variants:
            res.append(Hyp("panel[%s|%s]>>%s" % (dname, rname, hit.name),
                           _PanelRule(dec, bg, hit, tb, mk, rl, cm),
                           dcost + 2.0 + hit.cost, SOLVER))
        if len(res) > 30:
            break
    return res
