"""Panel stacks combined by a *learned* table rather than a fixed logic op.

``partition`` can already fold a grid's panels together, but only through a
closed list of boolean operators -- and, or, xor, majority -- with the "on" and
"off" colours picked from a short list.  That covers the classic two-panel
puzzles and nothing else.  The moment a task says "where the first panel is red
and the second is blue, write green; where both are empty, write grey", or
stacks four panels instead of two, the family is silent.

The generalisation is to stop naming the operator and learn it.  For every cell
position, the stack of panels supplies a *key* -- the tuple of colours at that
position, or an order-insensitive reading of it -- and the training outputs
supply the value.  If one conflict-free table explains every cell of every
training pair, that table *is* the operator, and it can express all eight
boolean ops, colour-preserving overlays, priority rules and arbitrary
combinations besides.

Several keys are offered because the right invariance differs by task:

``ordered``   which panel a colour came from matters;
``sorted``    only the multiset of colours matters (panel order is irrelevant);
``set``       only which colours are present;
``count``     only how many panels are non-empty there;
``count1``    how many are non-empty, plus which colour they carried.

The guard against memorisation is compression, the same rule ``blocks`` uses: a
table is rejected unless it sees, on average, at least two observations per
entry.  A table with one row per cell is a transcript of the training outputs,
and it would otherwise outrank the family that actually knows the rule.

A second family here works one level up.  Instead of folding the panels
together cell by cell, it *classifies* each panel and writes a learned block for
it: "this shape means red, that shape means blue".  The blocks are then laid out
either the way the panels were, or transposed, or strung into a single row or
column -- ARC uses all four, and which one it is cannot be read off a single
panel.  Every entry of the classification table must have been demonstrated at
least twice, so a panel shape seen once cannot carry an answer.
"""

from collections import Counter

from .. import grid as G
from ..task import Hyp
from .partition import _decompositions

SOLVER = "partition"

_MIN_OBS_PER_ENTRY = 2.0


def _flat(mat):
    return [p for row in mat for p in row]


def _keyfns(bg):
    def ordered(vals):
        return tuple(vals)

    def srt(vals):
        return tuple(sorted(vals))

    def st(vals):
        return frozenset(vals)

    def count(vals):
        return sum(1 for v in vals if v != bg)

    def count1(vals):
        nz = [v for v in vals if v != bg]
        return (len(nz), nz[0] if nz else bg)

    def first(vals):
        for v in vals:
            if v != bg:
                return v
        return bg

    return [("ordered", 0.0, ordered), ("sorted", 0.4, srt),
            ("set", 0.6, st), ("count", 0.8, count),
            ("count1", 0.7, count1), ("first", 0.9, first)]


class _TableRule:
    __slots__ = ("dec", "keyfn", "table", "bg", "default")

    def __init__(self, dec, keyfn, table, bg, default):
        self.dec = dec
        self.keyfn = keyfn
        self.table = table
        self.bg = bg
        self.default = default

    def __call__(self, g):
        try:
            mat = self.dec(g)
        except Exception:
            return None
        if not mat:
            return None
        ps = _flat(mat)
        if len(ps) < 2:
            return None
        d = G.dims(ps[0])
        if any(G.dims(p) != d for p in ps):
            return None
        h, w = d
        bg = self.bg
        out = []
        for r in range(h):
            row = []
            for c in range(w):
                k = self.keyfn(tuple(p[r][c] for p in ps))
                v = self.table.get(k, self.default)
                if v is None:
                    return None
                row.append(v)
            out.append(tuple(row))
        return tuple(out)


def _observe(ctx, dec):
    """Per-pair panel stacks aligned with their outputs, or ``None``."""
    rows = []
    for a, b in ctx.train:
        try:
            mat = dec(a)
        except Exception:
            return None
        if not mat:
            return None
        ps = _flat(mat)
        if len(ps) < 2:
            return None
        d = G.dims(ps[0])
        if any(G.dims(p) != d for p in ps) or d != G.dims(b):
            return None
        rows.append((ps, b, d))
    if not rows:
        return None
    if len({len(ps) for ps, _b, _d in rows}) != 1:
        return None
    return rows


def _fit(rows, keyfn, bg):
    table = {}
    obs = 0
    for ps, b, (h, w) in rows:
        for r in range(h):
            brow = b[r]
            for c in range(w):
                k = keyfn(tuple(p[r][c] for p in ps))
                v = brow[c]
                if table.setdefault(k, v) != v:
                    return None
                obs += 1
    if not table:
        return None
    if obs < _MIN_OBS_PER_ENTRY * len(table):
        return None                       # a transcript, not a rule
    return table, obs


_LAYOUTS = ("same", "transpose", "column", "row")


def _panel_key(panel, bg, kind):
    if kind == "content":
        return panel
    if kind == "mask":
        return tuple(tuple(1 if v != bg else 0 for v in row) for row in panel)
    if kind == "count":
        return sum(1 for row in panel for v in row if v != bg)
    if kind == "palette":
        return tuple(sorted({v for row in panel for v in row}))
    return None


def _layout_dims(mat, layout):
    rows, cols = len(mat), len(mat[0])
    if layout == "same":
        return rows, cols
    if layout == "transpose":
        return cols, rows
    if layout == "column":
        return rows * cols, 1
    return 1, rows * cols


def _layout_order(mat, layout):
    rows, cols = len(mat), len(mat[0])
    if layout == "transpose":
        return [mat[r][c] for c in range(cols) for r in range(rows)]
    return [mat[r][c] for r in range(rows) for c in range(cols)]


class _DictRule:
    __slots__ = ("dec", "kind", "layout", "table", "bg", "bh", "bw")

    def __init__(self, dec, kind, layout, table, bg, bh, bw):
        self.dec = dec
        self.kind = kind
        self.layout = layout
        self.table = table
        self.bg = bg
        self.bh = bh
        self.bw = bw

    def __call__(self, g):
        try:
            mat = self.dec(g)
        except Exception:
            return None
        if not mat or not mat[0]:
            return None
        lr, lc = _layout_dims(mat, self.layout)
        seq = _layout_order(mat, self.layout)
        if len(seq) != lr * lc:
            return None
        out = [[self.bg] * (lc * self.bw) for _ in range(lr * self.bh)]
        for i, panel in enumerate(seq):
            block = self.table.get(_panel_key(panel, self.bg, self.kind))
            if block is None:
                return None
            r0 = (i // lc) * self.bh
            c0 = (i % lc) * self.bw
            for r in range(self.bh):
                for c in range(self.bw):
                    out[r0 + r][c0 + c] = block[r][c]
        return tuple(tuple(r) for r in out)


def _dict_rules(ctx, dname, dcost, dec, bg, res, seen):
    obs = []
    for a, b in ctx.train:
        try:
            mat = dec(a)
        except Exception:
            return
        if not mat or not mat[0]:
            return
        obs.append((mat, b))
    for layout in _LAYOUTS:
        lr, lc = _layout_dims(obs[0][0], layout)
        dims = set()
        ok = True
        for mat, b in obs:
            r, c = _layout_dims(mat, layout)
            bh, bw = len(b), len(b[0])
            if r <= 0 or c <= 0 or bh % r or bw % c:
                ok = False
                break
            dims.add((bh // r, bw // c))
        if not ok or len(dims) != 1:
            continue
        bh, bw = dims.pop()
        if bh * bw > 64:
            continue
        for kind in ("content", "mask", "count", "palette"):
            table, counts = {}, {}
            good = True
            for mat, b in obs:
                r, c = _layout_dims(mat, layout)
                for i, panel in enumerate(_layout_order(mat, layout)):
                    k = _panel_key(panel, bg, kind)
                    if k is None:
                        good = False
                        break
                    r0, c0 = (i // c) * bh, (i % c) * bw
                    block = tuple(tuple(b[r0 + y][c0 + x] for x in range(bw))
                                  for y in range(bh))
                    if table.setdefault(k, block) != block:
                        good = False
                        break
                    counts[k] = counts.get(k, 0) + 1
                if not good:
                    break
            if not good or not table:
                continue
            if min(counts.values()) < 2:
                continue          # a class seen once is an answer, not a class
            rule = _DictRule(dec, kind, layout, table, bg, bh, bw)
            hp = Hyp("pdict[%s/%s/%s|%d]" % (dname, kind, layout, len(table)),
                     rule, 2.8 + dcost * 0.3 + 0.05 * len(table), SOLVER)
            if not hp.fits(ctx.train):
                continue
            try:
                sig = tuple(hp.apply(g) for g in ctx.test_inputs)
            except Exception:
                continue
            if sig in seen:
                continue
            seen.add(sig)
            res.append(hp)
            return


def generate(ctx):
    if ctx.same_shape:
        return []
    bg = ctx.bg
    res, seen = [], set()
    for dname, dcost, dec in _decompositions(ctx):
        if ctx.timed_out() or len(res) >= 16:
            break
        try:
            _dict_rules(ctx, dname, dcost, dec, bg, res, seen)
        except Exception:
            pass
        rows = _observe(ctx, dec)
        if not rows:
            continue
        n_panels = len(rows[0][0])
        for kname, kcost, keyfn in _keyfns(bg):
            got = _fit(rows, keyfn, bg)
            if got is None:
                continue
            table, obs = got
            modal = Counter(table.values()).most_common(1)[0][0]
            for default, extra in ((None, 0.0), (modal, 0.4)):
                rule = _TableRule(dec, keyfn, table, bg, default)
                hp = Hyp("ptab[%s/%s|%d>%d]" % (dname, kname, n_panels,
                                                len(table)),
                         rule, 2.4 + dcost * 0.3 + kcost
                         + 0.02 * len(table) + extra, SOLVER)
                if not hp.fits(ctx.train):
                    continue
                try:
                    sig = tuple(hp.apply(g) for g in ctx.test_inputs)
                except Exception:
                    continue
                if sig in seen:
                    break
                seen.add(sig)
                res.append(hp)
                break
    return res
