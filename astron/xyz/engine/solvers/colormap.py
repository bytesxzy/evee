"""Colour-level rules that generalise beyond the colours actually seen.

A literal lookup table (learned by ``cellwise``) fails the moment a test grid
introduces a colour that never appeared in training.  The rules here are
*relational* -- they key on rank, on frequency, on role -- so they transfer.
"""

from collections import Counter

from .. import grid as G
from ..task import Hyp

SOLVER = "colormap"


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


def _rank_order(g, bg, asc):
    hist = G.histogram(g)
    hist.pop(bg, None)
    items = sorted(hist.items(), key=lambda kv: (kv[1], kv[0]))
    if not asc:
        items = items[::-1]
    return [k for k, _ in items]


def generate(ctx):
    res = []
    bg = ctx.bg

    # ---- rank-preserving recolour ---------------------------------------
    if ctx.same_shape:
        for asc in (False, True):
            perm = _fit_rank_perm(ctx, bg, asc)
            if perm is not None:
                res.append(_h("rankmap_%s" % ("desc" if asc is False else "asc"),
                              lambda g, p=perm, b=bg, a=asc: _apply_rank(g, p, b, a),
                              4.0))
        # swap the two most frequent non-background colours
        res.append(_h("swap_top2", lambda g, b=bg: _swap_top(g, b, 0, 1), 4.5))
        res.append(_h("swap_minmax", lambda g, b=bg: _swap_minmax(g, b), 4.5))

    # ---- monochrome outputs ---------------------------------------------
    for c in sorted(ctx.out_palette):
        res.append(_h("solid#%d" % c,
                      lambda g, c=c: G.const_grid(len(g), len(g[0]), c), 6.0))

    # ---- keep only one colour -------------------------------------------
    for c in sorted(ctx.in_palette):
        res.append(_h("keep#%d" % c,
                      lambda g, c=c, b=bg: _keep(g, c, b), 4.5))
        res.append(_h("drop#%d" % c,
                      lambda g, c=c, b=bg: G.replace_color(g, c, b), 4.0))

    # ---- keep the rarest / commonest colour, computed per grid ----------
    res.append(_h("keep_rarest", lambda g, b=bg: _keep_rank(g, b, True), 5.0))
    res.append(_h("keep_commonest", lambda g, b=bg: _keep_rank(g, b, False), 5.0))
    res.append(_h("recolor_all_rarest", lambda g, b=bg: _all_to(g, b, True), 5.5))
    res.append(_h("recolor_all_commonest", lambda g, b=bg: _all_to(g, b, False), 5.5))

    # ---- conditional map keyed on a global grid property ----------------
    res.extend(_conditional_maps(ctx, bg))
    return res


def _fit_rank_perm(ctx, bg, asc):
    perm = {}
    for a, b in ctx.train:
        if G.dims(a) != G.dims(b):
            return None
        order = _rank_order(a, bg, asc)
        idx = {c: i for i, c in enumerate(order)}
        for ra, rb in zip(a, b):
            for x, y in zip(ra, rb):
                if x == bg:
                    if y != bg:
                        return None
                    continue
                i = idx.get(x)
                if i is None:
                    return None
                if perm.setdefault(i, y) != y:
                    return None
    return perm or None


def _apply_rank(g, perm, bg, asc):
    order = _rank_order(g, bg, asc)
    m = {}
    for i, c in enumerate(order):
        if i in perm:
            m[c] = perm[i]
        else:
            return None
    return G.apply_cmap(g, m)


def _swap_top(g, bg, i, j):
    order = _rank_order(g, bg, False)
    if len(order) <= max(i, j):
        return None
    a, b = order[i], order[j]
    return G.apply_cmap(g, {a: b, b: a})


def _swap_minmax(g, bg):
    order = _rank_order(g, bg, False)
    if len(order) < 2:
        return None
    a, b = order[0], order[-1]
    return G.apply_cmap(g, {a: b, b: a})


def _keep(g, c, bg):
    return tuple(tuple(v if v == c else bg for v in r) for r in g)


def _keep_rank(g, bg, rarest):
    order = _rank_order(g, bg, rarest)
    if not order:
        return None
    return _keep(g, order[0], bg)


def _all_to(g, bg, rarest):
    order = _rank_order(g, bg, rarest)
    if not order:
        return None
    c = order[0]
    return tuple(tuple(c if v != bg else bg for v in r) for r in g)


_GRID_PROPS = {
    "ncolors": lambda g, bg: len(G.palette(g)),
    "nnz_parity": lambda g, bg: sum(1 for r in g for v in r if v != bg) % 2,
    "shape": lambda g, bg: G.dims(g),
    "maxcolor": lambda g, bg: max(G.palette(g)),
    "nobj": lambda g, bg: len(G.flood_regions(g, bg, True, True)),
    "symm": lambda g, bg: tuple(G.symmetries(g)),
}


def _conditional_maps(ctx, bg):
    """out = const(P(input)) -- the answer is a lookup on a global property."""
    res = []
    for name, prop in _GRID_PROPS.items():
        table = {}
        ok = True
        for a, b in ctx.train:
            try:
                k = prop(a, bg)
            except Exception:
                ok = False
                break
            if table.setdefault(k, b) != b:
                ok = False
                break
        # A lookup keyed so finely that every training pair gets its own row has
        # memorised the data and predicts nothing; require real compression.
        if ok and 1 < len(table) <= len(ctx.train) - 2:
            res.append(_h("lookup_" + name,
                          lambda g, t=table, p=prop, b=bg: t.get(_safe(p, g, b)),
                          9.0))
    return res


def _safe(p, g, bg):
    try:
        return p(g, bg)
    except Exception:
        return None
