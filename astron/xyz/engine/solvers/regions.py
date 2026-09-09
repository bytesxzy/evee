"""Region extraction: windows, frames, marked rectangles, colour answers.

Covers the "show me the part that matters" family.  The window scanner is the
workhorse: when the output shape is known and small, every window of that shape
in the input is a candidate and the task reduces to *choosing* one -- again a
relational selection rather than a hand-written rule.
"""

from collections import Counter

from .. import grid as G
from .. import objects as O
from ..task import Hyp

SOLVER = "select"


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


# --------------------------------------------------------------------------
# window scanning
# --------------------------------------------------------------------------

def _windows(g, oh, ow, step=1):
    h, w = G.dims(g)
    if oh > h or ow > w:
        return []
    return [(r, c, tuple(row[c:c + ow] for row in g[r:r + oh]))
            for r in range(0, h - oh + 1, step)
            for c in range(0, w - ow + 1, step)]


_WIN_CRIT = {
    "ncolors": lambda p, bg: len(G.palette(p)),
    "nnz": lambda p, bg: sum(1 for r in p for v in r if v != bg),
    "nbg": lambda p, bg: sum(1 for r in p for v in r if v == bg),
    "nsym": lambda p, bg: len(G.symmetries(p)),
    "distinct": lambda p, bg: -len(set(p)),
}


_WCACHE = {}


def _win_analysis(g, oh, ow, bg, tiled):
    """Windows plus every criterion value, computed once and reused.

    Without this the scanner recomputes the same few hundred windows once per
    criterion, which dominated the whole engine's runtime.
    """
    key = (g, oh, ow, bg, tiled)
    hit = _WCACHE.get(key)
    if hit is not None:
        return hit
    wins = _windows(g, oh, ow, 1)
    if tiled:
        wins = [(r, c, p) for r, c, p in wins if r % oh == 0 and c % ow == 0]
    pats = [p for _r, _c, p in wins]
    vals = {}
    if pats:
        for k, f in _WIN_CRIT.items():
            if k == "nsym" and len(pats) > 300:
                continue            # symmetry testing is the expensive one
            vals[k] = [f(p, bg) for p in pats]
    cnt = Counter(pats)
    res = (pats, vals, cnt)
    if len(_WCACHE) > 64:
        _WCACHE.clear()
    _WCACHE[key] = res
    return res


def _window_pick(g, oh, ow, how, bg, tiled):
    if oh < 1 or ow < 1 or oh > len(g) or ow > len(g[0]):
        return None
    n_win = (len(g) - oh + 1) * (len(g[0]) - ow + 1)
    if n_win > 1200:
        return None
    pats, vals, cnt = _win_analysis(g, oh, ow, bg, tiled)
    if not pats:
        return None
    if how == "unique":
        hits = [p for p in pats if cnt[p] == 1]
        return hits[0] if len(hits) == 1 else None
    if how == "modal":
        k, n = cnt.most_common(1)[0]
        return k if n > 1 else None
    v = vals.get(how[4:])
    if v is None:
        return None
    tgt = max(v) if how.startswith("max_") else min(v)
    hits = {p for p, x in zip(pats, v) if x == tgt}
    return next(iter(hits)) if len(hits) == 1 else None


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def _frames(g, bg):
    """Hollow single-colour rectangles, as (r0, c0, r1, c1, colour)."""
    out = []
    for o in O.segment(g, "c8", bg):
        if o.height < 3 or o.width < 3:
            continue
        m = o.mask
        ok = True
        for r in range(o.height):
            for c in range(o.width):
                edge = (r in (0, o.height - 1)) or (c in (0, o.width - 1))
                if edge and not m[r][c]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append((o.r0, o.c0, o.r1, o.c1, o.color))
    return out


def _frame_interior(g, bg, which):
    fr = _frames(g, bg)
    if not fr:
        return None
    if len(fr) > 1:
        fr.sort(key=lambda f: -((f[2] - f[0]) * (f[3] - f[1])))
        f = fr[0] if which == "largest" else fr[-1]
    else:
        f = fr[0]
    return G.subgrid(g, f[0] + 1, f[1] + 1, f[2] - 1, f[3] - 1)


def _frame_content(g, bg, which):
    fr = _frames(g, bg)
    if not fr:
        return None
    fr.sort(key=lambda f: -((f[2] - f[0]) * (f[3] - f[1])))
    f = fr[0] if which == "largest" else fr[-1]
    return G.subgrid(g, f[0], f[1], f[2], f[3])


def _marked_rect(g, bg, marker, inclusive, pr=0, pc=0):
    """Bounding box of a marker colour, optionally grown or shrunk.

    Growing matters because markers usually delimit a structure rather than
    being it -- a pair of ruled lines with capped ends, for instance.
    """
    cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row)
             if v == marker]
    if len(cells) < 2:
        return None
    h, w = G.dims(g)
    r0, c0, r1, c1 = G.bbox_of(cells)
    if not inclusive:
        r0, c0, r1, c1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    r0, c0 = max(0, r0 - pr), max(0, c0 - pc)
    r1, c1 = min(h - 1, r1 + pr), min(w - 1, c1 + pc)
    return G.subgrid(g, r0, c0, r1, c1)


def _color_component_box(g, bg, color, seg):
    """Crop to the components that *contain* a given colour.

    Distinct from cropping to the colour itself: the interesting object is
    usually the whole connected structure the marker is part of.
    """
    objs = O.segment(g, seg, bg)
    cells = set()
    for o in objs:
        if color in o.colors():
            cells |= o.cells
    if not cells:
        return None
    return G.subgrid(g, *G.bbox_of(cells))


def _cut_color_region(g, bg, color):
    """Bounding box of everything that is *not* the given colour."""
    cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row)
             if v != color]
    if not cells:
        return None
    return G.subgrid(g, *G.bbox_of(cells))


# --------------------------------------------------------------------------
# colour answers (1x1 and monochrome outputs)
# --------------------------------------------------------------------------

_COLOR_PICKS = {
    "most": lambda g, bg: _rank(g, bg, -1),
    "least": lambda g, bg: _rank(g, bg, 0),
    "second": lambda g, bg: _rank(g, bg, -2),
    "largest_obj": lambda g, bg: _obj_color(g, bg, True),
    "smallest_obj": lambda g, bg: _obj_color(g, bg, False),
    "unique_shape_obj": lambda g, bg: _uniq_color(g, bg),
    "center": lambda g, bg: g[len(g) // 2][len(g[0]) // 2],
    "corner": lambda g, bg: g[0][0],
}


def _rank(g, bg, i):
    hist = G.histogram(g)
    hist.pop(bg, None)
    items = sorted(hist.items(), key=lambda kv: (kv[1], kv[0]))
    try:
        return items[i][0]
    except IndexError:
        return None


def _obj_color(g, bg, biggest):
    objs = O.segment(g, "c8", bg)
    if not objs:
        return None
    o = O.select_extreme(objs, "size", biggest)
    return None if o is None else o.color


def _uniq_color(g, bg):
    objs = O.segment(g, "c8", bg)
    if not objs:
        return None
    o = O.select_unique_shape(objs)
    return None if o is None else o.color


def _color_grid(g, pick, bg, shape):
    c = pick(g, bg)
    if c is None:
        return None
    if shape is None:
        return ((c,),)
    return G.const_grid(shape[0], shape[1], c)


# --------------------------------------------------------------------------

def generate(ctx):
    res = []
    bg = ctx.bg
    cs = ctx.const_out_shape

    # -- window scan ------------------------------------------------------
    crits = (["unique", "modal"] +
             ["max_" + k for k in _WIN_CRIT] + ["min_" + k for k in _WIN_CRIT])
    if cs and cs[0] * cs[1] <= 400:
        for how in crits:
            for tiled in (False, True):
                res.append(_h("win%dx%d.%s%s" % (cs[0], cs[1], how,
                                                 "_t" if tiled else ""),
                              (lambda s, how, bg, t:
                               lambda g: _window_pick(g, s[0], s[1], how, bg, t))(cs, how, bg, tiled),
                              4.0 + (0.0 if tiled else 0.5)))
    ir = ctx.inv_shape_ratio
    if not cs and ir and ir != (1, 1):
        ky, kx = ir
        for how in crits:
            res.append(_h("winr%dx%d.%s" % (ky, kx, how),
                          (lambda ky, kx, how, bg:
                           lambda g: _window_pick(g, len(g) // ky, len(g[0]) // kx,
                                                  how, bg, True))(ky, kx, how, bg),
                          4.5))

    # -- frames -----------------------------------------------------------
    for which in ("largest", "smallest"):
        res.append(_h("frame_in_" + which,
                      (lambda w, bg: lambda g: _frame_interior(g, bg, w))(which, bg), 3.5))
        res.append(_h("frame_all_" + which,
                      (lambda w, bg: lambda g: _frame_content(g, bg, w))(which, bg), 3.8))
    for c in sorted(ctx.in_palette):
        res.append(_h("mark#%d_in" % c,
                      (lambda c, bg: lambda g: _marked_rect(g, bg, c, False))(c, bg), 4.5))
        res.append(_h("mark#%d_all" % c,
                      (lambda c, bg: lambda g: _marked_rect(g, bg, c, True))(c, bg), 4.5))
        res.append(_h("notbox#%d" % c,
                      (lambda c, bg: lambda g: _cut_color_region(g, bg, c))(c, bg), 4.5))
        for pr, pc in ((1, 0), (0, 1), (1, 1)):
            res.append(_h("mark#%d_pad%d%d" % (c, pr, pc),
                          (lambda c, bg, pr, pc:
                           lambda g: _marked_rect(g, bg, c, True, pr, pc))(c, bg, pr, pc),
                          4.8))
        for seg in ("m8", "m4"):
            res.append(_h("compbox#%d.%s" % (c, seg),
                          (lambda c, bg, seg: lambda g: _color_component_box(g, bg, c, seg))(c, bg, seg),
                          4.2))

    # -- colour answers ---------------------------------------------------
    shapes = [None]
    if cs and cs[0] * cs[1] <= 25:
        shapes.append(cs)
    for name, pick in _COLOR_PICKS.items():
        for sh in shapes:
            res.append(_h("color_%s%s" % (name, "" if sh is None else "_fill"),
                          (lambda p, bg, sh: lambda g: _color_grid(g, p, bg, sh))(pick, bg, sh),
                          5.0))
    return res
