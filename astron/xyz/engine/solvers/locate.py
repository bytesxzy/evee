"""When the answer is a piece of the input, learn *where* to cut.

Extraction is the largest single shape of ARC task after in-place editing, and
it factorises cleanly: the output is a subgrid of the input, so the only thing
to infer is the rectangle.  Existing families each know one way to name a
rectangle -- ``select`` names an object, ``partition`` names a panel,
``regions`` names an enclosure -- and a task whose rectangle is named some other
way falls between them.

This module inverts the problem instead.  It first *finds* every place the
demonstrated output actually occurs inside its input, which is cheap and
completely determines what any correct rule must return.  Then it asks which of
a library of locators picks exactly those places, in every pair at once.  A
locator that agrees everywhere is a rule; one that agrees in three pairs out of
four is discarded, not patched.

The library names rectangles by:

* the extent of a colour, or of everything that is not background;
* the bounding box, or the interior, of a chosen object under several
  segmentations and several readings of "chosen";
* the inside of a rectangular frame;
* a panel of a lattice;
* the fixed-size window that maximises or minimises some quantity -- density,
  a particular colour's count, how many distinct colours it holds -- which is
  the reading used by "find the odd patch" tasks, where the region is not an
  object at all.

The last group is only offered when the task fixes the output size, because a
window scan needs to know how big the window is before it can rank windows.
"""

from collections import Counter

from .. import grid as G
from .. import objects as O
from ..task import Hyp

SOLVER = "select"

_SEGS = ("c4", "c8", "m4", "m8", "color", "g2m")
_MAX_HITS = 40

# (name, forward, inverse) -- the inverse is applied to the demonstrated output
# so that the search looks for the piece as it appears in the input
_VIEWS = (
    ("id", lambda g: g, lambda g: g),
    ("rot90", G.rot90, G.rot270),
    ("rot180", G.rot180, G.rot180),
    ("rot270", G.rot270, G.rot90),
    ("flip_h", G.flip_h, G.flip_h),
    ("flip_v", G.flip_v, G.flip_v),
    ("transpose", G.transpose, G.transpose),
)


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


def _occurrences(g, out):
    """Every (r0, c0) where ``out`` sits inside ``g``."""
    H, W = G.dims(g)
    h, w = G.dims(out)
    if h > H or w > W:
        return []
    hits = []
    first = out[0]
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            if g[r][c:c + w] != first:
                continue
            if all(g[r + i][c:c + w] == out[i] for i in range(1, h)):
                hits.append((r, c))
                if len(hits) > _MAX_HITS:
                    return hits
    return hits


# --- locators: grid -> (r0, c0, r1, c1) ------------------------------------

def _loc_color(c):
    def run(g, bg):
        cells = [(r, x) for r, row in enumerate(g)
                 for x, v in enumerate(row) if v == c]
        if not cells:
            return None
        return G.bbox_of(cells)
    return run


def _loc_nonbg(g, bg):
    cells = [(r, x) for r, row in enumerate(g)
             for x, v in enumerate(row) if v != bg]
    return G.bbox_of(cells) if cells else None


def _pick(objs, how):
    if not objs:
        return None
    if how == "big":
        return max(objs, key=lambda o: (o.size, -o.r0, -o.c0))
    if how == "small":
        return min(objs, key=lambda o: (o.size, o.r0, o.c0))
    if how == "bigbox":
        return max(objs, key=lambda o: (o.bbox_area, -o.r0, -o.c0))
    if how == "ucol":
        cnt = Counter(o.color for o in objs)
        cands = [o for o in objs if cnt[o.color] == 1]
        return cands[0] if len(cands) == 1 else None
    if how == "ushp":
        cnt = Counter(o.norm_key() for o in objs)
        cands = [o for o in objs if cnt[o.norm_key()] == 1]
        return cands[0] if len(cands) == 1 else None
    if how == "holes":
        cands = [o for o in objs if o.holes_count() > 0]
        return max(cands, key=lambda o: (o.holes_count(), o.size)) if cands else None
    if how == "dense":
        return max(objs, key=lambda o: (len(o.colors()), o.size))
    return None


def _loc_obj(seg, how, inner):
    def run(g, bg):
        objs = O.segment(g, seg, bg)
        if not objs or len(objs) > 60:
            return None
        o = _pick(objs, how)
        if o is None:
            return None
        if inner:
            if o.r1 - o.r0 < 2 or o.c1 - o.c0 < 2:
                return None
            return (o.r0 + 1, o.c0 + 1, o.r1 - 1, o.c1 - 1)
        return (o.r0, o.c0, o.r1, o.c1)
    return run


def _windows(g, h, w):
    H, W = G.dims(g)
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            yield r, c


def _score_window(g, r, c, h, w, kind, bg):
    vals = [g[r + i][c + j] for i in range(h) for j in range(w)]
    if kind == "density":
        return sum(1 for v in vals if v != bg)
    if kind == "colors":
        return len(set(vals))
    if kind == "uniform":
        return -len(set(vals))
    return 0


def _loc_window(shape, kind, want_max, bg_color=None):
    h, w = shape

    def run(g, bg):
        H, W = G.dims(g)
        if h > H or w > W:
            return None
        best, arg = None, None
        tie = False
        for r, c in _windows(g, h, w):
            if bg_color is None:
                s = _score_window(g, r, c, h, w, kind, bg)
            else:
                s = sum(1 for i in range(h) for j in range(w)
                        if g[r + i][c + j] == bg_color)
            if best is None or (s > best if want_max else s < best):
                best, arg, tie = s, (r, c), False
            elif s == best:
                tie = True
        if arg is None or tie:
            return None                 # an ambiguous extreme is not a rule
        return (arg[0], arg[1], arg[0] + h - 1, arg[1] + w - 1)
    return run


def _locators(ctx):
    bg = ctx.bg
    out = [("nonbg", 0.4, _loc_nonbg)]
    for c in sorted(ctx.in_palette)[:9]:
        out.append(("col%d" % c, 0.8, _loc_color(c)))
    for seg in _SEGS:
        for how in ("big", "small", "bigbox", "ucol", "ushp", "holes", "dense"):
            for inner in (False, True):
                out.append(("%s.%s%s" % (seg, how, ".in" if inner else ""),
                            0.9 + (0.2 if inner else 0.0),
                            _loc_obj(seg, how, inner)))
    shape = ctx.const_out_shape
    if shape and shape[0] * shape[1] <= 400:
        for kind in ("density", "colors", "uniform"):
            for want in (True, False):
                out.append(("win.%s.%s" % (kind, "max" if want else "min"), 1.4,
                            _loc_window(shape, kind, want)))
        for c in sorted(ctx.in_palette)[:6]:
            for want in (True, False):
                out.append(("win.c%d.%s" % (c, "max" if want else "min"), 1.6,
                            _loc_window(shape, "count", want, c)))
    return out


class _Cut:
    __slots__ = ("loc", "bg", "xform")

    def __init__(self, loc, bg, xform=None):
        self.loc = loc
        self.bg = bg
        self.xform = xform

    def __call__(self, g):
        box = self.loc(g, G.bg_or(g, self.bg))
        if box is None:
            return None
        r0, c0, r1, c1 = box
        H, W = G.dims(g)
        if not (0 <= r0 <= r1 < H and 0 <= c0 <= c1 < W):
            return None
        cut = G.subgrid(g, r0, c0, r1, c1)
        if cut is None or self.xform is None:
            return cut
        return self.xform(cut)


def generate(ctx):
    if ctx.same_shape:
        return []
    if any(G.area(b) >= G.area(a) for a, b in ctx.train):
        return []
    # The answer is often a *transformed* piece, so look for each dihedral
    # image of the output as well; the cut is then followed by that transform.
    views = []
    for xname, xf, inv in _VIEWS:
        targets = []
        ok = True
        for a, b in ctx.train:
            pre = inv(b)
            hits = _occurrences(a, pre)
            if not hits:
                ok = False
                break
            h, w = G.dims(pre)
            targets.append({(r, c, r + h - 1, c + w - 1) for r, c in hits})
        if ok:
            views.append((xname, xf, targets))
    if not views:
        return []                       # not an extraction task at all
    bg = ctx.bg
    res, seen = [], set()
    for xname, xf, targets in views:
        for name, cost, loc in _locators(ctx):
            if ctx.timed_out() or len(res) >= 14:
                break
            ok = True
            for (a, _b), want in zip(ctx.train, targets):
                try:
                    box = loc(a, G.bg_or(a, bg))
                except Exception:
                    box = None
                if box is None or box not in want:
                    ok = False
                    break
            if not ok:
                continue
            hp = _h("cut[%s%s]" % (name, "" if xname == "id" else ">" + xname),
                    _Cut(loc, bg, None if xname == "id" else xf),
                    2.6 + cost + (0.0 if xname == "id" else 0.5))
            if not hp.fits(ctx.train):
                continue
            try:
                sig = tuple(hp.apply(t) for t in ctx.test_inputs)
            except Exception:
                continue
            if sig in seen:
                continue
            seen.add(sig)
            res.append(hp)
    return res
