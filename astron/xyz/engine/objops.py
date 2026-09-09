"""Object-level operators for the general enumerator.

The synthesiser in ``enum_core`` composes grid-to-grid functions, and its whole
library was written at the level of pixels and whole grids: rotate, crop, tile,
shift, fill.  Nothing in it can say *which thing* to act on.  That is the single
largest hole in the program space, because a great many ARC rules are of the
form "do this to the odd one out" or "throw away everything that touches the
edge, then crop".

These operators close it.  Each one is still an ordinary grid-to-grid function,
so it drops straight into the existing search -- and therefore into every family
built on that search: the depth-4 enumerator, the shallow ``compose`` pass, the
forward ``cascade`` and the backward ``refine``.  The composition comes for
free; only the vocabulary is new.

An operator is a **selector** crossed with an **action**:

    selector  largest, smallest, unique colour, unique shape, majority shape,
              touching the border, enclosed, holed, square, symmetric, ...
    action    delete it, keep only it, crop to it, recolour it, slide it,
              snap it to an edge, outline it, fill its holes, fill its box

Cost control matters more here than anywhere else in the engine, because the
enumerator applies its whole library at every node.  Three things keep this
affordable: the operators are offered as *seeds* (depth 1 only, where the
frontier is one grid); segmentation is memoised per grid so a hundred operators
share one connected-components pass; and the vocabulary is generated from the
task -- colours that the task never produces are not offered as paint.
"""

from collections import Counter

from . import grid as G
from . import objects as O

# Ablation switch: when off, the enumerator sees only its pixel-level library,
# which is what the engine looked like before object operators existed.
ENABLED = True

_SEG_CACHE = {}
_SEG_CACHE_KEYS = []


def segment(g, seg, bg):
    """Memoised segmentation: a hundred operators, one components pass."""
    key = (id(g), len(g), g[0] if g else (), seg, bg)
    hit = _SEG_CACHE.get(key)
    if hit is not None and hit[0] is g:
        return hit[1]
    objs = O.segment(g, seg, bg)
    _SEG_CACHE[key] = (g, objs)
    _SEG_CACHE_KEYS.append(key)
    if len(_SEG_CACHE_KEYS) > 400:
        for k in _SEG_CACHE_KEYS[:200]:
            _SEG_CACHE.pop(k, None)
        del _SEG_CACHE_KEYS[:200]
    return objs


# --------------------------------------------------------------------------
# selectors: scene -> the objects the operator acts on
# --------------------------------------------------------------------------

def _sym(o):
    p = o.filled(0)
    return p == G.flip_h(p) or p == G.flip_v(p)


def _selectors():
    def by(fn):
        return fn

    def largest(objs):
        m = max(o.size for o in objs)
        return [o for o in objs if o.size == m]

    def smallest(objs):
        m = min(o.size for o in objs)
        return [o for o in objs if o.size == m]

    def widest(objs):
        m = max(o.bbox_area for o in objs)
        return [o for o in objs if o.bbox_area == m]

    def uniq_color(objs):
        c = Counter(o.color for o in objs)
        return [o for o in objs if c[o.color] == 1]

    def common_color(objs):
        c = Counter(o.color for o in objs)
        m = max(c.values())
        return [o for o in objs if c[o.color] == m]

    def uniq_shape(objs):
        c = Counter(o.norm_key() for o in objs)
        return [o for o in objs if c[o.norm_key()] == 1]

    def common_shape(objs):
        c = Counter(o.norm_key() for o in objs)
        m = max(c.values())
        return [o for o in objs if c[o.norm_key()] == m]

    def uniq_size(objs):
        c = Counter(o.size for o in objs)
        return [o for o in objs if c[o.size] == 1]

    def holed(objs):
        return [o for o in objs if o.holes_count() > 0]

    def solid(objs):
        return [o for o in objs if o.holes_count() == 0]

    def border(objs):
        return [o for o in objs if o.touches_border()]

    def inner(objs):
        return [o for o in objs if not o.touches_border()]

    def square(objs):
        return [o for o in objs if o.is_square()]

    def rect(objs):
        return [o for o in objs if o.is_rect()]

    def symmetric(objs):
        return [o for o in objs if _sym(o)]

    def multi(objs):
        return [o for o in objs if len(o.colors()) > 1]

    return [
        ("all", 0.0, by(lambda objs: list(objs))),
        ("big", 0.4, largest), ("small", 0.4, smallest),
        ("wide", 0.6, widest),
        ("ucol", 0.5, uniq_color), ("mcol", 0.7, common_color),
        ("ushp", 0.5, uniq_shape), ("mshp", 0.6, common_shape),
        ("usz", 0.7, uniq_size),
        ("hole", 0.5, holed), ("nohole", 0.7, solid),
        ("edge", 0.5, border), ("in", 0.5, inner),
        ("sq", 0.6, square), ("rc", 0.7, rect),
        ("sym", 0.7, symmetric), ("multi", 0.7, multi),
    ]


_DIRS = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}


# --------------------------------------------------------------------------
# actions: (grid, chosen objects, all objects, bg) -> grid
# --------------------------------------------------------------------------

def _act_delete(g, sel, objs, bg):
    out = [list(r) for r in g]
    for o in sel:
        for r, c in o.cells:
            out[r][c] = bg
    return tuple(tuple(r) for r in out)


def _act_keep(g, sel, objs, bg):
    keep = set()
    for o in sel:
        keep |= o.cells
    h, w = G.dims(g)
    return tuple(tuple(g[r][c] if (r, c) in keep else bg for c in range(w))
                 for r in range(h))


def _act_crop(g, sel, objs, bg):
    if not sel:
        return None
    r0 = min(o.r0 for o in sel)
    c0 = min(o.c0 for o in sel)
    r1 = max(o.r1 for o in sel)
    c1 = max(o.c1 for o in sel)
    return G.subgrid(g, r0, c0, r1, c1)


def _act_cropmask(g, sel, objs, bg):
    if len(sel) != 1:
        return None
    return sel[0].filled(bg)


def _act_recolor(color):
    def run(g, sel, objs, bg):
        out = [list(r) for r in g]
        for o in sel:
            for r, c in o.cells:
                out[r][c] = color
        return tuple(tuple(r) for r in out)
    return run


def _act_fill(color):
    def run(g, sel, objs, bg):
        out = [list(r) for r in g]
        touched = False
        for o in sel:
            for r, c in _interior(o):
                out[r][c] = color
                touched = True
        return tuple(tuple(r) for r in out) if touched else None
    return run


def _act_box(color):
    def run(g, sel, objs, bg):
        out = [list(r) for r in g]
        for o in sel:
            for r in range(o.r0, o.r1 + 1):
                for c in range(o.c0, o.c1 + 1):
                    out[r][c] = color
        return tuple(tuple(r) for r in out)
    return run


def _act_outline(color):
    def run(g, sel, objs, bg):
        h, w = G.dims(g)
        out = [list(r) for r in g]
        touched = False
        for o in sel:
            for r, c in o.cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if (0 <= rr < h and 0 <= cc < w
                                and (rr, cc) not in o.cells and g[rr][cc] == bg):
                            out[rr][cc] = color
                            touched = True
        return tuple(tuple(r) for r in out) if touched else None
    return run


def _act_move(d, mode):
    dr, dc = _DIRS[d]

    def run(g, sel, objs, bg):
        h, w = G.dims(g)
        blocked = set()
        for o in objs:
            if o not in sel:
                blocked |= o.cells
        out = [list(r) for r in g]
        draws = []
        for o in sel:
            if mode == "edge":
                k = _to_edge(o, h, w, dr, dc)
            else:
                k = _until_blocked(o, h, w, dr, dc, blocked)
            if k <= 0:
                continue
            for r, c in o.cells:
                out[r][c] = bg
            for r, c in o.cells:
                draws.append((r + dr * k, c + dc * k, g[r][c]))
        if not draws:
            return None
        for r, c, v in draws:
            if not (0 <= r < h and 0 <= c < w):
                return None
            out[r][c] = v
        return tuple(tuple(r) for r in out)
    return run


def _to_edge(o, h, w, dr, dc):
    if dr < 0:
        return o.r0
    if dr > 0:
        return h - 1 - o.r1
    if dc < 0:
        return o.c0
    return w - 1 - o.c1


def _until_blocked(o, h, w, dr, dc, blocked):
    limit = _to_edge(o, h, w, dr, dc)
    for k in range(1, limit + 1):
        for r, c in o.cells:
            if (r + dr * k, c + dc * k) in blocked:
                return k - 1
    return limit


def _interior(o):
    key = "interior0"
    got = o._holes.get(key)
    if got is not None:
        return got
    hh = o.r1 - o.r0 + 1
    ww = o.c1 - o.c0 + 1
    inside = [[True] * ww for _ in range(hh)]
    stack = [(r, c) for r in range(hh) for c in (0, ww - 1)]
    stack += [(r, c) for c in range(ww) for r in (0, hh - 1)]
    while stack:
        r, c = stack.pop()
        if not (0 <= r < hh and 0 <= c < ww) or not inside[r][c]:
            continue
        if (r + o.r0, c + o.c0) in o.cells:
            continue
        inside[r][c] = False
        stack.extend(((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)))
    out = [(r + o.r0, c + o.c0) for r in range(hh) for c in range(ww)
           if inside[r][c] and (r + o.r0, c + o.c0) not in o.cells]
    o._holes[key] = out
    return out


def _actions(ctx):
    """Paint colours come from the task, never from the full palette."""
    news = sorted(ctx.new_colors)[:2]
    hot = [c for c, _n in Counter(
        c for b in ctx.outputs for row in b for c in row).most_common(4)]
    paints = []
    for c in news + hot:
        if c not in paints:
            paints.append(c)
    paints = paints[:4]
    acts = [("del", 0.6, _act_delete), ("only", 0.6, _act_keep),
            ("crop", 0.8, _act_crop), ("cropm", 1.0, _act_cropmask)]
    for c in paints[:3]:
        acts.append(("rec%d" % c, 0.9, _act_recolor(c)))
    for c in paints[:2]:
        acts.append(("fil%d" % c, 1.0, _act_fill(c)))
    for c in paints[:1]:
        acts.append(("box%d" % c, 1.1, _act_box(c)))
        acts.append(("out%d" % c, 1.1, _act_outline(c)))
    for d in "udlr":
        acts.append(("slide" + d, 1.0, _act_move(d, "block")))
        acts.append(("snap" + d, 1.1, _act_move(d, "edge")))
    return acts


def _mk(seg, bg, sel_fn, act_fn, cap):
    def run(g):
        b = G.bg_or(g, bg)
        objs = segment(g, seg, b)
        if not objs or len(objs) > cap:
            return None
        sel = sel_fn(objs)
        if not sel:
            return None
        out = act_fn(g, sel, objs, b)
        if out is None or not G.valid(out):
            return None
        return out if out != g else None
    return run


def object_ops(ctx, cap=28, segs=("c8",)):
    """(name, cost, fn) triples: one per selector x action x segmentation."""
    if not ENABLED:
        return []
    sels = _selectors()
    acts = _actions(ctx)
    bg = None if ctx.bg_varies else ctx.bg
    ops = []
    for seg in segs:
        for sname, scost, sfn in sels:
            for aname, acost, afn in acts:
                ops.append(("%s.%s.%s" % (seg, sname, aname),
                            1.6 + scost + acost,
                            _mk(seg, bg, sfn, afn, cap)))
    return ops
