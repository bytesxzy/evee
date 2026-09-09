"""Counting: the answer is a number, expressed as a grid.

``ARCHITECTURE.md`` §10 states the gap plainly -- "no notion of counting
arguments" -- and it is a real one.  A recurring ARC family asks how *many* of
something there are and renders the answer as a grid: an ``n x n`` block, a bar
of length ``n``, the input repeated ``n`` times, or the input scaled by ``n``.
Every specialist in the portfolio is blind to those tasks, because none of them
can express "the output's size is a function of a property of the input".

This module fits rather than searches, like the rest of the portfolio.  For
each of a small set of **counters** it reads ``n`` off every training input,
then asks whether one **shape law** and one **colour law** explain every
training output.  A law that contradicts a single pair is discarded; nothing
here searches and nothing is scored on partial agreement.

Two guards, both borrowed from the modules that needed them first:

* **A counter that is constant across the training inputs proves nothing.**
  If every input happens to contain three objects, "n = number of objects"
  and "n = 3" are indistinguishable on this evidence, and the constant is the
  shorter explanation.  Such fits are emitted with a heavy cost penalty so the
  ranking prefers a real rule, exactly as ``cellwise`` pays for its capacity.
* **A counter must vary with the answer.**  If ``n`` is identical on every pair
  the hypothesis is memorisation, so it is only offered when at least two
  distinct values of ``n`` appear -- unless nothing else fits at all, in which
  case it is offered at the penalised cost.

Cost is description length in the portfolio's units: the counter and the two
laws are three choices, so the base cost sits just above the geometric
specialists and well below the enumerator.
"""

from collections import Counter

from .. import grid as G
from .. import objects as O
from ..task import Hyp

SOLVER = "select"          # ranks with the other "the answer is a selection"
                           # families; the portfolio prior is tuned for those
PHASE = 1

MAX_COUNT = 30             # a grid side longer than this is not a count


def _h(name, fn, cost):
    return Hyp("count:" + name, fn, cost, SOLVER)


# --------------------------------------------------------------------------
# counters.  Each is (name, fn(grid, bg) -> int or None)
# --------------------------------------------------------------------------

def _n_objects(mode):
    def f(g, bg):
        try:
            return len(O.segment(g, mode, bg))
        except Exception:
            return None
    return f


def _n_colors(g, bg):
    return len([c for c in G.palette(g) if c != bg])


def _n_cells(g, bg):
    return sum(1 for row in g for v in row if v != bg)


def _n_holes(g, bg):
    try:
        return len(G.holes(g, bg))
    except Exception:
        return None


def _max_object_size(mode):
    def f(g, bg):
        try:
            objs = O.segment(g, mode, bg)
        except Exception:
            return None
        return max((o.size for o in objs), default=None)
    return f


def _n_distinct_shapes(mode):
    def f(g, bg):
        try:
            objs = O.segment(g, mode, bg)
        except Exception:
            return None
        return len({o.norm_key() for o in objs}) or None
    return f


def _n_largest_color(g, bg):
    hist = [(v, c) for c, v in G.histogram(g).items() if c != bg]
    return max((v for v, _c in hist), default=None)


def _n_most_common_object_color(mode):
    def f(g, bg):
        try:
            objs = O.segment(g, mode, bg)
        except Exception:
            return None
        if not objs:
            return None
        counts = Counter(o.color for o in objs)
        return counts.most_common(1)[0][1]
    return f


def _n_rows_nonuniform(g, bg):
    return sum(1 for row in g if len(set(row)) > 1)


COUNTERS = (
    [("obj_" + m, _n_objects(m)) for m in ("c4", "c8", "m4", "m8", "color")] +
    [("shapes_" + m, _n_distinct_shapes(m)) for m in ("c4", "c8")] +
    [("maxsize_" + m, _max_object_size(m)) for m in ("c4", "c8")] +
    [("objcolor_" + m, _n_most_common_object_color(m)) for m in ("c4", "c8")] +
    [("colors", _n_colors), ("cells", _n_cells), ("holes", _n_holes),
     ("maxcolor", _n_largest_color), ("rows", _n_rows_nonuniform)]
)


# --------------------------------------------------------------------------
# colour laws.  Each is (name, fn(grid, bg) -> colour or None)
# --------------------------------------------------------------------------

def _c_most_common(g, bg):
    hist = [(v, c) for c, v in G.histogram(g).items() if c != bg]
    return max(hist)[1] if hist else None


def _c_least_common(g, bg):
    hist = [(v, c) for c, v in G.histogram(g).items() if c != bg]
    return min(hist)[1] if hist else None


def _c_largest_object(mode):
    def f(g, bg):
        try:
            objs = O.segment(g, mode, bg)
        except Exception:
            return None
        if not objs:
            return None
        return max(objs, key=lambda o: (o.size, -o.r0, -o.c0)).color
    return f


COLOR_LAWS = (
    [("most", _c_most_common), ("least", _c_least_common),
     ("bg", lambda g, bg: bg)] +
    [("largest_" + m, _c_largest_object(m)) for m in ("c4", "c8")]
)


# --------------------------------------------------------------------------
# shape laws.  Each maps (n, input dims) -> output dims
# --------------------------------------------------------------------------

SHAPE_LAWS = (
    ("nxn", lambda n, h, w: (n, n)),
    ("1xn", lambda n, h, w: (1, n)),
    ("nx1", lambda n, h, w: (n, 1)),
    ("nxw", lambda n, h, w: (n, w)),
    ("hxn", lambda n, h, w: (h, n)),
)


def _solid(h, w, c):
    if h <= 0 or w <= 0 or h > MAX_COUNT * 2 or w > MAX_COUNT * 2:
        return None
    return G.const_grid(h, w, c)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def _counts_for(ctx, counter, bg):
    """n for every training input, or None if the counter ever declines."""
    out = []
    for a in ctx.inputs:
        n = counter(a, bg)
        if n is None or not isinstance(n, int) or n <= 0 or n > MAX_COUNT:
            return None
        out.append(n)
    return out


def _fit_solid(ctx, bg, cname, counter, counts, varies):
    """Output is a solid block whose size is n and whose colour follows a law."""
    hyps = []
    for sname, shape in SHAPE_LAWS:
        ok = True
        for (a, b), n in zip(ctx.train, counts):
            ah, aw = G.dims(a)
            try:
                want = shape(n, ah, aw)
            except Exception:
                ok = False
                break
            if G.dims(b) != want or len(G.palette(b)) != 1:
                ok = False
                break
        if not ok:
            continue

        # colour: a constant, or one of the laws read off the input
        colours = {G.palette(b)[0] for _, b in ctx.train}
        laws = []
        if len(colours) == 1:
            const = colours.pop()
            laws.append(("const%d" % const, (lambda cc: lambda g, bgv: cc)(const)))
        for lname, law in COLOR_LAWS:
            good = True
            for a, b in ctx.train:
                if law(a, bg) != G.palette(b)[0]:
                    good = False
                    break
            if good:
                laws.append((lname, law))
        if not laws:
            continue

        lname, law = laws[0]
        cost = 4.0 + (0.0 if varies else 6.0)

        def make(shape=shape, law=law, counter=counter, bgv=bg):
            def run(g):
                n = counter(g, bgv)
                if n is None or n <= 0 or n > MAX_COUNT:
                    return None
                h, w = G.dims(g)
                try:
                    oh, ow = shape(n, h, w)
                except Exception:
                    return None
                c = law(g, bgv)
                if c is None:
                    return None
                return _solid(oh, ow, c)
            return run

        hyps.append(_h("%s.%s.%s" % (cname, sname, lname), make(), cost))
    return hyps


def _fit_repeat(ctx, bg, cname, counter, counts, varies):
    """Output is the input tiled or scaled n times."""
    hyps = []
    modes = (
        ("tile_h", lambda g, n: G.tile(g, 1, n)),
        ("tile_v", lambda g, n: G.tile(g, n, 1)),
        ("tile_hv", lambda g, n: G.tile(g, n, n)),
        ("scale", lambda g, n: G.upscale(g, n, n)),
        ("scale_h", lambda g, n: G.upscale(g, 1, n)),
        ("scale_v", lambda g, n: G.upscale(g, n, 1)),
    )
    for mname, fn in modes:
        ok = True
        for (a, b), n in zip(ctx.train, counts):
            try:
                got = fn(a, n)
            except Exception:
                ok = False
                break
            if got != b:
                ok = False
                break
        if not ok:
            continue
        cost = 4.5 + (0.0 if varies else 6.0)

        def make(fn=fn, counter=counter, bgv=bg):
            def run(g):
                n = counter(g, bgv)
                if n is None or n <= 0 or n > 12:
                    return None
                h, w = G.dims(g)
                if h * n > 60 or w * n > 60:
                    return None
                try:
                    return fn(g, n)
                except Exception:
                    return None
            return run

        hyps.append(_h("%s.%s" % (cname, mname), make(), cost))
    return hyps


def _fit_bar(ctx, bg, cname, counter, counts, varies):
    """Output is a fixed-shape grid with n cells filled from one end.

    "Show the count as a filled bar" -- the output shape is constant across the
    task and the number of coloured cells is the count.
    """
    shape = ctx.const_out_shape
    if not shape:
        return []
    oh, ow = shape
    if oh * ow > 400:
        return []
    fills = (
        ("rows", lambda n, h, w: [(r, c) for r in range(min(n, h)) for c in range(w)]),
        ("cols", lambda n, h, w: [(r, c) for c in range(min(n, w)) for r in range(h)]),
        ("rows_up", lambda n, h, w: [(h - 1 - r, c) for r in range(min(n, h)) for c in range(w)]),
        ("cols_right", lambda n, h, w: [(r, w - 1 - c) for c in range(min(n, w)) for r in range(h)]),
        ("cells", lambda n, h, w: [(i // w, i % w) for i in range(min(n, h * w))]),
    )
    hyps = []
    for fname, fill in fills:
        for cname2, law in ([("bgconst", None)] + list(COLOR_LAWS)):
            ok = True
            fg_const, bgc_const = set(), set()
            for (a, b), n in zip(ctx.train, counts):
                cells = set(fill(n, oh, ow))
                fg = {b[r][c] for r, c in cells}
                bgc = {b[r][c] for r in range(oh) for c in range(ow)
                       if (r, c) not in cells}
                if len(fg) != 1 or len(bgc) > 1:
                    ok = False
                    break
                want = fg.pop()
                if law is not None and law(a, bg) != want:
                    ok = False
                    break
                fg_const.add(want)
                bgc_const |= bgc
            if not ok:
                continue
            if law is None and len(fg_const) != 1:
                continue
            if len(bgc_const) > 1:
                continue
            fill_bg = bgc_const.pop() if bgc_const else bg
            fixed = fg_const.pop() if law is None else None
            cost = 5.0 + (0.0 if varies else 6.0)

            def make(fill=fill, law=law, fixed=fixed, fill_bg=fill_bg,
                     counter=counter, bgv=bg, oh=oh, ow=ow):
                def run(g):
                    n = counter(g, bgv)
                    if n is None or n <= 0 or n > MAX_COUNT:
                        return None
                    c = fixed if law is None else law(g, bgv)
                    if c is None:
                        return None
                    out = [[fill_bg] * ow for _ in range(oh)]
                    for r, cc in fill(n, oh, ow):
                        if 0 <= r < oh and 0 <= cc < ow:
                            out[r][cc] = c
                    return tuple(tuple(r) for r in out)
                return run

            hyps.append(_h("%s.bar_%s.%s" % (cname, fname, cname2), make(), cost))
            break                      # one colour law per fill is enough
    return hyps


def generate(ctx):
    bg = ctx.bg
    out = []
    for cname, counter in COUNTERS:
        if ctx.timed_out():
            break
        counts = _counts_for(ctx, counter, bg)
        if counts is None:
            continue
        # A counter that never changes cannot be distinguished from a constant.
        varies = len(set(counts)) > 1
        out.extend(_fit_solid(ctx, bg, cname, counter, counts, varies))
        out.extend(_fit_repeat(ctx, bg, cname, counter, counts, varies))
        out.extend(_fit_bar(ctx, bg, cname, counter, counts, varies))
        if len(out) > 120:
            break
    return out
