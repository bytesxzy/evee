"""Per-object edits learned as a decision function over object features.

Pipeline:

1. segment the input;
2. read off, for each object, the *action* the target output applied to it
   (keep / paint a solid colour / delete / fill its bounding box);
3. find a single object feature (or a pair) whose value determines the action
   consistently across every training object;
4. at test time, segment, evaluate the feature, apply the action.

Step 3 is deliberately restricted to low-capacity hypotheses.  With a dozen
training objects, any sufficiently rich feature set can separate them; only
rules that survive on one or two features carry information.
"""

from collections import Counter

from .. import grid as G
from .. import objects as O
from ..task import Hyp

SOLVER = "objects"

_SEGS = ("c4", "c8", "m4", "m8", "color", "g2", "g2m", "rows", "cols")

_FEATURE_KEYS = ("color", "size", "h", "w", "bbox", "square", "rect", "holes",
                 "ncolors", "border", "size_rank", "is_largest", "is_smallest",
                 "shape_freq", "shape_unique", "color_freq", "color_unique",
                 "r0", "c0", "row_band", "col_band", "container", "n_contains")

_SHAPE_KEY = "__shape__"


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


# --- actions ---------------------------------------------------------------
# An action is a small tuple; ``_do`` applies it to a mutable grid.

def _read_action(inp, outp, o, bg):
    """What did the output do to this object? ``None`` if not expressible."""
    vals = {outp[r][c] for r, c in o.cells}
    same = all(outp[r][c] == inp[r][c] for r, c in o.cells)
    if same:
        # distinguish 'keep' from 'bbox filled with own colour'
        return ("keep",)
    if len(vals) == 1:
        v = vals.pop()
        return ("del",) if v == bg else ("solid", v)
    # per-cell colour permutation inside the object
    m = {}
    for r, c in o.cells:
        a, b = inp[r][c], outp[r][c]
        if m.setdefault(a, b) != b:
            return None
    return ("cmap", tuple(sorted(m.items())))


def _do(out, inp, o, act, bg):
    if act[0] == "keep":
        return
    if act[0] == "del":
        for r, c in o.cells:
            out[r][c] = bg
    elif act[0] == "solid":
        for r, c in o.cells:
            out[r][c] = act[1]
    elif act[0] == "cmap":
        m = dict(act[1])
        for r, c in o.cells:
            out[r][c] = m.get(inp[r][c], inp[r][c])
    elif act[0] == "bbox":
        for r in range(o.r0, o.r1 + 1):
            for c in range(o.c0, o.c1 + 1):
                out[r][c] = act[1]


def _feat(o, objs, grid, key):
    if key == _SHAPE_KEY:
        return o.norm_key()
    return O.object_features(o, objs, grid)[key]


class _ObjRule:
    __slots__ = ("seg", "bg", "keys", "table", "default")

    def __init__(self, seg, bg, keys, table, default):
        self.seg = seg
        self.bg = bg
        self.keys = keys
        self.table = table
        self.default = default

    def __call__(self, g):
        bg = G.bg_or(g, self.bg)
        objs, feats = O.features_of(g, self.seg, bg)
        if not objs or len(objs) > 120:
            return None
        out = [list(r) for r in g]
        for i, o in enumerate(objs):
            if self.keys == (_SHAPE_KEY,):
                k = (o.norm_key(),)
            else:
                k = tuple(feats[i][x] for x in self.keys)
            act = self.table.get(k, self.default)
            if act is None:
                return None
            _do(out, g, o, act, bg)
        return tuple(tuple(r) for r in out)


def _bg_candidates(ctx):
    """Background colours worth segmenting against.

    Using the task background finds the *objects*; using a wall colour instead
    finds the *regions those walls enclose*, which is what tasks like "colour
    each pen by how big it is" are actually about.
    """
    cands = [ctx.bg]
    cnt = Counter()
    for a in ctx.inputs:
        for c, n in G.histogram(a).items():
            cnt[c] += n
    for c, _n in cnt.most_common(3):
        if c not in cands and all(c in G.palette(a) for a in ctx.all_inputs):
            cands.append(c)
    # A colour that rules the grid as a lattice is the most likely "wall", and
    # it is not always among the most frequent -- a thin grid of lines is a
    # small fraction of the cells.
    try:
        from .partition import sep_color_of
        seps = {sep_color_of(a) for a in ctx.all_inputs}
        if len(seps) == 1:
            sc = seps.pop()
            if sc is not None and sc not in cands:
                cands.append(sc)
    except Exception:
        pass
    if ctx.bg_varies:
        cands.append(None)          # resolve the background per grid
    return cands[:5]


def _learn(ctx, seg, bg, keys):
    table = {}
    nobj = 0
    for a, b in ctx.train:
        if G.dims(a) != G.dims(b):
            return None
        objs, feats = O.features_of(a, seg, bg)
        if not objs or len(objs) > 120:
            return None
        for i, o in enumerate(objs):
            act = _read_action(a, b, o, G.bg_or(a, bg))
            if act is None:
                return None
            k = (o.norm_key(),) if keys == (_SHAPE_KEY,) else tuple(feats[i][x] for x in keys)
            prev = table.get(k)
            if prev is None:
                table[k] = act
            elif prev != act:
                return None
            nobj += 1
    if not table or nobj < 2:
        return None
    # capacity guard
    if len(table) * 2 > nobj:
        return None
    default = None
    acts = Counter(table.values())
    if len(acts) > 1:
        default = acts.most_common(1)[0][0]
    return _ObjRule(seg, bg, keys, table, default)


def generate(ctx):
    res = []
    bg = ctx.bg
    # counting answers change the grid shape by construction, so they must be
    # offered before the same-shape gate below
    res.extend(_count_outputs(ctx, bg))
    if not ctx.same_shape:
        return res
    singles = [(k,) for k in _FEATURE_KEYS] + [(_SHAPE_KEY,)]
    pairs = [("size", "color"), ("color", "holes"), ("size", "holes"),
             ("h", "w"), ("color", "shape_freq"), ("size", "border"),
             ("color", "border"), ("ncolors", "size"),
             ("container", "size"), ("container", "color")]
    for bi, sbg in enumerate(_bg_candidates(ctx)):
        extra = 0.0 if bi == 0 else 1.0
        if sbg is None:
            extra = 0.5
        for seg in _SEGS:
            if ctx.timed_out():
                break
            # A row or column is present in every grid whether or not anything
            # is drawn in it, so a table over bands sees far more capacity per
            # observation than one over discovered objects.
            band = 1.6 if seg in ("rows", "cols") else 0.0
            for keys in singles:
                r = _learn(ctx, seg, sbg, keys)
                if r is not None:
                    res.append(_h("%s@%s.by_%s" % (seg, sbg, "+".join(keys)), r,
                                  3.5 + extra + band + 0.05 * len(r.table)))
            for keys in pairs:
                r = _learn(ctx, seg, sbg, keys)
                if r is not None:
                    res.append(_h("%s@%s.by_%s" % (seg, sbg, "+".join(keys)), r,
                                  5.0 + extra + band + 0.05 * len(r.table)))
    res.extend(_bbox_rules(ctx, bg))
    return res


# --- object counting -------------------------------------------------------

_COUNTERS = {
    "n_obj4": lambda g, bg: len(O.segment(g, "c4", bg)),
    "n_obj8": lambda g, bg: len(O.segment(g, "c8", bg)),
    "n_multi8": lambda g, bg: len(O.segment(g, "m8", bg)),
    "n_colors": lambda g, bg: len(G.palette(g) - {bg}),
    "max_size": lambda g, bg: max([o.size for o in O.segment(g, "c8", bg)] or [0]),
    "n_cells": lambda g, bg: sum(1 for r in g for v in r if v != bg),
}


def _count_outputs(ctx, bg):
    """Outputs that are an n x n / 1 x n block whose size is a count."""
    res = []
    for name, f in _COUNTERS.items():
        for shape in ("sq", "row", "col"):
            for cmode in ("fixed", "modal", "rarest"):
                res.append(_h("count_%s_%s_%s" % (name, shape, cmode),
                              (lambda f, shape, cmode, bg:
                               lambda g: _count_grid(g, f, shape, cmode, bg))(f, shape, cmode, bg),
                              6.5))
    return res


def _count_grid(g, f, shape, cmode, bg):
    bg = G.bg_or(g, bg)
    n = f(g, bg)
    if n < 1 or n > 30:
        return None
    if cmode == "fixed":
        c = 5
    else:
        hist = G.histogram(g)
        hist.pop(bg, None)
        if not hist:
            return None
        items = sorted(hist.items(), key=lambda kv: (kv[1], kv[0]))
        c = items[-1][0] if cmode == "modal" else items[0][0]
    if shape == "sq":
        return G.const_grid(n, n, c)
    if shape == "row":
        return G.const_grid(1, n, c)
    return G.const_grid(n, 1, c)


# --- bounding-box rendering ------------------------------------------------

def _bbox_rules(ctx, bg):
    res = []
    for seg in ("c4", "c8", "m8", "g2", "g2m"):
        for mode in ("fill_own", "outline_own", "fill_hole", "halo_own"):
            res.append(_h("%s.%s" % (seg, mode),
                          (lambda seg, mode, bg:
                           lambda g: _bbox_render(g, seg, mode, bg))(seg, mode, bg),
                          5.0))
        for c in sorted(ctx.out_palette):
            res.append(_h("%s.fill_bbox#%d" % (seg, c),
                          (lambda seg, c, bg:
                           lambda g: _bbox_render(g, seg, ("fill", c), bg))(seg, c, bg),
                          5.5))
            res.append(_h("%s.halo_bbox#%d" % (seg, c),
                          (lambda seg, c, bg:
                           lambda g: _bbox_render(g, seg, ("halo", c), bg))(seg, c, bg),
                          5.5))
    return res


def _bbox_render(g, seg, mode, bg):
    bg = G.bg_or(g, bg)
    objs = O.segment(g, seg, bg)
    if not objs or len(objs) > 120:
        return None
    out = [list(r) for r in g]
    for o in objs:
        if mode == "fill_own":
            for r in range(o.r0, o.r1 + 1):
                for c in range(o.c0, o.c1 + 1):
                    out[r][c] = o.color
        elif mode == "outline_own":
            for r in range(o.r0, o.r1 + 1):
                out[r][o.c0] = o.color
                out[r][o.c1] = o.color
            for c in range(o.c0, o.c1 + 1):
                out[o.r0][c] = o.color
                out[o.r1][c] = o.color
        elif mode == "fill_hole":
            m = o.mask
            hh, ww = o.height, o.width
            seen = [[False] * ww for _ in range(hh)]
            st = []
            for r in range(hh):
                for c in (0, ww - 1):
                    if not m[r][c] and not seen[r][c]:
                        seen[r][c] = True
                        st.append((r, c))
            for c in range(ww):
                for r in (0, hh - 1):
                    if not m[r][c] and not seen[r][c]:
                        seen[r][c] = True
                        st.append((r, c))
            while st:
                cr, cc = st.pop()
                for dr, dc in G.N4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < hh and 0 <= nc < ww and not seen[nr][nc] and not m[nr][nc]:
                        seen[nr][nc] = True
                        st.append((nr, nc))
            for r in range(hh):
                for c in range(ww):
                    if not m[r][c] and not seen[r][c]:
                        out[o.r0 + r][o.c0 + c] = o.color
        elif mode == "halo_own":
            for r in range(o.r0, o.r1 + 1):
                for c in range(o.c0, o.c1 + 1):
                    if (r, c) not in o.cells:
                        out[r][c] = o.color
        elif isinstance(mode, tuple) and mode[0] == "halo":
            # fill the bounding box *around* the object, leaving it intact
            for r in range(o.r0, o.r1 + 1):
                for c in range(o.c0, o.c1 + 1):
                    if (r, c) not in o.cells:
                        out[r][c] = mode[1]
        elif isinstance(mode, tuple):
            for r in range(o.r0, o.r1 + 1):
                for c in range(o.c0, o.c1 + 1):
                    out[r][c] = mode[1]
    return tuple(tuple(r) for r in out)
