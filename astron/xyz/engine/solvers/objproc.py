"""Per-object *process* induction over a compositional action algebra.

``objects_map`` already learns "which colour does each object become".  That is
one action family out of many, and the narrowness is the problem: an ARC rule
like "slide every loose block down until it lands, delete the ones touching the
border, and outline what is left" is not expressible as a colour table, so it is
never proposed and therefore never ranked.

This module widens the action space and keeps the induction discipline:

1. segment the input under several (segmentation, background) readings;
2. for every object, *propose* actions from an algebra -- recolour, erase,
   bounding-box paints, halo, hole fill, translate, slide-until-blocked,
   snap-to-edge, in-place dihedral, ray casting -- and keep only those whose
   local effect is consistent with the demonstrated output;
3. find a low-capacity decision function from object features to action that is
   consistent with *every* object of *every* training pair simultaneously;
4. apply the whole labelling and require an exact match on all pairs.

Step 2 is what makes step 3 affordable.  Proposing blind and verifying globally
would be exponential in the number of objects; proposing locally collapses each
object to a small set of survivors first, so the global search is a consistency
problem over feature groups rather than a search over labellings.

Step 3 is deliberately capacity-limited.  With a dozen training objects and
twenty-odd features almost any labelling can be separated; only rules that
survive on one feature (or one pair) carry information, and a table with nearly
as many rows as it has training objects is rejected as memorisation.

Actions are pure descriptions -- ``(name, args...)`` tuples -- and are applied
in two passes, erasures before draws, so that an object moving into the space
another object vacated composes correctly instead of depending on iteration
order.
"""

from collections import Counter

from .. import grid as G
from .. import objects as O
from ..task import Hyp

SOLVER = "objects"

_SEGS = ("c4", "c8", "m4", "m8", "color", "g2m", "rows", "cols")

_FEATURE_KEYS = ("color", "size", "h", "w", "bbox", "square", "rect", "holes",
                 "ncolors", "border", "size_rank", "is_largest", "is_smallest",
                 "shape_freq", "shape_unique", "color_freq", "color_unique",
                 "row_band", "col_band", "container", "n_contains")

_PAIR_KEYS = ("color", "size_rank", "shape_unique", "is_largest", "holes",
              "border", "square", "color_unique")

_SHAPE_KEY = "__shape__"

_DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# structural cost of each action family; cheap actions are preferred when
# several explain the same evidence
_ACTION_COST = {
    "keep": 0.0, "del": 0.6, "solid": 0.8, "bbox": 1.2, "bboxout": 1.4,
    "bboxin": 1.4, "halo": 1.4, "fill": 1.2, "move": 1.6, "slide": 1.3,
    "edge": 1.3, "geom": 1.5, "ray": 1.6, "rays4": 1.6, "cross": 1.6,
    "patch": 3.2, "toward": 1.4, "step": 1.5, "into": 1.6,
    "copy": 1.7, "reflect": 1.5,
}


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


# ---------------------------------------------------------------------------
# the action algebra: an action turns (grid, object) into a set of cell writes
# ---------------------------------------------------------------------------

def _dist(a, b):
    """Chebyshev distance between two cell sets, cheap bbox bound first."""
    dr = max(a.r0 - b.r1, b.r0 - a.r1, 0)
    dc = max(a.c0 - b.c1, b.c0 - a.c1, 0)
    lo = max(dr, dc)
    if lo > 0 and (len(a.cells) > 60 or len(b.cells) > 60):
        return lo
    best = None
    for r1, c1 in a.cells:
        for r2, c2 in b.cells:
            d = max(abs(r1 - r2), abs(c1 - c2))
            if best is None or d < best:
                best = d
                if best <= lo:
                    return best
    return best if best is not None else 10 ** 6


def _resolve(color, scene, o=None):
    """Turn a relational colour reference into a concrete colour.

    A rule like "paint the shape the colour of the lone marker" cannot be
    written with a literal, because the literal changes from example to example.
    Naming the colour by *where it comes from* is what makes it a rule.
    """
    if not isinstance(color, tuple):
        return color
    kind = color[1]
    objs = scene.get("objs") or ()
    if not objs:
        return None
    if kind == "big":
        pick = max(objs, key=lambda x: (x.size, -x.r0, -x.c0))
    elif kind == "small":
        pick = min(objs, key=lambda x: (x.size, x.r0, x.c0))
    elif kind == "ucol":
        counts = Counter(x.color for x in objs)
        cands = [x for x in objs if counts[x.color] == 1]
        if len(cands) != 1:
            return None
        pick = cands[0]
    elif kind == "ushp":
        counts = Counter(x.norm_key() for x in objs)
        cands = [x for x in objs if counts[x.norm_key()] == 1]
        if len(cands) != 1:
            return None
        pick = cands[0]
    elif kind == "usz":
        counts = Counter(x.size for x in objs)
        cands = [x for x in objs if counts[x.size] == 1]
        if len(cands) != 1:
            return None
        pick = cands[0]
    elif kind in ("near", "far", "neardiff", "nearbig"):
        if o is None or len(objs) < 2:
            return None
        others = [x for x in objs if x is not o]
        if kind == "neardiff":
            # peers of the same colour are usually siblings, not anchors
            others = [x for x in others if x.color != o.color]
        elif kind == "nearbig":
            others = [x for x in others if x.size > o.size]
        if not others or len(others) > 40:
            return None
        ds = [(_dist(o, x), x) for x in others]
        want = (max(d for d, _x in ds) if kind == "far"
                else min(d for d, _x in ds))
        hits = {x.color for d, x in ds if d == want}
        if len(hits) != 1:
            return None                # ambiguous anchor is not a rule
        return hits.pop()
    else:
        return None
    return pick.color


_REL_COLORS = (("rel", "big"), ("rel", "small"), ("rel", "ucol"),
               ("rel", "ushp"), ("rel", "usz"), ("rel", "near"), ("rel", "far"),
               ("rel", "neardiff"), ("rel", "nearbig"))


def _writes(g, o, act, bg, scene=None):
    """Cell writes an action performs, or ``None`` when it does not apply."""
    kind = act[0]
    if len(act) > 1 and isinstance(act[1], tuple) and act[1][:1] == ("rel",):
        c = _resolve(act[1], scene or {}, o)
        if c is None:
            return None
        act = (kind, c) + tuple(act[2:])
    h, w = G.dims(g)
    if kind == "keep":
        return {}
    if kind == "del":
        return {rc: bg for rc in o.cells}
    if kind == "solid":
        return {rc: act[1] for rc in o.cells}
    if kind == "bbox":
        c = act[1]
        return {(r, cc): c for r in range(o.r0, o.r1 + 1)
                for cc in range(o.c0, o.c1 + 1)}
    if kind == "bboxout":
        c = act[1]
        out = {}
        for r in range(o.r0, o.r1 + 1):
            for cc in range(o.c0, o.c1 + 1):
                if r in (o.r0, o.r1) or cc in (o.c0, o.c1):
                    out[(r, cc)] = c
        return out
    if kind == "bboxin":
        c = act[1]
        out = {}
        for r in range(o.r0, o.r1 + 1):
            for cc in range(o.c0, o.c1 + 1):
                if (r, cc) not in o.cells:
                    out[(r, cc)] = c
        return out
    if kind == "halo":
        c = act[1]
        out = {}
        for r, cc in o.cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, c2 = r + dr, cc + dc
                    if 0 <= rr < h and 0 <= c2 < w and (rr, c2) not in o.cells \
                            and g[rr][c2] == bg:
                        out[(rr, c2)] = c
        return out or None
    if kind == "fill":
        c = act[1]
        out = {}
        for (r, cc) in _interior(o, g, bg):
            out[(r, cc)] = c
        return out or None
    if kind in ("move", "slide", "edge"):
        if kind == "move":
            dr, dc = act[1], act[2]
        elif kind == "slide":
            dr, dc = _slide_offset(g, o, act[1], bg)
        else:
            dr, dc = _edge_offset(g, o, act[1])
        if dr == 0 and dc == 0:
            return None
        out = {rc: bg for rc in o.cells}
        for r, cc in o.cells:
            rr, c2 = r + dr, cc + dc
            if not (0 <= rr < h and 0 <= c2 < w):
                return None
            out[(rr, c2)] = g[r][cc]
        return out
    if kind in ("toward", "step", "into"):
        off = _anchor_offset(g, o, act, bg, (scene or {}).get("objs") or ())
        if off is None:
            return None
        dr, dc = off
        if dr == 0 and dc == 0:
            return None
        out = {rc: bg for rc in o.cells}
        for r, cc in o.cells:
            rr, c2 = r + dr, cc + dc
            if not (0 <= rr < h and 0 <= c2 < w):
                return None
            out[(rr, c2)] = g[r][cc]
        return out
    if kind == "copy":
        # the object stays and a duplicate appears elsewhere; "move" cannot say
        # this, and duplication is a large ARC family on its own
        dr, dc = act[1], act[2]
        out = {}
        for r, cc in o.cells:
            rr, c2 = r + dr, cc + dc
            if not (0 <= rr < h and 0 <= c2 < w):
                return None
            out[(rr, c2)] = g[r][cc]
        return out
    if kind == "reflect":
        return _reflect_writes(g, o, act[1], bg, h, w)
    if kind == "geom":
        return _geom_writes(g, o, act[1], bg)
    if kind in ("ray", "rays4", "cross"):
        return _ray_writes(g, o, act, bg, h, w)
    if kind == "patch":
        # a stencil learned from the demonstrations: the content the output puts
        # in the object's bounding box, optionally grown by ``pad`` cells so the
        # stencil can also draw just outside the object
        pad, rows = act[1], act[2]
        r0, c0 = o.r0 - pad, o.c0 - pad
        out = {}
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                if v is None:
                    continue
                r, c = r0 + i, c0 + j
                if not (0 <= r < h and 0 <= c < w):
                    return None
                out[(r, c)] = v
        return out
    return None


def _interior(o, g, bg):
    """Cells of the object's bounding box enclosed by it (its holes)."""
    key = ("interior", bg)
    cached = o._holes.get(key)
    if cached is not None:
        return cached
    hh = o.r1 - o.r0 + 1
    ww = o.c1 - o.c0 + 1
    inside = [[True] * ww for _ in range(hh)]
    stack = []
    for r in range(hh):
        for c in (0, ww - 1):
            stack.append((r, c))
    for c in range(ww):
        for r in (0, hh - 1):
            stack.append((r, c))
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


def _pick_anchor(objs, o, kind):
    others = [x for x in objs if x is not o]
    if not others:
        return None
    if kind == "big":
        return max(others, key=lambda x: (x.size, -x.r0, -x.c0))
    if kind == "small":
        return min(others, key=lambda x: (x.size, x.r0, x.c0))
    if kind == "diff":
        cands = [x for x in others if x.color != o.color]
        if not cands:
            return None
        ds = [(_dist(o, x), x.r0, x.c0, x) for x in cands]
        ds.sort(key=lambda t: t[:3])
        if len(ds) > 1 and ds[0][0] == ds[1][0] and ds[0][3].color != ds[1][3].color:
            return None                # ambiguous anchor is not a rule
        return ds[0][3]
    if kind == "bigger":
        cands = [x for x in others if x.size > o.size]
        if not cands:
            return None
        ds = sorted(((_dist(o, x), x.r0, x.c0, x) for x in cands),
                    key=lambda t: t[:3])
        return ds[0][3]
    return None


def _anchor_offset(g, o, act, bg, objs):
    """Displacement that carries ``o`` to its anchor.

    Three readings of "toward", because ARC uses all of them: slide until you
    touch it, take one step, or land inside it.  The direction is read off the
    two bounding boxes rather than the centroids alone, so an object that is
    already aligned on one axis moves purely along the other.
    """
    kind, anchor_kind = act[0], act[1]
    anchor = _pick_anchor(objs, o, anchor_kind)
    if anchor is None:
        return None
    if kind == "into":
        dr = (anchor.r0 + anchor.r1 - o.r0 - o.r1) // 2
        dc = (anchor.c0 + anchor.c1 - o.c0 - o.c1) // 2
        return (dr, dc)
    overlap_cols = o.c0 <= anchor.c1 and anchor.c0 <= o.c1
    overlap_rows = o.r0 <= anchor.r1 and anchor.r0 <= o.r1
    sr = (anchor.r0 + anchor.r1) - (o.r0 + o.r1)
    sc = (anchor.c0 + anchor.c1) - (o.c0 + o.c1)
    if overlap_cols and not overlap_rows:
        dr, dc = (1 if sr > 0 else -1), 0
    elif overlap_rows and not overlap_cols:
        dr, dc = 0, (1 if sc > 0 else -1)
    elif not overlap_rows and not overlap_cols:
        dr, dc = (1 if sr > 0 else -1), (1 if sc > 0 else -1)
    else:
        return None
    if kind == "step":
        return (dr, dc)
    h, w = G.dims(g)
    blocked = set()
    for x in objs:
        if x is not o:
            blocked |= x.cells
    k = 0
    while True:
        n = k + 1
        for r, c in o.cells:
            rr, cc = r + dr * n, c + dc * n
            if not (0 <= rr < h and 0 <= cc < w) or (rr, cc) in blocked:
                return (dr * k, dc * k)
        k = n
        if k > h + w:
            return (0, 0)


def _slide_offset(g, o, d, bg):
    """Largest shift in ``d`` before the object hits a non-background cell."""
    dr, dc = _DIRS[d]
    h, w = G.dims(g)
    k = 0
    while True:
        n = k + 1
        for r, c in o.cells:
            rr, cc = r + dr * n, c + dc * n
            if not (0 <= rr < h and 0 <= cc < w):
                return (dr * k, dc * k)
            if (rr, cc) not in o.cells and g[rr][cc] != bg:
                return (dr * k, dc * k)
        k = n
        if k > h + w:
            return (0, 0)


def _edge_offset(g, o, d):
    h, w = G.dims(g)
    if d == "up":
        return (-o.r0, 0)
    if d == "down":
        return (h - 1 - o.r1, 0)
    if d == "left":
        return (0, -o.c0)
    return (0, w - 1 - o.c1)


_GEOMS = {"rot180": G.rot180, "flip_h": G.flip_h, "flip_v": G.flip_v,
          "rot90": G.rot90, "rot270": G.rot270, "transpose": G.transpose}


def _reflect_writes(g, o, axis, bg, h, w):
    """A mirrored duplicate of the object about a grid axis."""
    out = {}
    for r, c in o.cells:
        if axis == "h":
            rr, cc = r, w - 1 - c
        elif axis == "v":
            rr, cc = h - 1 - r, c
        elif axis == "both":
            rr, cc = h - 1 - r, w - 1 - c
        elif axis == "d" and h == w:
            rr, cc = c, r
        else:
            return None
        if (rr, cc) in o.cells:
            continue
        out[(rr, cc)] = g[r][c]
    return out or None


def _geom_writes(g, o, name, bg):
    fn = _GEOMS.get(name)
    if fn is None:
        return None
    hh, ww = o.height, o.width
    if name in ("rot90", "rot270", "transpose") and hh != ww:
        return None
    patch = o.patch                      # None outside the object's own cells
    filled = tuple(tuple(bg if v is None else v for v in row) for row in patch)
    mask = o.mask
    tp = fn(filled)
    tm = fn(mask)
    if G.dims(tp) != (hh, ww):
        return None
    out = {rc: bg for rc in o.cells}
    for r in range(hh):
        for c in range(ww):
            if tm[r][c]:
                out[(o.r0 + r, o.c0 + c)] = tp[r][c]
    return out


def _ray_writes(g, o, act, bg, h, w):
    kind = act[0]
    if kind == "ray":
        dirs, c = [act[1]], act[2]
    elif kind == "rays4":
        dirs, c = ["up", "down", "left", "right"], act[1]
    else:
        dirs, c = ["up", "down", "left", "right"], act[1]
    out = {}
    for d in dirs:
        dr, dc = _DIRS[d]
        for r, cc in o.cells:
            rr, c2 = r + dr, cc + dc
            while 0 <= rr < h and 0 <= c2 < w:
                if (rr, c2) in o.cells:
                    break
                if g[rr][c2] != bg:
                    break
                out[(rr, c2)] = c
                rr += dr
                c2 += dc
    return out or None


# ---------------------------------------------------------------------------
# proposal and local consistency
# ---------------------------------------------------------------------------

def _candidate_actions(colors, geoms=True):
    acts = [("keep",), ("del",)]
    for c in colors:
        acts.append(("solid", c))
        acts.append(("bbox", c))
        acts.append(("fill", c))
        acts.append(("halo", c))
        acts.append(("bboxout", c))
        acts.append(("bboxin", c))
    for d in ("up", "down", "left", "right"):
        acts.append(("slide", d))
        acts.append(("edge", d))
    if geoms:
        for n in ("rot180", "flip_h", "flip_v", "rot90", "rot270", "transpose"):
            acts.append(("geom", n))
    for c in colors:
        acts.append(("rays4", c))
        for d in ("up", "down", "left", "right"):
            acts.append(("ray", d, c))
    return acts


def _match_offsets(g, out, o, bg, limit=4, raw=False):
    """Displacements at which the object's exact patch reappears in the output."""
    h, w = G.dims(g)
    found = []
    hh, ww = o.height, o.width
    cells = [(r - o.r0, c - o.c0, g[r][c]) for r, c in o.cells]
    for r0 in range(0, h - hh + 1):
        for c0 in range(0, w - ww + 1):
            dr, dc = r0 - o.r0, c0 - o.c0
            if dr == 0 and dc == 0:
                continue
            ok = True
            for r, c, v in cells:
                if out[r0 + r][c0 + c] != v:
                    ok = False
                    break
            if ok:
                found.append((abs(dr) + abs(dc), dr, dc))
                if len(found) > 24:
                    break
        if len(found) > 24:
            break
    found.sort()
    if raw:
        return found[:limit]
    return [("move", dr, dc) for _d, dr, dc in found[:limit]]


def _stencils(g, out, o, pads=(0, 1)):
    """Read the output's content around the object as a reusable stencil.

    This is the dictionary end of the algebra: instead of guessing a paint
    operation, take what the demonstration actually drew there.  It is powerful
    enough to be dangerous, so it costs more than any generated action and the
    induction below refuses a stencil that only ever applies to one object.
    """
    h, w = G.dims(g)
    acts = []
    for pad in pads:
        r0, c0 = o.r0 - pad, o.c0 - pad
        r1, c1 = o.r1 + pad, o.c1 + pad
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        if (r1 - r0 + 1) * (c1 - c0 + 1) > 90:
            continue
        rows = tuple(tuple(out[r][c] for c in range(c0, c1 + 1))
                     for r in range(r0, r1 + 1))
        acts.append(("patch", pad, rows))
    return acts


def _consistent(g, out, o, writes):
    if writes is None:
        return False
    for (r, c), v in writes.items():
        if out[r][c] != v:
            return False
    for rc in o.cells:
        if rc not in writes and out[rc[0]][rc[1]] != g[rc[0]][rc[1]]:
            return False
    return True


def _apply_labelling(g, objs, actions, bg):
    """Two passes: every erasure first, then every draw."""
    out = [list(r) for r in g]
    draws = []
    scene = {"objs": objs}
    for o, act in zip(objs, actions):
        wr = _writes(g, o, act, bg, scene)
        if wr is None:
            return None
        for rc, v in wr.items():
            if v == bg:
                out[rc[0]][rc[1]] = v
            else:
                draws.append((rc, v))
    for (r, c), v in draws:
        out[r][c] = v
    return tuple(tuple(r) for r in out)


# ---------------------------------------------------------------------------
# the learned rule
# ---------------------------------------------------------------------------

class _ProcRule:
    __slots__ = ("seg", "bg", "keys", "table", "default", "maxobj")

    def __init__(self, seg, bg, keys, table, default, maxobj=80):
        self.seg = seg
        self.bg = bg
        self.keys = keys
        self.table = table
        self.default = default
        self.maxobj = maxobj

    def __call__(self, g):
        bg = G.bg_or(g, self.bg)
        objs = O.segment(g, self.seg, bg)
        if not objs or len(objs) > self.maxobj:
            return None
        shared = O._shared_stats(objs, g)
        acts = []
        for o in objs:
            k = _key_of(o, objs, g, self.keys, shared)
            act = self.table.get(k, self.default)
            if act is None:
                return None
            acts.append(act)
        return _apply_labelling(g, objs, acts, bg)


def _key_of(o, objs, g, keys, shared):
    if not keys:
        return ()
    feats = None
    vals = []
    for k in keys:
        if k == _SHAPE_KEY:
            vals.append(o.norm_key())
        else:
            if feats is None:
                feats = O.object_features(o, objs, g, shared)
            vals.append(feats[k])
    return tuple(vals)


# ---------------------------------------------------------------------------
# induction
# ---------------------------------------------------------------------------

def _bg_candidates(ctx):
    cands = [ctx.bg]
    if ctx.bg_varies:
        cands.append(None)
    counts = Counter()
    for a in ctx.inputs:
        for c, n in G.histogram(a).items():
            counts[c] += n
    for c, _n in counts.most_common(2):
        if c not in cands:
            cands.append(c)
    return cands[:3]


def _colors(ctx):
    cols = set(ctx.out_palette) | {ctx.bg}
    for a, b in ctx.train:
        for ra, rb in zip(a, b):
            for x, y in zip(ra, rb):
                if x != y:
                    cols.add(y)
    return sorted(cols)[:9]


def _penalty(ctx):
    """Task-level prior over action families.

    When no demonstration ever removes anything, an action that removes
    something is not a cheap explanation -- it is a wrong one that happens to
    look cheap locally.
    """
    pen = {}
    if all(G.histogram(a) == G.histogram(b) for a, b in ctx.train):
        pen["del"] = 2.5
        pen["solid"] = 1.5
        pen["bbox"] = 1.5
    return pen


def _observe(ctx, seg, bg, colors, base_acts):
    """Per-object survivor sets, or ``None`` when this reading does not apply."""
    rows = []
    for a, b in ctx.train:
        if G.dims(a) != G.dims(b):
            return None
        abg = G.bg_or(a, bg)
        objs = O.segment(a, seg, abg)
        if not objs or len(objs) > 45:
            return None
        shared = O._shared_stats(objs, a)
        per = []
        for o in objs:
            scene = {"objs": objs}
            ok = set()
            for act in base_acts:
                if _consistent(a, b, o, _writes(a, o, act, abg, scene)):
                    ok.add(act)
            for ref in _REL_COLORS:
                for kind in ("solid", "fill", "bbox"):
                    act = (kind, ref)
                    if _consistent(a, b, o, _writes(a, o, act, abg, scene)):
                        ok.add(act)
            for kind in ("toward", "step", "into"):
                for anchor in ("big", "small", "diff", "bigger"):
                    act = (kind, anchor)
                    if _consistent(a, b, o, _writes(a, o, act, abg, scene)):
                        ok.add(act)
            for axis in ("h", "v", "both", "d"):
                act = ("reflect", axis)
                if _consistent(a, b, o, _writes(a, o, act, abg, scene)):
                    ok.add(act)
            for _k, dr, dc in _match_offsets(a, b, o, abg, limit=6, raw=True):
                act = ("copy", dr, dc)
                if _consistent(a, b, o, _writes(a, o, act, abg, scene)):
                    ok.add(act)
            for act in _match_offsets(a, b, o, abg):
                if act not in ok and _consistent(a, b, o,
                                                 _writes(a, o, act, abg, scene)):
                    ok.add(act)
            for act in _stencils(a, b, o):
                ok.add(act)
            if not ok:
                return None
            per.append((o, ok))
        rows.append((a, b, objs, shared, per))
    return rows


def _cost_of(act):
    base = _ACTION_COST.get(act[0], 2.0)
    if len(act) > 1 and isinstance(act[1], tuple) and act[1][:1] == ("rel",):
        base -= 0.25          # names its colour instead of hard-coding one
    return base


def _rank(ok, penalty=None):
    """Cheapest first, with the task-level prior folded in."""
    def key(a):
        c = _cost_of(a)
        if penalty:
            c += penalty.get(a[0], 0.0)
        return (c, len(a), repr(a))
    return sorted(ok, key=key)


def _pick(ok, penalty=None):
    return _rank(ok, penalty)[0]


def _tables(ranked, limit=14):
    """Assemble candidate tables from the per-group shortlists.

    Taking the cheapest survivor in every group is not enough.  Local
    consistency cannot tell "this object was deleted" from "this object moved
    away": both leave background behind where it used to be, and deletion is
    the cheaper reading, so it wins the group and the assembled rule then fails
    globally.  The cheapest assembly is therefore tried first, then single
    substitutions from each group's shortlist.  One group being wrong is
    overwhelmingly the common case, so this stays linear in the number of
    groups rather than exponential.
    """
    keys = list(ranked)
    base = {k: ranked[k][0] for k in keys}
    out = [base]
    for k in keys:
        for alt in ranked[k][1:]:
            t = dict(base)
            t[k] = alt
            out.append(t)
            if len(out) >= limit:
                return out
    if keys and all(len(ranked[k]) > 1 for k in keys):
        out.append({k: ranked[k][1] for k in keys})
    return out[:limit]


def _induce(rows, keys, penalty=None):
    """Per-group action shortlists consistent with every object, or ``None``."""
    groups = {}
    sizes = Counter()
    n = 0
    for a, _b, objs, shared, per in rows:
        for o, ok in per:
            n += 1
            k = _key_of(o, objs, a, keys, shared)
            sizes[k] += 1
            cur = groups.get(k)
            groups[k] = ok if cur is None else (cur & ok)
            if not groups[k]:
                return None
    if not groups:
        return None
    if keys and len(groups) > max(2, int(0.7 * n)):
        return None                      # a row per object is memorisation
    ranked = {k: _rank(v, penalty)[:3] for k, v in groups.items()}
    # a stencil that was only ever seen once is a copy of the answer, not a rule
    for k, opts in ranked.items():
        if opts[0][0] == "patch" and sizes[k] < 2:
            return None
    return ranked, n


def _verify(rule, ctx):
    for a, b in ctx.train:
        if rule(a) != b:
            return False
    return True


def _modal(table):
    c = Counter(table.values())
    return c.most_common(1)[0][0] if c else None


def generate(ctx):
    if not ctx.same_shape:
        return []
    colors = _colors(ctx)
    base = _candidate_actions(colors)
    penalty = _penalty(ctx)
    out, seen = [], set()
    for bg in _bg_candidates(ctx):
        if ctx.timed_out():
            break
        for seg in _SEGS:
            if ctx.timed_out():
                break
            try:
                rows = _observe(ctx, seg, bg, colors, base)
            except Exception:
                rows = None
            if not rows:
                continue
            keysets = [()]
            keysets.extend((k,) for k in _FEATURE_KEYS)
            keysets.append((_SHAPE_KEY,))
            found_single = False
            for keys in keysets:
                if ctx.timed_out():
                    break
                got = _induce(rows, keys, penalty)
                if got is None:
                    continue
                ranked, _n = got
                hit = False
                for table in _tables(ranked):
                    if ctx.timed_out() or hit:
                        break
                    if keys and len(set(table.values())) < 2:
                        continue         # degenerate: same as the uniform rule
                    defaults = [None, _modal(table)] if keys else [table[()]]
                    for default in defaults:
                        rule = _ProcRule(seg, bg, keys, table, default)
                        try:
                            if not _verify(rule, ctx):
                                continue
                            sig = tuple(rule(g) for g in ctx.test_inputs)
                        except Exception:
                            continue
                        if sig in seen:
                            hit = True
                            break
                        seen.add(sig)
                        cost = (2.2 + 0.7 * len(keys) + 0.28 * len(table)
                                + (1.6 if seg in ("rows", "cols") else 0.0)
                                + sum(_cost_of(a) for a in table.values())
                                / float(max(1, len(table)))
                                + (0.4 if default is not None else 0.0))
                        name = "proc[%s/%s/%s|%d]" % (
                            seg, "auto" if bg is None else bg,
                            "+".join(keys) if keys else "all", len(table))
                        out.append(_h(name, rule, cost))
                        if not keys:
                            found_single = True
                        hit = True
                        break
                if len(out) >= 40:
                    break
            if found_single and len(out) >= 12:
                break
        if len(out) >= 40:
            break
    if len(out) < 40 and not ctx.timed_out():
        out.extend(_pairs(ctx, colors, base, seen, 40 - len(out)))
    return out


def _pairs(ctx, colors, base, seen, room):
    """Two-feature tables, tried only when one feature was not enough."""
    out = []
    for bg in _bg_candidates(ctx)[:2]:
        for seg in _SEGS[:4]:
            if ctx.timed_out() or len(out) >= room:
                return out
            try:
                rows = _observe(ctx, seg, bg, colors, base)
            except Exception:
                rows = None
            if not rows:
                continue
            for i, ka in enumerate(_PAIR_KEYS):
                for kb in _PAIR_KEYS[i + 1:]:
                    if ctx.timed_out() or len(out) >= room:
                        return out
                    got = _induce(rows, (ka, kb), _penalty(ctx))
                    if got is None:
                        continue
                    ranked, _n = got
                    table = {k: v[0] for k, v in ranked.items()}
                    if len(set(table.values())) < 2:
                        continue
                    for default in (None, _modal(table)):
                        rule = _ProcRule(seg, bg, (ka, kb), table, default)
                        try:
                            if not _verify(rule, ctx):
                                continue
                            sig = tuple(rule(g) for g in ctx.test_inputs)
                        except Exception:
                            continue
                        if sig in seen:
                            continue
                        seen.add(sig)
                        cost = (3.4 + 0.3 * len(table)
                                + (1.6 if seg in ("rows", "cols") else 0.0)
                                + (0.4 if default is not None else 0.0))
                        out.append(_h("proc2[%s/%s/%s+%s|%d]" % (
                            seg, "auto" if bg is None else bg, ka, kb,
                            len(table)), rule, cost))
                        break
    return out
