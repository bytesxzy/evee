"""Constraint-fitted and relational per-object translations.

"Everything red moves two down, everything blue moves one left" is a rule about
objects that no pixel rule and no in-place object edit can express: the object
survives, unchanged, somewhere else. We recover the displacement by matching
each input object to the output object with the same patch, then learn a table
from an object property to its displacement.

Identical objects are matched jointly through shared displacement constraints,
not greedily to the nearest copy. Relative rules align objects to an unambiguous
anchor; distances are recomputed on each grid, with no offline training.
"""

from itertools import product

from .. import grid as G
from .. import objects as O
from ..task import Hyp

SOLVER = "objects"

_SEGS = ("c8", "m8", "c4")
_KEYS = ("color", "size", "shape", "dims", "all")


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


def _key_of(o, kind):
    if kind == "color":
        return o.color
    if kind == "size":
        return o.size
    if kind == "shape":
        return o.mask
    if kind == "dims":
        return (o.height, o.width)
    return 0


def _fit(ctx, seg, kind, bg):
    domains = {}
    n = 0
    changed = False
    for a, b in ctx.train:
        if G.dims(a) != G.dims(b):
            return None
        abg = G.bg_or(a, bg)
        ins = O.segment(a, seg, abg)
        outs = O.segment(b, seg, abg)
        if not ins or len(ins) > 30 or len(ins) != len(outs):
            return None
        changed |= a != b
        by_patch = {}
        for p in outs:
            by_patch.setdefault(p.patch, []).append(p)
        for o in ins:
            # Every occurrence of a property must admit the SAME displacement.
            possible = {(p.r0 - o.r0, p.c0 - o.c0)
                        for p in by_patch.get(o.patch, ())}
            k = _key_of(o, kind)
            if k in domains:
                domains[k] &= possible
            else:
                domains[k] = possible
            if not domains[k]:
                return None
            n += 1
    if not changed or not domains or len(domains) * 2 > n:
        return None
    keys = list(domains)
    choices = [sorted(domains[k], key=lambda d: (abs(d[0]) + abs(d[1]), d))
               for k in keys]
    # Exact rendering enforces global correspondence, including occupancy.
    # Bounded ambiguity search prevents repeated shapes from exploding work.
    for i, values in enumerate(product(*choices)):
        if i >= 128 or ctx.timed_out():
            break
        table = dict(zip(keys, values))
        if all(_apply(a, seg, kind, table, bg) == b for a, b in ctx.train):
            return table
    return None


def _apply(g, seg, kind, table, bg):
    bg = G.bg_or(g, bg)
    objs = O.segment(g, seg, bg)
    if not objs or len(objs) > 30:
        return None
    h, w = G.dims(g)
    out = [[bg] * w for _ in range(h)]
    for o in objs:
        d = table.get(_key_of(o, kind))
        if d is None:
            return None
        for r, c in o.cells:
            nr, nc = r + d[0], c + d[1]
            if not (0 <= nr < h and 0 <= nc < w):
                return None
            if out[nr][nc] != bg:
                return None
            out[nr][nc] = g[r][c]
    return tuple(tuple(r) for r in out)


def generate(ctx):
    if not ctx.same_shape:
        return []
    res = []
    for bg in ([ctx.bg, None] if ctx.bg_varies else [ctx.bg]):
        for seg in _SEGS:
            if ctx.timed_out():
                break
            for kind in _KEYS:
                try:
                    t = _fit(ctx, seg, kind, bg)
                except Exception:
                    t = None
                if t is None:
                    continue
                res.append(_h("move_%s_by_%s" % (seg, kind),
                              (lambda s, k, t, b:
                               lambda g: _apply(g, s, k, t, b))(seg, kind, t, bg),
                              4.0 + 0.1 * len(t)))
        res.extend(_relative_rules(ctx, bg))
    return res


def _anchor(objs, selector):
    """An anchor must be uniquely identifiable; traversal order is no evidence."""
    if selector == "largest":
        size = max(o.size for o in objs)
        candidates = [o for o in objs if o.size == size]
    elif selector == "smallest":
        size = min(o.size for o in objs)
        candidates = [o for o in objs if o.size == size]
    else:
        candidates = [o for o in objs if o.color == int(selector.split("#")[1])]
    return candidates[0] if len(candidates) == 1 else None


def _offset(start, size, anchor_start, anchor_size, mode):
    if mode == "keep":
        return 0
    if mode == "near":
        target = anchor_start
    elif mode == "far":
        target = anchor_start + anchor_size - size
    elif mode == "center":
        # A half-cell centre gives two placements; do not break that tie.
        if (anchor_size - size) % 2:
            return None
        target = anchor_start + (anchor_size - size) // 2
    elif mode == "before":
        target = anchor_start - size
    elif mode == "after":
        target = anchor_start + anchor_size
    else:
        raise ValueError("unknown alignment mode")
    return target - start


def _apply_relative(g, seg, bg, selector, row_mode, col_mode):
    bg = G.bg_or(g, bg)
    objs = O.segment(g, seg, bg)
    if not 2 <= len(objs) <= 30:
        return None
    anchor = _anchor(objs, selector)
    if anchor is None:
        return None
    h, w = G.dims(g)
    out = [[bg] * w for _ in range(h)]
    occupied = set()
    for o in objs:
        if o is anchor:
            dr = dc = 0
        else:
            dr = _offset(o.r0, o.height, anchor.r0, anchor.height, row_mode)
            dc = _offset(o.c0, o.width, anchor.c0, anchor.width, col_mode)
        if dr is None or dc is None:
            return None
        for r, c in o.cells:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in occupied:
                return None
            occupied.add((nr, nc))
            out[nr][nc] = g[r][c]
    return tuple(tuple(row) for row in out)


def _relative_rules(ctx, bg):
    """Small reusable geometry grammar; accept only complete demonstrated rules."""
    if all(a == b for a, b in ctx.train):
        return []
    selectors = ["largest", "smallest"] + ["color#%d" % c
                  for c in sorted(ctx.in_palette) if c != bg]
    modes = ("near", "far", "center", "before", "after")
    alignments = [(m, "keep") for m in modes] + [("keep", m) for m in modes]
    alignments += [(r, c) for r in modes for c in modes
                   if (r in ("before", "after")) != (c in ("before", "after"))]
    res = []
    for seg in _SEGS:
        for selector in selectors:
            # Reject ambiguous anchor schemas before trying any alignments.
            scenes = [O.segment(a, seg, G.bg_or(a, bg)) for a in ctx.inputs]
            if any(not 2 <= len(objs) <= 30 or _anchor(objs, selector) is None
                   for objs in scenes):
                continue
            for rm, cm in alignments:
                if ctx.timed_out() or len(res) >= 12:
                    return res
                hyp = _h("align_%s_%s_%s_%s_bg%s" % (seg, selector, rm, cm, bg),
                         lambda g, s=seg, b=bg, a=selector, r=rm, c=cm:
                         _apply_relative(g, s, b, a, r, c),
                         5.0 + (0.3 if selector.startswith("color#") else 0.0)
                         + (0.4 if rm != "keep" and cm != "keep" else 0.0))
                if hyp.fits(ctx.train):
                    res.append(hyp)
    return res
