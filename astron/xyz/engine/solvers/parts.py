"""Induce a small vocabulary of parts and solve their joint exact cover.

This makes subobjects reachable when touching parts have merged into one input
component. A part must recur in at least two demonstrations; an ambiguous cover
abstains. No part is indexed by an example or a task, and every rule is verified.
"""
from collections import defaultdict
from functools import lru_cache
from .. import grid as G, objects as O
from ..task import Hyp

SOLVER = "objects"


def _variants(mask):
    return tuple(sorted({fn(mask) for _, fn in G.DIHEDRAL}))


def _vocabulary(ctx, bg):
    support = defaultdict(set)
    for index, (a, b) in enumerate(ctx.train):
        abg = G.bg_or(a, bg)
        if len(G.palette(a) - {abg}) != 1:
            return None
        if any((x == abg) != (y == abg) for ra, rb in zip(a, b) for x, y in zip(ra, rb)):
            return None
        for obj in O.segment(b, "c4", abg):
            if 2 <= obj.size <= 16:
                canonical = min(_variants(obj.mask))
                support[(canonical, obj.color)].add(index)
    prototypes = [(mask, col) for (mask, col), examples in sorted(support.items())
                  if len(examples) >= 2]
    if not 2 <= len(prototypes) <= 6 or len({c for _, c in prototypes}) < 2:
        return None
    return tuple(prototypes)


def _cover(cells, prototypes, max_states=4000):
    cells = sorted(cells)
    if not cells or len(cells) > 180:
        return None
    indices = {rc: i for i, rc in enumerate(cells)}
    colors = sorted({c for _, c in prototypes})
    color_index = {c: i for i, c in enumerate(colors)}
    r0, c0, r1, c1 = G.bbox_of(cells)
    placements, seen = [], set()
    for patch, col in prototypes:
        for mask in _variants(patch):
            offsets = [(r, c) for r, row in enumerate(mask) for c, v in enumerate(row) if v]
            mh, mw = G.dims(mask)
            for y in range(r0, r1-mh+2):
                for x in range(c0, c1-mw+2):
                    coords = [(y+r, x+c) for r, c in offsets]
                    if not all(rc in indices for rc in coords):
                        continue
                    bits = sum(1 << indices[rc] for rc in coords)
                    rec = (bits, color_index[col])
                    if rec not in seen:
                        seen.add(rec)
                        placements.append(rec)
    by_cell = [[] for _ in cells]
    for bits, color in placements:
        cur = bits
        while cur:
            low = cur & -cur
            by_cell[low.bit_length()-1].append((bits, color))
            cur ^= low
    if any(not opts for opts in by_cell):
        return None
    visited = 0

    @lru_cache(maxsize=4000)
    def search(left):
        nonlocal visited
        visited += 1
        if visited > max_states:
            raise OverflowError("part-cover state cap")
        if not left:
            return ((0,) * len(colors),)
        best = None
        cur = left
        while cur:
            low = cur & -cur
            options = [(bits, c) for bits, c in by_cell[low.bit_length()-1] if bits & left == bits]
            if not options:
                return ()
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
            cur ^= low
        answers = set()
        for bits, c in best:
            for rest in search(left ^ bits):
                result = rest[:c] + (rest[c] | bits,) + rest[c+1:]
                answers.add(result)
                if len(answers) == 2:
                    return tuple(sorted(answers))
        return tuple(answers)

    try:
        answers = search((1 << len(cells))-1)
    except (OverflowError, RecursionError):
        return None
    if len(answers) != 1:
        return None
    writes = {}
    for c, bits in zip(colors, answers[0]):
        for i, rc in enumerate(cells):
            if bits & (1 << i):
                writes[rc] = c
    return writes


class Rule:
    def __init__(self, prototypes, bg):
        self.prototypes, self.bg = prototypes, bg

    def __call__(self, g):
        bg = G.bg_or(g, self.bg)
        if len(G.palette(g)-{bg}) != 1:
            return None
        out = [list(row) for row in g]
        for obj in O.segment(g, "m4", bg):
            writes = _cover(obj.cells, self.prototypes)
            if writes is None:
                return None
            for (r, c), value in writes.items():
                out[r][c] = value
        return tuple(tuple(row) for row in out)


def generate(ctx):
    if not ctx.same_shape:
        return []
    results = []
    for bg in [ctx.bg] + ([None] if ctx.bg_varies else []):
        if ctx.timed_out():
            break
        prototypes = _vocabulary(ctx, bg)
        if not prototypes:
            continue
        rule = Rule(prototypes, bg)
        if any(rule(a) != b for a, b in ctx.train):
            continue
        cost = 4.0 + .5*len(prototypes) + .08*sum(sum(map(sum, mask)) for mask, _ in prototypes)
        results.append(Hyp("parts[%s/%s]" % (bg, repr(prototypes)), rule, cost, SOLVER))
    return results

