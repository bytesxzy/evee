"""In-grid analogy: one complete exemplar, several partial copies.

A recurring ARC pattern is that the grid itself contains the demonstration --
a fully drawn motif somewhere, plus fragments of the same motif elsewhere that
must be completed.  The rule is not a function of the pixel or of the object in
isolation; it is "make the fragments look like the exemplar".

We take the richest object as the exemplar, then for every other object search
for a placement of some dihedral image (optionally upscaled) of the exemplar
that is consistent with every cell the fragment already has.  A placement is
accepted only if it is unique and conflicts with nothing else on the grid, so
ambiguous fragments are left alone rather than guessed at.
"""

from .. import grid as G
from .. import objects as O
from ..task import Hyp

SOLVER = "objects"
MAX_OBJ = 30


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


def _patch_cells(patch):
    return [(r, c, v) for r, row in enumerate(patch)
            for c, v in enumerate(row) if v is not None]


def _scale_patch(patch, k):
    if k == 1:
        return patch
    out = []
    for row in patch:
        nr = tuple(v for v in row for _ in range(k))
        out.extend([nr] * k)
    return tuple(out)


def _dihedral_patch(patch, name):
    f = G.DIHEDRAL_MAP[name]
    # the dihedral helpers work on rectangular tuples; None passes through
    return f(patch)


def _placements(grid, patch, frag, bg, h, w):
    """Offsets where ``patch`` explains every cell of ``frag`` without clash."""
    cells = _patch_cells(patch)
    if not cells:
        return []
    ph, pw = len(patch), len(patch[0])
    frag_cells = sorted(frag.cells)
    fr0, fc0 = frag_cells[0]
    fcolor = grid[fr0][fc0]
    out = []
    for pr, pc, pv in cells:
        if pv != fcolor:
            continue
        dr, dc = fr0 - pr, fc0 - pc
        if dr < 0 or dc < 0 or dr + ph > h or dc + pw > w:
            continue
        ok = True
        for r, c, v in cells:
            gr, gc = r + dr, c + dc
            cur = grid[gr][gc]
            if cur != bg and cur != v:
                ok = False
                break
        if not ok:
            continue
        for r, c in frag_cells:
            pv2 = patch[r - dr][c - dc] if (0 <= r - dr < ph and 0 <= c - dc < pw) else None
            if pv2 is None or pv2 != grid[r][c]:
                ok = False
                break
        if ok:
            out.append((dr, dc))
            if len(out) > 2:
                return out
    return out


def _complete(grid, bg, seg, pick, scales, keep_exemplar):
    bg = G.bg_or(grid, bg)
    objs = O.segment(grid, seg, bg)
    if len(objs) < 2 or len(objs) > MAX_OBJ:
        return None
    if pick == "size":
        ex = max(objs, key=lambda o: (o.size, o.bbox_area))
    elif pick == "colors":
        ex = max(objs, key=lambda o: (len(o.colors()), o.size))
    else:
        ex = max(objs, key=lambda o: (o.bbox_area, o.size))
    base = ex.patch
    h, w = G.dims(grid)
    out = [list(r) for r in grid]
    changed = False
    variants = []
    for dname, _f in G.DIHEDRAL:
        try:
            p = _dihedral_patch(base, dname)
        except Exception:
            continue
        for k in scales:
            sp = _scale_patch(p, k)
            if len(sp) <= h and len(sp[0]) <= w:
                variants.append(sp)
    seen = set()
    uniq = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    for o in objs:
        if o is ex:
            continue
        best = None
        nhit = 0
        for sp in uniq:
            pl = _placements(grid, sp, o, bg, h, w)
            if len(pl) == 1:
                nhit += 1
                best = (sp, pl[0])
                if nhit > 1:
                    break
        if nhit != 1 or best is None:
            continue
        sp, (dr, dc) = best
        for r, c, v in _patch_cells(sp):
            gr, gc = r + dr, c + dc
            if out[gr][gc] == bg:
                out[gr][gc] = v
                changed = True
    if not changed:
        return None
    return tuple(tuple(r) for r in out)


def generate(ctx):
    if not ctx.same_shape:
        return []
    res = []
    for bg in ([ctx.bg, None] if ctx.bg_varies else [ctx.bg]):
        res.extend(_rules(ctx, bg))
    return res


def _rules(ctx, bg):
    res = []
    sizes = {G.area(a) for a in ctx.all_inputs}
    scale_sets = [(1,)]
    if max(sizes) >= 100:
        scale_sets.append((1, 2, 3))
    for seg in ("m8", "m4", "c8"):
        for pick in ("size", "colors", "bbox"):
            for scales in scale_sets:
                res.append(_h("analogy_%s_%s_s%d" % (seg, pick, len(scales)),
                              (lambda seg, pick, sc, bg:
                               lambda g: _complete(g, bg, seg, pick, sc, True))(seg, pick, scales, bg),
                              6.0 + 0.5 * len(scales)))
    return res
