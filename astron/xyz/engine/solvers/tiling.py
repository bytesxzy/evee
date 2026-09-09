"""Periodicity: extracting the repeating tile, and extending a pattern.

Distinct from :mod:`symmetry` -- here the structure is translational and the
answer is often *smaller* than the input (the motif) or a continuation of it.
"""

from .. import grid as G
from ..task import Hyp

SOLVER = "tiling"


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


def _periods(g, ignore=None):
    """Smallest (p, q) such that g is p-periodic in rows and q in cols."""
    h, w = G.dims(g)
    p = h
    for cand in range(1, h):
        ok = True
        for r in range(h - cand):
            for c in range(w):
                a, b = g[r][c], g[r + cand][c]
                if ignore is not None and (a == ignore or b == ignore):
                    continue
                if a != b:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            p = cand
            break
    q = w
    for cand in range(1, w):
        ok = True
        for r in range(h):
            row = g[r]
            for c in range(w - cand):
                a, b = row[c], row[c + cand]
                if ignore is not None and (a == ignore or b == ignore):
                    continue
                if a != b:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            q = cand
            break
    return p, q


def _motif(g, ignore=None):
    p, q = _periods(g, ignore)
    if p == len(g) and q == len(g[0]):
        return None
    tile = [[None] * q for _ in range(p)]
    h, w = G.dims(g)
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            if ignore is not None and v == ignore:
                continue
            t = tile[r % p][c % q]
            if t is None:
                tile[r % p][c % q] = v
            elif t != v:
                return None
    if any(v is None for row in tile for v in row):
        return None
    return tuple(tuple(r) for r in tile)


def _fill_periodic(g, ignore):
    m = _motif(g, ignore)
    if m is None:
        return None
    p, q = G.dims(m)
    h, w = G.dims(g)
    return tuple(tuple(g[r][c] if g[r][c] != ignore else m[r % p][c % q]
                       for c in range(w)) for r in range(h))


def _fill_diag(g, ignore, sign):
    """Fill unknown cells where colour is a function of (r +/- c) mod k.

    Diagonally striped patterns have no row or column period, so the periodic
    filler above cannot see them at all; this recovers the smallest modulus
    consistent with every observed cell, and refuses to guess unless every
    residue class has actually been observed.
    """
    h, w = G.dims(g)
    for k in range(2, 13):
        table = {}
        ok = True
        for r in range(h):
            row = g[r]
            for c in range(w):
                v = row[c]
                if v == ignore:
                    continue
                key = (r + sign * c) % k
                prev = table.get(key)
                if prev is None:
                    table[key] = v
                elif prev != v:
                    ok = False
                    break
            if not ok:
                break
        if not ok or len(table) < k:
            continue
        return tuple(tuple(g[r][c] if g[r][c] != ignore
                           else table[(r + sign * c) % k]
                           for c in range(w)) for r in range(h))
    return None


def _extend(g, oh, ow):
    h, w = G.dims(g)
    if oh > 60 or ow > 60 or oh < 1 or ow < 1:
        return None
    return tuple(tuple(g[r % h][c % w] for c in range(ow)) for r in range(oh))


def _dedup_blocks(g):
    """Collapse runs of identical rows/cols to one each (structure skeleton)."""
    return G.dedup(g)


def generate(ctx):
    res = []
    bg = ctx.bg
    res.append(_h("motif", lambda g: _motif(g), 4.0))
    for c in sorted(ctx.in_palette):
        res.append(_h("motif_ig#%d" % c, lambda g, c=c: _motif(g, c), 4.5))
        res.append(_h("fillper#%d" % c, lambda g, c=c: _fill_periodic(g, c), 4.0))
        for sign in (1, -1):
            res.append(_h("filldiag%+d#%d" % (sign, c),
                          (lambda c, s: lambda g: _fill_diag(g, c, s))(c, sign),
                          4.2))
    cs = ctx.const_out_shape
    if cs:
        res.append(_h("extend%dx%d" % cs,
                      lambda g, s=cs: _extend(g, s[0], s[1]), 4.0))
        for name, f in G.DIHEDRAL[1:]:
            res.append(_h("extend_%s" % name,
                          lambda g, s=cs, f=f: _extend(f(g), s[0], s[1]), 5.0))
    r = ctx.shape_ratio
    if r:
        ky, kx = r
        res.append(_h("extend_ratio%dx%d" % (ky, kx),
                      lambda g, ky=ky, kx=kx: _extend(g, len(g) * ky, len(g[0]) * kx),
                      3.5))
    # motif of the content-cropped grid
    res.append(_h("crop_motif", lambda g, b=bg: _motif_of(G.crop_to_content(g, b)), 5.0))
    res.append(_h("skeleton", _dedup_blocks, 4.5))
    return res


def _motif_of(g):
    return None if g is None else _motif(g)
