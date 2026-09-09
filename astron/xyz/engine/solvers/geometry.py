"""Whole-grid geometric rules: dihedral maps, crops, tilings, scalings.

These are cheap and account for a substantial slice of ARC-1.  Everything here
is expressed as a closure over an input grid so the harness can validate it on
the train pairs before it is ever applied to a test input.
"""

from collections import Counter

from .. import grid as G
from ..task import Hyp

SOLVER = "geometry"


def _h(name, fn, cost):
    return Hyp(name, fn, cost, SOLVER)


def generate(ctx):
    out = []
    train = ctx.train
    bg = ctx.bg

    # ---- constant output ------------------------------------------------
    outs = {b for _, b in train}
    if len(outs) == 1:
        k = train[0][1]
        out.append(_h("const", lambda g, k=k: k, 12.0))

    # ---- identity + dihedral -------------------------------------------
    for name, f in G.DIHEDRAL:
        out.append(_h(name, f, 1.0 if name == "id" else 2.0))

    # ---- crop / trim / dedup -------------------------------------------
    out.append(_h("crop_content", lambda g, b=bg: G.crop_to_content(g, b), 3.0))
    for c in sorted(ctx.in_palette):
        out.append(_h("crop_content#%d" % c,
                      lambda g, c=c: G.crop_to_content(g, c), 4.5))
    out.append(_h("trim1", lambda g: G.trim_border(g, 1), 3.0))
    out.append(_h("trim2", lambda g: G.trim_border(g, 2), 4.0))
    out.append(_h("dedup", G.dedup, 3.5))
    out.append(_h("dedup_rows", G.dedup_rows, 4.0))
    out.append(_h("dedup_cols", G.dedup_cols, 4.0))

    for w in ("top", "bottom", "left", "right"):
        out.append(_h("half_" + w, lambda g, w=w: G.half(g, w), 3.0))
    for i in range(4):
        out.append(_h("quad%d" % i, lambda g, i=i: G.quadrant(g, i), 3.5))

    # ---- padding / borders ---------------------------------------------
    for c in sorted(ctx.out_palette | {0}):
        out.append(_h("pad1#%d" % c, lambda g, c=c: G.pad(g, 1, c), 4.0))

    # ---- scaling --------------------------------------------------------
    r = ctx.shape_ratio
    if r and r != (1, 1):
        ky, kx = r
        out.append(_h("upscale%dx%d" % (ky, kx),
                      lambda g, ky=ky, kx=kx: G.upscale(g, ky, kx), 2.5))
        out.append(_h("tile%dx%d" % (ky, kx),
                      lambda g, ky=ky, kx=kx: G.tile(g, ky, kx), 3.0))
        out.extend(_mirror_tilings(ky, kx))
        out.extend(_fitted_tiling(ctx, ky, kx, bg))
        out.extend(_fractal(ky, kx, ctx, bg))
    ir = ctx.inv_shape_ratio
    if ir and ir != (1, 1):
        ky, kx = ir
        out.append(_h("downscale%dx%d" % (ky, kx),
                      lambda g, ky=ky, kx=kx: G.downscale(g, ky, kx), 2.5))
        out.append(_h("modescale%dx%d" % (ky, kx),
                      lambda g, ky=ky, kx=kx: G.block_reduce_mode(g, ky, kx), 3.5))

    # scale factor that varies with the input (e.g. by colour count)
    out.extend(_dynamic_scale(ctx, bg))

    # ---- symmetric self-concatenation ----------------------------------
    for name, f in G.DIHEDRAL:
        if name == "id":
            continue
        out.append(_h("hcat_" + name,
                      lambda g, f=f: G.hconcat(g, f(g)), 4.0))
        out.append(_h("vcat_" + name,
                      lambda g, f=f: G.vconcat(g, f(g)), 4.0))
        out.append(_h("hcat_" + name + "_rev",
                      lambda g, f=f: G.hconcat(f(g), g), 4.5))
        out.append(_h("vcat_" + name + "_rev",
                      lambda g, f=f: G.vconcat(f(g), g), 4.5))
    out.append(_h("hcat_self", lambda g: G.hconcat(g, g), 4.0))
    out.append(_h("vcat_self", lambda g: G.vconcat(g, g), 4.0))
    # union of the grid with a dihedral image of itself (mirror completion)
    for name, f in G.DIHEDRAL[1:]:
        out.append(_h("union_" + name,
                      (lambda f, b=bg: lambda g: _union(g, f(g), b))(f), 3.2))
        out.append(_h("union_" + name + "_rev",
                      (lambda f, b=bg: lambda g: _union(f(g), g, b))(f), 3.4))
    out.append(_h("quad_mirror", _quad_mirror, 4.5))
    out.append(_h("quad_mirror_r", _quad_mirror_r, 5.0))

    # ---- gravity --------------------------------------------------------
    for d in ("down", "up", "left", "right"):
        out.append(_h("gravity_" + d,
                      lambda g, d=d, b=bg: G.gravity(g, b, d), 4.0))

    # ---- translation / wrap --------------------------------------------
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)):
        out.append(_h("wrap%+d%+d" % (dr, dc),
                      lambda g, dr=dr, dc=dc: G.wrap_translate(g, dr, dc), 4.5))
        out.append(_h("shift%+d%+d" % (dr, dc),
                      lambda g, dr=dr, dc=dc, b=bg: G.translate(g, dr, dc, b), 4.5))

    # ---- fill enclosed holes -------------------------------------------
    for c in sorted(ctx.out_palette):
        out.append(_h("fill_holes#%d" % c,
                      lambda g, c=c, b=bg: G.fill_holes(g, c, b), 3.5))
        out.append(_h("fill_holes8#%d" % c,
                      lambda g, c=c, b=bg: G.fill_holes(g, c, b, True), 4.0))
    return out


def _union(a, b, bg):
    """Overlay ``b`` onto ``a`` wherever ``a`` is background."""
    if a is None or b is None or G.dims(a) != G.dims(b):
        return None
    return tuple(tuple(x if x != bg else y for x, y in zip(ra, rb))
                 for ra, rb in zip(a, b))


def _quad_mirror(g):
    top = G.hconcat(g, G.flip_h(g))
    return G.vconcat(top, G.flip_v(top)) if top else None


def _quad_mirror_r(g):
    top = G.hconcat(G.flip_h(g), g)
    return G.vconcat(top, G.flip_v(top)) if top else None


def _fitted_tiling(ctx, ky, kx, bg):
    """Read each tile's transform off the training pairs instead of guessing.

    There are 8^(k*m) ways to fill a k x m tiling with dihedral images, so
    enumerating patterns only ever covers the handful someone thought of.
    Every tile position is independently determined by the demonstrations, so
    fit it: for each position, keep the transforms consistent with every
    training pair, and take the cheapest survivor.
    """
    if ky * kx > 16:
        return []
    choice = {}
    for i in range(ky):
        for j in range(kx):
            live = None
            for a, b in ctx.train:
                h, w = G.dims(a)
                blk = G.subgrid(b, i * h, j * w, (i + 1) * h - 1, (j + 1) * w - 1)
                if blk is None:
                    return []
                ok = set()
                for name, f in G.DIHEDRAL:
                    t = f(a)
                    if G.dims(t) == (h, w) and t == blk:
                        ok.add(name)
                if len(set(v for row in blk for v in row)) == 1:
                    ok.add("#%d" % blk[0][0])
                live = ok if live is None else (live & ok)
                if not live:
                    return []
            order = [n for n, _f in G.DIHEDRAL] + sorted(live)
            choice[(i, j)] = next(n for n in order if n in live)
    return [_h("fit_tile%dx%d" % (ky, kx),
               (lambda ch, ky, kx: lambda g: _apply_fitted(g, ch, ky, kx))(choice, ky, kx),
               2.8)]


def _apply_fitted(g, choice, ky, kx):
    h, w = G.dims(g)
    if h * ky > 60 or w * kx > 60:
        return None
    rows = []
    for i in range(ky):
        band = None
        for j in range(kx):
            nm = choice[(i, j)]
            if nm.startswith("#"):
                t = G.const_grid(h, w, int(nm[1:]))
            else:
                t = G.DIHEDRAL_MAP[nm](g)
                if G.dims(t) != (h, w):
                    return None
            band = t if band is None else G.hconcat(band, t)
        if band is None:
            return None
        rows.append(band)
    res = rows[0]
    for r in rows[1:]:
        res = G.vconcat(res, r)
    return res


def _mirror_tilings(ky, kx):
    """Tilings where tile (i, j) is a dihedral image chosen by parity."""
    res = []
    combos = (
        ("mirror", lambda i, j, g: _par(g, i, j)),
        ("mirror_r", lambda i, j, g: _par(g, i + 1, j + 1)),
        ("mirror_rows", lambda i, j, g: G.flip_v(g) if i % 2 else g),
        ("mirror_cols", lambda i, j, g: G.flip_h(g) if j % 2 else g),
        ("rot_cycle", lambda i, j, g: _rotc(g, (i + j) % 4)),
    )
    for nm, sel in combos:
        res.append(_h("tile_%s%dx%d" % (nm, ky, kx),
                      lambda g, ky=ky, kx=kx, sel=sel: _build_tiles(g, ky, kx, sel),
                      4.0))
    return res


def _par(g, i, j):
    if i % 2:
        g = G.flip_v(g)
    if j % 2:
        g = G.flip_h(g)
    return g


def _rotc(g, k):
    for _ in range(k % 4):
        g = G.rot90(g)
    return g


def _build_tiles(g, ky, kx, sel):
    h, w = G.dims(g)
    if h * ky > 60 or w * kx > 60:
        return None
    rows = []
    for i in range(ky):
        band = None
        for j in range(kx):
            t = sel(i, j, g)
            if G.dims(t) != (h, w):
                return None
            band = t if band is None else G.hconcat(band, t)
        rows.append(band)
    res = rows[0]
    for r in rows[1:]:
        res = G.vconcat(res, r)
    return res


def _fractal(ky, kx, ctx, bg):
    """out[i*h+r][j*w+c] = g[r][c] gated on the value of g[i][j]."""
    res = []
    if ky != kx:
        pass
    for polarity in (True, False):
        for fillc in sorted({bg} | (ctx.out_palette - ctx.in_palette) | {0}):
            res.append(_h("fractal%s#%d" % ("" if polarity else "_inv", fillc),
                          lambda g, p=polarity, b=bg, f=fillc: _frac(g, p, b, f),
                          4.5))
    return res


def _frac(g, polarity, bg, fill):
    h, w = G.dims(g)
    if h * h > 60 or w * w > 60:
        return None
    out = [[fill] * (w * w) for _ in range(h * h)]
    for i in range(h):
        for j in range(w):
            on = (g[i][j] != bg) if polarity else (g[i][j] == bg)
            if on:
                for r in range(h):
                    orow = out[i * h + r]
                    grow = g[r]
                    for c in range(w):
                        orow[j * w + c] = grow[c]
    return tuple(tuple(r) for r in out)


def _dynamic_scale(ctx, bg):
    """Scale factors that are a function of the input, not a constant."""
    feats = {
        "ncolors": lambda g: len(G.palette(g)),
        "ncolors_nb": lambda g: len(G.palette(g) - {bg}),
        "maxcount": lambda g: max(v for k, v in G.histogram(g).items() if k != bg) if len(G.palette(g)) > 1 else 1,
        "nnz": lambda g: sum(1 for r in g for v in r if v != bg),
        "nobj": lambda g: len(G.flood_regions(g, bg, True, True)),
    }
    res = []
    for nm, f in feats.items():
        # verify the factor is consistent before emitting (cheap guard)
        ok_up = ok_tile = True
        for a, b in ctx.train:
            ah, aw = G.dims(a)
            bh, bw = G.dims(b)
            try:
                k = f(a)
            except Exception:
                ok_up = ok_tile = False
                break
            if k < 1 or bh != ah * k or bw != aw * k:
                ok_up = ok_tile = False
                break
        if ok_up:
            res.append(_h("upscale_by_" + nm,
                          lambda g, f=f: G.upscale(g, f(g), f(g)), 5.0))
            res.append(_h("tile_by_" + nm,
                          lambda g, f=f: G.tile(g, f(g), f(g)), 5.0))
    return res
