"""Meta-solver: re-pose the task, then reuse the whole specialist portfolio.

Many tasks are a known transformation wearing a disguise -- the real rule
applies to the *cropped* grid, or to the grid with its background rows removed,
or the answer is a 3x upscale of a rule that lives at the original resolution.
Rather than duplicating every specialist for every disguise, we rewrite the
task and hand the rewritten version back to the specialists.

Input rewrites are applied to train and test inputs alike; output rewrites are
inverted on the way out.  Both directions are validated on the training pairs
by the orchestrator exactly like any other hypothesis, so a rewrite that
happens to fit by luck is no more trusted than anything else.
"""

import time

from .. import grid as G
from .. import objects as O
from .. import hypcache
from ..task import Ctx, Hyp

SOLVER = "compose"
PHASE = 2


def _crop(g, bg):
    return G.crop_to_content(g, bg)


def _compress(g, bg):
    rows = [r for r in g if any(v != bg for v in r)]
    if not rows:
        return None
    t = G.transpose(tuple(rows))
    cols = [c for c in t if any(v != bg for v in c)]
    if not cols:
        return None
    return G.transpose(tuple(cols))


def _largest(g, bg):
    objs = O.segment(g, "c8", bg)
    if not objs:
        return None
    o = O.select_extreme(objs, "size", True)
    return None if o is None else G.subgrid(g, o.r0, o.c0, o.r1, o.c1)


def _strip_lines(g, bg):
    """Remove full uniform rows/cols (separator scaffolding)."""
    h, w = G.dims(g)
    keep_r = [r for r in range(h) if len(set(g[r])) > 1]
    if not keep_r or len(keep_r) == h:
        return None
    t = tuple(g[r] for r in keep_r)
    tt = G.transpose(t)
    keep_c = [c for c in range(len(tt)) if len(set(tt[c])) > 1]
    if not keep_c:
        return None
    return G.transpose(tuple(tt[c] for c in keep_c))


def _in_rewrites(ctx):
    bg = ctx.bg
    out = [("crop", lambda g, b=bg: _crop(g, b)),
           ("compress", lambda g, b=bg: _compress(g, b)),
           ("largest", lambda g, b=bg: _largest(g, b)),
           ("dedup", G.dedup),
           ("strip", lambda g, b=bg: _strip_lines(g, b))]
    return out


def _out_rewrites(ctx):
    """(name, forward_on_output, inverse_on_prediction)."""
    out = []
    r = ctx.shape_ratio
    if r and r != (1, 1):
        ky, kx = r
        ok = all(G.downscale(b, ky, kx) is not None for _a, b in ctx.train)
        if ok:
            out.append(("down%dx%d" % (ky, kx),
                        lambda g, ky=ky, kx=kx: G.downscale(g, ky, kx),
                        lambda g, ky=ky, kx=kx: G.upscale(g, ky, kx)))
    return out


def _rank_perm(g, bg):
    """A colour permutation derived from this grid alone: background first,
    then colours in order of how much of the grid they cover.

    Tasks that use a different palette in every example are structurally one
    task; without this they look like several. The permutation is a bijection
    on 0-9 and is computed from the *input*, so it can be inverted at test time
    without knowing the answer.
    """
    hist = G.histogram(g)
    hist.pop(bg, None)
    order = [bg] + [c for c, _n in sorted(hist.items(),
                                          key=lambda kv: (-kv[1], kv[0]))]
    rest = [c for c in range(10) if c not in order]
    order += rest
    return {c: i for i, c in enumerate(order)}


def _invert(perm):
    return {v: k for k, v in perm.items()}


def _pair_rewrites(ctx):
    """(name, per-grid forward permutation) applied to input and output alike."""
    bg = ctx.bg
    pals = {tuple(sorted(G.palette(a) - {bg})) for a in ctx.all_inputs}
    if len(pals) < 2:
        return []          # one palette: normalising buys nothing
    return [("rankpal", lambda g, b=bg: _rank_perm(g, b))]


_MODULE_NAMES = ("geometry", "colormap", "cellwise", "symmetry", "partition",
                 "objects_map", "sequence", "tiling", "regions", "select")


def _modules():
    from ..solvers import (cellwise, colormap, extend, geometry, objects_map,
                           objproc, paneltable, partition, regions, select,
                           selfstamp, sequence, symmetry, tiling)
    from ..solvers import blocks, compose, paint, substitute
    return (geometry, colormap, symmetry, partition, tiling, blocks, regions,
            cellwise, objects_map, substitute, sequence, paint, select,
            compose, paneltable, selfstamp, extend)


def generate(ctx):
    res = []
    deadline = ctx.deadline or (time.time() + 6.0)
    mods = _modules()
    # Share the module's slice across its variants instead of giving the first
    # one a fixed 1.2s and letting the rest run out of clock: before this the
    # later rewrites were effectively never tried.
    n_var = max(1, len(_in_rewrites(ctx)) + len(_pair_rewrites(ctx)) + 3)
    slice_s = max(0.35, (deadline - time.time()) / n_var)

    # ---- input-side rewrites -------------------------------------------
    for rname, T in _in_rewrites(ctx):
        if time.time() > deadline:
            break
        pairs = []
        ok = True
        for a, b in ctx.train:
            ta = _safe(T, a)
            if ta is None or ta == a:
                ok = False
                break
            pairs.append((G.to_list(ta), G.to_list(b)))
        if not ok:
            continue
        tins = [_safe(T, t) for t in ctx.test_inputs]
        if any(t is None for t in tins):
            continue
        sub = Ctx(pairs, [G.to_list(t) for t in tins])
        sub.deadline = min(deadline, time.time() + slice_s)
        for mod in mods:
            if time.time() > deadline:
                break
            try:
                hyps = list(hypcache.generate(mod, sub))
            except Exception:
                continue
            for hp in hyps:
                if hp.fits(sub.train):
                    res.append(Hyp("%s|%s" % (rname, hp.name),
                                   _chain_in(T, hp), 3.0 + hp.cost, SOLVER))
                    if len(res) > 60:
                        break
            if len(res) > 60:
                break

    # ---- paired (invertible) colour rewrites ---------------------------
    for rname, permf in _pair_rewrites(ctx):
        if time.time() > deadline:
            break
        pairs = []
        ok = True
        for a, b in ctx.train:
            p = permf(a)
            if p is None:
                ok = False
                break
            pairs.append((G.to_list(G.apply_cmap(a, p)),
                          G.to_list(G.apply_cmap(b, p))))
        if not ok:
            continue
        tins = []
        for t in ctx.test_inputs:
            p = permf(t)
            if p is None:
                tins = None
                break
            tins.append(G.to_list(G.apply_cmap(t, p)))
        if tins is None:
            continue
        sub = Ctx(pairs, tins)
        sub.deadline = min(deadline, time.time() + slice_s)
        for mod in mods:
            if time.time() > deadline:
                break
            try:
                hyps = list(hypcache.generate(mod, sub))
            except Exception:
                continue
            for hp in hyps:
                if hp.fits(sub.train):
                    res.append(Hyp("%s~%s" % (rname, hp.name),
                                   _chain_pair(permf, hp), 3.5 + hp.cost, SOLVER))
                    break

    # ---- paired geometric rewrites -------------------------------------
    # Several families are row-biased by construction (row dictionaries, half
    # selection, downward gravity).  Solving the transposed task and undoing
    # the transpose gives every one of them its column-wise twin for free.
    for gname, fwd, inv in (("T", G.transpose, G.transpose),
                            ("R", G.rot90, G.rot270)):
        if time.time() > deadline:
            break
        pairs = [(G.to_list(fwd(a)), G.to_list(fwd(b))) for a, b in ctx.train]
        tins = [G.to_list(fwd(t)) for t in ctx.test_inputs]
        sub = Ctx(pairs, tins)
        sub.deadline = min(deadline, time.time() + slice_s)
        for mod in mods:
            if time.time() > deadline:
                break
            try:
                hyps = list(hypcache.generate(mod, sub))
            except Exception:
                continue
            for hp in hyps:
                if hp.fits(sub.train):
                    res.append(Hyp("%s@%s" % (gname, hp.name),
                                   _chain_geo(fwd, inv, hp), 3.5 + hp.cost,
                                   SOLVER))
                    break

    # ---- output-side rewrites ------------------------------------------
    for rname, fwd, inv in _out_rewrites(ctx):
        if time.time() > deadline:
            break
        pairs = []
        ok = True
        for a, b in ctx.train:
            tb = _safe(fwd, b)
            if tb is None:
                ok = False
                break
            pairs.append((G.to_list(a), G.to_list(tb)))
        if not ok:
            continue
        sub = Ctx(pairs, [G.to_list(t) for t in ctx.test_inputs])
        sub.deadline = min(deadline, time.time() + slice_s)
        for mod in mods:
            if time.time() > deadline:
                break
            try:
                hyps = list(hypcache.generate(mod, sub))
            except Exception:
                continue
            for hp in hyps:
                if hp.fits(sub.train):
                    res.append(Hyp("%s^%s" % (rname, hp.name),
                                   _chain_out(inv, hp), 3.0 + hp.cost, SOLVER))
                    if len(res) > 90:
                        break
    return res


def _safe(f, g):
    try:
        r = f(g)
    except Exception:
        return None
    return r if (r is not None and G.valid(r)) else None


def _chain_in(T, hp):
    def run(g):
        t = _safe(T, g)
        return None if t is None else hp.fn(t)
    return run


def _chain_geo(fwd, inv, hp):
    def run(g):
        r = hp.fn(fwd(g))
        return None if r is None else inv(r)
    return run


def _chain_pair(permf, hp):
    def run(g):
        p = permf(g)
        if p is None:
            return None
        r = hp.fn(G.apply_cmap(g, p))
        return None if r is None else G.apply_cmap(r, _invert(p))
    return run


def _chain_out(inv, hp):
    def run(g):
        r = hp.fn(g)
        return None if r is None else inv(r)
    return run
