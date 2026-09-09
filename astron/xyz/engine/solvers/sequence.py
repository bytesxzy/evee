"""Motion and drawing: gravity with collision, rays, and connections.

These are *procedural* rules -- they cannot be written as a pixel lookup, and
they are common enough in ARC to deserve first-class treatment.
"""

from collections import Counter

from .. import grid as G
from .. import objects as O
from ..task import Hyp

SOLVER = "sequence"

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
DIAG = {"ul": (-1, -1), "ur": (-1, 1), "dl": (1, -1), "dr": (1, 1)}


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


# --- gravity with collision ------------------------------------------------

def _move_objects(g, seg, bg, d, once=False):
    bg = G.bg_or(g, bg)
    dr, dc = DIRS[d]
    objs = O.segment(g, seg, bg)
    if not objs or len(objs) > 60:
        return None
    h, w = G.dims(g)
    occupied = [[False] * w for _ in range(h)]
    # move in order of proximity to the target wall
    def key(o):
        if d == "down":
            return -o.r1
        if d == "up":
            return o.r0
        if d == "right":
            return -o.c1
        return o.c0
    out = [[bg] * w for _ in range(h)]
    for o in sorted(objs, key=key):
        best = 0
        step = 1
        while True:
            ok = True
            for r, c in o.cells:
                nr, nc = r + dr * step, c + dc * step
                if not (0 <= nr < h and 0 <= nc < w) or occupied[nr][nc]:
                    ok = False
                    break
            if not ok:
                break
            best = step
            if once:
                break
            step += 1
        for r, c in o.cells:
            nr, nc = r + dr * best, c + dc * best
            occupied[nr][nc] = True
            out[nr][nc] = g[r][c]
    return tuple(tuple(r) for r in out)


# --- rays ------------------------------------------------------------------

def _rays(g, bg, dirs, stop_at_obstacle, color_mode, only_color=None):
    bg = G.bg_or(g, bg)
    h, w = G.dims(g)
    out = [list(r) for r in g]
    src = [(r, c, g[r][c]) for r in range(h) for c in range(w) if g[r][c] != bg]
    if not src or len(src) > 200:
        return None
    for r, c, v in src:
        if only_color is not None and v != only_color:
            continue
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w:
                if g[nr][nc] != bg:
                    if stop_at_obstacle:
                        break
                else:
                    out[nr][nc] = v if color_mode is None else color_mode
                nr += dr
                nc += dc
    return tuple(tuple(r) for r in out)


# --- connect equal-coloured pairs -----------------------------------------

def _connect(g, bg, fill_mode, diag=False, max_gap=None):
    bg = G.bg_or(g, bg)
    h, w = G.dims(g)
    out = [list(r) for r in g]
    pts = {}
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            if v != bg:
                pts.setdefault(v, []).append((r, c))
    any_drawn = False
    for v, ps in pts.items():
        if len(ps) > 40:
            continue
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                (r1, c1), (r2, c2) = ps[i], ps[j]
                cells = None
                if max_gap is not None:
                    d = max(abs(r2 - r1), abs(c2 - c1)) - 1
                    if d > max_gap:
                        continue
                if r1 == r2 and abs(c2 - c1) > 1:
                    cells = [(r1, c) for c in range(min(c1, c2) + 1, max(c1, c2))]
                elif c1 == c2 and abs(r2 - r1) > 1:
                    cells = [(r, c1) for r in range(min(r1, r2) + 1, max(r1, r2))]
                elif diag and abs(r2 - r1) == abs(c2 - c1) > 1:
                    sr = 1 if r2 > r1 else -1
                    sc = 1 if c2 > c1 else -1
                    cells = [(r1 + k * sr, c1 + k * sc)
                             for k in range(1, abs(r2 - r1))]
                if not cells:
                    continue
                if any(g[r][c] != bg for r, c in cells):
                    continue
                col = v if fill_mode is None else fill_mode
                for r, c in cells:
                    out[r][c] = col
                any_drawn = True
    if not any_drawn:
        return None
    return tuple(tuple(r) for r in out)


# --- outline / halo --------------------------------------------------------

def _halo(g, bg, color, diag, replace):
    bg = G.bg_or(g, bg)
    h, w = G.dims(g)
    out = [list(r) for r in g]
    nb = G.N8 if diag else G.N4
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg:
                continue
            for dr, dc in nb:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr][nc] == bg:
                    out[nr][nc] = color if color is not None else g[r][c]
    if replace:
        for r in range(h):
            for c in range(w):
                if g[r][c] != bg:
                    out[r][c] = bg
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
    for seg in ("c4", "c8", "m8"):
        for d in DIRS:
            res.append(_h("move_%s_%s" % (seg, d),
                          (lambda s, d, bg: lambda g: _move_objects(g, s, bg, d))(seg, d, bg),
                          4.5))
            res.append(_h("step_%s_%s" % (seg, d),
                          (lambda s, d, bg: lambda g: _move_objects(g, s, bg, d, True))(seg, d, bg),
                          5.5))
    dirsets = [("4", tuple(DIRS.values())), ("d", tuple(DIAG.values())),
               ("8", tuple(DIRS.values()) + tuple(DIAG.values()))]
    for dn, dv in dirsets:
        for stop in (True, False):
            res.append(_h("ray%s%s" % (dn, "_stop" if stop else ""),
                          (lambda dv, s, bg: lambda g: _rays(g, bg, dv, s, None))(dv, stop, bg),
                          4.5))
    for dn, (dr, dc) in list(DIRS.items()) + list(DIAG.items()):
        res.append(_h("ray1_" + dn,
                      (lambda v, bg: lambda g: _rays(g, bg, (v,), True, None))((dr, dc), bg),
                      4.5))
    for diag in (False, True):
        for gap in (None, 1, 2):
            sfx = ("_d" if diag else "") + ("" if gap is None else "_g%d" % gap)
            res.append(_h("connect" + sfx,
                          (lambda d, bg, gp: lambda g: _connect(g, bg, None, d, gp))(diag, bg, gap),
                          4.0 if gap is None else 4.4))
            for c in sorted(ctx.out_palette):
                res.append(_h("connect%s#%d" % (sfx, c),
                              (lambda d, c, bg, gp: lambda g: _connect(g, bg, c, d, gp))(diag, c, bg, gap),
                              5.0 if gap is None else 5.4))
    for diag in (False, True):
        res.append(_h("halo%s" % ("8" if diag else "4"),
                      (lambda d, bg: lambda g: _halo(g, bg, None, d, False))(diag, bg),
                      4.5))
        for c in sorted(ctx.out_palette):
            res.append(_h("halo%s#%d" % ("8" if diag else "4", c),
                          (lambda d, c, bg: lambda g: _halo(g, bg, c, d, False))(diag, c, bg),
                          5.0))
            res.append(_h("ring%s#%d" % ("8" if diag else "4", c),
                          (lambda d, c, bg: lambda g: _halo(g, bg, c, d, True))(diag, c, bg),
                          6.0))
    return res
