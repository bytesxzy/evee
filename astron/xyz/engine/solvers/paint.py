"""Drawing rules that need geometry rather than a lookup.

* extend every straight segment to the grid edge;
* complete a rectangle from its corner markers;
* reflect the contents of one side of a separator line onto the other;
* make each object symmetric within its own bounding box.

Each is a genuine construction: no table over local contexts can express
"continue this line until it hits something", because the answer at a cell
depends on structure arbitrarily far away.
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


# --- extend segments -------------------------------------------------------

def _extend_lines(g, bg, both, stop):
    bg = G.bg_or(g, bg)
    h, w = G.dims(g)
    out = [list(r) for r in g]
    drawn = False
    for o in O.segment(g, "c8", bg):
        if o.size < 2:
            continue
        if o.height == 1:
            dirs = [(0, 1), (0, -1)]
        elif o.width == 1:
            dirs = [(1, 0), (-1, 0)]
        else:
            continue
        if not o.is_rect():
            continue
        for dr, dc in dirs:
            r = o.r1 if dr > 0 else o.r0
            c = o.c1 if dc > 0 else o.c0
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w:
                if g[nr][nc] != bg:
                    if stop:
                        break
                else:
                    out[nr][nc] = o.color
                    drawn = True
                nr += dr
                nc += dc
            if not both:
                break
    return tuple(tuple(r) for r in out) if drawn else None


# --- rectangle from corners ------------------------------------------------

def _rect_from_corners(g, bg, fill, outline_only):
    bg = G.bg_or(g, bg)
    pts = {}
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v != bg:
                pts.setdefault(v, []).append((r, c))
    out = [list(r) for r in g]
    drawn = False
    for v, ps in pts.items():
        if len(ps) not in (2, 4):
            continue
        r0, c0, r1, c1 = G.bbox_of(ps)
        if r1 - r0 < 1 or c1 - c0 < 1:
            continue
        col = v if fill is None else fill
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                edge = r in (r0, r1) or c in (c0, c1)
                if outline_only and not edge:
                    continue
                if g[r][c] == bg:
                    out[r][c] = col
                    drawn = True
    return tuple(tuple(r) for r in out) if drawn else None


# --- reflect across a separator -------------------------------------------

def _axis_lines(g):
    h, w = G.dims(g)
    rows = [r for r in range(h) if len(set(g[r])) == 1]
    t = G.transpose(g)
    cols = [c for c in range(w) if len(set(t[c])) == 1]
    return rows, cols


def _reflect_across(g, bg, use_row, overwrite):
    bg = G.bg_or(g, bg)
    rows, cols = _axis_lines(g)
    idx = rows if use_row else cols
    if len(idx) != 1:
        return None
    a = idx[0]
    h, w = G.dims(g)
    out = [list(r) for r in g]
    drawn = False
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            if v == bg:
                continue
            if use_row:
                nr, nc = 2 * a - r, c
            else:
                nr, nc = r, 2 * a - c
            if 0 <= nr < h and 0 <= nc < w and (r, c) != (nr, nc):
                if overwrite or g[nr][nc] == bg:
                    if out[nr][nc] != v:
                        out[nr][nc] = v
                        drawn = True
    return tuple(tuple(r) for r in out) if drawn else None


# --- connect small objects to a big one -----------------------------------

def _connect_to_anchor(g, bg, seg, anchor_mode):
    """Each satellite aligned with the anchor object grows a line towards it.

    "Aligned" means the satellite's row or column band overlaps the anchor's;
    satellites that are not aligned are deliberately left untouched, which is
    what distinguishes this from a plain ray.
    """
    bg = G.bg_or(g, bg)
    objs = O.segment(g, seg, bg)
    if len(objs) < 2 or len(objs) > 40:
        return None
    if anchor_mode == "largest":
        anchor = max(objs, key=lambda o: (o.size, o.bbox_area))
    else:
        anchor = max(objs, key=lambda o: (o.bbox_area, o.size))
    h, w = G.dims(g)
    out = [list(r) for r in g]
    drawn = False
    for o in objs:
        if o is anchor:
            continue
        v = o.color
        # horizontal alignment
        rows = range(max(o.r0, anchor.r0), min(o.r1, anchor.r1) + 1)
        if rows:
            if o.c0 > anchor.c1:
                span = range(anchor.c1 + 1, o.c0)
            elif o.c1 < anchor.c0:
                span = range(o.c1 + 1, anchor.c0)
            else:
                span = ()
            for r in rows:
                if all(g[r][c] == bg for c in span):
                    for c in span:
                        out[r][c] = v
                        drawn = True
        cols = range(max(o.c0, anchor.c0), min(o.c1, anchor.c1) + 1)
        if cols:
            if o.r0 > anchor.r1:
                span = range(anchor.r1 + 1, o.r0)
            elif o.r1 < anchor.r0:
                span = range(o.r1 + 1, anchor.r0)
            else:
                span = ()
            for c in cols:
                if all(g[r][c] == bg for r in span):
                    for r in span:
                        out[r][c] = v
                        drawn = True
    return tuple(tuple(r) for r in out) if drawn else None


# --- move satellites onto the anchor ---------------------------------------

def _move_to_anchor(g, bg, seg, diag_touch, keep_anchor):
    """Slide every satellite object straight at the anchor until it touches.

    Distinct from gravity (which has a fixed direction) and from the anchor
    *line* rule (which draws rather than moves): the direction is per object,
    read off the geometry, and the object arrives intact.
    """
    bg = G.bg_or(g, bg)
    objs = O.segment(g, seg, bg)
    if len(objs) < 2 or len(objs) > 40:
        return None
    anchor = max(objs, key=lambda o: (o.size, o.bbox_area))
    if anchor.size < 2:
        return None
    h, w = G.dims(g)
    occupied = set(anchor.cells)
    out = [[bg] * w for _ in range(h)]
    for r, c in anchor.cells:
        out[r][c] = g[r][c]
    nb = G.N8 if diag_touch else G.N4
    moved = False
    for o in sorted(objs, key=lambda o: -min(
            abs(o.r0 - anchor.r0) + abs(o.c0 - anchor.c0), 10 ** 6)):
        if o is anchor:
            continue
        # Direction by band overlap, not by centre distance: an object already
        # lined up with the anchor's columns must travel straight, and a
        # centre comparison nudges it diagonally instead.
        if o.r1 < anchor.r0:
            dr = 1
        elif o.r0 > anchor.r1:
            dr = -1
        else:
            dr = 0
        if o.c1 < anchor.c0:
            dc = 1
        elif o.c0 > anchor.c1:
            dc = -1
        else:
            dc = 0
        if dr == 0 and dc == 0:
            return None
        best = 0
        step = 0
        while step < 60:
            cells = [(r + dr * step, c + dc * step) for r, c in o.cells]
            if not all(0 <= r < h and 0 <= c < w for r, c in cells):
                break
            if any((r, c) in occupied for r, c in cells):
                break
            touching = any((r + a, c + b) in occupied
                           for r, c in cells for a, b in nb)
            best = step
            if touching:
                break
            step += 1
        cells = [(r + dr * best, c + dc * best) for r, c in o.cells]
        for (nr, nc), (r, c) in zip(cells, o.cells):
            out[nr][nc] = g[r][c]
            occupied.add((nr, nc))
        if best:
            moved = True
    return tuple(tuple(r) for r in out) if moved else None


# --- striped rays ----------------------------------------------------------

def _ray_striped(g, bg, dr, dc, alt, period):
    """A trail that alternates between the source colour and a second colour."""
    bg = G.bg_or(g, bg)
    h, w = G.dims(g)
    src = [(r, c, g[r][c]) for r in range(h) for c in range(w) if g[r][c] != bg]
    if not src or len(src) > 60:
        return None
    out = [list(r) for r in g]
    drawn = False
    for r, c, v in src:
        k = 1
        nr, nc = r + dr, c + dc
        while 0 <= nr < h and 0 <= nc < w:
            if g[nr][nc] == bg:
                out[nr][nc] = alt if (k % period) else v
                drawn = True
            nr += dr
            nc += dc
            k += 1
    return tuple(tuple(r) for r in out) if drawn else None


# --- repeat a translation witnessed by two copies --------------------------

def _repeat_translation(g, bg, seg, both):
    """Two copies of a shape define a step; continue it to the edges.

    The evidence for "keep going" is inside the grid itself: a pair of
    identical objects fixes a translation vector, and the task is to iterate it.
    """
    bg = G.bg_or(g, bg)
    objs = O.segment(g, seg, bg)
    if len(objs) < 2 or len(objs) > 40:
        return None
    h, w = G.dims(g)
    groups = {}
    for o in objs:
        groups.setdefault((o.patch,), []).append(o)
    out = [list(r) for r in g]
    drawn = False
    for _k, members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda o: (o.r0, o.c0))
        deltas = {(b.r0 - a.r0, b.c0 - a.c0)
                  for a, b in zip(members, members[1:])}
        if len(deltas) != 1:
            continue
        dr, dc = deltas.pop()
        if dr == 0 and dc == 0:
            continue
        proto = members[0]
        steps = [(members[-1].r0 + dr, members[-1].c0 + dc, dr, dc)]
        if both:
            steps.append((proto.r0 - dr, proto.c0 - dc, -dr, -dc))
        for r0, c0, sr, sc in steps:
            while True:
                cells = [(r0 + (r - proto.r0), c0 + (c - proto.c0))
                         for r, c in proto.cells]
                if not all(0 <= r < h and 0 <= c < w for r, c in cells):
                    break
                for r, c in proto.cells:
                    nr, nc = r0 + (r - proto.r0), c0 + (c - proto.c0)
                    out[nr][nc] = g[r][c]
                    drawn = True
                r0 += sr
                c0 += sc
    return tuple(tuple(r) for r in out) if drawn else None


# --- stamp a template at every marker --------------------------------------

def _stamp_at_markers(g, bg, seg, recolor, keep_marker):
    bg = G.bg_or(g, bg)
    objs = O.segment(g, seg, bg)
    if len(objs) < 2 or len(objs) > 40:
        return None
    big = max(objs, key=lambda o: o.size)
    if big.size < 3:
        return None
    markers = [o for o in objs if o.size == 1]
    if not markers:
        return None
    h, w = G.dims(g)
    cr = (big.r0 + big.r1) // 2
    cc = (big.c0 + big.c1) // 2
    out = [list(r) for r in g]
    drawn = False
    for m in markers:
        mr, mc = next(iter(m.cells))
        for r, c in big.cells:
            nr, nc = mr + (r - cr), mc + (c - cc)
            if 0 <= nr < h and 0 <= nc < w:
                v = m.color if recolor else g[r][c]
                if out[nr][nc] != v:
                    out[nr][nc] = v
                    drawn = True
        if keep_marker:
            out[mr][mc] = m.color
    return tuple(tuple(r) for r in out) if drawn else None


# --- bouncing rays ---------------------------------------------------------

def _bounce(g, bg, start_color, diag, max_steps=200):
    """A ray that reflects off the grid edges until it runs out of steps.

    Reflection is not expressible as a local rule or as a straight ray: the
    direction at a cell depends on the whole path taken to reach it.
    """
    bg = G.bg_or(g, bg)
    h, w = G.dims(g)
    src = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row)
           if v == start_color]
    if len(src) != 1:
        return None
    r, c = src[0]
    dirs = ((1, 1), (1, -1), (-1, 1), (-1, -1)) if diag else ((1, 0), (0, 1))
    out = [list(x) for x in g]
    drawn = False
    for dr, dc in dirs:
        cr, cc, vr, vc = r, c, dr, dc
        for _ in range(max_steps):
            nr, nc = cr + vr, cc + vc
            if not (0 <= nr < h):
                vr = -vr
                nr = cr + vr
            if not (0 <= nc < w):
                vc = -vc
                nc = cc + vc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if g[nr][nc] == bg:
                out[nr][nc] = start_color
                drawn = True
            cr, cc = nr, nc
    return tuple(tuple(x) for x in out) if drawn else None


# --- histogram / bar output ------------------------------------------------

def _bars(g, bg, vertical, order_desc, out_shape):
    """Turn a colour histogram into a bar chart."""
    bg = G.bg_or(g, bg)
    hist = G.histogram(g)
    hist.pop(bg, None)
    if not hist or len(hist) > 10:
        return None
    items = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))
    if not order_desc:
        items = items[::-1]
    n = len(items)
    m = max(v for _k, v in items)
    if m > 30 or n > 30:
        return None
    h, w = (m, n) if vertical else (n, m)
    if out_shape is not None:
        if out_shape != (h, w):
            return None
    out = [[bg] * w for _ in range(h)]
    for i, (col, cnt) in enumerate(items):
        for j in range(cnt):
            if vertical:
                out[h - 1 - j][i] = col
            else:
                out[i][j] = col
    return tuple(tuple(r) for r in out)


# --- move a shape onto a marked target -------------------------------------

def _move_to_markers(g, bg, marker, mode):
    """Translate the main object onto the box staked out by marker cells."""
    bg = G.bg_or(g, bg)
    h, w = G.dims(g)
    marks = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row)
             if v == marker]
    if len(marks) < 2 or len(marks) > 8:
        return None
    mr0, mc0, mr1, mc1 = G.bbox_of(marks)
    body = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row)
            if v != bg and v != marker]
    if not body:
        return None
    br0, bc0, br1, bc1 = G.bbox_of(body)
    if mode == "center":
        dr = ((mr0 + mr1) - (br0 + br1)) // 2
        dc = ((mc0 + mc1) - (bc0 + bc1)) // 2
    elif mode == "inside":
        dr, dc = (mr0 + 1) - br0, (mc0 + 1) - bc0
    else:
        dr, dc = mr0 - br0, mc0 - bc0
    if dr == 0 and dc == 0:
        return None
    out = [[bg] * w for _ in range(h)]
    for r, c in marks:
        out[r][c] = marker
    for r, c in body:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        out[nr][nc] = g[r][c]
    return tuple(tuple(r) for r in out)


# --- growth to a fixed point ----------------------------------------------

def _voronoi(g, bg, diag, tie_bg):
    """Grow every coloured region simultaneously until the grid is full.

    A one-step local rule cannot express this: the colour a far cell ends up
    with depends on a distance computed across the whole grid. Ties (cells
    equidistant from two seeds) are either left as background or resolved to
    the lower colour, which are genuinely different tasks.
    """
    from collections import deque
    bg = G.bg_or(g, bg)
    h, w = G.dims(g)
    dist = [[-1] * w for _ in range(h)]
    col = [[bg] * w for _ in range(h)]
    q = deque()
    for r in range(h):
        for c in range(w):
            if g[r][c] != bg:
                dist[r][c] = 0
                col[r][c] = g[r][c]
                q.append((r, c))
    if not q or len(q) == h * w:
        return None
    nb = G.N8 if diag else G.N4
    tie = [[False] * w for _ in range(h)]
    while q:
        cr, cc = q.popleft()
        for dr, dc in nb:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if dist[nr][nc] == -1:
                dist[nr][nc] = dist[cr][cc] + 1
                col[nr][nc] = col[cr][cc]
                q.append((nr, nc))
            elif (dist[nr][nc] == dist[cr][cc] + 1
                  and col[nr][nc] != col[cr][cc]):
                tie[nr][nc] = True
                if not tie_bg and col[cr][cc] < col[nr][nc]:
                    col[nr][nc] = col[cr][cc]
    out = []
    for r in range(h):
        row = []
        for c in range(w):
            if g[r][c] != bg:
                row.append(g[r][c])
            elif tie[r][c] and tie_bg:
                row.append(bg)
            else:
                row.append(col[r][c])
        out.append(tuple(row))
    return tuple(out)


def _fill_enclosed(g, bg, diag, color=None, only_rect=False):
    """Paint each enclosed background pocket with the colour that encloses it."""
    from collections import deque
    bg = G.bg_or(g, bg)
    h, w = G.dims(g)
    seen = [[False] * w for _ in range(h)]
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if g[r][c] == bg and not seen[r][c]:
                seen[r][c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if g[r][c] == bg and not seen[r][c]:
                seen[r][c] = True
                q.append((r, c))
    nb = G.N8 if diag else G.N4
    while q:
        cr, cc = q.popleft()
        for dr, dc in nb:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g[nr][nc] == bg:
                seen[nr][nc] = True
                q.append((nr, nc))
    out = [list(r) for r in g]
    drawn = False
    vis = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g[r][c] != bg or seen[r][c] or vis[r][c]:
                continue
            comp = []
            border = Counter()
            st = [(r, c)]
            vis[r][c] = True
            while st:
                cr, cc = st.pop()
                comp.append((cr, cc))
                for dr, dc in G.N4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if g[nr][nc] == bg:
                        if not vis[nr][nc] and not seen[nr][nc]:
                            vis[nr][nc] = True
                            st.append((nr, nc))
                    else:
                        border[g[nr][nc]] += 1
            if not border:
                continue
            if only_rect:
                r0, c0, r1, c1 = G.bbox_of(comp)
                if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
                    continue
            col = color if color is not None else border.most_common(1)[0][0]
            for cr, cc in comp:
                out[cr][cc] = col
                drawn = True
    return tuple(tuple(r) for r in out) if drawn else None


# --- symmetrise each object -----------------------------------------------

def _symmetrise(g, bg, seg, mode):
    bg = G.bg_or(g, bg)
    objs = O.segment(g, seg, bg)
    if not objs or len(objs) > 60:
        return None
    out = [list(r) for r in g]
    drawn = False
    for o in objs:
        p = o.filled(bg)
        variants = [p]
        if mode in ("h", "both"):
            variants.append(G.flip_h(p))
        if mode in ("v", "both"):
            variants.append(G.flip_v(p))
        if mode == "both":
            variants.append(G.rot180(p))
        for var in variants[1:]:
            for r in range(o.height):
                for c in range(o.width):
                    v = var[r][c]
                    if v != bg and out[o.r0 + r][o.c0 + c] == bg:
                        out[o.r0 + r][o.c0 + c] = v
                        drawn = True
    return tuple(tuple(r) for r in out) if drawn else None


_RAY_MODES = ("none", "h", "v", "both")


def _color_rays(g, bg, assign, mode_order):
    """Each colour emits rays in the directions its own colour dictates.

    A single global direction cannot express "dots of this colour draw rows and
    dots of that colour draw columns", which is a whole ARC family on its own.
    Later colours overwrite earlier ones, so the draw order is part of the rule.
    """
    bg = G.bg_or(g, bg)
    h, w = G.dims(g)
    out = [list(r) for r in g]
    src = [(r, c, g[r][c]) for r in range(h) for c in range(w) if g[r][c] != bg]
    if not src or len(src) > 120:
        return None
    drawn = False
    # Draw order is by *direction*, not by colour: where a row line crosses a
    # column line, which one shows through is a property of the two directions.
    for want in mode_order:
        for r, c, v in src:
            mode = assign.get(v, "none")
            if mode != want:
                continue
            if mode in ("h", "both"):
                for cc in range(w):
                    if g[r][cc] == bg:
                        out[r][cc] = v
                        drawn = True
            if mode in ("v", "both"):
                for rr in range(h):
                    if g[rr][c] == bg:
                        out[rr][c] = v
                        drawn = True
    return tuple(tuple(r) for r in out) if drawn else None


def _ray_assignments(colors):
    # 4^k assignments: past three colours this stops being a search and starts
    # being a way to spend the whole budget on one family
    if not colors or len(colors) > 3:
        return []
    out = [{}]
    for col in colors:
        nxt = []
        for base in out:
            for m in _RAY_MODES:
                d = dict(base)
                d[col] = m
                nxt.append(d)
        out = nxt
        if len(out) > 300:
            return out
    return [a for a in out if any(v != "none" for v in a.values())]


def generate(ctx):
    res = []
    bgs = [ctx.bg, None] if ctx.bg_varies else [ctx.bg]
    # bar charts change the grid shape, so they sit outside the same-shape gate
    for bg in bgs:
        for vert in (True, False):
            for desc in (True, False):
                res.append(_h("bars%d%d" % (vert, desc),
                              (lambda v, d, b: lambda g: _bars(g, b, v, d, None))(vert, desc, bg),
                              5.5))
    if not ctx.same_shape:
        return res
    for bg in bgs:
        res.extend(_rules(ctx, bg))
    return res


def _rules(ctx, bg):
    res = []
    for both in (True, False):
        for stop in (True, False):
            res.append(_h("extend_lines%d%d" % (both, stop),
                          (lambda b, s, bg: lambda g: _extend_lines(g, bg, b, s))(both, stop, bg),
                          4.5))
    for outline in (True, False):
        res.append(_h("rect_corners%d" % outline,
                      (lambda o, bg: lambda g: _rect_from_corners(g, bg, None, o))(outline, bg),
                      4.8))
        for c in sorted(ctx.out_palette):
            res.append(_h("rect_corners%d#%d" % (outline, c),
                          (lambda o, c, bg: lambda g: _rect_from_corners(g, bg, c, o))(outline, c, bg),
                          5.5))
    for use_row in (True, False):
        for ov in (True, False):
            res.append(_h("reflect_%s%d" % ("row" if use_row else "col", ov),
                          (lambda u, o, bg: lambda g: _reflect_across(g, bg, u, o))(use_row, ov, bg),
                          4.5))
    cols = sorted(ctx.in_palette - {bg if bg is not None else -1})
    busiest = max((sum(1 for r in a for v in r if v != G.bg_or(a, bg))
                   for a in ctx.all_inputs), default=0)
    orders = (("v", "both", "h"), ("h", "both", "v"))
    for assign in (_ray_assignments(cols) if busiest <= 40 else []):
        for oi, order in enumerate(orders):
            res.append(_h("rays_%s%d" % ("".join("%d%s" % (k, v[0]) for k, v in
                                                 sorted(assign.items())), oi),
                          (lambda a, o, bg: lambda g: _color_rays(g, bg, a, o))(assign, order, bg),
                          5.5 + 0.2 * sum(1 for v in assign.values() if v != "none")))
    for c in sorted(ctx.in_palette - {bg if bg is not None else -1}):
        for diag in (True, False):
            res.append(_h("bounce#%d%d" % (c, diag),
                          (lambda c, d, bg: lambda g: _bounce(g, bg, c, d))(c, diag, bg),
                          5.6))
    for mc in sorted(ctx.in_palette - {bg if bg is not None else -1}):
        for mode in ("center", "corner", "inside"):
            res.append(_h("moveto#%d_%s" % (mc, mode),
                          (lambda m, md, bg: lambda g: _move_to_markers(g, bg, m, md))(mc, mode, bg),
                          5.4))
    for diag in (False, True):
        for tie_bg in (True, False):
            res.append(_h("voronoi%d%d" % (diag, tie_bg),
                          (lambda d, t, bg: lambda g: _voronoi(g, bg, d, t))(diag, tie_bg, bg),
                          4.4))
        res.append(_h("fill_enclosed%d" % diag,
                      (lambda d, bg: lambda g: _fill_enclosed(g, bg, d))(diag, bg),
                      4.2))
        for c in sorted(ctx.out_palette):
            for rect in (False, True):
                res.append(_h("fill_enclosed%d#%d%s" % (diag, c, "r" if rect else ""),
                              (lambda d, c, r, bg: lambda g: _fill_enclosed(g, bg, d, c, r))(diag, c, rect, bg),
                              4.6))
    for seg in ("c8", "m8", "c4"):
        for dt in (True, False):
            res.append(_h("moveanchor_%s%d" % (seg, dt),
                          (lambda s, d, bg: lambda g: _move_to_anchor(g, bg, s, d, True))(seg, dt, bg),
                          4.8))
    for dn, (dr, dc) in list(DIRS.items()) + list(DIAG.items()):
        for alt in sorted(ctx.out_palette):
            res.append(_h("stripe_%s#%d" % (dn, alt),
                          (lambda v, a, bg: lambda g: _ray_striped(g, bg, v[0], v[1], a, 2))((dr, dc), alt, bg),
                          5.6))
    for seg in ("c8", "m8"):
        for both in (True, False):
            res.append(_h("repeat_%s%d" % (seg, both),
                          (lambda s, b, bg: lambda g: _repeat_translation(g, bg, s, b))(seg, both, bg),
                          5.0))
        for rec in (False, True):
            for keep in (True, False):
                res.append(_h("stamp_%s%d%d" % (seg, rec, keep),
                              (lambda s, r, k, bg: lambda g: _stamp_at_markers(g, bg, s, r, k))(seg, rec, keep, bg),
                              5.2))
    for seg in ("c8", "m8", "c4"):
        for am in ("largest", "bbox"):
            res.append(_h("anchor_%s_%s" % (seg, am),
                          (lambda s, a, bg: lambda g: _connect_to_anchor(g, bg, s, a))(seg, am, bg),
                          4.6))
    for seg in ("c8", "m8"):
        for mode in ("h", "v", "both"):
            res.append(_h("symmetrise_%s_%s" % (seg, mode),
                          (lambda s, m, bg: lambda g: _symmetrise(g, bg, s, m))(seg, mode, bg),
                          5.0))
    return res
