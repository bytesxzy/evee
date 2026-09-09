"""Core grid algebra for ASTRA.

A Grid is an immutable ``tuple`` of ``tuple`` of small ints (0-9).  Immutability
is load-bearing: every solver in the portfolio caches on grid identity, and the
bottom-up enumerator prunes by observational equivalence, which requires grids
to be hashable.
"""

from collections import Counter, deque

NCOLORS = 10


# --------------------------------------------------------------------------
# basics
# --------------------------------------------------------------------------

def dims(g):
    return len(g), len(g[0]) if g else 0


def area(g):
    return len(g) * (len(g[0]) if g else 0)


def valid(g):
    """Rectangular immutable array, including internal masks containing None."""
    if not isinstance(g, tuple) or not g or not isinstance(g[0], tuple):
        return False
    w = len(g[0])
    if w == 0 or w > 60 or len(g) > 60:
        return False
    for r in g:
        if not isinstance(r, tuple) or len(r) != w:
            return False
    return True


def is_grid(g):
    """True only for a legal immutable ARC grid (never coerces cell values)."""
    return valid(g) and all(type(v) is int and 0 <= v < NCOLORS
                            for row in g for v in row)


def as_grid(value):
    """Normalize rows without silently truncating floats or parsing strings."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("grid must be a list or tuple of rows")
    if any(not isinstance(row, (list, tuple)) for row in value):
        raise ValueError("grid rows must be lists or tuples")
    g = tuple(tuple(row) for row in value)
    if not is_grid(g):
        raise ValueError("grid must be rectangular, 1..60 by 1..60, with integer colors 0..9")
    return g


def const_grid(h, w, c):
    row = (c,) * w
    return (row,) * h


def from_list(ll):
    return tuple(tuple(int(v) for v in r) for r in ll)


def to_list(g):
    return [list(r) for r in g]


# --------------------------------------------------------------------------
# dihedral group
# --------------------------------------------------------------------------

def transpose(g):
    return tuple(zip(*g))


def flip_h(g):
    return tuple(r[::-1] for r in g)


def flip_v(g):
    return g[::-1]


def rot90(g):
    return tuple(zip(*g[::-1]))


def rot180(g):
    return tuple(r[::-1] for r in g[::-1])


def rot270(g):
    return tuple(zip(*g))[::-1]


def anti_transpose(g):
    return tuple(zip(*[r[::-1] for r in g]))[::-1]


DIHEDRAL = (
    ("id", lambda g: g),
    ("rot90", rot90),
    ("rot180", rot180),
    ("rot270", rot270),
    ("flip_h", flip_h),
    ("flip_v", flip_v),
    ("transpose", transpose),
    ("anti_transpose", anti_transpose),
)
DIHEDRAL_MAP = dict(DIHEDRAL)


def orbit(g):
    """All 8 dihedral images (deduplicated, name -> grid)."""
    return {n: f(g) for n, f in DIHEDRAL}


# --------------------------------------------------------------------------
# colour statistics
# --------------------------------------------------------------------------

def histogram(g):
    c = Counter()
    for row in g:
        c.update(row)
    return c


def palette(g):
    s = set()
    for row in g:
        s.update(row)
    return s


def most_common_color(g):
    h = histogram(g)
    return max(h.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def least_common_color(g):
    h = histogram(g)
    return min(h.items(), key=lambda kv: (kv[1], kv[0]))[0]


def background(g):
    """Best-guess background: 0 when present in quantity, else modal colour."""
    h = histogram(g)
    n = area(g)
    if h.get(0, 0) * 4 >= n:
        return 0
    mc = max(h.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    if h.get(0, 0) and h[0] >= 0.25 * h[mc]:
        return 0
    return mc


def bg_or(g, bg):
    """Resolve a background: an explicit colour, or infer it from this grid."""
    return background(g) if bg is None else bg


def count_color(g, c):
    n = 0
    for row in g:
        n += row.count(c)
    return n


def replace_color(g, a, b):
    return tuple(tuple(b if v == a else v for v in r) for r in g)


def apply_cmap(g, m):
    return tuple(tuple(m.get(v, v) for v in r) for r in g)


# --------------------------------------------------------------------------
# cropping / subgrids
# --------------------------------------------------------------------------

def bbox_of(cells):
    r0 = min(p[0] for p in cells)
    r1 = max(p[0] for p in cells)
    c0 = min(p[1] for p in cells)
    c1 = max(p[1] for p in cells)
    return r0, c0, r1, c1


def subgrid(g, r0, c0, r1, c1):
    """Inclusive corners; returns None when out of range."""
    h, w = dims(g)
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w or r1 < r0 or c1 < c0:
        return None
    return tuple(r[c0:c1 + 1] for r in g[r0:r1 + 1])


def crop_to_content(g, bg=None):
    if bg is None:
        bg = background(g)
    cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != bg]
    if not cells:
        return None
    return subgrid(g, *bbox_of(cells))


def trim_border(g, n=1):
    h, w = dims(g)
    if h <= 2 * n or w <= 2 * n:
        return None
    return subgrid(g, n, n, h - 1 - n, w - 1 - n)


def pad(g, n, c):
    h, w = dims(g)
    row = (c,) * (w + 2 * n)
    body = tuple((c,) * n + r + (c,) * n for r in g)
    return (row,) * n + body + (row,) * n


def half(g, which):
    h, w = dims(g)
    if which == "top":
        return g[:h // 2] if h >= 2 else None
    if which == "bottom":
        return g[(h + 1) // 2:] if h >= 2 else None
    if which == "left":
        return tuple(r[:w // 2] for r in g) if w >= 2 else None
    if which == "right":
        return tuple(r[(w + 1) // 2:] for r in g) if w >= 2 else None
    return None


def quadrant(g, i):
    h, w = dims(g)
    hh, hw = h // 2, w // 2
    if hh == 0 or hw == 0:
        return None
    if i == 0:
        return tuple(r[:hw] for r in g[:hh])
    if i == 1:
        return tuple(r[w - hw:] for r in g[:hh])
    if i == 2:
        return tuple(r[:hw] for r in g[h - hh:])
    return tuple(r[w - hw:] for r in g[h - hh:])


# --------------------------------------------------------------------------
# scaling / tiling / joining
# --------------------------------------------------------------------------

def upscale(g, ky, kx):
    if ky < 1 or kx < 1 or len(g) * ky > 60 or len(g[0]) * kx > 60:
        return None
    out = []
    for row in g:
        nr = tuple(v for v in row for _ in range(kx))
        out.extend([nr] * ky)
    return tuple(out)


def downscale(g, ky, kx):
    h, w = dims(g)
    if ky < 1 or kx < 1 or h % ky or w % kx:
        return None
    out = []
    for r in range(0, h, ky):
        row = []
        for c in range(0, w, kx):
            v = g[r][c]
            for rr in range(r, r + ky):
                for cc in range(c, c + kx):
                    if g[rr][cc] != v:
                        return None
            row.append(v)
        out.append(tuple(row))
    return tuple(out)


def block_reduce_mode(g, ky, kx):
    """Downscale by majority vote (tolerates noise where ``downscale`` fails)."""
    h, w = dims(g)
    if ky < 1 or kx < 1 or h % ky or w % kx:
        return None
    out = []
    for r in range(0, h, ky):
        row = []
        for c in range(0, w, kx):
            cnt = Counter()
            for rr in range(r, r + ky):
                cnt.update(g[rr][c:c + kx])
            row.append(max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0])
        out.append(tuple(row))
    return tuple(out)


def block_reduce_nonbg(g, ky, kx, bg):
    """Downscale by taking each block's single non-background colour.

    ``downscale`` needs uniform blocks and ``block_reduce_mode`` is swamped by
    background; neither can shrink a sparse grid whose blocks each hold one
    coloured cell.
    """
    h, w = dims(g)
    if ky < 1 or kx < 1 or h % ky or w % kx:
        return None
    out = []
    for r in range(0, h, ky):
        row = []
        for c in range(0, w, kx):
            vals = set()
            for rr in range(r, r + ky):
                for cc in range(c, c + kx):
                    v = g[rr][cc]
                    if v != bg:
                        vals.add(v)
            if len(vals) > 1:
                return None
            row.append(vals.pop() if vals else bg)
        out.append(tuple(row))
    return tuple(out)


def tile(g, ky, kx):
    h, w = dims(g)
    if ky < 1 or kx < 1 or h * ky > 60 or w * kx > 60:
        return None
    body = tuple(r * kx for r in g)
    return body * ky


def hconcat(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    if len(a[0]) + len(b[0]) > 60:
        return None
    return tuple(x + y for x, y in zip(a, b))


def vconcat(a, b):
    if a is None or b is None or len(a[0]) != len(b[0]):
        return None
    if len(a) + len(b) > 60:
        return None
    return a + b


def paste(base, patch, r0, c0):
    """Overwrite ``patch`` onto a copy of ``base`` at (r0, c0); clipped."""
    h, w = dims(base)
    ph, pw = dims(patch)
    out = [list(r) for r in base]
    for r in range(ph):
        rr = r0 + r
        if 0 <= rr < h:
            prow = patch[r]
            orow = out[rr]
            for c in range(pw):
                cc = c0 + c
                if 0 <= cc < w:
                    orow[cc] = prow[c]
    return tuple(tuple(r) for r in out)


def paste_masked(base, patch, r0, c0, transparent):
    h, w = dims(base)
    ph, pw = dims(patch)
    out = [list(r) for r in base]
    for r in range(ph):
        rr = r0 + r
        if 0 <= rr < h:
            prow = patch[r]
            orow = out[rr]
            for c in range(pw):
                v = prow[c]
                cc = c0 + c
                if v != transparent and 0 <= cc < w:
                    orow[cc] = v
    return tuple(tuple(r) for r in out)


# --------------------------------------------------------------------------
# structural analysis
# --------------------------------------------------------------------------

def uniform_rows(g):
    return [r for r, row in enumerate(g) if len(set(row)) == 1]


def uniform_cols(g):
    return uniform_rows(transpose(g))


def dedup_rows(g):
    out = [g[0]]
    for r in g[1:]:
        if r != out[-1]:
            out.append(r)
    return tuple(out)


def dedup_cols(g):
    return transpose(dedup_rows(transpose(g)))


def dedup(g):
    return dedup_cols(dedup_rows(g))


def row_period(g):
    """Smallest p with g[r] == g[r+p] for all valid r (p == h means none)."""
    h = len(g)
    for p in range(1, h + 1):
        if all(g[r] == g[r + p] for r in range(h - p)):
            return p
    return h


def col_period(g):
    return row_period(transpose(g))


def symmetries(g, ignore=None):
    """Names of dihedral maps under which ``g`` is invariant.

    ``ignore`` marks cells that are unknown; they never contradict.
    """
    out = []
    for name, f in DIHEDRAL:
        if name == "id":
            continue
        t = f(g)
        if dims(t) != dims(g):
            continue
        if ignore is None:
            if t == g:
                out.append(name)
        else:
            ok = True
            for r, row in enumerate(g):
                tr = t[r]
                for c, v in enumerate(row):
                    if v != ignore and tr[c] != ignore and v != tr[c]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append(name)
    return out


# --------------------------------------------------------------------------
# connectivity
# --------------------------------------------------------------------------

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def flood_regions(g, bg, diag=False, same_color=True):
    """Connected components of non-background cells.

    Returns a list of ``(color_or_None, frozenset_of_cells)``.
    """
    h, w = dims(g)
    nb = N8 if diag else N4
    seen = [[False] * w for _ in range(h)]
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or g[r][c] == bg:
                continue
            col = g[r][c]
            q = deque([(r, c)])
            seen[r][c] = True
            cells = []
            multi = False
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in nb:
                    nr, ncc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= ncc < w and not seen[nr][ncc]:
                        v = g[nr][ncc]
                        if v == bg:
                            continue
                        if same_color and v != col:
                            continue
                        if v != col:
                            multi = True
                        seen[nr][ncc] = True
                        q.append((nr, ncc))
            out.append((None if (multi or not same_color) else col, frozenset(cells)))
    return out


def holes(g, bg, diag=False):
    """Background cells not connected to the border (enclosed regions)."""
    h, w = dims(g)
    nb = N8 if diag else N4
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
    while q:
        cr, cc = q.popleft()
        for dr, dc in nb:
            nr, ncc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= ncc < w and not seen[nr][ncc] and g[nr][ncc] == bg:
                seen[nr][ncc] = True
                q.append((nr, ncc))
    return [(r, c) for r in range(h) for c in range(w)
            if g[r][c] == bg and not seen[r][c]]


def fill_holes(g, color, bg=None, diag=False):
    if bg is None:
        bg = background(g)
    hs = holes(g, bg, diag)
    if not hs:
        return g
    out = [list(r) for r in g]
    for r, c in hs:
        out[r][c] = color
    return tuple(tuple(r) for r in out)


# --------------------------------------------------------------------------
# masks / logic
# --------------------------------------------------------------------------

def cellwise(a, b, fn):
    if a is None or b is None or dims(a) != dims(b):
        return None
    return tuple(tuple(fn(x, y) for x, y in zip(ra, rb)) for ra, rb in zip(a, b))


def gravity(g, bg, direction):
    """Slide every non-bg cell as far as it can go, column/row-wise."""
    bg = bg_or(g, bg)
    h, w = dims(g)
    if direction in ("down", "up"):
        cols = []
        for c in range(w):
            vals = [g[r][c] for r in range(h) if g[r][c] != bg]
            padn = [bg] * (h - len(vals))
            cols.append(padn + vals if direction == "down" else vals + padn)
        return tuple(tuple(cols[c][r] for c in range(w)) for r in range(h))
    out = []
    for row in g:
        vals = [v for v in row if v != bg]
        padn = [bg] * (w - len(vals))
        out.append(tuple(padn + vals if direction == "right" else vals + padn))
    return tuple(out)


def translate(g, dr, dc, fill):
    fill = bg_or(g, fill)
    h, w = dims(g)
    out = [[fill] * w for _ in range(h)]
    for r in range(h):
        nr = r + dr
        if 0 <= nr < h:
            for c in range(w):
                nc = c + dc
                if 0 <= nc < w:
                    out[nr][nc] = g[r][c]
    return tuple(tuple(r) for r in out)


def wrap_translate(g, dr, dc):
    h, w = dims(g)
    return tuple(tuple(g[(r - dr) % h][(c - dc) % w] for c in range(w)) for r in range(h))
