"""Object segmentation and object-level features.

ARC tasks are overwhelmingly about *things* rather than pixels, so the quality
of the segmentation vocabulary bounds what any downstream solver can express.
We therefore expose several segmentations and let the portfolio try each: no
single notion of "object" is right for all tasks.
"""

from collections import Counter

from .grid import (N4, N8, background, bbox_of, dims, flood_regions, histogram,
                   subgrid)


class Obj:
    """A set of cells lifted out of a grid, with a normalised patch view."""

    __slots__ = ("cells", "color", "r0", "c0", "r1", "c1", "grid_shape",
                 "_patch", "_mask", "_norm", "_holes")

    def __init__(self, cells, grid, color=None):
        self.cells = cells                       # frozenset[(r, c)]
        self.r0, self.c0, self.r1, self.c1 = bbox_of(cells)
        self.grid_shape = dims(grid)
        if color is None:
            cc = Counter(grid[r][c] for r, c in cells)
            color = max(cc.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        self.color = color
        self._patch = None
        self._mask = None
        self._norm = None
        self._holes = {}
        self._build(grid)

    def _build(self, grid):
        h = self.r1 - self.r0 + 1
        w = self.c1 - self.c0 + 1
        patch = [[None] * w for _ in range(h)]
        for r, c in self.cells:
            patch[r - self.r0][c - self.c0] = grid[r][c]
        self._patch = tuple(tuple(row) for row in patch)
        self._mask = tuple(tuple(1 if v is not None else 0 for v in row)
                           for row in patch)

    # -- geometry ---------------------------------------------------------
    @property
    def height(self):
        return self.r1 - self.r0 + 1

    @property
    def width(self):
        return self.c1 - self.c0 + 1

    @property
    def size(self):
        return len(self.cells)

    @property
    def bbox_area(self):
        return self.height * self.width

    @property
    def patch(self):
        """Bounding-box patch; cells outside the object are ``None``."""
        return self._patch

    @property
    def mask(self):
        return self._mask

    def filled(self, bg=0):
        return tuple(tuple(bg if v is None else v for v in row) for row in self._patch)

    def is_rect(self):
        return self.size == self.bbox_area

    def is_square(self):
        return self.height == self.width

    def colors(self):
        s = set()
        for row in self._patch:
            for v in row:
                if v is not None:
                    s.add(v)
        return s

    def touches_border(self):
        h, w = self.grid_shape
        return self.r0 == 0 or self.c0 == 0 or self.r1 == h - 1 or self.c1 == w - 1

    def norm_key(self):
        """Shape identity modulo translation (not colour)."""
        if self._norm is None:
            self._norm = self._mask
        return self._norm

    def holes_count(self, diag=False):
        if diag in self._holes:
            return self._holes[diag]
        n = self._holes_count(diag)
        self._holes[diag] = n
        return n

    def _holes_count(self, diag):
        h, w = self.height, self.width
        seen = [[False] * w for _ in range(h)]
        m = self._mask
        stack = []
        for r in range(h):
            for c in (0, w - 1):
                if not m[r][c] and not seen[r][c]:
                    seen[r][c] = True
                    stack.append((r, c))
        for c in range(w):
            for r in (0, h - 1):
                if not m[r][c] and not seen[r][c]:
                    seen[r][c] = True
                    stack.append((r, c))
        nb = N8 if diag else N4
        while stack:
            cr, cc = stack.pop()
            for dr, dc in nb:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and not m[nr][nc]:
                    seen[nr][nc] = True
                    stack.append((nr, nc))
        # count enclosed components
        cnt = 0
        vis2 = [[False] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if m[r][c] or seen[r][c] or vis2[r][c]:
                    continue
                cnt += 1
                st = [(r, c)]
                vis2[r][c] = True
                while st:
                    cr, cc = st.pop()
                    for dr, dc in nb:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w and not m[nr][nc]
                                and not seen[nr][nc] and not vis2[nr][nc]):
                            vis2[nr][nc] = True
                            st.append((nr, nc))
        return cnt

    def __repr__(self):
        return "Obj(c=%s,n=%d,@%d,%d,%dx%d)" % (
            self.color, self.size, self.r0, self.c0, self.height, self.width)


# --------------------------------------------------------------------------
# segmentations
# --------------------------------------------------------------------------

def seg_connected(grid, bg, diag, same_color):
    return [Obj(cells, grid, col)
            for col, cells in flood_regions(grid, bg, diag, same_color)]


def seg_by_color(grid, bg):
    """One object per non-background colour (may be disconnected)."""
    h, w = dims(grid)
    buckets = {}
    for r in range(h):
        row = grid[r]
        for c in range(w):
            v = row[c]
            if v != bg:
                buckets.setdefault(v, []).append((r, c))
    return [Obj(frozenset(cs), grid, col) for col, cs in sorted(buckets.items())]


def seg_cells(grid, bg):
    h, w = dims(grid)
    return [Obj(frozenset([(r, c)]), grid, grid[r][c])
            for r in range(h) for c in range(w) if grid[r][c] != bg]


def seg_connected_gap(grid, bg, gap, same_color=True):
    """Components where cells within Chebyshev distance ``gap`` are linked.

    Real ARC objects are often *scattered* -- a dashed line, a row of marks
    with one cell between them.  Strict adjacency splits those into singletons
    and every object-level rule then misses.
    """
    h, w = dims(grid)
    pts = [(r, c) for r in range(h) for c in range(w) if grid[r][c] != bg]
    if len(pts) > 400:
        return []
    idx = {p: i for i, p in enumerate(pts)}
    parent = list(range(len(pts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, (r, c) in enumerate(pts):
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                if dr == 0 and dc == 0:
                    continue
                j = idx.get((r + dr, c + dc))
                if j is None or j < i:
                    continue
                if same_color and grid[r + dr][c + dc] != grid[r][c]:
                    continue
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a
    groups = {}
    for i, p in enumerate(pts):
        groups.setdefault(find(i), []).append(p)
    return [Obj(frozenset(cs), grid) for cs in groups.values()]


def seg_lines(grid, bg, axis):
    """Every row (or column) as one object, background included.

    A whole band of the grid is a thing a rule can talk about -- "rows that are
    all one colour become red", "the column with the most marks wins" -- and no
    connectivity-based segmentation can name it, because a row is not connected
    in any interesting way and usually is not even foreground.  Including the
    background cells is the point: the predicate is about the row as a band,
    not about the marks that happen to sit in it.
    """
    h, w = dims(grid)
    out = []
    if axis == "row":
        for r in range(h):
            out.append(Obj(frozenset((r, c) for c in range(w)), grid))
    else:
        for c in range(w):
            out.append(Obj(frozenset((r, c) for r in range(h)), grid))
    return out


SEGMENTATIONS = (
    ("rows", lambda g, bg: seg_lines(g, bg, "row")),
    ("cols", lambda g, bg: seg_lines(g, bg, "col")),
    ("g2", lambda g, bg: seg_connected_gap(g, bg, 2, True)),
    ("g2m", lambda g, bg: seg_connected_gap(g, bg, 2, False)),
    ("g3", lambda g, bg: seg_connected_gap(g, bg, 3, True)),
    ("c4", lambda g, bg: seg_connected(g, bg, False, True)),
    ("c8", lambda g, bg: seg_connected(g, bg, True, True)),
    ("m4", lambda g, bg: seg_connected(g, bg, False, False)),
    ("m8", lambda g, bg: seg_connected(g, bg, True, False)),
    ("color", seg_by_color),
    ("cells", seg_cells),
)


_SEG_CACHE = {}
_SEG_CAP = 512


def segment(grid, mode, bg=None):
    """Memoised segmentation.

    Hundreds of hypotheses per task ask for the same segmentation of the same
    grid; recomputing connected components each time dominated the runtime.
    Grids are immutable, so the cache is sound, and callers must not mutate the
    returned list.
    """
    if bg is None:
        bg = background(grid)
    key = (grid, mode, bg)
    hit = _SEG_CACHE.get(key)
    if hit is not None:
        return hit
    for name, fn in SEGMENTATIONS:
        if name == mode:
            res = fn(grid, bg)
            if len(_SEG_CACHE) > _SEG_CAP:
                _SEG_CACHE.clear()
            _SEG_CACHE[key] = res
            return res
    raise KeyError(mode)


# --------------------------------------------------------------------------
# object rankings (used as selection features by several solvers)
# --------------------------------------------------------------------------

def _rank_values(objs, key):
    return [key(o) for o in objs]


OBJ_RANKERS = {
    "size": lambda o: o.size,
    "bbox_area": lambda o: o.bbox_area,
    "height": lambda o: o.height,
    "width": lambda o: o.width,
    "holes": lambda o: o.holes_count(),
    "ncolors": lambda o: len(o.colors()),
    "top": lambda o: -o.r0,
    "bottom": lambda o: o.r1,
    "left": lambda o: -o.c0,
    "right": lambda o: o.c1,
    "density": lambda o: o.size / float(o.bbox_area),
}


def select_extreme(objs, ranker, want_max=True, strict=True):
    """Object with the extreme value of ``ranker``; None when tied."""
    if not objs:
        return None
    f = OBJ_RANKERS[ranker]
    vals = [f(o) for o in objs]
    tgt = max(vals) if want_max else min(vals)
    hits = [o for o, v in zip(objs, vals) if v == tgt]
    if strict and len(hits) != 1:
        return None
    return hits[0]


def select_unique_shape(objs):
    """The one object whose mask occurs exactly once."""
    cnt = Counter(o.norm_key() for o in objs)
    hits = [o for o in objs if cnt[o.norm_key()] == 1]
    return hits[0] if len(hits) == 1 else None


def select_majority_shape(objs):
    cnt = Counter(o.norm_key() for o in objs)
    if not cnt:
        return None
    k, n = cnt.most_common(1)[0]
    if n < 2:
        return None
    for o in objs:
        if o.norm_key() == k:
            return o
    return None


def select_unique_color(objs):
    cnt = Counter(o.color for o in objs)
    hits = [o for o in objs if cnt[o.color] == 1]
    return hits[0] if len(hits) == 1 else None


def select_symmetric(objs, want=True):
    from .grid import symmetries
    hits = [o for o in objs if bool(symmetries(o.filled(0))) == want]
    return hits[0] if len(hits) == 1 else None


SELECTORS = []
for _r in OBJ_RANKERS:
    SELECTORS.append(("max_" + _r, (lambda r: lambda os: select_extreme(os, r, True))(_r)))
    SELECTORS.append(("min_" + _r, (lambda r: lambda os: select_extreme(os, r, False))(_r)))
SELECTORS.extend([
    ("unique_shape", select_unique_shape),
    ("majority_shape", select_majority_shape),
    ("unique_color", select_unique_color),
    ("symmetric", lambda os: select_symmetric(os, True)),
    ("asymmetric", lambda os: select_symmetric(os, False)),
])


_FEAT_CACHE = {}


def features_of(grid, mode, bg):
    """``(objs, feature_dicts)`` for a grid, computed once and cached.

    The shared statistics (size ranking, shape and colour frequencies) are
    O(n) per object if recomputed naively, so the whole feature table was
    quadratic and was being rebuilt for every candidate rule.
    """
    key = (grid, mode, bg)
    hit = _FEAT_CACHE.get(key)
    if hit is not None:
        return hit
    objs = segment(grid, mode, bg)
    shared = _shared_stats(objs, grid)
    feats = [object_features(o, objs, grid, shared) for o in objs]
    if len(_FEAT_CACHE) > 256:
        _FEAT_CACHE.clear()
    _FEAT_CACHE[key] = (objs, feats)
    return objs, feats


def _shared_stats(objs, grid):
    return {
        "sizes": sorted((x.size for x in objs), reverse=True),
        "shape_cnt": Counter(x.norm_key() for x in objs),
        "color_cnt": Counter(x.color for x in objs),
        "dims": dims(grid),
        "containment": _containment(objs),
    }


def _containment(objs):
    """For each object: what encloses it, and how much it encloses.

    "Colour each shape like the box it sits in" is an object rule whose
    evidence is entirely relational -- nothing about the shape itself says what
    colour it should be.
    """
    n = len(objs)
    inner = [(-1, 0)] * n
    if n > 60:
        return inner
    for i, o in enumerate(objs):
        best = None
        cnt = 0
        for j, p in enumerate(objs):
            if i == j:
                continue
            if (p.r0 <= o.r0 and p.c0 <= o.c0 and p.r1 >= o.r1 and p.c1 >= o.c1
                    and p.bbox_area > o.bbox_area):
                if best is None or p.bbox_area < objs[best].bbox_area:
                    best = j
            if (o.r0 <= p.r0 and o.c0 <= p.c0 and o.r1 >= p.r1 and o.c1 >= p.c1
                    and o.bbox_area > p.bbox_area):
                cnt += 1
        inner[i] = (objs[best].color if best is not None else -1, cnt)
    return inner


def object_features(o, objs, grid, shared=None):
    """Feature dict used by the object-mapping decision-list learner."""
    n = len(objs)
    if shared is None:
        shared = _shared_stats(objs, grid)
    sizes = shared["sizes"]
    shape_cnt = shared["shape_cnt"]
    color_cnt = shared["color_cnt"]
    gh, gw = shared["dims"]
    try:
        cont, ncont = shared["containment"][objs.index(o)]
    except (ValueError, IndexError, KeyError):
        cont, ncont = -1, 0
    return {
        "container": cont,
        "n_contains": ncont,
        "color": o.color,
        "size": o.size,
        "h": o.height,
        "w": o.width,
        "bbox": o.bbox_area,
        "square": int(o.is_square()),
        "rect": int(o.is_rect()),
        "holes": o.holes_count(),
        "ncolors": len(o.colors()),
        "border": int(o.touches_border()),
        "size_rank": sizes.index(o.size),
        "is_largest": int(o.size == sizes[0]),
        "is_smallest": int(o.size == sizes[-1]),
        "shape_freq": shape_cnt[o.norm_key()],
        "shape_unique": int(shape_cnt[o.norm_key()] == 1),
        "color_freq": color_cnt[o.color],
        "color_unique": int(color_cnt[o.color] == 1),
        "n_objects": n,
        "r0": o.r0, "c0": o.c0,
        "row_band": 0 if o.r0 < gh / 3 else (1 if o.r0 < 2 * gh / 3 else 2),
        "col_band": 0 if o.c0 < gw / 3 else (1 if o.c0 < 2 * gw / 3 else 2),
    }


def grid_from_objects(objs, h, w, bg):
    out = [[bg] * w for _ in range(h)]
    for o in objs:
        p = o.patch
        for r in range(o.height):
            for c in range(o.width):
                v = p[r][c]
                if v is not None:
                    rr, cc = o.r0 + r, o.c0 + c
                    if 0 <= rr < h and 0 <= cc < w:
                        out[rr][cc] = v
    return tuple(tuple(r) for r in out)


def histogram_no_bg(g, bg):
    h = histogram(g)
    h.pop(bg, None)
    return h


def object_patch_grid(o, bg=0):
    return o.filled(bg)


def crop_obj(grid, o):
    return subgrid(grid, o.r0, o.c0, o.r1, o.c1)
