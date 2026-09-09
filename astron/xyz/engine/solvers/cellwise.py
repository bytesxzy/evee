"""Local-rule induction (cellular-automaton style).

For same-shape tasks we try to explain the output as a *function of a bounded
local context* around each cell.  A context family is accepted only if it is
conflict-free over every training cell; it is then applied to the test input.

This single family covers denoising, outlining, ray drawing, colour swaps,
parity stripes and a long tail of "each pixel becomes ..." tasks.  Capacity is
controlled explicitly: a rule whose context table is nearly as large as the
number of observations it was fitted on is memorisation, and is rejected.
"""

from collections import Counter

from .. import grid as G
from ..task import Hyp

SOLVER = "cellwise"
OOB = -1


# --------------------------------------------------------------------------
# context extractors: (name, cost, arity_hint, fn(grid, r, c, aux) -> key)
# --------------------------------------------------------------------------

def _get(g, r, c, h, w):
    return g[r][c] if 0 <= r < h and 0 <= c < w else OOB


def _mk_extractors(bg):
    ex = []

    def color(g, r, c, a):
        return (g[r][c],)
    ex.append(("color", 3.0, color))

    def n4o(g, r, c, a):
        h, w = a["dims"]
        return (g[r][c], _get(g, r - 1, c, h, w), _get(g, r + 1, c, h, w),
                _get(g, r, c - 1, h, w), _get(g, r, c + 1, h, w))
    ex.append(("n4", 6.0, n4o))

    def n4s(g, r, c, a):
        h, w = a["dims"]
        v = sorted((_get(g, r - 1, c, h, w), _get(g, r + 1, c, h, w),
                    _get(g, r, c - 1, h, w), _get(g, r, c + 1, h, w)))
        return (g[r][c],) + tuple(v)
    ex.append(("n4set", 5.5, n4s))

    def n8o(g, r, c, a):
        h, w = a["dims"]
        return tuple(_get(g, r + dr, c + dc, h, w)
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    ex.append(("n8", 8.0, n8o))

    def n8mask(g, r, c, a):
        h, w = a["dims"]
        v = g[r][c]
        m = 0
        for i, (dr, dc) in enumerate(G.N8):
            if _get(g, r + dr, c + dc, h, w) == v:
                m |= 1 << i
        return (v, m)
    ex.append(("n8mask", 6.0, n8mask))

    def n8bgmask(g, r, c, a):
        h, w = a["dims"]
        m = 0
        for i, (dr, dc) in enumerate(G.N8):
            u = _get(g, r + dr, c + dc, h, w)
            if u != bg and u != OOB:
                m |= 1 << i
        return (g[r][c], m)
    ex.append(("n8fg", 6.5, n8bgmask))

    def n8count(g, r, c, a):
        h, w = a["dims"]
        n = sum(1 for dr, dc in G.N8
                if _get(g, r + dr, c + dc, h, w) not in (bg, OOB))
        return (g[r][c], n)
    ex.append(("n8count", 5.0, n8count))

    def n4count(g, r, c, a):
        h, w = a["dims"]
        n = sum(1 for dr, dc in G.N4
                if _get(g, r + dr, c + dc, h, w) not in (bg, OOB))
        return (g[r][c], n)
    ex.append(("n4count", 5.0, n4count))

    def parity(g, r, c, a):
        return (g[r][c], r % 2, c % 2)
    ex.append(("parity", 4.5, parity))

    for k in (2, 3, 4):
        def modk(g, r, c, a, k=k):
            return (g[r][c], r % k, c % k)
        ex.append(("mod%d" % k, 5.0, modk))

    def border(g, r, c, a):
        h, w = a["dims"]
        return (g[r][c], int(r == 0 or c == 0 or r == h - 1 or c == w - 1))
    ex.append(("border", 4.5, border))

    def ring(g, r, c, a):
        h, w = a["dims"]
        return (g[r][c], min(r, c, h - 1 - r, w - 1 - c))
    ex.append(("ring", 5.0, ring))

    def rowcol_colors(g, r, c, a):
        rc = a["rowsets"][r]
        cc = a["colsets"][c]
        return (g[r][c], rc, cc)
    ex.append(("rowcol", 6.0, rowcol_colors))

    def rowmajor(g, r, c, a):
        return (g[r][c], a["rowmode"][r], a["colmode"][c])
    ex.append(("rcmode", 5.5, rowmajor))

    def cross(g, r, c, a):
        """Nearest non-background colour in each of the four directions."""
        return (g[r][c],) + a["rays"][r][c]
    ex.append(("rays", 7.0, cross))

    def diagpar(g, r, c, a):
        return (g[r][c], (r + c) % 2, (r - c) % 2)
    ex.append(("diagpar", 5.0, diagpar))

    def quad(g, r, c, a):
        h, w = a["dims"]
        return (g[r][c], int(r >= (h + 1) // 2), int(c >= (w + 1) // 2))
    ex.append(("quadpos", 5.0, quad))

    def rcuniform(g, r, c, a):
        """Is this cell on a wholly uniform row / column?"""
        return (g[r][c], a["rowuni"][r], a["coluni"][c])
    ex.append(("rcuni", 4.5, rcuniform))

    def rcempty(g, r, c, a):
        return (g[r][c], a["rowbg"][r], a["colbg"][c])
    ex.append(("rcempty", 4.5, rcempty))

    def objsize(g, r, c, a):
        return (g[r][c], a["osize"][r][c])
    ex.append(("objsize", 6.0, objsize))

    def objshape(g, r, c, a):
        return (g[r][c], a["oshape"][r][c])
    ex.append(("objshape", 7.0, objshape))

    def objdims(g, r, c, a):
        return (g[r][c], a["odims"][r][c])
    ex.append(("objdims", 6.5, objdims))

    def objrel(g, r, c, a):
        """Position of this cell inside its own object's bounding box."""
        return (g[r][c], a["orel"][r][c])
    ex.append(("objrel", 7.0, objrel))

    def colorcount(g, r, c, a):
        return (g[r][c], a["ccount"].get(g[r][c], 0))
    ex.append(("ccount", 4.5, colorcount))

    def pos(g, r, c, a):
        """Absolute position.  Only informative when the grids share a size --
        the capacity guard and leave-one-out check do the filtering."""
        return (g[r][c], r, c)
    ex.append(("pos", 7.5, pos))

    def rowpos(g, r, c, a):
        return (g[r][c], r)
    ex.append(("rowpos", 6.0, rowpos))

    def colpos(g, r, c, a):
        return (g[r][c], c)
    ex.append(("colpos", 6.0, colpos))

    def dist(g, r, c, a):
        """Distance to the nearest non-background cell (capped).

        Rings, halos and "colour by how far you are" rules are all functions of
        this and of nothing else local; without it they look like noise.
        """
        return (g[r][c], a["dist"][r][c])
    ex.append(("dist", 5.5, dist))

    def nearcol(g, r, c, a):
        return (g[r][c], a["near"][r][c])
    ex.append(("nearcol", 6.0, nearcol))

    def distcol(g, r, c, a):
        return (g[r][c], a["dist"][r][c], a["near"][r][c])
    ex.append(("distcol", 7.0, distcol))

    def colorrank(g, r, c, a):
        return (g[r][c], a["crank"].get(g[r][c], -1))
    ex.append(("crank", 4.5, colorrank))
    return ex


_AUX_CACHE = {}


_PAIRS = (
    ("n8fg", "parity"), ("n8fg", "border"), ("n8count", "parity"),
    ("n4count", "border"), ("objsize", "n8mask"), ("rays", "n8count"),
    ("crank", "n8fg"), ("rcuni", "n8fg"), ("objdims", "objrel"),
    ("ring", "n8fg"), ("ccount", "n8count"), ("rowpos", "colpos"),
)


def _mk_pair_extractors(bg):
    """Conjunctions of two context families.

    Single families are each a projection of the cell's situation, and plenty
    of ARC rules need two of them at once -- "a background cell next to exactly
    one filled neighbour, on an even row". The cost is a bigger table, which
    the capacity guard and the leave-one-out check are already there to police,
    so only a curated dozen pairs are offered rather than the full product.
    """
    base = {n: f for n, _c, f in _mk_extractors(bg)}
    out = []
    for a, b in _PAIRS:
        fa, fb = base.get(a), base.get(b)
        if fa is None or fb is None:
            continue
        out.append((a + "+" + b, 9.0,
                    (lambda fa, fb: lambda g, r, c, x: (fa(g, r, c, x),
                                                        fb(g, r, c, x)))(fa, fb)))
    return out


def _aux(g, bg):
    """Per-grid derived context, memoised: every extractor and every
    leave-one-out refit asks for the same analysis of the same grids."""
    key = (g, bg)
    hit = _AUX_CACHE.get(key)
    if hit is not None:
        return hit
    res = _aux_build(g, bg)
    if len(_AUX_CACHE) > 128:
        _AUX_CACHE.clear()
    _AUX_CACHE[key] = res
    return res


def _aux_build(g, bg):
    h, w = G.dims(g)
    rowsets = [tuple(sorted(set(row))) for row in g]
    cols = G.transpose(g)
    colsets = [tuple(sorted(set(col))) for col in cols]
    rowmode = [Counter(row).most_common(1)[0][0] for row in g]
    colmode = [Counter(col).most_common(1)[0][0] for col in cols]
    rays = _rays(g, bg, h, w)
    rowuni = [int(len(set(row)) == 1) for row in g]
    coluni = [int(len(set(col)) == 1) for col in cols]
    rowbg = [int(all(v == bg for v in row)) for row in g]
    colbg = [int(all(v == bg for v in col)) for col in cols]
    hist = Counter()
    for row in g:
        hist.update(row)
    order = [k for k, _v in sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))]
    crank = {c: i for i, c in enumerate(order)}
    osize = [[0] * w for _ in range(h)]
    oshape = [[None] * w for _ in range(h)]
    odims = [[None] * w for _ in range(h)]
    orel = [[None] * w for _ in range(h)]
    for _col, cells in G.flood_regions(g, bg, True, False):
        n = len(cells)
        r0, c0, r1, c1 = G.bbox_of(cells)
        dh, dw = r1 - r0 + 1, c1 - c0 + 1
        key = frozenset((r - r0, c - c0) for r, c in cells)
        for r, c in cells:
            osize[r][c] = n
            oshape[r][c] = key
            odims[r][c] = (dh, dw)
            orel[r][c] = (r - r0, c - c0, dh, dw)
    dmap, nmap = _dist_maps(g, bg, h, w)
    return {"dist": dmap, "near": nmap,
            "dims": (h, w), "rowsets": rowsets, "colsets": colsets,
            "rowmode": rowmode, "colmode": colmode, "rays": rays,
            "rowuni": rowuni, "coluni": coluni, "rowbg": rowbg, "colbg": colbg,
            "ccount": hist, "crank": crank, "osize": osize, "oshape": oshape,
            "odims": odims, "orel": orel}


def _dist_maps(g, bg, h, w):
    """Multi-source BFS: distance to, and colour of, the nearest filled cell."""
    from collections import deque
    dist = [[9] * w for _ in range(h)]
    near = [[OOB] * w for _ in range(h)]
    q = deque()
    for r in range(h):
        for c in range(w):
            if g[r][c] != bg:
                dist[r][c] = 0
                near[r][c] = g[r][c]
                q.append((r, c))
    while q:
        cr, cc = q.popleft()
        if dist[cr][cc] >= 6:
            continue
        for dr, dc in G.N4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr][nc] > dist[cr][cc] + 1:
                dist[nr][nc] = dist[cr][cc] + 1
                near[nr][nc] = near[cr][cc]
                q.append((nr, nc))
    return dist, near


def _rays(g, bg, h, w):
    up = [[OOB] * w for _ in range(h)]
    dn = [[OOB] * w for _ in range(h)]
    lf = [[OOB] * w for _ in range(h)]
    rt = [[OOB] * w for _ in range(h)]
    for c in range(w):
        last = OOB
        for r in range(h):
            up[r][c] = last
            if g[r][c] != bg:
                last = g[r][c]
        last = OOB
        for r in range(h - 1, -1, -1):
            dn[r][c] = last
            if g[r][c] != bg:
                last = g[r][c]
    for r in range(h):
        last = OOB
        for c in range(w):
            lf[r][c] = last
            if g[r][c] != bg:
                last = g[r][c]
        last = OOB
        for c in range(w - 1, -1, -1):
            rt[r][c] = last
            if g[r][c] != bg:
                last = g[r][c]
    return [[(up[r][c], dn[r][c], lf[r][c], rt[r][c]) for c in range(w)]
            for r in range(h)]


class _Rule:
    __slots__ = ("table", "ex", "bg", "strict")

    def __init__(self, table, ex, bg, strict):
        self.table = table
        self.ex = ex
        self.bg = bg
        self.strict = strict

    def __call__(self, g):
        h, w = G.dims(g)
        a = _aux(g, self.bg)
        t = self.table
        f = self.ex
        out = []
        for r in range(h):
            row = []
            for c in range(w):
                v = t.get(f(g, r, c, a))
                if v is None:
                    if self.strict:
                        return None
                    v = g[r][c]
                row.append(v)
            out.append(tuple(row))
        return tuple(out)


def _fit(train, fn, bg, ncells_cap=None):
    """Build the context table for one extractor; None on any conflict."""
    table = {}
    for a, b in train:
        aux = _aux(a, bg)
        h, w = G.dims(a)
        for r in range(h):
            brow = b[r]
            arow_get = fn
            for c in range(w):
                k = arow_get(a, r, c, aux)
                v = brow[c]
                prev = table.get(k)
                if prev is None:
                    table[k] = v
                elif prev != v:
                    return None
    return table or None


def _loo(train, fn, bg):
    """Fraction of training pairs predicted by a table fitted without them.

    This is the honest test of a local rule: a table that only reproduces the
    cells it was built from has memorised, and refitting without a pair exposes
    that immediately.  It costs one extra table build per pair, which is cheap
    compared with everything else in the portfolio.
    """
    n = len(train)
    if n < 2:
        return None
    ok = 0
    for i in range(n):
        sub = train[:i] + train[i + 1:]
        t = _fit(sub, fn, bg)
        if t is None:
            continue
        if _Rule(t, fn, bg, True)(train[i][0]) == train[i][1]:
            ok += 1
    return ok / float(n)


def generate(ctx):
    if not ctx.same_shape:
        return []
    bg = ctx.bg
    res = []
    ncells = sum(G.area(a) for a, _ in ctx.train)
    families = _mk_extractors(bg)
    if ncells >= 400:
        # paired contexts have far more capacity; only offer them when there
        # is enough evidence for the guards below to mean something
        families = families + _mk_pair_extractors(bg)
    for name, cost, fn in families:
        if ctx.timed_out():
            break
        table = _fit(ctx.train, fn, bg)
        if table is None:
            continue
        # capacity guard: a table nearly as big as the data has learnt nothing
        if len(table) * 3 > ncells:
            continue
        frac = _loo(ctx.train, fn, bg)
        # A rule that survives refitting without a pair is worth believing;
        # one that does not is a transcript of the training set.
        adj = 0.0 if frac is None else (2.0 - 4.0 * frac)
        pen = cost + len(table) * 0.02 + adj
        res.append(Hyp("ca_" + name, _Rule(table, fn, bg, True), pen, SOLVER))
        res.append(Hyp("ca_" + name + "_lax", _Rule(table, fn, bg, False),
                       pen + 1.0, SOLVER))
    return res
