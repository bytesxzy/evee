"""Find every occurrence of a learned window and stamp on it.

Some rules are stated over *positions that look a certain way* rather than over
objects: "wherever there is a two-by-two empty square, fill it", "wherever this
motif appears, mark it". Nothing in the object vocabulary sees those, because
the interesting region is not an object -- it is a hole, or a coincidence of
shape, and it may overlap its neighbours.

The window and the paint are both read off the demonstrations: group the cells
the output changed, check they all share one shape, one before-content and one
after-content, and then look for that before-content everywhere.
"""

from collections import Counter

from .. import grid as G
from ..task import Hyp

SOLVER = "sequence"


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


def _diff_groups(a, b, diag):
    """Connected groups of changed cells."""
    h, w = G.dims(a)
    diff = [[a[r][c] != b[r][c] for c in range(w)] for r in range(h)]
    seen = [[False] * w for _ in range(h)]
    nb = G.N8 if diag else G.N4
    out = []
    for r in range(h):
        for c in range(w):
            if not diff[r][c] or seen[r][c]:
                continue
            st = [(r, c)]
            seen[r][c] = True
            cells = []
            while st:
                cr, cc = st.pop()
                cells.append((cr, cc))
                for dr, dc in nb:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < h and 0 <= nc < w and diff[nr][nc]
                            and not seen[nr][nc]):
                        seen[nr][nc] = True
                        st.append((nr, nc))
            out.append(cells)
    return out


def _fit(ctx, diag):
    """Candidate (before, after) windows read off the changed regions.

    Adjacent occurrences merge into one larger changed region, so requiring a
    single window across every group is too strict -- the merged ones are the
    same rule applied twice.  Take the most frequent window and the smallest
    one as candidates; the orchestrator validates them like anything else.
    """
    cnt = Counter()
    for a, b in ctx.train:
        if G.dims(a) != G.dims(b):
            return []
        groups = _diff_groups(a, b, diag)
        if not groups or len(groups) > 60:
            return []
        for cells in groups:
            r0, c0, r1, c1 = G.bbox_of(cells)
            if (r1 - r0 + 1) * (c1 - c0 + 1) > 25:
                continue
            wa = G.subgrid(a, r0, c0, r1, c1)
            wb = G.subgrid(b, r0, c0, r1, c1)
            if wa is None or wb is None or wa == wb:
                continue
            cnt[(wa, wb)] += 1
    if not cnt:
        return []
    out = []
    modal = cnt.most_common(1)[0]
    if modal[1] >= 2:
        out.append(modal[0])
    smallest = min(cnt, key=lambda k: (G.area(k[0]), k))
    if smallest not in out:
        out.append(smallest)
    return out


def _apply(g, before, after, overlap):
    h, w = G.dims(g)
    ph, pw = G.dims(before)
    if ph > h or pw > w:
        return None
    hits = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            if all(g[r + i][c:c + pw] == before[i] for i in range(ph)):
                hits.append((r, c))
    if not hits:
        return None
    used = set()
    out = [list(x) for x in g]
    for r, c in hits:
        cells = [(r + i, c + j) for i in range(ph) for j in range(pw)]
        if not overlap and any(p in used for p in cells):
            continue
        for i in range(ph):
            for j in range(pw):
                out[r + i][c + j] = after[i][j]
        used.update(cells)
    return tuple(tuple(x) for x in out)


def generate(ctx):
    if not ctx.same_shape:
        return []
    res = []
    for diag in (False, True):
        try:
            fits = _fit(ctx, diag)
        except Exception:
            fits = []
        for before, after in fits:
            for overlap in (True, False):
                res.append(_h("stamp%dx%d%s%s" % (len(before), len(before[0]),
                                                  "d" if diag else "",
                                                  "o" if overlap else ""),
                              (lambda b, a, o: lambda g: _apply(g, b, a, o))(before, after, overlap),
                              4.0))
    return res
