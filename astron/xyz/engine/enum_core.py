"""Bottom-up program synthesis with observational-equivalence pruning.

The op library is *task-parameterised*: colours that occur in the task become
constants, the inferred background becomes the default fill.  Search proceeds
in levels, comparing programs by the tuple of grids they produce on train
inputs and test inputs (which they must also be total on). Equivalent programs
are pruned when another uses no more cost or depth. A bounded beam reserves
room for both cheap programs and programs near the training targets; inferred
color-map suffixes avoid enumerating every possible palette permutation.

The library is extensible at runtime: :func:`add_learned_op` installs an
abstraction mined from previously solved tasks (see ``engine/learn.py``), which
is how the engine's search space grows with experience.
"""

import time

from . import grid as G
from . import objects as O

LEARNED = []          # [(name, cost, factory(ctx) -> fn or None)]
OP_BIAS = {}          # op name -> search-order bonus, fitted by engine/learn.py


def add_learned_op(name, cost, factory):
    for i, (n, _c, _f) in enumerate(LEARNED):
        if n == name:
            LEARNED[i] = (name, cost, factory)
            return
    LEARNED.append((name, cost, factory))


def clear_learned():
    del LEARNED[:]


# --------------------------------------------------------------------------
# op library
# --------------------------------------------------------------------------

def unary_ops(ctx, level="full"):
    """Base library plus any abstractions learned from solved tasks."""
    ops = base_unary_ops(ctx, level)
    for name, cost, factory in LEARNED:
        try:
            fn = factory(ctx)
        except Exception:
            fn = None
        if fn is not None:
            ops.append((name, cost, fn))
    return ops


def seed_ops(ctx):
    """Operators too costly to apply at every node.

    They run once, on the raw input, seeding the frontier; the cheap library
    then composes on top of whatever they produce.  Applying a symmetry repair
    to fifteen hundred intermediate states is how a depth-4 search turns into a
    depth-1 search that ran out of time.
    """
    bg = ctx.bg
    pal = sorted(ctx.in_palette | ctx.out_palette)
    ops = [("frame_in", 2.0, (lambda b=bg: lambda g: _frame_in(g, b))()),
           ("frame_all", 2.2, (lambda b=bg: lambda g: _frame_all(g, b))()),
           ("repair", 2.2, (lambda b=bg: lambda g: _repair_op(g, b))()),
           ("complete", 2.4, (lambda b=bg: lambda g: _complete_op(g, b))()),
           ("outline", 2.2, (lambda b=bg: lambda g: _halo_op(g, b, None))()),
           ("connect", 2.2, (lambda b=bg: lambda g: _connect_op(g, b))())]
    for c in pal:
        ops.append(("halo#%d" % c, 2.4,
                    (lambda c, b=bg: lambda g: _halo_op(g, b, c))(c)))
        ops.append(("mark#%d" % c, 2.0,
                    (lambda c, b=bg: lambda g: _mark(g, b, c))(c)))
    # Object-level operators belong exactly here: they are the expensive kind
    # that pays for itself once, on the raw input, after which the cheap
    # library composes on whatever scene edit they produced.  This is what
    # gives every composition family -- compose, cascade, refine and the deep
    # enumerator -- the ability to say *which thing* to act on.
    try:
        from . import objops
        ops.extend(objops.object_ops(ctx))
    except Exception:
        pass
    return ops


def _frame_in(g, bg):
    from .solvers.regions import _frame_interior
    return _frame_interior(g, bg, "largest")


def _frame_all(g, bg):
    from .solvers.regions import _frame_content
    return _frame_content(g, bg, "largest")


def _mark(g, bg, c):
    from .solvers.regions import _marked_rect
    return _marked_rect(g, bg, c, False)


def base_unary_ops(ctx, level="full"):
    """(name, cost, fn) triples.  ``level`` trades breadth for speed."""
    bg = ctx.bg
    pal = sorted(ctx.in_palette | ctx.out_palette)
    ops = []
    a = ops.append

    for name, f in G.DIHEDRAL[1:]:
        a((name, 1.0, f))
    a(("crop", 1.2, lambda g, b=bg: G.crop_to_content(g, b)))
    a(("dedup", 1.5, G.dedup))
    a(("dedup_r", 1.8, G.dedup_rows))
    a(("dedup_c", 1.8, G.dedup_cols))
    a(("trim", 1.5, lambda g: G.trim_border(g, 1)))
    for w in ("top", "bottom", "left", "right"):
        a(("half_" + w, 1.4, (lambda w: lambda g: G.half(g, w))(w)))
    for i in range(4):
        a(("quad%d" % i, 1.6, (lambda i: lambda g: G.quadrant(g, i))(i)))
    for d in ("down", "up", "left", "right"):
        a(("grav_" + d, 1.8, (lambda d, b=bg: lambda g: G.gravity(g, b, d))(d)))
    a(("motif", 2.0, _motif))
    a(("compress", 2.0, lambda g, b=bg: _compress(g, b)))

    if level != "small":
        for c in pal:
            a(("crop#%d" % c, 1.8, (lambda c: lambda g: G.crop_to_content(g, c))(c)))
        for c in pal:
            a(("fill#%d" % c, 1.8, (lambda c, b=bg: lambda g: G.fill_holes(g, c, b))(c)))
        for c in pal:
            a(("del#%d" % c, 1.6, (lambda c, b=bg: lambda g: G.replace_color(g, c, b))(c)))
        for c in pal:
            a(("keep#%d" % c, 1.8,
               (lambda c, b=bg: lambda g: tuple(tuple(v if v == c else b for v in r) for r in g))(c)))
        a(("upx2", 1.6, lambda g: G.upscale(g, 2, 2)))
        a(("upx3", 1.8, lambda g: G.upscale(g, 3, 3)))
        a(("dnx2", 1.6, lambda g: G.downscale(g, 2, 2)))
        for k in (2, 3):
            a(("nzx%d" % k, 1.9,
               (lambda k, b=bg: lambda g: G.block_reduce_nonbg(g, k, k, b))(k)))
            a(("upx%d_" % k, 1.9, (lambda k: lambda g: G.upscale(g, k, k))(k)))
        a(("dnx3", 1.8, lambda g: G.downscale(g, 3, 3)))
        a(("tile2", 1.8, lambda g: G.tile(g, 2, 2)))
        a(("pad0", 1.8, lambda g: G.pad(g, 1, bg)))
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            a(("sh%+d%+d" % (dr, dc), 1.7,
               (lambda dr, dc, b=bg: lambda g: G.translate(g, dr, dc, b))(dr, dc)))
            a(("wr%+d%+d" % (dr, dc), 1.9,
               (lambda dr, dc: lambda g: G.wrap_translate(g, dr, dc))(dr, dc)))
        for c in pal:
            a(("paint#%d" % c, 1.7,
               (lambda c, b=bg: lambda g: tuple(
                   tuple(c if v != b else b for v in r) for r in g))(c)))
        a(("largest8", 2.0, (lambda b=bg: lambda g: _extreme(g, "c8", b, True))()))
        a(("smallest8", 2.0, (lambda b=bg: lambda g: _extreme(g, "c8", b, False))()))
        a(("largest4", 2.2, (lambda b=bg: lambda g: _extreme(g, "c4", b, True))()))
        a(("uniq_shape", 2.4, (lambda b=bg: lambda g: _uniq(g, "c8", b))()))
        a(("mode_color", 2.4, lambda g: ((G.most_common_color(g),),)))
        # Procedural operators.  Depth is expensive; breadth at depth 2 is
        # cheap, and most ARC programs are short over a *rich* library rather
        # than long over a poor one.
        a(("denoise", 2.0, (lambda b=bg: lambda g: _denoise(g, b))()))
        a(("keep_big", 2.2, (lambda b=bg: lambda g: _keep_big(g, b, True))()))
        a(("drop_big", 2.4, (lambda b=bg: lambda g: _keep_big(g, b, False))()))
        a(("bbox_fill", 2.4, (lambda b=bg: lambda g: _bbox_fill(g, b))()))
        a(("sortrows", 2.6, lambda g: tuple(sorted(g))))
        a(("sortcols", 2.6, lambda g: G.transpose(tuple(sorted(G.transpose(g))))))

    return ops


def _repair_op(g, bg):
    from .solvers.symmetry import _repair
    return _repair(g, bg, True)


def _complete_op(g, bg):
    from .solvers.symmetry import _repair_bounded
    return _repair_bounded(g, bg, False, 0.0, 2, 0)


def _denoise(g, bg):
    """Drop every single-cell object."""
    objs = O.segment(g, "c8", bg)
    if not objs or len(objs) > 300:
        return None
    out = [list(r) for r in g]
    hit = False
    for o in objs:
        if o.size == 1:
            for r, c in o.cells:
                out[r][c] = bg
            hit = True
    return tuple(tuple(r) for r in out) if hit else None


def _halo_op(g, bg, color):
    from .solvers.sequence import _halo
    return _halo(g, bg, color, False, False)


def _connect_op(g, bg):
    from .solvers.sequence import _connect
    return _connect(g, bg, None, False)


def _keep_big(g, bg, keep):
    objs = O.segment(g, "c8", bg)
    if len(objs) < 2 or len(objs) > 300:
        return None
    o = O.select_extreme(objs, "size", True)
    if o is None:
        return None
    h, w = G.dims(g)
    if keep:
        out = [[bg] * w for _ in range(h)]
        for r, c in o.cells:
            out[r][c] = g[r][c]
    else:
        out = [list(r) for r in g]
        for r, c in o.cells:
            out[r][c] = bg
    return tuple(tuple(r) for r in out)


def _bbox_fill(g, bg):
    objs = O.segment(g, "c8", bg)
    if not objs or len(objs) > 120:
        return None
    out = [list(r) for r in g]
    for o in objs:
        for r in range(o.r0, o.r1 + 1):
            for c in range(o.c0, o.c1 + 1):
                out[r][c] = o.color
    return tuple(tuple(r) for r in out)


def _compress(g, bg):
    """Drop rows/cols that are entirely background."""
    rows = [r for r in g if any(v != bg for v in r)]
    if not rows:
        return None
    t = G.transpose(tuple(rows))
    cols = [c for c in t if any(v != bg for v in c)]
    if not cols:
        return None
    return G.transpose(tuple(cols))


def _motif(g):
    from .solvers.tiling import _motif as m
    return m(g)


def _extreme(g, seg, bg, biggest):
    objs = O.segment(g, seg, bg)
    if not objs or len(objs) > 200:
        return None
    o = O.select_extreme(objs, "size", biggest)
    return None if o is None else o.filled(bg)


def _uniq(g, seg, bg):
    objs = O.segment(g, seg, bg)
    if not objs or len(objs) > 200:
        return None
    o = O.select_unique_shape(objs)
    return None if o is None else o.filled(bg)


def binary_ops(bg=0):
    """Pairwise combinators.  Parameterised by background: "empty" is not
    always colour 0, and hard-coding it silently breaks every task drawn on a
    coloured field."""
    return (
        ("hcat", 2.0, G.hconcat),
        ("vcat", 2.0, G.vconcat),
        ("and", 2.2, (lambda b: lambda x, y: _log(x, y, "and", b))(bg)),
        ("or", 2.2, (lambda b: lambda x, y: _log(x, y, "or", b))(bg)),
        ("xor", 2.2, (lambda b: lambda x, y: _log(x, y, "xor", b))(bg)),
        ("diff", 2.4, (lambda b: lambda x, y: _log(x, y, "diff", b))(bg)),
        ("over", 2.2, (lambda b: lambda x, y: _log(x, y, "over", b))(bg)),
        ("under", 2.4, (lambda b: lambda x, y: _log(y, x, "over", b))(bg)),
    )


def _log(a, b, op, bg=0):
    if a is None or b is None or G.dims(a) != G.dims(b):
        return None
    out = []
    for ra, rb in zip(a, b):
        row = []
        for x, y in zip(ra, rb):
            fx, fy = x != bg, y != bg
            if op == "and":
                row.append(x if (fx and fy) else bg)
            elif op == "or":
                row.append(x if fx else (y if fy else bg))
            elif op == "xor":
                row.append(bg if (fx == fy) else (x if fx else y))
            elif op == "diff":
                row.append(x if (fx and not fy) else bg)
            else:
                row.append(y if fy else x)
        out.append(tuple(row))
    return tuple(out)


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

class _Node:
    __slots__ = ("state", "chain", "cost", "name", "depth", "bonus")

    def __init__(self, state, chain, cost, name="$", depth=0, bonus=0.0):
        self.state = state
        self.chain = chain
        self.cost = cost
        self.name = name
        self.depth = depth
        self.bonus = bonus


def _apply_chain(chain):
    def run(g):
        for f in chain:
            try:
                g = f(g)
                if g is None or not G.valid(g):
                    return None
            except Exception:
                return None
        return g
    return run


def _apply_state(fn, state, deadline=None):
    """Run a primitive on all evidence, containing partial/invalid operators."""
    out = []
    for g in state:
        if deadline is not None and time.time() >= deadline:
            return None
        try:
            r = fn(g)
            if r is None or not G.valid(r):
                return None
            hash(r)  # Learned extensions must return immutable grids too.
        except Exception:
            return None
        out.append(r)
    return tuple(out)


def _admit(seen, node):
    """Keep the cost/depth Pareto frontier of each observable state.

    A cheaper but deeper program cannot replace a shallow one: the shallow
    program has more remaining composition depth. Test *inputs* are included
    in the state, so hypotheses with different test predictions stay distinct.
    """
    labels = seen.get(node.state, ())
    if any(d <= node.depth and c <= node.cost for d, c in labels):
        return False
    seen[node.state] = [(d, c) for d, c in labels
                        if not (node.depth <= d and node.cost <= c)] + [
                            (node.depth, node.cost)]
    return True


def _fit_cmap(state, target):
    """Infer a single consistent, supported recoloring from training only.

    Every changed color needs at least two observed cells. This prevents a
    per-cell palette from turning an arbitrary grid into a memorized answer.
    Unseen colors retain their values when the resulting program is applied.
    """
    mapping, counts = {}, {}
    for a, b in zip(state, target):
        if G.dims(a) != G.dims(b):
            return None
        for ra, rb in zip(a, b):
            for x, y in zip(ra, rb):
                if mapping.setdefault(x, y) != y:
                    return None
                counts[x] = counts.get(x, 0) + 1
    changed = {x: y for x, y in mapping.items() if x != y}
    if not changed or any(counts[x] < 2 for x in changed):
        return None
    return changed


def _goal_distance(node, target):
    """A ranking hint, never a correctness or pruning condition."""
    distance = 0.0
    for a, b in zip(node.state, target):
        ah, aw = G.dims(a)
        bh, bw = G.dims(b)
        if (ah, aw) != (bh, bw):
            distance += 1.0 + abs(ah - bh) / max(ah, bh) + abs(aw - bw) / max(aw, bw)
        else:
            distance += sum(x != y for ra, rb in zip(a, b)
                            for x, y in zip(ra, rb)) / float(ah * aw)
    return distance / len(target)


def _beam(nodes, width, target, deadline=None):
    """Reserve capacity for cheap programs and for target-near programs."""
    ranked = sorted(nodes, key=lambda n: (n.cost - n.bonus, n.cost, n.name))
    if len(ranked) <= width:
        return ranked
    cheap = ranked[:width // 2]
    chosen = {id(n) for n in cheap}
    near = []
    for n in ranked:
        if id(n) in chosen:
            continue
        if deadline is not None and time.time() >= deadline:
            return ranked[:width]
        near.append((_goal_distance(n, target) + 0.1 * (n.cost - n.bonus),
                     n.cost, n.name, len(near), n))
    near.sort(key=lambda item: item[:4])
    return cheap + [item[-1] for item in near[:width - len(cheap)]]


def _binary_shape_ok(name, left, right, target):
    """Reject impossible terminal combinations before constructing grids."""
    for a, b, t in zip(left, right, target):
        ah, aw = G.dims(a)
        bh, bw = G.dims(b)
        th, tw = G.dims(t)
        if name == "hcat":
            if ah != bh or (ah, aw + bw) != (th, tw):
                return False
        elif name == "vcat":
            if aw != bw or (ah + bh, aw) != (th, tw):
                return False
        elif (ah, aw) != (bh, bw) or (ah, aw) != (th, tw):
            return False
    return True


def search(ctx, depth=3, max_states=1400, deadline=None, level="full",
           use_binary=True, prior=None):
    """Return cost-ranked (name, cost, fn) programs fitting every train pair.

    This is bounded, incomplete synthesis: each level retains at most
    ``max_states`` programs and considers at most four times that many new
    states (at least 64), finishing the current parent's operator list.
    ``depth`` counts primitives along the longest branch, including seeds and
    inferred recoloring. Binary programs can end in a fitted recoloring.
    """
    if depth < 0 or max_states < 1:
        raise ValueError("depth must be nonnegative and max_states positive")
    n_tr = len(ctx.train)
    if not n_tr:
        return []
    if deadline is None:
        deadline = ctx.deadline
    grids = ctx.inputs + ctx.test_inputs
    target = ctx.outputs
    bias = dict(OP_BIAS)
    if prior:
        bias.update(prior)
    found = {}
    seen = {}
    root = _Node(grids, (), 0.0)
    _admit(seen, root)
    frontier = [root]
    every = [root]

    def expired(limit=deadline):
        return limit is not None and time.time() >= limit

    def results():
        return sorted(found.values(), key=lambda h: (h[1], h[0]))[:6]

    def record(node):
        previous = found.get(node.state)
        if previous is None or (node.cost, node.name) < (previous[1], previous[0]):
            found[node.state] = (node.name, node.cost, _apply_chain(node.chain))

    def check(node):
        if node.state[:n_tr] == target:
            record(node)
            return True
        if node.depth < depth:
            mapping = _fit_cmap(node.state[:n_tr], target)
            if mapping is not None:
                fn = lambda g, m=mapping: G.apply_cmap(g, m)
                st = _apply_state(fn, node.state, deadline)
                if st is not None:
                    label = ",".join("%d>%d" % p for p in sorted(mapping.items()))
                    record(_Node(st, node.chain + (fn,),
                                 node.cost + 1.6 + 0.2 * len(mapping),
                                 "cmap[%s](%s)" % (label, node.name), node.depth + 1))
        return False

    if expired():
        return []
    check(root)
    if depth == 0:
        return results()
    ops = unary_ops(ctx, level)
    ops.sort(key=lambda t: (t[1] - bias.get(t[0], 0.0), t[1], t[0]))
    seeds = seed_ops(ctx) if level != "small" else []
    # Reserve a bounded share of the remaining wall-clock budget for binary
    # reasoning; otherwise a broad unary layer can starve it entirely.
    unary_deadline = deadline
    if use_binary and deadline is not None:
        now = time.time()
        unary_deadline = now + max(0.0, deadline - now) * 0.8
    generation_limit = max(64, max_states * 4)
    for current_depth in range(1, depth + 1):
        nxt = {}
        current_ops = ops + seeds if current_depth == 1 else ops
        for node in frontier:
            if expired(unary_deadline):
                break
            for oname, ocost, f in current_ops:
                if expired(unary_deadline):
                    break
                st = _apply_state(f, node.state, unary_deadline)
                if st is None:
                    continue
                nn = _Node(st, node.chain + (f,), node.cost + ocost,
                           "%s(%s)" % (oname, node.name), current_depth,
                           node.bonus + bias.get(oname, 0.0))
                if not _admit(seen, nn):
                    continue
                check(nn)
                nxt[st] = nn
            if len(nxt) >= generation_limit:
                break
        frontier = _beam(list(nxt.values()), max_states, target, unary_deadline)
        every.extend(frontier)
        if not frontier or expired(unary_deadline) or len(found) >= 6:
            break

    if use_binary and not found:
        # Preserve shallow operands so a binary result still fits the depth
        # budget. Self-pairs are meaningful for concatenation and tiling.
        pool = _beam([n for n in every if n.depth < depth], 120, target, deadline)
        bops = binary_ops(ctx.bg)
        for na in pool:
            for nb in pool:
                for oname, ocost, bf in bops:
                    if expired():
                        return results()
                    if not _binary_shape_ok(oname, na.state, nb.state, target):
                        continue
                    st = []
                    for x, y in zip(na.state, nb.state):
                        if expired():
                            return results()
                        try:
                            r = bf(x, y)
                            if r is None or not G.valid(r):
                                break
                            hash(r)
                        except Exception:
                            break
                        st.append(r)
                    if len(st) != len(grids):
                        continue
                    nn = _Node(tuple(st), (_mk_binary(na.chain, nb.chain, bf),),
                               na.cost + nb.cost + ocost,
                               "%s(%s,%s)" % (oname, na.name, nb.name),
                               1 + max(na.depth, nb.depth))
                    if _admit(seen, nn):
                        check(nn)
    return results()


def _mk_binary(ca, cb, bf):
    ra, rb = _apply_chain(ca), _apply_chain(cb)

    def run(g):
        x, y = ra(g), rb(g)
        if x is None or y is None:
            return None
        try:
            result = bf(x, y)
            return result if result is not None and G.valid(result) else None
        except Exception:
            return None
    return run
