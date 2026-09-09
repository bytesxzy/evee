"""Compositional search over object programs.

``objops`` gives the general enumerator a vocabulary for saying *which thing* to
act on, but only as seed operators -- applied once, to the raw input, after
which the pixel-level library takes over.  That leaves the obvious next step
unreachable: an object operator applied to the result of another one.  "Throw
away everything that touches the edge, then drop what is left, then keep the
largest" is three object steps and nothing in the engine can express it.

This is a small, dedicated search over exactly that space:

    program := op | op(program)      where op ranges over
               (selector x action) plus a handful of cheap whole-grid moves

It is a separate module rather than a wider setting on the main enumerator
because the two spaces have different economics.  The object vocabulary is
large (a few hundred operators) but its branching collapses immediately --
most selectors pick the same objects on any given grid, so most operators are
observationally identical and are pruned after one level.  The pixel library is
the opposite.  Mixing them at every depth multiplies the two branching factors
for no gain; keeping them apart lets each run at the depth it can afford.

The pruning is the same principle the main enumerator uses: programs are
compared by the tuple of grids they produce on this task's own inputs, never by
their syntax, so two different ways of saying the same thing cost one node.  A
fitted colour map is offered as a final step, which lets the search stop at the
right *shape* and let the recolouring be read off rather than searched for.
"""

import time

from .. import grid as G
from .. import objops
from ..task import Hyp

SOLVER = "objects"
PHASE = 2

_BEAM = 140
_MAX_DEPTH = 3


def _h(n, f, c):
    return Hyp(n, f, c, SOLVER)


def _grid_ops(ctx):
    """A few cheap whole-grid moves, so a chain can finish with a crop."""
    bg = ctx.bg
    ops = [("crop", 1.0, lambda g, b=bg: G.crop_to_content(g, b))]
    for name, f in G.DIHEDRAL[1:]:
        ops.append((name, 1.0, f))
    for d in ("down", "up", "left", "right"):
        ops.append(("grav_" + d, 1.4,
                    (lambda d, b=bg: lambda g: G.gravity(g, b, d))(d)))
    ops.append(("dedup", 1.4, G.dedup))
    return ops


def _apply(fn, state, deadline):
    out = []
    for g in state:
        if time.time() > deadline:
            return None
        try:
            r = fn(g)
        except Exception:
            return None
        if r is None or not isinstance(r, tuple) or not G.valid(r):
            return None
        out.append(r)
    return tuple(out)


def _fit_cmap(state, target):
    m = {}
    for a, b in zip(state, target):
        if G.dims(a) != G.dims(b):
            return None
        for ra, rb in zip(a, b):
            for x, y in zip(ra, rb):
                if m.setdefault(x, y) != y:
                    return None
    return m if any(k != v for k, v in m.items()) else None


def _chain(fns):
    def run(g):
        for f in fns:
            g = f(g)
            if g is None:
                return None
        return g
    return run


def _distance(state, target):
    """How near a state is to the demonstrated outputs; ties broken by shape."""
    n = shape = 0.0
    for g, t in zip(state, target):
        if G.dims(g) == G.dims(t):
            shape += 1.0
            a = sum(1 for ra, rb in zip(g, t) for x, y in zip(ra, rb) if x == y)
            n += a / float(max(1, G.area(t)))
    k = float(len(target))
    return shape / k * 1.5 + n / k


def generate(ctx):
    deadline = ctx.deadline or (time.time() + 4.0)
    if time.time() > deadline:
        return []
    n_tr = len(ctx.train)
    grids = ctx.inputs + ctx.test_inputs
    target = ctx.outputs
    try:
        ops = objops.object_ops(ctx) + _grid_ops(ctx)
    except Exception:
        return []
    seen = {grids}
    frontier = [(grids, (), "$", 0.0)]
    found = {}

    def record(state, fns, name, cost):
        prev = found.get(state)
        if prev is None or cost < prev[0]:
            found[state] = (cost, name, _chain(fns))

    for depth in range(1, _MAX_DEPTH + 1):
        nxt = []
        for state, fns, name, cost in frontier:
            if time.time() > deadline:
                break
            for oname, ocost, f in ops:
                if time.time() > deadline:
                    break
                st = _apply(f, state, deadline)
                if st is None or st in seen:
                    continue
                seen.add(st)
                nm = "%s(%s)" % (oname, name)
                cc = cost + ocost
                if st[:n_tr] == target:
                    record(st, fns + (f,), nm, cc)
                    continue
                m = _fit_cmap(st[:n_tr], target)
                if m is not None:
                    cf = (lambda mm: lambda g: G.apply_cmap(g, mm))(m)
                    st2 = _apply(cf, st, deadline)
                    if st2 is not None and st2[:n_tr] == target:
                        record(st2, fns + (f, cf), "cmap(%s)" % nm,
                               cc + 1.4 + 0.15 * len(m))
                nxt.append((_distance(st[:n_tr], target), (st, fns + (f,), nm, cc)))
            if len(found) >= 4:
                break
        if not nxt or time.time() > deadline or len(found) >= 4:
            break
        nxt.sort(key=lambda r: -r[0])
        frontier = [rec for _s, rec in nxt[:_BEAM]]
    out = []
    for cost, name, fn in sorted(found.values())[:6]:
        out.append(_h("chain:" + name, fn, 2.6 + cost))
    return out
