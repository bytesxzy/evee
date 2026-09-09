"""Low-capacity object programs whose actions take another object as argument.

Only demonstrations are used to induce a rule. Relations and actions contain no
task identifiers, coordinates learned from answers, or output stencils. Local
consistency prunes the action vocabulary; whole-scene verification is mandatory.
"""
from collections import Counter
from itertools import product
import heapq

from .. import grid as G, objects as O
from ..task import Hyp
from . import objproc as P

SOLVER = "objects"

_SELECTORS = ("near", "same", "diff", "larger", "smaller", "row", "col",
              "rowdiff", "coldiff", "aligneddiff", "inside", "contains", "shape", "samebig") + tuple("c%d" % c for c in range(10))
_KEYS = ("color", "size_rank", "is_largest", "is_smallest", "size",
         "h", "w", "ncolors", "border", "shape_unique", "color_unique",
         "square", "rect")
_VERBS = ("paint", "fill", "box", "touch", "center", "alignrow", "aligncol",
          "mirrorh", "mirrorv", "copycenter", "copyorigin", "complete")


def _relation(a, b, key):
    row = a.r0 <= b.r1 and b.r0 <= a.r1
    col = a.c0 <= b.c1 and b.c0 <= a.c1
    if len(key) == 2 and key[0] == "c" and key[1].isdigit():
        return b.color == int(key[1])
    if key == "near":
        return True
    if key == "same":
        return a.color == b.color
    if key == "diff":
        return a.color != b.color
    if key == "larger":
        return b.size > a.size
    if key == "smaller":
        return b.size < a.size
    if key == "samebig":
        return a.color == b.color and b.size > a.size
    if key in ("row", "rowdiff"):
        return row and (key == "row" or a.color != b.color)
    if key in ("col", "coldiff"):
        return col and (key == "col" or a.color != b.color)
    if key == "aligneddiff":
        return (row or col) and a.color != b.color
    if key == "inside":
        return (b.r0 <= a.r0 <= a.r1 <= b.r1 and
                b.c0 <= a.c0 <= a.c1 <= b.c1)
    if key == "contains":
        return _relation(b, a, "inside")
    if key == "shape":
        return a.norm_key() == b.norm_key()
    return False


class Scene:
    def __init__(self, g, seg, bg):
        self.g, self.bg = g, G.bg_or(g, bg)
        self.objs = O.segment(g, seg, self.bg)
        self.features, self.peers = {}, {}
        self.writes, self.write_cells = {}, 0
        if not 2 <= len(self.objs) <= 35:
            return
        shared = O._shared_stats(self.objs, g)
        for a in self.objs:
            f = O.object_features(a, self.objs, g, shared)
            peers = {}
            distances = {id(b): P._dist(a, b) for b in self.objs if b is not a}
            for key in _SELECTORS:
                candidates = [b for b in self.objs if b is not a and _relation(a, b, key)]
                nearest = min((distances[id(b)] for b in candidates), default=None)
                hits = [b for b in candidates if distances[id(b)] == nearest]
                peers[key] = hits
                f["has_" + key] = bool(hits)
                f["color_" + key] = (hits[0].color if hits and
                    len({b.color for b in hits}) == 1 else -1)
                f["touch_" + key] = nearest == 1
            self.features[id(a)], self.peers[id(a)] = f, peers

    @property
    def valid(self):
        return bool(self.features)


def _peer(scene, obj, selector, color_only=False):
    hits = scene.peers[id(obj)][selector]
    if len(hits) == 1:
        return hits[0]
    if color_only and hits and len({p.color for p in hits}) == 1:
        return hits[0]
    return None


def _scene(g, seg, bg, cache):
    key = (g, seg, bg)
    if key not in cache:
        result = Scene(g, seg, bg)
        if len(cache) < 64:
            cache[key] = result
        return result
    return cache[key]


def _writes(scene, a, action):
    if action[0] in ("keep", "del", "solid"):
        key = (id(a), action)
    else:
        peer = _peer(scene, a, action[1], action[0] in ("paint", "fill", "box"))
        key = (id(a), action[0], id(peer) if peer else None, action[2:])
    if key in scene.writes:
        return scene.writes[key]
    result = _writes_uncached(scene, a, action)
    size = len(result) if result else 0
    if len(scene.writes) < 2048 and scene.write_cells + size <= 4000:
        scene.writes[key] = result
        scene.write_cells += size
    return result


def _writes_uncached(scene, a, action):
    g, bg = scene.g, scene.bg
    kind = action[0]
    if kind == "keep":
        return {}
    if kind == "del":
        return dict.fromkeys(a.cells, bg)
    if kind == "solid":
        return dict.fromkeys(a.cells, action[1])
    b = _peer(scene, a, action[1], kind in ("paint", "fill", "box"))
    if b is None:
        return {} if kind == "complete" else None
    h, w = G.dims(g)
    if kind == "complete":
        return _complete_from_peer(scene, a, b) or {}
    if kind == "paint":
        return dict.fromkeys(a.cells, b.color)
    if kind == "fill":
        return dict.fromkeys(P._interior(a, g, bg), b.color)
    if kind == "box":
        return {(r, c): b.color for r in range(a.r0, a.r1 + 1)
                for c in range(a.c0, a.c1 + 1) if g[r][c] == bg}
    if kind in ("copycenter", "copyorigin"):
        # The receiving object's color supplies the new role color.
        if kind == "copycenter":
            dr, dc = a.r0 + a.r1 - b.r0 - b.r1, a.c0 + a.c1 - b.c0 - b.c1
            if dr % 2 or dc % 2:
                return None
            dr, dc = dr // 2, dc // 2
        else:
            dr, dc = a.r0 - b.r0, a.c0 - b.c0
        writes = {}
        for r, c in b.cells:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            v = a.color if g[r][c] == b.color else g[r][c]
            if g[rr][cc] != bg and (rr, cc) not in a.cells and g[rr][cc] != v:
                return None
            writes[rr, cc] = v
        return writes
    if kind.startswith("connect"):
        # Connect singleton centers; the path direction is part of the program.
        if a.size != 1 or b.size != 1:
            return None
        ar, ac, br, bc = a.r0, a.c0, b.r0, b.c0
        mode, color = action[2], action[3]
        color = a.color if color == "self" else b.color if color == "peer" else color
        cells = []
        if mode == "straight":
            dy, dx = br - ar, bc - ac
            if dy and dx and abs(dy) != abs(dx):
                return None
            n = max(abs(dy), abs(dx))
            if not n:
                return None
            cells = [(ar + i * (dy // n), ac + i * (dx // n)) for i in range(1, n)]
        else:
            corner = (ar, bc) if mode == "rowfirst" else (br, ac)
            for (r0, c0), (r1, c1) in (((ar, ac), corner), (corner, (br, bc))):
                n = max(abs(r1-r0), abs(c1-c0))
                if n:
                    cells.extend((r0+i*((r1-r0)//n), c0+i*((c1-c0)//n)) for i in range(n+1))
            cells = [rc for rc in cells if rc not in a.cells and rc not in b.cells]
        if any(g[r][c] not in (bg, color) for r, c in cells):
            return None
        return dict.fromkeys(cells, color)
    if kind == "center":
        dr, dc = b.r0 + b.r1 - a.r0 - a.r1, b.c0 + b.c1 - a.c0 - a.c1
        if dr % 2 or dc % 2:
            return None
        dr, dc = dr // 2, dc // 2
    elif kind == "alignrow":
        dr, dc = (b.r0 + b.r1 - a.r0 - a.r1) // 2, 0
    elif kind == "aligncol":
        dr, dc = 0, (b.c0 + b.c1 - a.c0 - a.c1) // 2
    elif kind == "mirrorh":
        dr, dc = 0, b.c0 + b.c1 - a.c0 - a.c1
    elif kind == "mirrorv":
        dr, dc = b.r0 + b.r1 - a.r0 - a.r1, 0
    elif kind == "touch":
        rows = a.r0 <= b.r1 and b.r0 <= a.r1
        cols = a.c0 <= b.c1 and b.c0 <= a.c1
        if rows and not cols:
            dr, dc = 0, b.c0 - a.c1 - 1 if a.c1 < b.c0 else b.c1 - a.c0 + 1
        elif cols and not rows:
            dr, dc = b.r0 - a.r1 - 1 if a.r1 < b.r0 else b.r1 - a.r0 + 1, 0
        else:
            return None
    else:
        return None
    if not (dr or dc):
        return {}
    writes = dict.fromkeys(a.cells, bg)
    for r, c in a.cells:
        rr, cc = r + dr, c + dc
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if (rr, cc) not in a.cells and g[rr][cc] != bg:
            return None
        writes[rr, cc] = g[r][c]
    return writes


def _complete_from_peer(scene, receiver, donor):
    """Transfer a peer's structure through an inferred color-role renaming.

    Anchor cells constrain placement, orientation, and scale. Distinct geometric
    descriptions producing the same writes are one interpretation; different
    resulting grids are ambiguity and cause abstention.
    """
    from .analogy import _scale_patch, _placements
    g, bg = scene.g, scene.bg
    h, w = G.dims(g)
    src, dst = donor.colors(), receiver.colors()
    old, new = src - dst, dst - src
    if new and (len(old) != 1 or len(new) != 1):
        return None
    rename = dict(zip(sorted(old), sorted(new))) if new else {}
    source_counts = Counter(g[r][c] for r, c in donor.cells)
    target_counts = Counter(g[r][c] for r, c in receiver.cells)
    patch = tuple(tuple(rename.get(v, v) for v in row) for row in donor.patch)
    variants, results = set(), set()
    for _, transform in G.DIHEDRAL[:1]:
        base = transform(patch)
        for scale in range(1, min(h // len(base), w // len(base[0])) + 1):
            if new and any(source_counts[c]*scale*scale != target_counts[c] for c in src & dst):
                continue
            if donor.size * scale * scale <= receiver.size:
                continue
            if len(base)*scale > h or len(base[0])*scale > w:
                continue
            candidate = _scale_patch(base, scale)
            if candidate in variants:
                continue
            variants.add(candidate)
            for dr, dc in _placements(g, candidate, receiver, bg, h, w):
                writes = tuple(sorted(((r+dr, c+dc), v)
                    for r, row in enumerate(candidate) for c, v in enumerate(row)
                    if v is not None))
                results.add(writes)
                if len(results) > 1:
                    return None
    return dict(next(iter(results))) if results else None


def _cost(action):
    if action[0] == "keep":
        return 0.0
    if action[0] == "del":
        return 0.7
    if action[0] == "solid":
        return 1.2
    return 1.8 + (0.4 if action[0].startswith("copy") else 0.0)


class Rule:
    def __init__(self, seg, bg, keys, table, scenes=None):
        self.seg, self.bg, self.keys, self.table = seg, bg, keys, table
        self.scenes = {} if scenes is None else scenes

    def __call__(self, g):
        scene = _scene(g, self.seg, self.bg, self.scenes)
        if not scene.valid:
            return None
        draws, erase = {}, set()
        for a in scene.objs:
            f = scene.features[id(a)]
            key = tuple(f[k] for k in self.keys)
            action = self.table.get(key)
            if action is None:
                return None
            writes = _writes(scene, a, action)
            if writes is None:
                return None
            for rc, v in writes.items():
                if v == scene.bg:
                    erase.add(rc)
                elif rc in draws and draws[rc] != v:
                    return None
                else:
                    draws[rc] = v
        out = [list(row) for row in g]
        for r, c in erase:
            out[r][c] = scene.bg
        for (r, c), v in draws.items():
            out[r][c] = v
        return tuple(tuple(row) for row in out)


def _tables(groups, cap=40):
    keys = list(groups)
    opts = [groups[k] for k in keys]
    start = (0,) * len(keys)
    heap, seen = [(0.0, start)], {start}
    for _ in range(cap):
        if not heap:
            return
        _, ix = heapq.heappop(heap)
        yield {key: opts[j][ix[j]] for j, key in enumerate(keys)}
        for j in range(len(keys)):
            if ix[j] + 1 >= len(opts[j]):
                continue
            nxt = ix[:j] + (ix[j] + 1,) + ix[j+1:]
            if nxt not in seen:
                seen.add(nxt)
                heapq.heappush(heap, (sum(_cost(opts[k][n]) for k, n in enumerate(nxt)), nxt))


def generate(ctx):
    if not ctx.same_shape:
        return []
    colors = sorted(ctx.out_palette - {ctx.bg})
    actions = [("keep",), ("del",)] + [("solid", c) for c in colors]
    actions += [(verb, sel) for sel in _SELECTORS for verb in _VERBS]
    actions += [("connect", sel, mode, col) for sel in ("same", "diff", "near")
                for mode in ("straight", "rowfirst", "colfirst")
                for col in ("self", "peer", *colors)]
    keys = [()] + [(k,) for k in _KEYS]
    keys += [(prefix + sel,) for sel in _SELECTORS for prefix in ("has_", "color_", "touch_")]
    keys += [(own, "has_" + sel) for sel in _SELECTORS
             for own in ("is_largest", "is_smallest", "color_unique")]
    results, seen, scenes, layouts_seen = [], set(), {}, set()
    backgrounds = [ctx.bg] + ([None] if ctx.bg_varies else [])
    for bg, seg in product(backgrounds, ("c4", "c8", "m4", "m8", "color")):
        if ctx.timed_out():
            break
        layout = tuple((G.bg_or(g, bg), frozenset(o.cells for o in
                       O.segment(g, seg, G.bg_or(g, bg)))) for g in ctx.all_inputs)
        if layout in layouts_seen:
            continue
        layouts_seen.add(layout)
        rows = []
        failed = False
        for g, target in ctx.train:
            scene = _scene(g, seg, bg, scenes)
            if not scene.valid:
                failed = True
                break
            for obj in scene.objs:
                if ctx.timed_out():
                    return results
                opts, covered = set(), {}
                for action in actions:
                    writes = _writes(scene, obj, action)
                    if P._consistent(g, target, obj, writes):
                        opts.add(action)
                        covered[action] = sum(g[r][c] != v for (r, c), v in writes.items())
                if not opts:
                    failed = True
                    break
                rows.append((scene.features[id(obj)], opts, covered))
            if failed:
                break
        if failed:
            continue
        for keyset in keys:
            if ctx.timed_out():
                return results
            groups, counts, coverage = {}, Counter(), {}
            for feats, opts, covered in rows:
                key = tuple(feats[k] for k in keyset)
                counts[key] += 1
                groups[key] = groups[key] & opts if key in groups else set(opts)
                cov = coverage.setdefault(key, Counter())
                cov.update(covered)
            if not groups or any(not opts for opts in groups.values()):
                continue
            if keyset and (len(groups) > min(6, max(2, len(rows)//2)) or min(counts.values()) < 2):
                continue
            ranked = {key: sorted(opts, key=lambda a: (-coverage[key][a], _cost(a), repr(a)))[:8]
                      for key, opts in groups.items()}
            for table in _tables(ranked):
                if ctx.timed_out():
                    return results
                if not any(len(a) >= 2 and a[1] in _SELECTORS for a in table.values()):
                    continue
                if keyset and len(set(table.values())) == 1:
                    continue
                rule = Rule(seg, bg, keyset, table, scenes)
                if any(rule(a) != b for a, b in ctx.train):
                    continue
                sig = tuple(rule(g) for g in ctx.test_inputs)
                if any(g is None for g in sig) or sig in seen:
                    continue
                seen.add(sig)
                name = "relproc[%s/%s/%s/%s]" % (seg, bg, "+".join(keyset) or "all",
                                                  repr(sorted(table.items())))
                cost = 3.2 + .7 * len(keyset) + .3 * len(table) + .6 * sum(map(_cost, set(table.values())))
                results.append(Hyp(name, rule, cost, SOLVER))
                break
            if len(results) >= 16:
                return results
    return results
