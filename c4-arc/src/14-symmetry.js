/* Port of engine/solvers/symmetry.py -- occlusion repair by symmetry-group
 * closure.
 *
 * Collect candidate cell-permutations (mirrors about every half-integer axis,
 * translations by every period, diagonals on square grids). Keep one only if
 * it never contradicts an observed cell and has enough overlap. Union-find
 * over the survivors yields the orbit partition, and because union-find closes
 * under composition it recovers the whole group without enumerating it.
 */

var SYMM = {};

(function () {
  var _h = mkHyp("symmetry");
  var MIN_SUPPORT = 0.30;
  var _MAP_CACHE = new Map();

  function _candidateMaps(h, w) {
    var key = h + "x" + w, hit = _MAP_CACHE.get(key);
    if (hit !== undefined) return hit;
    var maps = [], a, p, q;
    for (a = 1; a < 2 * w - 2; a++)
      maps.push((function (aa) { return function (r, c) { return [r, aa - c]; }; })(a));
    for (a = 1; a < 2 * h - 2; a++)
      maps.push((function (aa) { return function (r, c) { return [aa - r, c]; }; })(a));
    for (p = 1; p < h; p++)
      maps.push((function (pp) { return function (r, c) { return [r + pp, c]; }; })(p));
    for (q = 1; q < w; q++)
      maps.push((function (qq) { return function (r, c) { return [r, c + qq]; }; })(q));
    /* Diagonal translations: a grid whose colour depends on (r+c) mod k is
       invariant under these and nothing else in the list. */
    for (p = 1; p < Math.min(h, w); p++) {
      maps.push((function (pp) { return function (r, c) { return [r + pp, c + pp]; }; })(p));
      maps.push((function (pp) { return function (r, c) { return [r + pp, c - pp]; }; })(p));
    }
    if (h === w) {
      maps.push(function (r, c) { return [c, r]; });
      maps.push((function (n) { return function (r, c) { return [n - c, n - r]; }; })(h - 1));
      maps.push((function (n) { return function (r, c) { return [c, n - r]; }; })(h - 1));
    }
    if (_MAP_CACHE.size > 64) _MAP_CACHE.clear();
    _MAP_CACHE.set(key, maps);
    return maps;
  }

  var _PART_CACHE = new Map();

  function _partition(g, unknown, minSupport, minPairs) {
    var key = G.gkey(g) + "#" + unknown + "#" + minSupport + "#" + minPairs;
    var hit = _PART_CACHE.get(key);
    if (hit !== undefined) return hit;
    var res = _partitionBuild(g, unknown, minSupport, minPairs);
    if (_PART_CACHE.size > 96) _PART_CACHE.clear();
    _PART_CACHE.set(key, res);
    return res;
  }

  function _partitionBuild(g, unknown, minSupport, minPairs) {
    var h = g.length, w = g[0].length, n = h * w, flat = new Array(n), known = new Uint8Array(n);
    var r, c, i, j, nk = 0;
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) {
      flat[r * w + c] = g[r][c];
      known[r * w + c] = g[r][c] !== unknown ? 1 : 0;
      if (known[r * w + c]) nk++;
    }
    if (nk === n || nk < 3) return null;

    var maps = _candidateMaps(h, w), cands = [], m, pairs, bad, links, t, tr, tc, base;
    var floor = Math.max(minPairs, Math.floor(minSupport * nk));
    for (m = 0; m < maps.length; m++) {
      pairs = 0; bad = false; links = [];
      for (r = 0; r < h && !bad; r++) {
        base = r * w;
        for (c = 0; c < w; c++) {
          t = maps[m](r, c);
          if (t === null) continue;
          tr = t[0]; tc = t[1];
          if (!(tr >= 0 && tr < h && tc >= 0 && tc < w)) continue;
          i = base + c; j = tr * w + tc;
          if (i >= j) continue;
          links.push(i); links.push(j);
          if (known[i] && known[j]) {
            pairs++;
            if (flat[i] !== flat[j]) { bad = true; break; }
          }
        }
      }
      if (bad || pairs < floor) continue;
      cands.push([pairs, links]);
    }
    if (!cands.length) return null;
    cands.sort(function (a, b) { return b[0] - a[0]; });

    var parent = new Int32Array(n);
    for (i = 0; i < n; i++) parent[i] = i;
    function find(p, x) { while (p[x] !== x) { p[x] = p[p[x]]; x = p[x]; } return x; }

    var vals = new Map();
    for (i = 0; i < n; i++) if (known[i]) vals.set(i, flat[i]);

    var k, trial, tvals, ok, ri, rj, vi, vj, li;
    for (k = 0; k < cands.length; k++) {
      links = cands[k][1];
      trial = parent.slice();
      tvals = new Map(vals);
      ok = true;
      for (li = 0; li < links.length; li += 2) {
        i = links[li]; j = links[li + 1];
        ri = find(trial, i); rj = find(trial, j);
        if (ri === rj) continue;
        vi = tvals.has(ri) ? tvals.get(ri) : null;
        vj = tvals.has(rj) ? tvals.get(rj) : null;
        if (vi !== null && vj !== null && vi !== vj) { ok = false; break; }
        trial[rj] = ri;
        if (vi === null && vj !== null) tvals.set(ri, vj);
        tvals.delete(rj);
      }
      if (ok) { parent = trial; vals = tvals; }
    }
    for (i = 0; i < n; i++) find(parent, i);
    return { parent: parent, vals: vals, known: known, h: h, w: w };
  }

  function _repairFrom(g, unknown, strict, part) {
    if (!part) return null;
    var parent = part.parent, vals = part.vals, known = part.known;
    var h = part.h, w = part.w, n = h * w, out = G.copyGrid(g), filled = false, i, v;
    for (i = 0; i < n; i++) {
      if (!known[i]) {
        v = vals.has(parent[i]) ? vals.get(parent[i]) : null;
        if (v === null) { if (strict) return null; continue; }
        out[Math.floor(i / w)][i % w] = v;
        filled = true;
      }
    }
    if (!filled && strict) return null;
    return out;
  }

  /* Maps are adopted incrementally, best-supported first, and committed only
     if merging their orbits keeps every orbit single-valued: one coincidental
     map that happens to line up on three cells would otherwise merge two
     genuine orbits and destroy a perfect reconstruction. */
  function _repair(g, unknown, strict, minSupport, minPairs) {
    if (strict === undefined) strict = true;
    if (minSupport === undefined) minSupport = MIN_SUPPORT;
    if (minPairs === undefined) minPairs = 6;
    return _repairFrom(g, unknown, strict, _partition(g, unknown, minSupport, minPairs));
  }

  /* Complete a pattern, but only inside the region it already occupies:
     unbounded completion of a sparse motif wallpapers the canvas. */
  function _repairBounded(g, unknown, strict, minSupport, minPairs, grow) {
    if (strict === undefined) strict = false;
    if (minSupport === undefined) minSupport = 0.0;
    if (minPairs === undefined) minPairs = 2;
    if (grow === undefined) grow = 0;
    var rep = _repair(g, unknown, strict, minSupport, minPairs);
    if (rep === null) return null;
    var cells = [], r, c;
    for (r = 0; r < g.length; r++) for (c = 0; c < g[r].length; c++)
      if (g[r][c] !== unknown) cells.push(r * 64 + c);
    if (!cells.length) return null;
    var h = g.length, w = g[0].length, bb = G.bboxOf(cells);
    var r0 = Math.max(0, bb[0] - grow), c0 = Math.max(0, bb[1] - grow);
    var r1 = Math.min(h - 1, bb[2] + grow), c1 = Math.min(w - 1, bb[3] + grow);
    var out = G.copyGrid(g), changed = false;
    for (r = r0; r <= r1; r++) for (c = c0; c <= c1; c++)
      if (g[r][c] === unknown && rep[r][c] !== unknown) { out[r][c] = rep[r][c]; changed = true; }
    return changed ? out : null;
  }

  function _occludedBbox(g, unknown) {
    var cells = [], r, c;
    for (r = 0; r < g.length; r++) for (c = 0; c < g[r].length; c++)
      if (g[r][c] === unknown) cells.push(r * 64 + c);
    if (!cells.length) return null;
    return G.bboxOf(cells);
  }

  function _patch(g, unknown, strict, minSupport, minPairs) {
    var rep = _repair(g, unknown, strict, minSupport, minPairs);
    if (rep === null) return null;
    var bb = _occludedBbox(g, unknown);
    if (bb === null) return null;
    return G.subgrid(rep, bb[0], bb[1], bb[2], bb[3]);
  }

  /* Colours plausibly acting as the "hole" marker. */
  function _unknownCandidates(ctx) {
    var cnt = new Map(), order = [], t, a, b, r, c, diff, i, bb, pal;
    function bump(k) {
      if (!cnt.has(k)) { cnt.set(k, 0); order.push(k); }
      cnt.set(k, cnt.get(k) + 1);
    }
    if (ctx.same_shape()) {
      for (t = 0; t < ctx.train.length; t++) {
        a = ctx.train[t][0]; b = ctx.train[t][1];
        diff = 0;
        for (r = 0; r < a.length; r++) for (c = 0; c < a[r].length; c++)
          if (a[r][c] !== b[r][c]) diff |= 1 << a[r][c];
        if (G.csSize(diff) === 1) bump(G.csList(diff)[0]);
      }
    } else {
      for (t = 0; t < ctx.train.length; t++) {
        a = ctx.train[t][0]; b = ctx.train[t][1];
        pal = G.csList(G.palette(a));
        for (i = 0; i < pal.length; i++) {
          bb = _occludedBbox(a, pal[i]);
          if (bb && (bb[2] - bb[0] + 1) === b.length && (bb[3] - bb[1] + 1) === b[0].length)
            bump(pal[i]);
        }
      }
    }
    var got = [];
    for (i = 0; i < order.length; i++) if (cnt.get(order[i]) === ctx.train.length) got.push(order[i]);
    if (got.length) return got;
    var ranked = order.slice();
    ranked.sort(function (x, y) { return cnt.get(y) - cnt.get(x); });
    return ranked.slice(0, 3);
  }

  function generate(ctx) {
    var res = [], cands = _unknownCandidates(ctx), i, u, s, grow;
    if (!cands.length) cands = [ctx.bg()];
    for (i = 0; i < Math.min(4, cands.length); i++) {
      u = cands[i];
      var stricts = [true, false];
      for (s = 0; s < 2; s++) {
        (function (uu, st) {
          var sfx = st ? "" : "_lax";
          res.push(_h("repair#" + uu + sfx, function (g) { return _repair(g, uu, st); }, st ? 3.0 : 4.0));
          res.push(_h("patch#" + uu + sfx, function (g) { return _patch(g, uu, st); }, st ? 3.2 : 4.2));
        })(u, stricts[s]);
      }
      (function (uu) {
        res.push(_h("repair_loose#" + uu, function (g) { return _repair(g, uu, true, 0.12); }, 5.0));
        res.push(_h("patch_loose#" + uu, function (g) { return _patch(g, uu, true); }, 5.2));
      })(u);
      var order2 = [false, true];
      for (s = 0; s < 2; s++) {
        (function (uu, st) {
          res.push(_h("complete#" + uu + (st ? "" : "_lax"),
                      function (g) { return _repair(g, uu, st, 0.0, 2); }, st ? 5.5 : 6.0));
        })(u, order2[s]);
      }
      (function (uu) {
        res.push(_h("complete_patch#" + uu, function (g) { return _patch(g, uu, true, 0.0, 2); }, 6.0));
      })(u);
      for (grow = 0; grow <= 1; grow++) {
        (function (uu, gr) {
          res.push(_h("complete_box#" + uu + "+" + gr,
                      function (g) { return _repairBounded(g, uu, false, 0.0, 2, gr); }, 4.8 + 0.3 * gr));
          res.push(_h("complete_box_s#" + uu + "+" + gr,
                      function (g) { return _repairBounded(g, uu, false, 0.25, 4, gr); }, 5.0 + 0.3 * gr));
        })(u, grow);
      }
    }
    return res;
  }

  SYMM.repair = _repair;
  SYMM.repairBounded = _repairBounded;
  SYMM.patch = _patch;
  SYMM.occludedBbox = _occludedBbox;
  HOOKS.repair = _repair;
  HOOKS.repairBounded = _repairBounded;

  defSolver("symmetry", "symmetry", generate);
})();
