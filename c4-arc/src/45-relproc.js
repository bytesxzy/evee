/* Port of engine/solvers/relproc.py -- low-capacity object programs whose
 * actions take another object as argument.
 *
 * Only demonstrations induce a rule. Relations and actions contain no task
 * identifiers, no coordinates learned from answers and no output stencils.
 * Local consistency prunes the action vocabulary; whole-scene verification is
 * mandatory.
 */

(function () {
  var _SELECTORS = ["near", "same", "diff", "larger", "smaller", "row", "col",
    "rowdiff", "coldiff", "aligneddiff", "inside", "contains", "shape", "samebig"];
  (function () { var c; for (c = 0; c < 10; c++) _SELECTORS.push("c" + c); })();

  var _KEYS = ["color", "size_rank", "is_largest", "is_smallest", "size",
    "h", "w", "ncolors", "border", "shape_unique", "color_unique",
    "square", "rect"];
  var _VERBS = ["paint", "fill", "box", "touch", "center", "alignrow", "aligncol",
    "mirrorh", "mirrorv", "copycenter", "copyorigin", "complete"];

  function _relation(a, b, key) {
    var row = a.r0 <= b.r1 && b.r0 <= a.r1;
    var col = a.c0 <= b.c1 && b.c0 <= a.c1;
    if (key.length === 2 && key.charAt(0) === "c" && key.charAt(1) >= "0" && key.charAt(1) <= "9")
      return b.color === parseInt(key.charAt(1), 10);
    if (key === "near") return true;
    if (key === "same") return a.color === b.color;
    if (key === "diff") return a.color !== b.color;
    if (key === "larger") return b.size() > a.size();
    if (key === "smaller") return b.size() < a.size();
    if (key === "samebig") return a.color === b.color && b.size() > a.size();
    if (key === "row" || key === "rowdiff") return row && (key === "row" || a.color !== b.color);
    if (key === "col" || key === "coldiff") return col && (key === "col" || a.color !== b.color);
    if (key === "aligneddiff") return (row || col) && a.color !== b.color;
    if (key === "inside")
      return b.r0 <= a.r0 && a.r0 <= a.r1 && a.r1 <= b.r1 && b.c0 <= a.c0 && a.c0 <= a.c1 && a.c1 <= b.c1;
    if (key === "contains") return _relation(b, a, "inside");
    if (key === "shape") return a.norm_key() === b.norm_key();
    return false;
  }

  function Scene(g, seg, bg) {
    this.g = g;
    this.bg = G.bgOr(g, bg);
    this.objs = O.segment(g, seg, this.bg);
    this.features = new Map();
    this.peers = new Map();
    this.writes = new Map();
    this.write_cells = 0;
    this.ids = new Map();
    if (!(this.objs.length >= 2 && this.objs.length <= 35)) return;
    var i, j, k;
    for (i = 0; i < this.objs.length; i++) this.ids.set(this.objs[i], i);
    var shared = O.sharedStats(this.objs, g);
    for (i = 0; i < this.objs.length; i++) {
      var a = this.objs[i];
      var f = O.objectFeatures(a, this.objs, g, shared, i);
      var peers = {}, distances = new Map();
      for (j = 0; j < this.objs.length; j++)
        if (this.objs[j] !== a) distances.set(j, OBJPROC.dist(a, this.objs[j]));
      for (k = 0; k < _SELECTORS.length; k++) {
        var key = _SELECTORS[k], candidates = [], nearest = null;
        for (j = 0; j < this.objs.length; j++)
          if (this.objs[j] !== a && _relation(a, this.objs[j], key)) candidates.push(j);
        for (j = 0; j < candidates.length; j++)
          if (nearest === null || distances.get(candidates[j]) < nearest) nearest = distances.get(candidates[j]);
        var hits = [];
        for (j = 0; j < candidates.length; j++)
          if (distances.get(candidates[j]) === nearest) hits.push(this.objs[candidates[j]]);
        peers[key] = hits;
        f["has_" + key] = hits.length > 0;
        var m = 0;
        for (j = 0; j < hits.length; j++) m |= 1 << hits[j].color;
        f["color_" + key] = (hits.length && G.csSize(m) === 1) ? hits[0].color : -1;
        f["touch_" + key] = nearest === 1;
      }
      this.features.set(a, f);
      this.peers.set(a, peers);
    }
  }
  Scene.prototype.valid = function () { return this.features.size > 0; };

  function _peer(scene, obj, selector, colorOnly) {
    var hits = scene.peers.get(obj)[selector];
    if (hits.length === 1) return hits[0];
    if (colorOnly && hits.length) {
      var m = 0, i;
      for (i = 0; i < hits.length; i++) m |= 1 << hits[i].color;
      if (G.csSize(m) === 1) return hits[0];
    }
    return null;
  }

  var _sceneCacheKey = function (g, seg, bg) { return G.gkey(g) + "#" + seg + "#" + bg; };

  function _scene(g, seg, bg, cache) {
    var key = _sceneCacheKey(g, seg, bg);
    if (!cache.has(key)) {
      var result = new Scene(g, seg, bg);
      if (cache.size < 64) cache.set(key, result);
      return result;
    }
    return cache.get(key);
  }

  function _actKey(action) {
    var parts = [], i;
    for (i = 0; i < action.length; i++) parts.push(String(action[i]));
    return parts.join(":");
  }

  function _writes(scene, a, action) {
    var key;
    if (action[0] === "keep" || action[0] === "del" || action[0] === "solid")
      key = scene.ids.get(a) + "|" + _actKey(action);
    else {
      var peer = _peer(scene, a, action[1],
                       action[0] === "paint" || action[0] === "fill" || action[0] === "box");
      key = scene.ids.get(a) + "|" + action[0] + "|" + (peer ? scene.ids.get(peer) : "n") +
            "|" + action.slice(2).join(",");
    }
    if (scene.writes.has(key)) return scene.writes.get(key);
    var result = _writesUncached(scene, a, action);
    var size = result ? result.size : 0;
    if (scene.writes.size < 2048 && scene.write_cells + size <= 4000) {
      scene.writes.set(key, result);
      scene.write_cells += size;
    }
    return result;
  }

  function _fillMap(cells, v) {
    var m = new Map(), i;
    if (cells instanceof Set) {
      var it = cells.values(), s = it.next();
      while (!s.done) { m.set(s.value, v); s = it.next(); }
      return m;
    }
    for (i = 0; i < cells.length; i++) m.set(cells[i], v);
    return m;
  }

  function _writesUncached(scene, a, action) {
    var g = scene.g, bg = scene.bg, kind = action[0], h = g.length, w = g[0].length;
    if (kind === "keep") return new Map();
    if (kind === "del") return _fillMap(a.cells, bg);
    if (kind === "solid") return _fillMap(a.cells, action[1]);
    var b = _peer(scene, a, action[1], kind === "paint" || kind === "fill" || kind === "box");
    if (b === null) return kind === "complete" ? new Map() : null;
    var out, it, s, r, c, rr, cc, dr, dc, v;
    if (kind === "complete") return _completeFromPeer(scene, a, b) || new Map();
    if (kind === "paint") return _fillMap(a.cells, b.color);
    if (kind === "fill") return _fillMap(OBJPROC.interior(a, g, bg), b.color);
    if (kind === "box") {
      out = new Map();
      for (r = a.r0; r <= a.r1; r++) for (c = a.c0; c <= a.c1; c++)
        if (g[r][c] === bg) out.set(r * 64 + c, b.color);
      return out;
    }
    if (kind === "copycenter" || kind === "copyorigin") {
      if (kind === "copycenter") {
        dr = a.r0 + a.r1 - b.r0 - b.r1; dc = a.c0 + a.c1 - b.c0 - b.c1;
        if (dr % 2 || dc % 2) return null;
        dr = dr / 2; dc = dc / 2;
      } else { dr = a.r0 - b.r0; dc = a.c0 - b.c0; }
      out = new Map();
      it = b.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        rr = r + dr; cc = c + dc;
        if (!(rr >= 0 && rr < h && cc >= 0 && cc < w)) return null;
        v = g[r][c] === b.color ? a.color : g[r][c];
        if (g[rr][cc] !== bg && !a.cells.has(rr * 64 + cc) && g[rr][cc] !== v) return null;
        out.set(rr * 64 + cc, v);
        s = it.next();
      }
      return out;
    }
    if (kind.indexOf("connect") === 0) {
      if (a.size() !== 1 || b.size() !== 1) return null;
      var ar = a.r0, ac = a.c0, br = b.r0, bc = b.c0;
      var mode = action[2], color = action[3];
      color = color === "self" ? a.color : (color === "peer" ? b.color : color);
      var cells = [], n, i, dy, dx;
      if (mode === "straight") {
        dy = br - ar; dx = bc - ac;
        if (dy && dx && Math.abs(dy) !== Math.abs(dx)) return null;
        n = Math.max(Math.abs(dy), Math.abs(dx));
        if (!n) return null;
        for (i = 1; i < n; i++) cells.push([ar + i * (dy / n), ac + i * (dx / n)]);
      } else {
        var corner = mode === "rowfirst" ? [ar, bc] : [br, ac];
        var segs = [[[ar, ac], corner], [corner, [br, bc]]], sgi;
        for (sgi = 0; sgi < 2; sgi++) {
          var p0 = segs[sgi][0], p1 = segs[sgi][1];
          n = Math.max(Math.abs(p1[0] - p0[0]), Math.abs(p1[1] - p0[1]));
          if (n) for (i = 0; i <= n; i++) cells.push([p0[0] + i * ((p1[0] - p0[0]) / n), p0[1] + i * ((p1[1] - p0[1]) / n)]);
        }
        cells = cells.filter(function (rc) {
          return !a.cells.has(rc[0] * 64 + rc[1]) && !b.cells.has(rc[0] * 64 + rc[1]);
        });
      }
      for (i = 0; i < cells.length; i++) {
        v = g[cells[i][0]][cells[i][1]];
        if (v !== bg && v !== color) return null;
      }
      out = new Map();
      for (i = 0; i < cells.length; i++) out.set(cells[i][0] * 64 + cells[i][1], color);
      return out;
    }
    if (kind === "center") {
      dr = b.r0 + b.r1 - a.r0 - a.r1; dc = b.c0 + b.c1 - a.c0 - a.c1;
      if (dr % 2 || dc % 2) return null;
      dr = dr / 2; dc = dc / 2;
    } else if (kind === "alignrow") {
      dr = Math.floor((b.r0 + b.r1 - a.r0 - a.r1) / 2); dc = 0;
    } else if (kind === "aligncol") {
      dr = 0; dc = Math.floor((b.c0 + b.c1 - a.c0 - a.c1) / 2);
    } else if (kind === "mirrorh") {
      dr = 0; dc = b.c0 + b.c1 - a.c0 - a.c1;
    } else if (kind === "mirrorv") {
      dr = b.r0 + b.r1 - a.r0 - a.r1; dc = 0;
    } else if (kind === "touch") {
      var rows = a.r0 <= b.r1 && b.r0 <= a.r1, cols = a.c0 <= b.c1 && b.c0 <= a.c1;
      if (rows && !cols) { dr = 0; dc = a.c1 < b.c0 ? b.c0 - a.c1 - 1 : b.c1 - a.c0 + 1; }
      else if (cols && !rows) { dr = a.r1 < b.r0 ? b.r0 - a.r1 - 1 : b.r1 - a.r0 + 1; dc = 0; }
      else return null;
    } else return null;
    if (!dr && !dc) return new Map();
    out = _fillMap(a.cells, bg);
    it = a.cells.values(); s = it.next();
    while (!s.done) {
      r = s.value >> 6; c = s.value & 63;
      rr = r + dr; cc = c + dc;
      if (!(rr >= 0 && rr < h && cc >= 0 && cc < w)) return null;
      if (!a.cells.has(rr * 64 + cc) && g[rr][cc] !== bg) return null;
      out.set(rr * 64 + cc, g[r][c]);
      s = it.next();
    }
    return out;
  }

  /* Transfer a peer's structure through an inferred colour-role renaming.
     Distinct geometric descriptions producing the same writes are one
     interpretation; different resulting grids are ambiguity and cause
     abstention. */
  function _completeFromPeer(scene, receiver, donor) {
    var g = scene.g, bg = scene.bg, h = g.length, w = g[0].length;
    var src = donor.colors(), dst = receiver.colors();
    var old = G.csDiff(src, dst), nw = G.csDiff(dst, src);
    if (nw && (G.csSize(old) !== 1 || G.csSize(nw) !== 1)) return null;
    var rename = new Map();
    if (nw) rename.set(G.csList(old)[0], G.csList(nw)[0]);
    var sourceCounts = new Int32Array(G.NCOLORS), targetCounts = new Int32Array(G.NCOLORS);
    var it = donor.cells.values(), s = it.next();
    while (!s.done) { sourceCounts[g[s.value >> 6][s.value & 63]]++; s = it.next(); }
    it = receiver.cells.values(); s = it.next();
    while (!s.done) { targetCounts[g[s.value >> 6][s.value & 63]]++; s = it.next(); }
    var dp = donor.patch(), patch = [], r, c, row;
    for (r = 0; r < dp.length; r++) {
      row = new Array(dp[r].length);
      for (c = 0; c < dp[r].length; c++)
        row[c] = (dp[r][c] !== null && rename.has(dp[r][c])) ? rename.get(dp[r][c]) : dp[r][c];
      patch.push(row);
    }
    var variants = new Set(), results = new Map();
    var base = patch, scale, shared = src & dst, ok, k, candidate, key, places, i, writes;
    for (scale = 1; scale <= Math.min(Math.floor(h / base.length), Math.floor(w / base[0].length)); scale++) {
      if (nw) {
        ok = true;
        for (k = 0; k < G.NCOLORS; k++)
          if (G.csHas(shared, k) && sourceCounts[k] * scale * scale !== targetCounts[k]) { ok = false; break; }
        if (!ok) continue;
      }
      if (donor.size() * scale * scale <= receiver.size()) continue;
      if (base.length * scale > h || base[0].length * scale > w) continue;
      candidate = ANALOGY.scalePatch(base, scale);
      key = "";
      for (r = 0; r < candidate.length; r++) {
        for (c = 0; c < candidate[r].length; c++) key += (candidate[r][c] === null ? "n" : candidate[r][c]) + ",";
        key += "|";
      }
      if (variants.has(key)) continue;
      variants.add(key);
      places = ANALOGY.placements(g, candidate, receiver, bg, h, w);
      for (i = 0; i < places.length; i++) {
        var pairs = [];
        for (r = 0; r < candidate.length; r++) for (c = 0; c < candidate[r].length; c++)
          if (candidate[r][c] !== null)
            pairs.push([(r + places[i][0]) * 64 + (c + places[i][1]), candidate[r][c]]);
        pairs.sort(function (x, y) { return x[0] - y[0]; });
        var wk = pairs.map(function (p) { return p[0] + ">" + p[1]; }).join(",");
        if (!results.has(wk)) results.set(wk, pairs);
        if (results.size > 1) return null;
      }
    }
    if (!results.size) return null;
    var chosen = results.values().next().value, m = new Map();
    for (i = 0; i < chosen.length; i++) m.set(chosen[i][0], chosen[i][1]);
    return m;
  }

  function _cost(action) {
    if (action[0] === "keep") return 0.0;
    if (action[0] === "del") return 0.7;
    if (action[0] === "solid") return 1.2;
    return 1.8 + (action[0].indexOf("copy") === 0 ? 0.4 : 0.0);
  }

  function _rule(seg, bg, keys, table, scenes) {
    return function (g) {
      var scene = _scene(g, seg, bg, scenes);
      if (!scene.valid()) return null;
      var draws = new Map(), erase = new Set(), i, j, f, key, action, writes, bad = false;
      for (i = 0; i < scene.objs.length; i++) {
        f = scene.features.get(scene.objs[i]);
        var parts = [];
        for (j = 0; j < keys.length; j++) parts.push(f[keys[j]]);
        key = parts.join("|");
        action = table.get(key);
        if (action === undefined) return null;
        writes = _writes(scene, scene.objs[i], action);
        if (writes === null || writes === undefined) return null;
        writes.forEach(function (v, rc) {
          if (v === scene.bg) erase.add(rc);
          else if (draws.has(rc) && draws.get(rc) !== v) bad = true;
          else draws.set(rc, v);
        });
        if (bad) return null;
      }
      var out = G.copyGrid(g);
      erase.forEach(function (rc) { out[rc >> 6][rc & 63] = scene.bg; });
      draws.forEach(function (v, rc) { out[rc >> 6][rc & 63] = v; });
      return out;
    };
  }

  /* Cheapest assemblies first, expanded lazily from the per-group shortlists. */
  function _tables(groups, order, cap) {
    if (cap === undefined) cap = 40;
    var opts = [], i;
    for (i = 0; i < order.length; i++) opts.push(groups.get(order[i]));
    var start = new Array(order.length).fill(0);
    var heap = [[0.0, start]], seen = new Set([start.join(",")]), out = [];
    function popMin() {
      var bi = 0, j;
      for (j = 1; j < heap.length; j++) if (heap[j][0] < heap[bi][0]) bi = j;
      return heap.splice(bi, 1)[0];
    }
    var n;
    for (n = 0; n < cap; n++) {
      if (!heap.length) return out;
      var cur = popMin(), ix = cur[1], t = new Map(), j;
      for (j = 0; j < order.length; j++) t.set(order[j], opts[j][ix[j]]);
      out.push(t);
      for (j = 0; j < order.length; j++) {
        if (ix[j] + 1 >= opts[j].length) continue;
        var nxt = ix.slice();
        nxt[j] = ix[j] + 1;
        var nk = nxt.join(",");
        if (seen.has(nk)) continue;
        seen.add(nk);
        var cost = 0, m;
        for (m = 0; m < order.length; m++) cost += _cost(opts[m][nxt[m]]);
        heap.push([cost, nxt]);
      }
    }
    return out;
  }

  function generate(ctx) {
    if (!ctx.same_shape()) return [];
    var colors = G.csList(G.csDiff(ctx.out_palette(), 1 << ctx.bg()));
    var actions = [["keep"], ["del"]], i, j, k;
    for (i = 0; i < colors.length; i++) actions.push(["solid", colors[i]]);
    for (i = 0; i < _SELECTORS.length; i++)
      for (j = 0; j < _VERBS.length; j++) actions.push([_VERBS[j], _SELECTORS[i]]);
    var csels = ["same", "diff", "near"], cmodes = ["straight", "rowfirst", "colfirst"];
    var ccols = ["self", "peer"].concat(colors);
    for (i = 0; i < csels.length; i++) for (j = 0; j < cmodes.length; j++) for (k = 0; k < ccols.length; k++)
      actions.push(["connect", csels[i], cmodes[j], ccols[k]]);

    var keys = [[]];
    for (i = 0; i < _KEYS.length; i++) keys.push([_KEYS[i]]);
    var prefixes = ["has_", "color_", "touch_"];
    for (i = 0; i < _SELECTORS.length; i++) for (j = 0; j < prefixes.length; j++)
      keys.push([prefixes[j] + _SELECTORS[i]]);
    var owns = ["is_largest", "is_smallest", "color_unique"];
    for (i = 0; i < _SELECTORS.length; i++) for (j = 0; j < owns.length; j++)
      keys.push([owns[j], "has_" + _SELECTORS[i]]);

    var results = [], seen = new Set(), scenes = new Map(), layoutsSeen = new Set();
    var backgrounds = ctx.bg_varies() ? [ctx.bg(), null] : [ctx.bg()];
    var segs = ["c4", "c8", "m4", "m8", "color"], bi, si, bg, seg;
    var all = ctx.all_inputs();
    for (bi = 0; bi < backgrounds.length; bi++) for (si = 0; si < segs.length; si++) {
      bg = backgrounds[bi]; seg = segs[si];
      if (ctx.timed_out()) break;
      var layoutParts = [];
      for (i = 0; i < all.length; i++) {
        var b2 = G.bgOr(all[i], bg), objs2 = O.segment(all[i], seg, b2), cellKeys = [];
        for (j = 0; j < objs2.length; j++) {
          var arr = [], it = objs2[j].cells.values(), s = it.next();
          while (!s.done) { arr.push(s.value); s = it.next(); }
          arr.sort(function (x, y) { return x - y; });
          cellKeys.push(arr.join(","));
        }
        cellKeys.sort();
        layoutParts.push(b2 + ":" + cellKeys.join(";"));
      }
      var layout = layoutParts.join("||");
      if (layoutsSeen.has(layout)) continue;
      layoutsSeen.add(layout);

      var rows = [], failed = false, t;
      for (t = 0; t < ctx.train.length && !failed; t++) {
        var g = ctx.train[t][0], target = ctx.train[t][1];
        var scene = _scene(g, seg, bg, scenes);
        if (!scene.valid()) { failed = true; break; }
        for (i = 0; i < scene.objs.length; i++) {
          if (ctx.timed_out()) return results;
          var obj = scene.objs[i], opts = new Map(), covered = new Map();
          for (j = 0; j < actions.length; j++) {
            var writes = _writes(scene, obj, actions[j]);
            if (OBJPROC.consistent(g, target, obj, writes)) {
              var ak = _actKey(actions[j]);
              opts.set(ak, actions[j]);
              var n = 0;
              writes.forEach(function (v, rc) { if (g[rc >> 6][rc & 63] !== v) n++; });
              covered.set(ak, n);
            }
          }
          if (!opts.size) { failed = true; break; }
          rows.push([scene.features.get(obj), opts, covered]);
        }
      }
      if (failed) continue;

      var ki;
      for (ki = 0; ki < keys.length; ki++) {
        if (ctx.timed_out()) return results;
        var keyset = keys[ki], groups = new Map(), order = [], counts = new Map(), coverage = new Map();
        var bad = false;
        for (i = 0; i < rows.length; i++) {
          var feats = rows[i][0], parts = [];
          for (j = 0; j < keyset.length; j++) parts.push(feats[keyset[j]]);
          var key = parts.join("|");
          counts.set(key, (counts.get(key) || 0) + 1);
          if (!groups.has(key)) { groups.set(key, new Map(rows[i][1])); order.push(key); }
          else {
            var cur = groups.get(key), inter = new Map();
            cur.forEach(function (act, ak) { if (rows[i][1].has(ak)) inter.set(ak, act); });
            groups.set(key, inter);
          }
          if (!coverage.has(key)) coverage.set(key, new Map());
          var cov = coverage.get(key);
          rows[i][2].forEach(function (v, ak) { cov.set(ak, (cov.get(ak) || 0) + v); });
        }
        if (!groups.size) continue;
        for (i = 0; i < order.length; i++) if (!groups.get(order[i]).size) { bad = true; break; }
        if (bad) continue;
        if (keyset.length) {
          var minCount = Infinity;
          counts.forEach(function (v) { if (v < minCount) minCount = v; });
          if (groups.size > Math.min(6, Math.max(2, Math.floor(rows.length / 2))) || minCount < 2) continue;
        }
        var ranked = new Map();
        for (i = 0; i < order.length; i++) {
          var arr = [], cov2 = coverage.get(order[i]);
          groups.get(order[i]).forEach(function (act, ak) { arr.push([act, ak]); });
          arr.sort(function (x, y) {
            return ((cov2.get(y[1]) || 0) - (cov2.get(x[1]) || 0)) ||
                   (_cost(x[0]) - _cost(y[0])) || cmpStr(x[1], y[1]);
          });
          ranked.set(order[i], arr.slice(0, 8).map(function (x) { return x[0]; }));
        }
        var tables = _tables(ranked, order), ti;
        for (ti = 0; ti < tables.length; ti++) {
          if (ctx.timed_out()) return results;
          var table = tables[ti], anyRel = false, distinct = new Set();
          table.forEach(function (act) {
            if (act.length >= 2 && _SELECTORS.indexOf(act[1]) >= 0) anyRel = true;
            distinct.add(_actKey(act));
          });
          if (!anyRel) continue;
          if (keyset.length && distinct.size === 1) continue;
          var rule = _rule(seg, bg, keyset, table, scenes), okAll = true, p;
          for (t = 0; t < ctx.train.length; t++) {
            p = rule(ctx.train[t][0]);
            if (p === null || !G.gEq(p, ctx.train[t][1])) { okAll = false; break; }
          }
          if (!okAll) continue;
          var sigParts = [], anyNull = false;
          for (i = 0; i < ctx.test_inputs.length; i++) {
            p = rule(ctx.test_inputs[i]);
            if (p === null || p === undefined) { anyNull = true; break; }
            sigParts.push(G.gkey(p));
          }
          if (anyNull) continue;
          var sig = sigParts.join("~");
          if (seen.has(sig)) continue;
          seen.add(sig);
          var entries = [];
          table.forEach(function (act, kk) { entries.push(kk + "=" + _actKey(act)); });
          entries.sort();
          var name = "relproc[" + seg + "/" + bg + "/" + (keyset.join("+") || "all") + "/" + entries.join(",") + "]";
          var sum = 0;
          distinct.forEach(function () {});
          var seenActs = new Map();
          table.forEach(function (act) { seenActs.set(_actKey(act), act); });
          seenActs.forEach(function (act) { sum += _cost(act); });
          var cost = 3.2 + 0.7 * keyset.length + 0.3 * table.size + 0.6 * sum;
          results.push(new Hyp(name, rule, cost, "objects"));
          break;
        }
        if (results.length >= 16) return results;
      }
    }
    return results;
  }

  defSolver("relproc", "objects", generate);
})();
