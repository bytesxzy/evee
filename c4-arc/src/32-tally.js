/* Port of engine/solvers/tally.py -- answers that are a count, drawn onto a
 * fixed canvas.
 *
 * counting covers the case where the size of the output is the number; here
 * the canvas is fixed and the number is expressed by how much of it is filled.
 * The same canvas serves a neighbouring family: scattered marks gathered in
 * reading order and laid into a small grid. A counter that never varies across
 * the training inputs proves nothing, so it is rejected.
 */

(function () {
  var _h = mkHyp("counting");
  var _SEGS = ["c4", "c8", "m4", "m8", "color"];

  function _topNonbg(g, bg) {
    var h = G.histogram(g), best = -1, bestN = -1, k;
    for (k = 0; k < G.NCOLORS; k++) if (k !== bg && h[k] > 0 && h[k] > bestN) { bestN = h[k]; best = k; }
    return best;
  }

  function _counters(ctx) {
    var bg = ctx.bg(), out = [], s, i, pal;
    for (s = 0; s < _SEGS.length; s++) {
      (function (seg) {
        out.push(["n" + seg, 0.4, function (g) { return O.segment(g, seg, G.bgOr(g, bg)).length; }]);
        out.push(["big" + seg, 0.8, function (g) {
          var objs = O.segment(g, seg, G.bgOr(g, bg)), m = 0, j;
          for (j = 0; j < objs.length; j++) if (objs[j].size() > m) m = objs[j].size();
          return m;
        }]);
        out.push(["shp" + seg, 0.9, function (g) {
          var objs = O.segment(g, seg, G.bgOr(g, bg)), set = new Set(), j;
          for (j = 0; j < objs.length; j++) set.add(objs[j].norm_key());
          return set.size;
        }]);
      })(_SEGS[s]);
    }
    out.push(["ncol", 0.5, function (g) { return G.csSize(G.csDiff(G.palette(g), 1 << G.bgOr(g, bg))); }]);
    out.push(["ncell", 0.6, function (g) {
      var b = G.bgOr(g, bg), n = 0, r, c;
      for (r = 0; r < g.length; r++) for (c = 0; c < g[r].length; c++) if (g[r][c] !== b) n++;
      return n;
    }]);
    pal = G.csList(ctx.in_palette()).slice(0, 8);
    for (i = 0; i < pal.length; i++)
      out.push(["cnt" + pal[i], 0.9,
                (function (c) { return function (g) { return G.countColor(g, c); }; })(pal[i])]);
    return out;
  }

  function _fillN(shape, n, on, off, order) {
    var h = shape[0], w = shape[1], cells = [], r, c;
    if (order === "col" || order === "rcol") {
      for (c = 0; c < w; c++) for (r = 0; r < h; r++) cells.push([r, c]);
      if (order === "rcol") cells.reverse();
    } else if (order === "up") {
      for (r = h - 1; r >= 0; r--) for (c = 0; c < w; c++) cells.push([r, c]);
    } else {
      for (r = 0; r < h; r++) for (c = 0; c < w; c++) cells.push([r, c]);
      if (order === "rrow") cells.reverse();
    }
    if (n < 0 || n > h * w) return null;
    var out = G.constGrid(h, w, off), i;
    for (i = 0; i < n; i++) out[cells[i][0]][cells[i][1]] = on;
    return out;
  }

  function _fillRows(shape, n, on, off, axis) {
    var h = shape[0], w = shape[1], r, c;
    if (n < 0 || n > (axis === "row" ? h : w)) return null;
    var out = G.constGrid(h, w, off);
    if (axis === "row") { for (r = 0; r < n; r++) for (c = 0; c < w; c++) out[r][c] = on; }
    else { for (r = 0; r < h; r++) for (c = 0; c < n; c++) out[r][c] = on; }
    return out;
  }

  /* Reading order for the destination; serpentine is a common ARC idiom. */
  function _place(cells, h, w, bg, snake) {
    var out = G.constGrid(h, w, bg), i, r, c;
    for (i = 0; i < cells.length; i++) {
      r = Math.floor(i / w); c = i % w;
      if (snake && r % 2) c = w - 1 - c;
      out[r][c] = cells[i];
    }
    return out;
  }

  function _gather(g, shape, bg, order, mode, snake) {
    var h = shape[0], w = shape[1], cells = [], gh = g.length, gw = g[0].length, r, c, i;
    if (order === "col") {
      for (c = 0; c < gw; c++) for (r = 0; r < gh; r++) if (g[r][c] !== bg) cells.push(g[r][c]);
    } else {
      for (r = 0; r < gh; r++) for (c = 0; c < gw; c++) if (g[r][c] !== bg) cells.push(g[r][c]);
    }
    if (mode === "uniq") {
      var seen = new Set(), keep = [];
      for (i = 0; i < cells.length; i++) if (!seen.has(cells[i])) { seen.add(cells[i]); keep.push(cells[i]); }
      cells = keep;
    } else if (mode === "freq") {
      var cnt = new Map();
      for (i = 0; i < cells.length; i++) cnt.set(cells[i], (cnt.get(cells[i]) || 0) + 1);
      var items = [];
      cnt.forEach(function (n, v) { items.push([v, n]); });
      items.sort(function (a, b) { return (b[1] - a[1]) || (a[0] - b[0]); });
      cells = items.map(function (x) { return x[0]; });
    }
    if (cells.length > h * w) return null;
    return _place(cells, h, w, bg, snake);
  }

  function generate(ctx) {
    var shape = ctx.const_out_shape();
    if (shape === null || shape[0] * shape[1] > 120) return [];
    var res = [], seen = new Set();

    function offer(name, fn, cost) {
      if (res.length >= 20) return;
      var hp = _h(name, fn, cost);
      if (!hp.fits(ctx.train)) return;
      var parts = [], i, p;
      try {
        for (i = 0; i < ctx.test_inputs.length; i++) {
          p = hp.apply(ctx.test_inputs[i]);
          parts.push(p === null ? "*" : G.gkey(p));
        }
      } catch (e) { return; }
      var sig = parts.join("~");
      if (seen.has(sig)) return;
      seen.add(sig);
      res.push(hp);
    }

    var pal = G.csList(ctx.out_palette()), i, j, k, orders = ["row", "col", "rrow", "rcol", "up"];
    if (pal.length <= 3) {
      var counters = _counters(ctx), ci, vals, ok, distinct;
      for (ci = 0; ci < counters.length; ci++) {
        if (ctx.timed_out() || res.length >= 20) break;
        vals = []; ok = true;
        var ins = ctx.inputs();
        for (i = 0; i < ins.length; i++) {
          try { vals.push(counters[ci][2](ins[i])); } catch (e) { ok = false; break; }
        }
        if (!ok) continue;
        distinct = new Set(vals);
        if (distinct.size < 2) continue;   /* a constant counter is not evidence */
        for (i = 0; i < pal.length; i++) for (j = 0; j < pal.length; j++) {
          if (pal[i] === pal[j]) continue;
          for (k = 0; k < orders.length; k++)
            offer("tally[" + counters[ci][0] + "/" + orders[k] + "/" + pal[i] + ">" + pal[j] + "]",
                  (function (f, o, a, b) { return function (g) { return _fillN(shape, f(g), a, b, o); }; })(counters[ci][2], orders[k], pal[i], pal[j]),
                  2.8 + counters[ci][1]);
          var axes = ["row", "col"], ax;
          for (ax = 0; ax < 2; ax++)
            offer("bars[" + counters[ci][0] + "/" + axes[ax] + "/" + pal[i] + ">" + pal[j] + "]",
                  (function (f, x, a, b) { return function (g) { return _fillRows(shape, f(g), a, b, x); }; })(counters[ci][2], axes[ax], pal[i], pal[j]),
                  3.0 + counters[ci][1]);
        }
      }
    }

    var bg = ctx.bg(), gorders = ["row", "col"], modes = ["all", "uniq", "freq"], snakes = [false, true];
    for (i = 0; i < gorders.length; i++) for (j = 0; j < modes.length; j++) for (k = 0; k < 2; k++)
      offer("gather[" + gorders[i] + "/" + modes[j] + (snakes[k] ? "/snake" : "") + "]",
            (function (o, m, s) { return function (g) { return _gather(g, shape, G.bgOr(g, bg), o, m, s); }; })(gorders[i], modes[j], snakes[k]),
            3.0 + (snakes[k] ? 0.3 : 0.0));
    return res;
  }

  defSolver("tally", "counting", generate);
})();
