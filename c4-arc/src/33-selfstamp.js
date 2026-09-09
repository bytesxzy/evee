/* Port of engine/solvers/selfstamp.py -- the grid used as its own brush.
 *
 * Gated self-stamping: out[i*h+r][j*w+c] = g[r][c] wherever cell (i, j) passes
 * a test. geometry offers only the gate "is not background"; the interesting
 * tasks pick a particular colour, often named relationally because it differs
 * per example. Offset repetition lays a copy down again at a fixed step, which
 * tiling cannot express because the step is not the grid size.
 */

(function () {
  var _h = mkHyp("tiling");
  var _MAX_OUT = 1400;

  function _topNonbg(g, bg) {
    var h = G.histogram(g), best = -1, bestN = -1, k;
    for (k = 0; k < G.NCOLORS; k++) if (k !== bg && h[k] > 0 && h[k] > bestN) { bestN = h[k]; best = k; }
    return best;
  }

  function _gates(ctx) {
    var pal = G.csList(ctx.in_palette()), bg = ctx.bg(), gates = [], i;
    gates.push(["nonbg", 0.0, function (g) { return function (v) { return v !== bg; }; }]);
    gates.push(["isbg", 0.3, function (g) { return function (v) { return v === bg; }; }]);
    for (i = 0; i < pal.length; i++) {
      (function (c) {
        gates.push(["is" + c, 0.6, function (g) { return function (v) { return v === c; }; }]);
        gates.push(["not" + c, 0.9, function (g) { return function (v) { return v !== c; }; }]);
      })(pal[i]);
    }
    gates.push(["top", 0.5, function (g) { var m = G.mostCommonColor(g); return function (v) { return v === m; }; }]);
    gates.push(["nottop", 0.7, function (g) { var m = G.mostCommonColor(g); return function (v) { return v !== m; }; }]);
    gates.push(["rare", 0.5, function (g) { var m = G.leastCommonColor(g); return function (v) { return v === m; }; }]);
    gates.push(["topnb", 0.6, function (g) { var m = _topNonbg(g, bg); return function (v) { return v === m; }; }]);
    return gates;
  }

  function _stamp(g, gatefn, fill, xform) {
    var h = g.length, w = g[0].length;
    if (h * w > 40 || (h * h) * (w * w) > _MAX_OUT) return null;
    var src = xform(g);
    if (src.length !== h || src[0].length !== w) return null;
    var out = G.constGrid(h * h, w * w, fill), test = gatefn(g), i, j, r, c, orow, srow;
    for (i = 0; i < h; i++) for (j = 0; j < w; j++) {
      if (test(g[i][j])) {
        for (r = 0; r < h; r++) {
          orow = out[i * h + r]; srow = src[r];
          for (c = 0; c < w; c++) orow[j * w + c] = srow[c];
        }
      }
    }
    return out;
  }

  function _maskOnly(g, gatefn, on, off) {
    var test = gatefn(g), out = [], r, c, row;
    for (r = 0; r < g.length; r++) {
      row = new Array(g[r].length);
      for (c = 0; c < g[r].length; c++) row[c] = test(g[r][c]) ? on : off;
      out.push(row);
    }
    return out;
  }

  function _outShape(ctx, g) {
    var cs = ctx.const_out_shape();
    if (cs) return cs;
    var sr = ctx.shape_ratio();
    if (sr) return [g.length * sr[0], g[0].length * sr[1]];
    return null;
  }

  function _repeat(ctx, g, dr, dc, fill, back, xform) {
    var d = _outShape(ctx, g);
    if (d === null) return null;
    var H = d[0], W = d[1];
    if (H * W > _MAX_OUT) return null;
    var h = g.length, w = g[0].length, src = xform(g);
    if (src.length !== h || src[0].length !== w) return null;
    var out = G.constGrid(H, W, fill);
    var kFrom = back ? -back : 0, kTo = Math.max(H, W), k, r0, c0, r, c, rr, cc, srow, orow;
    for (k = kFrom; k <= kTo; k++) {
      r0 = dr * k; c0 = dc * k;
      if (r0 >= H || c0 >= W || r0 + h <= 0 || c0 + w <= 0) continue;
      for (r = 0; r < h; r++) {
        rr = r0 + r;
        if (!(rr >= 0 && rr < H)) continue;
        srow = src[r]; orow = out[rr];
        for (c = 0; c < w; c++) {
          cc = c0 + c;
          if (cc >= 0 && cc < W && srow[c] !== fill) orow[cc] = srow[c];
        }
      }
    }
    return out;
  }

  function generate(ctx) {
    var res = [], seen = new Set();

    function offer(name, fn, cost) {
      if (res.length >= 24) return;
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

    var xforms = [["id", 0.0, function (g) { return g; }]], i, j, k, x;
    for (i = 1; i < G.DIHEDRAL.length; i++) xforms.push([G.DIHEDRAL[i][0], 0.5, G.DIHEDRAL[i][1]]);
    var fills = G.csList(G.csAdd(G.csAdd(G.csDiff(ctx.out_palette(), ctx.in_palette()), ctx.bg()), 0)).slice(0, 3);

    var ratio = ctx.shape_ratio(), inv = ctx.inv_shape_ratio(), squareLaw = true, a, b;
    for (i = 0; i < ctx.train.length; i++) {
      a = ctx.train[i][0]; b = ctx.train[i][1];
      if (!(b.length === a.length * a.length && b[0].length === a[0].length * a[0].length)) { squareLaw = false; break; }
    }
    var gates = _gates(ctx), gi;
    if (squareLaw) {
      for (gi = 0; gi < gates.length; gi++) {
        if (ctx.timed_out()) return res;
        for (x = 0; x < xforms.length; x++) {
          for (k = 0; k < fills.length; k++)
            offer("stamp[" + gates[gi][0] + "/" + xforms[x][0] + "/" + fills[k] + "]",
                  (function (gt, f, xf) { return function (g) { return _stamp(g, gt, f, xf); }; })(gates[gi][2], fills[k], xforms[x][2]),
                  3.2 + gates[gi][1] + xforms[x][1]);
          if (res.length >= 24) return res;
        }
      }
    }

    if (ctx.same_shape() || inv) {
      var pal = G.csList(ctx.out_palette()).slice(0, 4);
      for (gi = 0; gi < gates.length; gi++) {
        if (ctx.timed_out()) return res;
        for (i = 0; i < pal.length; i++) for (j = 0; j < pal.length; j++) {
          if (pal[i] === pal[j]) continue;
          offer("gate[" + gates[gi][0] + "->" + pal[i] + "/" + pal[j] + "]",
                (function (gt, on, off) { return function (g) { return _maskOnly(g, gt, on, off); }; })(gates[gi][2], pal[i], pal[j]),
                3.4 + gates[gi][1]);
        }
      }
    }

    if (ratio || ctx.const_out_shape()) {
      var all = ctx.all_inputs(), hs = 0, ws = 0;
      for (i = 0; i < all.length; i++) {
        if (all[i].length > hs) hs = all[i].length;
        if (all[i][0].length > ws) ws = all[i][0].length;
      }
      if (hs <= 14 && ws <= 14) {
        var dr, dc, backs = [0, 1];
        for (dr = -hs; dr <= hs; dr++) {
          if (ctx.timed_out()) return res;
          for (dc = -ws; dc <= ws; dc++) {
            if (dr === 0 && dc === 0) continue;
            for (k = 0; k < Math.min(2, fills.length); k++)
              for (j = 0; j < 2; j++)
                offer("rep[" + sgn(dr) + sgn(dc) + "/" + fills[k] + (backs[j] ? "b" : "") + "]",
                      (function (y, xx, f, bk) {
                        return function (g) { return _repeat(ctx, g, y, xx, f, bk * 40, function (t) { return t; }); };
                      })(dr, dc, fills[k], backs[j]),
                      3.6 + 0.05 * (Math.abs(dr) + Math.abs(dc)));
          }
        }
      }
    }
    return res;
  }

  defSolver("selfstamp", "tiling", generate);
})();
