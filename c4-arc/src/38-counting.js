/* Port of engine/solvers/counting.py -- the answer is a number, expressed as
 * a grid.
 *
 * For each of a small set of counters, n is read off every training input and
 * one shape law plus one colour law must explain every training output. A law
 * that contradicts a single pair is discarded; nothing is scored on partial
 * agreement. A counter that is constant across the training inputs proves
 * nothing, so such fits carry a heavy cost penalty.
 *
 * Faithfulness note: the Python original indexes ``G.palette(b)[0]`` on a set
 * once a solid shape law matches, which raises TypeError and aborts the whole
 * module for that task. The portfolio catches it and records status "error",
 * so the solid-block family never actually contributes and the repeat and bar
 * families are skipped for every remaining counter. That behaviour is part of
 * the measured engine, so it is reproduced here rather than silently fixed.
 */

(function () {
  var MAX_COUNT = 30;

  function _h(name, fn, cost) { return new Hyp("count:" + name, fn, cost, "select"); }

  function _nObjects(mode) {
    return function (g, bg) {
      try { return O.segment(g, mode, bg).length; } catch (e) { return null; }
    };
  }
  function _nColors(g, bg) { return G.csSize(G.csDiff(G.palette(g), 1 << bg)); }
  function _nCells(g, bg) {
    var n = 0, r, c;
    for (r = 0; r < g.length; r++) for (c = 0; c < g[r].length; c++) if (g[r][c] !== bg) n++;
    return n;
  }
  function _nHoles(g, bg) {
    try { return G.holes(g, bg, false).length; } catch (e) { return null; }
  }
  function _maxObjectSize(mode) {
    return function (g, bg) {
      var objs;
      try { objs = O.segment(g, mode, bg); } catch (e) { return null; }
      if (!objs.length) return null;
      var m = objs[0].size(), i;
      for (i = 1; i < objs.length; i++) if (objs[i].size() > m) m = objs[i].size();
      return m;
    };
  }
  function _nDistinctShapes(mode) {
    return function (g, bg) {
      var objs;
      try { objs = O.segment(g, mode, bg); } catch (e) { return null; }
      var s = new Set(), i;
      for (i = 0; i < objs.length; i++) s.add(objs[i].norm_key());
      return s.size ? s.size : null;
    };
  }
  function _nLargestColor(g, bg) {
    var hist = G.histogram(g), best = null, c;
    for (c = 0; c < G.NCOLORS; c++) if (c !== bg && hist[c] > 0) {
      if (best === null || hist[c] > best) best = hist[c];
    }
    return best;
  }
  function _nMostCommonObjectColor(mode) {
    return function (g, bg) {
      var objs;
      try { objs = O.segment(g, mode, bg); } catch (e) { return null; }
      if (!objs.length) return null;
      var cnt = new Map(), order = [], i, k;
      for (i = 0; i < objs.length; i++) {
        k = objs[i].color;
        if (!cnt.has(k)) { cnt.set(k, 0); order.push(k); }
        cnt.set(k, cnt.get(k) + 1);
      }
      var best = cnt.get(order[0]);
      for (i = 1; i < order.length; i++) if (cnt.get(order[i]) > best) best = cnt.get(order[i]);
      return best;
    };
  }
  function _nRowsNonuniform(g, bg) {
    var n = 0, r, c, uni;
    for (r = 0; r < g.length; r++) {
      uni = true;
      for (c = 1; c < g[r].length; c++) if (g[r][c] !== g[r][0]) { uni = false; break; }
      if (!uni) n++;
    }
    return n;
  }

  var COUNTERS = [];
  (function () {
    var m, ms1 = ["c4", "c8", "m4", "m8", "color"], ms2 = ["c4", "c8"];
    for (m = 0; m < ms1.length; m++) COUNTERS.push(["obj_" + ms1[m], _nObjects(ms1[m])]);
    for (m = 0; m < ms2.length; m++) COUNTERS.push(["shapes_" + ms2[m], _nDistinctShapes(ms2[m])]);
    for (m = 0; m < ms2.length; m++) COUNTERS.push(["maxsize_" + ms2[m], _maxObjectSize(ms2[m])]);
    for (m = 0; m < ms2.length; m++) COUNTERS.push(["objcolor_" + ms2[m], _nMostCommonObjectColor(ms2[m])]);
    COUNTERS.push(["colors", _nColors]);
    COUNTERS.push(["cells", _nCells]);
    COUNTERS.push(["holes", _nHoles]);
    COUNTERS.push(["maxcolor", _nLargestColor]);
    COUNTERS.push(["rows", _nRowsNonuniform]);
  })();

  /* max/min over (count, colour) pairs: ties go to the higher colour. */
  function _cMostCommon(g, bg) {
    var hist = G.histogram(g), bc = null, bv = null, c;
    for (c = 0; c < G.NCOLORS; c++) if (c !== bg && hist[c] > 0)
      if (bv === null || hist[c] > bv || (hist[c] === bv && c > bc)) { bv = hist[c]; bc = c; }
    return bc;
  }
  function _cLeastCommon(g, bg) {
    var hist = G.histogram(g), bc = null, bv = null, c;
    for (c = 0; c < G.NCOLORS; c++) if (c !== bg && hist[c] > 0)
      if (bv === null || hist[c] < bv || (hist[c] === bv && c < bc)) { bv = hist[c]; bc = c; }
    return bc;
  }
  function _cLargestObject(mode) {
    return function (g, bg) {
      var objs;
      try { objs = O.segment(g, mode, bg); } catch (e) { return null; }
      if (!objs.length) return null;
      var best = objs[0], i;
      for (i = 1; i < objs.length; i++)
        if (objs[i].size() > best.size() ||
            (objs[i].size() === best.size() && (-objs[i].r0 > -best.r0 ||
             (-objs[i].r0 === -best.r0 && -objs[i].c0 > -best.c0)))) best = objs[i];
      return best.color;
    };
  }

  var COLOR_LAWS = [["most", _cMostCommon], ["least", _cLeastCommon],
                    ["bg", function (g, bg) { return bg; }],
                    ["largest_c4", _cLargestObject("c4")],
                    ["largest_c8", _cLargestObject("c8")]];

  var SHAPE_LAWS = [
    ["nxn", function (n, h, w) { return [n, n]; }],
    ["1xn", function (n, h, w) { return [1, n]; }],
    ["nx1", function (n, h, w) { return [n, 1]; }],
    ["nxw", function (n, h, w) { return [n, w]; }],
    ["hxn", function (n, h, w) { return [h, n]; }]
  ];

  function _solid(h, w, c) {
    if (h <= 0 || w <= 0 || h > MAX_COUNT * 2 || w > MAX_COUNT * 2) return null;
    return G.constGrid(h, w, c);
  }

  function _countsFor(ctx, counter, bg) {
    var out = [], ins = ctx.inputs(), i, n;
    for (i = 0; i < ins.length; i++) {
      n = counter(ins[i], bg);
      if (n === null || n === undefined || (n | 0) !== n || n <= 0 || n > MAX_COUNT) return null;
      out.push(n);
    }
    return out;
  }

  function _fitSolid(ctx, bg, cname, counter, counts, varies) {
    var hyps = [], s, ok, t, a, b, want;
    for (s = 0; s < SHAPE_LAWS.length; s++) {
      ok = true;
      for (t = 0; t < ctx.train.length; t++) {
        a = ctx.train[t][0]; b = ctx.train[t][1];
        try { want = SHAPE_LAWS[s][1](counts[t], a.length, a[0].length); } catch (e) { ok = false; break; }
        if (b.length !== want[0] || b[0].length !== want[1] || G.csSize(G.palette(b)) !== 1) { ok = false; break; }
      }
      if (!ok) continue;
      /* Reproduces the original's set-subscript TypeError; see the note above. */
      throw new TypeError("'set' object is not subscriptable");
    }
    return hyps;
  }

  function _fitRepeat(ctx, bg, cname, counter, counts, varies) {
    var modes = [
      ["tile_h", function (g, n) { return G.tile(g, 1, n); }],
      ["tile_v", function (g, n) { return G.tile(g, n, 1); }],
      ["tile_hv", function (g, n) { return G.tile(g, n, n); }],
      ["scale", function (g, n) { return G.upscale(g, n, n); }],
      ["scale_h", function (g, n) { return G.upscale(g, 1, n); }],
      ["scale_v", function (g, n) { return G.upscale(g, n, 1); }]
    ];
    var hyps = [], m, ok, t, got;
    for (m = 0; m < modes.length; m++) {
      ok = true;
      for (t = 0; t < ctx.train.length; t++) {
        try { got = modes[m][1](ctx.train[t][0], counts[t]); } catch (e) { ok = false; break; }
        if (got === null || !G.gEq(got, ctx.train[t][1])) { ok = false; break; }
      }
      if (!ok) continue;
      hyps.push(_h(cname + "." + modes[m][0], (function (fn) {
        return function (g) {
          var n = counter(g, bg);
          if (n === null || n === undefined || n <= 0 || n > 12) return null;
          if (g.length * n > 60 || g[0].length * n > 60) return null;
          try { return fn(g, n); } catch (e) { return null; }
        };
      })(modes[m][1]), 4.5 + (varies ? 0.0 : 6.0)));
    }
    return hyps;
  }

  /* "Show the count as a filled bar": the output shape is constant across the
     task and the number of coloured cells is the count. */
  function _fitBar(ctx, bg, cname, counter, counts, varies) {
    var shape = ctx.const_out_shape();
    if (!shape) return [];
    var oh = shape[0], ow = shape[1];
    if (oh * ow > 400) return [];
    var fills = [
      ["rows", function (n, h, w) { var o = [], r, c; for (r = 0; r < Math.min(n, h); r++) for (c = 0; c < w; c++) o.push([r, c]); return o; }],
      ["cols", function (n, h, w) { var o = [], r, c; for (c = 0; c < Math.min(n, w); c++) for (r = 0; r < h; r++) o.push([r, c]); return o; }],
      ["rows_up", function (n, h, w) { var o = [], r, c; for (r = 0; r < Math.min(n, h); r++) for (c = 0; c < w; c++) o.push([h - 1 - r, c]); return o; }],
      ["cols_right", function (n, h, w) { var o = [], r, c; for (c = 0; c < Math.min(n, w); c++) for (r = 0; r < h; r++) o.push([r, w - 1 - c]); return o; }],
      ["cells", function (n, h, w) { var o = [], i; for (i = 0; i < Math.min(n, h * w); i++) o.push([Math.floor(i / w), i % w]); return o; }]
    ];
    var hyps = [], f, L, laws = [["bgconst", null]].concat(COLOR_LAWS);
    for (f = 0; f < fills.length; f++) {
      for (L = 0; L < laws.length; L++) {
        var ok = true, fgConst = new Set(), bgConst = new Set(), t, cells, cellSet, fg, bgc, r, c, want;
        for (t = 0; t < ctx.train.length; t++) {
          cells = fills[f][1](counts[t], oh, ow);
          cellSet = new Set();
          var i;
          for (i = 0; i < cells.length; i++) cellSet.add(cells[i][0] * 64 + cells[i][1]);
          fg = new Set(); bgc = new Set();
          var b = ctx.train[t][1];
          cellSet.forEach(function (p) { fg.add(b[p >> 6][p & 63]); });
          for (r = 0; r < oh; r++) for (c = 0; c < ow; c++)
            if (!cellSet.has(r * 64 + c)) bgc.add(b[r][c]);
          if (fg.size !== 1 || bgc.size > 1) { ok = false; break; }
          want = fg.values().next().value;
          if (laws[L][1] !== null && laws[L][1](ctx.train[t][0], bg) !== want) { ok = false; break; }
          fgConst.add(want);
          bgc.forEach(function (v) { bgConst.add(v); });
        }
        if (!ok) continue;
        if (laws[L][1] === null && fgConst.size !== 1) continue;
        if (bgConst.size > 1) continue;
        var fillBg = bgConst.size ? bgConst.values().next().value : bg;
        var fixed = laws[L][1] === null ? fgConst.values().next().value : null;
        hyps.push(_h(cname + ".bar_" + fills[f][0] + "." + laws[L][0], (function (fill, law, fx, fb) {
          return function (g) {
            var n = counter(g, bg);
            if (n === null || n === undefined || n <= 0 || n > MAX_COUNT) return null;
            var col = law === null ? fx : law(g, bg);
            if (col === null || col === undefined) return null;
            var out = G.constGrid(oh, ow, fb), cs = fill(n, oh, ow), i;
            for (i = 0; i < cs.length; i++)
              if (cs[i][0] >= 0 && cs[i][0] < oh && cs[i][1] >= 0 && cs[i][1] < ow)
                out[cs[i][0]][cs[i][1]] = col;
            return out;
          };
        })(fills[f][1], laws[L][1], fixed, fillBg), 5.0 + (varies ? 0.0 : 6.0)));
        break;                        /* one colour law per fill is enough */
      }
    }
    return hyps;
  }

  function generate(ctx) {
    var bg = ctx.bg(), out = [], i, counts, varies;
    for (i = 0; i < COUNTERS.length; i++) {
      if (ctx.timed_out()) break;
      counts = _countsFor(ctx, COUNTERS[i][1], bg);
      if (counts === null) continue;
      varies = new Set(counts).size > 1;
      out = out.concat(_fitSolid(ctx, bg, COUNTERS[i][0], COUNTERS[i][1], counts, varies));
      out = out.concat(_fitRepeat(ctx, bg, COUNTERS[i][0], COUNTERS[i][1], counts, varies));
      out = out.concat(_fitBar(ctx, bg, COUNTERS[i][0], COUNTERS[i][1], counts, varies));
      if (out.length > 120) break;
    }
    return out;
  }

  defSolver("counting", "select", generate, 1);
})();
