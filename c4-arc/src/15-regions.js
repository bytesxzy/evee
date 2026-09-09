/* Port of engine/solvers/regions.py -- region extraction (family "select").
 *
 * The window scanner is the workhorse: when the output shape is known and
 * small, every window of that shape in the input is a candidate and the task
 * reduces to choosing one -- a relational selection rather than a rule.
 */

var REGIONS = {};

(function () {
  var _h = mkHyp("select");

  function _windows(g, oh, ow, step) {
    var h = g.length, w = g[0].length, out = [], r, c, i, patch;
    if (oh > h || ow > w) return [];
    for (r = 0; r <= h - oh; r += step)
      for (c = 0; c <= w - ow; c += step) {
        patch = [];
        for (i = r; i < r + oh; i++) patch.push(g[i].slice(c, c + ow));
        out.push([r, c, patch]);
      }
    return out;
  }

  var WIN_CRIT_NAMES = ["ncolors", "nnz", "nbg", "nsym", "distinct"];
  var WIN_CRIT = {
    ncolors: function (p, bg) { return G.csSize(G.palette(p)); },
    nnz: function (p, bg) {
      var n = 0, r, c;
      for (r = 0; r < p.length; r++) for (c = 0; c < p[r].length; c++) if (p[r][c] !== bg) n++;
      return n;
    },
    nbg: function (p, bg) {
      var n = 0, r, c;
      for (r = 0; r < p.length; r++) for (c = 0; c < p[r].length; c++) if (p[r][c] === bg) n++;
      return n;
    },
    nsym: function (p, bg) { return G.symmetries(p).length; },
    distinct: function (p, bg) {
      var s = new Set(), r;
      for (r = 0; r < p.length; r++) s.add(p[r].join(","));
      return -s.size;
    }
  };

  var _WCACHE = new Map();

  /* Windows plus every criterion value, computed once and reused: without
     this the scanner recomputes the same few hundred windows once per
     criterion, which dominated the whole engine's runtime. */
  function _winAnalysis(g, oh, ow, bg, tiled) {
    var key = G.gkey(g) + "#" + oh + "#" + ow + "#" + bg + "#" + (tiled ? 1 : 0);
    var hit = _WCACHE.get(key);
    if (hit !== undefined) return hit;
    var wins = _windows(g, oh, ow, 1), i;
    if (tiled) {
      var kept = [];
      for (i = 0; i < wins.length; i++)
        if (wins[i][0] % oh === 0 && wins[i][1] % ow === 0) kept.push(wins[i]);
      wins = kept;
    }
    var pats = [];
    for (i = 0; i < wins.length; i++) pats.push(wins[i][2]);
    var vals = {}, k;
    if (pats.length) {
      for (k = 0; k < WIN_CRIT_NAMES.length; k++) {
        if (WIN_CRIT_NAMES[k] === "nsym" && pats.length > 300) continue;
        var arr = [];
        for (i = 0; i < pats.length; i++) arr.push(WIN_CRIT[WIN_CRIT_NAMES[k]](pats[i], bg));
        vals[WIN_CRIT_NAMES[k]] = arr;
      }
    }
    var cnt = new Map();
    for (i = 0; i < pats.length; i++) {
      var gk = G.gkey(pats[i]);
      cnt.set(gk, (cnt.get(gk) || 0) + 1);
    }
    var res = [pats, vals, cnt];
    if (_WCACHE.size > 64) _WCACHE.clear();
    _WCACHE.set(key, res);
    return res;
  }

  function _windowPick(g, oh, ow, how, bg, tiled) {
    if (oh < 1 || ow < 1 || oh > g.length || ow > g[0].length) return null;
    var nWin = (g.length - oh + 1) * (g[0].length - ow + 1);
    if (nWin > 1200) return null;
    var an = _winAnalysis(g, oh, ow, bg, tiled), pats = an[0], vals = an[1], cnt = an[2], i;
    if (!pats.length) return null;
    if (how === "unique") {
      var hits = [];
      for (i = 0; i < pats.length; i++) if (cnt.get(G.gkey(pats[i])) === 1) hits.push(pats[i]);
      return hits.length === 1 ? hits[0] : null;
    }
    if (how === "modal") {
      var bestK = null, bestN = -1;
      cnt.forEach(function (n, k) { if (n > bestN) { bestN = n; bestK = k; } });
      if (bestN <= 1) return null;
      for (i = 0; i < pats.length; i++) if (G.gkey(pats[i]) === bestK) return pats[i];
      return null;
    }
    var v = vals[how.slice(4)];
    if (v === undefined) return null;
    var tgt = v[0], isMax = how.indexOf("max_") === 0;
    for (i = 1; i < v.length; i++) if (isMax ? v[i] > tgt : v[i] < tgt) tgt = v[i];
    var uniq = new Map();
    for (i = 0; i < pats.length; i++) if (v[i] === tgt) uniq.set(G.gkey(pats[i]), pats[i]);
    if (uniq.size !== 1) return null;
    var only = null;
    uniq.forEach(function (p) { only = p; });
    return only;
  }

  /* ------------------------------------------------------------- frames */

  /* Hollow single-colour rectangles, as [r0, c0, r1, c1, colour]. */
  function _frames(g, bg) {
    var objs = O.segment(g, "c8", bg), out = [], i, o, m, r, c, ok, edge;
    for (i = 0; i < objs.length; i++) {
      o = objs[i];
      if (o.height() < 3 || o.width() < 3) continue;
      m = o.mask(); ok = true;
      for (r = 0; r < o.height() && ok; r++)
        for (c = 0; c < o.width(); c++) {
          edge = (r === 0 || r === o.height() - 1 || c === 0 || c === o.width() - 1);
          if (edge && !m[r][c]) { ok = false; break; }
        }
      if (ok) out.push([o.r0, o.c0, o.r1, o.c1, o.color]);
    }
    return out;
  }

  function _byAreaDesc(fr) {
    fr.sort(function (a, b) {
      return ((b[2] - b[0]) * (b[3] - b[1])) - ((a[2] - a[0]) * (a[3] - a[1]));
    });
    return fr;
  }

  function _frameInterior(g, bg, which) {
    var fr = _frames(g, bg), f;
    if (!fr.length) return null;
    if (fr.length > 1) { _byAreaDesc(fr); f = which === "largest" ? fr[0] : fr[fr.length - 1]; }
    else f = fr[0];
    return G.subgrid(g, f[0] + 1, f[1] + 1, f[2] - 1, f[3] - 1);
  }

  function _frameContent(g, bg, which) {
    var fr = _frames(g, bg);
    if (!fr.length) return null;
    _byAreaDesc(fr);
    var f = which === "largest" ? fr[0] : fr[fr.length - 1];
    return G.subgrid(g, f[0], f[1], f[2], f[3]);
  }

  /* Bounding box of a marker colour, optionally grown or shrunk: markers
     usually delimit a structure rather than being it. */
  function _markedRect(g, bg, marker, inclusive, pr, pc) {
    if (pr === undefined) pr = 0;
    if (pc === undefined) pc = 0;
    var cells = [], r, c;
    for (r = 0; r < g.length; r++) for (c = 0; c < g[r].length; c++)
      if (g[r][c] === marker) cells.push(r * 64 + c);
    if (cells.length < 2) return null;
    var h = g.length, w = g[0].length, bb = G.bboxOf(cells);
    var r0 = bb[0], c0 = bb[1], r1 = bb[2], c1 = bb[3];
    if (!inclusive) { r0 += 1; c0 += 1; r1 -= 1; c1 -= 1; }
    r0 = Math.max(0, r0 - pr); c0 = Math.max(0, c0 - pc);
    r1 = Math.min(h - 1, r1 + pr); c1 = Math.min(w - 1, c1 + pc);
    return G.subgrid(g, r0, c0, r1, c1);
  }

  /* Crop to the components that contain a given colour: the interesting
     object is usually the whole connected structure the marker is part of. */
  function _colorComponentBox(g, bg, color, seg) {
    var objs = O.segment(g, seg, bg), cells = [], i, it, s;
    for (i = 0; i < objs.length; i++) {
      if (!G.csHas(objs[i].colors(), color)) continue;
      it = objs[i].cells.values(); s = it.next();
      while (!s.done) { cells.push(s.value); s = it.next(); }
    }
    if (!cells.length) return null;
    var bb = G.bboxOf(cells);
    return G.subgrid(g, bb[0], bb[1], bb[2], bb[3]);
  }

  function _cutColorRegion(g, bg, color) {
    var cells = [], r, c;
    for (r = 0; r < g.length; r++) for (c = 0; c < g[r].length; c++)
      if (g[r][c] !== color) cells.push(r * 64 + c);
    if (!cells.length) return null;
    var bb = G.bboxOf(cells);
    return G.subgrid(g, bb[0], bb[1], bb[2], bb[3]);
  }

  /* ------------------------------------------------------- colour answers */

  function _rank(g, bg, i) {
    var hist = G.histogram(g), items = [], k;
    for (k = 0; k < G.NCOLORS; k++) if (hist[k] > 0 && k !== bg) items.push([k, hist[k]]);
    items.sort(function (a, b) { return (a[1] - b[1]) || (a[0] - b[0]); });
    var idx = i < 0 ? items.length + i : i;
    if (idx < 0 || idx >= items.length) return null;
    return items[idx][0];
  }

  function _objColor(g, bg, biggest) {
    var objs = O.segment(g, "c8", bg);
    if (!objs.length) return null;
    var o = O.selectExtreme(objs, "size", biggest);
    return o ? o.color : null;
  }

  function _uniqColor(g, bg) {
    var objs = O.segment(g, "c8", bg);
    if (!objs.length) return null;
    var o = O.selectUniqueShape(objs);
    return o ? o.color : null;
  }

  var COLOR_PICKS = [
    ["most", function (g, bg) { return _rank(g, bg, -1); }],
    ["least", function (g, bg) { return _rank(g, bg, 0); }],
    ["second", function (g, bg) { return _rank(g, bg, -2); }],
    ["largest_obj", function (g, bg) { return _objColor(g, bg, true); }],
    ["smallest_obj", function (g, bg) { return _objColor(g, bg, false); }],
    ["unique_shape_obj", function (g, bg) { return _uniqColor(g, bg); }],
    ["center", function (g, bg) { return g[Math.floor(g.length / 2)][Math.floor(g[0].length / 2)]; }],
    ["corner", function (g, bg) { return g[0][0]; }]
  ];

  function _colorGrid(g, pick, bg, shape) {
    var c = pick(g, bg);
    if (c === null || c === undefined) return null;
    if (shape === null) return [[c]];
    return G.constGrid(shape[0], shape[1], c);
  }

  function generate(ctx) {
    var res = [], bg = ctx.bg(), cs = ctx.const_out_shape(), i, k, how;
    var crits = ["unique", "modal"];
    for (k = 0; k < WIN_CRIT_NAMES.length; k++) crits.push("max_" + WIN_CRIT_NAMES[k]);
    for (k = 0; k < WIN_CRIT_NAMES.length; k++) crits.push("min_" + WIN_CRIT_NAMES[k]);

    if (cs && cs[0] * cs[1] <= 400) {
      for (k = 0; k < crits.length; k++) {
        how = crits[k];
        var tiles = [false, true];
        for (i = 0; i < 2; i++)
          res.push(_h("win" + cs[0] + "x" + cs[1] + "." + how + (tiles[i] ? "_t" : ""),
                      (function (hw, t) { return function (g) { return _windowPick(g, cs[0], cs[1], hw, bg, t); }; })(how, tiles[i]),
                      4.0 + (tiles[i] ? 0.0 : 0.5)));
      }
    }
    var ir = ctx.inv_shape_ratio();
    if (!cs && ir && !(ir[0] === 1 && ir[1] === 1)) {
      var ky = ir[0], kx = ir[1];
      for (k = 0; k < crits.length; k++)
        res.push(_h("winr" + ky + "x" + kx + "." + crits[k],
                    (function (hw) {
                      return function (g) {
                        return _windowPick(g, Math.floor(g.length / ky), Math.floor(g[0].length / kx), hw, bg, true);
                      };
                    })(crits[k]), 4.5));
    }

    var whichs = ["largest", "smallest"];
    for (i = 0; i < 2; i++) {
      (function (wch) {
        res.push(_h("frame_in_" + wch, function (g) { return _frameInterior(g, bg, wch); }, 3.5));
        res.push(_h("frame_all_" + wch, function (g) { return _frameContent(g, bg, wch); }, 3.8));
      })(whichs[i]);
    }

    var pal = G.csList(ctx.in_palette());
    for (i = 0; i < pal.length; i++) {
      (function (c) {
        res.push(_h("mark#" + c + "_in", function (g) { return _markedRect(g, bg, c, false); }, 4.5));
        res.push(_h("mark#" + c + "_all", function (g) { return _markedRect(g, bg, c, true); }, 4.5));
        res.push(_h("notbox#" + c, function (g) { return _cutColorRegion(g, bg, c); }, 4.5));
        var pads = [[1, 0], [0, 1], [1, 1]], j;
        for (j = 0; j < pads.length; j++)
          res.push(_h("mark#" + c + "_pad" + pads[j][0] + pads[j][1],
                      (function (pr, pc) { return function (g) { return _markedRect(g, bg, c, true, pr, pc); }; })(pads[j][0], pads[j][1]),
                      4.8));
        var segs = ["m8", "m4"];
        for (j = 0; j < segs.length; j++)
          res.push(_h("compbox#" + c + "." + segs[j],
                      (function (sg) { return function (g) { return _colorComponentBox(g, bg, c, sg); }; })(segs[j]),
                      4.2));
      })(pal[i]);
    }

    var shapes = [null];
    if (cs && cs[0] * cs[1] <= 25) shapes.push(cs);
    for (k = 0; k < COLOR_PICKS.length; k++)
      for (i = 0; i < shapes.length; i++)
        res.push(_h("color_" + COLOR_PICKS[k][0] + (shapes[i] === null ? "" : "_fill"),
                    (function (p, sh) { return function (g) { return _colorGrid(g, p, bg, sh); }; })(COLOR_PICKS[k][1], shapes[i]),
                    5.0));
    return res;
  }

  REGIONS.frames = _frames;
  REGIONS.frameInterior = _frameInterior;
  REGIONS.frameContent = _frameContent;
  REGIONS.markedRect = _markedRect;
  REGIONS.windows = _windows;
  HOOKS.frameInterior = _frameInterior;
  HOOKS.frameContent = _frameContent;
  HOOKS.markedRect = _markedRect;

  defSolver("regions", "select", generate);
})();
