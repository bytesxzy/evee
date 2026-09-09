/* Port of engine/solvers/objects_map.py -- per-object edits learned as a
 * decision function over object features.
 *
 * Segment the input; read off the action the target output applied to each
 * object (keep / paint solid / delete / recolour); find a single feature (or a
 * pair) whose value determines the action consistently across every training
 * object; then apply it. Step three is deliberately low-capacity: with a dozen
 * training objects any rich feature set separates them, and only rules that
 * survive on one or two features carry information.
 */

(function () {
  var _h = mkHyp("objects");
  var _SEGS = ["c4", "c8", "m4", "m8", "color", "g2", "g2m", "rows", "cols"];
  var _FEATURE_KEYS = ["color", "size", "h", "w", "bbox", "square", "rect", "holes",
    "ncolors", "border", "size_rank", "is_largest", "is_smallest",
    "shape_freq", "shape_unique", "color_freq", "color_unique",
    "r0", "c0", "row_band", "col_band", "container", "n_contains"];
  var _SHAPE_KEY = "__shape__";

  function _actKey(act) {
    if (act === null) return "null";
    if (act[0] === "cmap") {
      var parts = [], i;
      for (i = 0; i < act[1].length; i++) parts.push(act[1][i][0] + ">" + act[1][i][1]);
      return "cmap:" + parts.join(",");
    }
    return act.join(":");
  }

  /* What did the output do to this object? null if not expressible. */
  function _readAction(inp, outp, o, bg) {
    var vals = new Set(), same = true, it = o.cells.values(), s = it.next(), r, c;
    while (!s.done) {
      r = s.value >> 6; c = s.value & 63;
      vals.add(outp[r][c]);
      if (outp[r][c] !== inp[r][c]) same = false;
      s = it.next();
    }
    if (same) return ["keep"];
    if (vals.size === 1) {
      var v = vals.values().next().value;
      return v === bg ? ["del"] : ["solid", v];
    }
    var m = new Map();
    it = o.cells.values(); s = it.next();
    while (!s.done) {
      r = s.value >> 6; c = s.value & 63;
      var a = inp[r][c], b = outp[r][c];
      if (!m.has(a)) m.set(a, b);
      else if (m.get(a) !== b) return null;
      s = it.next();
    }
    var pairs = [];
    m.forEach(function (vv, kk) { pairs.push([kk, vv]); });
    pairs.sort(function (x, y) { return x[0] - y[0]; });
    return ["cmap", pairs];
  }

  function _do(out, inp, o, act, bg) {
    var it, s, r, c, m, i;
    if (act[0] === "keep") return;
    if (act[0] === "del") {
      it = o.cells.values(); s = it.next();
      while (!s.done) { out[s.value >> 6][s.value & 63] = bg; s = it.next(); }
    } else if (act[0] === "solid") {
      it = o.cells.values(); s = it.next();
      while (!s.done) { out[s.value >> 6][s.value & 63] = act[1]; s = it.next(); }
    } else if (act[0] === "cmap") {
      m = new Map();
      for (i = 0; i < act[1].length; i++) m.set(act[1][i][0], act[1][i][1]);
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        out[r][c] = m.has(inp[r][c]) ? m.get(inp[r][c]) : inp[r][c];
        s = it.next();
      }
    } else if (act[0] === "bbox") {
      for (r = o.r0; r <= o.r1; r++) for (c = o.c0; c <= o.c1; c++) out[r][c] = act[1];
    }
  }

  function _objRule(seg, bg, keys, table, def) {
    return function (g) {
      var b = G.bgOr(g, bg), got = O.featuresOf(g, seg, b), objs = got[0], feats = got[1];
      if (!objs.length || objs.length > 120) return null;
      var out = G.copyGrid(g), i, j, k, act;
      for (i = 0; i < objs.length; i++) {
        if (keys.length === 1 && keys[0] === _SHAPE_KEY) k = objs[i].norm_key();
        else {
          var parts = [];
          for (j = 0; j < keys.length; j++) parts.push(feats[i][keys[j]]);
          k = parts.join("|");
        }
        act = table.get(k);
        if (act === undefined) act = def;
        if (act === null || act === undefined) return null;
        _do(out, g, objs[i], act, b);
      }
      return out;
    };
  }

  /* Background colours worth segmenting against: the task background finds the
     objects, a wall colour instead finds the regions those walls enclose. */
  function _bgCandidates(ctx) {
    var cands = [ctx.bg()], cnt = new Int32Array(G.NCOLORS), ins = ctx.inputs(), i, c, hist;
    for (i = 0; i < ins.length; i++) {
      hist = G.histogram(ins[i]);
      for (c = 0; c < G.NCOLORS; c++) cnt[c] += hist[c];
    }
    var order = [];
    for (c = 0; c < G.NCOLORS; c++) if (cnt[c] > 0) order.push(c);
    order.sort(function (a, b) { return cnt[b] - cnt[a]; });
    var all = ctx.all_inputs(), top = order.slice(0, 3), j, ok;
    for (i = 0; i < top.length; i++) {
      if (cands.indexOf(top[i]) >= 0) continue;
      ok = true;
      for (j = 0; j < all.length; j++) if (!G.csHas(G.palette(all[j]), top[i])) { ok = false; break; }
      if (ok) cands.push(top[i]);
    }
    /* A colour that rules the grid as a lattice is the most likely wall, and a
       thin grid of lines is a small fraction of the cells. */
    try {
      var seps = new Set();
      for (i = 0; i < all.length; i++) seps.add(PART.sepColorOf(all[i]));
      if (seps.size === 1) {
        var sc = seps.values().next().value;
        if (sc !== null && sc !== undefined && cands.indexOf(sc) < 0) cands.push(sc);
      }
    } catch (e) {}
    if (ctx.bg_varies()) cands.push(null);
    return cands.slice(0, 5);
  }

  function _learn(ctx, seg, bg, keys) {
    var table = new Map(), nobj = 0, t, a, b, got, objs, feats, i, j, act, k, prev;
    for (t = 0; t < ctx.train.length; t++) {
      a = ctx.train[t][0]; b = ctx.train[t][1];
      if (a.length !== b.length || a[0].length !== b[0].length) return null;
      got = O.featuresOf(a, seg, bg); objs = got[0]; feats = got[1];
      if (!objs.length || objs.length > 120) return null;
      for (i = 0; i < objs.length; i++) {
        act = _readAction(a, b, objs[i], G.bgOr(a, bg));
        if (act === null) return null;
        if (keys.length === 1 && keys[0] === _SHAPE_KEY) k = objs[i].norm_key();
        else {
          var parts = [];
          for (j = 0; j < keys.length; j++) parts.push(feats[i][keys[j]]);
          k = parts.join("|");
        }
        prev = table.get(k);
        if (prev === undefined) table.set(k, act);
        else if (_actKey(prev) !== _actKey(act)) return null;
        nobj += 1;
      }
    }
    if (!table.size || nobj < 2) return null;
    if (table.size * 2 > nobj) return null;      /* capacity guard */
    var acts = new Map(), order = [];
    table.forEach(function (v) {
      var ak = _actKey(v);
      if (!acts.has(ak)) { acts.set(ak, { n: 0, act: v }); order.push(ak); }
      acts.get(ak).n += 1;
    });
    var def = null;
    if (acts.size > 1) {
      var best = acts.get(order[0]);
      for (i = 1; i < order.length; i++) if (acts.get(order[i]).n > best.n) best = acts.get(order[i]);
      def = best.act;
    }
    return { fn: _objRule(seg, bg, keys, table, def), size: table.size };
  }

  var _COUNTERS = [
    ["n_obj4", function (g, bg) { return O.segment(g, "c4", bg).length; }],
    ["n_obj8", function (g, bg) { return O.segment(g, "c8", bg).length; }],
    ["n_multi8", function (g, bg) { return O.segment(g, "m8", bg).length; }],
    ["n_colors", function (g, bg) { return G.csSize(G.csDiff(G.palette(g), 1 << bg)); }],
    ["max_size", function (g, bg) {
      var objs = O.segment(g, "c8", bg), m = 0, i;
      for (i = 0; i < objs.length; i++) if (objs[i].size() > m) m = objs[i].size();
      return m;
    }],
    ["n_cells", function (g, bg) {
      var n = 0, r, c;
      for (r = 0; r < g.length; r++) for (c = 0; c < g[r].length; c++) if (g[r][c] !== bg) n++;
      return n;
    }]
  ];

  function _countGrid(g, f, shape, cmode, bg) {
    bg = G.bgOr(g, bg);
    var n = f(g, bg), c;
    if (n < 1 || n > 30) return null;
    if (cmode === "fixed") c = 5;
    else {
      var hist = G.histogram(g), items = [], k;
      for (k = 0; k < G.NCOLORS; k++) if (k !== bg && hist[k] > 0) items.push([k, hist[k]]);
      if (!items.length) return null;
      items.sort(function (a, b) { return (a[1] - b[1]) || (a[0] - b[0]); });
      c = cmode === "modal" ? items[items.length - 1][0] : items[0][0];
    }
    if (shape === "sq") return G.constGrid(n, n, c);
    if (shape === "row") return G.constGrid(1, n, c);
    return G.constGrid(n, 1, c);
  }

  function _countOutputs(ctx, bg) {
    var res = [], i, s, m, shapes = ["sq", "row", "col"], modes = ["fixed", "modal", "rarest"];
    for (i = 0; i < _COUNTERS.length; i++)
      for (s = 0; s < shapes.length; s++)
        for (m = 0; m < modes.length; m++)
          res.push(_h("count_" + _COUNTERS[i][0] + "_" + shapes[s] + "_" + modes[m],
                      (function (f, sh, cm) { return function (g) { return _countGrid(g, f, sh, cm, bg); }; })(_COUNTERS[i][1], shapes[s], modes[m]),
                      6.5));
    return res;
  }

  function _bboxRender(g, seg, mode, modeArg, bg) {
    bg = G.bgOr(g, bg);
    var objs = O.segment(g, seg, bg);
    if (!objs.length || objs.length > 120) return null;
    var out = G.copyGrid(g), i, o, r, c;
    for (i = 0; i < objs.length; i++) {
      o = objs[i];
      if (mode === "fill_own") {
        for (r = o.r0; r <= o.r1; r++) for (c = o.c0; c <= o.c1; c++) out[r][c] = o.color;
      } else if (mode === "outline_own") {
        for (r = o.r0; r <= o.r1; r++) { out[r][o.c0] = o.color; out[r][o.c1] = o.color; }
        for (c = o.c0; c <= o.c1; c++) { out[o.r0][c] = o.color; out[o.r1][c] = o.color; }
      } else if (mode === "fill_hole") {
        var m = o.mask(), hh = o.height(), ww = o.width(), seen = [], st = [], j, row;
        for (r = 0; r < hh; r++) { row = new Array(ww); for (c = 0; c < ww; c++) row[c] = false; seen.push(row); }
        for (r = 0; r < hh; r++) for (j = 0; j < 2; j++) {
          c = j ? ww - 1 : 0;
          if (!m[r][c] && !seen[r][c]) { seen[r][c] = true; st.push([r, c]); }
        }
        for (c = 0; c < ww; c++) for (j = 0; j < 2; j++) {
          r = j ? hh - 1 : 0;
          if (!m[r][c] && !seen[r][c]) { seen[r][c] = true; st.push([r, c]); }
        }
        var cur, d, nr, nc;
        while (st.length) {
          cur = st.pop();
          for (d = 0; d < G.N4.length; d++) {
            nr = cur[0] + G.N4[d][0]; nc = cur[1] + G.N4[d][1];
            if (nr >= 0 && nr < hh && nc >= 0 && nc < ww && !seen[nr][nc] && !m[nr][nc]) {
              seen[nr][nc] = true; st.push([nr, nc]);
            }
          }
        }
        for (r = 0; r < hh; r++) for (c = 0; c < ww; c++)
          if (!m[r][c] && !seen[r][c]) out[o.r0 + r][o.c0 + c] = o.color;
      } else if (mode === "halo_own") {
        for (r = o.r0; r <= o.r1; r++) for (c = o.c0; c <= o.c1; c++)
          if (!o.cells.has(r * 64 + c)) out[r][c] = o.color;
      } else if (mode === "halo") {
        for (r = o.r0; r <= o.r1; r++) for (c = o.c0; c <= o.c1; c++)
          if (!o.cells.has(r * 64 + c)) out[r][c] = modeArg;
      } else if (mode === "fill") {
        for (r = o.r0; r <= o.r1; r++) for (c = o.c0; c <= o.c1; c++) out[r][c] = modeArg;
      }
    }
    return out;
  }

  function _bboxRules(ctx, bg) {
    var res = [], segs = ["c4", "c8", "m8", "g2", "g2m"], modes = ["fill_own", "outline_own", "fill_hole", "halo_own"];
    var s, m, i, pal = G.csList(ctx.out_palette());
    for (s = 0; s < segs.length; s++) {
      for (m = 0; m < modes.length; m++)
        res.push(_h(segs[s] + "." + modes[m],
                    (function (sg, md) { return function (g) { return _bboxRender(g, sg, md, null, bg); }; })(segs[s], modes[m]),
                    5.0));
      for (i = 0; i < pal.length; i++) {
        (function (sg, c) {
          res.push(_h(sg + ".fill_bbox#" + c, function (g) { return _bboxRender(g, sg, "fill", c, bg); }, 5.5));
          res.push(_h(sg + ".halo_bbox#" + c, function (g) { return _bboxRender(g, sg, "halo", c, bg); }, 5.5));
        })(segs[s], pal[i]);
      }
    }
    return res;
  }

  function generate(ctx) {
    var res = [], bg = ctx.bg(), i, j, k;
    /* counting answers change the grid shape by construction, so they must be
       offered before the same-shape gate below */
    res = res.concat(_countOutputs(ctx, bg));
    if (!ctx.same_shape()) return res;
    var singles = [], pairs = [["size", "color"], ["color", "holes"], ["size", "holes"],
      ["h", "w"], ["color", "shape_freq"], ["size", "border"],
      ["color", "border"], ["ncolors", "size"],
      ["container", "size"], ["container", "color"]];
    for (i = 0; i < _FEATURE_KEYS.length; i++) singles.push([_FEATURE_KEYS[i]]);
    singles.push([_SHAPE_KEY]);
    var cands = _bgCandidates(ctx), bi, sbg, extra, seg, band, r;
    for (bi = 0; bi < cands.length; bi++) {
      sbg = cands[bi];
      extra = bi === 0 ? 0.0 : 1.0;
      if (sbg === null) extra = 0.5;
      for (j = 0; j < _SEGS.length; j++) {
        seg = _SEGS[j];
        if (ctx.timed_out()) break;
        /* A row or column is present whether or not anything is drawn in it,
           so a table over bands sees far more capacity per observation. */
        band = (seg === "rows" || seg === "cols") ? 1.6 : 0.0;
        for (k = 0; k < singles.length; k++) {
          r = _learn(ctx, seg, sbg, singles[k]);
          if (r !== null)
            res.push(_h(seg + "@" + sbg + ".by_" + singles[k].join("+"), r.fn,
                        3.5 + extra + band + 0.05 * r.size));
        }
        for (k = 0; k < pairs.length; k++) {
          r = _learn(ctx, seg, sbg, pairs[k]);
          if (r !== null)
            res.push(_h(seg + "@" + sbg + ".by_" + pairs[k].join("+"), r.fn,
                        5.0 + extra + band + 0.05 * r.size));
        }
      }
    }
    return res.concat(_bboxRules(ctx, bg));
  }

  defSolver("objects_map", "objects", generate);
})();
