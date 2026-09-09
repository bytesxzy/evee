/* Port of engine/solvers/assemble.py -- outputs assembled from the objects
 * rather than edited in the grid.
 *
 * select answers "which one object is it" and partition combines panels cut by
 * separator lines. Neither covers the family where the answer is built out of
 * several objects that were never laid out on a grid: stack the shapes in size
 * order, overlay them, or report one cell per object.
 */

(function () {
  var _h = mkHyp("select");
  var _SEGS = ["c4", "c8", "m4", "m8", "color"];

  var _ORDER_NAMES = ["size", "size_asc", "pos", "col", "color", "holes", "bbox"];
  var _ORDERS = {
    size: function (a, b) { return (b.size() - a.size()) || (a.r0 - b.r0) || (a.c0 - b.c0); },
    size_asc: function (a, b) { return (a.size() - b.size()) || (a.r0 - b.r0) || (a.c0 - b.c0); },
    pos: function (a, b) { return (a.r0 - b.r0) || (a.c0 - b.c0); },
    col: function (a, b) { return (a.c0 - b.c0) || (a.r0 - b.r0); },
    color: function (a, b) { return (a.color - b.color) || (a.r0 - b.r0) || (a.c0 - b.c0); },
    holes: function (a, b) { return (b.holes_count() - a.holes_count()) || (a.r0 - b.r0) || (a.c0 - b.c0); },
    bbox: function (a, b) { return (b.bbox_area() - a.bbox_area()) || (a.r0 - b.r0) || (a.c0 - b.c0); }
  };

  function _objs(g, seg, bg, cap) {
    if (cap === undefined) cap = 40;
    var b = G.bgOr(g, bg), objs = O.segment(g, seg, b);
    if (!objs.length || objs.length > cap) return [null, b];
    return [objs, b];
  }

  /* Stack every object's patch, all normalised to a shared size. */
  function _overlay(objs, bg, mode, order) {
    var h = objs[0].height(), w = objs[0].width(), i;
    for (i = 1; i < objs.length; i++) if (objs[i].height() !== h || objs[i].width() !== w) return null;
    if (h * w > 900) return null;
    var seq = objs.slice();
    seq.sort(_ORDERS[order]);
    var out = G.constGrid(h, w, bg), r, c, p, cnt, vals;
    if (mode === "first") {
      for (i = 0; i < seq.length; i++) {
        p = seq[i].patch();
        for (r = 0; r < h; r++) for (c = 0; c < w; c++)
          if (p[r][c] !== null && out[r][c] === bg) out[r][c] = p[r][c];
      }
    } else if (mode === "last") {
      for (i = 0; i < seq.length; i++) {
        p = seq[i].patch();
        for (r = 0; r < h; r++) for (c = 0; c < w; c++) if (p[r][c] !== null) out[r][c] = p[r][c];
      }
    } else if (mode === "count") {
      cnt = G.constGrid(h, w, 0);
      for (i = 0; i < seq.length; i++) {
        p = seq[i].patch();
        for (r = 0; r < h; r++) for (c = 0; c < w; c++) if (p[r][c] !== null) cnt[r][c] += 1;
      }
      var n = seq.length, p0 = seq[0].patch();
      for (r = 0; r < h; r++) for (c = 0; c < w; c++)
        if (cnt[r][c] === n) out[r][c] = p0[r][c] !== null ? p0[r][c] : bg;
    } else if (mode === "odd") {
      cnt = G.constGrid(h, w, 0);
      vals = G.constGrid(h, w, bg);
      for (i = 0; i < seq.length; i++) {
        p = seq[i].patch();
        for (r = 0; r < h; r++) for (c = 0; c < w; c++)
          if (p[r][c] !== null) { cnt[r][c] += 1; vals[r][c] = p[r][c]; }
      }
      for (r = 0; r < h; r++) for (c = 0; c < w; c++) if (cnt[r][c] % 2 === 1) out[r][c] = vals[r][c];
    } else return null;
    return out;
  }

  function _renderObj(o, bg, render) {
    var r, c, row, out;
    if (render === "patch") return o.filled(bg);
    if (render === "mask") {
      var m = o.mask();
      out = [];
      for (r = 0; r < m.length; r++) {
        row = new Array(m[r].length);
        for (c = 0; c < m[r].length; c++) row[c] = m[r][c] ? o.color : bg;
        out.push(row);
      }
      return out;
    }
    return G.constGrid(o.height(), o.width(), o.color);
  }

  function _stack(objs, bg, order, axis, dedup, render) {
    var seq = objs.slice();
    seq.sort(_ORDERS[order]);
    var parts = [], seen = new Set(), i, p, k;
    for (i = 0; i < seq.length; i++) {
      p = _renderObj(seq[i], bg, render);
      if (dedup) {
        k = G.gkey(p);
        if (seen.has(k)) continue;
        seen.add(k);
      }
      parts.push(p);
    }
    if (parts.length < 2) return null;
    var out = parts[0];
    if (axis === "v") {
      for (i = 1; i < parts.length; i++) if (parts[i][0].length !== parts[0][0].length) return null;
      for (i = 1; i < parts.length; i++) { out = G.vconcat(out, parts[i]); if (out === null) return null; }
    } else {
      for (i = 1; i < parts.length; i++) if (parts[i].length !== parts[0].length) return null;
      for (i = 1; i < parts.length; i++) { out = G.hconcat(out, parts[i]); if (out === null) return null; }
    }
    return (out !== null && G.area(out) <= 2500) ? out : null;
  }

  function _summary(objs, bg, order, shape) {
    var seq = objs.slice();
    seq.sort(_ORDERS[order]);
    var n = seq.length, i;
    if (n < 2 || n > 40) return null;
    var cols = [];
    for (i = 0; i < n; i++) cols.push(seq[i].color);
    if (shape === "row") return [cols];
    if (shape === "col") {
      var out = [];
      for (i = 0; i < n; i++) out.push([cols[i]]);
      return out;
    }
    if (shape === "square") {
      var k = Math.round(Math.sqrt(n));
      if (k * k !== n) return null;
      var g = [], r;
      for (r = 0; r < k; r++) g.push(cols.slice(r * k, (r + 1) * k));
      return g;
    }
    return null;
  }

  /* Objects laid out on a lattice -> one cell each, keeping their layout. */
  function _gridOfObjects(objs, bg) {
    var rs = new Set(), cs = new Set(), i;
    for (i = 0; i < objs.length; i++) { rs.add(objs[i].r0); cs.add(objs[i].c0); }
    var rows = Array.from(rs).sort(function (a, b) { return a - b; });
    var cols = Array.from(cs).sort(function (a, b) { return a - b; });
    if (rows.length * cols.length !== objs.length || rows.length > 12 || cols.length > 12) return null;
    var ri = new Map(), ci = new Map();
    for (i = 0; i < rows.length; i++) ri.set(rows[i], i);
    for (i = 0; i < cols.length; i++) ci.set(cols[i], i);
    var out = G.constGrid(rows.length, cols.length, bg);
    for (i = 0; i < objs.length; i++) out[ri.get(objs[i].r0)][ci.get(objs[i].c0)] = objs[i].color;
    return out;
  }

  function _nth(objs, bg, order, idx, render) {
    var seq = objs.slice();
    seq.sort(_ORDERS[order]);
    if (!(idx >= -seq.length && idx < seq.length)) return null;
    var o = seq[idx < 0 ? seq.length + idx : idx];
    return _renderObj(o, bg, render);
  }

  function _rule(seg, bg, fn) {
    return function (g) {
      var got = _objs(g, seg, bg);
      if (got[0] === null) return null;
      try { return fn(got[0], got[1]); } catch (e) { return null; }
    };
  }

  function generate(ctx) {
    if (ctx.same_shape()) return [];
    var res = [], seen = new Set();

    function offer(name, fn, cost) {
      if (res.length >= 60) return;
      /* this family is newer and less corroborated than ``select``; where both
         explain the demonstrations, the established one should win */
      var hp = _h(name, fn, cost + 0.7);
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

    var bgs = ctx.bg_varies() ? [ctx.bg(), null] : [ctx.bg()];
    var bi, s, oi, order, m, ax, dd, rd, sh, ix;
    var modes = ["first", "last", "count", "odd"], axes = ["v", "h"];
    var dedups = [false, true], renders = ["patch", "mask", "solid"];
    var shapes = ["row", "col", "square"], idxs = [0, 1, 2, -1, -2];
    for (bi = 0; bi < bgs.length; bi++) {
      for (s = 0; s < _SEGS.length; s++) {
        if (ctx.timed_out() || res.length >= 60) return res;
        for (oi = 0; oi < _ORDER_NAMES.length; oi++) {
          order = _ORDER_NAMES[oi];
          for (m = 0; m < modes.length; m++)
            offer("ov[" + _SEGS[s] + "/" + bgs[bi] + "/" + modes[m] + "/" + order + "]",
                  _rule(_SEGS[s], bgs[bi], (function (md, od) {
                    return function (o, b) { return _overlay(o, b, md, od); };
                  })(modes[m], order)), 3.0);
          if (ctx.timed_out()) return res;
        }
        for (oi = 0; oi < _ORDER_NAMES.length; oi++) {
          order = _ORDER_NAMES[oi];
          for (ax = 0; ax < 2; ax++) for (dd = 0; dd < 2; dd++) for (rd = 0; rd < renders.length; rd++)
            offer("st[" + _SEGS[s] + "/" + order + "/" + axes[ax] + (dedups[dd] ? "d" : "") + "/" + renders[rd] + "]",
                  _rule(_SEGS[s], bgs[bi], (function (od, a2, d2, q2) {
                    return function (o, b) { return _stack(o, b, od, a2, d2, q2); };
                  })(order, axes[ax], dedups[dd], renders[rd])), 3.4);
          if (ctx.timed_out()) return res;
        }
        for (oi = 0; oi < _ORDER_NAMES.length; oi++) {
          order = _ORDER_NAMES[oi];
          for (sh = 0; sh < shapes.length; sh++)
            offer("sum[" + _SEGS[s] + "/" + order + "/" + shapes[sh] + "]",
                  _rule(_SEGS[s], bgs[bi], (function (od, s2) {
                    return function (o, b) { return _summary(o, b, od, s2); };
                  })(order, shapes[sh])), 3.2);
          for (ix = 0; ix < idxs.length; ix++) for (rd = 0; rd < renders.length; rd++)
            offer("nth[" + _SEGS[s] + "/" + order + "/" + sgn(idxs[ix]).replace("+", "") + "/" + renders[rd] + "]",
                  _rule(_SEGS[s], bgs[bi], (function (od, i2, q2) {
                    return function (o, b) { return _nth(o, b, od, i2, q2); };
                  })(order, idxs[ix], renders[rd])), 3.0 + 0.2 * Math.abs(idxs[ix]));
        }
        offer("lattice[" + _SEGS[s] + "]", _rule(_SEGS[s], bgs[bi], _gridOfObjects), 3.0);
      }
    }
    return res;
  }

  defSolver("assemble", "select", generate);
})();
