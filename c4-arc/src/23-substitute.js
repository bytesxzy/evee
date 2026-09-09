/* Port of engine/solvers/substitute.py -- every object replaced by a learned
 * stencil.
 *
 * "Each seed grows into this motif" is neither a pixel lookup (the motif
 * extends beyond the seed) nor a per-object recolour (the shape changes). It
 * is a dictionary from shape to stencil, learned by matching each input object
 * to whatever the output drew over it, and rejected unless it compresses the
 * observations it was fitted on.
 */

(function () {
  var _h = mkHyp("objects");
  var _KEYS = ["mask", "patch", "color", "size", "dims"];

  function _keyOf(o, kind) {
    if (kind === "mask") return o.norm_key();
    if (kind === "patch") {
      var parts = [], r, row, c, vals;
      for (r = 0; r < o._patch.length; r++) {
        row = o._patch[r]; vals = [];
        for (c = 0; c < row.length; c++) vals.push(row[c] === null ? "n" : row[c]);
        parts.push(vals.join(","));
      }
      return parts.join("|");
    }
    if (kind === "color") return "c" + o.color;
    if (kind === "size") return "s" + o.size();
    return "d" + o.height() + "x" + o.width();
  }

  /* Union of output components overlapping ``o``, as [patch, dr, dc]. */
  function _imageOf(o, outObjs, outGrid) {
    var cells = new Set(), i, it, s, hit;
    for (i = 0; i < outObjs.length; i++) {
      hit = false;
      it = outObjs[i].cells.values(); s = it.next();
      while (!s.done) { if (o.cells.has(s.value)) { hit = true; break; } s = it.next(); }
      if (!hit) continue;
      it = outObjs[i].cells.values(); s = it.next();
      while (!s.done) { cells.add(s.value); s = it.next(); }
    }
    if (!cells.size) return null;
    var bb = G.bboxOf(cells), r0 = bb[0], c0 = bb[1], r1 = bb[2], c1 = bb[3];
    var h = r1 - r0 + 1, w = c1 - c0 + 1, patch = [], r, c, row;
    for (r = 0; r < h; r++) { row = new Array(w); for (c = 0; c < w; c++) row[c] = null; patch.push(row); }
    it = cells.values(); s = it.next();
    while (!s.done) {
      r = s.value >> 6; c = s.value & 63;
      patch[r - r0][c - c0] = outGrid[r][c];
      s = it.next();
    }
    return [patch, r0 - o.r0, c0 - o.c0];
  }

  function _imgKey(img) {
    var parts = [], r, c, row, vals;
    for (r = 0; r < img[0].length; r++) {
      row = img[0][r]; vals = [];
      for (c = 0; c < row.length; c++) vals.push(row[c] === null ? "n" : row[c]);
      parts.push(vals.join(","));
    }
    return parts.join("|") + "@" + img[1] + "," + img[2];
  }

  function _fit(ctx, seg, kind, bg) {
    var table = new Map(), n = 0, t, a, b, ins, outs, i, o, img, k, prev;
    for (t = 0; t < ctx.train.length; t++) {
      a = ctx.train[t][0]; b = ctx.train[t][1];
      if (a.length !== b.length || a[0].length !== b[0].length) return null;
      ins = O.segment(a, seg, bg);
      outs = O.segment(b, seg, bg);
      if (!ins.length || ins.length > 60 || outs.length > 200) return null;
      for (i = 0; i < ins.length; i++) {
        o = ins[i];
        img = _imageOf(o, outs, b);
        if (img === null) return null;
        k = _keyOf(o, kind);
        prev = table.get(k);
        if (prev === undefined) table.set(k, img);
        else if (_imgKey(prev) !== _imgKey(img)) return null;
        n += 1;
      }
    }
    if (!table.size || table.size >= n || n < 2) return null;
    return table;
  }

  function _apply(g, seg, kind, table, bg, overInput) {
    bg = G.bgOr(g, bg);
    var objs = O.segment(g, seg, bg);
    if (!objs.length || objs.length > 60) return null;
    var h = g.length, w = g[0].length;
    var out = overInput ? G.copyGrid(g) : G.constGrid(h, w, bg);
    var i, o, v, patch, dr, dc, ph, pw, r, c, gr, gc, prow;
    for (i = 0; i < objs.length; i++) {
      o = objs[i];
      v = table.get(_keyOf(o, kind));
      if (v === undefined) return null;
      patch = v[0]; dr = v[1]; dc = v[2];
      ph = patch.length; pw = patch[0].length;
      for (r = 0; r < ph; r++) {
        gr = o.r0 + dr + r;
        if (!(gr >= 0 && gr < h)) continue;
        prow = patch[r];
        for (c = 0; c < pw; c++) {
          gc = o.c0 + dc + c;
          if (gc >= 0 && gc < w && prow[c] !== null) out[gr][gc] = prow[c];
        }
      }
    }
    return out;
  }

  function _rules(ctx, bg) {
    var res = [], segs = ["c8", "m8", "c4", "cells"], s, k, t, ov;
    for (s = 0; s < segs.length; s++) {
      if (ctx.timed_out()) break;
      for (k = 0; k < _KEYS.length; k++) {
        try { t = _fit(ctx, segs[s], _KEYS[k], bg); } catch (e) { t = null; }
        if (t === null) continue;
        var overs = [true, false];
        for (ov = 0; ov < 2; ov++)
          res.push(_h("subst_" + segs[s] + "_" + _KEYS[k] + (overs[ov] ? "" : "_bg"),
                      (function (sg, kd, tb, over) {
                        return function (g) { return _apply(g, sg, kd, tb, bg, over); };
                      })(segs[s], _KEYS[k], t, overs[ov]),
                      4.0 + 0.1 * t.size));
      }
    }
    return res;
  }

  function generate(ctx) {
    if (!ctx.same_shape()) return [];
    var res = [], bgs = ctx.bg_varies() ? [ctx.bg(), null] : [ctx.bg()], i;
    for (i = 0; i < bgs.length; i++) res = res.concat(_rules(ctx, bgs[i]));
    return res;
  }

  defSolver("substitute", "objects", generate);
})();
