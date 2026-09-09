/* Port of engine/solvers/tiling.py -- periodicity.
 *
 * Distinct from symmetry: here the structure is translational and the answer
 * is often smaller than the input (the motif) or a continuation of it.
 */

var TILING = {};

(function () {
  var _h = mkHyp("tiling");

  /* Smallest (p, q) such that g is p-periodic in rows and q in cols. */
  function _periods(g, ignore) {
    var h = g.length, w = g[0].length, p = h, q = w, cand, r, c, a, b, ok, row;
    var useIg = (ignore !== null && ignore !== undefined);
    for (cand = 1; cand < h; cand++) {
      ok = true;
      for (r = 0; r < h - cand && ok; r++) {
        for (c = 0; c < w; c++) {
          a = g[r][c]; b = g[r + cand][c];
          if (useIg && (a === ignore || b === ignore)) continue;
          if (a !== b) { ok = false; break; }
        }
      }
      if (ok) { p = cand; break; }
    }
    for (cand = 1; cand < w; cand++) {
      ok = true;
      for (r = 0; r < h && ok; r++) {
        row = g[r];
        for (c = 0; c < w - cand; c++) {
          a = row[c]; b = row[c + cand];
          if (useIg && (a === ignore || b === ignore)) continue;
          if (a !== b) { ok = false; break; }
        }
      }
      if (ok) { q = cand; break; }
    }
    return [p, q];
  }

  function _motif(g, ignore) {
    var pq = _periods(g, ignore), p = pq[0], q = pq[1];
    if (p === g.length && q === g[0].length) return null;
    var tile = [], r, c, row, v, t;
    for (r = 0; r < p; r++) { row = new Array(q); for (c = 0; c < q; c++) row[c] = null; tile.push(row); }
    var h = g.length, w = g[0].length;
    var useIg = (ignore !== null && ignore !== undefined);
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) {
      v = g[r][c];
      if (useIg && v === ignore) continue;
      t = tile[r % p][c % q];
      if (t === null) tile[r % p][c % q] = v;
      else if (t !== v) return null;
    }
    for (r = 0; r < p; r++) for (c = 0; c < q; c++) if (tile[r][c] === null) return null;
    return tile;
  }

  function _fillPeriodic(g, ignore) {
    var m = _motif(g, ignore);
    if (m === null) return null;
    var p = m.length, q = m[0].length, h = g.length, w = g[0].length, out = [], r, c, row;
    for (r = 0; r < h; r++) {
      row = new Array(w);
      for (c = 0; c < w; c++) row[c] = g[r][c] !== ignore ? g[r][c] : m[r % p][c % q];
      out.push(row);
    }
    return out;
  }

  /* Fill unknown cells where colour is a function of (r +/- c) mod k.
     Diagonally striped patterns have no row or column period, so the periodic
     filler cannot see them at all. */
  function _fillDiag(g, ignore, sign) {
    var h = g.length, w = g[0].length, k, table, ok, r, c, v, key, prev, row, out;
    for (k = 2; k <= 12; k++) {
      table = new Map(); ok = true;
      for (r = 0; r < h && ok; r++) {
        row = g[r];
        for (c = 0; c < w; c++) {
          v = row[c];
          if (v === ignore) continue;
          key = (((r + sign * c) % k) + k) % k;
          prev = table.get(key);
          if (prev === undefined) table.set(key, v);
          else if (prev !== v) { ok = false; break; }
        }
      }
      if (!ok || table.size < k) continue;
      out = [];
      for (r = 0; r < h; r++) {
        row = new Array(w);
        for (c = 0; c < w; c++)
          row[c] = g[r][c] !== ignore ? g[r][c] : table.get((((r + sign * c) % k) + k) % k);
        out.push(row);
      }
      return out;
    }
    return null;
  }

  function _extend(g, oh, ow) {
    var h = g.length, w = g[0].length, out = [], r, c, row;
    if (oh > 60 || ow > 60 || oh < 1 || ow < 1) return null;
    for (r = 0; r < oh; r++) {
      row = new Array(ow);
      for (c = 0; c < ow; c++) row[c] = g[r % h][c % w];
      out.push(row);
    }
    return out;
  }

  function _motifOf(g) { return g === null ? null : _motif(g); }

  function generate(ctx) {
    var res = [], bg = ctx.bg(), pal = G.csList(ctx.in_palette()), i, s;
    res.push(_h("motif", function (g) { return _motif(g); }, 4.0));
    for (i = 0; i < pal.length; i++) {
      (function (c) {
        res.push(_h("motif_ig#" + c, function (g) { return _motif(g, c); }, 4.5));
        res.push(_h("fillper#" + c, function (g) { return _fillPeriodic(g, c); }, 4.0));
        var signs = [1, -1], j;
        for (j = 0; j < 2; j++)
          res.push(_h("filldiag" + sgn(signs[j]) + "#" + c,
                      (function (sn) { return function (g) { return _fillDiag(g, c, sn); }; })(signs[j]), 4.2));
      })(pal[i]);
    }
    var cs = ctx.const_out_shape();
    if (cs) {
      res.push(_h("extend" + cs[0] + "x" + cs[1], function (g) { return _extend(g, cs[0], cs[1]); }, 4.0));
      for (i = 1; i < G.DIHEDRAL.length; i++)
        res.push(_h("extend_" + G.DIHEDRAL[i][0],
                    (function (f) { return function (g) { return _extend(f(g), cs[0], cs[1]); }; })(G.DIHEDRAL[i][1]), 5.0));
    }
    var r = ctx.shape_ratio();
    if (r) {
      var ky = r[0], kx = r[1];
      res.push(_h("extend_ratio" + ky + "x" + kx,
                  function (g) { return _extend(g, g.length * ky, g[0].length * kx); }, 3.5));
    }
    res.push(_h("crop_motif", function (g) { return _motifOf(G.cropToContent(g, bg)); }, 5.0));
    res.push(_h("skeleton", G.dedup, 4.5));
    return res;
  }

  TILING.periods = _periods;
  TILING.motif = _motif;
  TILING.fillPeriodic = _fillPeriodic;
  TILING.extend = _extend;
  HOOKS.motif = function (g) { return _motif(g); };

  defSolver("tiling", "tiling", generate);
})();
