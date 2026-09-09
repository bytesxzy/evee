/* Port of engine/solvers/partition.py -- grid decomposition into panels and
 * rules defined over the panel set.
 *
 * Two decompositions: separator lines of one colour cutting the grid into a
 * matrix of panels, and an even split into k x m congruent panels. Given
 * panels, the rules pick one by a relational selector, combine them all
 * cellwise, or reduce each panel to a single cell.
 */

var PART = {};

(function () {
  var _h = mkHyp("partition");

  function _sepLines(g, color) {
    var h = g.length, w = g[0].length, rows = [], cols = [], r, c, ok;
    for (r = 0; r < h; r++) {
      ok = true;
      for (c = 0; c < w; c++) if (g[r][c] !== color) { ok = false; break; }
      if (ok) rows.push(r);
    }
    for (c = 0; c < w; c++) {
      ok = true;
      for (r = 0; r < h; r++) if (g[r][c] !== color) { ok = false; break; }
      if (ok) cols.push(c);
    }
    return [rows, cols];
  }

  /* Maximal index ranges not covered by ``marks``. */
  function _runs(marks, n) {
    var s = new Set(marks), out = [], cur = null, i;
    for (i = 0; i < n; i++) {
      if (s.has(i)) {
        if (cur !== null) { out.push([cur, i - 1]); cur = null; }
      } else if (cur === null) cur = i;
    }
    if (cur !== null) out.push([cur, n - 1]);
    return out;
  }

  function panelsBySeparator(g, color) {
    var h = g.length, w = g[0].length, sl = _sepLines(g, color);
    var rows = sl[0], cols = sl[1];
    if (!rows.length && !cols.length) return null;
    var rr = _runs(rows, h), cc = _runs(cols, w);
    if (!rr.length || !cc.length || (rr.length === 1 && cc.length === 1)) return null;
    if (rr.length * cc.length > 64) return null;
    var out = [], i, j, band;
    for (i = 0; i < rr.length; i++) {
      band = [];
      for (j = 0; j < cc.length; j++)
        band.push(G.subgrid(g, rr[i][0], cc[j][0], rr[i][1], cc[j][1]));
      out.push(band);
    }
    return out;
  }

  function panelsEven(g, ky, kx) {
    var h = g.length, w = g[0].length;
    if (h % ky || w % kx || (ky === 1 && kx === 1)) return null;
    var ph = h / ky, pw = w / kx, out = [], i, j, band;
    for (i = 0; i < ky; i++) {
      band = [];
      for (j = 0; j < kx; j++)
        band.push(G.subgrid(g, i * ph, j * pw, (i + 1) * ph - 1, (j + 1) * pw - 1));
      out.push(band);
    }
    return out;
  }

  /* Decompose using whichever colour rules this grid, chosen per grid: the
     separator colour is not always shared across a task's examples. */
  function panelsAuto(g) {
    var pal = G.csList(G.palette(g)), i, c, sl, mat, best = null, score;
    for (i = 0; i < pal.length; i++) {
      c = pal[i];
      sl = _sepLines(g, c);
      if (!sl[0].length && !sl[1].length) continue;
      mat = panelsBySeparator(g, c);
      if (!mat) continue;
      score = [sl[0].length + sl[1].length, -c];
      if (best === null || score[0] > best[0][0] ||
          (score[0] === best[0][0] && score[1] > best[0][1])) best = [score, mat];
    }
    return best ? best[1] : null;
  }

  function sepColorOf(g) {
    var pal = G.csList(G.palette(g)), i, c, sl, best = null, score;
    for (i = 0; i < pal.length; i++) {
      c = pal[i];
      sl = _sepLines(g, c);
      if (!sl[0].length && !sl[1].length) continue;
      if (!panelsBySeparator(g, c)) continue;
      score = [sl[0].length + sl[1].length, -c];
      if (best === null || score[0] > best[0][0] ||
          (score[0] === best[0][0] && score[1] > best[0][1])) best = [score, c];
    }
    return best ? best[1] : null;
  }

  function _flat(mat) {
    var out = [], i, j;
    if (!mat) return out;
    for (i = 0; i < mat.length; i++)
      for (j = 0; j < mat[i].length; j++) if (mat[i][j] !== null && mat[i][j] !== undefined) out.push(mat[i][j]);
    return out;
  }

  /* Colours that cut *every* training input into more than one panel. */
  function sepColorCandidates(ctx) {
    var cands = [], all = ctx.all_inputs(), pal = G.palette(ctx.train[0][0]), i, j, ok;
    for (i = 0; i < all.length; i++) pal = pal & G.palette(all[i]);
    var list = G.csList(pal);
    for (i = 0; i < list.length; i++) {
      ok = true;
      for (j = 0; j < all.length; j++) if (!panelsBySeparator(all[j], list[i])) { ok = false; break; }
      if (ok) cands.push(list[i]);
    }
    return cands;
  }

  function _decompositions(ctx) {
    var out = [], cands = sepColorCandidates(ctx), i, j, all = ctx.all_inputs(), ok;
    for (i = 0; i < cands.length; i++)
      out.push(["sep#" + cands[i], 3.0,
                (function (c) { return function (g) { return panelsBySeparator(g, c); }; })(cands[i])]);
    ok = true;
    for (i = 0; i < all.length; i++) if (!panelsAuto(all[i])) { ok = false; break; }
    if (ok) out.push(["sep_auto", 3.2, panelsAuto]);
    var shapes = [], seen = {};
    for (i = 0; i < all.length; i++) {
      var k = all[i].length + "x" + all[i][0].length;
      if (!seen[k]) { seen[k] = 1; shapes.push([all[i].length, all[i][0].length]); }
    }
    var ky, kx;
    for (ky = 1; ky <= 4; ky++) for (kx = 1; kx <= 4; kx++) {
      if (ky === 1 && kx === 1) continue;
      ok = true;
      for (j = 0; j < shapes.length; j++)
        if (shapes[j][0] % ky !== 0 || shapes[j][1] % kx !== 0) { ok = false; break; }
      if (ok) out.push(["even" + ky + "x" + kx, 3.5,
                        (function (y, x) { return function (g) { return panelsEven(g, y, x); }; })(ky, kx)]);
    }
    return out;
  }

  /* ------------------------------------------------------- panel selectors */

  var SCORE_KEYS = ["ncolors", "nnz", "nzero", "nobj", "nsym"];

  function _panelScore(p, key, bg) {
    var r, c, n = 0;
    if (key === "ncolors") return G.csSize(G.palette(p));
    if (key === "nnz") {
      for (r = 0; r < p.length; r++) for (c = 0; c < p[r].length; c++) if (p[r][c] !== bg) n++;
      return n;
    }
    if (key === "nzero") {
      for (r = 0; r < p.length; r++) for (c = 0; c < p[r].length; c++) if (p[r][c] === bg) n++;
      return n;
    }
    if (key === "nobj") return G.floodRegions(p, bg, true, true).length;
    return G.symmetries(p).length;
  }

  function _select(ps, how, bg) {
    if (!ps || !ps.length) return null;
    var i, cnt, hits, key, vals, tgt;
    if (how === "unique") {
      cnt = O.countBy(ps, function (p) { return G.gkey(p); });
      hits = [];
      for (i = 0; i < ps.length; i++) if (cnt.get(G.gkey(ps[i])) === 1) hits.push(ps[i]);
      return hits.length === 1 ? hits[0] : null;
    }
    if (how === "majority") {
      cnt = O.countBy(ps, function (p) { return G.gkey(p); });
      var bestK = null, bestN = -1;
      cnt.forEach(function (n, k) { if (n > bestN) { bestN = n; bestK = k; } });
      if (bestN <= 1) return null;
      for (i = 0; i < ps.length; i++) if (G.gkey(ps[i]) === bestK) return ps[i];
      return null;
    }
    if (how === "unique_shapewise") {
      var norm = [];
      for (i = 0; i < ps.length; i++) norm.push(G.gkey(G.dedup(ps[i])));
      cnt = new Map();
      for (i = 0; i < norm.length; i++) cnt.set(norm[i], (cnt.get(norm[i]) || 0) + 1);
      hits = [];
      for (i = 0; i < ps.length; i++) if (cnt.get(norm[i]) === 1) hits.push(ps[i]);
      return hits.length === 1 ? hits[0] : null;
    }
    var s;
    for (s = 0; s < SCORE_KEYS.length; s++) {
      key = SCORE_KEYS[s];
      if (how === "max_" + key || how === "min_" + key) {
        vals = [];
        for (i = 0; i < ps.length; i++) vals.push(_panelScore(ps[i], key, bg));
        tgt = vals[0];
        for (i = 1; i < vals.length; i++)
          if (how.indexOf("max") === 0 ? vals[i] > tgt : vals[i] < tgt) tgt = vals[i];
        hits = [];
        for (i = 0; i < ps.length; i++) if (vals[i] === tgt) hits.push(ps[i]);
        return hits.length === 1 ? hits[0] : null;
      }
    }
    return null;
  }

  var _SELECTORS = ["unique", "majority", "unique_shapewise"];
  (function () {
    var i;
    for (i = 0; i < SCORE_KEYS.length; i++) {
      _SELECTORS.push("max_" + SCORE_KEYS[i]);
      _SELECTORS.push("min_" + SCORE_KEYS[i]);
    }
  })();

  /* ------------------------------------------ cellwise panel combination */

  var LOGIC_NAMES = ["and", "or", "xor", "nand", "nor", "xnor", "majority", "exact2"];
  var LOGIC = {
    and: function (n, t) { return n === t; },
    or: function (n, t) { return n > 0; },
    xor: function (n, t) { return n === 1; },
    nand: function (n, t) { return n < t; },
    nor: function (n, t) { return n === 0; },
    xnor: function (n, t) { return n !== 1; },
    majority: function (n, t) { return 2 * n > t; },
    exact2: function (n, t) { return n === 2; }
  };

  function _combine(ps, bg, op, on, off) {
    if (!ps || ps.length < 2) return null;
    var h = ps[0].length, w = ps[0][0].length, i;
    for (i = 1; i < ps.length; i++) if (ps[i].length !== h || ps[i][0].length !== w) return null;
    var f = LOGIC[op], t = ps.length, out = [], r, c, n, row;
    for (r = 0; r < h; r++) {
      row = new Array(w);
      for (c = 0; c < w; c++) {
        n = 0;
        for (i = 0; i < t; i++) if (ps[i][r][c] !== bg) n++;
        row[c] = f(n, t) ? on : off;
      }
      out.push(row);
    }
    return out;
  }

  function _overlay(ps, bg, order) {
    if (!ps || ps.length < 2) return null;
    var h = ps[0].length, w = ps[0][0].length, i;
    for (i = 1; i < ps.length; i++) if (ps[i].length !== h || ps[i][0].length !== w) return null;
    var seq = order ? ps : ps.slice().reverse(), base = seq[0];
    for (i = 1; i < seq.length; i++) base = G.pasteMasked(base, seq[i], 0, 0, bg);
    return base;
  }

  function _reduceCells(mat, bg, mode) {
    if (!mat) return null;
    var out = [], i, j, p, hist, orow, k, nbCount, only, best, bestN;
    for (i = 0; i < mat.length; i++) {
      orow = [];
      for (j = 0; j < mat[i].length; j++) {
        p = mat[i][j];
        if (p === null || p === undefined) return null;
        hist = G.histogram(p);
        if (mode === "mode") orow.push(G.modalOf(hist));
        else if (mode === "nonbg") {
          nbCount = 0; only = -1; best = -1; bestN = -1;
          for (k = 0; k < G.NCOLORS; k++) if (k !== bg && hist[k] > 0) {
            nbCount++; only = k;
            if (hist[k] > bestN) { bestN = hist[k]; best = k; }
          }
          if (!nbCount) orow.push(bg);
          else if (nbCount === 1) orow.push(only);
          else orow.push(best);
        } else if (mode === "any") {
          var any = 0;
          for (k = 0; k < G.NCOLORS; k++) if (k !== bg && hist[k] > 0) { any = 1; break; }
          orow.push(any);
        } else return null;
      }
      out.push(orow);
    }
    return out;
  }

  function _at(mat, i, j) {
    if (!mat || i >= mat.length || j >= mat[0].length) return null;
    return mat[i][j];
  }

  function generate(ctx) {
    var res = [], bg = ctx.bg(), decs = _decompositions(ctx);
    if (!decs.length) return res;
    var colors = G.csList(G.csAdd(G.csAdd(ctx.out_palette(), bg), 0));
    var d, dname, dcost, dec, m0, nr, nc, i, j, s, op, on, order, mode;
    for (d = 0; d < Math.min(10, decs.length); d++) {
      dname = decs[d][0]; dcost = decs[d][1]; dec = decs[d][2];
      try { m0 = dec(ctx.train[0][0]); } catch (e) { continue; }
      if (!m0) continue;
      nr = m0.length; nc = m0[0].length;
      if (nr * nc <= 12) {
        for (i = 0; i < nr; i++) for (j = 0; j < nc; j++)
          res.push(_h(dname + ".at" + i + "," + j,
                      (function (dd, y, x) { return function (g) { return _at(dd(g), y, x); }; })(dec, i, j),
                      dcost + 2.0));
      }
      for (s = 0; s < _SELECTORS.length; s++)
        res.push(_h(dname + ".sel_" + _SELECTORS[s],
                    (function (dd, how) { return function (g) { return _select(_flat(dd(g)), how, bg); }; })(dec, _SELECTORS[s]),
                    dcost + 3.0));
      for (op = 0; op < LOGIC_NAMES.length; op++)
        for (i = 0; i < colors.length; i++) {
          on = colors[i];
          res.push(_h(dname + "." + LOGIC_NAMES[op] + "->" + on,
                      (function (dd, o, c) { return function (g) { return _combine(_flat(dd(g)), bg, o, c, bg); }; })(dec, LOGIC_NAMES[op], on),
                      dcost + 2.5));
        }
      var orders = [true, false];
      for (i = 0; i < 2; i++) {
        order = orders[i];
        res.push(_h(dname + ".overlay" + (order ? "1" : "0"),
                    (function (dd, o) { return function (g) { return _overlay(_flat(dd(g)), bg, o); }; })(dec, order),
                    dcost + 3.0));
      }
      var modes = ["mode", "nonbg", "any"];
      for (i = 0; i < modes.length; i++) {
        mode = modes[i];
        res.push(_h(dname + ".reduce_" + mode,
                    (function (dd, m) { return function (g) { return _reduceCells(dd(g), bg, m); }; })(dec, mode),
                    dcost + 3.5));
      }
      res.push(_h(dname + ".panel_id",
                  (function (dd) { return function (g) { return _reduceCells(dd(g), bg, "nonbg"); }; })(dec),
                  dcost + 4.5));
    }
    return res;
  }

  PART.panelsBySeparator = panelsBySeparator;
  PART.panelsEven = panelsEven;
  PART.panelsAuto = panelsAuto;
  PART.sepColorOf = sepColorOf;
  PART.sepColorCandidates = sepColorCandidates;
  PART.sepLines = _sepLines;
  PART.runs = _runs;
  PART.flat = _flat;
  PART.decompositions = _decompositions;
  HOOKS.sepColorCandidates = sepColorCandidates;

  defSolver("partition", "partition", generate);
})();
