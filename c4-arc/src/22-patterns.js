/* Port of engine/solvers/patterns.py -- find every occurrence of a learned
 * window and stamp on it.
 *
 * Some rules are stated over positions that look a certain way rather than
 * over objects. The window and the paint are both read off the demonstrations:
 * group the cells the output changed, check they share one shape, one
 * before-content and one after-content, then look for that content everywhere.
 */

(function () {
  var _h = mkHyp("sequence");

  function _diffGroups(a, b, diag) {
    var h = a.length, w = a[0].length, diff = [], seen = [], r, c, row, srow;
    for (r = 0; r < h; r++) {
      row = new Array(w); srow = new Array(w);
      for (c = 0; c < w; c++) { row[c] = a[r][c] !== b[r][c]; srow[c] = false; }
      diff.push(row); seen.push(srow);
    }
    var nb = diag ? G.N8 : G.N4, out = [], st, cells, cur, d, nr, nc;
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) {
      if (!diff[r][c] || seen[r][c]) continue;
      st = [[r, c]]; seen[r][c] = true; cells = [];
      while (st.length) {
        cur = st.pop();
        cells.push(cur[0] * 64 + cur[1]);
        for (d = 0; d < nb.length; d++) {
          nr = cur[0] + nb[d][0]; nc = cur[1] + nb[d][1];
          if (nr >= 0 && nr < h && nc >= 0 && nc < w && diff[nr][nc] && !seen[nr][nc]) {
            seen[nr][nc] = true; st.push([nr, nc]);
          }
        }
      }
      out.push(cells);
    }
    return out;
  }

  /* Adjacent occurrences merge into one larger changed region, so requiring a
     single window across every group is too strict. Take the most frequent
     window and the smallest one; the orchestrator validates them. */
  function _fit(ctx, diag) {
    var cnt = new Map(), t, a, b, groups, i, bb, wa, wb, key;
    for (t = 0; t < ctx.train.length; t++) {
      a = ctx.train[t][0]; b = ctx.train[t][1];
      if (a.length !== b.length || a[0].length !== b[0].length) return [];
      groups = _diffGroups(a, b, diag);
      if (!groups.length || groups.length > 60) return [];
      for (i = 0; i < groups.length; i++) {
        bb = G.bboxOf(groups[i]);
        if ((bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) > 25) continue;
        wa = G.subgrid(a, bb[0], bb[1], bb[2], bb[3]);
        wb = G.subgrid(b, bb[0], bb[1], bb[2], bb[3]);
        if (wa === null || wb === null || G.gEq(wa, wb)) continue;
        key = G.gkey(wa) + "=>" + G.gkey(wb);
        if (!cnt.has(key)) cnt.set(key, { n: 0, wa: wa, wb: wb });
        cnt.get(key).n += 1;
      }
    }
    if (!cnt.size) return [];
    var out = [], modal = null;
    cnt.forEach(function (v) { if (modal === null || v.n > modal.n) modal = v; });
    if (modal.n >= 2) out.push(modal);
    var smallest = null;
    cnt.forEach(function (v) {
      if (smallest === null) { smallest = v; return; }
      var da = G.area(v.wa) - G.area(smallest.wa);
      if (da < 0) { smallest = v; return; }
      if (da === 0) {
        var c1 = cmpGrid(v.wa, smallest.wa);
        if (c1 < 0 || (c1 === 0 && cmpGrid(v.wb, smallest.wb) < 0)) smallest = v;
      }
    });
    if (out.indexOf(smallest) < 0) out.push(smallest);
    return out;
  }

  function _apply(g, before, after, overlap) {
    var h = g.length, w = g[0].length, ph = before.length, pw = before[0].length;
    if (ph > h || pw > w) return null;
    var hits = [], r, c, i, j, ok;
    for (r = 0; r <= h - ph; r++) for (c = 0; c <= w - pw; c++) {
      ok = true;
      for (i = 0; i < ph && ok; i++)
        for (j = 0; j < pw; j++) if (g[r + i][c + j] !== before[i][j]) { ok = false; break; }
      if (ok) hits.push([r, c]);
    }
    if (!hits.length) return null;
    var used = new Set(), out = G.copyGrid(g), k, cells, clash;
    for (k = 0; k < hits.length; k++) {
      r = hits[k][0]; c = hits[k][1];
      cells = [];
      for (i = 0; i < ph; i++) for (j = 0; j < pw; j++) cells.push((r + i) * 64 + (c + j));
      if (!overlap) {
        clash = false;
        for (i = 0; i < cells.length; i++) if (used.has(cells[i])) { clash = true; break; }
        if (clash) continue;
      }
      for (i = 0; i < ph; i++) for (j = 0; j < pw; j++) out[r + i][c + j] = after[i][j];
      for (i = 0; i < cells.length; i++) used.add(cells[i]);
    }
    return out;
  }

  function generate(ctx) {
    if (!ctx.same_shape()) return [];
    var res = [], diags = [false, true], i, j, k, fits;
    for (i = 0; i < 2; i++) {
      try { fits = _fit(ctx, diags[i]); } catch (e) { fits = []; }
      for (j = 0; j < fits.length; j++) {
        var overlaps = [true, false];
        for (k = 0; k < 2; k++)
          res.push(_h("stamp" + fits[j].wa.length + "x" + fits[j].wa[0].length +
                      (diags[i] ? "d" : "") + (overlaps[k] ? "o" : ""),
                      (function (bfr, aft, ov) { return function (g) { return _apply(g, bfr, aft, ov); }; })(fits[j].wa, fits[j].wb, overlaps[k]),
                      4.0));
      }
    }
    return res;
  }

  defSolver("patterns", "sequence", generate);
})();
