/* Port of engine/solvers/paneltable.py -- panel stacks combined by a learned
 * table rather than a fixed logic op.
 *
 * For every cell position the stack of panels supplies a key -- the tuple of
 * colours there, or an order-insensitive reading of it -- and the training
 * outputs supply the value. One conflict-free table explaining every cell *is*
 * the operator, and expresses all eight boolean ops, colour-preserving
 * overlays, priority rules and more besides. A table is rejected unless it
 * sees at least two observations per entry: one row per cell is a transcript.
 */

(function () {
  var _MIN_OBS_PER_ENTRY = 2.0;

  function _flat(mat) {
    var out = [], i, j;
    for (i = 0; i < mat.length; i++) for (j = 0; j < mat[i].length; j++) out.push(mat[i][j]);
    return out;
  }

  function _keyfns(bg) {
    return [
      ["ordered", 0.0, function (vals) { return "o" + vals.join(","); }],
      ["sorted", 0.4, function (vals) { return "s" + vals.slice().sort(function (a, b) { return a - b; }).join(","); }],
      ["set", 0.6, function (vals) {
        var m = 0, i;
        for (i = 0; i < vals.length; i++) m |= 1 << vals[i];
        return "t" + m;
      }],
      ["count", 0.8, function (vals) {
        var n = 0, i;
        for (i = 0; i < vals.length; i++) if (vals[i] !== bg) n++;
        return "c" + n;
      }],
      ["count1", 0.7, function (vals) {
        var nz = [], i;
        for (i = 0; i < vals.length; i++) if (vals[i] !== bg) nz.push(vals[i]);
        return "1" + nz.length + "," + (nz.length ? nz[0] : bg);
      }],
      ["first", 0.9, function (vals) {
        var i;
        for (i = 0; i < vals.length; i++) if (vals[i] !== bg) return "f" + vals[i];
        return "f" + bg;
      }]
    ];
  }

  function _tableRule(dec, keyfn, table, bg, def) {
    return function (g) {
      var mat;
      try { mat = dec(g); } catch (e) { return null; }
      if (!mat) return null;
      var ps = _flat(mat), i;
      if (ps.length < 2) return null;
      for (i = 0; i < ps.length; i++) if (!ps[i]) return null;
      var h = ps[0].length, w = ps[0][0].length;
      for (i = 1; i < ps.length; i++) if (ps[i].length !== h || ps[i][0].length !== w) return null;
      var out = [], r, c, row, vals, v;
      for (r = 0; r < h; r++) {
        row = new Array(w);
        for (c = 0; c < w; c++) {
          vals = [];
          for (i = 0; i < ps.length; i++) vals.push(ps[i][r][c]);
          v = table.get(keyfn(vals));
          if (v === undefined) v = def;
          if (v === null || v === undefined) return null;
          row[c] = v;
        }
        out.push(row);
      }
      return out;
    };
  }

  /* Per-pair panel stacks aligned with their outputs, or null. */
  function _observe(ctx, dec) {
    var rows = [], t, mat, ps, i, h, w, b;
    for (t = 0; t < ctx.train.length; t++) {
      try { mat = dec(ctx.train[t][0]); } catch (e) { return null; }
      if (!mat) return null;
      ps = _flat(mat);
      if (ps.length < 2) return null;
      for (i = 0; i < ps.length; i++) if (!ps[i]) return null;
      h = ps[0].length; w = ps[0][0].length;
      b = ctx.train[t][1];
      for (i = 1; i < ps.length; i++) if (ps[i].length !== h || ps[i][0].length !== w) return null;
      if (b.length !== h || b[0].length !== w) return null;
      rows.push([ps, b, h, w]);
    }
    if (!rows.length) return null;
    for (i = 1; i < rows.length; i++) if (rows[i][0].length !== rows[0][0].length) return null;
    return rows;
  }

  function _fit(rows, keyfn) {
    var table = new Map(), obs = 0, i, r, c, ps, b, h, w, vals, p, k, v;
    for (i = 0; i < rows.length; i++) {
      ps = rows[i][0]; b = rows[i][1]; h = rows[i][2]; w = rows[i][3];
      for (r = 0; r < h; r++) for (c = 0; c < w; c++) {
        vals = [];
        for (p = 0; p < ps.length; p++) vals.push(ps[p][r][c]);
        k = keyfn(vals); v = b[r][c];
        if (!table.has(k)) table.set(k, v);
        else if (table.get(k) !== v) return null;
        obs += 1;
      }
    }
    if (!table.size) return null;
    if (obs < _MIN_OBS_PER_ENTRY * table.size) return null;   /* a transcript */
    return [table, obs];
  }

  var _LAYOUTS = ["same", "transpose", "column", "row"];

  function _panelKey(panel, bg, kind) {
    var r, c, n, m;
    if (kind === "content") return "c" + G.gkey(panel);
    if (kind === "mask") {
      var parts = [];
      for (r = 0; r < panel.length; r++) {
        var row = [];
        for (c = 0; c < panel[r].length; c++) row.push(panel[r][c] !== bg ? 1 : 0);
        parts.push(row.join(""));
      }
      return "m" + parts.join("|");
    }
    if (kind === "count") {
      n = 0;
      for (r = 0; r < panel.length; r++) for (c = 0; c < panel[r].length; c++) if (panel[r][c] !== bg) n++;
      return "n" + n;
    }
    if (kind === "palette") {
      m = 0;
      for (r = 0; r < panel.length; r++) for (c = 0; c < panel[r].length; c++) m |= 1 << panel[r][c];
      return "p" + m;
    }
    return null;
  }

  function _layoutDims(mat, layout) {
    var rows = mat.length, cols = mat[0].length;
    if (layout === "same") return [rows, cols];
    if (layout === "transpose") return [cols, rows];
    if (layout === "column") return [rows * cols, 1];
    return [1, rows * cols];
  }

  function _layoutOrder(mat, layout) {
    var rows = mat.length, cols = mat[0].length, out = [], r, c;
    if (layout === "transpose") {
      for (c = 0; c < cols; c++) for (r = 0; r < rows; r++) out.push(mat[r][c]);
      return out;
    }
    for (r = 0; r < rows; r++) for (c = 0; c < cols; c++) out.push(mat[r][c]);
    return out;
  }

  function _dictRule(dec, kind, layout, table, bg, bh, bw) {
    return function (g) {
      var mat;
      try { mat = dec(g); } catch (e) { return null; }
      if (!mat || !mat[0]) return null;
      var ld = _layoutDims(mat, layout), lr = ld[0], lc = ld[1];
      var seq = _layoutOrder(mat, layout);
      if (seq.length !== lr * lc) return null;
      var out = G.constGrid(lr * bh, lc * bw, bg), i, block, r0, c0, r, c;
      for (i = 0; i < seq.length; i++) {
        if (!seq[i]) return null;
        block = table.get(_panelKey(seq[i], bg, kind));
        if (block === undefined) return null;
        r0 = Math.floor(i / lc) * bh;
        c0 = (i % lc) * bw;
        for (r = 0; r < bh; r++) for (c = 0; c < bw; c++) out[r0 + r][c0 + c] = block[r][c];
      }
      return out;
    };
  }

  function _sigOf(hp, ctx) {
    var parts = [], i, p;
    for (i = 0; i < ctx.test_inputs.length; i++) {
      p = hp.apply(ctx.test_inputs[i]);
      parts.push(p === null ? "*" : G.gkey(p));
    }
    return parts.join("~");
  }

  function _dictRules(ctx, dname, dcost, dec, bg, res, seen) {
    var obs = [], t, mat, i, j;
    for (t = 0; t < ctx.train.length; t++) {
      try { mat = dec(ctx.train[t][0]); } catch (e) { return; }
      if (!mat || !mat[0]) return;
      obs.push([mat, ctx.train[t][1]]);
    }
    var L, layout, ok, dims, dimKeys, r, c, bh, bw, b, kinds = ["content", "mask", "count", "palette"];
    for (L = 0; L < _LAYOUTS.length; L++) {
      layout = _LAYOUTS[L];
      ok = true; dims = {}; dimKeys = [];
      for (i = 0; i < obs.length; i++) {
        var ld = _layoutDims(obs[i][0], layout);
        r = ld[0]; c = ld[1];
        b = obs[i][1];
        if (r <= 0 || c <= 0 || b.length % r || b[0].length % c) { ok = false; break; }
        var kk = (b.length / r) + "," + (b[0].length / c);
        if (!dims[kk]) { dims[kk] = [b.length / r, b[0].length / c]; dimKeys.push(kk); }
      }
      if (!ok || dimKeys.length !== 1) continue;
      bh = dims[dimKeys[0]][0]; bw = dims[dimKeys[0]][1];
      if (bh * bw > 64) continue;
      var kd;
      for (kd = 0; kd < kinds.length; kd++) {
        var table = new Map(), counts = new Map(), good = true, kind = kinds[kd];
        for (i = 0; i < obs.length && good; i++) {
          var ld2 = _layoutDims(obs[i][0], layout), cc = ld2[1];
          var seq = _layoutOrder(obs[i][0], layout);
          b = obs[i][1];
          for (j = 0; j < seq.length; j++) {
            if (!seq[j]) { good = false; break; }
            var k2 = _panelKey(seq[j], bg, kind);
            if (k2 === null) { good = false; break; }
            var r0 = Math.floor(j / cc) * bh, c0 = (j % cc) * bw, block = [], y, x, row;
            for (y = 0; y < bh; y++) {
              row = new Array(bw);
              for (x = 0; x < bw; x++) row[x] = b[r0 + y][c0 + x];
              block.push(row);
            }
            if (!table.has(k2)) table.set(k2, block);
            else if (!G.gEq(table.get(k2), block)) { good = false; break; }
            counts.set(k2, (counts.get(k2) || 0) + 1);
          }
        }
        if (!good || !table.size) continue;
        var minCount = Infinity;
        counts.forEach(function (v) { if (v < minCount) minCount = v; });
        if (minCount < 2) continue;   /* a class seen once is an answer */
        var hp = new Hyp("pdict[" + dname + "/" + kind + "/" + layout + "|" + table.size + "]",
                         _dictRule(dec, kind, layout, table, bg, bh, bw),
                         2.8 + dcost * 0.3 + 0.05 * table.size, "partition");
        if (!hp.fits(ctx.train)) continue;
        var sig;
        try { sig = _sigOf(hp, ctx); } catch (e) { continue; }
        if (seen.has(sig)) continue;
        seen.add(sig);
        res.push(hp);
        return;
      }
    }
  }

  function generate(ctx) {
    if (ctx.same_shape()) return [];
    var bg = ctx.bg(), res = [], seen = new Set(), decs = PART.decompositions(ctx), d, i;
    for (d = 0; d < decs.length; d++) {
      if (ctx.timed_out() || res.length >= 16) break;
      var dname = decs[d][0], dcost = decs[d][1], dec = decs[d][2];
      try { _dictRules(ctx, dname, dcost, dec, bg, res, seen); } catch (e) {}
      var rows = _observe(ctx, dec);
      if (!rows) continue;
      var nPanels = rows[0][0].length, kfs = _keyfns(bg), kf;
      for (kf = 0; kf < kfs.length; kf++) {
        var got = _fit(rows, kfs[kf][2]);
        if (got === null) continue;
        var table = got[0];
        var modalCnt = new Map(), modal = null, modalN = -1;
        table.forEach(function (v) { modalCnt.set(v, (modalCnt.get(v) || 0) + 1); });
        modalCnt.forEach(function (n, v) { if (n > modalN) { modalN = n; modal = v; } });
        var defs = [[null, 0.0], [modal, 0.4]], dfi;
        for (dfi = 0; dfi < 2; dfi++) {
          var hp = new Hyp("ptab[" + dname + "/" + kfs[kf][0] + "|" + nPanels + ">" + table.size + "]",
                           _tableRule(dec, kfs[kf][2], table, bg, defs[dfi][0]),
                           2.4 + dcost * 0.3 + kfs[kf][1] + 0.02 * table.size + defs[dfi][1],
                           "partition");
          if (!hp.fits(ctx.train)) continue;
          var sig;
          try { sig = _sigOf(hp, ctx); } catch (e) { continue; }
          if (seen.has(sig)) break;
          seen.add(sig);
          res.push(hp);
          break;
        }
      }
    }
    return res;
  }

  defSolver("paneltable", "partition", generate);
})();
