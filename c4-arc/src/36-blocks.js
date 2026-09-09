/* Port of engine/solvers/blocks.py -- learned block dictionaries.
 *
 * blockmap expands every input cell into a k x m patch chosen by its colour;
 * blockfold collapses every k x m block to one output cell; panelmap rewrites
 * each panel by a table keyed on its content. A table is accepted only if it
 * compresses -- at least two observations per entry -- because "fewer entries
 * than observations" still admits a transcript of the training data.
 */

(function () {
  var _h = mkHyp("tiling");

  function _blockOf(g, r, c, ky, kx) {
    var out = [], i;
    for (i = 0; i < ky; i++) out.push(g[r * ky + i].slice(c * kx, (c + 1) * kx));
    return out;
  }

  function _fitBlockmap(ctx, ky, kx, keyedOnShape) {
    var table = new Map(), nObs = 0, t, a, b, ah, aw, r, c, patch, k, prev;
    for (t = 0; t < ctx.train.length; t++) {
      a = ctx.train[t][0]; b = ctx.train[t][1];
      ah = a.length; aw = a[0].length;
      if (b.length !== ah * ky || b[0].length !== aw * kx) return null;
      for (r = 0; r < ah; r++) for (c = 0; c < aw; c++) {
        patch = _blockOf(b, r, c, ky, kx);
        k = keyedOnShape ? (a[r][c] + "," + (r % 2) + "," + (c % 2)) : String(a[r][c]);
        prev = table.get(k);
        if (prev === undefined) table.set(k, patch);
        else if (!G.gEq(prev, patch)) return null;
        nObs += 1;
      }
    }
    if (!table.size || table.size * 2 > nObs) return null;
    return function (g) {
      var h = g.length, w = g[0].length, rr, cc, i, j, p, kk;
      if (h * ky > 60 || w * kx > 60) return null;
      var out = G.constGrid(h * ky, w * kx, 0);
      for (rr = 0; rr < h; rr++) for (cc = 0; cc < w; cc++) {
        kk = keyedOnShape ? (g[rr][cc] + "," + (rr % 2) + "," + (cc % 2)) : String(g[rr][cc]);
        p = table.get(kk);
        if (p === undefined) return null;
        for (i = 0; i < ky; i++) for (j = 0; j < kx; j++) out[rr * ky + i][cc * kx + j] = p[i][j];
      }
      return out;
    };
  }

  function _fitBlockfold(ctx, ky, kx) {
    var table = new Map(), nObs = 0, t, a, b, ah, aw, bh, bw, r, c, patch, prev, k;
    for (t = 0; t < ctx.train.length; t++) {
      a = ctx.train[t][0]; b = ctx.train[t][1];
      ah = a.length; aw = a[0].length; bh = b.length; bw = b[0].length;
      if (ah !== bh * ky || aw !== bw * kx) return null;
      for (r = 0; r < bh; r++) for (c = 0; c < bw; c++) {
        patch = _blockOf(a, r, c, ky, kx);
        k = G.gkey(patch);
        prev = table.get(k);
        if (prev === undefined) table.set(k, b[r][c]);
        else if (prev !== b[r][c]) return null;
        nObs += 1;
      }
    }
    if (!table.size || table.size * 2 > nObs) return null;
    return function (g) {
      var h = g.length, w = g[0].length, r, c, out = [], row, v;
      if (h % ky || w % kx) return null;
      for (r = 0; r < h / ky; r++) {
        row = [];
        for (c = 0; c < w / kx; c++) {
          v = table.get(G.gkey(_blockOf(g, r, c, ky, kx)));
          if (v === undefined) return null;
          row.push(v);
        }
        out.push(row);
      }
      return out;
    };
  }

  function _fitPanelmap(ctx, dec, toCell) {
    var table = new Map(), nObs = 0, t, a, b, mat, omat, nr, nc, i, j, p, q, prev, k;
    for (t = 0; t < ctx.train.length; t++) {
      a = ctx.train[t][0]; b = ctx.train[t][1];
      mat = dec(a);
      if (!mat) return null;
      nr = mat.length; nc = mat[0].length;
      if (toCell) {
        if (b.length !== nr || b[0].length !== nc) return null;
        for (i = 0; i < nr; i++) for (j = 0; j < nc; j++) {
          p = mat[i][j];
          if (!p) return null;
          k = G.gkey(p);
          prev = table.get(k);
          if (prev === undefined) table.set(k, b[i][j]);
          else if (prev !== b[i][j]) return null;
          nObs += 1;
        }
      } else {
        omat = dec(b);
        if (!omat || omat.length !== nr || omat[0].length !== nc) return null;
        for (i = 0; i < nr; i++) for (j = 0; j < nc; j++) {
          p = mat[i][j]; q = omat[i][j];
          if (!p || !q) return null;
          k = G.gkey(p);
          prev = table.get(k);
          if (prev === undefined) table.set(k, q);
          else if (!G.gEq(prev, q)) return null;
          nObs += 1;
        }
      }
    }
    if (!table.size || table.size * 2 > nObs) return null;
    return function (g) {
      var m = dec(g), i, j, out, orow, v, rows, band, q, res;
      if (!m) return null;
      if (toCell) {
        out = [];
        for (i = 0; i < m.length; i++) {
          orow = [];
          for (j = 0; j < m[i].length; j++) {
            if (!m[i][j]) return null;
            v = table.get(G.gkey(m[i][j]));
            if (v === undefined) return null;
            orow.push(v);
          }
          out.push(orow);
        }
        return out;
      }
      rows = [];
      for (i = 0; i < m.length; i++) {
        band = null;
        for (j = 0; j < m[i].length; j++) {
          if (!m[i][j]) return null;
          q = table.get(G.gkey(m[i][j]));
          if (q === undefined) return null;
          band = band === null ? q : G.hconcat(band, q);
          if (band === null) return null;
        }
        if (band === null) return null;
        rows.push(band);
      }
      res = rows[0];
      for (i = 1; i < rows.length; i++) {
        res = G.vconcat(res, rows[i]);
        if (res === null) return null;
      }
      return res;
    };
  }

  function _linemapTable(train, axis) {
    var table = new Map(), n = 0, t, a, b, ra, rb, i, k, prev;
    for (t = 0; t < train.length; t++) {
      a = train[t][0]; b = train[t][1];
      if (a.length !== b.length || a[0].length !== b[0].length) return [null, 0];
      ra = axis === 0 ? a : G.transpose(a);
      rb = axis === 0 ? b : G.transpose(b);
      for (i = 0; i < ra.length; i++) {
        k = ra[i].join(",");
        prev = table.get(k);
        if (prev === undefined) table.set(k, rb[i]);
        else if (!G.rowEq(prev, rb[i])) return [null, 0];
        n += 1;
      }
    }
    return [table.size ? table : null, n];
  }

  /* Rows are a natural unit for striping, row-wise recolouring and sorting,
     and a row-level table compresses far harder than a cell-level one. A row
     dictionary generalises only if rows recur across grids, and refitting
     without a pair is the direct test of that. */
  function _fitLinemap(ctx, axis) {
    var got = _linemapTable(ctx.train, axis), table = got[0], n = got[1];
    if (table === null || table.size * 2 > n) return null;
    if (ctx.train.length >= 3) {
      var ok = 0, i, j, sub, t2, a, b, ra, rb, good;
      for (i = 0; i < ctx.train.length; i++) {
        sub = ctx.train.slice(0, i).concat(ctx.train.slice(i + 1));
        t2 = _linemapTable(sub, axis)[0];
        if (t2 === null) continue;
        a = ctx.train[i][0]; b = ctx.train[i][1];
        ra = axis === 0 ? a : G.transpose(a);
        rb = axis === 0 ? b : G.transpose(b);
        good = true;
        for (j = 0; j < ra.length; j++) {
          var v = t2.get(ra[j].join(","));
          if (v === undefined || !G.rowEq(v, rb[j])) { good = false; break; }
        }
        if (good) ok += 1;
      }
      if (ok * 2 < ctx.train.length) return null;
    }
    return function (g) {
      var rows = axis === 0 ? g : G.transpose(g), out = [], i, y;
      for (i = 0; i < rows.length; i++) {
        y = table.get(rows[i].join(","));
        if (y === undefined) return null;
        out.push(y.slice());
      }
      return axis === 0 ? out : G.transpose(out);
    };
  }

  /* Delete rows (or columns) that are empty, or uniform, or duplicated. */
  function _dropLines(g, axis, mode, bg) {
    bg = G.bgOr(g, bg);
    var rows = axis === 0 ? g : G.transpose(g), keep = [], seen = new Set(), i, j, r, skip, uni, k;
    for (i = 0; i < rows.length; i++) {
      r = rows[i]; skip = false;
      if (mode === "empty") {
        skip = true;
        for (j = 0; j < r.length; j++) if (r[j] !== bg) { skip = false; break; }
      } else if (mode === "uniform") {
        uni = true;
        for (j = 1; j < r.length; j++) if (r[j] !== r[0]) { uni = false; break; }
        skip = uni;
      } else if (mode === "dup") {
        k = r.join(",");
        if (seen.has(k)) skip = true;
        else seen.add(k);
      }
      if (!skip) keep.push(r.slice());
    }
    if (!keep.length || keep.length === rows.length) return null;
    return axis === 0 ? keep : G.transpose(keep);
  }

  function generate(ctx) {
    var res = [], r = ctx.shape_ratio(), i, f, keyed;
    if (r && !(r[0] === 1 && r[1] === 1) && r[0] * r[1] <= 36) {
      var keyeds = [false, true];
      for (i = 0; i < 2; i++) {
        keyed = keyeds[i];
        f = _fitBlockmap(ctx, r[0], r[1], keyed);
        if (f !== null) res.push(_h("blockmap" + r[0] + "x" + r[1] + (keyed ? "_p" : ""), f, 3.0));
      }
    }
    var ir = ctx.inv_shape_ratio();
    if (ir && !(ir[0] === 1 && ir[1] === 1) && ir[0] * ir[1] <= 64) {
      f = _fitBlockfold(ctx, ir[0], ir[1]);
      if (f !== null) res.push(_h("blockfold" + ir[0] + "x" + ir[1], f, 3.0));
    }
    var axis, modes = ["empty", "uniform", "dup"], m;
    for (axis = 0; axis <= 1; axis++) {
      f = _fitLinemap(ctx, axis);
      if (f !== null) res.push(_h("linemap" + axis, f, 5.0));
      for (m = 0; m < modes.length; m++)
        res.push(_h("drop_" + modes[m] + axis,
                    (function (ax, md, bb) { return function (g) { return _dropLines(g, ax, md, bb); }; })(axis, modes[m], ctx.bg()),
                    3.2));
    }
    var decs = PART.decompositions(ctx).slice(0, 6), d, tcs = [true, false];
    for (d = 0; d < decs.length; d++)
      for (i = 0; i < 2; i++) {
        try { f = _fitPanelmap(ctx, decs[d][2], tcs[i]); } catch (e) { f = null; }
        if (f !== null)
          res.push(_h("panelmap_" + decs[d][0] + "_" + (tcs[i] ? "cell" : "grid"), f, decs[d][1] + 1.5));
      }
    return res;
  }

  defSolver("blocks", "tiling", generate);
})();
