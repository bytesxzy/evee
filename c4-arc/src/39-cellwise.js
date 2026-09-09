/* Port of engine/solvers/cellwise.py -- local-rule induction (cellular
 * automaton style).
 *
 * For same-shape tasks the output is explained as a function of a bounded
 * local context around each cell. A context family is accepted only if it is
 * conflict-free over every training cell. This covers denoising, outlining,
 * ray drawing, colour swaps, parity stripes and a long tail of "each pixel
 * becomes ..." tasks; capacity is controlled explicitly, and a table nearly as
 * large as the data it was fitted on is rejected as memorisation.
 */

(function () {
  var OOB = -1;

  function _get(g, r, c, h, w) {
    return (r >= 0 && r < h && c >= 0 && c < w) ? g[r][c] : OOB;
  }

  function _mkExtractors(bg) {
    var ex = [];

    ex.push(["color", 3.0, function (g, r, c, a) { return "" + g[r][c]; }]);

    ex.push(["n4", 6.0, function (g, r, c, a) {
      var h = a.h, w = a.w;
      return g[r][c] + "," + _get(g, r - 1, c, h, w) + "," + _get(g, r + 1, c, h, w) +
             "," + _get(g, r, c - 1, h, w) + "," + _get(g, r, c + 1, h, w);
    }]);

    ex.push(["n4set", 5.5, function (g, r, c, a) {
      var h = a.h, w = a.w;
      var v = [_get(g, r - 1, c, h, w), _get(g, r + 1, c, h, w),
               _get(g, r, c - 1, h, w), _get(g, r, c + 1, h, w)];
      v.sort(function (x, y) { return x - y; });
      return g[r][c] + "," + v.join(",");
    }]);

    ex.push(["n8", 8.0, function (g, r, c, a) {
      var h = a.h, w = a.w, out = [], dr, dc;
      for (dr = -1; dr <= 1; dr++) for (dc = -1; dc <= 1; dc++)
        out.push(_get(g, r + dr, c + dc, h, w));
      return out.join(",");
    }]);

    ex.push(["n8mask", 6.0, function (g, r, c, a) {
      var h = a.h, w = a.w, v = g[r][c], m = 0, i;
      for (i = 0; i < G.N8.length; i++)
        if (_get(g, r + G.N8[i][0], c + G.N8[i][1], h, w) === v) m |= 1 << i;
      return v + "," + m;
    }]);

    ex.push(["n8fg", 6.5, function (g, r, c, a) {
      var h = a.h, w = a.w, m = 0, i, u;
      for (i = 0; i < G.N8.length; i++) {
        u = _get(g, r + G.N8[i][0], c + G.N8[i][1], h, w);
        if (u !== bg && u !== OOB) m |= 1 << i;
      }
      return g[r][c] + "," + m;
    }]);

    ex.push(["n8count", 5.0, function (g, r, c, a) {
      var h = a.h, w = a.w, n = 0, i, u;
      for (i = 0; i < G.N8.length; i++) {
        u = _get(g, r + G.N8[i][0], c + G.N8[i][1], h, w);
        if (u !== bg && u !== OOB) n++;
      }
      return g[r][c] + "," + n;
    }]);

    ex.push(["n4count", 5.0, function (g, r, c, a) {
      var h = a.h, w = a.w, n = 0, i, u;
      for (i = 0; i < G.N4.length; i++) {
        u = _get(g, r + G.N4[i][0], c + G.N4[i][1], h, w);
        if (u !== bg && u !== OOB) n++;
      }
      return g[r][c] + "," + n;
    }]);

    ex.push(["parity", 4.5, function (g, r, c, a) { return g[r][c] + "," + (r % 2) + "," + (c % 2); }]);

    var k;
    for (k = 2; k <= 4; k++)
      ex.push(["mod" + k, 5.0, (function (kk) {
        return function (g, r, c, a) { return g[r][c] + "," + (r % kk) + "," + (c % kk); };
      })(k)]);

    ex.push(["border", 4.5, function (g, r, c, a) {
      return g[r][c] + "," + ((r === 0 || c === 0 || r === a.h - 1 || c === a.w - 1) ? 1 : 0);
    }]);

    ex.push(["ring", 5.0, function (g, r, c, a) {
      return g[r][c] + "," + Math.min(r, c, a.h - 1 - r, a.w - 1 - c);
    }]);

    ex.push(["rowcol", 6.0, function (g, r, c, a) {
      return g[r][c] + "|" + a.rowsets[r] + "|" + a.colsets[c];
    }]);

    ex.push(["rcmode", 5.5, function (g, r, c, a) {
      return g[r][c] + "," + a.rowmode[r] + "," + a.colmode[c];
    }]);

    /* Nearest non-background colour in each of the four directions. */
    ex.push(["rays", 7.0, function (g, r, c, a) { return g[r][c] + "," + a.rays[r][c]; }]);

    ex.push(["diagpar", 5.0, function (g, r, c, a) {
      return g[r][c] + "," + (((r + c) % 2) + 2) % 2 + "," + (((r - c) % 2) + 2) % 2;
    }]);

    ex.push(["quadpos", 5.0, function (g, r, c, a) {
      return g[r][c] + "," + (r >= Math.floor((a.h + 1) / 2) ? 1 : 0) +
             "," + (c >= Math.floor((a.w + 1) / 2) ? 1 : 0);
    }]);

    ex.push(["rcuni", 4.5, function (g, r, c, a) { return g[r][c] + "," + a.rowuni[r] + "," + a.coluni[c]; }]);
    ex.push(["rcempty", 4.5, function (g, r, c, a) { return g[r][c] + "," + a.rowbg[r] + "," + a.colbg[c]; }]);
    ex.push(["objsize", 6.0, function (g, r, c, a) { return g[r][c] + "," + a.osize[r][c]; }]);
    ex.push(["objshape", 7.0, function (g, r, c, a) { return g[r][c] + "|" + a.oshape[r][c]; }]);
    ex.push(["objdims", 6.5, function (g, r, c, a) { return g[r][c] + "," + a.odims[r][c]; }]);
    ex.push(["objrel", 7.0, function (g, r, c, a) { return g[r][c] + "," + a.orel[r][c]; }]);
    ex.push(["ccount", 4.5, function (g, r, c, a) { return g[r][c] + "," + (a.ccount[g[r][c]] || 0); }]);
    ex.push(["pos", 7.5, function (g, r, c, a) { return g[r][c] + "," + r + "," + c; }]);
    ex.push(["rowpos", 6.0, function (g, r, c, a) { return g[r][c] + "," + r; }]);
    ex.push(["colpos", 6.0, function (g, r, c, a) { return g[r][c] + "," + c; }]);
    ex.push(["dist", 5.5, function (g, r, c, a) { return g[r][c] + "," + a.dist[r][c]; }]);
    ex.push(["nearcol", 6.0, function (g, r, c, a) { return g[r][c] + "," + a.near[r][c]; }]);
    ex.push(["distcol", 7.0, function (g, r, c, a) { return g[r][c] + "," + a.dist[r][c] + "," + a.near[r][c]; }]);
    ex.push(["crank", 4.5, function (g, r, c, a) {
      var v = a.crank[g[r][c]];
      return g[r][c] + "," + (v === undefined ? -1 : v);
    }]);
    return ex;
  }

  var _PAIRS = [
    ["n8fg", "parity"], ["n8fg", "border"], ["n8count", "parity"],
    ["n4count", "border"], ["objsize", "n8mask"], ["rays", "n8count"],
    ["crank", "n8fg"], ["rcuni", "n8fg"], ["objdims", "objrel"],
    ["ring", "n8fg"], ["ccount", "n8count"], ["rowpos", "colpos"]
  ];

  /* Conjunctions of two context families: plenty of ARC rules need two
     projections at once. Only a curated dozen pairs, not the full product. */
  function _mkPairExtractors(bg) {
    var base = {}, exs = _mkExtractors(bg), i, out = [], fa, fb;
    for (i = 0; i < exs.length; i++) base[exs[i][0]] = exs[i][2];
    for (i = 0; i < _PAIRS.length; i++) {
      fa = base[_PAIRS[i][0]]; fb = base[_PAIRS[i][1]];
      if (!fa || !fb) continue;
      out.push([_PAIRS[i][0] + "+" + _PAIRS[i][1], 9.0, (function (x, y) {
        return function (g, r, c, a) { return x(g, r, c, a) + "#" + y(g, r, c, a); };
      })(fa, fb)]);
    }
    return out;
  }

  var _AUX_CACHE = new Map();

  function _aux(g, bg) {
    var key = G.gkey(g) + "#" + bg, hit = _AUX_CACHE.get(key);
    if (hit !== undefined) return hit;
    var res = _auxBuild(g, bg);
    if (_AUX_CACHE.size > 128) _AUX_CACHE.clear();
    _AUX_CACHE.set(key, res);
    return res;
  }

  /* Multi-source BFS: distance to, and colour of, the nearest filled cell. */
  function _distMaps(g, bg, h, w) {
    var dist = [], near = [], r, c, row, nrow, q = [], qi = 0;
    for (r = 0; r < h; r++) {
      row = new Array(w); nrow = new Array(w);
      for (c = 0; c < w; c++) { row[c] = 9; nrow[c] = OOB; }
      dist.push(row); near.push(nrow);
    }
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) if (g[r][c] !== bg) {
      dist[r][c] = 0; near[r][c] = g[r][c]; q.push(r * 64 + c);
    }
    var cr, cc, d, nr, nc;
    while (qi < q.length) {
      cr = q[qi] >> 6; cc = q[qi] & 63; qi++;
      if (dist[cr][cc] >= 6) continue;
      for (d = 0; d < G.N4.length; d++) {
        nr = cr + G.N4[d][0]; nc = cc + G.N4[d][1];
        if (nr >= 0 && nr < h && nc >= 0 && nc < w && dist[nr][nc] > dist[cr][cc] + 1) {
          dist[nr][nc] = dist[cr][cc] + 1;
          near[nr][nc] = near[cr][cc];
          q.push(nr * 64 + nc);
        }
      }
    }
    return [dist, near];
  }

  function _raysOf(g, bg, h, w) {
    var up = [], dn = [], lf = [], rt = [], r, c, last, row;
    for (r = 0; r < h; r++) {
      up.push(new Array(w)); dn.push(new Array(w)); lf.push(new Array(w)); rt.push(new Array(w));
    }
    for (c = 0; c < w; c++) {
      last = OOB;
      for (r = 0; r < h; r++) { up[r][c] = last; if (g[r][c] !== bg) last = g[r][c]; }
      last = OOB;
      for (r = h - 1; r >= 0; r--) { dn[r][c] = last; if (g[r][c] !== bg) last = g[r][c]; }
    }
    for (r = 0; r < h; r++) {
      last = OOB;
      for (c = 0; c < w; c++) { lf[r][c] = last; if (g[r][c] !== bg) last = g[r][c]; }
      last = OOB;
      for (c = w - 1; c >= 0; c--) { rt[r][c] = last; if (g[r][c] !== bg) last = g[r][c]; }
    }
    var out = [];
    for (r = 0; r < h; r++) {
      row = new Array(w);
      for (c = 0; c < w; c++) row[c] = up[r][c] + "," + dn[r][c] + "," + lf[r][c] + "," + rt[r][c];
      out.push(row);
    }
    return out;
  }

  function _modeOf(row) {
    var cnt = new Map(), order = [], i, k, best, bestN;
    for (i = 0; i < row.length; i++) {
      k = row[i];
      if (!cnt.has(k)) { cnt.set(k, 0); order.push(k); }
      cnt.set(k, cnt.get(k) + 1);
    }
    best = order[0]; bestN = cnt.get(order[0]);
    for (i = 1; i < order.length; i++) if (cnt.get(order[i]) > bestN) { bestN = cnt.get(order[i]); best = order[i]; }
    return best;
  }

  function _setKey(row) {
    var m = 0, i;
    for (i = 0; i < row.length; i++) m |= 1 << row[i];
    return G.csList(m).join(",");
  }

  function _auxBuild(g, bg) {
    var h = g.length, w = g[0].length, cols = G.transpose(g), r, c, i;
    var rowsets = [], colsets = [], rowmode = [], colmode = [];
    var rowuni = [], coluni = [], rowbg = [], colbg = [];
    for (r = 0; r < h; r++) {
      rowsets.push(_setKey(g[r]));
      rowmode.push(_modeOf(g[r]));
      var uni = true, allbg = true;
      for (c = 0; c < w; c++) { if (g[r][c] !== g[r][0]) uni = false; if (g[r][c] !== bg) allbg = false; }
      rowuni.push(uni ? 1 : 0);
      rowbg.push(allbg ? 1 : 0);
    }
    for (c = 0; c < cols.length; c++) {
      colsets.push(_setKey(cols[c]));
      colmode.push(_modeOf(cols[c]));
      var uni2 = true, allbg2 = true;
      for (r = 0; r < cols[c].length; r++) { if (cols[c][r] !== cols[c][0]) uni2 = false; if (cols[c][r] !== bg) allbg2 = false; }
      coluni.push(uni2 ? 1 : 0);
      colbg.push(allbg2 ? 1 : 0);
    }
    var hist = G.histogram(g), items = [];
    for (i = 0; i < G.NCOLORS; i++) if (hist[i] > 0) items.push([i, hist[i]]);
    items.sort(function (a, b) { return (b[1] - a[1]) || (a[0] - b[0]); });
    var crank = {}, ccount = {};
    for (i = 0; i < items.length; i++) { crank[items[i][0]] = i; ccount[items[i][0]] = items[i][1]; }
    var osize = [], oshape = [], odims = [], orel = [];
    for (r = 0; r < h; r++) {
      osize.push(new Array(w).fill(0));
      oshape.push(new Array(w).fill("null"));
      odims.push(new Array(w).fill("null"));
      orel.push(new Array(w).fill("null"));
    }
    var regs = G.floodRegions(g, bg, true, false), k, cells, bb, dh, dw, key, arr, it, s;
    for (k = 0; k < regs.length; k++) {
      cells = regs[k][1];
      bb = G.bboxOf(cells);
      dh = bb[2] - bb[0] + 1; dw = bb[3] - bb[1] + 1;
      arr = [];
      it = cells.values(); s = it.next();
      while (!s.done) { arr.push(((s.value >> 6) - bb[0]) * 64 + ((s.value & 63) - bb[1])); s = it.next(); }
      arr.sort(function (a, b) { return a - b; });
      key = arr.join(",");
      it = cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        osize[r][c] = cells.size;
        oshape[r][c] = key;
        odims[r][c] = dh + "x" + dw;
        orel[r][c] = (r - bb[0]) + "," + (c - bb[1]) + "," + dh + "," + dw;
        s = it.next();
      }
    }
    var dm = _distMaps(g, bg, h, w);
    return { dist: dm[0], near: dm[1], h: h, w: w, rowsets: rowsets, colsets: colsets,
             rowmode: rowmode, colmode: colmode, rays: _raysOf(g, bg, h, w),
             rowuni: rowuni, coluni: coluni, rowbg: rowbg, colbg: colbg,
             ccount: ccount, crank: crank, osize: osize, oshape: oshape,
             odims: odims, orel: orel };
  }

  function _rule(table, ex, bg, strict) {
    return function (g) {
      var h = g.length, w = g[0].length, a = _aux(g, bg), out = [], r, c, row, v;
      for (r = 0; r < h; r++) {
        row = new Array(w);
        for (c = 0; c < w; c++) {
          v = table.get(ex(g, r, c, a));
          if (v === undefined) {
            if (strict) return null;
            v = g[r][c];
          }
          row[c] = v;
        }
        out.push(row);
      }
      return out;
    };
  }

  /* Build the context table for one extractor; null on any conflict. */
  function _fit(train, fn, bg) {
    var table = new Map(), t, a, b, aux, h, w, r, c, k, v, prev;
    for (t = 0; t < train.length; t++) {
      a = train[t][0]; b = train[t][1];
      aux = _aux(a, bg);
      h = a.length; w = a[0].length;
      for (r = 0; r < h; r++) for (c = 0; c < w; c++) {
        k = fn(a, r, c, aux);
        v = b[r][c];
        prev = table.get(k);
        if (prev === undefined) table.set(k, v);
        else if (prev !== v) return null;
      }
    }
    return table.size ? table : null;
  }

  /* Fraction of training pairs predicted by a table fitted without them: a
     table that only reproduces the cells it was built from has memorised. */
  function _loo(train, fn, bg) {
    var n = train.length, ok = 0, i, sub, t, p;
    if (n < 2) return null;
    for (i = 0; i < n; i++) {
      sub = train.slice(0, i).concat(train.slice(i + 1));
      t = _fit(sub, fn, bg);
      if (t === null) continue;
      p = _rule(t, fn, bg, true)(train[i][0]);
      if (p !== null && G.gEq(p, train[i][1])) ok += 1;
    }
    return ok / n;
  }

  function generate(ctx) {
    if (!ctx.same_shape()) return [];
    var bg = ctx.bg(), res = [], ncells = 0, i;
    for (i = 0; i < ctx.train.length; i++) ncells += G.area(ctx.train[i][0]);
    var families = _mkExtractors(bg);
    /* paired contexts have far more capacity; only offer them when there is
       enough evidence for the guards below to mean something */
    if (ncells >= 400) families = families.concat(_mkPairExtractors(bg));
    var table, frac, adj, pen;
    for (i = 0; i < families.length; i++) {
      if (ctx.timed_out()) break;
      table = _fit(ctx.train, families[i][2], bg);
      if (table === null) continue;
      if (table.size * 3 > ncells) continue;     /* capacity guard */
      frac = _loo(ctx.train, families[i][2], bg);
      adj = frac === null ? 0.0 : (2.0 - 4.0 * frac);
      pen = families[i][1] + table.size * 0.02 + adj;
      res.push(new Hyp("ca_" + families[i][0], _rule(table, families[i][2], bg, true), pen, "cellwise"));
      res.push(new Hyp("ca_" + families[i][0] + "_lax", _rule(table, families[i][2], bg, false), pen + 1.0, "cellwise"));
    }
    return res;
  }

  defSolver("cellwise", "cellwise", generate);
})();
