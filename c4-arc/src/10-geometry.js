/* Port of engine/solvers/geometry.py -- whole-grid geometric rules.
 *
 * Dihedral maps, crops, tilings and scalings. Cheap, and a substantial slice
 * of ARC-1. Everything is a closure over an input grid, so the harness can
 * validate it on the train pairs before it ever touches a test input.
 */

(function () {
  var _h = mkHyp("geometry");

  function _union(a, b, bg) {
    if (!a || !b || a.length !== b.length || a[0].length !== b[0].length) return null;
    var out = [], r, c, row;
    for (r = 0; r < a.length; r++) {
      row = new Array(a[r].length);
      for (c = 0; c < a[r].length; c++) row[c] = a[r][c] !== bg ? a[r][c] : b[r][c];
      out.push(row);
    }
    return out;
  }

  function _quadMirror(g) {
    var top = G.hconcat(g, G.flipH(g));
    return top ? G.vconcat(top, G.flipV(top)) : null;
  }

  function _quadMirrorR(g) {
    var top = G.hconcat(G.flipH(g), g);
    return top ? G.vconcat(top, G.flipV(top)) : null;
  }

  function _applyFitted(g, choice, ky, kx) {
    var h = g.length, w = g[0].length;
    if (h * ky > 60 || w * kx > 60) return null;
    var rows = [], i, j, nm, t, band;
    for (i = 0; i < ky; i++) {
      band = null;
      for (j = 0; j < kx; j++) {
        nm = choice[i + "," + j];
        if (nm.charAt(0) === "#") t = G.constGrid(h, w, parseInt(nm.slice(1), 10));
        else {
          t = G.DIHEDRAL_MAP[nm](g);
          if (t.length !== h || t[0].length !== w) return null;
        }
        band = band === null ? t : G.hconcat(band, t);
        if (band === null) return null;
      }
      if (band === null) return null;
      rows.push(band);
    }
    var res = rows[0], k;
    for (k = 1; k < rows.length; k++) {
      res = G.vconcat(res, rows[k]);
      if (res === null) return null;
    }
    return res;
  }

  /* Read each tile's transform off the training pairs instead of guessing:
     there are 8^(k*m) ways to fill a tiling with dihedral images, and every
     tile position is independently determined by the demonstrations. */
  function _fittedTiling(ctx, ky, kx, bg) {
    if (ky * kx > 16) return [];
    var choice = {}, i, j, t, n, a, b, h, w, blk, ok, live, uniform, v, r, c;
    for (i = 0; i < ky; i++) for (j = 0; j < kx; j++) {
      live = null;
      for (t = 0; t < ctx.train.length; t++) {
        a = ctx.train[t][0]; b = ctx.train[t][1];
        h = a.length; w = a[0].length;
        blk = G.subgrid(b, i * h, j * w, (i + 1) * h - 1, (j + 1) * w - 1);
        if (blk === null) return [];
        ok = new Set();
        for (n = 0; n < G.DIHEDRAL.length; n++) {
          var tt = G.DIHEDRAL[n][1](a);
          if (tt.length === h && tt[0].length === w && G.gEq(tt, blk)) ok.add(G.DIHEDRAL[n][0]);
        }
        uniform = true; v = blk[0][0];
        for (r = 0; r < blk.length && uniform; r++)
          for (c = 0; c < blk[r].length; c++) if (blk[r][c] !== v) { uniform = false; break; }
        if (uniform) ok.add("#" + v);
        if (live === null) live = ok;
        else {
          var inter = new Set();
          live.forEach(function (x) { if (ok.has(x)) inter.add(x); });
          live = inter;
        }
        if (!live.size) return [];
      }
      var order = [], extra = [];
      for (n = 0; n < G.DIHEDRAL.length; n++) order.push(G.DIHEDRAL[n][0]);
      live.forEach(function (x) { extra.push(x); });
      extra.sort();
      order = order.concat(extra);
      for (n = 0; n < order.length; n++) if (live.has(order[n])) { choice[i + "," + j] = order[n]; break; }
    }
    return [_h("fit_tile" + ky + "x" + kx,
               function (g) { return _applyFitted(g, choice, ky, kx); }, 2.8)];
  }

  function _par(g, i, j) {
    if (i % 2) g = G.flipV(g);
    if (j % 2) g = G.flipH(g);
    return g;
  }

  function _rotc(g, k) {
    var i;
    for (i = 0; i < ((k % 4) + 4) % 4; i++) g = G.rot90(g);
    return g;
  }

  function _buildTiles(g, ky, kx, sel) {
    var h = g.length, w = g[0].length;
    if (h * ky > 60 || w * kx > 60) return null;
    var rows = [], i, j, t, band;
    for (i = 0; i < ky; i++) {
      band = null;
      for (j = 0; j < kx; j++) {
        t = sel(i, j, g);
        if (!t || t.length !== h || t[0].length !== w) return null;
        band = band === null ? t : G.hconcat(band, t);
        if (band === null) return null;
      }
      rows.push(band);
    }
    var res = rows[0], k;
    for (k = 1; k < rows.length; k++) {
      res = G.vconcat(res, rows[k]);
      if (res === null) return null;
    }
    return res;
  }

  /* Tilings where tile (i, j) is a dihedral image chosen by parity. */
  function _mirrorTilings(ky, kx) {
    var combos = [
      ["mirror", function (i, j, g) { return _par(g, i, j); }],
      ["mirror_r", function (i, j, g) { return _par(g, i + 1, j + 1); }],
      ["mirror_rows", function (i, j, g) { return (i % 2) ? G.flipV(g) : g; }],
      ["mirror_cols", function (i, j, g) { return (j % 2) ? G.flipH(g) : g; }],
      ["rot_cycle", function (i, j, g) { return _rotc(g, (i + j) % 4); }]
    ];
    var res = [], i;
    for (i = 0; i < combos.length; i++)
      res.push(_h("tile_" + combos[i][0] + ky + "x" + kx,
                  (function (sel) { return function (g) { return _buildTiles(g, ky, kx, sel); }; })(combos[i][1]),
                  4.0));
    return res;
  }

  /* out[i*h+r][j*w+c] = g[r][c] gated on the value of g[i][j]. */
  function _frac(g, polarity, bg, fill) {
    var h = g.length, w = g[0].length;
    if (h * h > 60 || w * w > 60) return null;
    var out = G.constGrid(h * h, w * w, fill), i, j, r, c, on, orow, grow;
    for (i = 0; i < h; i++) for (j = 0; j < w; j++) {
      on = polarity ? (g[i][j] !== bg) : (g[i][j] === bg);
      if (on) for (r = 0; r < h; r++) {
        orow = out[i * h + r]; grow = g[r];
        for (c = 0; c < w; c++) orow[j * w + c] = grow[c];
      }
    }
    return out;
  }

  function _fractal(ky, kx, ctx, bg) {
    var res = [], p, i;
    var fills = G.csList(G.csAdd(G.csAdd(G.csDiff(ctx.out_palette(), ctx.in_palette()), bg), 0));
    for (p = 0; p < 2; p++) {
      var polarity = (p === 0);
      for (i = 0; i < fills.length; i++)
        res.push(_h("fractal" + (polarity ? "" : "_inv") + "#" + fills[i],
                    (function (pol, f) { return function (g) { return _frac(g, pol, bg, f); }; })(polarity, fills[i]),
                    4.5));
    }
    return res;
  }

  /* Scale factors that are a function of the input, not a constant. */
  function _dynamicScale(ctx, bg) {
    var feats = [
      ["ncolors", function (g) { return G.csSize(G.palette(g)); }],
      ["ncolors_nb", function (g) { return G.csSize(G.csDiff(G.palette(g), 1 << bg)); }],
      ["maxcount", function (g) {
        if (G.csSize(G.palette(g)) <= 1) return 1;
        var hist = G.histogram(g), best = -Infinity, k;
        for (k = 0; k < G.NCOLORS; k++) if (k !== bg && hist[k] > 0 && hist[k] > best) best = hist[k];
        if (best === -Infinity) throw new Error("empty");
        return best;
      }],
      ["nnz", function (g) {
        var n = 0, r, c;
        for (r = 0; r < g.length; r++) for (c = 0; c < g[r].length; c++) if (g[r][c] !== bg) n++;
        return n;
      }],
      ["nobj", function (g) { return G.floodRegions(g, bg, true, true).length; }]
    ];
    var res = [], i, t, a, b, ah, aw, bh, bw, k, okUp;
    for (i = 0; i < feats.length; i++) {
      okUp = true;
      for (t = 0; t < ctx.train.length; t++) {
        a = ctx.train[t][0]; b = ctx.train[t][1];
        ah = a.length; aw = a[0].length; bh = b.length; bw = b[0].length;
        try { k = feats[i][1](a); } catch (e) { okUp = false; break; }
        if (k < 1 || bh !== ah * k || bw !== aw * k) { okUp = false; break; }
      }
      if (okUp) {
        res.push(_h("upscale_by_" + feats[i][0],
                    (function (f) { return function (g) { return G.upscale(g, f(g), f(g)); }; })(feats[i][1]), 5.0));
        res.push(_h("tile_by_" + feats[i][0],
                    (function (f) { return function (g) { return G.tile(g, f(g), f(g)); }; })(feats[i][1]), 5.0));
      }
    }
    return res;
  }

  function generate(ctx) {
    var out = [], train = ctx.train, bg = ctx.bg(), i, c, pal;

    var seen = new Set(), t;
    for (t = 0; t < train.length; t++) seen.add(G.gkey(train[t][1]));
    if (seen.size === 1) {
      var k = train[0][1];
      out.push(_h("const", function () { return k; }, 12.0));
    }

    for (i = 0; i < G.DIHEDRAL.length; i++)
      out.push(_h(G.DIHEDRAL[i][0], G.DIHEDRAL[i][1], G.DIHEDRAL[i][0] === "id" ? 1.0 : 2.0));

    out.push(_h("crop_content", function (g) { return G.cropToContent(g, bg); }, 3.0));
    pal = G.csList(ctx.in_palette());
    for (i = 0; i < pal.length; i++)
      out.push(_h("crop_content#" + pal[i],
                  (function (cc) { return function (g) { return G.cropToContent(g, cc); }; })(pal[i]), 4.5));
    out.push(_h("trim1", function (g) { return G.trimBorder(g, 1); }, 3.0));
    out.push(_h("trim2", function (g) { return G.trimBorder(g, 2); }, 4.0));
    out.push(_h("dedup", G.dedup, 3.5));
    out.push(_h("dedup_rows", G.dedupRows, 4.0));
    out.push(_h("dedup_cols", G.dedupCols, 4.0));

    var halves = ["top", "bottom", "left", "right"];
    for (i = 0; i < halves.length; i++)
      out.push(_h("half_" + halves[i],
                  (function (w) { return function (g) { return G.half(g, w); }; })(halves[i]), 3.0));
    for (i = 0; i < 4; i++)
      out.push(_h("quad" + i, (function (q) { return function (g) { return G.quadrant(g, q); }; })(i), 3.5));

    pal = G.csList(G.csAdd(ctx.out_palette(), 0));
    for (i = 0; i < pal.length; i++)
      out.push(_h("pad1#" + pal[i],
                  (function (cc) { return function (g) { return G.pad(g, 1, cc); }; })(pal[i]), 4.0));

    var r = ctx.shape_ratio();
    if (r && !(r[0] === 1 && r[1] === 1)) {
      var ky = r[0], kx = r[1];
      out.push(_h("upscale" + ky + "x" + kx, function (g) { return G.upscale(g, ky, kx); }, 2.5));
      out.push(_h("tile" + ky + "x" + kx, function (g) { return G.tile(g, ky, kx); }, 3.0));
      out = out.concat(_mirrorTilings(ky, kx));
      out = out.concat(_fittedTiling(ctx, ky, kx, bg));
      out = out.concat(_fractal(ky, kx, ctx, bg));
    }
    var ir = ctx.inv_shape_ratio();
    if (ir && !(ir[0] === 1 && ir[1] === 1)) {
      var iy = ir[0], ix = ir[1];
      out.push(_h("downscale" + iy + "x" + ix, function (g) { return G.downscale(g, iy, ix); }, 2.5));
      out.push(_h("modescale" + iy + "x" + ix, function (g) { return G.blockReduceMode(g, iy, ix); }, 3.5));
    }

    out = out.concat(_dynamicScale(ctx, bg));

    for (i = 0; i < G.DIHEDRAL.length; i++) {
      if (G.DIHEDRAL[i][0] === "id") continue;
      (function (name, f) {
        out.push(_h("hcat_" + name, function (g) { return G.hconcat(g, f(g)); }, 4.0));
        out.push(_h("vcat_" + name, function (g) { return G.vconcat(g, f(g)); }, 4.0));
        out.push(_h("hcat_" + name + "_rev", function (g) { return G.hconcat(f(g), g); }, 4.5));
        out.push(_h("vcat_" + name + "_rev", function (g) { return G.vconcat(f(g), g); }, 4.5));
      })(G.DIHEDRAL[i][0], G.DIHEDRAL[i][1]);
    }
    out.push(_h("hcat_self", function (g) { return G.hconcat(g, g); }, 4.0));
    out.push(_h("vcat_self", function (g) { return G.vconcat(g, g); }, 4.0));
    for (i = 1; i < G.DIHEDRAL.length; i++) {
      (function (name, f) {
        out.push(_h("union_" + name, function (g) { return _union(g, f(g), bg); }, 3.2));
        out.push(_h("union_" + name + "_rev", function (g) { return _union(f(g), g, bg); }, 3.4));
      })(G.DIHEDRAL[i][0], G.DIHEDRAL[i][1]);
    }
    out.push(_h("quad_mirror", _quadMirror, 4.5));
    out.push(_h("quad_mirror_r", _quadMirrorR, 5.0));

    var dirs = ["down", "up", "left", "right"];
    for (i = 0; i < dirs.length; i++)
      out.push(_h("gravity_" + dirs[i],
                  (function (d) { return function (g) { return G.gravity(g, bg, d); }; })(dirs[i]), 4.0));

    var shifts = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1]];
    for (i = 0; i < shifts.length; i++) {
      (function (dr, dc) {
        out.push(_h("wrap" + sgn(dr) + sgn(dc), function (g) { return G.wrapTranslate(g, dr, dc); }, 4.5));
        out.push(_h("shift" + sgn(dr) + sgn(dc), function (g) { return G.translate(g, dr, dc, bg); }, 4.5));
      })(shifts[i][0], shifts[i][1]);
    }

    pal = G.csList(ctx.out_palette());
    for (i = 0; i < pal.length; i++) {
      (function (cc) {
        out.push(_h("fill_holes#" + cc, function (g) { return G.fillHoles(g, cc, bg); }, 3.5));
        out.push(_h("fill_holes8#" + cc, function (g) { return G.fillHoles(g, cc, bg, true); }, 4.0));
      })(pal[i]);
    }
    return out;
  }

  defSolver("geometry", "geometry", generate);
})();
