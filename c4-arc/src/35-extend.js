/* Port of engine/solvers/extend.py -- continuing a pattern the grid started.
 *
 * Periodic completion: the period is inferred from the cells that are present,
 * agreeing across every residue class, and must be the smallest such period --
 * a larger one would also fit the evidence and would copy the emptiness along
 * with the pattern. Reflective extension: a grid extended past its own edge by
 * bouncing rather than wrapping, which plain tiling gets wrong at every seam.
 */

(function () {
  var _h = mkHyp("tiling");
  var _MAX = 2500;

  /* Fill values per residue class, or null if the period contradicts. */
  function _consistentPeriod(g, bg, py, px) {
    var h = g.length, w = g[0].length, table = new Map(), r, c, v, k, row;
    for (r = 0; r < h; r++) {
      row = g[r];
      for (c = 0; c < w; c++) {
        v = row[c];
        if (v === bg) continue;
        k = (r % py) * 64 + (c % px);
        if (!table.has(k)) table.set(k, v);
        else if (table.get(k) !== v) return null;
      }
    }
    return table;
  }

  function _contentBox(g, bg) {
    var r0 = Infinity, c0 = Infinity, r1 = -Infinity, c1 = -Infinity, r, c, any = false;
    for (r = 0; r < g.length; r++) for (c = 0; c < g[r].length; c++) if (g[r][c] !== bg) {
      any = true;
      if (r < r0) r0 = r;
      if (r > r1) r1 = r;
      if (c < c0) c0 = c;
      if (c > c1) c1 = c;
    }
    return any ? [r0, c0, r1, c1] : null;
  }

  /* Read the period off the drawn region, then continue it into the empty one:
     the empty cells inside the drawn pattern are part of the pattern, and only
     the ones outside it are missing. */
  function _continueOutward(g, bg, axis) {
    var box = _contentBox(g, bg);
    if (box === null) return null;
    var r0 = box[0], c0 = box[1], r1 = box[2], c1 = box[3];
    var h = g.length, w = g[0].length, bh = r1 - r0 + 1, bw = c1 - c0 + 1;
    var cands = [], py, px, ys = [], xs = [], i;
    if (axis === "both" || axis === "y") { for (py = 1; py < bh; py++) ys.push(py); }
    else ys.push(Math.max(bh, 1));
    if (axis === "both" || axis === "x") { for (px = 1; px < bw; px++) xs.push(px); }
    else xs.push(Math.max(bw, 1));
    for (i = 0; i < ys.length; i++) for (var j = 0; j < xs.length; j++)
      cands.push([ys[i] * xs[j], ys[i], xs[j]]);
    cands.sort(function (a, b) { return (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]); });
    var ci, table, ok, r, c, k, v, out, changed, row, inBand, inside, offAxis;
    for (ci = 0; ci < cands.length; ci++) {
      py = cands[ci][1]; px = cands[ci][2];
      table = new Map(); ok = true;
      for (r = r0; r <= r1 && ok; r++) {
        for (c = c0; c <= c1; c++) {
          k = ((r - r0) % py) * 64 + ((c - c0) % px);
          v = g[r][c];
          if (!table.has(k)) table.set(k, v);
          else if (table.get(k) !== v) { ok = false; break; }
        }
      }
      if (!ok || table.size < py * px) continue;
      out = []; changed = false;
      for (r = 0; r < h; r++) {
        row = new Array(w);
        inBand = (r0 <= r && r <= r1);
        for (c = 0; c < w; c++) {
          inside = inBand && c0 <= c && c <= c1;
          /* a horizontal continuation says nothing about rows the pattern never
             occupied, and a vertical one says nothing about columns */
          offAxis = ((axis === "x" && !inBand) || (axis === "y" && !(c0 <= c && c <= c1)));
          if (inside || offAxis) { row[c] = g[r][c]; continue; }
          v = table.get((((r - r0) % py) + py) % py * 64 + ((((c - c0) % px) + px) % px));
          if (v !== g[r][c]) changed = true;
          row[c] = v;
        }
        out.push(row);
      }
      if (changed) return out;
    }
    return null;
  }

  function _periodicFill(g, bg, py, px, onlyBg) {
    var table = _consistentPeriod(g, bg, py, px);
    if (table === null) return null;
    var h = g.length, w = g[0].length, out = [], r, c, v, row, t;
    for (r = 0; r < h; r++) {
      row = new Array(w);
      for (c = 0; c < w; c++) {
        v = g[r][c];
        if (onlyBg && v !== bg) row[c] = v;
        else {
          t = table.get((r % py) * 64 + (c % px));
          row[c] = t === undefined ? v : t;
        }
      }
      out.push(row);
    }
    return out;
  }

  /* Smallest period consistent with the cells that are actually there,
     inferred per grid: in real tasks it changes from one example to the next. */
  function _bestPeriod(g, bg, axis) {
    var h = g.length, w = g[0].length, cands = [], ys = [], xs = [], py, px, i, j;
    if (axis === "both" || axis === "y") { for (py = 1; py <= h; py++) ys.push(py); } else ys.push(h);
    if (axis === "both" || axis === "x") { for (px = 1; px <= w; px++) xs.push(px); } else xs.push(w);
    for (i = 0; i < ys.length; i++) for (j = 0; j < xs.length; j++) {
      if (ys[i] === h && xs[j] === w) continue;
      cands.push([ys[i] * xs[j], ys[i], xs[j]]);
    }
    cands.sort(function (a, b) { return (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]); });
    for (i = 0; i < cands.length; i++) {
      var table = _consistentPeriod(g, bg, cands[i][1], cands[i][2]);
      if (table === null) continue;
      if (table.size < cands[i][1] * cands[i][2]) continue;
      return [cands[i][1], cands[i][2]];
    }
    return null;
  }

  function _periodicFillAuto(g, bg, axis, onlyBg) {
    var got = _bestPeriod(g, bg, axis);
    if (got === null) return null;
    return _periodicFill(g, bg, got[0], got[1], onlyBg);
  }

  function _mirrorIndex(i, n) {
    if (n <= 1) return 0;
    var period = 2 * n - 2, j = i % period;
    return j < n ? j : period - j;
  }

  function _extendGrid(g, H, W, modeY, modeX) {
    var h = g.length, w = g[0].length, rows = [], i, j, r, src, row;
    if (H * W > _MAX || H <= 0 || W <= 0) return null;
    for (i = 0; i < H; i++) {
      r = modeY === "mirror" ? _mirrorIndex(i, h) : i % h;
      src = g[r]; row = new Array(W);
      for (j = 0; j < W; j++) row[j] = src[modeX === "mirror" ? _mirrorIndex(j, w) : j % w];
      rows.push(row);
    }
    return rows;
  }

  function _target(ctx, g) {
    var h = g.length, w = g[0].length, cs = ctx.const_out_shape();
    if (cs) return cs;
    var sr = ctx.shape_ratio();
    if (sr) return [h * sr[0], w * sr[1]];
    var law = ctx.affine_shape();
    if (law) {
      var H = law[0][0] * h + law[0][1], W = law[1][0] * w + law[1][1];
      if (H > 0 && W > 0) return [H, W];
    }
    return null;
  }

  function generate(ctx) {
    var res = [], seen = new Set();

    function offer(name, fn, cost) {
      if (res.length >= 18) return;
      var hp = _h(name, fn, cost);
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

    var bg = ctx.bg(), axes = ["both", "x", "y"], i, j;
    if (ctx.same_shape()) {
      for (i = 0; i < axes.length; i++)
        offer("continue[" + axes[i] + "]",
              (function (ax) { return function (g) { return _continueOutward(g, G.bgOr(g, bg), ax); }; })(axes[i]),
              2.5 + (axes[i] === "both" ? 0.2 : 0.0));
      var obs = [true, false];
      for (i = 0; i < axes.length; i++) for (j = 0; j < 2; j++)
        offer("period[" + axes[i] + (obs[j] ? "/bg" : "") + "]",
              (function (ax, ob) { return function (g) { return _periodicFillAuto(g, G.bgOr(g, bg), ax, ob); }; })(axes[i], obs[j]),
              2.6 + (axes[i] === "both" ? 0.2 : 0.0));
    }

    if (!ctx.same_shape()) {
      var ms = ["mirror", "wrap"];
      for (i = 0; i < 2; i++) for (j = 0; j < 2; j++)
        offer("ext[" + ms[i] + "/" + ms[j] + "]",
              (function (my, mx) {
                return function (g) {
                  var d = _target(ctx, g);
                  return d ? _extendGrid(g, d[0], d[1], my, mx) : null;
                };
              })(ms[i], ms[j]),
              2.8 + (ms[i] === ms[j] ? 0.0 : 0.3));
    }
    return res;
  }

  defSolver("extend", "tiling", generate);
})();
