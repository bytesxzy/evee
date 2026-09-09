/* Port of engine/solvers/sequence.py -- motion and drawing.
 *
 * Gravity with collision, rays and connections: procedural rules that cannot
 * be written as a pixel lookup, and common enough in ARC to deserve
 * first-class treatment.
 */

var SEQ = {};

(function () {
  var _h = mkHyp("sequence");
  var DIR_NAMES = ["up", "down", "left", "right"];
  var DIRS = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };
  var DIAG_NAMES = ["ul", "ur", "dl", "dr"];
  var DIAG = { ul: [-1, -1], ur: [-1, 1], dl: [1, -1], dr: [1, 1] };

  function _moveObjects(g, seg, bg, d, once) {
    bg = G.bgOr(g, bg);
    var dr = DIRS[d][0], dc = DIRS[d][1];
    var objs = O.segment(g, seg, bg);
    if (!objs.length || objs.length > 60) return null;
    var h = g.length, w = g[0].length;
    var occupied = new Uint8Array(h * w), out = G.constGrid(h, w, bg);
    function key(o) {
      if (d === "down") return -o.r1;
      if (d === "up") return o.r0;
      if (d === "right") return -o.c1;
      return o.c0;
    }
    var order = objs.slice();
    order.sort(function (a, b) { return key(a) - key(b); });
    var i, o, best, step, ok, it, s, r, c, nr, nc;
    for (i = 0; i < order.length; i++) {
      o = order[i];
      best = 0; step = 1;
      for (;;) {
        ok = true;
        it = o.cells.values(); s = it.next();
        while (!s.done) {
          r = s.value >> 6; c = s.value & 63;
          nr = r + dr * step; nc = c + dc * step;
          if (!(nr >= 0 && nr < h && nc >= 0 && nc < w) || occupied[nr * w + nc]) { ok = false; break; }
          s = it.next();
        }
        if (!ok) break;
        best = step;
        if (once) break;
        step++;
      }
      it = o.cells.values(); s = it.next();
      while (!s.done) {
        r = s.value >> 6; c = s.value & 63;
        nr = r + dr * best; nc = c + dc * best;
        occupied[nr * w + nc] = 1;
        out[nr][nc] = g[r][c];
        s = it.next();
      }
    }
    return out;
  }

  function _rays(g, bg, dirs, stopAtObstacle, colorMode, onlyColor) {
    bg = G.bgOr(g, bg);
    var h = g.length, w = g[0].length, out = G.copyGrid(g), src = [], r, c;
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) if (g[r][c] !== bg) src.push([r, c, g[r][c]]);
    if (!src.length || src.length > 200) return null;
    var i, d, v, dr, dc, nr, nc;
    for (i = 0; i < src.length; i++) {
      r = src[i][0]; c = src[i][1]; v = src[i][2];
      if (onlyColor !== null && onlyColor !== undefined && v !== onlyColor) continue;
      for (d = 0; d < dirs.length; d++) {
        dr = dirs[d][0]; dc = dirs[d][1];
        nr = r + dr; nc = c + dc;
        while (nr >= 0 && nr < h && nc >= 0 && nc < w) {
          if (g[nr][nc] !== bg) {
            if (stopAtObstacle) break;
          } else out[nr][nc] = (colorMode === null || colorMode === undefined) ? v : colorMode;
          nr += dr; nc += dc;
        }
      }
    }
    return out;
  }

  function _connect(g, bg, fillMode, diag, maxGap) {
    bg = G.bgOr(g, bg);
    var h = g.length, w = g[0].length, out = G.copyGrid(g), pts = new Map(), r, c, v;
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) {
      v = g[r][c];
      if (v !== bg) {
        if (!pts.has(v)) pts.set(v, []);
        pts.get(v).push([r, c]);
      }
    }
    var anyDrawn = false;
    pts.forEach(function (ps, val) {
      if (ps.length > 40) return;
      var i, j, r1, c1, r2, c2, cells, k, sr, sc, d, blocked, col;
      for (i = 0; i < ps.length; i++) for (j = i + 1; j < ps.length; j++) {
        r1 = ps[i][0]; c1 = ps[i][1]; r2 = ps[j][0]; c2 = ps[j][1];
        cells = null;
        if (maxGap !== null && maxGap !== undefined) {
          d = Math.max(Math.abs(r2 - r1), Math.abs(c2 - c1)) - 1;
          if (d > maxGap) continue;
        }
        if (r1 === r2 && Math.abs(c2 - c1) > 1) {
          cells = [];
          for (k = Math.min(c1, c2) + 1; k < Math.max(c1, c2); k++) cells.push([r1, k]);
        } else if (c1 === c2 && Math.abs(r2 - r1) > 1) {
          cells = [];
          for (k = Math.min(r1, r2) + 1; k < Math.max(r1, r2); k++) cells.push([k, c1]);
        } else if (diag && Math.abs(r2 - r1) === Math.abs(c2 - c1) && Math.abs(r2 - r1) > 1) {
          sr = r2 > r1 ? 1 : -1; sc = c2 > c1 ? 1 : -1;
          cells = [];
          for (k = 1; k < Math.abs(r2 - r1); k++) cells.push([r1 + k * sr, c1 + k * sc]);
        }
        if (!cells || !cells.length) continue;
        blocked = false;
        for (k = 0; k < cells.length; k++) if (g[cells[k][0]][cells[k][1]] !== bg) { blocked = true; break; }
        if (blocked) continue;
        col = (fillMode === null || fillMode === undefined) ? val : fillMode;
        for (k = 0; k < cells.length; k++) out[cells[k][0]][cells[k][1]] = col;
        anyDrawn = true;
      }
    });
    return anyDrawn ? out : null;
  }

  function _halo(g, bg, color, diag, replace) {
    bg = G.bgOr(g, bg);
    var h = g.length, w = g[0].length, out = G.copyGrid(g), nb = diag ? G.N8 : G.N4;
    var r, c, d, nr, nc;
    for (r = 0; r < h; r++) for (c = 0; c < w; c++) {
      if (g[r][c] === bg) continue;
      for (d = 0; d < nb.length; d++) {
        nr = r + nb[d][0]; nc = c + nb[d][1];
        if (nr >= 0 && nr < h && nc >= 0 && nc < w && g[nr][nc] === bg)
          out[nr][nc] = (color === null || color === undefined) ? g[r][c] : color;
      }
    }
    if (replace)
      for (r = 0; r < h; r++) for (c = 0; c < w; c++) if (g[r][c] !== bg) out[r][c] = bg;
    return out;
  }

  function _rules(ctx, bg) {
    var res = [], segs = ["c4", "c8", "m8"], i, j, k, seg, d;
    for (i = 0; i < segs.length; i++) {
      seg = segs[i];
      for (j = 0; j < DIR_NAMES.length; j++) {
        d = DIR_NAMES[j];
        res.push(_h("move_" + seg + "_" + d,
                    (function (s, dd) { return function (g) { return _moveObjects(g, s, bg, dd, false); }; })(seg, d), 4.5));
        res.push(_h("step_" + seg + "_" + d,
                    (function (s, dd) { return function (g) { return _moveObjects(g, s, bg, dd, true); }; })(seg, d), 5.5));
      }
    }
    var dv4 = [], dvd = [];
    for (i = 0; i < DIR_NAMES.length; i++) dv4.push(DIRS[DIR_NAMES[i]]);
    for (i = 0; i < DIAG_NAMES.length; i++) dvd.push(DIAG[DIAG_NAMES[i]]);
    var dirsets = [["4", dv4], ["d", dvd], ["8", dv4.concat(dvd)]];
    for (i = 0; i < dirsets.length; i++) {
      var stops = [true, false];
      for (j = 0; j < 2; j++)
        res.push(_h("ray" + dirsets[i][0] + (stops[j] ? "_stop" : ""),
                    (function (dv, s) { return function (g) { return _rays(g, bg, dv, s, null); }; })(dirsets[i][1], stops[j]),
                    4.5));
    }
    var singles = [];
    for (i = 0; i < DIR_NAMES.length; i++) singles.push([DIR_NAMES[i], DIRS[DIR_NAMES[i]]]);
    for (i = 0; i < DIAG_NAMES.length; i++) singles.push([DIAG_NAMES[i], DIAG[DIAG_NAMES[i]]]);
    for (i = 0; i < singles.length; i++)
      res.push(_h("ray1_" + singles[i][0],
                  (function (v) { return function (g) { return _rays(g, bg, [v], true, null); }; })(singles[i][1]), 4.5));

    var pal = G.csList(ctx.out_palette()), diags = [false, true], gaps = [null, 1, 2];
    for (i = 0; i < 2; i++) for (j = 0; j < gaps.length; j++) {
      (function (dg, gp) {
        var sfx = (dg ? "_d" : "") + (gp === null ? "" : "_g" + gp);
        res.push(_h("connect" + sfx, function (g) { return _connect(g, bg, null, dg, gp); },
                    gp === null ? 4.0 : 4.4));
        for (k = 0; k < pal.length; k++)
          res.push(_h("connect" + sfx + "#" + pal[k],
                      (function (c) { return function (g) { return _connect(g, bg, c, dg, gp); }; })(pal[k]),
                      gp === null ? 5.0 : 5.4));
      })(diags[i], gaps[j]);
    }
    for (i = 0; i < 2; i++) {
      (function (dg) {
        res.push(_h("halo" + (dg ? "8" : "4"), function (g) { return _halo(g, bg, null, dg, false); }, 4.5));
        var m;
        for (m = 0; m < pal.length; m++) {
          (function (c) {
            res.push(_h("halo" + (dg ? "8" : "4") + "#" + c,
                        function (g) { return _halo(g, bg, c, dg, false); }, 5.0));
            res.push(_h("ring" + (dg ? "8" : "4") + "#" + c,
                        function (g) { return _halo(g, bg, c, dg, true); }, 6.0));
          })(pal[m]);
        }
      })(diags[i]);
    }
    return res;
  }

  function generate(ctx) {
    if (!ctx.same_shape()) return [];
    var res = [], bgs = ctx.bg_varies() ? [ctx.bg(), null] : [ctx.bg()], i;
    for (i = 0; i < bgs.length; i++) res = res.concat(_rules(ctx, bgs[i]));
    return res;
  }

  SEQ.halo = _halo;
  SEQ.connect = _connect;
  SEQ.rays = _rays;
  SEQ.moveObjects = _moveObjects;
  SEQ.DIRS = DIRS;
  SEQ.DIAG = DIAG;
  SEQ.DIR_NAMES = DIR_NAMES;
  SEQ.DIAG_NAMES = DIAG_NAMES;
  HOOKS.halo = _halo;
  HOOKS.connect = _connect;

  defSolver("sequence", "sequence", generate);
})();
