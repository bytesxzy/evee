/* Port of engine/solvers/select.py -- answer = one object, chosen relationally.
 *
 * "Find the odd one out / the biggest / the one that is different and show it"
 * is a large ARC family whose space factorises cleanly into
 * (segmentation) x (selector) x (rendering), so the product is enumerated
 * rather than one rule written per task.
 */

(function () {
  var _h = mkHyp("select");
  var _SEGS = ["c4", "c8", "m4", "m8", "color"];

  function _renderPatch(grid, o, bg) { return o.filled(bg); }
  function _renderCrop(grid, o, bg) { return G.subgrid(grid, o.r0, o.c0, o.r1, o.c1); }
  function _renderMask(grid, o, bg) {
    var m = o.mask(), out = [], r, c, row;
    for (r = 0; r < m.length; r++) {
      row = new Array(m[r].length);
      for (c = 0; c < m[r].length; c++) row[c] = m[r][c] ? o.color : bg;
      out.push(row);
    }
    return out;
  }
  function _renderSolid(grid, o, bg) { return G.constGrid(o.height(), o.width(), o.color); }
  function _renderCell(grid, o, bg) { return [[o.color]]; }
  function _renderIsolate(grid, o, bg) {
    var out = G.constGrid(grid.length, grid[0].length, bg), it = o.cells.values(), s = it.next();
    while (!s.done) { out[s.value >> 6][s.value & 63] = grid[s.value >> 6][s.value & 63]; s = it.next(); }
    return out;
  }
  function _renderDelete(grid, o, bg) {
    var out = G.copyGrid(grid), it = o.cells.values(), s = it.next();
    while (!s.done) { out[s.value >> 6][s.value & 63] = bg; s = it.next(); }
    return out;
  }

  var RENDERERS = [
    ["patch", _renderPatch, 0.0],
    ["crop", _renderCrop, 0.2],
    ["mask", _renderMask, 1.0],
    ["solid", _renderSolid, 1.5],
    ["cell", _renderCell, 1.5],
    ["isolate", _renderIsolate, 1.0],
    ["delete", _renderDelete, 1.0]
  ];

  function _pick(grid, seg, sel, bg) {
    var objs = O.segment(grid, seg, bg);
    if (!objs.length || objs.length > 200) return null;
    return sel(objs);
  }

  var _OVERLAY_OPS = ["or", "and", "xor", "mode", "first", "last"];

  /* Superimpose every object of equal bounding-box size: "stack the shapes and
     report what they have in common" is invisible to the panel-logic solver
     because the pieces are scattered objects rather than lattice panels. */
  function _overlayObjects(g, seg, op, bg, useMask) {
    bg = G.bgOr(g, bg);
    var objs = O.segment(g, seg, bg);
    if (objs.length < 2 || objs.length > 30) return null;
    var h = objs[0].height(), w = objs[0].width(), i;
    for (i = 1; i < objs.length; i++) if (objs[i].height() !== h || objs[i].width() !== w) return null;
    if (h * w > 400) return null;
    var patches = [], p, r, c, row;
    for (i = 0; i < objs.length; i++) {
      p = objs[i].filled(bg);
      if (useMask) {
        var mp = [];
        for (r = 0; r < p.length; r++) {
          row = new Array(p[r].length);
          for (c = 0; c < p[r].length; c++) row[c] = p[r][c] !== bg ? 1 : bg;
          mp.push(row);
        }
        p = mp;
      }
      patches.push(p);
    }
    var out = [], vals, live, cnt, order, bestV, bestN;
    for (r = 0; r < h; r++) {
      row = new Array(w);
      for (c = 0; c < w; c++) {
        vals = []; live = [];
        for (i = 0; i < patches.length; i++) {
          vals.push(patches[i][r][c]);
          if (patches[i][r][c] !== bg) live.push(patches[i][r][c]);
        }
        if (op === "or") row[c] = live.length ? live[0] : bg;
        else if (op === "and") row[c] = live.length === vals.length ? live[0] : bg;
        else if (op === "xor") row[c] = live.length === 1 ? live[0] : bg;
        else if (op === "mode") {
          cnt = new Map(); order = [];
          for (i = 0; i < vals.length; i++) {
            if (!cnt.has(vals[i])) { cnt.set(vals[i], 0); order.push(vals[i]); }
            cnt.set(vals[i], cnt.get(vals[i]) + 1);
          }
          bestV = order[0]; bestN = cnt.get(order[0]);
          for (i = 1; i < order.length; i++) if (cnt.get(order[i]) > bestN) { bestN = cnt.get(order[i]); bestV = order[i]; }
          row[c] = bestV;
        } else if (op === "first") row[c] = vals[0];
        else row[c] = vals[vals.length - 1];
      }
      out.push(row);
    }
    return out;
  }

  function _apply(g, seg, sel, rend, bg) {
    var b = G.bgOr(g, bg);
    var o = _pick(g, seg, sel, b);
    if (o === null) return null;
    return rend(g, o, b);
  }

  function _nth(g, seg, key, idx, bg) {
    var b = G.bgOr(g, bg);
    var objs = O.segment(g, seg, b);
    if (!objs.length || objs.length > 200) return null;
    var f = O.OBJ_RANKERS[key], sorted = objs.slice();
    sorted.sort(function (x, y) { return f(y) - f(x); });
    var i = idx < 0 ? sorted.length + idx : idx;
    if (i < 0 || i >= sorted.length) return null;
    return sorted[i].filled(b);
  }

  function _rules(ctx, bg) {
    var res = [], segs1 = ["c8", "m8", "c4"], s, o, m, i, j;
    for (s = 0; s < segs1.length; s++)
      for (o = 0; o < _OVERLAY_OPS.length; o++) {
        var ums = [false, true];
        for (m = 0; m < 2; m++)
          res.push(_h("overlay_" + segs1[s] + "_" + _OVERLAY_OPS[o] + (ums[m] ? "_m" : ""),
                      (function (sg, op, um) { return function (g) { return _overlayObjects(g, sg, op, bg, um); }; })(segs1[s], _OVERLAY_OPS[o], ums[m]),
                      4.5));
      }
    for (s = 0; s < _SEGS.length; s++)
      for (i = 0; i < O.SELECTORS.length; i++)
        for (j = 0; j < RENDERERS.length; j++)
          res.push(_h(_SEGS[s] + "." + O.SELECTORS[i][0] + "." + RENDERERS[j][0],
                      (function (sg, sel, rend) { return function (g) { return _apply(g, sg, sel, rend, bg); }; })(_SEGS[s], O.SELECTORS[i][1], RENDERERS[j][1]),
                      3.0 + RENDERERS[j][2]));
    var segs2 = ["c4", "c8", "m8"], keys = ["size", "bbox_area", "top", "left"], idxs = [0, 1, -1];
    for (s = 0; s < segs2.length; s++)
      for (i = 0; i < keys.length; i++)
        for (j = 0; j < idxs.length; j++)
          res.push(_h(segs2[s] + ".nth_" + keys[i] + sgn(idxs[j]),
                      (function (sg, ky, ix) { return function (g) { return _nth(g, sg, ky, ix, bg); }; })(segs2[s], keys[i], idxs[j]),
                      5.0));
    return res;
  }

  function generate(ctx) {
    var res = [], bgs = ctx.bg_varies() ? [ctx.bg(), null] : [ctx.bg()], i;
    for (i = 0; i < bgs.length; i++) res = res.concat(_rules(ctx, bgs[i]));
    return res;
  }

  defSolver("select", "select", generate);
})();
