/* Port of engine/solvers/analogy.py -- in-grid analogy.
 *
 * The grid itself contains the demonstration: a fully drawn motif somewhere
 * plus fragments of the same motif elsewhere. Take the richest object as the
 * exemplar, then for every other object search for a placement of some
 * dihedral image (optionally upscaled) consistent with every cell the fragment
 * already has. A placement is accepted only if it is unique and conflicts with
 * nothing, so ambiguous fragments are left alone rather than guessed at.
 */

var ANALOGY = {};

(function () {
  var _h = mkHyp("objects");
  var MAX_OBJ = 30;

  function _patchCells(patch) {
    var out = [], r, c, row;
    for (r = 0; r < patch.length; r++) {
      row = patch[r];
      for (c = 0; c < row.length; c++) if (row[c] !== null) out.push([r, c, row[c]]);
    }
    return out;
  }

  function _scalePatch(patch, k) {
    if (k === 1) return patch;
    var out = [], r, c, i, row, nr;
    for (r = 0; r < patch.length; r++) {
      row = patch[r]; nr = [];
      for (c = 0; c < row.length; c++) for (i = 0; i < k; i++) nr.push(row[c]);
      for (i = 0; i < k; i++) out.push(nr.slice());
    }
    return out;
  }

  function _patchKey(p) {
    var parts = [], r, c, row, vals;
    for (r = 0; r < p.length; r++) {
      row = p[r]; vals = [];
      for (c = 0; c < row.length; c++) vals.push(row[c] === null ? "n" : row[c]);
      parts.push(vals.join(","));
    }
    return parts.join("|");
  }

  /* Offsets where ``patch`` explains every cell of ``frag`` without clash. */
  function _placements(grid, patch, frag, bg, h, w) {
    var cells = _patchCells(patch);
    if (!cells.length) return [];
    var ph = patch.length, pw = patch[0].length;
    var fragCells = [], it = frag.cells.values(), s = it.next();
    while (!s.done) { fragCells.push(s.value); s = it.next(); }
    fragCells.sort(function (a, b) { return a - b; });
    var fr0 = fragCells[0] >> 6, fc0 = fragCells[0] & 63;
    var fcolor = grid[fr0][fc0], out = [], i, j, dr, dc, ok, gr, gc, cur, r, c, pv2;
    for (i = 0; i < cells.length; i++) {
      if (cells[i][2] !== fcolor) continue;
      dr = fr0 - cells[i][0]; dc = fc0 - cells[i][1];
      if (dr < 0 || dc < 0 || dr + ph > h || dc + pw > w) continue;
      ok = true;
      for (j = 0; j < cells.length; j++) {
        gr = cells[j][0] + dr; gc = cells[j][1] + dc;
        cur = grid[gr][gc];
        if (cur !== bg && cur !== cells[j][2]) { ok = false; break; }
      }
      if (!ok) continue;
      for (j = 0; j < fragCells.length; j++) {
        r = fragCells[j] >> 6; c = fragCells[j] & 63;
        pv2 = (r - dr >= 0 && r - dr < ph && c - dc >= 0 && c - dc < pw) ? patch[r - dr][c - dc] : null;
        if (pv2 === null || pv2 !== grid[r][c]) { ok = false; break; }
      }
      if (ok) {
        out.push([dr, dc]);
        if (out.length > 2) return out;
      }
    }
    return out;
  }

  function _complete(grid, bg, seg, pick, scales) {
    bg = G.bgOr(grid, bg);
    var objs = O.segment(grid, seg, bg);
    if (objs.length < 2 || objs.length > MAX_OBJ) return null;
    var ex = objs[0], i, k, o;
    function better(a, b) {
      if (pick === "size") return a.size() > b.size() || (a.size() === b.size() && a.bbox_area() > b.bbox_area());
      if (pick === "colors") {
        var ca = G.csSize(a.colors()), cb = G.csSize(b.colors());
        return ca > cb || (ca === cb && a.size() > b.size());
      }
      return a.bbox_area() > b.bbox_area() || (a.bbox_area() === b.bbox_area() && a.size() > b.size());
    }
    for (i = 1; i < objs.length; i++) if (better(objs[i], ex)) ex = objs[i];
    var base = ex.patch(), h = grid.length, w = grid[0].length;
    var out = G.copyGrid(grid), changed = false, variants = [], p, sp, d;
    for (d = 0; d < G.DIHEDRAL.length; d++) {
      try { p = G.DIHEDRAL[d][1](base); } catch (e) { continue; }
      for (k = 0; k < scales.length; k++) {
        sp = _scalePatch(p, scales[k]);
        if (sp.length <= h && sp[0].length <= w) variants.push(sp);
      }
    }
    var seen = new Set(), uniq = [], key;
    for (i = 0; i < variants.length; i++) {
      key = _patchKey(variants[i]);
      if (!seen.has(key)) { seen.add(key); uniq.push(variants[i]); }
    }
    var best, nhit, pl, j, cells2;
    for (i = 0; i < objs.length; i++) {
      o = objs[i];
      if (o === ex) continue;
      best = null; nhit = 0;
      for (j = 0; j < uniq.length; j++) {
        pl = _placements(grid, uniq[j], o, bg, h, w);
        if (pl.length === 1) {
          nhit++;
          best = [uniq[j], pl[0]];
          if (nhit > 1) break;
        }
      }
      if (nhit !== 1 || best === null) continue;
      cells2 = _patchCells(best[0]);
      for (j = 0; j < cells2.length; j++) {
        var gr2 = cells2[j][0] + best[1][0], gc2 = cells2[j][1] + best[1][1];
        if (out[gr2][gc2] === bg) { out[gr2][gc2] = cells2[j][2]; changed = true; }
      }
    }
    return changed ? out : null;
  }

  function _rules(ctx, bg) {
    var res = [], all = ctx.all_inputs(), maxArea = 0, i;
    for (i = 0; i < all.length; i++) maxArea = Math.max(maxArea, G.area(all[i]));
    var scaleSets = [[1]];
    if (maxArea >= 100) scaleSets.push([1, 2, 3]);
    var segs = ["m8", "m4", "c8"], picks = ["size", "colors", "bbox"], s, p, sc;
    for (s = 0; s < segs.length; s++)
      for (p = 0; p < picks.length; p++)
        for (sc = 0; sc < scaleSets.length; sc++)
          res.push(_h("analogy_" + segs[s] + "_" + picks[p] + "_s" + scaleSets[sc].length,
                      (function (sg, pk, scl) {
                        return function (g) { return _complete(g, bg, sg, pk, scl); };
                      })(segs[s], picks[p], scaleSets[sc]),
                      6.0 + 0.5 * scaleSets[sc].length));
    return res;
  }

  function generate(ctx) {
    if (!ctx.same_shape()) return [];
    var res = [], bgs = ctx.bg_varies() ? [ctx.bg(), null] : [ctx.bg()], i;
    for (i = 0; i < bgs.length; i++) res = res.concat(_rules(ctx, bgs[i]));
    return res;
  }

  ANALOGY.scalePatch = _scalePatch;
  ANALOGY.placements = _placements;

  defSolver("analogy", "objects", generate);
})();
