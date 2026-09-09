/* Port of engine/solvers/locate.py -- when the answer is a piece of the input,
 * learn where to cut.
 *
 * Extraction factorises cleanly: the output is a subgrid, so the only thing to
 * infer is the rectangle. This module first finds every place the demonstrated
 * output actually occurs inside its input, which completely determines what a
 * correct rule must return, then asks which locator picks exactly those places
 * in every pair at once. A locator that agrees in three pairs out of four is
 * discarded, not patched.
 */

(function () {
  var _h = mkHyp("select");
  var _SEGS = ["c4", "c8", "m4", "m8", "color", "g2m"];
  var _MAX_HITS = 40;

  /* (name, forward, inverse): the inverse is applied to the demonstrated
     output so the search looks for the piece as it appears in the input. */
  var _VIEWS = [
    ["id", function (g) { return g; }, function (g) { return g; }],
    ["rot90", G.rot90, G.rot270],
    ["rot180", G.rot180, G.rot180],
    ["rot270", G.rot270, G.rot90],
    ["flip_h", G.flipH, G.flipH],
    ["flip_v", G.flipV, G.flipV],
    ["transpose", G.transpose, G.transpose]
  ];

  function _occurrences(g, out) {
    var H = g.length, W = g[0].length, h = out.length, w = out[0].length, hits = [], r, c, i, j, ok;
    if (h > H || w > W) return [];
    for (r = 0; r <= H - h; r++) for (c = 0; c <= W - w; c++) {
      ok = true;
      for (i = 0; i < h && ok; i++)
        for (j = 0; j < w; j++) if (g[r + i][c + j] !== out[i][j]) { ok = false; break; }
      if (ok) {
        hits.push([r, c]);
        if (hits.length > _MAX_HITS) return hits;
      }
    }
    return hits;
  }

  function _locColor(c) {
    return function (g, bg) {
      var cells = [], r, x;
      for (r = 0; r < g.length; r++) for (x = 0; x < g[r].length; x++) if (g[r][x] === c) cells.push(r * 64 + x);
      if (!cells.length) return null;
      return G.bboxOf(cells);
    };
  }

  function _locNonbg(g, bg) {
    var cells = [], r, x;
    for (r = 0; r < g.length; r++) for (x = 0; x < g[r].length; x++) if (g[r][x] !== bg) cells.push(r * 64 + x);
    return cells.length ? G.bboxOf(cells) : null;
  }

  function _pick(objs, how) {
    if (!objs.length) return null;
    var i, best, cnt, cands;
    if (how === "big") {
      best = objs[0];
      for (i = 1; i < objs.length; i++)
        if (objs[i].size() > best.size() ||
            (objs[i].size() === best.size() && (-objs[i].r0 > -best.r0 ||
             (-objs[i].r0 === -best.r0 && -objs[i].c0 > -best.c0)))) best = objs[i];
      return best;
    }
    if (how === "small") {
      best = objs[0];
      for (i = 1; i < objs.length; i++)
        if (objs[i].size() < best.size() ||
            (objs[i].size() === best.size() && (objs[i].r0 < best.r0 ||
             (objs[i].r0 === best.r0 && objs[i].c0 < best.c0)))) best = objs[i];
      return best;
    }
    if (how === "bigbox") {
      best = objs[0];
      for (i = 1; i < objs.length; i++)
        if (objs[i].bbox_area() > best.bbox_area() ||
            (objs[i].bbox_area() === best.bbox_area() && (-objs[i].r0 > -best.r0 ||
             (-objs[i].r0 === -best.r0 && -objs[i].c0 > -best.c0)))) best = objs[i];
      return best;
    }
    if (how === "ucol") {
      cnt = O.countBy(objs, function (o) { return o.color; });
      cands = [];
      for (i = 0; i < objs.length; i++) if (cnt.get(objs[i].color) === 1) cands.push(objs[i]);
      return cands.length === 1 ? cands[0] : null;
    }
    if (how === "ushp") {
      cnt = O.countBy(objs, function (o) { return o.norm_key(); });
      cands = [];
      for (i = 0; i < objs.length; i++) if (cnt.get(objs[i].norm_key()) === 1) cands.push(objs[i]);
      return cands.length === 1 ? cands[0] : null;
    }
    if (how === "holes") {
      cands = [];
      for (i = 0; i < objs.length; i++) if (objs[i].holes_count() > 0) cands.push(objs[i]);
      if (!cands.length) return null;
      best = cands[0];
      for (i = 1; i < cands.length; i++)
        if (cands[i].holes_count() > best.holes_count() ||
            (cands[i].holes_count() === best.holes_count() && cands[i].size() > best.size())) best = cands[i];
      return best;
    }
    if (how === "dense") {
      best = objs[0];
      for (i = 1; i < objs.length; i++) {
        var ci = G.csSize(objs[i].colors()), cb = G.csSize(best.colors());
        if (ci > cb || (ci === cb && objs[i].size() > best.size())) best = objs[i];
      }
      return best;
    }
    return null;
  }

  function _locObj(seg, how, inner) {
    return function (g, bg) {
      var objs = O.segment(g, seg, bg);
      if (!objs.length || objs.length > 60) return null;
      var o = _pick(objs, how);
      if (o === null) return null;
      if (inner) {
        if (o.r1 - o.r0 < 2 || o.c1 - o.c0 < 2) return null;
        return [o.r0 + 1, o.c0 + 1, o.r1 - 1, o.c1 - 1];
      }
      return [o.r0, o.c0, o.r1, o.c1];
    };
  }

  function _scoreWindow(g, r, c, h, w, kind, bg) {
    var i, j, n = 0, m = 0;
    if (kind === "density") {
      for (i = 0; i < h; i++) for (j = 0; j < w; j++) if (g[r + i][c + j] !== bg) n++;
      return n;
    }
    if (kind === "colors" || kind === "uniform") {
      for (i = 0; i < h; i++) for (j = 0; j < w; j++) m |= 1 << g[r + i][c + j];
      return kind === "colors" ? G.csSize(m) : -G.csSize(m);
    }
    return 0;
  }

  function _locWindow(shape, kind, wantMax, bgColor) {
    var h = shape[0], w = shape[1];
    return function (g, bg) {
      var H = g.length, W = g[0].length, best = null, arg = null, tie = false, r, c, s, i, j;
      if (h > H || w > W) return null;
      for (r = 0; r <= H - h; r++) for (c = 0; c <= W - w; c++) {
        if (bgColor === null || bgColor === undefined) s = _scoreWindow(g, r, c, h, w, kind, bg);
        else {
          s = 0;
          for (i = 0; i < h; i++) for (j = 0; j < w; j++) if (g[r + i][c + j] === bgColor) s++;
        }
        if (best === null || (wantMax ? s > best : s < best)) { best = s; arg = [r, c]; tie = false; }
        else if (s === best) tie = true;
      }
      if (arg === null || tie) return null;   /* an ambiguous extreme is not a rule */
      return [arg[0], arg[1], arg[0] + h - 1, arg[1] + w - 1];
    };
  }

  function _locators(ctx) {
    var out = [["nonbg", 0.4, _locNonbg]], i, s, hw, inr;
    var pal = G.csList(ctx.in_palette()).slice(0, 9);
    for (i = 0; i < pal.length; i++) out.push(["col" + pal[i], 0.8, _locColor(pal[i])]);
    var hows = ["big", "small", "bigbox", "ucol", "ushp", "holes", "dense"], inners = [false, true];
    for (s = 0; s < _SEGS.length; s++)
      for (hw = 0; hw < hows.length; hw++)
        for (inr = 0; inr < 2; inr++)
          out.push([_SEGS[s] + "." + hows[hw] + (inners[inr] ? ".in" : ""),
                    0.9 + (inners[inr] ? 0.2 : 0.0),
                    _locObj(_SEGS[s], hows[hw], inners[inr])]);
    var shape = ctx.const_out_shape();
    if (shape && shape[0] * shape[1] <= 400) {
      var kinds = ["density", "colors", "uniform"], wants = [true, false], k, wt;
      for (k = 0; k < kinds.length; k++)
        for (wt = 0; wt < 2; wt++)
          out.push(["win." + kinds[k] + "." + (wants[wt] ? "max" : "min"), 1.4,
                    _locWindow(shape, kinds[k], wants[wt])]);
      var pal2 = G.csList(ctx.in_palette()).slice(0, 6);
      for (i = 0; i < pal2.length; i++)
        for (wt = 0; wt < 2; wt++)
          out.push(["win.c" + pal2[i] + "." + (wants[wt] ? "max" : "min"), 1.6,
                    _locWindow(shape, "count", wants[wt], pal2[i])]);
    }
    return out;
  }

  function _cut(loc, bg, xform) {
    return function (g) {
      var box = loc(g, G.bgOr(g, bg));
      if (box === null) return null;
      var H = g.length, W = g[0].length;
      if (!(box[0] >= 0 && box[0] <= box[2] && box[2] < H && box[1] >= 0 && box[1] <= box[3] && box[3] < W)) return null;
      var cut = G.subgrid(g, box[0], box[1], box[2], box[3]);
      if (cut === null || xform === null) return cut;
      return xform(cut);
    };
  }

  function generate(ctx) {
    if (ctx.same_shape()) return [];
    var i, t;
    for (t = 0; t < ctx.train.length; t++)
      if (G.area(ctx.train[t][1]) >= G.area(ctx.train[t][0])) return [];
    /* The answer is often a transformed piece, so look for each dihedral image
       of the output as well; the cut is then followed by that transform. */
    var views = [], v, targets, ok, pre, hits, h, w, set2, j;
    for (v = 0; v < _VIEWS.length; v++) {
      targets = []; ok = true;
      for (t = 0; t < ctx.train.length; t++) {
        pre = _VIEWS[v][2](ctx.train[t][1]);
        hits = _occurrences(ctx.train[t][0], pre);
        if (!hits.length) { ok = false; break; }
        h = pre.length; w = pre[0].length;
        set2 = new Set();
        for (j = 0; j < hits.length; j++)
          set2.add(hits[j][0] + "," + hits[j][1] + "," + (hits[j][0] + h - 1) + "," + (hits[j][1] + w - 1));
        targets.push(set2);
      }
      if (ok) views.push([_VIEWS[v][0], _VIEWS[v][1], targets]);
    }
    if (!views.length) return [];   /* not an extraction task at all */
    var bg = ctx.bg(), res = [], seen = new Set(), locs = _locators(ctx), L, box;
    for (v = 0; v < views.length; v++) {
      for (L = 0; L < locs.length; L++) {
        if (ctx.timed_out() || res.length >= 14) break;
        ok = true;
        for (t = 0; t < ctx.train.length; t++) {
          try { box = locs[L][2](ctx.train[t][0], G.bgOr(ctx.train[t][0], bg)); } catch (e) { box = null; }
          if (box === null || !views[v][2][t].has(box.join(","))) { ok = false; break; }
        }
        if (!ok) continue;
        var hp = _h("cut[" + locs[L][0] + (views[v][0] === "id" ? "" : ">" + views[v][0]) + "]",
                    _cut(locs[L][2], bg, views[v][0] === "id" ? null : views[v][1]),
                    2.6 + locs[L][1] + (views[v][0] === "id" ? 0.0 : 0.5));
        if (!hp.fits(ctx.train)) continue;
        var parts = [], p;
        try {
          for (i = 0; i < ctx.test_inputs.length; i++) {
            p = hp.apply(ctx.test_inputs[i]);
            parts.push(p === null ? "*" : G.gkey(p));
          }
        } catch (e) { continue; }
        var sig = parts.join("~");
        if (seen.has(sig)) continue;
        seen.add(sig);
        res.push(hp);
      }
    }
    return res;
  }

  defSolver("locate", "select", generate);
})();
