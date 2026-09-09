/* Port of engine/solvers/rewrite.py -- re-pose the task, then reuse the whole
 * specialist portfolio.
 *
 * Many tasks are a known transformation wearing a disguise: the real rule
 * applies to the cropped grid, or to the grid with its background rows
 * removed, or the answer is an upscale of a rule at the original resolution.
 * Rather than duplicating every specialist for every disguise, the task is
 * rewritten and handed back to the specialists.
 */

(function () {
  function _crop(g, bg) { return G.cropToContent(g, bg); }

  function _compress(g, bg) {
    var rows = [], r, c, any;
    for (r = 0; r < g.length; r++) {
      any = false;
      for (c = 0; c < g[r].length; c++) if (g[r][c] !== bg) { any = true; break; }
      if (any) rows.push(g[r]);
    }
    if (!rows.length) return null;
    var t = G.transpose(rows), cols = [];
    for (r = 0; r < t.length; r++) {
      any = false;
      for (c = 0; c < t[r].length; c++) if (t[r][c] !== bg) { any = true; break; }
      if (any) cols.push(t[r]);
    }
    if (!cols.length) return null;
    return G.transpose(cols);
  }

  function _largest(g, bg) {
    var objs = O.segment(g, "c8", bg);
    if (!objs.length) return null;
    var o = O.selectExtreme(objs, "size", true);
    return o === null ? null : G.subgrid(g, o.r0, o.c0, o.r1, o.c1);
  }

  /* Remove full uniform rows/cols (separator scaffolding). */
  function _stripLines(g, bg) {
    var h = g.length, w = g[0].length, keepR = [], r, c, uni;
    for (r = 0; r < h; r++) {
      uni = true;
      for (c = 1; c < w; c++) if (g[r][c] !== g[r][0]) { uni = false; break; }
      if (!uni) keepR.push(g[r]);
    }
    if (!keepR.length || keepR.length === h) return null;
    var tt = G.transpose(keepR), keepC = [];
    for (r = 0; r < tt.length; r++) {
      uni = true;
      for (c = 1; c < tt[r].length; c++) if (tt[r][c] !== tt[r][0]) { uni = false; break; }
      if (!uni) keepC.push(tt[r]);
    }
    if (!keepC.length) return null;
    return G.transpose(keepC);
  }

  function _inRewrites(ctx) {
    var bg = ctx.bg();
    return [
      ["crop", function (g) { return _crop(g, bg); }],
      ["compress", function (g) { return _compress(g, bg); }],
      ["largest", function (g) { return _largest(g, bg); }],
      ["dedup", G.dedup],
      ["strip", function (g) { return _stripLines(g, bg); }]
    ];
  }

  function _outRewrites(ctx) {
    var out = [], r = ctx.shape_ratio(), i, ok;
    if (r && !(r[0] === 1 && r[1] === 1)) {
      var ky = r[0], kx = r[1];
      ok = true;
      for (i = 0; i < ctx.train.length; i++)
        if (G.downscale(ctx.train[i][1], ky, kx) === null) { ok = false; break; }
      if (ok) out.push(["down" + ky + "x" + kx,
                        function (g) { return G.downscale(g, ky, kx); },
                        function (g) { return G.upscale(g, ky, kx); }]);
    }
    return out;
  }

  /* A colour permutation derived from this grid alone: background first, then
     colours by how much of the grid they cover. Tasks that use a different
     palette in every example are structurally one task. */
  function _rankPerm(g, bg) {
    var hist = G.histogram(g), items = [], c;
    for (c = 0; c < G.NCOLORS; c++) if (c !== bg && hist[c] > 0) items.push([c, hist[c]]);
    items.sort(function (a, b) { return (b[1] - a[1]) || (a[0] - b[0]); });
    var order = [bg], i;
    for (i = 0; i < items.length; i++) order.push(items[i][0]);
    for (c = 0; c < 10; c++) if (order.indexOf(c) < 0) order.push(c);
    var perm = new Map();
    for (i = 0; i < order.length; i++) perm.set(order[i], i);
    return perm;
  }

  function _invert(perm) {
    var inv = new Map();
    perm.forEach(function (v, k) { inv.set(v, k); });
    return inv;
  }

  function _pairRewrites(ctx) {
    var bg = ctx.bg(), all = ctx.all_inputs(), pals = new Set(), i;
    for (i = 0; i < all.length; i++) pals.add(G.csDiff(G.palette(all[i]), 1 << bg));
    if (pals.size < 2) return [];   /* one palette: normalising buys nothing */
    return [["rankpal", function (g) { return _rankPerm(g, bg); }]];
  }

  function _modules() {
    var names = ["geometry", "colormap", "symmetry", "partition", "tiling", "blocks",
                 "regions", "cellwise", "objects_map", "substitute", "sequence",
                 "paint", "select", "compose", "paneltable", "selfstamp", "extend"];
    var out = [], i, m;
    for (i = 0; i < names.length; i++) { m = moduleByName(names[i]); if (m) out.push(m); }
    return out;
  }

  function _safe(f, g) {
    var r;
    try { r = f(g); } catch (e) { return null; }
    return (r !== null && r !== undefined && G.valid(r)) ? r : null;
  }

  function _chainIn(T, hp) {
    return function (g) {
      var t = _safe(T, g);
      return t === null ? null : hp.fn(t);
    };
  }

  function _chainGeo(fwd, inv, hp) {
    return function (g) {
      var r = hp.fn(fwd(g));
      return (r === null || r === undefined) ? null : inv(r);
    };
  }

  function _chainPair(permf, hp) {
    return function (g) {
      var p = permf(g);
      if (p === null) return null;
      var r = hp.fn(G.applyCmap(g, p));
      return (r === null || r === undefined) ? null : G.applyCmap(r, _invert(p));
    };
  }

  function _chainOut(inv, hp) {
    return function (g) {
      var r = hp.fn(g);
      return (r === null || r === undefined) ? null : inv(r);
    };
  }

  function generate(ctx) {
    var res = [], deadline = ctx.deadline === null ? (nowMs() + 6000) : ctx.deadline;
    var mods = _modules(), i, j, m, hyps, hp;
    /* Share the module's slice across its variants instead of giving the first
       one a fixed budget and letting the rest run out of clock. */
    var inR = _inRewrites(ctx), pairR = _pairRewrites(ctx);
    var nVar = Math.max(1, inR.length + pairR.length + 3);
    var sliceMs = Math.max(350, (deadline - nowMs()) / nVar);

    for (i = 0; i < inR.length; i++) {
      if (nowMs() > deadline) break;
      var T = inR[i][1], pairs = [], ok = true, ta, t;
      for (t = 0; t < ctx.train.length; t++) {
        ta = _safe(T, ctx.train[t][0]);
        if (ta === null || G.gEq(ta, ctx.train[t][0])) { ok = false; break; }
        pairs.push([ta, ctx.train[t][1]]);
      }
      if (!ok) continue;
      var tins = [], bad = false;
      for (t = 0; t < ctx.test_inputs.length; t++) {
        var tt = _safe(T, ctx.test_inputs[t]);
        if (tt === null) { bad = true; break; }
        tins.push(tt);
      }
      if (bad) continue;
      var sub = new Ctx(pairs, tins);
      sub.deadline = Math.min(deadline, nowMs() + sliceMs);
      for (m = 0; m < mods.length; m++) {
        if (nowMs() > deadline) break;
        try { hyps = hypcacheGenerate(mods[m], sub); } catch (e) { continue; }
        for (j = 0; j < hyps.length; j++) {
          hp = hyps[j];
          if (hp.fits(sub.train)) {
            res.push(new Hyp(inR[i][0] + "|" + hp.name, _chainIn(T, hp), 3.0 + hp.cost, "compose"));
            if (res.length > 60) break;
          }
        }
        if (res.length > 60) break;
      }
    }

    for (i = 0; i < pairR.length; i++) {
      if (nowMs() > deadline) break;
      var permf = pairR[i][1], p2 = [], ok2 = true, p, t2;
      for (t2 = 0; t2 < ctx.train.length; t2++) {
        p = permf(ctx.train[t2][0]);
        if (p === null) { ok2 = false; break; }
        p2.push([G.applyCmap(ctx.train[t2][0], p), G.applyCmap(ctx.train[t2][1], p)]);
      }
      if (!ok2) continue;
      var ti = [], bad2 = false;
      for (t2 = 0; t2 < ctx.test_inputs.length; t2++) {
        p = permf(ctx.test_inputs[t2]);
        if (p === null) { bad2 = true; break; }
        ti.push(G.applyCmap(ctx.test_inputs[t2], p));
      }
      if (bad2) continue;
      var sub2 = new Ctx(p2, ti);
      sub2.deadline = Math.min(deadline, nowMs() + sliceMs);
      for (m = 0; m < mods.length; m++) {
        if (nowMs() > deadline) break;
        try { hyps = hypcacheGenerate(mods[m], sub2); } catch (e) { continue; }
        for (j = 0; j < hyps.length; j++) {
          hp = hyps[j];
          if (hp.fits(sub2.train)) {
            res.push(new Hyp(pairR[i][0] + "~" + hp.name, _chainPair(permf, hp), 3.5 + hp.cost, "compose"));
            break;
          }
        }
      }
    }

    /* Several families are row-biased by construction; solving the transposed
       task and undoing the transpose gives each its column-wise twin free. */
    var geos = [["T", G.transpose, G.transpose], ["R", G.rot90, G.rot270]], gi;
    for (gi = 0; gi < geos.length; gi++) {
      if (nowMs() > deadline) break;
      var gp = [], gt = [], t3;
      for (t3 = 0; t3 < ctx.train.length; t3++)
        gp.push([geos[gi][1](ctx.train[t3][0]), geos[gi][1](ctx.train[t3][1])]);
      for (t3 = 0; t3 < ctx.test_inputs.length; t3++) gt.push(geos[gi][1](ctx.test_inputs[t3]));
      var sub3 = new Ctx(gp, gt);
      sub3.deadline = Math.min(deadline, nowMs() + sliceMs);
      for (m = 0; m < mods.length; m++) {
        if (nowMs() > deadline) break;
        try { hyps = hypcacheGenerate(mods[m], sub3); } catch (e) { continue; }
        for (j = 0; j < hyps.length; j++) {
          hp = hyps[j];
          if (hp.fits(sub3.train)) {
            res.push(new Hyp(geos[gi][0] + "@" + hp.name,
                             _chainGeo(geos[gi][1], geos[gi][2], hp), 3.5 + hp.cost, "compose"));
            break;
          }
        }
      }
    }

    var outR = _outRewrites(ctx), oi;
    for (oi = 0; oi < outR.length; oi++) {
      if (nowMs() > deadline) break;
      var op = [], ok3 = true, tb, t4;
      for (t4 = 0; t4 < ctx.train.length; t4++) {
        tb = _safe(outR[oi][1], ctx.train[t4][1]);
        if (tb === null) { ok3 = false; break; }
        op.push([ctx.train[t4][0], tb]);
      }
      if (!ok3) continue;
      var sub4 = new Ctx(op, ctx.test_inputs);
      sub4.deadline = Math.min(deadline, nowMs() + sliceMs);
      for (m = 0; m < mods.length; m++) {
        if (nowMs() > deadline) break;
        try { hyps = hypcacheGenerate(mods[m], sub4); } catch (e) { continue; }
        for (j = 0; j < hyps.length; j++) {
          hp = hyps[j];
          if (hp.fits(sub4.train)) {
            res.push(new Hyp(outR[oi][0] + "^" + hp.name, _chainOut(outR[oi][2], hp), 3.0 + hp.cost, "compose"));
            if (res.length > 90) break;
          }
        }
      }
    }
    return res;
  }

  defSolver("rewrite", "compose", generate, 2);
})();
