/* Port of engine/solvers/objwise.py -- solve one object, apply it to all.
 *
 * The object analogue of panelwise. A task with three grids and five objects
 * each gives the sub-task fifteen pairs instead of three, which is often the
 * difference between a rule that looks like memorisation and one that is not.
 */

(function () {
  var _SEGS = ["c8", "m8", "c4", "m4"];

  function _objwiseRule(seg, bg, hyp) {
    return function (g) {
      var b = G.bgOr(g, bg);
      var objs = O.segment(g, seg, b);
      if (!objs.length || objs.length > 40) return null;
      var out = G.copyGrid(g), order = objs.slice(), i, o, patch, q, r, c;
      order.sort(function (x, y) { return (x.r0 - y.r0) || (x.c0 - y.c0); });
      for (i = 0; i < order.length; i++) {
        o = order[i];
        patch = G.subgrid(g, o.r0, o.c0, o.r1, o.c1);
        if (patch === null) return null;
        q = hyp.fn(patch);
        if (q === null || q === undefined || !G.valid(q) ||
            q.length !== patch.length || q[0].length !== patch[0].length) return null;
        for (r = 0; r < o.height(); r++)
          for (c = 0; c < o.width(); c++) out[o.r0 + r][o.c0 + c] = q[r][c];
      }
      return out;
    };
  }

  function _modules() {
    var names = ["geometry", "colormap", "symmetry", "tiling", "cellwise",
                 "objects_map", "selfstamp", "extend"], out = [], i, m;
    for (i = 0; i < names.length; i++) { m = moduleByName(names[i]); if (m) out.push(m); }
    return out;
  }

  function _one(ctx, seg, bg, deadline) {
    var res = [], pairs = [], changed = false, i, t, a, b, abg, objs, o, pa, pb, tins = [];
    try {
      for (t = 0; t < ctx.train.length; t++) {
        a = ctx.train[t][0]; b = ctx.train[t][1];
        abg = G.bgOr(a, bg);
        objs = O.segment(a, seg, abg);
        if (!objs.length || objs.length > 40) return res;
        for (i = 0; i < objs.length; i++) {
          o = objs[i];
          pa = G.subgrid(a, o.r0, o.c0, o.r1, o.c1);
          pb = G.subgrid(b, o.r0, o.c0, o.r1, o.c1);
          if (pa === null || pb === null) return res;
          if (!G.gEq(pa, pb)) changed = true;
          pairs.push([pa, pb]);
        }
      }
      if (!changed || pairs.length < 3 || pairs.length > 120) return res;
      for (t = 0; t < ctx.test_inputs.length; t++) {
        var tg = ctx.test_inputs[t], tbg = G.bgOr(tg, bg);
        objs = O.segment(tg, seg, tbg);
        for (i = 0; i < objs.length; i++) {
          o = objs[i];
          var p = G.subgrid(tg, o.r0, o.c0, o.r1, o.c1);
          if (p === null) return res;
          tins.push(p);
        }
      }
      if (!tins.length) return res;
    } catch (e) { return res; }

    var sub = new Ctx(pairs, tins.slice(0, 8));
    sub.deadline = Math.min(deadline, nowMs() + 1000);
    var mods = _modules(), m, hyps, hp;
    for (m = 0; m < mods.length; m++) {
      if (nowMs() > sub.deadline) break;
      try { hyps = hypcacheGenerate(mods[m], sub); } catch (e) { continue; }
      for (i = 0; i < hyps.length; i++) {
        hp = hyps[i];
        if (hp.fits(sub.train)) {
          res.push(new Hyp("objwise[" + seg + "]>>" + hp.name,
                           _objwiseRule(seg, bg, hp), 3.5 + hp.cost, "objects"));
          break;
        }
      }
      if (res.length > 10) break;
    }
    return res;
  }

  function generate(ctx) {
    if (!ctx.same_shape()) return [];
    var deadline = ctx.deadline === null ? (nowMs() + 3000) : ctx.deadline;
    var res = [], bgs = ctx.bg_varies() ? [ctx.bg(), null] : [ctx.bg()], i, j;
    for (i = 0; i < bgs.length; i++)
      for (j = 0; j < _SEGS.length; j++) {
        if (nowMs() > deadline || res.length > 10) break;
        res = res.concat(_one(ctx, _SEGS[j], bgs[i], deadline));
      }
    return res;
  }

  defSolver("objwise", "objects", generate, 2);
})();
