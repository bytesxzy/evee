/* Port of engine/solvers/panelwise.py -- solve one panel, apply it to all.
 *
 * Complementary to panelabs: that one asks what happens between panels, this
 * one asks what happens inside each. A task with three training grids and a
 * 3x3 lattice supplies twenty-seven training pairs to the sub-task instead of
 * three, which is exactly the regime where the capacity guards elsewhere in
 * the engine stop rejecting things.
 */

(function () {
  /* Swap this grid's separator colour with a canonical slot: a lattice task
     often draws the same rule in a different colour in every grid. The swap is
     an involution, so undoing it after rendering is the same operation. */
  function _normMap(g, k) {
    var sep = PART.sepColorOf(g);
    if (sep === null || sep === undefined || sep === k) return null;
    var m = new Map();
    m.set(sep, k); m.set(k, sep);
    return m;
  }

  function _panelwiseRule(dec, hyp, norm) {
    return function (g) {
      var cmap = null;
      if (norm !== null && norm !== undefined) {
        cmap = _normMap(g, norm);
        if (cmap === null) return null;
        g = G.applyCmap(g, cmap);
      }
      var mat = dec(g);
      if (!mat) return null;
      var pos = PANELABS.positions(g, null, dec);
      if (pos === null) return null;
      var rows = pos[0], cols = pos[1], ph = pos[2], pw = pos[3];
      var out = G.copyGrid(g), i, j, p, q, rr, cc;
      for (i = 0; i < rows.length; i++) for (j = 0; j < cols.length; j++) {
        p = mat[i][j];
        if (p === null || p === undefined || p.length !== ph || p[0].length !== pw) return null;
        q = hyp.fn(p);
        if (q === null || q === undefined || !G.valid(q) || q.length !== ph || q[0].length !== pw) return null;
        for (rr = 0; rr < ph; rr++)
          for (cc = 0; cc < pw; cc++) out[rows[i] + rr][cols[j] + cc] = q[rr][cc];
      }
      return cmap ? G.applyCmap(out, cmap) : out;
    };
  }

  function _modules() {
    var names = ["geometry", "colormap", "symmetry", "tiling", "cellwise",
                 "objects_map", "substitute", "sequence", "selfstamp", "extend"];
    var out = [], i, m;
    for (i = 0; i < names.length; i++) { m = moduleByName(names[i]); if (m) out.push(m); }
    return out;
  }

  function _canonicalSlot(ctx) {
    var used = G.csUnion(ctx.in_palette(), ctx.out_palette()), k;
    for (k = 9; k >= 0; k--) if (!G.csHas(used, k)) return k;
    return 9;
  }

  function _one(ctx, dname, dcost, dec, norm, deadline) {
    var res = [], pairs = [], changed = false, shape = null, t, a, b, cm, ma, mb, i, j, pa, pb, tins = [];
    try {
      for (t = 0; t < ctx.train.length; t++) {
        a = ctx.train[t][0]; b = ctx.train[t][1];
        if (norm !== null && norm !== undefined) {
          cm = _normMap(a, norm);
          if (cm === null) return res;
          a = G.applyCmap(a, cm); b = G.applyCmap(b, cm);
        }
        ma = dec(a); mb = dec(b);
        if (!ma || !mb || ma.length !== mb.length || ma[0].length !== mb[0].length) return res;
        for (i = 0; i < ma.length; i++) for (j = 0; j < ma[i].length; j++) {
          pa = ma[i][j]; pb = mb[i][j];
          if (!pa || !pb || pa.length !== pb.length || pa[0].length !== pb[0].length) return res;
          if (shape === null) shape = [pa.length, pa[0].length];
          else if (pa.length !== shape[0] || pa[0].length !== shape[1]) return res;
          if (!G.gEq(pa, pb)) changed = true;
          pairs.push([pa, pb]);
        }
      }
      if (!changed || !pairs.length || pairs.length > 80) return res;
      for (t = 0; t < ctx.test_inputs.length; t++) {
        var tg = ctx.test_inputs[t];
        if (norm !== null && norm !== undefined) {
          cm = _normMap(tg, norm);
          if (cm === null) return res;
          tg = G.applyCmap(tg, cm);
        }
        var m2 = dec(tg);
        if (!m2) return res;
        for (i = 0; i < m2.length; i++) for (j = 0; j < m2[i].length; j++) {
          var p = m2[i][j];
          if (!p || p.length !== shape[0] || p[0].length !== shape[1]) return res;
          tins.push(p);
        }
      }
    } catch (e) { return res; }

    var sub = new Ctx(pairs, tins);
    sub.deadline = Math.min(deadline, nowMs() + 1200);
    var tag = (norm === null || norm === undefined) ? dname : (dname + "~" + norm);
    var mods = _modules(), m, hyps, hp;
    for (m = 0; m < mods.length; m++) {
      if (nowMs() > sub.deadline) break;
      try { hyps = hypcacheGenerate(mods[m], sub); } catch (e) { continue; }
      for (i = 0; i < hyps.length; i++) {
        hp = hyps[i];
        if (hp.fits(sub.train)) {
          res.push(new Hyp("panelwise[" + tag + "]>>" + hp.name,
                           _panelwiseRule(dec, hp, norm), dcost + 1.5 + hp.cost, "partition"));
          break;
        }
      }
      if (res.length > 12) break;
    }
    return res;
  }

  function generate(ctx) {
    if (!ctx.same_shape()) return [];
    var deadline = ctx.deadline === null ? (nowMs() + 4000) : ctx.deadline;
    var res = [], slot = _canonicalSlot(ctx), decs = PART.decompositions(ctx).slice(0, 4), d, i;
    var norms = [null, slot];
    for (d = 0; d < decs.length; d++)
      for (i = 0; i < 2; i++) {
        if (nowMs() > deadline || res.length > 12) break;
        res = res.concat(_one(ctx, decs[d][0], decs[d][1], decs[d][2], norms[i], deadline));
      }
    return res;
  }

  defSolver("panelwise", "partition", generate, 2);
})();
