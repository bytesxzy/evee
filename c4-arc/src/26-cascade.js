/* Port of engine/solvers/cascade.py -- two-stage solving: get close, then
 * solve the residual.
 *
 * Most solvers are all-or-nothing, which throws away the most useful signal in
 * the portfolio: the transform that gets the grid almost right. Run a shallow
 * search scored by cell agreement, keep the transforms that make the most
 * progress, re-pose each as a fresh task (stage1(input), output), and hand it
 * to the specialists. The composition is validated end-to-end, so a promising
 * first stage costs nothing when the second fails to close the gap.
 */

(function () {
  function _agreement(a, b) {
    if (a === null || a.length !== b.length || a[0].length !== b[0].length) return 0.0;
    var n = 0, m = 0, r, c;
    for (r = 0; r < a.length; r++) for (c = 0; c < a[r].length; c++) {
      n++;
      if (a[r][c] === b[r][c]) m++;
    }
    return m / n;
  }

  function _shapeMatch(a, b) {
    return (a.length === b.length && a[0].length === b[0].length) ? 1.0 : 0.0;
  }

  /* Shallow BFS ranked by how close each state gets to the outputs. */
  function _nearStates(ctx, depth, keep, deadline) {
    var nTr = ctx.train.length;
    var grids = ctx.inputs().concat(ctx.test_inputs);
    var target = ctx.outputs();
    var ops = unaryOps(ctx, "full").concat(seedOps(ctx));
    var startScore = 0, i, j, d;
    for (i = 0; i < nTr; i++) startScore += _agreement(grids[i], target[i]);
    startScore /= nTr;
    var seen = new Set([stateKey(grids)]);
    var frontier = [[grids, [], "$", 0.0]], scored = [], nxt, node, st, ok, r, sc, shp, rec, sk;
    for (d = 0; d < depth; d++) {
      nxt = [];
      for (i = 0; i < frontier.length; i++) {
        node = frontier[i];
        if (nowMs() > deadline) break;
        for (j = 0; j < ops.length; j++) {
          st = []; ok = true;
          var g2;
          for (g2 = 0; g2 < node[0].length; g2++) {
            try { r = ops[j][2](node[0][g2]); } catch (e) { ok = false; break; }
            if (r === null || r === undefined || !G.valid(r)) { ok = false; break; }
            st.push(r);
          }
          if (!ok) continue;
          sk = stateKey(st);
          if (seen.has(sk)) continue;
          seen.add(sk);
          sc = 0; shp = 0;
          for (g2 = 0; g2 < nTr; g2++) {
            sc += _agreement(st[g2], target[g2]);
            shp += _shapeMatch(st[g2], target[g2]);
          }
          sc /= nTr; shp /= nTr;
          rec = [st, node[1].concat([ops[j][2]]), ops[j][0] + "(" + node[2] + ")", node[3] + ops[j][1]];
          /* rank shape-correct states first: a second stage can repaint cells
             but cannot resize the grid */
          nxt.push([shp * 2.0 + sc - 0.02 * (node[3] + ops[j][1]), rec]);
        }
        if (nowMs() > deadline) break;
      }
      if (!nxt.length) break;
      nxt.sort(function (a, b) { return b[0] - a[0]; });
      for (i = 0; i < Math.min(keep, nxt.length); i++) scored.push(nxt[i]);
      frontier = [];
      for (i = 0; i < Math.min(keep, nxt.length); i++) frontier.push(nxt[i][1]);
    }
    scored.sort(function (a, b) { return b[0] - a[0]; });
    var out = [], seenStates = new Set(), s;
    for (i = 0; i < scored.length; i++) {
      s = scored[i][0]; rec = scored[i][1];
      sk = stateKey(rec[0]);
      if (seenStates.has(sk)) continue;
      seenStates.add(sk);
      /* a stage that destroys the grid outright is not worth continuing */
      if (s <= 0.0 && startScore > 0.0) continue;
      out.push(rec);
      if (out.length >= keep) break;
    }
    return out;
  }

  function _modules() {
    var names = ["geometry", "colormap", "symmetry", "partition", "tiling",
                 "blocks", "regions", "select", "cellwise", "objects_map",
                 "substitute", "sequence", "paneltable", "selfstamp", "extend", "tally"];
    var out = [], i, m;
    for (i = 0; i < names.length; i++) { m = moduleByName(names[i]); if (m) out.push(m); }
    return out;
  }

  function _compose(run1, hp) {
    return function (g) {
      var t = run1(g);
      if (t === null || t === undefined) return null;
      return hp.fn(t);
    };
  }

  function generate(ctx) {
    var deadline = ctx.deadline === null ? (nowMs() + 6000) : ctx.deadline;
    if (nowMs() > deadline) return [];
    var stageDl = Math.min(deadline, nowMs() + Math.max(1000, (deadline - nowMs()) * 0.4));
    var states;
    try { states = _nearStates(ctx, 2, 8, stageDl); } catch (e) { return []; }
    if (!states.length) return [];
    var res = [], mods = _modules(), nTr = ctx.train.length, i, j, m, pairs, tins, sub, share, run1, hyps, hp;
    for (i = 0; i < states.length; i++) {
      if (nowMs() > deadline || res.length > 40) break;
      pairs = [];
      for (j = 0; j < nTr; j++) pairs.push([states[i][0][j], ctx.train[j][1]]);
      tins = states[i][0].slice(nTr);
      sub = new Ctx(pairs, tins);
      share = Math.max(400, (deadline - nowMs()) / Math.max(1, states.length));
      sub.deadline = Math.min(deadline, nowMs() + share);
      run1 = applyChain(states[i][1]);
      for (m = 0; m < mods.length; m++) {
        if (nowMs() > sub.deadline) break;
        try { hyps = hypcacheGenerate(mods[m], sub); } catch (e) { continue; }
        for (j = 0; j < hyps.length; j++) {
          hp = hyps[j];
          if (hp.fits(sub.train)) {
            res.push(new Hyp(states[i][2] + ">>" + hp.name, _compose(run1, hp),
                             4.0 + states[i][3] + hp.cost, "compose"));
            break;   /* one per module per stage-1 is plenty */
          }
        }
      }
    }
    return res;
  }

  defSolver("cascade", "compose", generate, 2);
})();
