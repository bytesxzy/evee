/* Port of engine/solvers/refine.py -- specialist first, correction after.
 *
 * cascade composes forwards; the reverse composition
 * output = correction(specialist(input)) is a large family none of whose
 * members is reachable by a single specialist or a forward cascade. The
 * specialists' near misses -- wrong but close by cell and shape agreement --
 * each become a starting state for a short operator search over corrections,
 * so the search stays shallow even when the composed program is not.
 */

(function () {
  var _MAX_SEEDS = 34;
  var _MAX_PER_MODULE = 26;

  function _modules() {
    var names = ["geometry", "select", "partition", "symmetry", "tiling", "blocks",
                 "regions", "colormap", "objects_map", "panelabs", "paneltable",
                 "panelwise", "patterns", "assemble", "selfstamp", "extend"];
    var out = [], i, m;
    for (i = 0; i < names.length; i++) { m = moduleByName(names[i]); if (m) out.push(m); }
    return out;
  }

  function _agreement(a, b) {
    if (a === null || a.length !== b.length || a[0].length !== b[0].length) return 0.0;
    var n = 0, m = 0, r, c;
    for (r = 0; r < a.length; r++) for (c = 0; c < a[r].length; c++) {
      n++;
      if (a[r][c] === b[r][c]) m++;
    }
    return n ? m / n : 0.0;
  }

  function _score(state, target) {
    var n = target.length, agree = 0, shape = 0, i;
    for (i = 0; i < n; i++) {
      agree += _agreement(state[i], target[i]);
      if (state[i] !== null && state[i].length === target[i].length &&
          state[i][0].length === target[i][0].length) shape += 1.0;
    }
    return shape / n * 1.5 + agree / n;
  }

  /* Near-miss states produced by the specialists, best first. */
  function _seeds(ctx, deadline) {
    var nTr = ctx.train.length;
    var grids = ctx.inputs().concat(ctx.test_inputs);
    var target = ctx.outputs();
    var seen = new Set([stateKey(grids)]), out = [], mods = _modules(), m, hyps, i, j, hp, st, ok, taken, sk, s;
    for (m = 0; m < mods.length; m++) {
      if (nowMs() > deadline) break;
      try { hyps = hypcacheGenerate(mods[m], ctx); } catch (e) { continue; }
      taken = 0;
      for (i = 0; i < hyps.length; i++) {
        if (taken >= _MAX_PER_MODULE || nowMs() > deadline) break;
        hp = hyps[i];
        st = []; ok = true;
        for (j = 0; j < grids.length; j++) {
          var p = hp.apply(grids[j]);
          if (p === null) { ok = false; break; }
          st.push(p);
        }
        if (!ok) continue;
        sk = stateKey(st);
        if (seen.has(sk)) continue;
        seen.add(sk);
        taken++;
        var exact = true;
        for (j = 0; j < nTr; j++) if (!G.gEq(st[j], target[j])) { exact = false; break; }
        if (exact) continue;              /* the portfolio finds those anyway */
        s = _score(st.slice(0, nTr), target);
        if (s <= 0.05) continue;
        out.push([s, hp.cost, hp, st]);
      }
    }
    out.sort(function (a, b) { return (b[0] - a[0]) || (a[1] - b[1]); });
    return out.slice(0, _MAX_SEEDS);
  }

  function _correct(state, ops, target, nTr, depth, deadline, seen) {
    var frontier = [[[], 0.0, "", state]], results = [], d, i, j, node, nw, ok, r, sk, rec, nxt, g2;
    for (d = 0; d < depth; d++) {
      nxt = [];
      for (i = 0; i < frontier.length; i++) {
        node = frontier[i];
        if (nowMs() > deadline) return results;
        for (j = 0; j < ops.length; j++) {
          nw = []; ok = true;
          for (g2 = 0; g2 < node[3].length; g2++) {
            try { r = ops[j][2](node[3][g2]); } catch (e) { ok = false; break; }
            if (r === null || r === undefined || !G.valid(r)) { ok = false; break; }
            nw.push(r);
          }
          if (!ok) continue;
          sk = stateKey(nw);
          if (seen.has(sk)) continue;
          seen.add(sk);
          rec = [node[0].concat([ops[j][2]]), node[1] + ops[j][1],
                 ops[j][0] + "(" + node[2] + ")", nw];
          var exact = true;
          for (g2 = 0; g2 < nTr; g2++) if (!G.gEq(nw[g2], target[g2])) { exact = false; break; }
          if (exact) {
            results.push(rec);
            if (results.length >= 3) return results;
          }
          nxt.push([_score(nw.slice(0, nTr), target), rec]);
        }
      }
      if (!nxt.length || nowMs() > deadline) break;
      nxt.sort(function (a, b) { return b[0] - a[0]; });
      frontier = [];
      for (i = 0; i < Math.min(14, nxt.length); i++) frontier.push(nxt[i][1]);
    }
    return results;
  }

  function _compose(hp, chain) {
    var run = applyChain(chain);
    return function (g) {
      var t = hp.fn(g);
      if (t === null || t === undefined) return null;
      return run(t);
    };
  }

  function generate(ctx) {
    var deadline = ctx.deadline === null ? (nowMs() + 6000) : ctx.deadline;
    if (nowMs() > deadline) return [];
    var nTr = ctx.train.length, target = ctx.outputs();
    var seedDl = Math.min(deadline, nowMs() + Math.max(600, (deadline - nowMs()) * 0.45));
    var seeds;
    try { seeds = _seeds(ctx, seedDl); } catch (e) { return []; }
    if (!seeds.length) return [];
    var ops = unaryOps(ctx, "full"), res = [], seen = new Set(), i, j, now, left, share, depth, found;
    for (i = 0; i < seeds.length; i++) {
      now = nowMs();
      if (now > deadline || res.length >= 24) break;
      left = deadline - now;
      share = Math.max(80, left / Math.max(1, seeds.length - i));
      depth = i < 8 ? 2 : 1;
      try {
        found = _correct(seeds[i][3], ops, target, nTr, depth, Math.min(deadline, now + share), seen);
      } catch (e) { continue; }
      for (j = 0; j < found.length; j++)
        res.push(new Hyp("refine:" + seeds[i][2].name + ">>" + found[j][2],
                         _compose(seeds[i][2], found[j][0]),
                         3.6 + seeds[i][1] + found[j][1], "compose"));
    }
    return res;
  }

  defSolver("refine", "compose", generate, 2);
})();
