/* Port of engine/solvers/conditional.py -- rules that branch on a property of
 * the input.
 *
 * Every other hypothesis is a single total function, which makes one class of
 * ARC rule unstateable: the ones where the demonstrations genuinely do
 * different things and a property of the input selects between them. The
 * construction is deliberately conservative, because a branch is a licence to
 * memorise: every group must carry at least two pairs, a group of three or
 * more must survive its own leave-one-out refit, and a task some single
 * hypothesis already explains is left alone.
 */

(function () {
  var _MAX_BRANCHES = 4;

  function _modules() {
    var names = ["geometry", "colormap", "tiling", "symmetry", "select", "partition",
                 "blocks", "objects_map", "objproc", "cellwise"];
    var out = [], i, m;
    for (i = 0; i < names.length; i++) { m = moduleByName(names[i]); if (m) out.push(m); }
    return out;
  }

  function _predicates(ctx) {
    var pal = G.csList(ctx.in_palette()), preds = [], i, seg, segs = ["c4", "c8"];
    preds.push(["shape", function (g) { return g.length + "x" + g[0].length; }]);
    preds.push(["h", function (g) { return "" + g.length; }]);
    preds.push(["w", function (g) { return "" + g[0].length; }]);
    preds.push(["square", function (g) { return g.length === g[0].length ? "T" : "F"; }]);
    preds.push(["tall", function (g) { return g.length > g[0].length ? "T" : "F"; }]);
    preds.push(["ncol", function (g) { return "" + G.csSize(G.palette(g)); }]);
    preds.push(["bg", function (g) { return "" + G.background(g); }]);
    preds.push(["top", function (g) { return "" + G.mostCommonColor(g); }]);
    preds.push(["rare", function (g) { return "" + G.leastCommonColor(g); }]);
    preds.push(["symh", function (g) { return G.gEq(g, G.flipH(g)) ? "T" : "F"; }]);
    preds.push(["symv", function (g) { return G.gEq(g, G.flipV(g)) ? "T" : "F"; }]);
    preds.push(["sym", function (g) { return G.symmetries(g).length ? "T" : "F"; }]);
    preds.push(["corner", function (g) { return "" + g[0][0]; }]);
    for (i = 0; i < Math.min(6, pal.length); i++) {
      (function (c) {
        preds.push(["has" + c, function (g) { return G.csHas(G.palette(g), c) ? "T" : "F"; }]);
        preds.push(["cnt" + c, function (g) { return "" + G.countColor(g, c); }]);
      })(pal[i]);
    }
    for (i = 0; i < segs.length; i++) {
      (function (sg) {
        preds.push(["n_" + sg, function (g) {
          return "" + Math.min(40, O.segment(g, sg, G.background(g)).length);
        }]);
        preds.push(["np_" + sg, function (g) {
          return "" + (O.segment(g, sg, G.background(g)).length % 2);
        }]);
        preds.push(["big_" + sg, function (g) {
          var objs = O.segment(g, sg, G.background(g)), j, m = -Infinity, best = -1;
          if (!objs.length) return "-1";
          for (j = 0; j < objs.length; j++) if (objs[j].size() > m) m = objs[j].size();
          for (j = 0; j < objs.length; j++) if (objs[j].size() === m && objs[j].color > best) best = objs[j].color;
          return "" + best;
        }]);
      })(segs[i]);
    }
    return preds;
  }

  function _keys(pred, grids) {
    var out = [], i, k;
    for (i = 0; i < grids.length; i++) {
      try { k = pred(grids[i]); } catch (e) { return null; }
      out.push(k);
    }
    return out;
  }

  function _branching(pred, table, def) {
    return function (g) {
      var k;
      try { k = pred(g); } catch (e) { return null; }
      var fn = table.has(k) ? table.get(k) : def;
      if (fn === null || fn === undefined) return null;
      try { return fn(g); } catch (e) { return null; }
    };
  }

  /* Ordinary hypotheses refitted on one group of demonstrations. */
  function _branchHyps(ctx, pairs, testInputs, deadline) {
    var sub = new Ctx(pairs, testInputs, deadline), out = [], mods = _modules(), m, hyps, i, best;
    for (m = 0; m < mods.length; m++) {
      if (nowMs() > deadline) break;
      try { hyps = hypcacheGenerate(mods[m], sub); } catch (e) { continue; }
      best = null;
      for (i = 0; i < hyps.length; i++)
        if (hyps[i].fits(sub.train) && (best === null || hyps[i].cost < best.cost)) best = hyps[i];
      if (best !== null) out.push(best);
    }
    out.sort(function (a, b) { return a.cost - b.cost; });
    return out.slice(0, 3);
  }

  /* Refit this branch with one pair withheld and check it back. A two-pair
     branch has nothing to withhold, so it is passed through and carries the
     extra cost instead. */
  function _looOk(ctx, idxs, deadline) {
    if (idxs.length < 3) return true;
    var rest = [], i;
    for (i = 0; i < idxs.length - 1; i++) rest.push(ctx.train[idxs[i]]);
    var heldIn = ctx.train[idxs[idxs.length - 1]][0], heldOut = ctx.train[idxs[idxs.length - 1]][1];
    var hyps = _branchHyps(ctx, rest, [heldIn], deadline), p;
    for (i = 0; i < hyps.length; i++) {
      p = hyps[i].apply(heldIn);
      if (p !== null && G.gEq(p, heldOut)) return true;
    }
    return false;
  }

  /* Whether an ordinary, unbranched rule already explains the task. */
  function _singleFitExists(ctx, deadline) {
    var mods = _modules(), m, hyps, i;
    for (m = 0; m < mods.length; m++) {
      if (nowMs() > deadline) return false;
      try { hyps = hypcacheGenerate(mods[m], ctx); } catch (e) { continue; }
      for (i = 0; i < hyps.length; i++) if (hyps[i].fits(ctx.train)) return true;
    }
    return false;
  }

  function generate(ctx) {
    var n = ctx.train.length;
    if (n < 4) return [];             /* every branch needs two pairs of its own */
    var deadline = ctx.deadline === null ? (nowMs() + 5000) : ctx.deadline;
    if (nowMs() > deadline) return [];
    var inputs = ctx.inputs();
    if (_singleFitExists(ctx, Math.min(deadline, nowMs() + 1200))) return [];
    var res = [], seen = new Set(), tried = 0, preds = _predicates(ctx), pi, keys, i;
    for (pi = 0; pi < preds.length; pi++) {
      if (nowMs() > deadline || res.length >= 4) break;
      keys = _keys(preds[pi][1], inputs);
      if (keys === null) continue;
      var groups = new Map(), order = [];
      for (i = 0; i < keys.length; i++) {
        if (!groups.has(keys[i])) { groups.set(keys[i], []); order.push(keys[i]); }
        groups.get(keys[i]).push(i);
      }
      if (groups.size < 2 || groups.size > Math.min(_MAX_BRANCHES, Math.floor(n / 2))) continue;
      var minLen = Infinity;
      groups.forEach(function (v) { if (v.length < minLen) minLen = v.length; });
      if (minLen < 2) continue;
      var sigParts = [];
      groups.forEach(function (v) { sigParts.push(v.join(",")); });
      sigParts.sort();
      var sig = sigParts.join("|");
      if (seen.has(sig)) continue;    /* another predicate already cut this way */
      seen.add(sig);
      tried++;
      if (tried > 5) break;
      var share = Math.max(250, (deadline - nowMs()) / 4.0);
      var table = new Map(), ok = true, support = new Map(), k2;
      for (i = 0; i < order.length && ok; i++) {
        k2 = order[i];
        var idxs = groups.get(k2), pairs = [], j;
        for (j = 0; j < idxs.length; j++) pairs.push(ctx.train[idxs[j]]);
        var hyps = _branchHyps(ctx, pairs, ctx.test_inputs, Math.min(deadline, nowMs() + share));
        if (!hyps.length) { ok = false; break; }
        table.set(k2, hyps[0].fn);
        support.set(k2, [idxs.length, hyps[0].cost, hyps[0].name]);
      }
      if (!ok) continue;
      var names = new Set();
      support.forEach(function (v) { names.add(v[2]); });
      if (names.size < 2) continue;   /* one rule everywhere is not a branch */
      var looOk = true;
      for (i = 0; i < order.length; i++)
        if (!_looOk(ctx, groups.get(order[i]), Math.min(deadline, nowMs() + share))) { looOk = false; break; }
      if (!looOk) continue;
      var sumCost = 0;
      support.forEach(function (v) { sumCost += v[1]; });
      var cost = 4.2 + 0.8 * table.size + sumCost / support.size;
      var labelKeys = order.slice();
      labelKeys.sort();
      var labelParts = [];
      for (i = 0; i < labelKeys.length; i++) labelParts.push(labelKeys[i] + ":" + support.get(labelKeys[i])[2]);
      var label = "if[" + preds[pi][0] + "]{" + labelParts.join(",").slice(0, 90) + "}";
      var modal = order[0];
      for (i = 1; i < order.length; i++) {
        var a = support.get(order[i]), b = support.get(modal);
        if (a[0] > b[0] || (a[0] === b[0] && -a[1] > -b[1])) modal = order[i];
      }
      var variants = [[null, 0.0, ""], [table.get(modal), 0.5, "+fallback"]], vi;
      for (vi = 0; vi < 2; vi++) {
        var hp = new Hyp(label + variants[vi][2], _branching(preds[pi][1], table, variants[vi][0]),
                         cost + variants[vi][1], "compose");
        if (hp.fits(ctx.train)) { res.push(hp); break; }
      }
    }
    return res;
  }

  defSolver("conditional", "compose", generate, 2);
})();
