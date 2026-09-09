/* Port of engine/solvers/objchain.py -- compositional search over object
 * programs.
 *
 * objops gives the enumerator a vocabulary for saying which thing to act on,
 * but only as seed operators applied once to the raw input. "Throw away
 * everything touching the edge, then drop what is left, then keep the largest"
 * is three object steps and nothing else in the engine can express it. Kept
 * separate from the pixel library because the two spaces have different
 * economics: mixing them at every depth multiplies both branching factors.
 */

(function () {
  var _h = mkHyp("objects");
  var _BEAM = 140;
  var _MAX_DEPTH = 3;

  /* A few cheap whole-grid moves, so a chain can finish with a crop. */
  function _gridOps(ctx) {
    var bg = ctx.bg(), ops = [["crop", 1.0, function (g) { return G.cropToContent(g, bg); }]], i;
    for (i = 1; i < G.DIHEDRAL.length; i++) ops.push([G.DIHEDRAL[i][0], 1.0, G.DIHEDRAL[i][1]]);
    var dirs = ["down", "up", "left", "right"];
    for (i = 0; i < dirs.length; i++)
      ops.push(["grav_" + dirs[i], 1.4,
                (function (d) { return function (g) { return G.gravity(g, bg, d); }; })(dirs[i])]);
    ops.push(["dedup", 1.4, G.dedup]);
    return ops;
  }

  function _applyAll(fn, state, deadline) {
    var out = [], i, r;
    for (i = 0; i < state.length; i++) {
      if (nowMs() > deadline) return null;
      try { r = fn(state[i]); } catch (e) { return null; }
      if (r === null || r === undefined || !G.valid(r)) return null;
      out.push(r);
    }
    return out;
  }

  function _fitCmap(state, target) {
    var m = new Map(), i, r, c, x, y, any = false;
    for (i = 0; i < state.length; i++) {
      if (state[i].length !== target[i].length || state[i][0].length !== target[i][0].length) return null;
      for (r = 0; r < state[i].length; r++) for (c = 0; c < state[i][r].length; c++) {
        x = state[i][r][c]; y = target[i][r][c];
        if (!m.has(x)) m.set(x, y);
        else if (m.get(x) !== y) return null;
      }
    }
    m.forEach(function (v, k) { if (k !== v) any = true; });
    return any ? m : null;
  }

  function _chain(fns) {
    return function (g) {
      var i;
      for (i = 0; i < fns.length; i++) {
        g = fns[i](g);
        if (g === null || g === undefined) return null;
      }
      return g;
    };
  }

  /* How near a state is to the demonstrated outputs; ties broken by shape. */
  function _distance(state, target) {
    var n = 0.0, shape = 0.0, i, r, c, a, t, agree;
    for (i = 0; i < target.length; i++) {
      a = state[i]; t = target[i];
      if (a.length === t.length && a[0].length === t[0].length) {
        shape += 1.0;
        agree = 0;
        for (r = 0; r < a.length; r++) for (c = 0; c < a[r].length; c++) if (a[r][c] === t[r][c]) agree++;
        n += agree / Math.max(1, G.area(t));
      }
    }
    var k = target.length;
    return shape / k * 1.5 + n / k;
  }

  function generate(ctx) {
    var deadline = ctx.deadline === null ? (nowMs() + 4000) : ctx.deadline;
    if (nowMs() > deadline) return [];
    var nTr = ctx.train.length;
    var grids = ctx.inputs().concat(ctx.test_inputs);
    var target = ctx.outputs(), ops;
    try { ops = objectOps(ctx).concat(_gridOps(ctx)); } catch (e) { return []; }
    var seen = new Set([stateKey(grids)]);
    var frontier = [[grids, [], "$", 0.0]];
    var found = new Map();

    function record(state, fns, name, cost) {
      var key = stateKey(state), prev = found.get(key);
      if (prev === undefined || cost < prev[0]) found.set(key, [cost, name, _chain(fns)]);
    }
    function prefixMatch(st) {
      var i;
      for (i = 0; i < nTr; i++) if (!G.gEq(st[i], target[i])) return false;
      return true;
    }

    var depth, nxt, i, j, node, st, sk, nm, cc, m, cf, st2;
    for (depth = 1; depth <= _MAX_DEPTH; depth++) {
      nxt = [];
      for (i = 0; i < frontier.length; i++) {
        node = frontier[i];
        if (nowMs() > deadline) break;
        for (j = 0; j < ops.length; j++) {
          if (nowMs() > deadline) break;
          st = _applyAll(ops[j][2], node[0], deadline);
          if (st === null) continue;
          sk = stateKey(st);
          if (seen.has(sk)) continue;
          seen.add(sk);
          nm = ops[j][0] + "(" + node[2] + ")";
          cc = node[3] + ops[j][1];
          if (prefixMatch(st)) { record(st, node[1].concat([ops[j][2]]), nm, cc); continue; }
          m = _fitCmap(st.slice(0, nTr), target);
          if (m !== null) {
            cf = (function (mm) { return function (g) { return G.applyCmap(g, mm); }; })(m);
            st2 = _applyAll(cf, st, deadline);
            if (st2 !== null && prefixMatch(st2))
              record(st2, node[1].concat([ops[j][2], cf]), "cmap(" + nm + ")", cc + 1.4 + 0.15 * m.size);
          }
          nxt.push([_distance(st.slice(0, nTr), target), [st, node[1].concat([ops[j][2]]), nm, cc]]);
        }
        if (found.size >= 4) break;
      }
      if (!nxt.length || nowMs() > deadline || found.size >= 4) break;
      nxt.sort(function (a, b) { return b[0] - a[0]; });
      frontier = [];
      for (i = 0; i < Math.min(_BEAM, nxt.length); i++) frontier.push(nxt[i][1]);
    }
    var vals = [];
    found.forEach(function (v) { vals.push(v); });
    vals.sort(function (a, b) { return (a[0] - b[0]) || cmpStr(a[1], b[1]); });
    var out = [];
    for (i = 0; i < Math.min(6, vals.length); i++)
      out.push(_h("chain:" + vals[i][1], vals[i][2], 2.6 + vals[i][0]));
    return out;
  }

  defSolver("objchain", "objects", generate, 2);
})();
