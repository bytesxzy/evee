/* Port of engine/enum_core.py -- bottom-up synthesis with observational
 * equivalence pruning.
 *
 * The op library is task-parameterised: colours occurring in the task become
 * constants and the inferred background becomes the default fill. Search
 * proceeds in levels, comparing programs by the tuple of grids they produce on
 * train and test inputs, and pruning equivalents that cost no less at no less
 * depth.
 */

var LEARNED = [];        /* [name, cost, factory(ctx) -> fn|null] */
var OP_BIAS = {};        /* op name -> search-order bonus, fitted by learn.js */

/* Late-bound entry points into individual solvers. The Python original
   imported them lazily inside each op; the bundle fills this table as the
   solver sources load, which keeps the load order acyclic. */
var HOOKS = {};

function addLearnedOp(name, cost, factory) {
  var i;
  for (i = 0; i < LEARNED.length; i++)
    if (LEARNED[i][0] === name) { LEARNED[i] = [name, cost, factory]; return; }
  LEARNED.push([name, cost, factory]);
}

function clearLearned() { LEARNED.length = 0; }

function stateKey(st) {
  var parts = [], i;
  for (i = 0; i < st.length; i++) parts.push(G.gkey(st[i]));
  return parts.join("~");
}

/* --------------------------------------------------------------- op library */

function unaryOps(ctx, level) {
  var ops = baseUnaryOps(ctx, level), i, fn;
  for (i = 0; i < LEARNED.length; i++) {
    try { fn = LEARNED[i][2](ctx); } catch (e) { fn = null; }
    if (fn) ops.push([LEARNED[i][0], LEARNED[i][1], fn]);
  }
  return ops;
}

/* Operators too costly to apply at every node: they run once, on the raw
   input, seeding the frontier. */
function seedOps(ctx) {
  var bg = ctx.bg();
  var pal = G.csList(G.csUnion(ctx.in_palette(), ctx.out_palette()));
  var ops = [
    ["frame_in", 2.0, function (g) { return HOOKS.frameInterior(g, bg, "largest"); }],
    ["frame_all", 2.2, function (g) { return HOOKS.frameContent(g, bg, "largest"); }],
    ["repair", 2.2, function (g) { return HOOKS.repair(g, bg, true); }],
    ["complete", 2.4, function (g) { return HOOKS.repairBounded(g, bg, false, 0.0, 2, 0); }],
    ["outline", 2.2, function (g) { return HOOKS.halo(g, bg, null, false, false); }],
    ["connect", 2.2, function (g) { return HOOKS.connect(g, bg, null, false); }]
  ];
  var i;
  for (i = 0; i < pal.length; i++) {
    ops.push(["halo#" + pal[i], 2.4,
              (function (c) { return function (g) { return HOOKS.halo(g, bg, c, false, false); }; })(pal[i])]);
    ops.push(["mark#" + pal[i], 2.0,
              (function (c) { return function (g) { return HOOKS.markedRect(g, bg, c, false); }; })(pal[i])]);
  }
  try { ops = ops.concat(objectOps(ctx)); } catch (e) {}
  return ops;
}

function _enDenoise(g, bg) {
  var objs = O.segment(g, "c8", bg);
  if (!objs.length || objs.length > 300) return null;
  var out = G.copyGrid(g), hit = false, i, it, s;
  for (i = 0; i < objs.length; i++) if (objs[i].size() === 1) {
    it = objs[i].cells.values(); s = it.next();
    while (!s.done) { out[s.value >> 6][s.value & 63] = bg; s = it.next(); }
    hit = true;
  }
  return hit ? out : null;
}

function _enKeepBig(g, bg, keep) {
  var objs = O.segment(g, "c8", bg);
  if (objs.length < 2 || objs.length > 300) return null;
  var o = O.selectExtreme(objs, "size", true);
  if (!o) return null;
  var h = g.length, w = g[0].length, out, it, s;
  if (keep) {
    out = G.constGrid(h, w, bg);
    it = o.cells.values(); s = it.next();
    while (!s.done) { out[s.value >> 6][s.value & 63] = g[s.value >> 6][s.value & 63]; s = it.next(); }
  } else {
    out = G.copyGrid(g);
    it = o.cells.values(); s = it.next();
    while (!s.done) { out[s.value >> 6][s.value & 63] = bg; s = it.next(); }
  }
  return out;
}

function _enBboxFill(g, bg) {
  var objs = O.segment(g, "c8", bg);
  if (!objs.length || objs.length > 120) return null;
  var out = G.copyGrid(g), i, r, c, o;
  for (i = 0; i < objs.length; i++) {
    o = objs[i];
    for (r = o.r0; r <= o.r1; r++) for (c = o.c0; c <= o.c1; c++) out[r][c] = o.color;
  }
  return out;
}

function _enCompress(g, bg) {
  var rows = [], r, c, row, any;
  for (r = 0; r < g.length; r++) {
    row = g[r]; any = false;
    for (c = 0; c < row.length; c++) if (row[c] !== bg) { any = true; break; }
    if (any) rows.push(row);
  }
  if (!rows.length) return null;
  var t = G.transpose(rows), cols = [];
  for (r = 0; r < t.length; r++) {
    row = t[r]; any = false;
    for (c = 0; c < row.length; c++) if (row[c] !== bg) { any = true; break; }
    if (any) cols.push(row);
  }
  if (!cols.length) return null;
  return G.transpose(cols);
}

function _enExtreme(g, seg, bg, biggest) {
  var objs = O.segment(g, seg, bg);
  if (!objs.length || objs.length > 200) return null;
  var o = O.selectExtreme(objs, "size", biggest);
  return o ? o.filled(bg) : null;
}

function _enUniq(g, seg, bg) {
  var objs = O.segment(g, seg, bg);
  if (!objs.length || objs.length > 200) return null;
  var o = O.selectUniqueShape(objs);
  return o ? o.filled(bg) : null;
}

function sortGridRows(g) {
  var rows = G.copyGrid(g);
  rows.sort(function (a, b) {
    var i, n = Math.min(a.length, b.length);
    for (i = 0; i < n; i++) if (a[i] !== b[i]) return a[i] - b[i];
    return a.length - b.length;
  });
  return rows;
}

function baseUnaryOps(ctx, level) {
  var bg = ctx.bg();
  var pal = G.csList(G.csUnion(ctx.in_palette(), ctx.out_palette()));
  var ops = [], i, c, k, d, dr, dc;
  function a(t) { ops.push(t); }

  for (i = 1; i < G.DIHEDRAL.length; i++) a([G.DIHEDRAL[i][0], 1.0, G.DIHEDRAL[i][1]]);
  a(["crop", 1.2, function (g) { return G.cropToContent(g, bg); }]);
  a(["dedup", 1.5, G.dedup]);
  a(["dedup_r", 1.8, G.dedupRows]);
  a(["dedup_c", 1.8, G.dedupCols]);
  a(["trim", 1.5, function (g) { return G.trimBorder(g, 1); }]);
  var halves = ["top", "bottom", "left", "right"];
  for (i = 0; i < halves.length; i++)
    a(["half_" + halves[i], 1.4, (function (w) { return function (g) { return G.half(g, w); }; })(halves[i])]);
  for (i = 0; i < 4; i++)
    a(["quad" + i, 1.6, (function (q) { return function (g) { return G.quadrant(g, q); }; })(i)]);
  var grav = ["down", "up", "left", "right"];
  for (i = 0; i < grav.length; i++)
    a(["grav_" + grav[i], 1.8, (function (dd) { return function (g) { return G.gravity(g, bg, dd); }; })(grav[i])]);
  a(["motif", 2.0, function (g) { return HOOKS.motif(g); }]);
  a(["compress", 2.0, function (g) { return _enCompress(g, bg); }]);

  if (level !== "small") {
    for (i = 0; i < pal.length; i++) {
      c = pal[i];
      a(["crop#" + c, 1.8, (function (cc) { return function (g) { return G.cropToContent(g, cc); }; })(c)]);
    }
    for (i = 0; i < pal.length; i++) {
      c = pal[i];
      a(["fill#" + c, 1.8, (function (cc) { return function (g) { return G.fillHoles(g, cc, bg); }; })(c)]);
    }
    for (i = 0; i < pal.length; i++) {
      c = pal[i];
      a(["del#" + c, 1.6, (function (cc) { return function (g) { return G.replaceColor(g, cc, bg); }; })(c)]);
    }
    for (i = 0; i < pal.length; i++) {
      c = pal[i];
      a(["keep#" + c, 1.8, (function (cc) {
        return function (g) {
          var out = [], r, j, row, nr;
          for (r = 0; r < g.length; r++) {
            row = g[r]; nr = new Array(row.length);
            for (j = 0; j < row.length; j++) nr[j] = row[j] === cc ? cc : bg;
            out.push(nr);
          }
          return out;
        };
      })(c)]);
    }
    a(["upx2", 1.6, function (g) { return G.upscale(g, 2, 2); }]);
    a(["upx3", 1.8, function (g) { return G.upscale(g, 3, 3); }]);
    a(["dnx2", 1.6, function (g) { return G.downscale(g, 2, 2); }]);
    for (k = 2; k <= 3; k++) {
      a(["nzx" + k, 1.9, (function (kk) { return function (g) { return G.blockReduceNonbg(g, kk, kk, bg); }; })(k)]);
      a(["upx" + k + "_", 1.9, (function (kk) { return function (g) { return G.upscale(g, kk, kk); }; })(k)]);
    }
    a(["dnx3", 1.8, function (g) { return G.downscale(g, 3, 3); }]);
    a(["tile2", 1.8, function (g) { return G.tile(g, 2, 2); }]);
    a(["pad0", 1.8, function (g) { return G.pad(g, 1, bg); }]);
    var shifts = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    for (i = 0; i < shifts.length; i++) {
      dr = shifts[i][0]; dc = shifts[i][1];
      a(["sh" + sgn(dr) + sgn(dc), 1.7,
         (function (y, x) { return function (g) { return G.translate(g, y, x, bg); }; })(dr, dc)]);
      a(["wr" + sgn(dr) + sgn(dc), 1.9,
         (function (y, x) { return function (g) { return G.wrapTranslate(g, y, x); }; })(dr, dc)]);
    }
    for (i = 0; i < pal.length; i++) {
      c = pal[i];
      a(["paint#" + c, 1.7, (function (cc) {
        return function (g) {
          var out = [], r, j, row, nr;
          for (r = 0; r < g.length; r++) {
            row = g[r]; nr = new Array(row.length);
            for (j = 0; j < row.length; j++) nr[j] = row[j] !== bg ? cc : bg;
            out.push(nr);
          }
          return out;
        };
      })(c)]);
    }
    a(["largest8", 2.0, function (g) { return _enExtreme(g, "c8", bg, true); }]);
    a(["smallest8", 2.0, function (g) { return _enExtreme(g, "c8", bg, false); }]);
    a(["largest4", 2.2, function (g) { return _enExtreme(g, "c4", bg, true); }]);
    a(["uniq_shape", 2.4, function (g) { return _enUniq(g, "c8", bg); }]);
    a(["mode_color", 2.4, function (g) { return [[G.mostCommonColor(g)]]; }]);
    a(["denoise", 2.0, function (g) { return _enDenoise(g, bg); }]);
    a(["keep_big", 2.2, function (g) { return _enKeepBig(g, bg, true); }]);
    a(["drop_big", 2.4, function (g) { return _enKeepBig(g, bg, false); }]);
    a(["bbox_fill", 2.4, function (g) { return _enBboxFill(g, bg); }]);
    a(["sortrows", 2.6, sortGridRows]);
    a(["sortcols", 2.6, function (g) { return G.transpose(sortGridRows(G.transpose(g))); }]);
  }
  return ops;
}

function sgn(v) { return (v >= 0 ? "+" : "-") + Math.abs(v); }

/* Pairwise combinators, parameterised by background: "empty" is not always
   colour 0, and hard-coding it breaks every task drawn on a coloured field. */
function binaryOps(bg) {
  if (bg === undefined) bg = 0;
  return [
    ["hcat", 2.0, G.hconcat],
    ["vcat", 2.0, G.vconcat],
    ["and", 2.2, function (x, y) { return _logOp(x, y, "and", bg); }],
    ["or", 2.2, function (x, y) { return _logOp(x, y, "or", bg); }],
    ["xor", 2.2, function (x, y) { return _logOp(x, y, "xor", bg); }],
    ["diff", 2.4, function (x, y) { return _logOp(x, y, "diff", bg); }],
    ["over", 2.2, function (x, y) { return _logOp(x, y, "over", bg); }],
    ["under", 2.4, function (x, y) { return _logOp(y, x, "over", bg); }]
  ];
}

function _logOp(a, b, op, bg) {
  if (!a || !b || a.length !== b.length || a[0].length !== b[0].length) return null;
  var out = [], r, c, row, x, y, fx, fy;
  for (r = 0; r < a.length; r++) {
    row = new Array(a[r].length);
    for (c = 0; c < a[r].length; c++) {
      x = a[r][c]; y = b[r][c];
      fx = x !== bg; fy = y !== bg;
      if (op === "and") row[c] = (fx && fy) ? x : bg;
      else if (op === "or") row[c] = fx ? x : (fy ? y : bg);
      else if (op === "xor") row[c] = (fx === fy) ? bg : (fx ? x : y);
      else if (op === "diff") row[c] = (fx && !fy) ? x : bg;
      else row[c] = fy ? y : x;
    }
    out.push(row);
  }
  return out;
}

/* -------------------------------------------------------------------- search */

function _Node(state, chain, cost, name, depth, bonus) {
  this.state = state;
  this.chain = chain;
  this.cost = cost;
  this.name = name === undefined ? "$" : name;
  this.depth = depth === undefined ? 0 : depth;
  this.bonus = bonus === undefined ? 0.0 : bonus;
  this.key = null;
}
_Node.prototype.stateKey = function () {
  if (this.key === null) this.key = stateKey(this.state);
  return this.key;
};

function applyChain(chain) {
  return function (g) {
    var i;
    for (i = 0; i < chain.length; i++) {
      try {
        g = chain[i](g);
        if (g === null || g === undefined || !G.valid(g)) return null;
      } catch (e) { return null; }
    }
    return g;
  };
}

function applyStateFn(fn, state, deadline) {
  var out = [], i, r;
  for (i = 0; i < state.length; i++) {
    if (deadline !== null && deadline !== undefined && nowMs() >= deadline) return null;
    try {
      r = fn(state[i]);
      if (r === null || r === undefined || !G.valid(r)) return null;
    } catch (e) { return null; }
    out.push(r);
  }
  return out;
}

/* Keep the cost/depth Pareto frontier of each observable state: a cheaper but
   deeper program cannot replace a shallow one, which has more remaining
   composition depth. */
function admit(seen, node) {
  var k = node.stateKey(), labels = seen.get(k) || [], i;
  for (i = 0; i < labels.length; i++)
    if (labels[i][0] <= node.depth && labels[i][1] <= node.cost) return false;
  var kept = [];
  for (i = 0; i < labels.length; i++)
    if (!(node.depth <= labels[i][0] && node.cost <= labels[i][1])) kept.push(labels[i]);
  kept.push([node.depth, node.cost]);
  seen.set(k, kept);
  return true;
}

/* Infer a single consistent, supported recolouring from training only: every
   changed colour needs at least two observed cells, so a per-cell palette
   cannot turn an arbitrary grid into a memorised answer. */
function fitCmap(state, target) {
  var mapping = new Map(), counts = new Map(), i, r, c, x, y;
  for (i = 0; i < state.length; i++) {
    if (state[i].length !== target[i].length || state[i][0].length !== target[i][0].length) return null;
    for (r = 0; r < state[i].length; r++)
      for (c = 0; c < state[i][r].length; c++) {
        x = state[i][r][c]; y = target[i][r][c];
        if (!mapping.has(x)) mapping.set(x, y);
        else if (mapping.get(x) !== y) return null;
        counts.set(x, (counts.get(x) || 0) + 1);
      }
  }
  var changed = [], ok = true;
  mapping.forEach(function (v, k2) { if (k2 !== v) changed.push(k2); });
  if (!changed.length) return null;
  for (i = 0; i < changed.length; i++) if (counts.get(changed[i]) < 2) ok = false;
  if (!ok) return null;
  return mapping;
}

/* A ranking hint, never a correctness or pruning condition. */
function goalDistance(node, target) {
  var distance = 0.0, i, a, b, ah, aw, bh, bw, r, c, n;
  for (i = 0; i < target.length; i++) {
    a = node.state[i]; b = target[i];
    ah = a.length; aw = a[0].length; bh = b.length; bw = b[0].length;
    if (ah !== bh || aw !== bw) {
      distance += 1.0 + Math.abs(ah - bh) / Math.max(ah, bh) + Math.abs(aw - bw) / Math.max(aw, bw);
    } else {
      n = 0;
      for (r = 0; r < ah; r++) for (c = 0; c < aw; c++) if (a[r][c] !== b[r][c]) n++;
      distance += n / (ah * aw);
    }
  }
  return distance / target.length;
}

function cmpNum(a, b) { return a < b ? -1 : (a > b ? 1 : 0); }
function cmpStr(a, b) { return a < b ? -1 : (a > b ? 1 : 0); }

/* Reserve capacity for cheap programs and for target-near programs. */
function beam(nodes, width, target, deadline) {
  var ranked = nodes.slice();
  ranked.sort(function (x, y) {
    return cmpNum(x.cost - x.bonus, y.cost - y.bonus) || cmpNum(x.cost, y.cost) ||
           cmpStr(x.name, y.name);
  });
  if (ranked.length <= width) return ranked;
  var cheap = ranked.slice(0, Math.floor(width / 2));
  var chosen = new Set(cheap), near = [], i, n;
  for (i = 0; i < ranked.length; i++) {
    n = ranked[i];
    if (chosen.has(n)) continue;
    if (deadline !== null && deadline !== undefined && nowMs() >= deadline) return ranked.slice(0, width);
    near.push([goalDistance(n, target) + 0.1 * (n.cost - n.bonus), n.cost, n.name, near.length, n]);
  }
  near.sort(function (x, y) {
    return cmpNum(x[0], y[0]) || cmpNum(x[1], y[1]) || cmpStr(x[2], y[2]) || cmpNum(x[3], y[3]);
  });
  var out = cheap.slice();
  for (i = 0; i < near.length && out.length < width; i++) out.push(near[i][4]);
  return out;
}

/* Reject impossible terminal combinations before constructing grids. */
function binaryShapeOk(name, left, right, target) {
  var i, ah, aw, bh, bw, th, tw;
  for (i = 0; i < target.length; i++) {
    ah = left[i].length; aw = left[i][0].length;
    bh = right[i].length; bw = right[i][0].length;
    th = target[i].length; tw = target[i][0].length;
    if (name === "hcat") {
      if (ah !== bh || ah !== th || aw + bw !== tw) return false;
    } else if (name === "vcat") {
      if (aw !== bw || ah + bh !== th || aw !== tw) return false;
    } else if (ah !== bh || aw !== bw || ah !== th || aw !== tw) return false;
  }
  return true;
}

function mkBinary(ca, cb, bf) {
  var ra = applyChain(ca), rb = applyChain(cb);
  return function (g) {
    var x = ra(g), y = rb(g), result;
    if (x === null || y === null) return null;
    try {
      result = bf(x, y);
      return (result !== null && result !== undefined && G.valid(result)) ? result : null;
    } catch (e) { return null; }
  };
}

/* Return cost-ranked [name, cost, fn] programs fitting every train pair. */
function enumSearch(ctx, depth, maxStates, deadline, level, useBinary, prior) {
  if (depth === undefined) depth = 3;
  if (maxStates === undefined) maxStates = 1400;
  if (level === undefined) level = "full";
  if (useBinary === undefined) useBinary = true;
  if (depth < 0 || maxStates < 1) throw new Error("bad search bounds");
  var nTr = ctx.train.length;
  if (!nTr) return [];
  if (deadline === undefined || deadline === null) deadline = ctx.deadline;
  var grids = ctx.inputs().concat(ctx.test_inputs);
  var target = ctx.outputs();
  var bias = {}, k;
  for (k in OP_BIAS) if (Object.prototype.hasOwnProperty.call(OP_BIAS, k)) bias[k] = OP_BIAS[k];
  if (prior) for (k in prior) if (Object.prototype.hasOwnProperty.call(prior, k)) bias[k] = prior[k];
  var found = new Map(), seen = new Map();
  var root = new _Node(grids, [], 0.0);
  admit(seen, root);
  var frontier = [root], every = [root];

  function expired(limit) {
    if (limit === undefined) limit = deadline;
    return limit !== null && limit !== undefined && nowMs() >= limit;
  }

  function results() {
    var arr = [];
    found.forEach(function (v) { arr.push(v); });
    arr.sort(function (x, y) { return cmpNum(x[1], y[1]) || cmpStr(x[0], y[0]); });
    return arr.slice(0, 6);
  }

  function record(node) {
    var key = node.stateKey(), prev = found.get(key);
    if (prev === undefined || node.cost < prev[1] ||
        (node.cost === prev[1] && node.name < prev[0]))
      found.set(key, [node.name, node.cost, applyChain(node.chain)]);
  }

  function statePrefixMatches(st) {
    var i;
    for (i = 0; i < nTr; i++) if (!G.gEq(st[i], target[i])) return false;
    return true;
  }

  function check(node) {
    if (statePrefixMatches(node.state)) { record(node); return true; }
    if (node.depth < depth) {
      var mapping = fitCmap(node.state.slice(0, nTr), target);
      if (mapping !== null) {
        var fn = function (g) { return G.applyCmap(g, mapping); };
        var st = applyStateFn(fn, node.state, deadline);
        if (st !== null) {
          var pairs = [];
          mapping.forEach(function (v, kk) { pairs.push([kk, v]); });
          pairs.sort(function (x, y) { return x[0] - y[0]; });
          var label = pairs.map(function (p) { return p[0] + ">" + p[1]; }).join(",");
          record(new _Node(st, node.chain.concat([fn]),
                           node.cost + 1.6 + 0.2 * mapping.size,
                           "cmap[" + label + "](" + node.name + ")", node.depth + 1));
        }
      }
    }
    return false;
  }

  if (expired()) return [];
  check(root);
  if (depth === 0) return results();
  var ops = unaryOps(ctx, level);
  ops.sort(function (x, y) {
    var bx = bias[x[0]] || 0.0, by = bias[y[0]] || 0.0;
    return cmpNum(x[1] - bx, y[1] - by) || cmpNum(x[1], y[1]) || cmpStr(x[0], y[0]);
  });
  var seeds = (level !== "small") ? seedOps(ctx) : [];
  var unaryDeadline = deadline;
  if (useBinary && deadline !== null && deadline !== undefined) {
    var now = nowMs();
    unaryDeadline = now + Math.max(0.0, deadline - now) * 0.8;
  }
  var generationLimit = Math.max(64, maxStates * 4);
  var currentDepth, node, oname, ocost, f, st2, nn, i, j, currentOps, nxt;
  for (currentDepth = 1; currentDepth <= depth; currentDepth++) {
    nxt = new Map();
    currentOps = (currentDepth === 1) ? ops.concat(seeds) : ops;
    for (i = 0; i < frontier.length; i++) {
      node = frontier[i];
      if (expired(unaryDeadline)) break;
      for (j = 0; j < currentOps.length; j++) {
        if (expired(unaryDeadline)) break;
        oname = currentOps[j][0]; ocost = currentOps[j][1]; f = currentOps[j][2];
        st2 = applyStateFn(f, node.state, unaryDeadline);
        if (st2 === null) continue;
        nn = new _Node(st2, node.chain.concat([f]), node.cost + ocost,
                       oname + "(" + node.name + ")", currentDepth,
                       node.bonus + (bias[oname] || 0.0));
        if (!admit(seen, nn)) continue;
        check(nn);
        nxt.set(nn.stateKey(), nn);
      }
      if (nxt.size >= generationLimit) break;
    }
    var arr = [];
    nxt.forEach(function (v) { arr.push(v); });
    frontier = beam(arr, maxStates, target, unaryDeadline);
    every = every.concat(frontier);
    if (!frontier.length || expired(unaryDeadline) || found.size >= 6) break;
  }

  if (useBinary && !found.size) {
    var shallow = [];
    for (i = 0; i < every.length; i++) if (every[i].depth < depth) shallow.push(every[i]);
    var pool = beam(shallow, 120, target, deadline);
    var bops = binaryOps(ctx.bg()), na, nb, bf, stl, x, y, r, ok;
    for (i = 0; i < pool.length; i++) {
      na = pool[i];
      for (j = 0; j < pool.length; j++) {
        nb = pool[j];
        for (k = 0; k < bops.length; k++) {
          if (expired()) return results();
          oname = bops[k][0]; ocost = bops[k][1]; bf = bops[k][2];
          if (!binaryShapeOk(oname, na.state, nb.state, target)) continue;
          stl = []; ok = true;
          for (x = 0; x < na.state.length; x++) {
            if (expired()) return results();
            try {
              r = bf(na.state[x], nb.state[x]);
              if (r === null || r === undefined || !G.valid(r)) { ok = false; break; }
            } catch (e) { ok = false; break; }
            stl.push(r);
          }
          if (!ok || stl.length !== grids.length) continue;
          nn = new _Node(stl, [mkBinary(na.chain, nb.chain, bf)],
                         na.cost + nb.cost + ocost,
                         oname + "(" + na.name + "," + nb.name + ")",
                         1 + Math.max(na.depth, nb.depth));
          if (admit(seen, nn)) check(nn);
        }
      }
    }
  }
  return results();
}

var ENUM = {
  LEARNED: LEARNED, OP_BIAS: OP_BIAS, HOOKS: HOOKS,
  addLearnedOp: addLearnedOp, clearLearned: clearLearned,
  unaryOps: unaryOps, baseUnaryOps: baseUnaryOps, seedOps: seedOps,
  binaryOps: binaryOps, search: enumSearch, applyChain: applyChain
};
