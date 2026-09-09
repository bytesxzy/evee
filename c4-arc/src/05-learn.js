/* Port of engine/learn.py -- task signatures and the learned policy.
 *
 * Only the parts the solver needs at run time are ported: signature extraction,
 * policy application (family bias, skips, operator bias, mined abstractions).
 * Fitting stays offline in the Python tree, where the paired evaluation that
 * gates every policy change also lives.
 */

function signatures(ctx) {
  var sig = [], same = ctx.same_shape(), i, a, b;
  sig.push(same ? "shape:same" : "shape:diff");
  if (!same) {
    var bigger = true, smaller = true;
    for (i = 0; i < ctx.train.length; i++) {
      a = ctx.train[i][0]; b = ctx.train[i][1];
      if (!(G.area(b) > G.area(a))) bigger = false;
      if (!(G.area(b) < G.area(a))) smaller = false;
    }
    sig.push(bigger ? "size:up" : (smaller ? "size:down" : "size:mix"));
    var sr = ctx.shape_ratio();
    if (sr) sig.push("ratio:" + sr[0] + "x" + sr[1]);
    var ir = ctx.inv_shape_ratio();
    if (ir) sig.push("iratio:" + ir[0] + "x" + ir[1]);
    if (ctx.const_out_shape()) sig.push("outshape:const");
  }
  var ip = ctx.in_palette(), op = ctx.out_palette();
  if (G.csDiff(op, ip)) sig.push("pal:new");
  if (G.csDiff(ip, op)) sig.push("pal:drop");
  if (op === ip) sig.push("pal:same");
  sig.push("ntrain:" + Math.min(ctx.train.length, 5));
  sig.push("bg:" + ctx.bg());
  var a0 = ctx.train[0][0], h = a0.length, w = a0[0].length;
  sig.push("in:" + (h * w <= 64 ? "small" : (h * w <= 400 ? "mid" : "big")));
  var npal = G.csSize(ctx.in_palette());
  sig.push("ncol:" + (npal <= 3 ? "low" : (npal <= 5 ? "mid" : "high")));
  try {
    if (HOOKS.sepColorCandidates && HOOKS.sepColorCandidates(ctx).length) sig.push("sep:yes");
  } catch (e) {}
  var allSym = true, ins = ctx.inputs();
  for (i = 0; i < ins.length; i++) if (!G.symmetries(ins[i]).length) { allSym = false; break; }
  if (allSym) sig.push("sym:in");
  return sig;
}

var _NAME_RE = /^([A-Za-z0-9_#-]+)\((.*)\)$/;

/* "crop(rot90($))" -> ["rot90", "crop"] (application order). */
function parseChain(name) {
  var ops = [], s = name, m;
  while (s !== "$") {
    m = _NAME_RE.exec(s);
    if (!m) return null;
    if (m[2].indexOf(",") >= 0) return null;   /* binary node: not linear */
    ops.push(m[1]);
    s = m[2];
  }
  ops.reverse();
  return ops;
}

function Policy(data) {
  var d = data || {};
  this.solver_prior = d.solver_prior || {};
  this.feature_bias = d.feature_bias || {};
  this.abstractions = d.abstractions || [];
  this.op_bias = d.op_bias || {};
  this.module_order = d.module_order || [];
  this.module_skip = d.module_skip || {};
  this.meta = d.meta || {};
}

/* Prior offset per solver family for a task with these signatures. */
Policy.prototype.bias_for = function (sigs) {
  var out = {}, i, s, row, fam;
  for (i = 0; i < sigs.length; i++) {
    s = sigs[i];
    row = this.feature_bias[s];
    if (row) for (fam in row) if (Object.prototype.hasOwnProperty.call(row, fam))
      out[fam] = (out[fam] || 0) + row[fam];
  }
  for (fam in this.solver_prior) if (Object.prototype.hasOwnProperty.call(this.solver_prior, fam))
    out[fam] = (out[fam] || 0) + this.solver_prior[fam];
  /* clamp so a learned bias can reorder but never silence a family */
  for (fam in out) if (Object.prototype.hasOwnProperty.call(out, fam))
    out[fam] = Math.max(-3.0, Math.min(3.0, out[fam]));
  return out;
};

/* Families to leave out entirely for a task with these signatures. */
Policy.prototype.skips_for = function (sigs, minVotes) {
  if (minVotes === undefined) minVotes = 2;
  var votes = {}, alive = {}, i, j, s, list, row, f;
  for (i = 0; i < sigs.length; i++) {
    s = sigs[i];
    list = this.module_skip[s] || [];
    for (j = 0; j < list.length; j++) votes[list[j]] = (votes[list[j]] || 0) + 1;
  }
  for (i = 0; i < sigs.length; i++) {
    row = this.feature_bias[sigs[i]];
    if (row) for (f in row) if (Object.prototype.hasOwnProperty.call(row, f)) alive[f] = 1;
  }
  var out = {};
  for (f in votes) if (Object.prototype.hasOwnProperty.call(votes, f))
    if (votes[f] >= minVotes && !alive[f]) out[f] = 1;
  return out;
};

/* Install abstractions and operator bias into the enumerator. */
Policy.prototype.install = function () {
  clearLearned();
  var i;
  for (i = 0; i < this.abstractions.length; i++) installAbstraction(this.abstractions[i]);
  var k;
  for (k in OP_BIAS) if (Object.prototype.hasOwnProperty.call(OP_BIAS, k)) delete OP_BIAS[k];
  for (k in this.op_bias) if (Object.prototype.hasOwnProperty.call(this.op_bias, k))
    OP_BIAS[k] = this.op_bias[k];
};

function installAbstraction(ab) {
  var names = ab.ops.slice(), cost = Number(ab.cost === undefined ? 1.0 : ab.cost);
  addLearnedOp(ab.name, cost, function (ctx) {
    var base = {}, ops = baseUnaryOps(ctx, "full"), i;
    for (i = 0; i < ops.length; i++) base[ops[i][0]] = ops[i][2];
    var fns = [];
    for (i = 0; i < names.length; i++) {
      if (!base[names[i]]) return null;
      fns.push(base[names[i]]);
    }
    return function (g) {
      var j;
      for (j = 0; j < fns.length; j++) {
        g = fns[j](g);
        if (g === null || g === undefined) return null;
      }
      return g;
    };
  });
}

var _ACTIVE_POLICY = null;

function activatePolicy(policy) {
  _ACTIVE_POLICY = policy;
  if (policy) policy.install();
  else {
    clearLearned();
    var k;
    for (k in OP_BIAS) if (Object.prototype.hasOwnProperty.call(OP_BIAS, k)) delete OP_BIAS[k];
  }
  PORTFOLIO_STATE.POLICY = policy;
}

function activePolicy() { return _ACTIVE_POLICY; }

var LEARN = {
  signatures: signatures, parseChain: parseChain, Policy: Policy,
  activate: activatePolicy, active: activePolicy
};
