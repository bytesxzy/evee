/* Port of engine/portfolio.py -- validate and rank independent program
 * generators without retraining.
 *
 * Description length supplies prior weights. Bounded leave-one-out refits
 * provide additional evidence for the same rule and test behaviour, never an
 * unrelated rule in its family. Ensemble weights are evidence, not calibrated
 * probabilities.
 */

var SOLVER_PRIOR = {
  geometry: 0.0, cellwise: 1.0, partition: 0.0, symmetry: 0.0,
  objects: 1.5, tiling: 0.5, colormap: 0.0, select: 1.0,
  compose: 2.5, enumerate: 3.0, sequence: 1.0
};

/* The registration order of engine/portfolio.py::_load_default. Module order
   sets the per-module time shares, so it is part of the engine's behaviour. */
var MODULE_ORDER = ["geometry", "colormap", "partition", "paneltable", "symmetry",
  "tiling", "blocks", "selfstamp", "extend", "select", "locate", "regions",
  "counting", "cellwise", "objects_map", "objproc", "relproc", "tally", "motion",
  "substitute", "sequence", "paint", "patterns", "assemble", "analogy", "compose",
  "panelabs", "panelwise", "objwise", "objchain", "rewrite", "cascade", "refine",
  "conditional", "enumerate_dsl"];

function orderedModules() {
  var byName = {}, i, out = [];
  for (i = 0; i < SOLVER_MODULES.length; i++) byName[SOLVER_MODULES[i].__name__] = SOLVER_MODULES[i];
  for (i = 0; i < MODULE_ORDER.length; i++) if (byName[MODULE_ORDER[i]]) out.push(byName[MODULE_ORDER[i]]);
  for (i = 0; i < SOLVER_MODULES.length; i++)
    if (MODULE_ORDER.indexOf(SOLVER_MODULES[i].__name__) < 0) out.push(SOLVER_MODULES[i]);
  return out;
}

function Result() {
  this.predictions = [];
  this.hyps = [];
  this.chosen = [];
  this.elapsed = 0.0;
  this.n_hyps = 0;
  this.n_fit = 0;
  this.solver = null;
  this.diagnostics = { modules: [], loo: [], predictions: [] };
}

/* Keep custom generators behind the same output boundary as built-ins. */
function _prediction(hyp, grid) {
  try {
    var out = hyp.apply(grid);
    if (out === null || out === undefined) return null;
    return G.isGrid(out) ? out : null;
  } catch (e) { return null; }
}

function _sigKey(sig) {
  var parts = [], i;
  for (i = 0; i < sig.length; i++) parts.push(sig[i] === null ? "*" : G.gkey(sig[i]));
  return parts.join("~");
}

/* Validation, isolating generator failures. Returns [[hyp, cost], ...]. */
function _candidates(mod, ctx, stats, validationDeadline) {
  stats.generated = 0; stats.fitted = 0; stats.invalid = 0; stats.status = "complete";
  var started = nowMs();
  if (validationDeadline === undefined || validationDeadline === null)
    validationDeadline = ctx.deadline;
  function expired() {
    return validationDeadline !== null && validationDeadline !== undefined && nowMs() > validationDeadline;
  }
  var out = [];
  try {
    var candidates = hypcacheGenerate(mod, ctx), i, hyp, cost, fits, t, p;
    for (i = 0; i < candidates.length; i++) {
      if (expired()) { stats.status = "timed_out"; break; }
      hyp = candidates[i];
      stats.generated++;
      cost = Number(hyp.cost);
      if (!isFinite(cost) || typeof hyp.name !== "string" || typeof hyp.solver !== "string") {
        stats.invalid++;
        continue;
      }
      fits = true;
      for (t = 0; t < ctx.train.length; t++) {
        if (expired()) { stats.status = "timed_out"; fits = false; break; }
        p = _prediction(hyp, ctx.train[t][0]);
        if (p === null || !G.gEq(p, ctx.train[t][1])) { fits = false; break; }
      }
      if (fits) { stats.fitted++; out.push([hyp, cost]); }
    }
  } catch (exc) {
    stats.status = "error";
    stats.error = String(exc && exc.message ? exc.message : exc).slice(0, 160);
  }
  stats.generation_timed_out = ctx.timed_out();
  stats.elapsed = Math.round((nowMs() - started) / 10) / 100;
  return out;
}

/* Refit once per fold for all (solver, name, test-prediction) keys.
   Incomplete folds are missing evidence; matching-name refits split credit
   when ambiguous, so one lucky rule among many cannot earn full credit. */
function _looEvidence(mod, ctx, targets) {
  var evidence = new Map(), i;
  for (i = 0; i < targets.length; i++) evidence.set(targets[i].key, [0.0, 0]);
  if (ctx.train.length < 3) return evidence;
  var identities = new Set();
  for (i = 0; i < targets.length; i++) identities.add(targets[i].solver + "" + targets[i].name);
  var f, heldIn, heldOut, sub, started, foldEnd, subCtx, stats, observations, cands, j, hyp, sig, obs, idKey;
  for (f = 0; f < ctx.train.length; f++) {
    if (ctx.timed_out()) break;
    heldIn = ctx.train[f][0]; heldOut = ctx.train[f][1];
    sub = ctx.train.slice(0, f).concat(ctx.train.slice(f + 1));
    started = nowMs();
    foldEnd = Math.min(ctx.deadline === null ? Infinity : ctx.deadline, started + 700);
    subCtx = new Ctx(sub, [heldIn], started + (foldEnd - started) * 0.9);
    subCtx.op_prior = ctx.op_prior;
    stats = {}; observations = new Map();
    cands = _candidates(mod, subCtx, stats, foldEnd);
    for (j = 0; j < cands.length; j++) {
      hyp = cands[j][0];
      idKey = hyp.solver + "" + hyp.name;
      if (!identities.has(idKey)) continue;
      sig = [];
      for (i = 0; i < ctx.test_inputs.length; i++) sig.push(_prediction(hyp, ctx.test_inputs[i]));
      obs = _sigKey([_prediction(hyp, heldIn)]) + "||" + _sigKey(sig);
      if (!observations.has(idKey)) observations.set(idKey, new Map());
      observations.get(idKey).set(obs, [_prediction(hyp, heldIn), _sigKey(sig)]);
    }
    if (stats.status !== "complete" || subCtx.timed_out()) continue;
    for (i = 0; i < targets.length; i++) {
      var t = targets[i], counts = evidence.get(t.key);
      var variants = observations.get(t.solver + "" + t.name);
      counts[1] += 1;
      if (variants && variants.size) {
        var wins = 0;
        variants.forEach(function (v) {
          if (v[0] !== null && G.gEq(v[0], heldOut) && v[1] === t.sigKey) wins += 1;
        });
        counts[0] += wins / variants.size;
      }
    }
  }
  return evidence;
}

/* --------------------------------------------------- generation scheduling */

function _MinHeap(cmp) { this.a = []; this.cmp = cmp; }
_MinHeap.prototype.push = function (v) {
  var a = this.a, i = a.length, p;
  a.push(v);
  while (i > 0) {
    p = (i - 1) >> 1;
    if (this.cmp(a[i], a[p]) < 0) { var t = a[i]; a[i] = a[p]; a[p] = t; i = p; }
    else break;
  }
};
_MinHeap.prototype.replaceRoot = function (v) {
  var a = this.a, n = a.length, i = 0, l, r, s;
  a[0] = v;
  for (;;) {
    l = 2 * i + 1; r = l + 1; s = i;
    if (l < n && this.cmp(a[l], a[s]) < 0) s = l;
    if (r < n && this.cmp(a[r], a[s]) < 0) s = r;
    if (s === i) break;
    var t = a[i]; a[i] = a[s]; a[s] = t; i = s;
  }
};

/* Items are [negScore, negOrder, hyp, mod]; the heap root is the worst kept. */
function _itemCmp(x, y) {
  if (x[0] !== y[0]) return x[0] < y[0] ? -1 : 1;
  if (x[1] !== y[1]) return x[1] < y[1] ? -1 : 1;
  return 0;
}

function _moduleKey(mod) { return mod.__name__; }

/* Run one generator for its slice and fold what fits into the reservoir. */
function _harvest(mod, ctx, moduleEnd, bias, reservoir, order, res) {
  var now = nowMs();
  /* Eager search generators commonly return at their own deadline; keep a
     separate validation slice so their useful results are not discarded. */
  ctx.deadline = now + Math.max(0.0, moduleEnd - now) * 0.9;
  var fam = mod.SOLVER || "", prior;
  try {
    prior = Number(SOLVER_PRIOR[fam] === undefined ? 2.0 : SOLVER_PRIOR[fam]) + Number(bias[fam] || 0.0);
    if (!isFinite(prior)) prior = 2.0;
  } catch (e) { prior = 2.0; }
  var stats = { solver: fam, module: _moduleKey(mod) };
  var cands = _candidates(mod, ctx, stats, moduleEnd), i, hyp, cost, score, item;
  for (i = 0; i < cands.length; i++) {
    hyp = cands[i][0]; cost = cands[i][1];
    score = cost + prior;
    if (!isFinite(score)) { stats.invalid++; continue; }
    item = [-score, -order, hyp, mod];
    order += 1;
    if (reservoir.a.length < 600) reservoir.push(item);
    else if (_itemCmp(item, reservoir.a[0]) > 0) reservoir.replaceRoot(item);
  }
  res.n_hyps += stats.generated;
  res.diagnostics.modules.push(stats);
  return order;
}

/* A deeper pass of the general enumerator, available as a plan action. */
var _DeepEnumerator = {
  __name__: "deepen",
  SOLVER: "enumerate",
  PHASE: 2,
  generate: function (ctx) {
    var found = enumSearch(ctx, 5, 2600, ctx.deadline, "full", true, ctx.op_prior);
    var out = [], i;
    for (i = 0; i < found.length; i++)
      out.push(new Hyp("deep:" + found[i][0], found[i][2], 3.4 + found[i][1], "enumerate"));
    return out;
  }
};

function _deepen(ctx, res, bias, reservoir, order, moduleEnd) {
  return _harvest(_DeepEnumerator, ctx, moduleEnd, bias, reservoir, order, res);
}

function _plannerFor(ctx, res) {
  try {
    var pl = activePlanner();
    if (pl === null || !pl.trained) return null;
    return [pl, signatures(ctx)];
  } catch (exc) {
    res.diagnostics.planner_error = String(exc && exc.name ? exc.name : exc);
    return null;
  }
}

/* Choose the next generator, and its share, from P(action | state). The loop
   is closed: after each generator reports back the reasoning state changes and
   the distribution is recomputed. Every family still runs; the planner moves
   time between them rather than silencing any. */
function _plannedGeneration(ctx, res, plan, phase1, phase2, bias, reservoir, order, t0, generationEnd) {
  var pl = plan[0], sigs = plan[1], trace = [], ran = [], deepened = false;
  var groups = [phase1, phase2], gi, group, pool, i, now, span, fracLeft, avail, dist, key, p, mod;
  for (gi = 0; gi < 2; gi++) {
    group = groups[gi];
    if (!group.length) continue;
    pool = new Map();
    for (i = 0; i < group.length; i++) pool.set(_moduleKey(group[i]), group[i]);
    while (pool.size) {
      now = nowMs();
      if (now >= generationEnd) break;
      span = Math.max(1e-6, generationEnd - t0);
      fracLeft = Math.max(0.0, generationEnd - now) / span;
      avail = new Set();
      pool.forEach(function (v, k) { avail.add(k); });
      if (gi === 1 && !deepened && ran.indexOf("enumerate_dsl") >= 0) avail.add("deepen");
      try {
        dist = pl.distribution(sigs, ran, reservoir.a.length, fracLeft, ran.length, avail);
      } catch (e) { dist = {}; }
      var keys = Object.keys(dist);
      if (!keys.length) {
        var sorted = [];
        pool.forEach(function (v, k) { sorted.push(k); });
        sorted.sort();
        key = sorted[0];
        p = 1.0 / pool.size;
      } else {
        key = keys[0];
        for (i = 1; i < keys.length; i++)
          if (dist[keys[i]] > dist[key] || (dist[keys[i]] === dist[key] && keys[i] > key)) key = keys[i];
        p = dist[key];
      }
      var remainingTime = Math.max(0.0, generationEnd - now), slice_;
      if (key === "deepen") {
        deepened = true;
        slice_ = Math.min(remainingTime * 0.5, Math.max(500, p * remainingTime * 1.5));
        order = _deepen(ctx, res, bias, reservoir, order, now + slice_);
        trace.push({ action: "deepen", p: p, slice: slice_, fit_after: reservoir.a.length });
        continue;
      }
      mod = pool.get(key);
      pool.delete(key);
      var nLeft = pool.size;
      var floor = Math.min(120, remainingTime / Math.max(1, nLeft + 1));
      var want = p * remainingTime * 1.7;
      var cap = remainingTime - floor * nLeft;
      slice_ = Math.max(floor, Math.min(want, Math.max(floor, cap)));
      var before = reservoir.a.length;
      order = _harvest(mod, ctx, now + slice_, bias, reservoir, order, res);
      ran.push(key);
      trace.push({ action: key, p: p, slice: slice_, fit_gain: reservoir.a.length - before });
    }
  }
  res.diagnostics.plan = trace;
  return order;
}

/* Keep all demonstrated shape laws when training does not distinguish them. */
function _shapeOptions(ctx, tg) {
  var shapes = new Map();
  if (!ctx.train.length) return shapes;
  function add(h, w) { shapes.set(h + "," + w, [h, w]); }
  var th = tg.length, tw = tg[0].length;
  var cs = ctx.const_out_shape();
  if (cs) add(cs[0], cs[1]);
  if (ctx.same_shape()) add(th, tw);
  var sr = ctx.shape_ratio();
  if (sr) add(th * sr[0], tw * sr[1]);
  var ir = ctx.inv_shape_ratio();
  if (ir && !(th % ir[0]) && !(tw % ir[1])) add(th / ir[0], tw / ir[1]);
  var flipped = true, i, a, b;
  for (i = 0; i < ctx.train.length; i++) {
    a = ctx.train[i][0]; b = ctx.train[i][1];
    if (!(b.length === a[0].length && b[0].length === a.length)) { flipped = false; break; }
  }
  if (flipped) add(tw, th);
  return shapes;
}

function solveInner(train, testInputs, timeBudget, k, loo, modules, collectAll) {
  timeBudget = Number(timeBudget);
  if (!isFinite(timeBudget) || timeBudget < 0) throw new Error("time_budget must be finite and nonnegative");
  if (typeof k !== "number" || (k | 0) !== k || k < 0) throw new Error("k must be a nonnegative integer");
  var mods = modules ? modules.slice() : orderedModules();
  var t0 = nowMs(), deadline = t0 + timeBudget * 1000;
  var ctx = new Ctx(train, testInputs, deadline);
  var res = new Result(), bias = {}, i, j;
  var POLICY = PORTFOLIO_STATE.POLICY;
  if (POLICY !== null && POLICY !== undefined) {
    try {
      var sigs = signatures(ctx);
      bias = POLICY.bias_for(sigs);
      ctx.op_prior = POLICY.op_bias;
      if (POLICY.module_order && POLICY.module_order.length) {
        var rank = {};
        for (i = 0; i < POLICY.module_order.length; i++) rank[POLICY.module_order[i]] = i;
        mods = mods.map(function (m, idx) { return [m, idx]; });
        mods.sort(function (a, b) {
          var ra = rank[a[0].SOLVER] === undefined ? 99 : rank[a[0].SOLVER];
          var rb = rank[b[0].SOLVER] === undefined ? 99 : rank[b[0].SOLVER];
          return (ra - rb) || (a[1] - b[1]);
        });
        mods = mods.map(function (x) { return x[0]; });
      }
      var skip = POLICY.skips_for(sigs), kept = [];
      for (i = 0; i < mods.length; i++) if (!skip[mods[i].SOLVER]) kept.push(mods[i]);
      if (kept.length) mods = kept;
    } catch (exc) {
      bias = {};
      res.diagnostics.policy_error = String(exc && exc.name ? exc.name : exc);
    }
  }

  /* Cheap modules donate unused time to later search. Reserve enough budget to
     evaluate test predictions and to refit rather than silently skip LOO. */
  var reserve = (loo && ctx.train.length >= 3) ? Math.min(3000, timeBudget * 1000 * 0.15) : 0.0;
  var generationEnd = deadline - reserve - Math.min(200, timeBudget * 1000 * 0.03);
  var phase1 = [], phase2 = [];
  for (i = 0; i < mods.length; i++) ((mods[i].PHASE === 2) ? phase2 : phase1).push(mods[i]);
  var reservoir = new _MinHeap(_itemCmp), order = 0;
  var plan = _plannerFor(ctx, res);
  if (plan !== null) {
    order = _plannedGeneration(ctx, res, plan, phase1, phase2, bias, reservoir, order, t0, generationEnd);
  } else {
    var p1End = t0 + (generationEnd - t0) * (phase2.length ? 0.45 : 1.0);
    var share1 = (p1End - t0) / Math.max(1, phase1.length);
    var all = phase1.concat(phase2), moduleEnd, now, remaining;
    for (i = 0; i < all.length; i++) {
      now = nowMs();
      if (now >= generationEnd) break;
      if (i < phase1.length)
        moduleEnd = Math.min(generationEnd, Math.max(now + share1 * 0.5, t0 + share1 * (i + 1)));
      else {
        remaining = all.length - i;
        moduleEnd = Math.min(generationEnd, now + (generationEnd - now) / remaining);
      }
      order = _harvest(all[i], ctx, moduleEnd, bias, reservoir, order, res);
    }
  }

  var fitted = [];
  for (i = 0; i < reservoir.a.length; i++) {
    var it = reservoir.a[i];
    fitted.push([-it[0], -it[1], it[2], it[3]]);
  }
  fitted.sort(function (a, b) { return (a[0] - b[0]) || (a[1] - b[1]); });
  res.n_fit = fitted.length;
  res.diagnostics.total_fitted = order;
  ctx.deadline = deadline;

  /* Cache predictions once; deduplication below prevents aliases from crowding
     distinct behaviours out of the finite voting pool. */
  var sigsByIdx = new Map();
  for (i = 0; i < fitted.length; i++) {
    if (sigsByIdx.size && ctx.timed_out()) break;
    var sig = [];
    for (j = 0; j < ctx.test_inputs.length; j++) sig.push(_prediction(fitted[i][2], ctx.test_inputs[j]));
    sigsByIdx.set(fitted[i][1], sig);
  }
  var kept2 = [];
  for (i = 0; i < fitted.length; i++) if (sigsByIdx.has(fitted[i][1])) kept2.push(fitted[i]);
  fitted = kept2;

  if (loo && ctx.train.length >= 3) {
    var groups = new Map(), selected = new Set(), rec, key, fullKey, modIdx = new Map(), nextModId = 0;
    function modId(m) {
      if (!modIdx.has(m)) modIdx.set(m, nextModId++);
      return modIdx.get(m);
    }
    for (i = 0; i < fitted.length; i++) {
      rec = fitted[i];
      var sk = _sigKey(sigsByIdx.get(rec[1]));
      key = rec[2].solver + "" + rec[2].name + "" + sk;
      fullKey = modId(rec[3]) + "" + key;
      if (!selected.has(fullKey) && selected.size < 12) {
        selected.add(fullKey);
        if (!groups.has(rec[3])) groups.set(rec[3], []);
        groups.get(rec[3]).push({ key: key, solver: rec[2].solver, name: rec[2].name, sigKey: sk });
      }
    }
    var adjustments = new Map();
    var groupList = [];
    groups.forEach(function (targets, mod) { groupList.push([mod, targets]); });
    for (i = 0; i < groupList.length; i++) {
      if (ctx.timed_out()) break;
      var evid = _looEvidence(groupList[i][0], ctx, groupList[i][1]);
      var targets = groupList[i][1];
      for (j = 0; j < targets.length; j++) {
        var wt = evid.get(targets[j].key), wins = wt[0], trials = wt[1];
        /* Partial cross-validation carries proportionally less weight. */
        var adjustment = trials ? ((1.5 - 4.5 * wins / trials) * trials / ctx.train.length) : 0.0;
        adjustments.set(modId(groupList[i][0]) + "" + targets[j].key, adjustment);
        res.diagnostics.loo.push({ solver: targets[j].solver, name: targets[j].name,
          wins: wins, trials: trials, folds: ctx.train.length, adjustment: adjustment });
      }
    }
    for (i = 0; i < fitted.length; i++) {
      rec = fitted[i];
      key = modId(rec[3]) + "" + rec[2].solver + "" + rec[2].name + "" +
            _sigKey(sigsByIdx.get(rec[1]));
      rec[0] += adjustments.has(key) ? adjustments.get(key) : 0.0;
    }
  }
  fitted.sort(function (a, b) { return (a[0] - b[0]) || (a[1] - b[1]); });
  res.hyps = [];
  for (i = 0; i < Math.min(8, fitted.length); i++)
    res.hyps.push([fitted[i][2].solver + ":" + fitted[i][2].name, Math.round(fitted[i][0] * 100) / 100]);

  var pool = [], seenBehaviours = new Set();
  for (i = 0; i < fitted.length; i++) {
    var signature = sigsByIdx.get(fitted[i][1]);
    var bkey = fitted[i][2].solver + "" + _sigKey(signature);
    var anyGrid = false;
    for (j = 0; j < signature.length; j++) if (signature[j] !== null) { anyGrid = true; break; }
    if (!seenBehaviours.has(bkey) && anyGrid) {
      seenBehaviours.add(bkey);
      pool.push(fitted[i]);
      if (pool.length >= 150) break;
    }
  }
  res.diagnostics.voting_hypotheses = pool.length;

  var preservesColors = true;
  for (i = 0; i < ctx.train.length; i++)
    if (!G.csSubset(G.palette(ctx.train[i][1]), G.palette(ctx.train[i][0]))) { preservesColors = false; break; }

  var ti;
  for (ti = 0; ti < ctx.test_inputs.length; ti++) {
    var tg = ctx.test_inputs[ti];
    var shapes = _shapeOptions(ctx, tg);
    var allowed = preservesColors ? G.csUnion(G.palette(tg), ctx.out_palette()) : null;
    var best = new Map(), first = new Map(), author = new Map(), gridByKey = new Map();
    var rank2;
    for (rank2 = 0; rank2 < pool.length; rank2++) {
      var score = pool[rank2][0], idx = pool[rank2][1], hyp = pool[rank2][2];
      var gg = sigsByIdx.get(idx)[ti];
      if (gg === null) continue;
      var gk = G.gkey(gg);
      if (!gridByKey.has(gk)) gridByKey.set(gk, gg);
      if (!author.has(gk)) author.set(gk, [hyp.solver, hyp.name]);
      if (!best.has(gk)) best.set(gk, new Map());
      var fam2 = best.get(gk);
      if (!fam2.has(hyp.solver) || score < fam2.get(hyp.solver)) fam2.set(hyp.solver, score);
      if (!first.has(gk)) first.set(gk, rank2);
    }
    var scored = [];
    best.forEach(function (families, gk) {
      var logits = [], peak = -Infinity;
      families.forEach(function (s) { var v = -s / 2.0; logits.push(v); if (v > peak) peak = v; });
      var sum = 0, m;
      for (m = 0; m < logits.length; m++) sum += Math.exp(logits[m] - peak);
      var weight = peak + Math.log(sum);
      var gg2 = gridByKey.get(gk);
      var violations = 0;
      if (shapes.size && !shapes.has(gg2.length + "," + gg2[0].length)) violations += 1;
      if (allowed !== null && !G.csSubset(G.palette(gg2), allowed)) violations += 1;
      weight += violations * Math.log(0.25);
      scored.push([-weight, first.get(gk), gg2, violations, families.size]);
    });
    scored.sort(function (a, b) { return (a[0] - b[0]) || (a[1] - b[1]); });
    var predictions = [];
    for (i = 0; i < scored.length; i++) predictions.push(scored[i][2]);
    res.predictions.push(collectAll ? predictions : predictions.slice(0, k));
    res.chosen.push(predictions.length ? author.get(G.gkey(predictions[0])) : null);
    res.diagnostics.predictions.push({
      distinct: scored.length,
      top_support: scored.length ? scored[0][4] : 0,
      top_violations: scored.length ? scored[0][3] : 0,
      log_weight_margin: scored.length > 1 ? (scored[1][0] - scored[0][0]) : null
    });
  }
  res.solver = null;
  for (i = 0; i < res.chosen.length; i++) if (res.chosen[i]) { res.solver = res.chosen[i][0]; break; }
  res.elapsed = (nowMs() - t0) / 1000;
  res.diagnostics.timed_out = res.elapsed > timeBudget;
  res.diagnostics.unrun_modules = mods.length - res.diagnostics.modules.length;
  return res;
}

var solve = hypcacheScoped(function (train, testInputs, opts) {
  opts = opts || {};
  return solveInner(train, testInputs,
                    opts.time_budget === undefined ? 30.0 : opts.time_budget,
                    opts.k === undefined ? 2 : opts.k,
                    opts.loo === undefined ? true : opts.loo,
                    opts.modules || null,
                    !!opts.collect_all);
});
