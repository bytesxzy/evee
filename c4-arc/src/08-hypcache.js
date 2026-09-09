/* Port of engine/hypcache.py -- per-solve reuse of completed generation.
 *
 * Partial, expired and failed streams are never cached. Cache lifetime is one
 * top-level solve; transformed and leave-one-out contexts have distinct keys.
 */

var HYPCACHE_ENABLED = true;
var _HC_STATE = null;

var _HC_CACHEABLE = {};
(function () {
  var names = ["geometry", "colormap", "partition", "symmetry", "tiling",
    "blocks", "select", "regions", "counting", "cellwise", "objects_map",
    "objproc", "relproc", "motion", "substitute", "sequence", "paint", "patterns",
    "analogy", "assemble", "paneltable", "selfstamp", "tally", "extend", "locate"];
  var i;
  for (i = 0; i < names.length; i++) _HC_CACHEABLE[names[i]] = true;
})();

function hypcacheScoped(fn) {
  return function () {
    var state = HYPCACHE_ENABLED ? { items: new Map(), hits: 0, misses: 0, stored_hyps: 0 } : null;
    var prev = _HC_STATE;
    _HC_STATE = state;
    try {
      var result = fn.apply(null, arguments);
      if (state && result && result.diagnostics) {
        result.diagnostics.hypothesis_cache = {
          hits: state.hits, misses: state.misses,
          stored_hyps: state.stored_hyps, entries: state.items.size
        };
      }
      return result;
    } finally {
      _HC_STATE = prev;
    }
  };
}

function _ctxKey(ctx) {
  var parts = [], i;
  for (i = 0; i < ctx.train.length; i++)
    parts.push(G.gkey(ctx.train[i][0]), G.gkey(ctx.train[i][1]));
  parts.push("|");
  for (i = 0; i < ctx.test_inputs.length; i++) parts.push(G.gkey(ctx.test_inputs[i]));
  return parts.join("~");
}

function hypcacheGenerate(module, ctx) {
  var state = _HC_STATE, name = module.__name__ || "";
  if (state === null || !_HC_CACHEABLE[name]) return module.generate(ctx);
  var priorKey = "", k, keys = [];
  if (ctx.op_prior) {
    for (k in ctx.op_prior) if (Object.prototype.hasOwnProperty.call(ctx.op_prior, k)) keys.push(k);
    keys.sort();
    for (k = 0; k < keys.length; k++) priorKey += keys[k] + "=" + ctx.op_prior[keys[k]] + ";";
  }
  var key = name + "||" + _ctxKey(ctx) + "||" + priorKey;
  var cached = state.items.get(key);
  if (cached !== undefined) { state.hits++; return cached; }
  state.misses++;
  var complete = module.generate(ctx);
  if (!ctx.timed_out() && state.items.size < 96 &&
      state.stored_hyps + complete.length <= 12000) {
    state.items.set(key, complete);
    state.stored_hyps += complete.length;
  }
  return complete;
}
