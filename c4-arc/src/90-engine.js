/* Public surface of the bundle. */

function loadPolicy(data) { return new Policy(data); }
function loadPlanner(data) { return data ? new Planner(data) : null; }

/* Solve one ARC task. ``task`` is the standard ARC JSON shape:
   {train: [{input, output}, ...], test: [{input, output?}, ...]}.
   Test outputs, when present, are never read here -- the caller scores. */
function solveTask(task, opts) {
  opts = opts || {};
  var train = [], testInputs = [], i;
  for (i = 0; i < task.train.length; i++) train.push([task.train[i].input, task.train[i].output]);
  for (i = 0; i < task.test.length; i++) testInputs.push(task.test[i].input);
  return solve(train, testInputs, opts);
}

/* True when every test pair of the task is reproduced within the top ``k``
   ranked guesses. A task counts as solved only when every pair is exact. */
function scoreTask(task, res, k) {
  if (k === undefined) k = 2;
  var i, j, ok1 = true, okk = true, ans, preds, hit;
  for (i = 0; i < task.test.length; i++) {
    if (task.test[i].output === undefined) return null;
    ans = G.asGrid(task.test[i].output);
    preds = res.predictions[i] || [];
    if (!preds.length || !G.gEq(preds[0], ans)) ok1 = false;
    hit = false;
    for (j = 0; j < Math.min(k, preds.length); j++) if (G.gEq(preds[j], ans)) { hit = true; break; }
    if (!hit) okk = false;
  }
  return { top1: ok1, topk: okk };
}

var ENGINE = {
  G: G, O: O, OPS: OPS, ENUM: ENUM, LEARN: LEARN, PLANNER: PLANNER, PART: PART,
  TILING: TILING, SYMM: SYMM, REGIONS: REGIONS, SEQ: SEQ,
  Ctx: Ctx, Hyp: Hyp, Result: Result,
  SOLVER_PRIOR: SOLVER_PRIOR, SOLVER_MODULES: SOLVER_MODULES,
  orderedModules: orderedModules,
  solve: solve, solveTask: solveTask, scoreTask: scoreTask,
  signatures: signatures,
  loadPolicy: loadPolicy, loadPlanner: loadPlanner,
  activatePolicy: activatePolicy, activatePlanner: activatePlanner,
  moduleNames: function () {
    var out = [], i, mods = orderedModules();
    for (i = 0; i < mods.length; i++) out.push(mods[i].__name__);
    return out;
  }
};
