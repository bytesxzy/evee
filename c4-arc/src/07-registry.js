/* Solver module registry. Each solver source appends one module object; the
 * portfolio reads this list exactly as the Python engine reads _REGISTRY. */

var SOLVER_MODULES = [];
var PORTFOLIO_STATE = { POLICY: null };

function defSolver(moduleName, solverFamily, generate, phase) {
  var mod = {
    __name__: moduleName,
    SOLVER: solverFamily,
    PHASE: phase === undefined ? 1 : phase,
    generate: generate
  };
  SOLVER_MODULES.push(mod);
  return mod;
}

function mkHyp(solverFamily) {
  return function (name, fn, cost) { return new Hyp(name, fn, cost, solverFamily); };
}

function moduleByName(name) {
  var i;
  for (i = 0; i < SOLVER_MODULES.length; i++)
    if (SOLVER_MODULES[i].__name__ === name) return SOLVER_MODULES[i];
  return null;
}

/* Python compares tuples of tuples of ints element-wise; several solvers use
   that ordering as a deterministic tie-break, so it is reproduced here. */
function cmpRow(a, b) {
  var n = Math.min(a.length, b.length), i;
  for (i = 0; i < n; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  return a.length - b.length;
}

function cmpGrid(a, b) {
  var n = Math.min(a.length, b.length), i, r;
  for (i = 0; i < n; i++) { r = cmpRow(a[i], b[i]); if (r) return r; }
  return a.length - b.length;
}
