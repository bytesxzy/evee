/* Measurement harness for the JavaScript engine.
 *
 * Mirrors bench/run_arc.py: a solver is handed train pairs and test inputs
 * only, the answers stay here, and a task counts as solved only when every
 * test pair is exact. Workers are forked so the 550-task corpus finishes in
 * reasonable wall-clock time; each worker is otherwise identical to a browser.
 *
 *   node bench.js --budget 20 --jobs 4 --out ev/JS_FULL.json [--tasks DIR]
 *   node bench.js --worker            (internal)
 */

var fs = require("fs");
var path = require("path");
var cp = require("child_process");

var ROOT = path.resolve(__dirname, "..");
var ENGINE_PATH = path.join(ROOT, "c4-arc-engine.js");
var DATA = path.join(ROOT, "astron", "xyz", "data", "arc");
var PLANNER = path.join(ROOT, "astron", "xyz", "policy", "arc_planner.json");

function parseArgs(argv) {
  var a = { budget: 20, jobs: 4, k: 2, out: null, tasks: DATA, limit: 0,
            filter: "", planner: PLANNER, worker: false, quiet: false };
  var i;
  for (i = 0; i < argv.length; i++) {
    if (argv[i] === "--worker") a.worker = true;
    else if (argv[i] === "--quiet") a.quiet = true;
    else if (argv[i] === "--budget") a.budget = parseFloat(argv[++i]);
    else if (argv[i] === "--jobs") a.jobs = parseInt(argv[++i], 10);
    else if (argv[i] === "--k") a.k = parseInt(argv[++i], 10);
    else if (argv[i] === "--out") a.out = argv[++i];
    else if (argv[i] === "--tasks") a.tasks = argv[++i];
    else if (argv[i] === "--limit") a.limit = parseInt(argv[++i], 10);
    else if (argv[i] === "--filter") a.filter = argv[++i];
    else if (argv[i] === "--planner") a.planner = argv[++i];
    else if (argv[i] === "--no-planner") a.planner = "";
  }
  return a;
}

function runOne(E, taskPath, budget, k) {
  var task = JSON.parse(fs.readFileSync(taskPath, "utf8"));
  var id = path.basename(taskPath).replace(/\.json$/, "");
  var t0 = Date.now(), res, err = null;
  try {
    res = E.solveTask(task, { time_budget: budget, k: k });
  } catch (e) {
    err = String(e && e.message ? e.message : e).slice(0, 200);
    res = { predictions: [], hyps: [], solver: null, chosen: [], n_fit: 0 };
  }
  var el = (Date.now() - t0) / 1000;
  var top1 = true, top2 = true, i, j, ans, preds, hit;
  for (i = 0; i < task.test.length; i++) {
    ans = E.G.asGrid(task.test[i].output);
    preds = res.predictions[i] || [];
    if (!preds.length || !E.G.gEq(preds[0], ans)) top1 = false;
    hit = false;
    for (j = 0; j < Math.min(k, preds.length); j++) if (E.G.gEq(preds[j], ans)) { hit = true; break; }
    if (!hit) top2 = false;
  }
  var part = 0.0, n, m, r, c;
  for (i = 0; i < task.test.length; i++) {
    ans = E.G.asGrid(task.test[i].output);
    preds = res.predictions[i] || [];
    if (preds.length && preds[0].length === ans.length && preds[0][0].length === ans[0].length) {
      n = E.G.area(ans); m = 0;
      for (r = 0; r < ans.length; r++) for (c = 0; c < ans[r].length; c++) if (preds[0][r][c] === ans[r][c]) m++;
      part += m / n;
    }
  }
  part /= Math.max(1, task.test.length);
  return { id: id, solved: top1 ? 1 : 0, solved2: top2 ? 1 : 0,
           partial: Math.round(part * 10000) / 10000,
           time: Math.round(el * 1000) / 1000,
           solver: res.solver, error: err, n_test: task.test.length,
           n_fit: res.n_fit };
}

function main() {
  var a = parseArgs(process.argv.slice(2));
  var E = require(ENGINE_PATH);
  if (a.planner) {
    try { E.activatePlanner(E.loadPlanner(JSON.parse(fs.readFileSync(a.planner, "utf8")))); }
    catch (e) { console.error("planner load failed: " + e.message); }
  }

  if (a.worker) {
    var lines = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", function (d) { lines += d; });
    process.stdin.on("end", function () {
      var files = lines.split("\n").filter(function (s) { return s.length; });
      var out = [], i;
      for (i = 0; i < files.length; i++) {
        out.push(runOne(E, files[i], a.budget, a.k));
        process.stderr.write(".");
      }
      process.stdout.write(JSON.stringify(out));
    });
    return;
  }

  var files = fs.readdirSync(a.tasks).filter(function (f) { return /\.json$/.test(f); }).sort();
  if (a.filter) files = files.filter(function (f) { return f.indexOf(a.filter) >= 0; });
  if (a.limit) files = files.slice(0, a.limit);
  var paths = files.map(function (f) { return path.join(a.tasks, f); });

  if (a.jobs <= 1) {
    var out = [], i;
    for (i = 0; i < paths.length; i++) {
      out.push(runOne(E, paths[i], a.budget, a.k));
      if (!a.quiet) console.log(out[out.length - 1].id, out[out.length - 1].solved,
                                out[out.length - 1].time, out[out.length - 1].solver || "");
    }
    finish(out, a);
    return;
  }

  var buckets = [], j;
  for (j = 0; j < a.jobs; j++) buckets.push([]);
  for (j = 0; j < paths.length; j++) buckets[j % a.jobs].push(paths[j]);
  var results = [], done = 0, t0 = Date.now();
  buckets.forEach(function (bucket, bi) {
    var args = ["--worker", "--budget", String(a.budget), "--k", String(a.k)];
    if (a.planner) args.push("--planner", a.planner); else args.push("--no-planner");
    var child = cp.spawn(process.execPath, [__filename].concat(args),
                         { stdio: ["pipe", "pipe", "inherit"] });
    var buf = "";
    child.stdout.on("data", function (d) { buf += d; });
    child.on("close", function () {
      try { results = results.concat(JSON.parse(buf)); }
      catch (e) { console.error("worker " + bi + " produced no result"); }
      done += 1;
      if (done === a.jobs) {
        console.error("");
        console.error("wall " + Math.round((Date.now() - t0) / 1000) + "s");
        finish(results, a);
      }
    });
    child.stdin.write(bucket.join("\n"));
    child.stdin.end();
  });
}

function finish(out, a) {
  out.sort(function (x, y) { return x.id < y.id ? -1 : (x.id > y.id ? 1 : 0); });
  var n = out.length, s1 = 0, s2 = 0, a1 = 0, a2 = 0, n1 = 0, n2 = 0, part = 0, errs = 0, i;
  for (i = 0; i < n; i++) {
    s1 += out[i].solved; s2 += out[i].solved2; part += out[i].partial;
    if (out[i].error) errs++;
    if (out[i].id.indexOf("arc1") === 0) { n1++; a1 += out[i].solved; }
    if (out[i].id.indexOf("arc2") === 0) { n2++; a2 += out[i].solved; }
  }
  var summary = {
    n: n, solved: s1, solved_top2: s2,
    rate: Math.round(s1 / n * 10000) / 10000,
    rate_top2: Math.round(s2 / n * 10000) / 10000,
    arc1: { n: n1, solved: a1, rate: n1 ? Math.round(a1 / n1 * 10000) / 10000 : 0 },
    arc2: { n: n2, solved: a2, rate: n2 ? Math.round(a2 / n2 * 10000) / 10000 : 0 },
    partial_mean: Math.round(part / n * 10000) / 10000,
    errors: errs, budget: a.budget, k: a.k,
    planner: a.planner || null
  };
  console.log(JSON.stringify(summary, null, 1));
  if (a.out) {
    fs.writeFileSync(a.out, JSON.stringify(Object.assign({}, summary, { results: out })));
    console.log("wrote " + a.out);
  }
}

main();
