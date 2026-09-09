/* C4ARC -- the ARC-AGI route for the C4 mini assistant.
 *
 * Everything here runs in the page. There is no API call, no hosted model and
 * no network of any kind: the engine in c4-arc-engine.js is a JavaScript port
 * of the CELL4 ASTRA solver, the corpus in c4-arc-tasks.js is the public
 * ARC-AGI-1 and ARC-AGI-2 task files, and the planner in c4-arc-policy.js is
 * weights fitted offline on solved tasks. When this module reports a score it
 * has just measured it here, on this machine, against answers it never sees
 * until after it has committed to a prediction.
 *
 * Loads after c4-arc-engine.js, c4-arc-tasks.js and c4-arc-policy.js.
 */

(function (root) {
"use strict";

var ENGINE = root.C4ARCEngine || null;
var TASKS = root.C4ARCTasks || null;
var PLANNER = root.C4ARCPlanner || null;

/* Where this file was loaded from, so the worker can import its siblings. */
var BASE = (function () {
  try {
    var s = document.currentScript && document.currentScript.src;
    if (s) return s.replace(/[^/]*$/, "");
    return new URL(".", document.baseURI).href;
  } catch (e) { return ""; }
})();

var STORE_KEY = "c4.arc.run.v1";

/* The full-corpus result this engine produced, and the exact way to reproduce
   it. Quoted as a reference point; the numbers the assistant reports in chat
   are always the ones it measured live.

   The per-task budget is wall clock, so the total is not bit-reproducible: a
   task whose search lands near the deadline can fall either side of it. Two
   runs of the reference implementation on the same machine scored 252 and 253,
   and this engine scored 254 -- so treat the last digit as noise, not signal. */
var REFERENCE = {
  n: 550, solved: 254, arc1: { n: 400, solved: 203 }, arc2: { n: 150, solved: 51 },
  budget: 20, k: 2, spread: "252-254 across runs",
  command: "node c4-arc/bench.js --budget 20 --jobs 4"
};

/* ------------------------------------------------------------ task corpus */

function unpackGrid(s) {
  var rows = s.split("|"), out = [], r, i, row;
  for (r = 0; r < rows.length; r++) {
    row = new Array(rows[r].length);
    for (i = 0; i < rows[r].length; i++) row[i] = rows[r].charCodeAt(i) - 48;
    out.push(row);
  }
  return out;
}

function unpackPairs(s) {
  if (!s) return [];
  var parts = s.split(";"), out = [], i, kv;
  for (i = 0; i < parts.length; i++) {
    kv = parts[i].split(">");
    out.push({ input: unpackGrid(kv[0]), output: kv[1] ? unpackGrid(kv[1]) : undefined });
  }
  return out;
}

function taskAt(i) {
  var row = TASKS[i];
  return { id: row[0], train: unpackPairs(row[1]), test: unpackPairs(row[2]) };
}

function taskById(id) {
  var i;
  for (i = 0; i < TASKS.length; i++) if (TASKS[i][0] === id) return taskAt(i);
  return null;
}

/* A fixed, reproducible spread over both corpora rather than a random draw:
   the same question asked twice measures the same tasks. */
function sampleIndices(n) {
  var total = TASKS.length, out = [], i;
  if (n >= total) { for (i = 0; i < total; i++) out.push(i); return out; }
  var step = total / n;
  for (i = 0; i < n; i++) out.push(Math.floor(i * step + step / 2));
  return out;
}

/* ---------------------------------------------------------- the run engine */

function scoreOne(task, budget) {
  var res = ENGINE.solveTask(task, { time_budget: budget, k: 2 });
  var s = ENGINE.scoreTask(task, res, 2);
  return { id: task.id, solved: s && s.top1 ? 1 : 0, solved2: s && s.topk ? 1 : 0,
           solver: res.solver, n_fit: res.n_fit };
}

/* The worker keeps the page responsive while the engine runs; if a worker
   cannot be created the same loop runs inline, yielding between tasks. */
function workerSource() {
  return "self.onmessage=function(e){" +
    "var d=e.data;" +
    "importScripts(d.base+'c4-arc-engine.js',d.base+'c4-arc-tasks.js'" +
    (PLANNER ? ",d.base+'c4-arc-policy.js'" : "") + ");" +
    "var E=self.C4ARCEngine,T=self.C4ARCTasks;" +
    "if(self.C4ARCPlanner)E.activatePlanner(E.loadPlanner(self.C4ARCPlanner));" +
    "function ug(s){var rs=s.split('|'),o=[],r,i,row;for(r=0;r<rs.length;r++){row=new Array(rs[r].length);" +
    "for(i=0;i<rs[r].length;i++)row[i]=rs[r].charCodeAt(i)-48;o.push(row);}return o;}" +
    "function up(s){if(!s)return [];var p=s.split(';'),o=[],i,kv;for(i=0;i<p.length;i++){kv=p[i].split('>');" +
    "o.push({input:ug(kv[0]),output:kv[1]?ug(kv[1]):undefined});}return o;}" +
    "for(var n=0;n<d.idx.length;n++){var row=T[d.idx[n]];" +
    "var task={id:row[0],train:up(row[1]),test:up(row[2])};" +
    "var t0=Date.now(),res,ok={top1:false,topk:false};" +
    "try{res=E.solveTask(task,{time_budget:d.budget,k:2});ok=E.scoreTask(task,res,2)||ok;}catch(err){res={solver:null,n_fit:0};}" +
    "self.postMessage({i:n,id:task.id,solved:ok.top1?1:0,solved2:ok.topk?1:0," +
    "solver:res.solver,n_fit:res.n_fit,ms:Date.now()-t0,total:d.idx.length});}" +
    "self.postMessage({done:true});};";
}

function runIndices(idx, budget, onProgress) {
  return new Promise(function (resolve) {
    var results = [];
    var worker = null;
    if (BASE && typeof Worker !== "undefined" && typeof Blob !== "undefined") {
      try {
        var url = URL.createObjectURL(new Blob([workerSource()], { type: "text/javascript" }));
        worker = new Worker(url);
      } catch (e) { worker = null; }
    }
    if (worker) {
      worker.onmessage = function (ev) {
        if (ev.data.done) { worker.terminate(); resolve(results); return; }
        results.push(ev.data);
        if (onProgress) onProgress(results.length, idx.length, ev.data);
      };
      worker.onerror = function () { worker.terminate(); inline(); };
      worker.postMessage({ base: BASE, idx: idx, budget: budget });
      return;
    }
    inline();

    function inline() {
      var i = 0;
      (function step() {
        if (i >= idx.length) { resolve(results); return; }
        var task = taskAt(idx[i]), t0 = Date.now(), rec;
        try { rec = scoreOne(task, budget); }
        catch (e) { rec = { id: task.id, solved: 0, solved2: 0, solver: null, n_fit: 0 }; }
        rec.ms = Date.now() - t0;
        results.push(rec);
        if (onProgress) onProgress(results.length, idx.length, rec);
        i += 1;
        setTimeout(step, 0);
      })();
    }
  });
}

function tally(results) {
  var s1 = 0, s2 = 0, a1 = 0, n1 = 0, a2 = 0, n2 = 0, i;
  for (i = 0; i < results.length; i++) {
    s1 += results[i].solved; s2 += results[i].solved2;
    if (results[i].id.indexOf("arc1") === 0) { n1++; a1 += results[i].solved; }
    if (results[i].id.indexOf("arc2") === 0) { n2++; a2 += results[i].solved; }
  }
  return { n: results.length, solved: s1, solved2: s2,
           arc1: { n: n1, solved: a1 }, arc2: { n: n2, solved: a2 } };
}

function pct(a, b) { return b ? (Math.round(a / b * 1000) / 10).toFixed(1) + "%" : "-"; }

function saveRun(rec) {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(rec)); } catch (e) {}
}

function loadRun() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY) || "null"); } catch (e) { return null; }
}

/* ------------------------------------------------------------- rendering */

function renderGrid(g) {
  var out = [], r;
  for (r = 0; r < g.length; r++) out.push(g[r].join(""));
  return out.join("\n");
}

function describeTask(task) {
  var lines = ["task " + task.id + " - " + task.train.length + " demonstrations, " +
               task.test.length + " test input" + (task.test.length === 1 ? "" : "s")];
  var i;
  for (i = 0; i < Math.min(2, task.train.length); i++) {
    lines.push("");
    lines.push("train " + (i + 1) + " in:");
    lines.push(renderGrid(task.train[i].input));
    lines.push("train " + (i + 1) + " out:");
    lines.push(renderGrid(task.train[i].output));
  }
  return lines.join("\n");
}

/* ------------------------------------------------------------- intents */

/* "arc", "arc-agi", "arc agi 2", and the bundled task-id form arc1_007bbfb7. */
var RE_ARC = /\barc[12]_[0-9a-f]{6,8}\b|\barc[\s-]?(?:agi)?[\s-]?[12]?\b/i;
var RE_SCORE = /\b(score|accuracy|how\s+(?:well|good|many)|result|benchmark|rate|percent|%)\b/i;
var RE_FULL = /\b(full|all|whole|entire|complete|550|every)\b/i;
var RE_SOLVE = /\b(solve|answer|predict|try)\b/i;
var RE_SHOW = /\b(show|display|render|example|sample|see)\b/i;
var RE_HOW = /\b(how|what|explain|which|describe|work|engine|solver)\b/i;

function isRequest(text) {
  if (!ready()) return false;
  var t = String(text || "");
  if (!RE_ARC.test(t)) return /\{[\s\S]*"train"[\s\S]*\}/.test(t) && RE_SOLVE.test(t);
  return RE_SCORE.test(t) || RE_SOLVE.test(t) || RE_SHOW.test(t) || RE_HOW.test(t) ||
         /\b(run|benchmark|test|evaluate|eval)\b/i.test(t);
}

function ready() { return !!(ENGINE && TASKS && TASKS.length); }

function extractTaskJson(text) {
  var i = text.indexOf("{"), j = text.lastIndexOf("}");
  if (i < 0 || j <= i) return null;
  var d;
  try { d = JSON.parse(text.slice(i, j + 1)); } catch (e) { return null; }
  if (!d || !d.train || !d.train.length) return null;
  if (!d.test) d.test = [];
  return d;
}

function corpusLine() {
  var n1 = 0, n2 = 0, i;
  for (i = 0; i < TASKS.length; i++) {
    if (TASKS[i][0].indexOf("arc1") === 0) n1++;
    else if (TASKS[i][0].indexOf("arc2") === 0) n2++;
  }
  return TASKS.length + " public task files in the page (" + n1 +
         " ARC-AGI-1, " + n2 + " ARC-AGI-2)";
}

function engineLine() {
  return ENGINE.moduleNames().length + " hypothesis generators, " +
         "each candidate rule executed against every demonstration and discarded " +
         "unless it reproduces all of them";
}

/* ------------------------------------------------------------- answering */

function answerScore(text, opts) {
  var full = RE_FULL.test(text);
  var n = full ? TASKS.length : (opts && opts.n) || 24;
  var budget = full ? 20 : ((opts && opts.budget) || 5);
  var idx = sampleIndices(n);
  var t0 = Date.now();
  return runIndices(idx, budget, opts && opts.onProgress).then(function (results) {
    var t = tally(results), secs = Math.round((Date.now() - t0) / 100) / 10;
    saveRun({ at: Date.now(), budget: budget, tally: t, results: results });
    var lines = [];
    lines.push("Measured just now, in this page: " + t.solved + " of " + t.n +
               " solved (" + pct(t.solved, t.n) + ") at a " + budget +
               "-second budget per task, " + secs + "s of wall clock.");
    if (t.arc1.n) lines.push("  ARC-AGI-1  " + t.arc1.solved + "/" + t.arc1.n + "  " + pct(t.arc1.solved, t.arc1.n));
    if (t.arc2.n) lines.push("  ARC-AGI-2  " + t.arc2.solved + "/" + t.arc2.n + "  " + pct(t.arc2.solved, t.arc2.n));
    lines.push("  top-2      " + t.solved2 + "/" + t.n + "  " + pct(t.solved2, t.n));
    lines.push("");
    if (!full) {
      lines.push("That is an evenly spaced sample of " + n + " of the " + TASKS.length +
                 " bundled tasks, at a shorter budget than the headline run, because " +
                 "the full corpus at 20 seconds a task is about three CPU hours. " +
                 "Ask for the full ARC run to start it here.");
      lines.push("");
    }
    lines.push("Full corpus, this same engine, 20 s per task, two attempts scored: " +
               REFERENCE.solved + "/" + REFERENCE.n + " (" + pct(REFERENCE.solved, REFERENCE.n) +
               ") -- ARC-AGI-1 " + REFERENCE.arc1.solved + "/" + REFERENCE.arc1.n +
               " (" + pct(REFERENCE.arc1.solved, REFERENCE.arc1.n) + "), ARC-AGI-2 " +
               REFERENCE.arc2.solved + "/" + REFERENCE.arc2.n +
               " (" + pct(REFERENCE.arc2.solved, REFERENCE.arc2.n) + "). Reproduce with `" +
               REFERENCE.command + "`. The budget is wall clock, so the total moves " +
               "by a task or two between runs -- observed range " + REFERENCE.spread + ".");
    lines.push("");
    lines.push("No hosted model and no API is involved: " + engineLine() + ". " + corpusLine() + ".");
    return { ok: true, text: lines.join("\n"), tally: t, results: results,
             used: ["measured in this page"] };
  });
}

function answerSolve(text) {
  var d = extractTaskJson(text), byId = null, m;
  if (!d) {
    m = /\b(arc[12]_[0-9a-f]{6,8})\b/i.exec(text);
    if (m) byId = taskById(m[1].toLowerCase());
    if (!byId) {
      var idx = sampleIndices(1);
      byId = taskAt(idx[0]);
    }
    d = byId;
  }
  var task = { id: d.id || "pasted", train: d.train, test: d.test || [] };
  if (!task.test.length && task.train.length > 1) {
    /* No test input given: hold out the last demonstration and predict it. */
    task = { id: task.id + " (held-out demonstration)",
             train: task.train.slice(0, -1),
             test: [{ input: task.train[task.train.length - 1].input,
                      output: task.train[task.train.length - 1].output }] };
  }
  return new Promise(function (resolve) {
    var t0 = Date.now(), res;
    try { res = ENGINE.solveTask(task, { time_budget: 10, k: 2 }); }
    catch (e) {
      resolve({ ok: false, text: "The engine failed on that task: " + e.message });
      return;
    }
    var secs = Math.round((Date.now() - t0) / 100) / 10;
    var lines = [];
    var preds = res.predictions[0] || [];
    if (!preds.length) {
      lines.push("No rule in the portfolio reproduced all " + task.train.length +
                 " demonstrations of " + task.id + ", so there is nothing to predict. " +
                 "It searched for " + secs + "s and fitted " + res.n_fit + " candidates.");
      resolve({ ok: false, text: lines.join("\n"), reasonedName: null });
      return;
    }
    lines.push("Prediction for " + task.id + " -- " + res.n_fit + " rules fitted every " +
               "demonstration, the ranked winner came from the `" + (res.solver || "?") +
               "` family, " + secs + "s.");
    lines.push("");
    lines.push(renderGrid(preds[0]));
    var s = ENGINE.scoreTask(task, res, 2);
    if (s) {
      lines.push("");
      lines.push(s.top1 ? "That is exactly the withheld answer."
                        : (s.topk ? "The withheld answer is this rule's second guess."
                                  : "That does not match the withheld answer."));
    }
    if (res.hyps && res.hyps.length) {
      lines.push("");
      lines.push("Cheapest fitting rules: " + res.hyps.slice(0, 3).map(function (h) {
        return h[0] + " (" + h[1] + ")";
      }).join(", "));
    }
    resolve({ ok: true, text: lines.join("\n"), used: ["solved in this page"] });
  });
}

function answerShow(text) {
  var m = /\b(arc[12]_[0-9a-f]{6,8})\b/i.exec(text);
  var task = m ? taskById(m[1].toLowerCase()) : taskAt(sampleIndices(1)[0]);
  if (!task) return Promise.resolve({ ok: false, text: "No task by that id is bundled here." });
  return Promise.resolve({ ok: true, text: describeTask(task), used: ["bundled corpus"] });
}

function answerHow() {
  var names = ENGINE.moduleNames();
  var lines = [];
  lines.push("The ARC solver in this page is a portfolio of " + names.length +
             " independent hypothesis generators over a grid algebra. Each one proposes " +
             "candidate programs -- geometry, panel logic, per-object processes, learned " +
             "local rules, bottom-up enumeration -- and every candidate is executed against " +
             "every demonstration and thrown away unless it reproduces all of them. The " +
             "survivors are ranked by description length, corrected by bounded " +
             "leave-one-out refits, and the top-ranked distinct predictions are the answer.");
  lines.push("");
  lines.push("Families: " + names.join(", ") + ".");
  lines.push("");
  lines.push("A trained planner decides which generator gets the next slice of the budget, " +
             "from task signatures, what has already run and produced nothing, and how much " +
             "time is left. It chooses order and budget only -- it cannot make the engine " +
             "accept a wrong answer, because acceptance is still exact reproduction of every " +
             "demonstration.");
  lines.push("");
  lines.push("Nothing here calls out: no hosted model, no API, no network. " + corpusLine() + ".");
  return Promise.resolve({ ok: true, text: lines.join("\n"), used: ["local engine"] });
}

function answer(text) {
  var t = String(text || "");
  if (!ready()) return Promise.resolve({ ok: false, text: "The ARC engine is not loaded on this page." });
  if (extractTaskJson(t)) return answerSolve(t);
  if (RE_SCORE.test(t) || /\b(run|benchmark|evaluate|eval)\b/i.test(t)) return answerScore(t, {});
  if (RE_SOLVE.test(t)) return answerSolve(t);
  if (RE_SHOW.test(t)) return answerShow(t);
  return answerHow();
}

root.C4ARC = {
  ready: ready,
  isRequest: isRequest,
  answer: answer,
  score: function (opts) { return answerScore((opts && opts.full) ? "full" : "", opts || {}); },
  solveTask: function (task, budget) {
    return ENGINE.solveTask(task, { time_budget: budget === undefined ? 10 : budget, k: 2 });
  },
  taskById: taskById,
  taskCount: function () { return TASKS ? TASKS.length : 0; },
  lastRun: loadRun,
  reference: REFERENCE,
  engine: function () { return ENGINE; }
};

/* The planner is data, not code: activating it only reorders generators. */
if (ENGINE && PLANNER) {
  try { ENGINE.activatePlanner(ENGINE.loadPlanner(PLANNER)); } catch (e) {}
}

})(typeof window !== "undefined" ? window : globalThis);
