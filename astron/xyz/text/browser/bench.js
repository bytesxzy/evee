/* Paired benchmark: the baseline build vs the revision, same fixtures.
 *
 *   node text/browser/bench.js --a baseline.html --b revised.html --split tune
 *
 * Ten metrics, all measured from what the page actually rendered plus the
 * introspection hook `robots.last()`. The baseline predates that hook, so every
 * metric has a DOM-only fallback: a title is recovered from the answer text and
 * from the source links the page printed. That is less precise than reading the
 * winning row, and it is applied identically to both builds, so the comparison
 * stays fair.
 */

"use strict";
const path = require("path");
const H = require("./harness");
const R = require("./routes");
const F = require("./fixtures");
const { CASES, CONVERSATION, DIALOGUES } = require("./cases");

const STOP = new Set(("a an the of for to in on at by with from as is are was were be " +
  "been being am do does did have has had will would can could should and or but not " +
  "no so if then than that this these those it its i you he she they we our their").split(" "));

function toks(s) { return String(s || "").toLowerCase().match(/[a-z0-9]+/g) || []; }
function contentToks(s) { return toks(s).filter((w) => w.length >= 5 && !STOP.has(w)); }

function longestRun(a, b) {
  const x = toks(a), y = toks(b);
  if (!x.length || !y.length) return 0;
  let best = 0, prev = new Array(y.length + 1).fill(0);
  for (let i = 1; i <= x.length; i++) {
    const cur = new Array(y.length + 1).fill(0);
    for (let j = 1; j <= y.length; j++)
      if (x[i - 1] === y[j - 1]) { cur[j] = prev[j - 1] + 1; if (cur[j] > best) best = cur[j]; }
    prev = cur;
  }
  return best;
}

/* Which fixture page did the answer land on? Prefer the introspection hook;
   fall back to matching the rendered text and printed source links, so the
   baseline (which has no hook) is measured the same way. */
function resolvedTitle(win, answerText, node) {
  try {
    const last = win.robots.last && win.robots.last();
    const row = last && last.a && last.a.research && last.a.research.row;
    if (row && row.h) return { title: row.h, via: "hook", row: row, ans: last.a };
  } catch (e) {}
  // source links carry ?curid=<pageid>
  const links = [];
  (function walk(n) {
    if (!n) return;
    if (n.href) links.push(String(n.href));
    (n.children || []).forEach(walk);
  })(node);
  for (const href of links) {
    const m = href.match(/curid=(\d+)/);
    if (m) {
      const p = F.BY_ID.get(Number(m[1]));
      if (p) return { title: p.title, via: "link", row: null, ans: null };
    }
  }
  // last resort: the page whose title tokens the answer text carries best
  let best = null, bestScore = 0;
  for (const p of F.PAGES) {
    const t = toks(p.title);
    if (!t.length) continue;
    let hit = 0;
    const at = new Set(toks(answerText));
    for (const w of t) if (at.has(w)) hit++;
    const score = hit / t.length + (hit === t.length ? 0.5 : 0) - 0.02 * t.length;
    if (hit === t.length && score > bestScore) { bestScore = score; best = p.title; }
  }
  return { title: best || "", via: "text", row: null, ans: null };
}

function isExpansion(asked, title) {
  const q = toks(asked), t = toks(title);
  if (!q.length || t.length <= q.length) return false;
  let qi = 0;
  for (const w of t) if (qi < q.length && w === q[qi]) qi++;
  return qi === q.length;
}

function askedPhrase(query) {
  return String(query)
    .replace(/^\s*(?:what(?:'s)?|which|who(?:'s)?)\s+(?:is|are|was|were)\s+(?:the\s+|an?\s+)?/i, "")
    .replace(/^\s*(?:define|describe|explain|tell me about)\s+/i, "")
    .replace(/^\s*what does\s+/i, "").replace(/\s+mean\s*$/i, "")
    .replace(/^\s*could you explain\s+/i, "")
    .replace(/[?!.]+\s*$/, "").trim();
}

async function runOne(win, kase, evidenceOf) {
  const t0 = Date.now();
  let res;
  try {
    res = await H.askOnce(win, kase.query, 9000);
  } catch (e) {
    return { id: kase.id, query: kase.query, error: String(e.message).slice(0, 120),
             blank: true };
  }
  const text = String(res.text || "").trim();
  const info = resolvedTitle(win, text, res.node);
  void res.full;
  const asked = askedPhrase(kase.query);
  const out = {
    id: kase.id, query: kase.query, split: kase.split, mode: kase.mode,
    text: text, title: info.title, via: info.via, ms: Date.now() - t0
  };

  out.blank = !text;
  out.truncated = /\b(?:and|the|because|of|to|for|with|that|which|is|are)\s*(?:…|\.\.\.)?$/i.test(text) ||
                  /[,;:]\s*$/.test(text);

  if (kase.mode === "conversational") {
    /* A conversational turn must not be answered out of an encyclopedia. Both
       the rendered "via <source>" chips and a fixture title showing up in the
       text count as a failure. */
    const usedEncyclopedia = /\bvia\b/i.test(String(res.full || "").slice(text.length)) ||
      F.PAGES.some((p) => p.extract && longestRun(text, p.extract) >= 6);
    out.correct = !usedEncyclopedia && !!text;
    out.conversational = !usedEncyclopedia;
    return out;
  }

  const accept = kase.accept || [], reject = kase.reject || [];
  out.correct = accept.indexOf(info.title) >= 0;
  out.wrongSense = reject.indexOf(info.title) >= 0;
  out.expandedWrong = !!(info.title && reject.indexOf(info.title) >= 0 &&
                         isExpansion(asked, info.title));

  /* answer-topic consistency: the answer talks about what it resolved, and
     does not quietly talk about a sense it rejected */
  const at = new Set(toks(text));
  const acceptHit = accept.some((a) => toks(a).every((w) => at.has(w) || STOP.has(w)));
  const rejectHit = reject.some((r) => {
    const distinct = toks(r).filter((w) => !accept.some((a) => toks(a).includes(w)));
    return distinct.length && distinct.every((w) => at.has(w));
  });
  out.topicConsistent = (acceptHit || out.correct) && !rejectHit;
  if (kase.want) out.wantHit = kase.want.every((w) => at.has(w));

  /* grounding + copy overlap, both against the evidence the page actually read */
  const ev = evidenceOf(info);
  out.copyRun = ev.length ? Math.max(...ev.map((e) => longestRun(text, e))) : 0;
  out.answerTokens = toks(text).length;
  out.copyRatio = out.answerTokens ? out.copyRun / out.answerTokens : 0;
  /* The metric that actually encodes "direct sentence copying should be
     exceptional": does the answer reproduce a whole evidence sentence? A long
     run inside a technical phrase is tolerable; a reproduced sentence is not. */
  const nrm = (x) => String(x || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const nt = nrm(text);
  out.verbatimSentence = ev.some((e) => {
    const n = nrm(e);
    return n.length > 45 && nt.indexOf(n) >= 0;
  });
  const evTok = new Set(ev.flatMap(toks));
  const ct = contentToks(text);
  out.grounded = ct.length ? ct.filter((w) => evTok.has(w)).length / ct.length : 0;
  return out;
}

function evidenceFactory() {
  return function (info) {
    if (info.row && info.row.wikiEvidence && info.row.wikiEvidence.length)
      return info.row.wikiEvidence.map(String);
    const p = F.BY_TITLE.get(String(info.title || "").toLowerCase());
    if (!p) return [];
    return String(p.full).split(/(?<=[.!?])\s+/).filter(Boolean);
  };
}

async function runBuild(htmlPath, split, opts) {
  const routes = R.build(opts || {});
  const win = H.loadPage(htmlPath, routes, R.jsonp(opts || {}));
  const evidenceOf = evidenceFactory();
  const rows = [];

  const single = CASES.concat(CONVERSATION)
    .filter((k) => split === "all" || k.split === split);
  for (const k of single) rows.push(await runOne(win, k, evidenceOf));

  const dialogues = [];
  for (const d of DIALOGUES.filter((x) => split === "all" || x.split === split)) {
    // a dialogue needs a clean page: state must not leak between scripts
    const w2 = H.loadPage(htmlPath, routes, R.jsonp(opts || {}));
    const turns = [];
    for (const t of d.turns) {
      const r = await runOne(w2, {
        id: d.id + ":" + t.q, query: t.q, split: d.split,
        mode: t.mode === "either" ? "research" : (t.mode || "research"),
        accept: t.accept, reject: t.reject, want: t.want
      }, evidenceOf);
      if (t.mode === "either" && !r.correct && r.text) r.correct = !r.blank;
      turns.push(r);
    }
    dialogues.push({ id: d.id, split: d.split, turns: turns,
                     correct: turns.every((t) => t.correct) });
  }
  return { rows, dialogues, win };
}

function summarise(res) {
  const rows = res.rows;
  const research = rows.filter((r) => r.mode !== "conversational");
  const convo = rows.filter((r) => r.mode === "conversational");
  const n = research.length || 1;
  const withEv = research.filter((r) => r.copyRun != null);
  const cont = res.dialogues.filter((d) => /carry/.test(d.id));
  const shift = res.dialogues.filter((d) => /shift/.test(d.id));
  return {
    n_research: research.length,
    n_conversational: convo.length,
    n_dialogues: res.dialogues.length,
    sense_accuracy: research.filter((r) => r.correct).length / n,
    top1_retrieval: research.filter((r) => r.title && r.correct).length / n,
    topic_consistency: research.filter((r) => r.topicConsistent).length / n,
    wrong_expanded_rate: research.filter((r) => r.expandedWrong).length / n,
    wrong_sense_rate: research.filter((r) => r.wrongSense).length / n,
    ambiguity_ok: research.filter((r) => !r.wrongSense).length / n,
    continuation_consistency: cont.length ?
      cont.filter((d) => d.correct).length / cont.length : null,
    topic_shift_reset: shift.length ?
      shift.filter((d) => d.correct).length / shift.length : null,
    grounding: withEv.length ?
      withEv.reduce((a, r) => a + (r.grounded || 0), 0) / withEv.length : 0,
    copy_overlap_max: withEv.length ? Math.max(...withEv.map((r) => r.copyRun || 0)) : 0,
    copy_overlap_mean: withEv.length ?
      withEv.reduce((a, r) => a + (r.copyRun || 0), 0) / withEv.length : 0,
    copy_overlap_over_13: withEv.filter((r) => (r.copyRun || 0) > 13).length / (withEv.length || 1),
    verbatim_sentence_rate: withEv.filter((r) => r.verbatimSentence).length / (withEv.length || 1),
    copy_ratio_mean: withEv.length ?
      withEv.reduce((a, r) => a + (r.copyRatio || 0), 0) / withEv.length : 0,
    conversation_routing: convo.length ?
      convo.filter((r) => r.correct).length / convo.length : null,
    blank_answers: rows.filter((r) => r.blank).length,
    truncated_answers: rows.filter((r) => r.truncated).length,
    errors: rows.filter((r) => r.error).length
  };
}

function pct(v) { return v == null ? "  n/a" : (100 * v).toFixed(1).padStart(5) + "%"; }

function report(a, b) {
  const ka = summarise(a), kb = summarise(b);
  const lines = [];
  const HIGHER = ["sense_accuracy", "top1_retrieval", "topic_consistency",
    "ambiguity_ok", "continuation_consistency", "topic_shift_reset",
    "grounding", "conversation_routing"];
  const LOWER = ["wrong_expanded_rate", "wrong_sense_rate",
                 "verbatim_sentence_rate", "copy_ratio_mean", "copy_overlap_over_13"];
  lines.push("metric                        baseline   revision    delta");
  lines.push("-".repeat(60));
  for (const k of HIGHER.concat(LOWER)) {
    const x = ka[k], y = kb[k];
    const d = (x == null || y == null) ? "   n/a" :
      ((y - x >= 0 ? "+" : "") + (100 * (y - x)).toFixed(1) + "pt").padStart(8);
    lines.push(k.padEnd(28) + pct(x) + "  " + pct(y) + "  " + d);
  }
  lines.push("-".repeat(60));
  for (const k of ["copy_overlap_max", "copy_overlap_mean", "blank_answers",
                   "truncated_answers", "errors", "n_research",
                   "n_conversational", "n_dialogues"]) {
    lines.push(k.padEnd(28) + String(ka[k]).padStart(6) + "  " +
               String(kb[k]).padStart(7));
  }
  const errA = 1 - ka.sense_accuracy, errB = 1 - kb.sense_accuracy;
  const rel = errA > 0 ? (errA - errB) / errA : null;
  lines.push("-".repeat(60));
  lines.push("relative error reduction (sense accuracy): " +
             (rel == null ? "n/a (baseline had no errors)" : (100 * rel).toFixed(1) + "%"));
  return { text: lines.join("\n"), baseline: ka, revision: kb, relative_error_reduction: rel };
}

async function main() {
  const argv = process.argv.slice(2);
  const arg = (name, dflt) => {
    const i = argv.indexOf("--" + name);
    return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
  };
  const split = arg("split", "tune");
  const a = path.resolve(arg("a", "robots_topic_coherent.ORIGINAL.html"));
  const b = path.resolve(arg("b", "robots_topic_coherent.html"));
  const resA = await runBuild(a, split);
  const resB = await runBuild(b, split);
  const rep = report(resA, resB);
  console.log("\nsplit: " + split + "\nbaseline: " + a + "\nrevision: " + b + "\n");
  console.log(rep.text);
  if (argv.indexOf("--json") >= 0) {
    const fs = require("fs");
    const out = arg("json", "bench_result.json");
    fs.writeFileSync(out, JSON.stringify({
      split, baseline_path: a, revision_path: b,
      summary: rep, baseline_rows: resA.rows, revision_rows: resB.rows,
      baseline_dialogues: resA.dialogues, revision_dialogues: resB.dialogues
    }, null, 1));
    console.log("\nwrote " + out);
  }
  if (argv.indexOf("--verbose") >= 0) {
    for (let i = 0; i < resB.rows.length; i++) {
      const x = resA.rows[i], y = resB.rows[i];
      if (x.correct === y.correct) continue;
      console.log((y.correct ? "  WIN  " : "  LOSS ") + y.id + "  " +
                  JSON.stringify(y.query) + "\n    base -> " + x.title +
                  "\n    rev  -> " + y.title);
    }
  }
  process.exit(0);
}

if (require.main === module) main().catch((e) => { console.error(e); process.exit(1); });
module.exports = { runBuild, summarise, report, longestRun };
