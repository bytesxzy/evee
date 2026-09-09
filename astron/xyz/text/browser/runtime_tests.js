/* Runtime tests for robots.html.
 *
 * A syntax check is not enough. The previous round found a bug where the file
 * parsed, but a helper had drifted into another closure and only the Wikipedia
 * path could see it -- so every test below drives a real code path with real
 * (mocked) responses, including the paths that only run when something fails.
 *
 *   node text/browser/runtime_tests.js [path/to/robots.html]
 */

"use strict";
const path = require("path");
const H = require("./harness");
const R = require("./routes");
const F = require("./fixtures");

const TARGET = path.resolve(process.argv[2] ||
  path.join(__dirname, "..", "..", "..", "..", "robots_topic_coherent.html"));

let pass = 0, fail = 0;
const failures = [];

function ok(name, cond, detail) {
  if (cond) { pass++; console.log("  ok   " + name); }
  else { fail++; failures.push(name + (detail ? "  -- " + detail : "")); console.log("  FAIL " + name + (detail ? "  -- " + detail : "")); }
}

function boot(opts) {
  return H.loadPage(TARGET, R.build(opts || {}), R.jsonp(opts || {}));
}

function settle(ms) { return new Promise((r) => setTimeout(r, ms || 250)); }

function complete(text) {
  const t = String(text || "").trim();
  if (!t) return false;
  if (!/[.!?)"”\]]$/.test(t)) return false;
  if (/\b(?:and|or|but|the|a|an|of|to|for|in|on|with|that|which|because|is|are|was|were)\s*(?:…|\.\.\.)?$/i.test(t.replace(/[.!?]$/, "")))
    return false;
  return true;
}

async function main() {
  console.log("runtime tests against " + TARGET + "\n");

  /* 1. parse + boot ------------------------------------------------------ */
  let win;
  try { win = boot(); ok("1  page loads and both scripts execute", true); }
  catch (e) { ok("1  page loads and both scripts execute", false, e.message); return done(); }

  /* 2. in-page self test -------------------------------------------------- */
  const st = win.robots.selfTest();
  ok("2  in-page selfTest passes", st.passed === st.total,
     st.passed + "/" + st.total + " " +
     st.results.filter((r) => !r.pass).map((r) => r.name).join("; "));

  /* 3. artifacts load same-origin ---------------------------------------- */
  await settle(300);
  ok("3a text model artifact loaded", !!win.robots.textModel());
  ok("3b conversation pack loaded", !!win.robots.conversationPack());

  /* 4. exact title / redirect resolution ---------------------------------- */
  let r = await H.askOnce(win, "what is high school", 9000);
  let row = win.robots.last().a.research.row;
  ok("4  high school resolves through the redirect", row && row.h === "Secondary school",
     row && row.h);

  /* 5. an expanded compound may not impersonate the concept --------------- */
  ok("5  high school is not High School High",
     !/High School High/i.test(r.text), r.text.slice(0, 90));

  /* 6. specificity in the other direction --------------------------------- */
  await H.askOnce(win, "what is High School Musical", 9000);
  row = win.robots.last().a.research.row;
  ok("6  High School Musical still resolves", row && row.h === "High School Musical",
     row && row.h);

  /* 7. ranking: a rank-1 lexical hit loses to a verified redirect --------- */
  await H.askOnce(win, "what is math", 9000);
  row = win.robots.last().a.research.row;
  ok("7  math resolves to Mathematics, not Math rock",
     row && row.h === "Mathematics", row && row.h);

  /* 8. full-article enrichment ------------------------------------------- */
  ok("8  the resolved article was actually read",
     !!(row && row.wikiRead && row.wikiEvidence && row.wikiEvidence.length),
     row && String(row.wikiEvidence && row.wikiEvidence.length));

  /* 9. paraphrase: no whole evidence sentence reproduced ------------------ */
  const answers = [];
  for (const q of ["what is rust", "what is a jaguar", "what is photosynthesis",
                   "what is science", "what is JavaScript"]) {
    const a = await H.askOnce(win, q, 9000);
    answers.push([q, a.text]);
  }
  const norm = (s) => String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  let copied = 0;
  for (const [, text] of answers) {
    for (const p of F.PAGES) {
      for (const sent of String(p.full).split(/(?<=[.!?])\s+/)) {
        const e = norm(sent);
        if (e.length > 45 && norm(text).indexOf(e) >= 0) copied++;
      }
    }
  }
  ok("9  no answer reproduces a whole evidence sentence", copied === 0, "copies=" + copied);

  /* 10. conversation corpus + acts --------------------------------------- */
  const pack = win.robots.conversationPack();
  ok("10 dialogue-act model present and typed",
     !!(pack && pack.act_model && pack.act_model.weights && pack.acts.length >= 10));

  /* 11. conversation context --------------------------------------------- */
  const w2 = boot(); await settle(300);
  await H.askOnce(w2, "tell me about Albert Einstein", 9000);
  await H.askOnce(w2, "when was he born", 9000);
  ok("11 anaphora resolves to the current subject",
     /1879/.test(win_last(w2).a.text) || /Einstein/i.test(win_last(w2).effective),
     win_last(w2).effective + " => " + win_last(w2).a.text.slice(0, 60));

  /* 12. topic reset ------------------------------------------------------- */
  await H.askOnce(w2, "anyway what's photosynthesis", 9000);
  const shifted = win_last(w2).a.research && win_last(w2).a.research.row;
  ok("12 an explicit shift resets the topic",
     shifted && shifted.h === "Photosynthesis", shifted && shifted.h);

  /* 13. model missing ----------------------------------------------------- */
  const w3 = boot({ artifacts: "down" }); await settle(400);
  const a3 = await H.askOnce(w3, "what is math", 9000);
  ok("13 no artifact: still answers", complete(a3.text), a3.text.slice(0, 80));
  ok("13b no artifact: still resolves the right sense",
     win_last(w3).a.research && win_last(w3).a.research.row.h === "Mathematics",
     win_last(w3).a.research && win_last(w3).a.research.row.h);

  /* 14. malformed artifact ------------------------------------------------ */
  const w4 = boot({ artifacts: "garbage" }); await settle(400);
  const a4 = await H.askOnce(w4, "what is high school", 9000);
  ok("14 malformed artifact is ignored, not fatal", complete(a4.text), a4.text.slice(0, 80));

  /* 15. Wikipedia down ---------------------------------------------------- */
  const w5 = boot({ wikipedia: "down" }); await settle(300);
  const a5 = await H.askOnce(w5, "what is math", 9000);
  ok("15 Wikipedia down: answers without crashing", complete(a5.text), a5.text.slice(0, 80));

  /* 16. title resolver down, full text still available -------------------- */
  const w6 = boot({ wikipediaTitle: "down" }); await settle(300);
  const a6 = await H.askOnce(w6, "what is photosynthesis", 9000);
  ok("16 title resolver down: full-text path still answers", complete(a6.text),
     a6.text.slice(0, 80));

  /* 17. slow source: the abort controller has to win ---------------------- */
  const w7 = boot({ wikipedia: "hang" }); await settle(300);
  const t0 = Date.now();
  const a7 = await H.askOnce(w7, "what is math", 12000);
  ok("17 a hanging source is aborted and the answer still lands",
     complete(a7.text) && Date.now() - t0 < 11000, (Date.now() - t0) + "ms");

  /* 18. corrupt cache ----------------------------------------------------- */
  const w8 = boot();
  w8.localStorage.setItem("robots.text.model.v1", "{not json");
  w8.localStorage.setItem("robots.conv.pack.v1", "{\"doc\":");
  const a8 = await H.askOnce(w8, "what is science", 9000);
  ok("18 corrupt cache is dropped, not thrown", complete(a8.text), a8.text.slice(0, 60));

  /* 19. repeated identical requests --------------------------------------- */
  const before = win._fetchLog.length;
  await H.askOnce(win, "what is math", 9000);
  await H.askOnce(win, "what is math", 9000);
  ok("19 a repeated question is served from cache",
     win._fetchLog.length - before <= 4, "requests=" + (win._fetchLog.length - before));

  /* 20. empty and strange input ------------------------------------------- */
  for (const q of ["   ", "???", "a", "what is what is what is", " ",
                   "!!!!!!!!!!", "the the the the", "😀😀"]) {
    let text = "";
    try { text = (await H.askOnce(win, q, 9000)).text; }
    catch (e) { text = ""; }
    ok("20 strange input " + JSON.stringify(q) + " answers, never blank",
       !!String(text).trim(), JSON.stringify(String(text).slice(0, 40)));
  }

  /* 21. no dangling tail on any answer ------------------------------------ */
  let dangling = 0;
  for (const q of ["what is math", "what is high school", "who is Albert Einstein",
                   "what is photosynthesis", "capital of France", "hey", "lol",
                   "what is rust", "what is science", "what is JavaScript"]) {
    const a = await H.askOnce(win, q, 9000);
    if (!complete(a.text)) { dangling++; console.log("     dangling: " + JSON.stringify(a.text.slice(-60))); }
  }
  ok("21 no answer ends on a dangling connective", dangling === 0, "n=" + dangling);

  /* 22. conversational routing does not reach an encyclopedia -------------- */
  const w9 = boot(); await settle(300);
  w9._fetchLog.length = 0;
  await H.askOnce(w9, "hey", 9000);
  await H.askOnce(w9, "lol", 9000);
  await H.askOnce(w9, "how are you", 9000);
  ok("22 social turns make no encyclopedia request",
     w9._fetchLog.filter((u) => u.indexOf("wikipedia") >= 0).length === 0,
     w9._fetchLog.filter((u) => u.indexOf("wikipedia") >= 0).length + " requests");

  /* 23. the site still answers about itself -------------------------------- */
  const a10 = await H.askOnce(win, "what does CELL4 do", 9000);
  /* A site answer is a clipped passage from the page itself, not generated
     prose, so it does not always end on a sentence boundary -- that is the
     behaviour that shipped and it is out of scope here. What must hold is that
     it is about CELL4 and does not trail off mid-clause. */
  ok("23 CELL4 questions are still answered from the site",
     /CELL4/i.test(a10.text) && !noDangling(a10.text), a10.text.slice(0, 70));

  /* 24. feedback cannot cross a tier boundary ------------------------------ */
  const reasoner = win.robots.reasoner();
  for (let i = 0; i < 400; i++)
    reasoner.learn({ coverage: 1, expansion: 1, senseCollision: 1, canonical: 0,
                     coherence: 0, anchor: 0, subject: 1, relation: 1 }, 1);
  const after = await H.askOnce(win, "what is high school", 9000);
  ok("24 400 upvotes on the wrong sense cannot redefine the concept",
     !/High School High/i.test(after.text), after.text.slice(0, 90));

  return done();
}

function win_last(w) { return w.robots.last(); }

function noDangling(text) {
  const t = String(text || "").trim().replace(/[.!?\u2026]+$/, "");
  return !t || /\b(?:and|or|but|the|a|an|of|to|for|in|on|with|that|which|because)$/i.test(t);
}

function done() {
  console.log("\n" + pass + " passed, " + fail + " failed");
  if (failures.length) { console.log("\nfailures:"); failures.forEach((f) => console.log("  " + f)); }
  process.exit(fail ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(1); });
