/* Does the browser compute what the fit was fitted on?
 *
 * The model is trained in Python over signature strings and scored in
 * JavaScript over signature strings. If the two tokenisers, form regexes or
 * pair groups ever drift apart, the browser is scoring features the model has
 * no weights for -- and the failure is silent: the page still answers, just
 * worse, and nothing anywhere reports a problem.
 *
 * So the drift is made loud. `python3 -m text.bench --conformance` writes the
 * reference vectors; this compares them against what robots.html actually
 * computes, and exits non-zero on the first disagreement.
 *
 *   node text/browser/conformance_check.js [robots.html] [conformance.json]
 */

"use strict";
const fs = require("fs");
const path = require("path");
const H = require("./harness");
const R = require("./routes");

const HTML = path.resolve(process.argv[2] ||
  path.join(__dirname, "..", "..", "..", "..", "robots_topic_coherent.html"));
const VEC = path.resolve(process.argv[3] || path.join(__dirname, "conformance.json"));

const SITE_LEXICON = ["CELL4", "robots.js", "proof of work", "the team",
                      "your services", "contact", "inquire", "cell4.art"];

async function main() {
  if (!fs.existsSync(VEC)) {
    console.error("no conformance vectors at " + VEC +
                  "\nrun: python3 -m text.bench --conformance");
    process.exit(2);
  }
  const ref = JSON.parse(fs.readFileSync(VEC, "utf8"));
  const win = H.loadPage(HTML, R.build({}), R.jsonp({}));
  /* The artifact is fetched asynchronously. Comparing before it lands would
     compare the fitted distribution against the untrained uniform prior and
     report a difference that is not one. */
  await new Promise((r) => setTimeout(r, 500));
  if (!win.robots.textModel()) {
    console.error("the text model did not load; conformance cannot be checked");
    process.exit(2);
  }
  const B = win.robots.bridge;

  let checked = 0;
  const diffs = [];
  function eq(name, a, b) {
    checked++;
    const x = JSON.stringify(a), y = JSON.stringify(b);
    if (x !== y) diffs.push(name + "\n    python: " + x + "\n    browser: " + y);
  }

  for (const row of ref.queries) {
    const q = row.q;
    eq("form " + JSON.stringify(q), row.form, B.form(q));
    const sr = B.split(q);
    eq("subject " + JSON.stringify(q), row.subject, sr.subject);
    eq("relation " + JSON.stringify(q), row.relation, sr.relation);
    eq("parse " + JSON.stringify(q), row.parse, sr.parse);
    eq("signatures " + JSON.stringify(q), row.signatures,
       B.signatures(q, null, SITE_LEXICON));
    eq("families " + JSON.stringify(q), row.families,
       B.families(q, SITE_LEXICON, null));
    eq("identity " + JSON.stringify(q), row.identity, B.identityKey(sr.subject));
    if (row.family_dist) {
      const got = B.familyDist(q, null, SITE_LEXICON);
      const rounded = {};
      Object.keys(got).sort().forEach((k) => { rounded[k] = Number(got[k].toFixed(9)); });
      eq("family_dist " + JSON.stringify(q), row.family_dist, rounded);
    }
  }

  for (const t of ref.tiers) {
    const got = B.tier(t.asked, { title: t.title });
    eq("tier " + t.asked + " / " + t.title, [t.tier, t.reason], [got.tier, got.reason]);
    eq("dup " + t.asked + " / " + t.title, t.dup_mismatch,
       B.duplicateMismatch(t.asked, t.title));
    const c = B.orderedContainment(t.asked, t.title);
    eq("containment " + t.asked + " / " + t.title, t.containment, [c.ok, c.extras]);
  }

  console.log("conformance: " + checked + " comparisons, " + diffs.length + " differences");
  if (diffs.length) {
    diffs.forEach((d) => console.log("  DIFF " + d));
    process.exit(1);
  }
  console.log("the browser reproduces the Python side exactly.");
  process.exit(0);
}

main();
