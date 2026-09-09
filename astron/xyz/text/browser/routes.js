/* Fixture encyclopedia -> MediaWiki-shaped HTTP responses.
 *
 * The shapes matter as much as the content. `wikipediaTitle` reads
 * `query.redirects[0].from` to recover the spelling the user typed, and
 * `generator=search` stores relevance in `page.index` rather than in key order.
 * A fixture that got either wrong would test a different program.
 */

"use strict";
const F = require("./fixtures");
const fs = require("fs");
const path = require("path");

function pageObj(p, extra) {
  return Object.assign({ pageid: p.id, ns: 0, title: p.title }, extra || {});
}

/* action=query&titles=X&redirects=1 -- "does the encyclopedia know this exact
   name?". Modelled with the real two-step: normalisation capitalises the query,
   then the redirect maps that to the target. */
function titlesResponse(asked) {
  const hit = F.resolveTitle(asked);
  const normalized = asked.charAt(0).toUpperCase() + asked.slice(1);
  if (!hit) {
    return { query: { pages: { "-1": { ns: 0, title: normalized, missing: "" } } } };
  }
  const q = { pages: {} };
  if (normalized !== asked) q.normalized = [{ from: asked, to: normalized }];
  if (hit.redirected || hit.page.title !== normalized)
    q.redirects = [{ from: normalized, to: hit.page.title }];
  const props = {};
  if (hit.page.disambiguation) props.disambiguation = "";
  q.pages[String(hit.page.id)] = pageObj(hit.page, {
    extract: hit.page.extract,
    pageprops: props
  });
  return { query: q };
}

function searchResponse(query, limit) {
  const rows = F.search(query, limit || 3), pages = {};
  rows.forEach((r) => {
    pages[String(r.page.id)] = pageObj(r.page, {
      index: r.index, extract: r.page.extract
    });
  });
  return { query: { pages: pages } };
}

function extractResponse(pageid) {
  const p = F.BY_ID.get(Number(pageid));
  if (!p) return { query: { pages: { "-1": { ns: 0, missing: "" } } } };
  return { query: { pages: { [String(p.id)]: pageObj(p, { extract: p.full }) } } };
}

function param(url, name) {
  const m = String(url).match(new RegExp("[?&]" + name + "=([^&]*)"));
  return m ? decodeURIComponent(m[1].replace(/\+/g, " ")) : "";
}

/* `opts` lets an individual test break one source without touching the rest,
   which is how the degradation cases are expressed. */
function build(opts) {
  opts = opts || {};
  const routes = [];

  /* The same-origin ASTRON artifacts, served off disk exactly as the web server
     would. `opts.artifacts === "down"` is the degradation case: the page must
     stay fully usable with neither file. */
  routes.push({
    match: (u) => /\/astron\/xyz\/(policy|data)\/[A-Za-z0-9_.-]+\.json$/.test(u),
    respond: (u) => {
      if (opts.artifacts === "down") return null;
      if (opts.artifacts === "garbage") return { body: { kind: "not-what-you-expected" } };
      const rel = u.replace(/^https?:\/\/[^/]+/, "").replace(/^\/astron\/xyz\//, "");
      const file = path.join(__dirname, "..", "..", rel);
      if (!fs.existsSync(file)) return { ok: false, status: 404, body: {} };
      try { return { body: JSON.parse(fs.readFileSync(file, "utf8")) }; }
      catch (e) { return { ok: false, status: 500, body: {} }; }
    }
  });

  /* The live-directory reader. A served directory has no JSON index, so this is
     a rejection in production too; the page is expected to shrug. */
  routes.push({
    match: (u) => /\/astron\/xyz\/?$/.test(u) || /\/astron\/xyz\/[^.]*$/.test(u),
    respond: () => null
  });

  routes.push({
    match: (u) => u.indexOf("en.wikipedia.org/w/api.php") >= 0,
    respond: (u) => {
      if (opts.wikipedia === "down") return null;
      if (opts.wikipedia === "hang") return "hang";
      if (opts.wikipedia === "garbage") return { body: { not: "a wiki response" } };
      if (u.indexOf("pageids=") >= 0) {
        if (opts.wikipediaFullText === "down") return null;
        return { body: extractResponse(param(u, "pageids")) };
      }
      if (u.indexOf("titles=") >= 0) {
        if (opts.wikipediaTitle === "down") return null;
        return { body: titlesResponse(param(u, "titles")) };
      }
      if (u.indexOf("gsrsearch=") >= 0)
        return { body: searchResponse(param(u, "gsrsearch"), 3) };
      return { body: { query: { pages: {} } } };
    }
  });

  routes.push({
    match: (u) => u.indexOf("wikidata.org") >= 0,
    respond: () => (opts.wikidata === "down" ? null : { body: { search: [], entities: {} } })
  });

  /* Every other keyless source answers, and answers with nothing. They stay
     "healthy" so the page does not disable them mid-run and quietly change the
     experiment underneath the later questions. */
  routes.push({
    match: () => true,
    respond: () => ({ body: {} })
  });
  return routes;
}

/* DuckDuckGo arrives over JSONP. Its instant answer is often the *wrong* sense
   too, so the fixture makes it agree with the naive lexical top hit: two
   sources agreeing on the wrong sense is the case the ladder has to survive. */
function jsonp(opts) {
  opts = opts || {};
  return function (src) {
    if (String(src).indexOf("duckduckgo") < 0) return {};
    if (opts.duckduckgo === "down") return null;
    const q = param(src, "q");
    const rows = F.search(q, 1);
    if (!rows.length) return {};
    const p = rows[0].page;
    return {
      Heading: p.title,
      AbstractText: p.extract,
      AbstractSource: "Wikipedia",
      AbstractURL: "https://en.wikipedia.org/wiki/" + encodeURIComponent(p.title),
      RelatedTopics: []
    };
  };
}

module.exports = { build, jsonp, titlesResponse, searchResponse, extractResponse };
