/* A real runtime for robots.html, not a syntax check.
 *
 * The previous round of work found bugs that parsed cleanly and only failed
 * when the Wikipedia path actually executed -- a helper that had drifted into
 * another closure. A syntax check cannot see that. So this harness gives the
 * page a DOM, a localStorage, a fetch, timers and an AbortController, executes
 * both <script> blocks in one `vm` context exactly as a browser would, and then
 * drives `window.robots.ask` with mocked network responses.
 *
 * Everything the page can reach is here, and nothing here reaches the network.
 */

"use strict";
const fs = require("fs");
const vm = require("vm");
const path = require("path");

function makeElement(tag) {
  /* className and classList are the same state in a browser. Keeping them as
     two independent fields is exactly the kind of shim bug that makes a test
     pass while the page is broken -- `el("div", "msg bot")` sets className, and
     a watcher asking classList.contains("bot") would silently never fire. */
  const classes = new Set();
  const el = {
    tagName: String(tag || "div").toUpperCase(),
    id: "", title: "", href: "", target: "", rel: "",
    _text: "", _html: "", children: [], parentNode: null, style: {},
    scrollTop: 0, scrollHeight: 0, value: "", async: false, src: "",
    onerror: null, referrerPolicy: "",
    dataset: {},
    classList: {
      add(c) { String(c).split(/\s+/).filter(Boolean).forEach((x) => classes.add(x)); },
      remove(c) { classes.delete(c); },
      contains(c) { return classes.has(c); },
      toggle(c, on) {
        if (on === undefined) { classes.has(c) ? classes.delete(c) : classes.add(c); }
        else if (on) { classes.add(c); } else { classes.delete(c); }
      }
    },
    appendChild(c) { c.parentNode = el; el.children.push(c); return c; },
    removeChild(c) {
      const i = el.children.indexOf(c);
      if (i >= 0) el.children.splice(i, 1);
      c.parentNode = null;
      return c;
    },
    addEventListener() {}, removeEventListener() {}, click() {}, focus() {},
    setAttribute(k, v) { el[k] = v; }, getAttribute(k) { return el[k]; },
    querySelectorAll() { return []; },
    querySelector() { return null; },
    get textContent() {
      if (el._text) return el._text;
      return el.children.map((c) => c.textContent).join("");
    },
    set textContent(v) { el._text = String(v); el.children.length = 0; },
    get innerHTML() { return el._html; },
    set innerHTML(v) { el._html = String(v); },
    get className() { return Array.from(classes).join(" "); },
    set className(v) {
      classes.clear();
      String(v || "").split(/\s+/).filter(Boolean).forEach((x) => classes.add(x));
    }
  };
  return el;
}

function makeDocument(jsonp) {
  const registry = new Map();
  const doc = {
    readyState: "complete",
    documentElement: makeElement("html"),
    head: makeElement("head"),
    body: makeElement("body"),
    createElement: (t) => makeElement(t),
    createTextNode: (t) => {
      const n = makeElement("#text");
      n.textContent = String(t);
      return n;
    },
    addEventListener() {},
    querySelector(sel) {
      if (!registry.has(sel)) {
        const el = makeElement("div");
        el.id = sel.startsWith("#") ? sel.slice(1) : "";
        el.className = sel.startsWith(".") ? sel.slice(1) : "";
        registry.set(sel, el);
      }
      return registry.get(sel);
    },
    _registry: registry
  };
  /* JSONP is a real transport in this page (DuckDuckGo refuses CORS), so the
     harness has to model it: appending a <script> with a callback= parameter
     invokes that callback, or fires onerror. Leaving it unmodelled would make
     every DuckDuckGo call sit until FED_TIMEOUT and quietly turn the benchmark
     into a timing test. */
  const realAppend = doc.head.appendChild.bind(doc.head);
  doc.head.appendChild = function (node) {
    realAppend(node);
    if (node.tagName === "SCRIPT" && node.src && jsonp) {
      setTimeout(() => {
        const m = String(node.src).match(/[?&]callback=([^&]+)/);
        const name = m ? decodeURIComponent(m[1]) : null;
        const payload = jsonp(String(node.src));
        if (name && payload !== null && doc._window && typeof doc._window[name] === "function")
          doc._window[name](payload);
        else if (typeof node.onerror === "function") node.onerror();
      }, 1);
    }
    return node;
  };
  doc.documentElement.appendChild = doc.head.appendChild;
  doc.body.classList.add = doc.body.classList.add.bind(doc.body.classList);
  doc.documentElement.setAttribute = function (k, v) { this[k] = v; };
  doc.documentElement.getAttribute = function (k) { return this[k]; };
  return doc;
}

function makeStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => { m.set(k, String(v)); },
    removeItem: (k) => { m.delete(k); },
    clear: () => m.clear(),
    _map: m
  };
}

/* A fetch that answers only from a route table. An unrouted URL is a failure
 * the page must survive, which is exactly what the degradation tests want. */
function makeFetch(routes, log) {
  return function (url, opts) {
    const u = String(url);
    log.push(u);
    if (opts && opts.signal && opts.signal.aborted)
      return Promise.reject(new Error("aborted"));
    for (const r of routes) {
      if (r.match(u)) {
        const res = r.respond(u);
        if (res === null) return Promise.reject(new Error("network"));
        if (res === "hang") return new Promise(() => {});
        return Promise.resolve({
          ok: res.ok !== false,
          status: res.status || 200,
          json: () => Promise.resolve(res.body),
          text: () => Promise.resolve(typeof res.body === "string" ? res.body : JSON.stringify(res.body))
        });
      }
    }
    return Promise.reject(new Error("unrouted: " + u));
  };
}

class FakeAbortController {
  constructor() {
    this.signal = { aborted: false, addEventListener() {} };
  }
  abort() { this.signal.aborted = true; }
}

function loadPage(htmlPath, routes, jsonp) {
  const html = fs.readFileSync(htmlPath, "utf8");
  const scripts = [];
  let i = 0;
  for (;;) {
    const a = html.indexOf("<script>", i);
    if (a < 0) break;
    const b = html.indexOf("</script>", a);
    if (b < 0) throw new Error("unterminated <script>");
    scripts.push(html.slice(a + 8, b));
    i = b + 9;
  }
  const fetchLog = [];
  const doc = makeDocument(jsonp);
  const sandbox = {
    console,
    setTimeout, clearTimeout, setInterval, clearInterval,
    Promise, Math, JSON, Date, Object, Array, String, Number, Boolean,
    RegExp, Error, TypeError, Map, Set, isFinite, isNaN, parseInt, parseFloat,
    encodeURIComponent, decodeURIComponent, Uint8Array, TextDecoder,
    document: doc,
    localStorage: makeStorage(),
    sessionStorage: makeStorage(),
    location: { href: "https://cell4.art/robots.html", protocol: "https:", hostname: "cell4.art" },
    matchMedia: () => ({ matches: false, addListener() {}, addEventListener() {} }),
    AbortController: FakeAbortController,
    URL,
    Blob: function () {},
    atob: (s) => Buffer.from(String(s), "base64").toString("binary"),
    btoa: (s) => Buffer.from(String(s), "binary").toString("base64"),
    fetch: makeFetch(routes, fetchLog),
    _fetchLog: fetchLog
  };
  doc._window = sandbox;
  sandbox.window = sandbox;
  sandbox.self = sandbox;
  sandbox.top = sandbox;
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  scripts.forEach((src, n) => {
    try {
      vm.runInContext(src, sandbox, { filename: path.basename(htmlPath) + "#script" + n });
    } catch (e) {
      throw new Error("script " + n + " threw at load: " + e.stack);
    }
  });
  return sandbox;
}

/* Drive one question and resolve with the rendered bot message. The page uses
 * a randomised setTimeout before it answers, so the harness watches the log
 * rather than guessing a delay. */
function askOnce(win, question, timeoutMs) {
  const log = win.document.querySelector("#log");
  const before = log.children.length;
  return new Promise((resolve, reject) => {
    let done = false;
    const started = Date.now();
    win.robots.ask(question);
    const tick = setInterval(() => {
      const kids = log.children.slice(before);
      const bots = kids.filter((k) => k.classList.contains("bot") && !k.classList.contains("think"));
      if (bots.length) {
        clearInterval(tick);
        if (done) return;
        done = true;
        const m = bots[bots.length - 1];
        /* The rendered node is answer text PLUS the source chips and the vote
           buttons ("via Wikipedia en.wikipedia.org up down"). Measuring
           completeness or copy overlap on the whole subtree scores the chrome,
           not the answer -- every answer looked truncated because it ended in
           an arrow glyph. The answer is the first text node. */
        const first = m.children.length ? m.children[0] : null;
        const answer = first && first.tagName === "#TEXT" ? first.textContent : m.textContent;
        resolve({ text: answer, full: m.textContent, node: m, ms: Date.now() - started });
      } else if (Date.now() - started > (timeoutMs || 8000)) {
        clearInterval(tick);
        if (done) return;
        done = true;
        reject(new Error("no answer for " + JSON.stringify(question)));
      }
    }, 5);
  });
}

module.exports = { loadPage, askOnce, makeElement, makeDocument, makeStorage };
