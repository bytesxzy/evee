/* The adversarial sense benchmark.
 *
 * Two rules keep this honest:
 *
 * 1. **Split.** Every case is `tune` or `hold`. Only `tune` was run while the
 *    implementation was being changed. `hold` was run once, at the end, and no
 *    code was edited after seeing it. `bench.js --split` enforces which set you
 *    are looking at.
 * 2. **Nothing here is in the runtime.** No title, no query and no expected
 *    answer from this file appears in robots.html or in the exported model.
 *    `bench.js --audit` greps for exactly that and fails if it finds one.
 *
 * A case is scored against sets, not a single string, because more than one
 * answer can be right: for a genuinely ambiguous term, disclosing the ambiguity
 * and picking a documented sense are both correct, and picking the wrong sense
 * silently is not.
 */

"use strict";

/* accept: titles that are a correct resolution
   reject: titles that are a wrong-sense resolution
   avoid:  tokens that must not appear in the answer text
   want:   tokens that should appear in the answer text                        */
function c(id, split, query, accept, reject, extra) {
  return Object.assign({ id, split, query, accept, reject: reject || [],
                         mode: "research" }, extra || {});
}

const WORDINGS = ["what is {x}", "define {x}", "tell me about {x}",
                  "what does {x} mean", "explain {x}", "{x}?", "what's {x}",
                  "could you explain {x}"];

/* Generate the same concept across many wordings: the router must not depend
   on one phrasing, and a fix that only works for "what is X" is not a fix. */
function spread(idBase, split, subject, accept, reject, wordings, extra) {
  return (wordings || WORDINGS).map((w, i) =>
    c(idBase + "#" + i, split, w.replace("{x}", subject), accept, reject, extra));
}

const CASES = [].concat(
  /* ---------------------------------------------------------------- tune */
  spread("high-school", "tune", "high school",
    ["Secondary school", "School"], ["High School High", "High School Musical"]),
  spread("high-school-musical", "tune", "High School Musical",
    ["High School Musical"], ["Secondary school", "High School High"]),
  spread("math", "tune", "math",
    ["Mathematics"], ["MathJax", "MathML", "Math rock"]),
  spread("math-rock", "tune", "math rock",
    ["Math rock"], ["Mathematics", "MathJax"]),
  spread("science", "tune", "science", ["Science"], ["MathJax"]),
  spread("ai", "tune", "AI", ["Artificial intelligence"], []),
  spread("photosynthesis", "tune", "photosynthesis", ["Photosynthesis"], []),
  spread("javascript", "tune", "JavaScript",
    ["JavaScript"], ["Java (programming language)", "Java"]),
  spread("school", "tune", "school",
    ["School", "Secondary school"], ["School of Rock"]),
  spread("rust", "tune", "rust",
    ["Rust"], ["Rust (programming language)"]),
  spread("rust-lang", "tune", "Rust programming language",
    ["Rust (programming language)"], ["Rust"]),
  spread("jaguar", "tune", "jaguar", ["Jaguar"], ["Jaguar Cars"]),
  spread("apple", "tune", "apple", ["Apple"], ["Apple Inc."]),
  spread("apple-inc", "tune", "Apple Inc.", ["Apple Inc."], ["Apple"]),
  spread("cell-bio", "tune", "cell",
    ["Cell", "Cell (biology)"], ["Prison cell"]),
  spread("quantum", "tune", "quantum mechanics", ["Quantum mechanics"], []),
  [
    c("einstein-who", "tune", "who is Albert Einstein", ["Albert Einstein"], []),
    c("einstein-tell", "tune", "tell me about Albert Einstein", ["Albert Einstein"], []),
    c("capital-france", "tune", "capital of France", ["Paris", "France"], [],
      { want: ["paris"] }),
    c("capital-france-2", "tune", "what is the capital of France",
      ["Paris", "France"], [], { want: ["paris"] }),
    c("java-lang", "tune", "what is Java the programming language",
      ["Java (programming language)"], ["Java", "Java coffee"]),
  ],

  /* --------------------------------------------------------------- hold
     Run once, after the implementation was frozen. */
  spread("matrix", "hold", "matrix",
    ["Matrix", "Matrix (mathematics)"], ["The Matrix"]),
  spread("the-matrix", "hold", "The Matrix",
    ["The Matrix"], ["Matrix (mathematics)"]),
  spread("mercury", "hold", "mercury",
    ["Mercury", "Mercury (planet)", "Mercury (element)"], []),
  spread("transformer", "hold", "transformer",
    ["Transformer"], ["Transformers (franchise)",
                      "Transformer (deep learning architecture)"]),
  spread("windows", "hold", "windows",
    ["Windows", "Window", "Microsoft Windows"], []),
  spread("window", "hold", "window", ["Window"], ["Microsoft Windows"]),
  spread("amazon", "hold", "amazon",
    ["Amazon", "Amazon River", "Amazon rainforest", "Amazon (company)"], []),
  spread("saturn", "hold", "saturn",
    ["Saturn"], ["Saturn (rocket family)"]),
  spread("python", "hold", "python",
    ["Python", "Python (programming language)", "Pythonidae"], ["Monty Python"]),
  spread("java", "hold", "java",
    ["Java", "Java (programming language)"], ["Java coffee"]),
  [
    c("ms-windows", "hold", "what is Microsoft Windows",
      ["Microsoft Windows"], ["Window"]),
    c("school-of-rock", "hold", "what is School of Rock",
      ["School of Rock"], ["School", "Secondary school"]),
    c("monty", "hold", "who are Monty Python", ["Monty Python"], ["Pythonidae"]),
    c("dl-transformer", "hold", "what is a transformer in deep learning",
      ["Transformer (deep learning architecture)"], ["Transformers (franchise)"]),
  ]
);

/* Conversation routing. A research answer here is the failure. */
const CONVERSATION = [
  ["hey", "tune"], ["yo", "tune"], ["how are you", "tune"],
  ["that's actually pretty cool", "tune"], ["lol", "tune"],
  ["what do you think about that", "tune"], ["can we talk", "tune"],
  ["why do you sound so formal", "tune"], ["I had a weird day", "tune"],
  ["that's not what I meant", "tune"], ["nah bro", "tune"],
  ["wait what", "tune"], ["continue", "tune"], ["tell me more", "tune"],
  ["hi", "hold"], ["thanks", "hold"], ["are you an ai", "hold"],
  ["i'm bored", "hold"], ["good morning", "hold"], ["fair enough", "hold"]
].map(([q, split], i) => ({ id: "convo#" + i, split, query: q,
                            mode: "conversational" }));

/* Multi-turn: continuation must hold the subject, an explicit shift must drop
   it, and neither may be decided by keyword overlap alone. */
const DIALOGUES = [
  {
    id: "carry-einstein", split: "tune",
    turns: [
      { q: "tell me about Albert Einstein", accept: ["Albert Einstein"] },
      { q: "when was he born", accept: ["Albert Einstein"], want: ["1879"] }
    ]
  },
  {
    id: "shift-photosynthesis", split: "tune",
    turns: [
      { q: "tell me about Albert Einstein", accept: ["Albert Einstein"] },
      { q: "anyway what's photosynthesis", accept: ["Photosynthesis"],
        reject: ["Albert Einstein"] }
    ]
  },
  {
    id: "carry-more", split: "tune",
    turns: [
      { q: "what is photosynthesis", accept: ["Photosynthesis"] },
      { q: "tell me more", accept: ["Photosynthesis"], mode: "either" }
    ]
  },
  {
    id: "shift-hard", split: "hold",
    turns: [
      { q: "what is math", accept: ["Mathematics"] },
      { q: "never mind, what is photosynthesis", accept: ["Photosynthesis"],
        reject: ["Mathematics"] }
    ]
  },
  {
    id: "carry-then-shift", split: "hold",
    turns: [
      { q: "what is JavaScript", accept: ["JavaScript"] },
      { q: "tell me more", accept: ["JavaScript"], mode: "either" },
      { q: "actually what is photosynthesis", accept: ["Photosynthesis"],
        reject: ["JavaScript"] }
    ]
  }
];

module.exports = { CASES, CONVERSATION, DIALOGUES, WORDINGS };
