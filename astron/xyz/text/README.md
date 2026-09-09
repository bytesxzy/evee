# ASTRON text — the sibling that resolves senses

GABRIEL is a sparse log-linear model over the ARC **program** language.  This
package is its sibling over natural-language **queries**.  It shares GABRIEL's
architecture and none of its vocabulary, because an ARC token cannot score
English and an English token cannot score an ARC program — and pretending
otherwise would be the fake integration this whole package exists to avoid.

```
ARC side (gabriel/)                    text side (text/)
─────────────────────                  ─────────────────────────────
task signatures                        query signatures
  ↓                                      ↓
P(solver family | task)                P(hypothesis family | query)
  ↓                                      ↓
operator probabilities                 candidate-sense scoring
  ↓                                      ↓
constrained program chains             constrained answer plans
  ↓                                      ↓
run on every training pair             the identity ladder
  ↓                                      ↓
MODEL PROPOSES, VERIFIER DISPOSES      MODEL PROPOSES, VERIFIER DISPOSES
```

`engine/`, `gabriel/`, `bench/` and `policy/gabriel_lm.json` are **untouched**.
`tests/test_text.py` asserts that no module here imports either package.

---

## 1. The one rule everything else serves

> **A token set is allowed for coverage.  A token set is not sufficient for
> identity.**

The bug this replaces was that `high school` and `High School High` had the same
*unique* token set `{high, school}`, because the duplicate disappeared.  That let
an expanded compound title impersonate an exact concept.

`families.identity_tokens` therefore keeps **order**, **multiplicity** and
**stop words**:

| phrase | identity tokens |
|---|---|
| `high school` | `high · school` |
| `High School High` | `high · school · high` |
| `matrix` | `matrix` |
| `the matrix` | `the · matrix` |
| `Matrix (mathematics)` | `matrix` (a parenthetical is a disambiguator, not a name) |

Coverage — "does this page mention the words" — may still use a set, and does.
The two are different questions and are now different functions.

## 2. The identity ladder

```
5  VERIFIED_TITLE     an exact title or a redirect the encyclopedia confirmed
4  VERIFIED_ALIAS     a confirmed canonical alias of what the user typed
3  EXACT_PHRASE       title identity equals query identity, order + multiplicity
2  COHERENT           a different name, but the evidence is about this concept
1  EXPANDED           the query is a prefix-compound of a longer title
0  FUZZY              tokens overlap and nothing else does
```

`families.prune` enforces one invariant: **once any candidate reaches tier 3 or
better, nothing at tier 1 or below may be answered.**

That single rule is what makes the failure case impossible.  `high school`
resolves through Wikipedia's redirect to `Secondary school` at tier 5;
`High School High` is tier 1; tier 1 is dropped.  And it works in the other
direction for free — `high school musical` is identity-equal to
`High School Musical`, so the specific title is tier 3 and wins.

When *no* candidate clears tier 1, the page does not silently answer with an
expansion: it discloses the sense it picked, or asks one clarifying question.

## 3. Where the learned model sits — and where it cannot

```
grammar proposes families        (families.grammatical_families)
        ↓
model weights them               (lm.TextLM family head, constrained softmax)
        ↓
IDENTITY LADDER PRUNES           (families.prune — hard, unlearned)
        ↓
model orders the survivors       (lm.TextLM candidate head)
        ↓
ordering never crosses a tier
```

Feedback cannot teach `math = MathJax`, because MathJax never reaches the
learned layer: `math` is not an ordered token subsequence of `mathjax`, and no
redirect says it is.  `tests/test_text.py::test_the_learned_head_never_sees_a_rejected_candidate`
asserts it, and the browser's `robots.selfTest()` re-asserts it in the page.
Runtime test 24 pushes 400 upvotes onto the wrong sense's feature vector and
checks the answer does not move.

## 4. Query signatures

Categorical, and **topic-blind by construction** — `what is photosynthesis` and
`what is cryptography` produce the *identical* signature.  A signature that
named entities would be a topic list in disguise.

```
form:      whatis whois define tellme howto howwork why compare when where
           list social meta bare bare_q
parse:     bare possessive relation
len:       meaningful subject tokens (0–4)
sub: rel:  subject / relation length buckets
case:      lower title mixed upper      ← named-looking phrase structure
dup:       does the subject repeat a token
punct:     trailing question mark, quotes
convo:     conversational markers, incl. multi-word social phrases
shift:     explicit topic-shift marker
cont:      explicit continuation marker
pron:      unresolved pronoun / bare anaphor
site:      a token from the host site's own lexicon
ctx:       fresh / same / new / carry, relative to dialogue state
depth:     how deep into the conversation
```

## 5. Dialogue state

Topic coherence has to survive more than one message *and die on command*.
Keyword overlap alone does neither.  `DialogueState` (in `robots.html`) tracks
subject, canonical entity, relation, confidence, turn depth, last act and a
short stack of prior entities, and decides by grammar:

```
"tell me about Einstein"   → set     subject = Einstein, entity = Albert Einstein
"when was he born"         → carry   rewritten to "when was Albert Einstein born"
"anyway what's photosynthesis" → reset   old subject dropped outright
"tell me more"             → carry + continuation → the plan skips what was said
```

The anaphor is resolved **before** anything is retrieved.  Retrieval never gets
to decide what "he" meant.

## 6. Reading, not copying

```
article → evidence windows → propositions → answer plan → generated sentence
```

* **Windows.** Sentences are scored against the article's own *topic profile*
  (vocabulary from its opening plus the asked phrase), not just against the
  title.  An article stops naming itself after its first sentence; scoring by
  title alone discarded the whole body, which is why answers were one sentence
  long and "tell me more" had nothing more to say.
* **Propositions.** Typed records — subject, copula, determiner, predicate,
  kind, source index — extracted, never generated.
* **Plan.** A definition establishes *what the thing is* first.  History,
  etymology and same-named films are `detail`/`event` propositions and come
  after, or not at all.
* **Synthesis.** Sentences are generated *from* the proposition, then
  restructured until they are no longer a reproduction: alternative openings,
  relative-clause splitting, participial-modifier splitting, prepositional-phrase
  fronting, coordinated-predicate splitting.  Every transform reorders the
  source's **own words** — none replaces a term — so a name, a number, a date or
  a technical phrase cannot be distorted.  Truth outranks novelty; where no safe
  transform exists the long technical span survives and only the *whole-sentence*
  copy is prevented.

## 7. Conversation

The old path retrieved the nearest line from a canned corpus and returned it
verbatim.  This replaces the mechanism, not just the corpus:

```
utterance → dialogue-act features → P(act | utterance) → authored response plan
          → slot filling from the user's own words and the dialogue state
```

The corpus controls **how to speak**.  It never supplies **what is true**, and
it ships no dataset text at all — `conversation.statistics_from_turns` reduces a
harvest to counts, and every string a user can see is written in
`conversation.py`.  That is the anti-copy mechanism by construction rather than
by filter, which is the only version that cannot leak.

A research form can never route conversationally: `grammatical_families` does
not offer `conversational` for `form:whatis`, and `conversationalAllowedHard`
in the page refuses it again as a hard rule above the model.

## 8. The integration is static, and that is deliberate

`README.OPS.md` is explicit that ASTRON runs no daemon and listens on no port.
`https://cell4.art/astron/` is a served directory.  A page that called it as an
inference API would be calling something that does not exist.  So:

```
text/train.py   fits offline (0.7 s, pure Python, stdlib only)
  → policy/text_gabriel_lm.json     family head + candidate head + contract
  → data/conversation_pack.json     act model + response plans + style stats
    ↓ fetched same-origin, versioned by content hash, cached in localStorage
robots.html runs the same arithmetic locally
```

`text/bench.py --conformance` writes reference vectors; `text/browser/conformance_check.js`
proves the browser reproduces them **exactly** (217 comparisons, 0 differences).
Without that check, a drift between the Python fit and the JavaScript inference
would be silent: the page would still answer, just worse.

## 9. Everything degrades

| failure | behaviour |
|---|---|
| `text_gabriel_lm.json` missing/malformed | deterministic ranking that shipped before; ladder still runs |
| `conversation_pack.json` missing | ChatterBot corpus, then built-in replies |
| Hugging Face unavailable | the exported artifact is already local; nothing is fetched at runtime |
| Wikipedia down | other sources; answer still lands |
| title resolver down | full-text search path still answers |
| a source hangs | aborted at 7 s, answer still lands |
| corrupt localStorage | entry dropped, not thrown |
| synthesis fails | the safest grounded compact form |

No blank replies, no uncaught promises, no hanging loading state, no dangling
`and…`.  `text/browser/runtime_tests.js` drives all of it: 33 checks, all
passing.

## 10. Files

```
text/signatures.py        query signatures (mirrored exactly in robots.html)
text/families.py          hypothesis families + the identity ladder (the verifier)
text/lm.py                the two log-linear heads
text/corpus.py            template-generated training data + split discipline
text/conversation.py      dialogue acts, response plans, harvest→statistics
text/harvest_oasst.py     the ONLY networked module; offline; keyless; licence-gated
text/train.py             one gated fitting pass
text/export.py            the two browser artifacts + the conformance vectors
text/bench.py             ladder gate, benchmark-leak audit, conformance export
text/browser/harness.js   a real DOM/fetch/timer runtime for robots.html
text/browser/fixtures.js  the fixture encyclopedia (prose written here)
text/browser/routes.js    fixtures → MediaWiki-shaped responses
text/browser/cases.js     the adversarial benchmark, tune/hold split
text/browser/bench.js     paired baseline-vs-revision measurement
text/browser/runtime_tests.js   33 runtime checks incl. every failure path
text/browser/conformance_check.js   Python ↔ JavaScript agreement
```

Numbers: [RESULTS.md](RESULTS.md).  Data and licences: [DATA.md](DATA.md).
Deployment: [DEPLOY.md](DEPLOY.md).  Bugs found on the way: [BUGS.md](BUGS.md).
