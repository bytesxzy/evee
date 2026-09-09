# Bugs found while implementing

Every one of these was found by a test or a measurement, not by reading.  Each
entry says what broke, how it surfaced, and where the fix is.

---

## In the shipped assistant (`robots_topic_coherent.html`)

### 1. The per-origin evidence cap could discard the verified sense
`rankEvidence` kept at most two rows per origin.  Wikipedia's title resolver,
Wikipedia's full-text search and DuckDuckGo's instant answer (which reports its
source as Wikipedia) all share the origin `wikipedia`, so three rows competed
for two slots — and the *verified redirect* row, which is the only one that
establishes which concept was asked for, lost to two lexical hits.

`what is math` returned **Math rock**.

Fix: a `wikiExact` row bypasses the cap and does not consume a slot.  A verified
title is not "more of the same source"; it is the row that settles the question.

### 2. The wrong article was read
`enrichWikipedia` fetched the full text of the *first* Wikipedia row in score
order.  It then scored that row higher because it had been read.  So a lexical
hit got the article, got the boost, and won — retrieval deciding the question
through the back door.

Fix: read the article of the row the identity ladder prefers (`wikiExact` first).

### 3. The strict relation gate ran before the article was read
`Federation.ask.finish` ran `rankEvidence` → `reasoner.rank` → enrich → rank
again.  `reasoner.rank` requires a direct question's relation to appear in the
evidence — but before enrichment the only evidence is the search snippet, and
"born in Ulm in 1879" is in the body, not the intro.  The row was discarded
before the fetch that would have justified it.

`when was Albert Einstein born` returned *"I could not find that on this site."*

Fix: lenient prefilter → read the article → **then** the strict gate.

### 4. `analyzeQuestion` read `what's X` as a possessive
`^(.+?)['’]s\s+(.+)$` matched `what's photosynthesis` as the relation
*photosynthesis* of the subject *what*.  Every contracted question was parsed
into garbage and then retrieved on.

Fix: the interrogative contraction is a copula.  Added the `what's|who's` lead
patterns and refused a bare interrogative on the left of a possessive.  Same fix
applied identically in `text/signatures.py` (`let's talk` was hitting it too).

### 5. `analyzeQuestion` never stripped a preamble
`anyway what's photosynthesis` parsed as subject *anyway what*.  An explicit
topic shift produced a garbage parse rather than a clean new subject.

Fix: the analyzer strips the same shift/greeting scaffolding the signature layer
strips.

### 6. `study` and `use` were rewritten as verbs in noun position
`WIKI_PHRASE_REWRITES` mapped `study → examine` unconditionally, so
*"the study of quantity"* was printed to a user as **"the examine of quantity."**

Fix: determiner-guarded rewrites, written without lookbehind (not every target
browser supports it).  Also: `created → developed` followed by
`developed → created` mapped *both* words onto "created"; replaced with one
simultaneous swap.

### 7. `process → procedure` changed a technical term
Photosynthesis is a *process*, not a *procedure*.  A paraphrase rule that shifts
a term's meaning is worse than no paraphrase.  Rule removed rather than tuned.

### 8. The site could answer any question containing a site word
`what is JavaScript` returned the **robots.js download page**, because that page
mentions JavaScript and the local-first rule fired on token coverage alone.

Fix: `localSenseAcceptable` — a site page must clear the same identity bar as
anyone else, unless the question names the site's own vocabulary.

### 9. Restructuring was applied on top of discourse frames
The overlap reducer ran after `"In short: "` had been prepended, producing
*"In the years between primary education and higher education or employment, in
short: High school — an institution that educates adolescents."*

Fix: restructure the **clause**, then attach the frame.  Never the other way
round.

### 10. Sentence splitting could change a fact
Splitting on `and` turned *"plants, algae and some bacteria convert light
energy"* into *"plants, algae. Some bacteria convert light energy"* — a
coordinated **noun phrase** split as if it were two clauses.  Not merely
awkward: false.

Fix: coordination splitting requires the right-hand side to begin with a finite
verb, which is what distinguishes a coordinated predicate from a noun list.

### 11. Prepositional fronting stranded participles and relative clauses
*"…change, carried out by stating assumptions"* fronted at `by` left
**"carried out"** at the end of the new sentence.  *"…, which distinguishes it
from the sciences"* left a relative clause attached to nothing.  And
*"born in Ulm in 1879 and died in Princeton in 1955"* fronted into *"In 1879 and
died in Princeton in 1955, he was born in Ulm."*

Fix: three guards — trailing participle (with particle: `carried out`), open
relative clause, and a tail carrying its own coordinated finite verb.

### 12. Support sentences lost their subject
The plan handed support forms a bare predicate, producing *"In practice, used
for this stage in the United States"* — a sentence with no subject.

Fix: support forms receive a complete clause.

### 13. The determiner was dropped from every definition
`PROP_DEF` discarded the `a|an|the` it matched, so the proposition rendered as
*"Math rock is style of rock music."*

Fix: capture the determiner and re-attach it.

### 14. An article that names itself differently fell through
`high school` reaches an article whose first sentence begins *"A secondary
school is…"*.  Matching the proposition's subject only against the display title
failed, so the answer read *"High school — A secondary school is…"*.

Fix: every known name of the page (display title, article title, redirect
target, alias) counts as its subject.

### 15. The evidence reader discarded the whole article body
Sentences were scored by how much of the **title** they repeated.  An article
stops naming itself after its first sentence, so exactly one sentence survived
the digest — which is why answers were one sentence long and `tell me more` had
nothing more to say.

Fix: a topic **profile** built from the article's own opening vocabulary plus
the asked phrase, and a coreference-opener bonus.

### 16. `cleanPhrase` stripped the article from a name
The disambiguation retry asked for `the matrix` and `cleanPhrase` turned it back
into `matrix`, so the retry re-asked the question it was retrying.  The same
function also rendered the offered sense as *"One strong match is Matrix"* when
the sense was **The Matrix**.

Fix: a `keepArticle` path for the retry, and `txtOf` rather than `cleanPhrase`
for displaying a title.

### 17. The two subject hypotheses were identical
`subjectAlt` (the article-keeping reading) was derived from a string the lead
regex had *already* stripped the article from, so the alternative hypothesis was
never different and never ran.

Fix: a second lead regex that does not consume the determiner.

### 18. A predicate pattern captured its own copula
`The capital of France is is Paris.`

Fix: `trimAnswerValue` drops a leading copula.

### 19. The browser grammar was missing a rule the Python side had
`grammatical_families` offered `conversational` for an anaphor or a
continuation; the JavaScript mirror did not.  `tell me more` and `when was he
born` could not route conversationally in the page even though the model had
been fitted assuming they could.

**Found by `text/browser/conformance_check.js`** — this is precisely the silent
drift that check exists to catch.

### 20. The family grammar admitted small talk for a factual lookup
`capital of France` is six tokens with no question word, so the
short-utterance conversational fallback fired and put a factual lookup on the
same ballot as small talk.

Found by `tests/test_text.py::test_a_research_form_can_never_route_conversationally`.

---

## In the supporting infrastructure

### 21. `unittest discover` cannot load the test directory
`astron/xyz/tests/` has no `__init__.py`, so
`python3 -m unittest discover -s tests -t .` fails with
*"Start directory is not importable"*.  `python3 -m unittest tests.test_x`
works because of implicit namespace packages.  Pre-existing; **not fixed**,
because adding the file would change how the existing suite is collected and
that is not this change's business.  The working invocation is in
[RESULTS.md](RESULTS.md).

### 22. The test harness let `className` and `classList` drift apart
`el("div", "msg bot")` sets `className`; the harness's watcher asked
`classList.contains("bot")`.  Both were plausible and independent, so every
answer was invisible to the harness and every question "timed out".

A shim bug, but the instructive kind: it is exactly the failure mode the brief
warned about — code that parses, and only breaks when a real path runs.

### 23. The harness measured the chrome, not the answer
`textContent` of a rendered message includes the source chips and vote buttons,
so every answer appeared to end in *"…viaWikipediaen.wikipedia.org↑↓"* and the
"no dangling tail" check failed on all of them.

Fix: the answer is the first text node.

### 24. The benchmark-leak audit fired on prose and on a different word
It flagged benchmark titles appearing in **comments**, in the in-page
`selfTest()` assertions (which the brief asked for by name), and in the site's
own knowledge base.  It also flagged the pre-existing OpenAlex health-check
probe `"transformers"` as the benchmark term `"transformer"`.

Fix: strip comments, docstrings, the test block and the site knowledge base
before scanning; match whole words.

### 25. The candidate corpus overlapped the benchmark holdout
`CANDIDATE_HOLDOUT` used `jaguar` and `the matrix`, and `CANDIDATE_SCENARIOS`
used `mercury` — all three appear in the browser benchmark, two of them in its
**holdout** split.  That would have made the holdout a report on the training
set.

Fix: replaced with subjects the benchmark never mentions, plus
`text/bench.py --audit`, which asserts the disjointness on every run.
