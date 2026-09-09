# Data, sources and licences

Written to be checkable rather than reassuring.  Where something could not be
verified in the build environment, that is said plainly instead of asserted.

---

## 1. What actually ships to a browser

Two files, both generated, both small, both containing **parameters and
authored text only**:

| artifact | bytes | contents | third-party text? |
|---|---:|---|---|
| `policy/text_gabriel_lm.json` | ~29 KB | two sparse weight tables keyed by categorical feature strings (`s\|form:whatis`, `x\|case:lower\|tier:expanded`, …), the family list, the stop list, the feature contract, provenance | **none** |
| `data/conversation_pack.json` | ~23 KB | dialogue-act model weights, the response plans, the act-marker regexes, style statistics, provenance | **none** |

Neither file contains a page title, an article sentence, a dataset reply, or
any string a user could recognise as coming from somewhere else.  The feature
keys are shapes (`tier:verified_title`, `exp:1`, `cov:3`), which is why the
model *cannot* encode "math means Mathematics" even if someone tried to teach
it that.

`text/bench.py --audit` greps the runtime and both artifacts for every
benchmark title and fails the build on a hit.  It currently reports clean.

---

## 2. OpenAssistant / OASST1

**Investigated as instructed, implemented in full, and not run here.**

* Dataset: `OpenAssistant/oasst1`
* Access: the public Hugging Face Dataset Viewer REST API,
  `https://datasets-server.huggingface.co` — no account, no token.
* Module: `text/harvest_oasst.py`

What the harvester does, in order:

1. **Reads the licence** from the Hub API (`cardData.license`, falling back to
   the `license:` tag) and **refuses to proceed** unless it is in
   `SAFE_LICENSES` — a commercial-safe allowlist (`apache-2.0`, `mit`, the BSD
   family, `cc0-1.0`, `cc-by-4.0`, `cc-by-3.0`, `odc-by`, `unlicense`).
   OASST1 is documented as Apache-2.0.  **That was not verifiable from this
   build environment** (see §5), which is exactly why the check is code rather
   than a sentence in a README: the guard runs on the machine that has network
   access, at the moment of harvest, against what the Hub says then.
2. **Discovers** the config and split from `/splits`.  Nothing is hard-coded —
   a re-publish under different names must not fail silently.
3. **Pages** `/rows` with a bounded row budget and a polite delay.
4. **Filters** on OASST1's own quality metadata:
   `lang == "en"`, `deleted == false`, `review_result != false`,
   `tree_state == "ready_for_export"`, `rank == 0` for assistant messages,
   `toxicity ≤ 0.2`, `spam ≤ 0.2`, `quality ≥ 0.5`, length sane.
5. **Reconstructs conversation structure** through `message_id` / `parent_id` /
   `message_tree_id` — assistant replies are paired with their actual prompter
   parent, not with whatever row happened to precede them in the page.
6. **Reduces to statistics** (`conversation.statistics_from_turns`): act
   transition counts, reply-length and sentence-count buckets, question-back
   rate, hedge rate, first-person rate.  **No message text is written to the
   output file.**  `tests/test_text.py::test_statistics_are_counts_only`
   asserts it.

Run it on a networked machine:

```bash
cd astron/xyz
python3 -m text.harvest_oasst --max-rows 4000 --out text/data/oasst_stats.json
python3 -m text.export --stats text/data/oasst_stats.json
```

`export.py` then folds those statistics into `conversation_pack.json` and
records the provenance block.  If the file is absent — as it is in this
checkout — the pack ships with `style: {}` and
`provenance.style_statistics: null`, which is honest rather than empty:
everything else in the pack still works.

### Why statistics and not text

Three reasons, in order of importance:

1. **It is the anti-copy mechanism.**  A retrieval system that has the replies
   can paste one.  A system that has only counts cannot, by construction.
2. **It removes the licensing question from the product.**  Apache-2.0 permits
   redistribution with attribution; derived aggregate statistics are not a
   redistribution of the corpus at all, so the B2B question never arises.
3. **It is smaller.**  23 KB versus megabytes, and it loads on a phone.

### DailyDialog — deliberately not used

Commonly distributed copies of DailyDialog carry **CC BY-NC-SA** terms.  A
non-commercial licence in a B2B product is a problem you cannot fix downstream,
so it is not wired in and not in `SAFE_LICENSES`.  If someone adds it later,
`--require-license` refuses it without being told to.

### ChatterBot corpus — demoted, not removed

The previous build loaded the BSD-licensed `gunthercox/chatterbot-corpus`
English YAML files.  That path still exists and is still keyless, but it is now
the **fallback**: the dialogue-act model answers first, the ChatterBot corpus
answers if the pack failed to load, and the built-in replies answer if that
failed too.  Its licence (BSD-3-Clause) was already compatible; the reason for
demotion is behavioural, not legal — it returned replies verbatim.

---

## 3. Facts at runtime

| source | used for | key | licence note |
|---|---|---|---|
| Wikipedia (MediaWiki API) | sense resolution, article full text | none | content CC BY-SA; the page **reads** it as evidence and synthesises, and the anti-copy layer exists so it does not redistribute sentences |
| Wikidata | structured relational facts | none | CC0 |
| the 15 other keyless APIs already in `robots.html` | unchanged | none | unchanged from the baseline |
| CELL4's own pages (`window.ROBOTS_BRAIN`) | anything about CELL4 | n/a | the customer's own content |

The separation the brief asks for is enforced structurally:

* **Wikipedia / Wikidata / structured sources** decide *what is true*.
* **The conversation pack** decides *how to speak*.
* **The CELL4 corpus** decides *CELL4 facts*, and now has to clear the same
  identity bar as everyone else — `localSenseAcceptable` stops a site page that
  merely *contains* the asked word from pre-empting research (that is why
  "what is JavaScript" no longer returns the robots.js download page).

OASST1-derived statistics cannot answer "what year was X born": the family
grammar does not offer `conversational` for a `form:whatis` or `form:when`
query, so that branch is unreachable.

---

## 4. The benchmark's article text

`text/browser/fixtures.js` contains a fixture encyclopedia.  **That prose was
written for this file.**  It is not copied from Wikipedia.

Two consequences worth stating:

* The benchmark is reproducible.  A suite whose expected answers move when
  Wikipedia's ranking changes is not a regression suite.
* The copy-overlap metric is honest.  Any long verbatim run in an answer came
  from text in that file, and the anti-copy layer is supposed to break it.

---

## 5. What could not be verified here

The build environment has **no outbound network access**: the egress proxy
refuses `datasets-server.huggingface.co` and `en.wikipedia.org` by
organisation policy.

```
$ curl -s -m 20 -o /dev/null -w "%{http_code}\n" \
    "https://datasets-server.huggingface.co/splits?dataset=OpenAssistant%2Foasst1"
000   (connect_rejected — the egress proxy denied the CONNECT)
```

Therefore:

* the OASST1 harvest **was not run**, and no OASST1-derived statistics are in
  this checkout;
* OASST1's licence was **not** read from the Hub here — the code reads it at
  harvest time and refuses on anything outside the allowlist;
* the browser benchmark runs against fixtures, not against live Wikipedia.

None of that is worked around, and none of it is presented as if it had
succeeded.  Everything that *was* run is in [RESULTS.md](RESULTS.md) with the
exact command next to it.

---

## 6. Transformations applied

| input | transformation | output |
|---|---|---|
| `text/corpus.py` templates × placeholder subjects | render → signature → family label; split by template hash into train / dev / holdout | 1 761 examples (1 342 / 359 / 60) |
| `text/corpus.py` candidate scenarios | identity ladder → candidate signature → binary label; scenarios that the ladder rejects never become examples | 91 examples (69 / 22), 20 held out |
| `text/conversation.py` SEED labels | act features → log-linear fit, early-stopped on a 16-utterance holdout | act model, 41 feature rows |
| OASST1 rows (when harvested) | quality filter → tree reconstruction → **counts only** | transition / length / rate distributions |
