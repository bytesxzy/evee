# Deployment

Three files change on the server.  Nothing else moves, no service is added, no
port is opened, and the page keeps working at every intermediate state.

## What goes where

| upload this | to this path | why |
|---|---|---|
| `robots_topic_coherent.html` | wherever the assistant is served today (the page itself, or the file the site iframes) | the revised assistant |
| `astron/xyz/policy/text_gabriel_lm.json` | `/astron/xyz/policy/text_gabriel_lm.json` | the fitted family + candidate heads |
| `astron/xyz/data/conversation_pack.json` | `/astron/xyz/data/conversation_pack.json` | the dialogue-act model + response plans |

Both artifact paths are **hard-coded, same-origin and absolute** in the page:

```js
var TEXT_MODEL_URL = "/astron/xyz/policy/text_gabriel_lm.json";
var CONV_PACK_URL  = "/astron/xyz/data/conversation_pack.json";
```

If the assistant is served from a different origin than `/astron/`, change those
two constants — they are adjacent, near the top of the bridge block, and nothing
else refers to the paths.

## Order of operations

Any order is safe.  Upload the JSON first if you want the new behaviour to
switch on the moment the HTML lands; upload the HTML first and the page runs on
its deterministic fallback until the artifacts appear.  There is no window in
which the page is broken:

* HTML present, artifacts absent → the identity ladder still runs (it is not
  learned), the family router falls back to the shipped heuristic, conversation
  falls back to the ChatterBot corpus and then to built-in replies.
* Artifacts present, old HTML → the files are simply never requested.

## Server requirements

* The two JSON files must be served with any `Content-Type` a browser will hand
  to `response.json()` (`application/json` is right; the default for `.json` on
  nginx/Apache is already correct).
* Same origin as the page.  No CORS headers are needed and none are requested.
* No compression requirement; they are ~29 KB and ~23 KB.
* Caching: the page keys its `localStorage` copy on the artifact's own
  `content_hash` and re-fetches after 24 h.  A normal `ETag`/`Last-Modified`
  setup is enough — a redeploy changes the hash, and the browser picks it up on
  its next TTL expiry or on a hard reload.

## Verifying a deploy

From the browser console on the live page:

```js
robots.selfTest()          // 19/19 — the identity-ladder regressions
robots.textModel()         // the parsed model, or null if it did not load
robots.conversationPack()  // the parsed pack, or null
robots.diagnose()          // which public databases answer from this browser
```

The status line under the chat log also shows it: `ASTRON text N feats` when the
model loaded, `ASTRON text off` when it did not, and hovering gives the exact
state (`cache`, `network`, `unavailable:timeout`, …).

## Regenerating the artifacts

```bash
cd astron/xyz
python3 -m text.export                                  # refit + write both
python3 -m text.export --stats text/data/oasst_stats.json   # with a harvest
```

`export.py` prints the byte size, the content hash and the deploy path of each
file.  The fit takes under a second and needs nothing outside the standard
library.

## Nothing else in ASTRON changes

`engine/`, `gabriel/`, `bench/`, `policy/gabriel_lm.json` and `policy/policy.json`
are untouched by this work, with one exception, which is additive:
`engine/solvers/counting.py` is a new solver module and one line of
`engine/portfolio.py` registers it.  The ARC cron
(`scripts/cron_evolve.sh`) needs no change.
