"""Export the browser artifacts.

The integration is deliberately *static*.  ``README.OPS.md`` is explicit that
ASTRON runs no daemon and listens on no port, so a page that called
``cell4.art/astron/`` as an inference API would be calling something that does
not exist.  What does exist is a served directory -- so the connection is:

    ASTRON offline fit  ->  compact JSON under policy/ and data/
                        ->  robots.html fetches it same-origin, versioned
                        ->  the browser runs the same arithmetic locally

Two artifacts:

``policy/text_gabriel_lm.json``   the family head and the candidate head
``data/conversation_pack.json``   the dialogue-act model, the response plans,
                                  and the style statistics

Both are small (tens of kilobytes), both are cache-keyed by a content hash so a
redeploy invalidates the browser's copy and a re-fetch of an unchanged file is a
304, and both are optional: ``robots.html`` degrades to its existing
deterministic ranking and its built-in conversational fallback if either is
missing, malformed, or blocked.

Usage::

    python3 -m text.export                     # write both artifacts
    python3 -m text.export --stats text/data/oasst_stats.json
"""

import argparse
import hashlib
import json
import os
import time

from . import conversation as CONV
from . import corpus as C
from . import families as FAM
from . import lm as LM
from . import signatures as S
from . import train as TRAIN

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POLICY_DIR = os.path.join(ROOT, "policy")
DATA_DIR = os.path.join(ROOT, "data")
PACK_PATH = os.path.join(DATA_DIR, "conversation_pack.json")

# What the browser needs to reproduce the Python side exactly.  Exported so a
# drift between the two implementations is a data mismatch, not a silent
# behaviour change.
CONTRACT = {
    "tokenizer": "lowercase; [a-z0-9][a-z0-9'+#._-]* ; trailing ._- stripped; "
                 "length > 1; stop list applied for coverage only",
    "identity": "ordered tokens with multiplicity, stop words KEPT, trailing "
                "parenthetical split off as a qualifier",
    "tiers": FAM.TIER_NAMES,
    "pair_groups": [list(p) for p in LM.PAIR_GROUPS],
    "families": list(FAM.FAMILIES),
}


def _hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(payload)
    payload["content_hash"] = _hash(
        {k: v for k, v in payload.items() if k != "content_hash"})
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, sort_keys=True, separators=(",", ":"))
    os.replace(tmp, path)
    return os.path.getsize(path), payload["content_hash"]


def export_model(model, report, path=TRAIN.MODEL_PATH):
    d = model.to_dict()
    d["stop"] = sorted(S.STOP)
    d["contract"] = CONTRACT
    d["provenance"] = {
        "producer": "astron/xyz/text/train.py",
        "corpus": "astron/xyz/text/corpus.py (generated templates x placeholder "
                  "subjects; topic-blind by construction)",
        "gate": "dev loss on a template-disjoint slice; holdout measured once "
                "and gates nothing",
        "external_data": "none",
        "built": int(time.time()),
    }
    d["report"] = {k: report[k] for k in ("gate", "holdout", "baseline")
                   if k in report}
    return _write(path, d)


def export_conversation_pack(stats_path=None, path=PACK_PATH):
    """The dialogue pack.  Parameters and authored plans; no harvested text."""
    extra = []
    stats = CONV.load_statistics(stats_path) if stats_path else None
    provenance = {
        "act_model": "astron/xyz/text/conversation.py SEED labels (authored "
                     "here, CC0 as part of this repository)",
        "plans": "authored in astron/xyz/text/conversation.py; every string a "
                 "user can see originates there",
        "style_statistics": None,
        "verbatim_text_from_datasets": False,
    }
    if stats:
        provenance["style_statistics"] = {
            "dataset": "OpenAssistant/oasst1",
            "access": "https://datasets-server.huggingface.co (public, keyless)",
            "used": "act-transition counts, reply-length and sentence-count "
                    "buckets, question-back / hedge / first-person rates",
            "not_used": "message text -- no reply string is stored or shipped",
            "turns": stats.get("turns"),
        }
    model, mstats = CONV.train_act_model(extra=extra)
    payload = {
        "kind": "astron-conversation-pack",
        "version": 1,
        "acts": list(CONV.ACTS),
        "act_model": model.to_dict(),
        "act_model_stats": mstats,
        "plans": {a: [[skel, needs] for skel, needs in plans]
                  for a, plans in CONV.PLANS.items()},
        "fallback": CONV.FALLBACK,
        "style": stats or {},
        "markers": [[n, rx.pattern] for n, rx in CONV._MARKERS],
        "provenance": provenance,
        "built": int(time.time()),
    }
    return _write(path, payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=os.path.join(HERE, "data", "oasst_stats.json"),
                    help="harvest statistics file; ignored if absent")
    ap.add_argument("--model-out", default=TRAIN.MODEL_PATH)
    ap.add_argument("--pack-out", default=PACK_PATH)
    ap.add_argument("--no-fit", action="store_true",
                    help="export the incumbent model instead of refitting")
    a = ap.parse_args()

    if a.no_fit:
        model = LM.TextLM.load(a.model_out)
        if model is None:
            raise SystemExit("no incumbent model at %s" % a.model_out)
        report = {}
    else:
        model, report = TRAIN.fit()
        TRAIN.adopt(model, report, path=a.model_out, force=True)

    msize, mhash = export_model(model, report, path=a.model_out)
    psize, phash = export_conversation_pack(
        stats_path=a.stats if os.path.exists(a.stats) else None,
        path=a.pack_out)

    out = {
        "model": {"path": a.model_out, "bytes": msize, "hash": mhash},
        "pack": {"path": a.pack_out, "bytes": psize, "hash": phash},
        "oasst_stats_used": os.path.exists(a.stats),
        "deploy": {
            "/astron/xyz/policy/text_gabriel_lm.json": a.model_out,
            "/astron/xyz/data/conversation_pack.json": a.pack_out,
        },
    }
    if report:
        out["report"] = {k: report[k] for k in ("gate", "holdout", "baseline")}
    print(json.dumps(out, indent=2, sort_keys=True))


# --------------------------------------------------------------------------
# conformance vectors: what the JS side must reproduce
# --------------------------------------------------------------------------

CONFORMANCE_QUERIES = [
    "what is high school", "what is High School Musical", "math",
    "what is math rock", "who is Albert Einstein", "capital of France",
    "hey", "how are you", "tell me more", "anyway what's photosynthesis",
    "Java vs JavaScript", "explain quantum mechanics", "what does CELL4 do",
    "define entropy", "python?", "why do you sound so formal",
    "when was he born", "what is the matrix", "how does a transformer work",
    "  ", "???", "a", "what is what is what is",
]


def conformance_vectors(site_lexicon=C.SITE_TERMS):
    """Reference outputs the browser implementation must match exactly."""
    model = LM.TextLM.load(TRAIN.MODEL_PATH)
    rows = []
    for q in CONFORMANCE_QUERIES:
        subj, rel, parse = S.split_subject_relation(q)
        row = {
            "q": q,
            "form": S.query_form(q),
            "subject": subj, "relation": rel, "parse": parse,
            "signatures": list(S.signatures(q, site_lexicon=site_lexicon)),
            "families": list(FAM.grammatical_families(q, site_lexicon=site_lexicon)),
            "identity": FAM.identity_key(subj),
        }
        if model is not None:
            dist = model.family_dist(q, site_lexicon=site_lexicon)
            row["family_dist"] = {k: round(v, 9) for k, v in sorted(dist.items())}
        rows.append(row)
    pairs = [("high school", "High School High"),
             ("high school", "Secondary school"),
             ("high school musical", "High School Musical"),
             ("math", "MathJax"), ("math", "Mathematics"),
             ("math rock", "Math rock"), ("matrix", "The Matrix"),
             ("matrix", "Matrix (mathematics)"), ("apple", "Apple Inc."),
             ("school", "School of Rock"), ("cell", "Cell (biology)")]
    tiers = [{"asked": a, "title": t,
              "tier": FAM.tier(a, {"title": t})[0],
              "reason": FAM.tier(a, {"title": t})[1],
              "dup_mismatch": FAM.duplicate_mismatch(a, t),
              "containment": list(FAM.ordered_containment(a, t))}
             for a, t in pairs]
    return {"queries": rows, "tiers": tiers}


if __name__ == "__main__":
    main()
