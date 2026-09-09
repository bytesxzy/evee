"""Training data for the text model, and the split discipline around it.

``gabriel/corpus.py`` turns the engine's own run records into examples and is
careful about two things: only the fit split is ever read, and a further dev
slice inside it gates adoption.  The same two rules apply here.

What an example is
------------------

**Family examples** are ``(query signature, family)``.  They are generated from
*templates crossed with topic placeholders*, not from a list of topics with
answers attached.  That is not a shortcut -- it is the only honest way to build
this corpus, because :mod:`signatures` is topic-blind by construction.  The
model sees ``form:whatis len:1 case:lower`` for ``what is math`` and for ``what
is photosynthesis`` alike, so a corpus that named topics would be teaching the
model nothing it could use and everything it must not memorise.

**Candidate examples** are ``(query signature, candidate signature, label)``,
where the candidate signature is likewise categorical evidence shape: tier,
expansion extras, coverage bucket, coherence bucket, source, rank.  A weight
learned here says "a verified redirect beats a rank-1 expanded compound"; there
is no representation in which it could say "math means Mathematics".

Splits
------

``holdout_key`` hashes the *template family*, not the rendered string, so a
paraphrase of a training template cannot leak into the holdout.  The holdout is
measured once by :mod:`train` and gates nothing; the dev slice gates adoption.

The adversarial benchmark in :mod:`bench` draws from a disjoint template pool
and asserts the disjointness at run time, so a benchmark score is never a
report on the training set.
"""

import hashlib
import json
import os

from . import families as FAM
from . import lm as LM
from . import signatures as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# --------------------------------------------------------------------------
# placeholders.  Deliberately meaningless strings of the right *shape*: the
# corpus must teach shape, and a real topic here would be a memorisable answer.
# --------------------------------------------------------------------------

LOWER_1 = ["photosynthesis", "entropy", "gravity", "inflation", "osmosis",
           "cryptography", "topology", "metabolism", "sonar", "algebra"]
LOWER_2 = ["machine learning", "black hole", "supply chain", "prime number",
           "heat capacity", "civil engineering", "public policy", "tidal force"]
LOWER_3 = ["central limit theorem", "second law thermodynamics",
           "general purpose language", "double slit experiment"]
TITLE_1 = ["Kubernetes", "Einstein", "Namibia", "Voyager", "Bauhaus"]
TITLE_2 = ["Albert Einstein", "Marie Curie", "Hudson Bay", "Ada Lovelace"]
TITLE_3 = ["The Great Gatsby", "High Noon Express", "Blue Ridge Parkway"]
UPPER = ["NASA", "HTTP", "GDP", "RNA", "SQL"]

SUBJECT_POOL = {
    "lower1": LOWER_1, "lower2": LOWER_2, "lower3": LOWER_3,
    "title1": TITLE_1, "title2": TITLE_2, "title3": TITLE_3, "upper": UPPER,
}

RELATIONS = ["capital", "population", "author", "founder", "height",
             "currency", "atomic number", "release date"]

SITE_TERMS = ["CELL4", "robots.js", "proof of work", "the team",
              "your services", "contact", "inquire", "cell4.art"]

# --------------------------------------------------------------------------
# template families.  (template, family, pools) -- the family is the label.
# --------------------------------------------------------------------------

DEFINITION_TEMPLATES = [
    "what is {s}", "what's {s}", "what is a {s}", "what are {s}",
    "define {s}", "describe {s}", "what does {s} mean",
    "meaning of {s}", "definition of {s}", "tell me about {s}",
    "{s}?", "{s}", "could you explain {s}", "explain {s} to me",
    "i want to know about {s}", "what is {s} exactly",
]
IDENTITY_TEMPLATES = [
    "who is {s}", "who was {s}", "who's {s}", "who are {s}",
    "tell me who {s} is", "who is {s}?",
]
RELATION_TEMPLATES = [
    "{r} of {s}", "what is the {r} of {s}", "{s}'s {r}",
    "what's the {r} of {s}", "tell me the {r} of {s}",
]
EXPLANATION_TEMPLATES = [
    "why is {s} important", "why does {s} happen", "explain how {s} works",
    "how does {s} work", "how is {s} made", "why do we need {s}",
]
HOWTO_TEMPLATES = [
    "how do i learn {s}", "how to use {s}", "how can i start with {s}",
    "how should i approach {s}", "how do you build {s}",
]
COMPARISON_TEMPLATES = [
    "{s} vs {s2}", "{s} versus {s2}", "difference between {s} and {s2}",
    "compare {s} and {s2}", "{s} compared to {s2}",
]
CONVERSATIONAL_TEMPLATES = [
    "hey", "hi", "hello", "yo", "sup", "hey there", "good morning",
    "how are you", "how's it going", "how have you been", "you doing ok",
    "that's actually pretty cool", "that's wild", "nice", "cool", "lol",
    "haha", "lmao", "nah bro", "wait what", "huh", "oh interesting",
    "what do you think about that", "can we talk", "let's talk",
    "why do you sound so formal", "you sound like a robot",
    "i had a weird day", "i'm bored", "i'm tired", "i feel weird today",
    "that's not what i meant", "no i meant something else", "never mind",
    "thanks", "thank you", "ty", "appreciate it", "cheers",
    "ok", "okay", "alright", "sure", "fair enough", "makes sense",
    "do you like anything", "what's your favorite thing",
    "are you an ai", "what are you", "who made you",
]
SITE_TEMPLATES = [
    "what does {t} do", "who is on {t}", "how do i contact {t}",
    "what is {t}", "tell me about {t}", "show me {t}",
    "{t}", "explain {t}",
]
CONTINUATION_TEMPLATES = [
    "tell me more", "continue", "go on", "more", "expand on that",
    "and then", "what else", "keep going",
]

# Held out from *every* fit.  These templates exist only so `train.py` can
# report a transfer number, exactly as evolve's holdout does for ARC.
HOLDOUT_TEMPLATES = [
    ("what would you call {s}", "definition"),
    ("give me a rundown on {s}", "definition"),
    ("who exactly is {s}", "identity"),
    ("remind me what {s} means", "definition"),
    ("break down how {s} works", "explanation"),
    ("walk me through using {s}", "howto"),
    ("set {s} against {s2}", "comparison"),
    ("man that's rough", "conversational"),
    ("you still there", "conversational"),
    ("anything else worth knowing", "conversational"),
]


def _render(template, subject, subject2=None, relation=None, term=None):
    out = template
    if "{s2}" in out:
        out = out.replace("{s2}", subject2 or subject)
    out = out.replace("{s}", subject)
    if relation is not None:
        out = out.replace("{r}", relation)
    if term is not None:
        out = out.replace("{t}", term)
    return out


def _pool_cycle(pools, i):
    flat = []
    for name in pools:
        flat.extend(SUBJECT_POOL[name])
    return flat[i % len(flat)]


def generate_family_examples(include_holdout=False):
    """Every (query, family) pair the template grid produces."""
    rows = []
    pools = ["lower1", "lower2", "lower3", "title1", "title2", "title3", "upper"]

    def add(text, family, template):
        rows.append({"text": text, "family": family, "template": template})

    n = 0
    for t in DEFINITION_TEMPLATES:
        for p in pools:
            for s in SUBJECT_POOL[p]:
                add(_render(t, s), "definition", t)
                n += 1
    for t in IDENTITY_TEMPLATES:
        for p in ("title1", "title2", "title3", "upper"):
            for s in SUBJECT_POOL[p]:
                add(_render(t, s), "identity", t)
    for t in RELATION_TEMPLATES:
        for i, r in enumerate(RELATIONS):
            for p in ("title1", "title2", "lower1"):
                for s in SUBJECT_POOL[p][:4]:
                    add(_render(t, s, relation=r), "relational_fact", t)
    for t in EXPLANATION_TEMPLATES:
        for p in ("lower1", "lower2", "title1"):
            for s in SUBJECT_POOL[p]:
                add(_render(t, s), "explanation", t)
    for t in HOWTO_TEMPLATES:
        for p in ("lower1", "lower2", "upper"):
            for s in SUBJECT_POOL[p]:
                add(_render(t, s), "howto", t)
    for t in COMPARISON_TEMPLATES:
        for i in range(24):
            a = _pool_cycle(pools, i)
            b = _pool_cycle(pools, i + 7)
            add(_render(t, a, subject2=b), "comparison", t)
    for t in CONVERSATIONAL_TEMPLATES:
        add(t, "conversational", t)
    for t in CONTINUATION_TEMPLATES:
        add(t, "conversational", t)
    for t in SITE_TEMPLATES:
        for term in SITE_TERMS:
            add(_render(t, term, term=term), "site", t)
    if include_holdout:
        for t, fam in HOLDOUT_TEMPLATES:
            for i in range(6):
                a = _pool_cycle(pools, i)
                b = _pool_cycle(pools, i + 3)
                add(_render(t, a, subject2=b), fam, t)
    return rows


def holdout_key(template):
    return hashlib.md5(("astron-text-holdout-v1" + str(template)).encode()).hexdigest()


def is_dev(template, frac=0.18):
    d = int(holdout_key(template)[:2], 16) / 255.0
    return d < frac


def build_family_corpus(site_lexicon=SITE_TERMS):
    """(train, dev, holdout, stats).  Split by template, never by rendered text."""
    rows = generate_family_examples(include_holdout=True)
    hold_templates = set(t for t, _f in HOLDOUT_TEMPLATES)
    train, dev, hold = [], [], []
    for r in rows:
        sigs = S.signatures(r["text"], site_lexicon=site_lexicon)
        allowed = FAM.grammatical_families(r["text"], site_lexicon=site_lexicon)
        ex = {"text": r["text"], "sigs": sigs, "family": r["family"],
              "allowed": allowed, "template": r["template"], "weight": 1.0}
        if r["template"] in hold_templates:
            hold.append(ex)
        elif is_dev(r["template"]):
            dev.append(ex)
        else:
            train.append(ex)
    stats = {"rows": len(rows), "train": len(train), "dev": len(dev),
             "holdout": len(hold),
             "templates": len(set(r["template"] for r in rows)),
             "grammar_misses": sum(1 for e in train + dev + hold
                                   if e["family"] not in e["allowed"])}
    return train, dev, hold, stats


# --------------------------------------------------------------------------
# candidate corpus: evidence shapes, not topics
# --------------------------------------------------------------------------

def _cand(title, **kw):
    d = {"title": title}
    d.update(kw)
    return d


CANDIDATE_SCENARIOS = [
    # (asked phrase, query template, [(candidate, is_intended)])
    # -- a verified redirect must beat a rank-1 expanded compound
    ("high school", "what is {s}", [
        (_cand("Secondary school", verified_title=True, alias="high school",
               source="wikipedia", rank=0, coherence=0.86, full_text=True), 1),
        (_cand("High School High", source="wikipedia", rank=1, coherence=0.42), 0),
        (_cand("High School Musical", source="wikipedia", rank=2, coherence=0.35), 0),
    ]),
    # -- an exact ordered phrase must beat a longer compound
    ("math rock", "what is {s}", [
        (_cand("Math rock", source="wikipedia", rank=1, coherence=0.81, full_text=True), 1),
        (_cand("Math rock discography", source="wikipedia", rank=2, coherence=0.30), 0),
    ]),
    # -- a lexical near-miss must lose to a verified alias
    ("ai", "what is {s}", [
        (_cand("Artificial intelligence", verified_title=True, alias="AI",
               source="wikipedia", rank=0, coherence=0.88, full_text=True), 1),
        (_cand("Ai (singer)", source="wikipedia", rank=1, coherence=0.20), 0),
    ]),
    # -- a specific named work asked for explicitly must win
    ("high school musical", "what is {s}", [
        (_cand("High School Musical", verified_title=True, alias="High School Musical",
               source="wikipedia", rank=0, coherence=0.90, full_text=True), 1),
        (_cand("High School Musical 2", source="wikipedia", rank=1, coherence=0.55), 0),
    ]),
    # -- a disambiguation page is not an answer when a sense is available.
    #    The subject is chosen from outside the benchmark on purpose; an earlier
    #    draft used "mercury", which the benchmark holds out.
    ("mercator", "what is {s}", [
        (_cand("Mercator projection", source="wikipedia", rank=2, coherence=0.72,
               full_text=True), 1),
        (_cand("Mercator", source="wikipedia", rank=1, coherence=0.30,
               disambiguation=True), 0),
    ]),
    # -- structured data with a matching relation wins a relational question
    ("france", "capital of {s}", [
        (_cand("France", source="wikidata", rank=0, coherence=0.80), 1),
        (_cand("Capital punishment in France", source="wikipedia", rank=1,
               coherence=0.25), 0),
    ]),
    # -- rank alone is not evidence: a rank-1 fuzzy hit loses to a rank-3 exact
    ("rust", "what is {s}", [
        (_cand("Rust (programming language)", source="wikipedia", rank=3,
               coherence=0.78, full_text=True), 1),
        (_cand("Rustenburg", source="wikipedia", rank=1, coherence=0.12), 0),
    ]),
    # -- the read article beats the search snippet for the same concept
    ("photosynthesis", "what is {s}", [
        (_cand("Photosynthesis", verified_title=True, alias="photosynthesis",
               source="wikipedia", rank=0, coherence=0.92, full_text=True), 1),
        (_cand("Photosynthesis in plants", source="duckduckgo", rank=1,
               coherence=0.60), 0),
    ]),
]

# Held out from the fit AND chosen so that no subject here appears in the
# browser benchmark -- `text/bench.py --audit` asserts that disjointness. An
# earlier draft used "jaguar" and "the matrix", both of which the benchmark
# also uses, which would have made the holdout a report on the training set.
CANDIDATE_HOLDOUT = [
    ("bauhaus", "what is {s}", [
        (_cand("Bauhaus", verified_title=True, alias="bauhaus", source="wikipedia",
               rank=0, coherence=0.85, full_text=True), 1),
        (_cand("Bauhaus (band)", source="wikipedia", rank=1, coherence=0.44), 0),
    ]),
    ("the hague", "what is {s}", [
        (_cand("The Hague", verified_title=True, alias="The Hague",
               source="wikipedia", rank=0, coherence=0.88, full_text=True), 1),
        (_cand("Hague Convention", source="wikipedia", rank=1, coherence=0.50), 0),
    ]),
]

QUERY_SHAPES = ["what is {s}", "what's {s}", "define {s}", "tell me about {s}",
                "{s}", "{s}?", "explain {s}", "could you explain {s}",
                "what does {s} mean", "who is {s}"]


def build_candidate_corpus():
    """(train, dev, holdout, stats) of (qsigs, csigs, label) triples."""
    def expand(scenarios):
        out = []
        for asked, template, cands in scenarios:
            shapes = QUERY_SHAPES if template in ("what is {s}",) else [template]
            for shape in shapes:
                text = shape.replace("{s}", asked)
                qsigs = S.signatures(text)
                pruned, _best, _n = FAM.prune(
                    asked, [c for c, _y in cands], focused=True)
                by_title = {c["title"]: c for c in pruned}
                for cand, label in cands:
                    merged = by_title.get(cand["title"])
                    if merged is None:
                        # the hard verifier already rejected it; the learned head
                        # never sees it, which is the point
                        continue
                    out.append({"text": text, "asked": asked,
                                "qsigs": qsigs,
                                "csigs": LM.candidate_signature(asked, merged),
                                "label": label, "title": cand["title"]})
        return out

    rows = expand(CANDIDATE_SCENARIOS)
    hold = expand(CANDIDATE_HOLDOUT)
    train, dev = [], []
    for i, r in enumerate(rows):
        (dev if is_dev(r["asked"] + "|" + r["text"], 0.2) else train).append(r)
    stats = {"rows": len(rows), "train": len(train), "dev": len(dev),
             "holdout": len(hold)}
    return train, dev, hold, stats


# --------------------------------------------------------------------------
# conversation pack input
# --------------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows
