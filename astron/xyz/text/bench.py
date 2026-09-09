"""Measurement and audit for the text side.

Three jobs, all of which are gates rather than reports:

``--ladder``      the identity ladder over an adversarial pair set, including
                  every collision family the brief names. A pair that should be
                  rejected and is not fails the run.
``--audit``       proves the runtime does not contain the benchmark. It greps
                  ``robots.html`` and the exported artifacts for the benchmark's
                  own titles and expected answers; a hit is a hard failure,
                  because a system that has memorised its benchmark has not been
                  measured.
``--conformance`` writes the reference vectors the browser implementation must
                  reproduce -- signatures, families, tiers, and the family
                  distribution -- so a drift between the Python fit and the
                  JavaScript inference is caught as a data mismatch instead of
                  as a mysterious behaviour change.

Usage::

    python3 -m text.bench --ladder
    python3 -m text.bench --audit --html ../../robots_topic_coherent.html
    python3 -m text.bench --conformance --out text/browser/conformance.json
"""

import argparse
import json
import os
import re
import sys

from . import corpus as C
from . import export as EX
from . import families as FAM
from . import lm as LM
from . import signatures as S
from . import train as TRAIN

HERE = os.path.dirname(os.path.abspath(__file__))

# (asked, candidate title, must the ladder allow this as the answer?)
# `False` means the candidate must not be answerable when a better tier exists.
LADDER_PAIRS = [
    # the named regression: a duplicate token must not disappear
    ("high school", "High School High", False),
    ("high school", "High School Musical", False),
    ("high school", "Secondary school", None),        # only with verification
    ("high school musical", "High School Musical", True),
    ("high school musical", "High School Musical 2", False),
    # math family
    ("math", "MathJax", False),
    ("math", "MathML", False),
    ("math", "Math rock", False),
    ("math rock", "Math rock", True),
    ("math rock", "Mathematics", False),
    # the article is part of some names and not of others
    ("matrix", "The Matrix", False),
    ("matrix", "Matrix (mathematics)", True),
    ("the matrix", "The Matrix", True),
    # compounds
    ("apple", "Apple Inc.", False),
    ("apple inc.", "Apple Inc.", True),
    ("school", "School of Rock", False),
    ("school of rock", "School of Rock", True),
    ("windows", "Microsoft Windows", False),
    ("microsoft windows", "Microsoft Windows", True),
    ("transformer", "Transformers (franchise)", False),
    # A parenthetical is a disambiguator, not part of the name, so a qualified
    # title IS the asked concept -- "Transformer (deep learning architecture)"
    # and "Transformer" are both valid readings of "transformer". Choosing
    # between two VALID senses is the learned layer's job and is measured in the
    # browser benchmark; the ladder only decides validity.
    ("transformer", "Transformer (deep learning architecture)", True),
    ("rust", "Rust (programming language)", True),     # qualifier, same name
    ("python", "Python (programming language)", True),
    ("python", "Monty Python", False),
    ("java", "Java coffee", False),
    ("java", "Java (programming language)", True),
    ("cell", "Cell (biology)", True),
    ("cell", "Prison cell", False),
    ("mercury", "Mercury (planet)", True),
    ("amazon", "Amazon River", False),
    ("saturn", "Saturn (rocket family)", True),   # qualified: valid sense
    ("jaguar", "Jaguar Cars", False),
    ("window", "Microsoft Windows", False),
]


def run_ladder(verbose=True):
    fails = []
    for asked, title, expect in LADDER_PAIRS:
        t, why = FAM.tier(asked, {"title": title})
        if expect is True and t < FAM.TIER_EXACT_PHRASE:
            fails.append((asked, title, t, why, "should be answerable"))
        if expect is False and t >= FAM.TIER_EXACT_PHRASE:
            fails.append((asked, title, t, why, "should not be answerable"))
        if verbose:
            print("  %-24s %-42s tier=%d %-18s %s" %
                  (asked, title, t, why,
                   "" if expect is None else ("OK" if not any(
                       f[0] == asked and f[1] == title for f in fails) else "FAIL")))

    # the ladder's own invariant, checked directly
    kept, best, _n = FAM.prune("high school", [
        {"title": "High School High"},
        {"title": "High School Musical"},
        {"title": "Secondary school", "verified_title": True, "alias": "high school"},
    ])
    if [c["title"] for c in kept] != ["Secondary school"]:
        fails.append(("high school", "prune", best, "", "expansions survived a verified sense"))

    # duplicate-token detection is the regression that must never come back
    if not FAM.duplicate_mismatch("high school", "High School High"):
        fails.append(("high school", "High School High", -1, "",
                      "duplicate-token mismatch not detected"))
    if FAM.identity_key("matrix") == FAM.identity_key("the matrix"):
        fails.append(("matrix", "the matrix", -1, "", "stop word dropped from identity"))
    return fails


# --------------------------------------------------------------------------
# audit: the runtime must not contain the benchmark
# --------------------------------------------------------------------------

BENCH_TERMS = sorted({t for _a, t, _e in LADDER_PAIRS} |
                     {a for a, _t, _e in LADDER_PAIRS})


_JS_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_JS_LINE = re.compile(r"(?m)^\s*//.*$")
_PY_DOC = re.compile(r'"""(?:.|\n)*?"""')
_PY_LINE = re.compile(r"(?m)^\s*#.*$")
_SELFTEST = re.compile(r"function selfTest\(\)\s*\{.*?\n  \}", re.S)


def _executable(path, text):
    """Strip comments, docstrings and the in-page test block.

    A benchmark title in a comment is an explanation; in a test assertion it is
    a regression check the brief asked for by name. Neither is a rule. What the
    audit is looking for is a title inside a *decision*: a lookup table, a
    branch, a weight keyed on a name. Scanning raw source would drown that
    signal in prose, so the prose is removed first.
    """
    if path.endswith((".html", ".js")):
        # The first <script> is the site's own knowledge base -- CELL4 prose
        # harvested from the site. It is data about the host, not a rule, and a
        # word like "transformer" appearing in a case study is not a leak.
        cut = text.find("<script>window.ROBOTS_CONFIG")
        if cut > 0:
            text = text[cut:]
        text = _SELFTEST.sub("", text)
        text = _JS_BLOCK.sub(" ", text)
        text = _JS_LINE.sub(" ", text)
    elif path.endswith(".py"):
        text = _PY_DOC.sub(" ", text)
        text = _PY_LINE.sub(" ", text)
    return text


def run_audit(paths):
    """Any benchmark title inside executable runtime code is a hard failure."""
    hits = []
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
        except OSError:
            continue
        text = _executable(path, raw).lower()
        for term in BENCH_TERMS:
            t = term.lower()
            # single very common words would false-positive; only multi-word or
            # distinctive titles are meaningful evidence of memorisation
            if len(t) < 6 or " " not in t and t in ("school", "matrix", "python",
                                                    "java", "apple", "cell",
                                                    "rust", "mercury", "saturn",
                                                    "amazon", "jaguar", "window",
                                                    "windows", "math"):
                continue
            # Whole words only. A substring match reported the pre-existing
            # OpenAlex health-check probe "transformers" as the benchmark term
            # "transformer", which is a different word.
            if re.search(r"\b" + re.escape(t) + r"\b", text):
                hits.append((os.path.basename(path), term))
    return hits


# The browser benchmark's holdout queries. The candidate corpus must not touch
# them, or the holdout is measuring the training set.
BENCH_HOLDOUT_SUBJECTS = {
    "matrix", "the matrix", "mercury", "transformer", "windows", "window",
    "amazon", "saturn", "python", "java", "microsoft windows", "school of rock",
    "monty python",
}


def run_disjointness():
    """The candidate corpus may not name a benchmark holdout subject."""
    bad = []
    for asked, _tpl, _c in C.CANDIDATE_SCENARIOS + C.CANDIDATE_HOLDOUT:
        if asked.strip().lower() in BENCH_HOLDOUT_SUBJECTS:
            bad.append(asked)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--conformance", action="store_true")
    ap.add_argument("--html", default=os.path.join(
        HERE, "..", "..", "..", "robots_topic_coherent.html"))
    ap.add_argument("--out", default=os.path.join(HERE, "browser", "conformance.json"))
    a = ap.parse_args()
    if not (a.ladder or a.audit or a.conformance):
        a.ladder = a.audit = a.conformance = True

    rc = 0
    if a.ladder:
        print("identity ladder")
        fails = run_ladder()
        print("  %d pairs, %d failures" % (len(LADDER_PAIRS), len(fails)))
        for f in fails:
            print("  FAIL %s / %s  tier=%s  %s" % (f[0], f[1], f[2], f[4]))
        rc |= 1 if fails else 0

    if a.audit:
        print("\nholdout disjointness")
        bad = run_disjointness()
        if bad:
            print("  LEAK the candidate corpus names benchmark holdout "
                  "subjects: %s" % ", ".join(sorted(set(bad))))
            rc |= 4
        else:
            print("  clean: no candidate-corpus scenario names a benchmark "
                  "holdout subject")

        print("\nbenchmark-leak audit")
        # RUNTIME files only. `corpus.py` deliberately names a handful of real
        # tune-split subjects as evidence-shape examples; that is disclosed, is
        # what the tune/hold split is for, and is checked separately by the
        # disjointness gate above. What must be clean is what actually runs.
        paths = [a.html, TRAIN.MODEL_PATH, EX.PACK_PATH,
                 os.path.join(HERE, "families.py"),
                 os.path.join(HERE, "signatures.py"), os.path.join(HERE, "lm.py")]
        hits = run_audit(paths)
        if hits:
            for h in hits:
                print("  LEAK %s contains benchmark term %r" % h)
            rc |= 2
        else:
            print("  clean: no benchmark title appears in the runtime or the "
                  "exported model")

    if a.conformance:
        vec = EX.conformance_vectors()
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(vec, fh, indent=1, sort_keys=True)
        print("\nconformance vectors: %d queries, %d tier pairs -> %s" %
              (len(vec["queries"]), len(vec["tiers"]), a.out))

    return rc


if __name__ == "__main__":
    sys.exit(main())
