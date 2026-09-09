"""Competing hypothesis families, and the verifier that disposes of them.

ASTRA's strongest property is that a solver *proposes* and a checker *disposes*:
a hypothesis reproduces every training pair or it is discarded, however
confident the model that wrote it.  This module is that discipline for senses.

Two layers, and the order between them is the whole point:

**Hard semantic validity** (this file, below the learned model).  A candidate
either can or cannot be the thing the user named.  That is decided by ordered,
multiplicity-preserving token identity and by verified redirect evidence --
never by a weight, never by a vote, never by feedback.

**Learned preference** (``lm.py``, above).  Among candidates the verifier has
already accepted, a fitted log-linear model chooses.  It can reorder valid
hypotheses.  It cannot make an invalid one valid.

Feedback therefore cannot teach ``math = MathJax``: MathJax never reaches the
learned layer, because ``math`` is not an ordered token subsequence of
``mathjax`` and no redirect says it is.

The identity ladder
-------------------

::

    5  VERIFIED_TITLE      an exact title or a redirect the encyclopedia confirmed
    4  VERIFIED_ALIAS      a confirmed canonical alias of what the user typed
    3  EXACT_PHRASE        title identity equals query identity, order + multiplicity
    2  COHERENT            a different name, but the evidence is about this concept
    1  EXPANDED            the query is a prefix-compound of a longer title
    0  FUZZY               tokens overlap and nothing else does

Rule enforced by :func:`prune`: once any candidate reaches tier 3 or better, no
candidate at tier 1 or below may be answered.  ``high school`` cannot become
``High School High`` while ``Secondary school`` is on the table -- and if it is
not on the table, the answer is a disclosed sense or a clarification, never a
silent substitution.

A TOKEN SET IS ALLOWED FOR COVERAGE.  A TOKEN SET IS NOT SUFFICIENT FOR
IDENTITY.  Every function below that says "identity" uses ordered tokens with
multiplicity; every function that says "coverage" may use a set.
"""

import re

from . import signatures as S

# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------

FAMILIES = (
    "definition",             # "what is X" -- establish what X *is*
    "identity",               # "who is X"  -- establish who X *is*
    "relational_fact",        # "capital of France"
    "explanation",            # "why/how does X work"
    "comparison",             # "X vs Y"
    "howto",                  # "how do I X"
    "conversational",         # social turn, no factual claim owed
    "site",                   # the host site is the authority
    "ambiguous",              # several senses survive; say so
    "clarification_required",  # nothing survives verification; ask
)

RESEARCH_FAMILIES = frozenset(
    ("definition", "identity", "relational_fact", "explanation",
     "comparison", "howto", "ambiguous"))

TIER_VERIFIED_TITLE = 5
TIER_VERIFIED_ALIAS = 4
TIER_EXACT_PHRASE = 3
TIER_COHERENT = 2
TIER_EXPANDED = 1
TIER_FUZZY = 0

TIER_NAMES = {5: "verified_title", 4: "verified_alias", 3: "exact_phrase",
              2: "coherent", 1: "expanded", 0: "fuzzy"}

# A title's trailing parenthetical is Wikipedia's disambiguator, not part of the
# concept name: "Matrix (mathematics)" is the concept "matrix", qualified.
_QUALIFIER = re.compile(r"\s*\(([^)]{1,60})\)\s*$")
_PUNCT = re.compile(r"[^a-z0-9'\s]+")


# --------------------------------------------------------------------------
# identity -- ordered, with multiplicity.  Never a set.
# --------------------------------------------------------------------------

def split_qualifier(title):
    """``"Matrix (mathematics)"`` -> ``("Matrix", "mathematics")``."""
    t = re.sub(r"\s+", " ", str(title or "")).strip()
    m = _QUALIFIER.search(t)
    if not m:
        return t, ""
    return t[:m.start()].strip(), m.group(1).strip()


def identity_tokens(phrase):
    """Every word of ``phrase``, lowercased, in order, duplicates kept.

    Stop words are **kept**.  ``matrix`` and ``the matrix`` are different
    concepts and the only thing separating them is a word a coverage filter
    would have thrown away.  Dropping duplicates here is the exact bug that let
    ``High School High`` impersonate ``high school``; dropping ``the`` is the
    same bug wearing a different hat.
    """
    base, _q = split_qualifier(phrase)
    base = _PUNCT.sub(" ", str(base or "").lower())
    return [w for w in base.split() if w]


def identity_key(phrase):
    """Canonical identity string.  Equality here means "the same name"."""
    return " ".join(identity_tokens(phrase))


def identity_equal(a, b):
    ka, kb = identity_key(a), identity_key(b)
    return bool(ka) and ka == kb


def ordered_containment(query, title):
    """How ``query`` sits inside ``title``, preserving order and multiplicity.

    Returns ``(contained, extras)``.  ``contained`` is True when every query
    token appears in ``title`` in the same order, each match consuming one
    title token -- so ``high school`` is contained in ``high school high`` with
    ``extras = 1``, and is *not* silently equal to it.
    """
    qt, ht = identity_tokens(query), identity_tokens(title)
    if not qt or not ht or len(ht) < len(qt):
        return False, 0
    qi = 0
    for tok in ht:
        if qi < len(qt) and tok == qt[qi]:
            qi += 1
    if qi != len(qt):
        return False, 0
    return True, len(ht) - len(qt)


def duplicate_mismatch(query, title):
    """True when a *set* comparison would call these equal and identity does not.

    This is the named regression: ``{high, school}`` is the token set of both
    ``high school`` and ``High School High``.  Any future change that starts
    comparing sets will make this function return True on a pair it must reject,
    and ``bench.py`` asserts on it.
    """
    qt, ht = identity_tokens(query), identity_tokens(title)
    if not qt or not ht:
        return False
    if set(qt) != set(ht):
        return False
    return qt != ht


# --------------------------------------------------------------------------
# coverage -- a set is fine here, and only here
# --------------------------------------------------------------------------

def coverage(phrase, text):
    need = set(S.tokens(phrase))
    if not need:
        return 0.0
    got = set(S.tokens(text))
    return len(need & got) / float(len(need))


def ordered_phrase_present(phrase, text, gap=0):
    """Does ``phrase`` occur in ``text`` in order, as a unit?"""
    need, hay = identity_tokens(phrase), identity_tokens(text)
    if not need:
        return False
    for i in range(len(hay)):
        if hay[i] != need[0]:
            continue
        at, ok = i, True
        for n in range(1, len(need)):
            found = -1
            for j in range(at + 1, min(at + 2 + gap, len(hay))):
                if hay[j] == need[n]:
                    found = j
                    break
            if found < 0:
                ok = False
                break
            at = found
        if ok:
            return True
    return False


# --------------------------------------------------------------------------
# the tier decision
# --------------------------------------------------------------------------

def tier(query_phrase, cand):
    """Identity tier of one candidate for one asked phrase.

    ``cand`` is a dict with at least ``title``; optionally ``verified_title``
    (the encyclopedia resolved the asked phrase to this page), ``alias`` (the
    spelling it resolved *from*), ``coherence`` (0..1 evidence-topic score) and
    ``disambiguation``.

    Returns ``(tier, reason)``.
    """
    asked = str(query_phrase or "").strip()
    title = str(cand.get("title") or "")
    if not asked or not title:
        return TIER_FUZZY, "no-phrase"

    if cand.get("verified_title"):
        # The encyclopedia was asked "do you know this exact name?" and said
        # yes, possibly through a redirect.  A redirect may legitimately change
        # every token ("high school" -> "Secondary school"): that is equivalence
        # established by the source, not a lexical guess, so it outranks
        # everything below and the expansion penalty must not apply to it.
        alias = str(cand.get("alias") or "")
        if alias and identity_equal(alias, asked):
            return TIER_VERIFIED_TITLE, "redirect-from-asked"
        if identity_equal(title, asked):
            return TIER_VERIFIED_TITLE, "exact-title"
        return TIER_VERIFIED_ALIAS, "verified-alias"

    if identity_equal(title, asked):
        return TIER_EXACT_PHRASE, "identity-equal"

    contained, extras = ordered_containment(asked, title)
    if contained and extras > 0:
        # An expanded compound title.  It is a *different, more specific*
        # concept than the one that was named, and it stays below every
        # candidate that is actually the named concept.
        return TIER_EXPANDED, "expanded+%d" % extras

    if float(cand.get("coherence") or 0.0) >= 0.55 and \
            coverage(asked, cand.get("text") or title) >= 0.5:
        return TIER_COHERENT, "coherent-sense"

    return TIER_FUZZY, "lexical"


def prune(query_phrase, candidates, focused=True):
    """Apply the ladder.  Returns ``(kept, best_tier, notes)``.

    The invariant, stated once and enforced here: **a candidate at a strictly
    worse tier can never be answered while a better tier is available.**  The
    learned model runs afterwards, on ``kept``, and orders within tiers.
    """
    notes = []
    scored = []
    for c in candidates:
        t, why = tier(query_phrase, c)
        c = dict(c)
        c["tier"] = t
        c["tier_reason"] = why
        c["tier_name"] = TIER_NAMES[t]
        scored.append(c)
        notes.append((c.get("title"), t, why))
    if not scored:
        return [], -1, notes
    best = max(c["tier"] for c in scored)
    if not focused:
        return scored, best, notes
    floor = TIER_EXPANDED + 1 if best >= TIER_EXACT_PHRASE else 0
    kept = [c for c in scored if c["tier"] >= floor]
    return kept, best, notes


# --------------------------------------------------------------------------
# family proposal (grammar) -- the model reweights, it does not overrule
# --------------------------------------------------------------------------

def grammatical_families(raw, site_lexicon=(), state=None):
    """Families the *grammar* of the query admits, before any learned score.

    This is the constrained-decoding grammar: the model may only choose among
    families this returns, exactly as ``beam_chains`` may only emit operators
    the engine actually has.
    """
    sig = set(S.signatures(raw, state=state, site_lexicon=site_lexicon))
    subj, rel, _parse = S.split_subject_relation(raw)
    form = S.query_form(raw)
    out = []

    if "site:yes" in sig:
        out.append("site")
    if form == "compare":
        out.append("comparison")
    if form == "howto":
        out.append("howto")
        out.append("explanation")
    if form == "howwork":
        out.append("explanation")
        out.append("definition")
    if form == "why":
        # "explain X" is a definition when X is a bare noun and an explanation
        # when X is a process.  Both stay on the table; the model decides.
        out.append("explanation")
        out.append("definition")
    if rel and subj:
        out.append("relational_fact")
    if form == "whois" or re.search(r"\bwho\b", str(raw or ""), re.I):
        out.append("identity")
    if form in ("whatis", "define", "tellme", "bare_q", "when", "where", "list"):
        out.append("definition")
    if form == "bare" and subj and "convo:no" in sig:
        out.append("definition")
    if "convo:yes" in sig or form in ("meta", "social") or (form == "bare" and not subj):
        out.append("conversational")
    # A short utterance with no interrogative shape can be a social turn -- but
    # only when it is not already a content question.  "capital of France" is
    # six tokens with no question word, and admitting `conversational` for it
    # would put a factual lookup on the same ballot as small talk.  The gate is
    # the same one the browser applies as a hard rule: no subject, a
    # conversational marker, an unresolved pronoun, a continuation, or an
    # explicitly second-person form.
    if form in ("bare", "bare_q", "social", "meta") and len(S.raw_tokens(raw)) <= 6:
        if (not subj) or (not rel and ("convo:yes" in sig or "convo:phrase" in sig or
                                       "pron:yes" in sig or "cont:yes" in sig)) or \
                form in ("social", "meta"):
            out.append("conversational")
    if "pron:yes" in sig or "cont:yes" in sig:
        # Anaphora resolves against dialogue state; the family is whatever the
        # previous turn was about, so both branches stay open.
        out.append("definition")
        out.append("conversational")
    if not out:
        out.append("definition")
        out.append("conversational")
    seen, uniq = set(), []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return tuple(uniq)
