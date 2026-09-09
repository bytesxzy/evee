"""Text query signatures -- the analogue of ``engine.learn.signatures``.

ASTRA reduces an ARC task to a handful of *categorical* facts (does the shape
change, does the palette grow, is there a separator colour) and conditions its
priors on them.  It works because the signature is cheap, computed from
information the solver is allowed to have, and coarse enough that a small
corpus covers it.

The same three properties are what a text router needs.  A query is reduced
here to categorical tokens describing its *shape*, never its topic: the word
"photosynthesis" contributes ``len:1`` and ``case:lower``, exactly as
"mitochondria" does.  A signature that named entities would be a lookup table
with a topic list inside it, which is the thing this whole package exists to
avoid.

Signature groups
----------------

``form:``     the interrogative / imperative shape of the utterance
``len:``      how many meaningful (non-stop) tokens the subject has
``sub:``      subject length bucket, ``rel:`` relation length bucket
``case:``     lowercase / Title Case / MIXED -- named-looking phrase structure
``dup:``      does the subject repeat a token (``high school high``)
``punct:``    trailing question mark, presence of quotes
``convo:``    conversational markers present
``pron:``     an unresolved pronoun / bare anaphor
``shift:``    an explicit topic-shift marker
``cont:``     an explicit continuation marker
``site:``     a token from the host-site lexicon
``ctx:``      dialogue-state relationship to the previous turn

Every token here is also computed, character-for-character identically, by
``robots.html``.  ``bench.py --conformance`` compares the two implementations
over a fixed query list and fails on any disagreement; if they drift, the
browser is scoring against features the model was not fitted on.
"""

import re

# --------------------------------------------------------------------------
# tokenising -- deliberately identical to the browser's `tok()`
# --------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9][a-z0-9'+#._-]*")

STOP = set("""
a an the of for to in on at by with from as is are was were be been being am
do does did done have has had having will would can could should may might
must shall about into over under and or but not no nor so yet if then than
that this these those it its it's i you he she they we me him her them us my
your his their our what which who whom whose when where why how
""".split())

# Words that carry query *shape* rather than topic.  They are stripped from the
# subject but never from the raw text, because their presence is a signal.
CONVO_MARKERS = [
    "hey", "hi", "hello", "yo", "sup", "howdy", "morning", "evening",
    "thanks", "thank", "thx", "ty", "cheers", "please", "sorry", "oops",
    "lol", "lmao", "haha", "hah", "heh", "bruh", "bro", "dude", "man",
    "nah", "yeah", "yep", "yup", "nope", "ok", "okay", "alright", "cool",
    "sure", "exactly", "mhm", "mm", "yea", "aight",
    "nice", "sweet", "damn", "wow", "huh", "hmm", "wait", "oh", "ah",
    "bored", "tired", "sad", "happy", "weird", "crazy", "fine", "meh",
]
SHIFT_MARKERS = [
    "anyway", "anyways", "actually", "nevermind", "never", "forget",
    "different", "instead", "switching", "switch", "unrelated", "aside",
    "changing", "moving", "new",
]
CONT_MARKERS = [
    "more", "continue", "go", "elaborate", "expand", "further", "else",
    "also", "and", "next", "keep", "again",
]
PRONOUNS = ["he", "she", "it", "they", "him", "her", "them", "his", "hers",
            "their", "theirs", "its", "that", "this", "those", "these", "one"]

# Multi-word social cues.  A single-token marker list cannot see "how have you
# been" or "that's not what I meant", and without them the grammar refused to
# offer `conversational` for turns that are obviously conversational.
SOCIAL_PHRASE = re.compile(
    r"\b(?:how (?:have you been|are you|is it going|'?s it going|you doing)"
    r"|can we talk|let'?s talk|talk to me|wanna chat"
    r"|(?:that'?s )?not what i meant|no worries|my bad|fair enough|makes sense"
    r"|good point|sounds good|got it|i see|never ?mind"
    r"|who made you|what are you|are you (?:an? )?(?:ai|bot|robot|human|real)"
    r"|that'?s (?:wild|cool|neat|nice|crazy|fair|true|right))\b", re.I)


def tokens(text):
    """Lowercased word tokens, stop words removed, order and multiplicity kept.

    Multiplicity is load-bearing.  ``high school high`` must tokenise to three
    tokens, not two: collapsing to a set is exactly the bug that let a compound
    title impersonate an exact concept.
    """
    out = []
    for m in _WORD.findall(str(text or "").lower()):
        w = m.rstrip("._-")
        if len(w) > 1 and w not in STOP:
            out.append(w)
    return out


def raw_tokens(text):
    """Every word token, stop words included -- for shape tests."""
    return [m.rstrip("._-") for m in _WORD.findall(str(text or "").lower())]


def _bucket(n, edges):
    for i, e in enumerate(edges):
        if n <= e:
            return i
    return len(edges)


def _has_any(toks, words):
    s = set(toks)
    return any(w in s for w in words)


# --------------------------------------------------------------------------
# query form
# --------------------------------------------------------------------------

# Order matters: the first match wins, so the more specific shape is listed
# above the shape that would otherwise swallow it.  "how does X work" is an
# explanation, not a how-to; "what do you think" is a social turn, not a
# definition lookup.  Getting this order wrong is not cosmetic -- it decides
# which families the grammar will even offer the model.
_FORMS = [
    ("social", re.compile(r"^\s*(?:what|how)(?:'s|\u2019s)?\s+(?:up|good|new|going)\b", re.I)),
    ("meta", re.compile(r"^\s*(?:what|who|how)(?:'s|\u2019s)?\s+(?:are|is|do|did|does)?\s*"
                        r"(?:you|u)\s*[?.!]*$"
                        r"|^\s*(?:what|which)(?:'s|\u2019s)?\s+(?:is\s+)?your\b"
                        r"|\b(?:you|your|u|ur)\s+(?:\w+\s+)?(?:sound|sounds|are|were|think|thinks|"
                        r"feel|feels|name|prefer|like|know|remember|talk|talking|doing|there|"
                        r"ok|okay|alive|real|human|bot|ai)\b", re.I)),
    ("compare", re.compile(r"\b(?:vs\.?|versus|compared? (?:to|with)|difference between)\b"
                           r"|^\s*compare\b|^\s*set\b.+\bagainst\b", re.I)),
    ("howwork", re.compile(r"^\s*(?:explain\s+)?how\s+(?:does|do|did|is|are|was|were)\b"
                           r".*\b(?:work|works|worked|happen|happens|made|formed|produced)\b"
                           r"|^\s*break\s+down\s+how\b", re.I)),
    ("howto", re.compile(r"^\s*how\s+(?:to|do|does|did|can|could|should|would)\b"
                         r"|^\s*walk\s+me\s+through\b", re.I)),
    ("why", re.compile(r"^\s*why\b|^\s*explain\b", re.I)),
    ("whois", re.compile(r"^\s*who(?:'s|’s)?\s+(?:is|are|was|were|exactly)\b"
                         r"|^\s*who\s+(?:is|are|was|were)\b"
                         r"|^\s*tell\s+me\s+who\b"
                         r"|^\s*who(?:'s|’s)\s", re.I)),
    ("whatis", re.compile(r"^\s*(?:what|which)(?:'s|’s|s)\s"
                          r"|^\s*(?:what|which)\s+(?:is|are|was|were)\b"
                          r"|^\s*(?:what|which)\s+(?:does|do|did)\s+(?!(?:you|u|ya|ur|your)\b)", re.I)),
    ("define", re.compile(r"^\s*(?:define|describe|definition of|meaning of|"
                          r"remind me what|give me a rundown|what would you call)\b", re.I)),
    ("tellme", re.compile(r"^\s*(?:tell me about|tell me|talk about|info on|information about|"
                          r"i want to know about)\b", re.I)),
    ("when", re.compile(r"^\s*when\s+(?:is|are|was|were|did|does|do)\b", re.I)),
    ("where", re.compile(r"^\s*where\s+(?:is|are|was|were)\b", re.I)),
    ("list", re.compile(r"^\s*(?:list|show me|give me|name)\b", re.I)),
    ("social", re.compile(r"^\s*(?:what|how)(?:'s|’s)?\s+(?:do|are|is)?\s*(?:you|u|ya)\b"
                          r"|^\s*(?:how|what)(?:'s|’s)?\s+(?:it|things|life)\b", re.I)),
]


def query_form(raw):
    text = str(raw or "").strip()
    stripped = _PREAMBLE.sub("", _FORGET.sub("", text)).strip()
    if stripped:
        text = stripped
    for name, rx in _FORMS:
        if rx.search(text):
            return name
    if re.search(r"\?\s*$", text):
        return "bare_q"
    return "bare"


# --------------------------------------------------------------------------
# subject / relation extraction (shape only -- no entity knowledge)
# --------------------------------------------------------------------------

_LEAD = re.compile(
    r"^\s*(?:what(?:'s|\u2019s|s)?|which|who(?:'s|\u2019s|s)?|whom|whose)\s+"
    r"(?:is|are|was|were)\s+(?:the\s+|an?\s+)?|"
    # "what's X" / "who's X": the contraction *is* the copula, so what follows
    # is the subject.  Without this the phrase never loses its scaffolding and
    # the whole question ends up in the subject slot.
    r"^\s*(?:what|which|who)(?:'s|\u2019s)\s+(?:the\s+|an?\s+)?|"
    r"^\s*(?:what(?:'s|s)?|which)\s+(?:does|do|did)\s+|"
    r"^\s*(?:define|describe)\s+|"
    r"^\s*(?:meaning|definition)\s+of\s+|"
    r"^\s*(?:tell\s+me\s+about|tell\s+me|talk\s+about|explain|info\s+on|information\s+about)\s+|"
    r"^\s*(?:could|can|would)\s+you\s+(?:please\s+)?(?:explain|describe|define|tell\s+me\s+about)\s+",
    re.I)

_TRAIL = re.compile(r"\s+(?:mean|means|meaning|about)\s*$", re.I)

# A topic shift or a greeting is scaffolding in front of a real question:
# "anyway, what's photosynthesis" asks about photosynthesis, not about
# "anyway".  The marker still fires its own signature -- it is stripped from
# the *subject*, never from the evidence that it was said.
_PREAMBLE = re.compile(
    r"^\s*(?:(?:ok(?:ay)?|so|well|right|alright|anyway|anyways|actually|"
    r"but|hey|hi|hello|yo|um|uh|hmm)\b[\s,]*)+", re.I)
_FORGET = re.compile(
    r"^\s*(?:never\s*mind|forget (?:that|it|about that)|"
    r"(?:new|different|unrelated) (?:question|topic|subject)|"
    r"(?:changing|switching) (?:the )?(?:topic|subject))\b[\s,:-]*", re.I)


def split_subject_relation(raw):
    """(subject, relation, parse_form) using grammar only.

    ``capital of France`` -> ("France", "capital", "relation")
    ``France's capital``  -> ("France", "capital", "possessive")
    ``high school``       -> ("high school", "", "bare")
    """
    text = re.sub(r"[?!.]+\s*$", "", str(raw or "").strip())
    text = _FORGET.sub("", text)
    text = _PREAMBLE.sub("", text)
    work = _LEAD.sub("", text)
    work = _TRAIL.sub("", work).strip()
    work = re.sub(r"^(?:the|a|an)\s+", "", work, flags=re.I).strip()
    if not work:
        return "", "", "bare"

    m = re.match(r"^(.+?)['’]s\s+(.+)$", work)
    # An interrogative contraction is a copula, not a possessive: "what's X"
    # asks about X, it does not describe something belonging to "what".
    if m and not re.match(r"^(?:what|which|who|that|there|here|it|he|she|let|lets|how|where|when|why)$",
                          m.group(1).strip(), re.I):
        return _clean(m.group(1)), _clean(m.group(2)), "possessive"

    m = re.match(r"^(.{1,70}?)\s+(?:of|for|about)\s+(.+)$", work, re.I)
    if m:
        return _clean(m.group(2)), _clean(m.group(1)), "relation"

    m = re.match(r"^([a-z][a-z0-9 _+.#-]{1,34})\s+in\s+(.+)$", work, re.I)
    if m:
        return _clean(m.group(2)), _clean(m.group(1)), "relation"

    return _clean(work), "", "bare"


def _clean(v):
    v = re.sub(r"\s+", " ", str(v or "")).strip(" ,;:-–—")
    return re.sub(r"^(?:the|a|an)\s+", "", v, flags=re.I).strip()


# --------------------------------------------------------------------------
# case shape -- "named-looking phrase structure"
# --------------------------------------------------------------------------

def case_shape(raw_subject):
    """lower / title / mixed / upper, from the *original* casing."""
    words = [w for w in re.split(r"\s+", str(raw_subject or "").strip()) if w]
    if not words:
        return "none"
    alpha = [w for w in words if re.search(r"[A-Za-z]", w)]
    if not alpha:
        return "none"
    upper = sum(1 for w in alpha if w.isupper() and len(w) > 1)
    titled = sum(1 for w in alpha if w[:1].isupper() and not w.isupper())
    if upper == len(alpha):
        return "upper"
    if titled == len(alpha):
        return "title"
    if titled or upper:
        return "mixed"
    return "lower"


# --------------------------------------------------------------------------
# the signature itself
# --------------------------------------------------------------------------

def signatures(raw, state=None, site_lexicon=()):
    """Categorical signature tokens for one query.

    ``state`` is the dialogue state dict the browser keeps (``subject``,
    ``entity``, ``turns``); ``site_lexicon`` is the host site's own vocabulary.
    Both are optional -- a signature computed without them is still valid, just
    coarser, which is the graceful-degradation rule applied to features.
    """
    text = str(raw or "").strip()
    subject_raw, relation_raw, parse = split_subject_relation(text)
    subj = tokens(subject_raw)
    rel = tokens(relation_raw)
    all_raw = raw_tokens(text)
    all_tok = tokens(text)

    sig = []
    sig.append("form:" + query_form(text))
    sig.append("parse:" + parse)
    sig.append("len:" + str(min(len(subj), 4)))
    sig.append("qlen:" + str(_bucket(len(all_raw), [1, 2, 4, 7, 12])))
    sig.append("sub:" + str(_bucket(len(subject_raw), [0, 6, 14, 30])))
    sig.append("rel:" + ("none" if not rel else str(min(len(rel), 3))))
    sig.append("case:" + case_shape(_subject_original(text, subject_raw)))
    sig.append("dup:" + ("yes" if len(subj) != len(set(subj)) else "no"))
    sig.append("punct:" + ("q" if text.endswith("?") else "none"))
    if '"' in text or "“" in text:
        sig.append("punct:quoted")
    sig.append("convo:" + ("yes" if _has_any(all_raw, CONVO_MARKERS) else "no"))
    if SOCIAL_PHRASE.search(text):
        sig.append("convo:phrase")
    sig.append("shift:" + ("yes" if _has_any(all_raw, SHIFT_MARKERS) else "no"))
    sig.append("cont:" + ("yes" if (_has_any(all_raw, CONT_MARKERS) and len(all_raw) <= 5) else "no"))
    sig.append("pron:" + ("yes" if (_has_any(all_raw, PRONOUNS) and len(subj) <= 2) else "no"))
    # The site lexicon arrives as phrases ("proof of work"); compare it as
    # tokens, or a multi-word entry can never match a single query token.
    lex = set()
    for w in site_lexicon or ():
        for t in tokens(w):
            lex.add(t)
    sig.append("site:" + ("yes" if (lex and any(t in lex for t in all_tok)) else "no"))

    if state:
        prior = tokens(state.get("subject") or "")
        overlap = len(set(prior) & set(subj)) if prior and subj else 0
        if not prior:
            sig.append("ctx:fresh")
        elif overlap and subj:
            sig.append("ctx:same")
        elif subj:
            sig.append("ctx:new")
        else:
            sig.append("ctx:carry")
        sig.append("depth:" + str(_bucket(int(state.get("turns") or 0), [0, 1, 3, 6])))
    else:
        sig.append("ctx:fresh")
        sig.append("depth:0")

    return tuple(sig)


def _subject_original(text, subject_lower_ok):
    """Recover the subject slice with its original casing.

    ``split_subject_relation`` lowercases nothing, but the leading-scaffold
    regexes are case-insensitive, so the returned subject already carries the
    user's own capitalisation.  This wrapper exists so the intent is explicit
    and so a future change to the splitter cannot silently destroy the ``case:``
    signal, which is the only evidence that ``High School Musical`` was typed as
    a name rather than as three ordinary words.
    """
    return subject_lower_ok if subject_lower_ok else text
