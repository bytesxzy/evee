"""Conversation: a dialogue-act model plus authored response plans.

The old path retrieved the nearest line from a small canned corpus and returned
it verbatim.  That fails in three separate ways at once: it copies, it has no
notion of what the turn was *doing*, and it cannot use dialogue state.

This module replaces the mechanism, not just the corpus.

    utterance
      -> act features (categorical, topic-blind)
      -> P(act | utterance)                  <- fitted, exportable
      -> response plan for that act          <- authored here, in this file
      -> slot filling from the *user's own* words and the dialogue state
      -> a sentence generated from the plan

The corpus therefore controls **how to speak** -- which act follows which, how
long a reply should be, how often to ask back -- and never **what is true**.  No
harvested reply text is stored, exported, or emitted: :func:`statistics_from_turns`
reduces a harvest to counts, and the only strings a user can ever see are the
ones written below.  That is the anti-copy mechanism by construction rather
than by filter, which is the only version of it that cannot leak.

Factual questions never reach this module.  ``families.grammatical_families``
does not offer ``conversational`` for a ``form:whatis`` query, so OASST-derived
statistics cannot influence the answer to "what year was X born".
"""

import math
import os
import re
import time

# --------------------------------------------------------------------------
# acts
# --------------------------------------------------------------------------

ACTS = (
    "greeting", "farewell", "thanks", "howareyou", "smalltalk",
    "reaction_positive", "reaction_negative", "agreement", "correction",
    "continuation", "meta_identity", "meta_style", "emotional_share",
    "request_chat", "confusion", "acknowledge", "boredom", "opinion_request",
)

_MARKERS = [
    ("greet", r"^\s*(?:hey|hi|hello|yo|sup|howdy|greetings|good (?:morning|afternoon|evening))\b"),
    ("bye", r"\b(?:bye|goodbye|see ya|see you|later|gtg|good night)\b"),
    ("thanks", r"\b(?:thanks|thank you|thx|ty|appreciate it|cheers)\b"),
    ("howru", r"\b(?:how are you|how's it going|how have you been|how you doing|you doing ok|what's up)\b"),
    ("laugh", r"\b(?:lol|lmao|haha+|hehe|rofl)\b"),
    ("praise", r"\b(?:cool|nice|awesome|great|sweet|dope|neat|interesting|wild|impressive)\b"),
    ("neg", r"\b(?:boring|lame|bad|terrible|awful|sucks|hate|annoying|useless)\b"),
    ("agree", r"^\s*(?:yeah|yep|yup|sure|ok|okay|alright|right|true|exactly|fair enough|makes sense|i see|got it)\b"),
    ("disagree", r"^\s*(?:no|nah|nope)\b|\b(?:that's not|not what i meant|you're wrong|incorrect)\b"),
    ("more", r"\b(?:tell me more|go on|continue|keep going|expand|elaborate|what else|and then)\b"),
    ("whatru", r"\b(?:are you (?:an? )?(?:ai|bot|robot|human|real)|what are you|who made you|your name)\b"),
    ("style", r"\b(?:sound|tone|formal|casual|robotic|stiff|weird|talk like)\b"),
    ("feel", r"\b(?:i feel|i'm feeling|im feeling|i had|my day|i'm tired|im tired|i'm sad|feeling)\b"),
    ("chat", r"\b(?:can we talk|let's talk|lets talk|wanna chat|just chat|talk to me)\b"),
    ("confused", r"\b(?:wait what|huh|what\?|i don't get it|confused|makes no sense)\b"),
    ("bored", r"\b(?:bored|nothing to do|boring)\b"),
    ("opinion", r"\b(?:what do you think|your opinion|do you like|do you prefer|favou?rite)\b"),
    ("q", r"\?\s*$"),
    ("first", r"\b(?:i|i'm|im|my|me)\b"),
    ("second", r"\b(?:you|your|u|ur)\b"),
]
_MARKERS = [(n, re.compile(p, re.I)) for n, p in _MARKERS]


def act_features(text, state=None):
    """Categorical features for one social turn.  No topic words appear."""
    s = str(text or "").strip()
    words = re.findall(r"[A-Za-z0-9']+", s)
    feats = ["b"]
    for name, rx in _MARKERS:
        if rx.search(s):
            feats.append("m|" + name)
    n = len(words)
    feats.append("n|" + ("0" if n <= 1 else "1" if n <= 3 else
                         "2" if n <= 7 else "3" if n <= 14 else "4"))
    feats.append("p|" + ("q" if s.endswith("?") else
                         "x" if s.endswith("!") else "."))
    if state:
        feats.append("t|" + ("0" if not state.get("turns") else
                             "1" if int(state["turns"]) <= 2 else "2"))
        if state.get("last_act"):
            feats.append("a|" + str(state["last_act"]))
    pairs = [f for f in feats if f.startswith("m|")]
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            feats.append("c|" + pairs[i] + "|" + pairs[j])
    return feats


class ActModel:
    """Sparse log-linear ``P(act | utterance)``.  Same form as the family head."""

    def __init__(self, data=None):
        d = data or {}
        self.w = {k: dict(v) for k, v in (d.get("weights") or {}).items()}
        self.acts = list(d.get("acts") or ACTS)

    def to_dict(self, prune=1e-4):
        w = {}
        for f, row in self.w.items():
            r = {a: round(v, 5) for a, v in row.items() if abs(v) >= prune}
            if r:
                w[f] = r
        return {"acts": list(self.acts), "weights": w}

    def dist(self, text, state=None):
        feats = act_features(text, state)
        acc = dict.fromkeys(self.acts, 0.0)
        for f in feats:
            row = self.w.get(f)
            if not row:
                continue
            for a in acc:
                v = row.get(a)
                if v:
                    acc[a] += v
        peak = max(acc.values())
        exp = {a: math.exp(v - peak) for a, v in acc.items()}
        z = sum(exp.values()) or 1.0
        return {a: v / z for a, v in exp.items()}

    def predict(self, text, state=None):
        p = self.dist(text, state)
        return max(p.items(), key=lambda kv: (kv[1], kv[0]))

    def train(self, examples, epochs=24, lr=0.5, l2=2e-6, seed=5, dev=None,
              patience=3):
        import random
        if not examples:
            return {"examples": 0}
        rng = random.Random(seed)
        acc_g = {}
        order = list(range(len(examples)))
        best, stale = (float("inf"), None, 0), 0
        loss, n = 0.0, 0
        for ep in range(epochs):
            rng.shuffle(order)
            loss, n = 0.0, 0
            for ei in order:
                text, gold, st = examples[ei]
                feats = act_features(text, st)
                acc = dict.fromkeys(self.acts, 0.0)
                for f in feats:
                    row = self.w.get(f)
                    if not row:
                        continue
                    for a in acc:
                        v = row.get(a)
                        if v:
                            acc[a] += v
                peak = max(acc.values())
                exp = {a: math.exp(v - peak) for a, v in acc.items()}
                z = sum(exp.values()) or 1.0
                p = {a: v / z for a, v in exp.items()}
                loss -= math.log(max(p.get(gold, 1e-12), 1e-12))
                n += 1
                for a in self.acts:
                    g = p[a] - (1.0 if a == gold else 0.0)
                    if g == 0.0:
                        continue
                    for f in feats:
                        row = self.w.setdefault(f, {})
                        ga = acc_g.setdefault(f, {})
                        gg = g + l2 * row.get(a, 0.0)
                        ga[a] = ga.get(a, 0.0) + gg * gg
                        row[a] = row.get(a, 0.0) - lr * gg / math.sqrt(ga[a] + 1e-8)
            dl = self.loss(dev) if dev else None
            if dl is not None:
                if dl < best[0] - 1e-9:
                    best = (dl, {f: dict(r) for f, r in self.w.items()}, ep + 1)
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
        if best[1] is not None:
            self.w = best[1]
        return {"examples": len(examples), "loss_per_example": round(loss / max(1, n), 4),
                "best_dev_loss": None if best[1] is None else round(best[0], 4),
                "best_epoch": best[2], "features": len(self.w)}

    def loss(self, examples):
        if not examples:
            return float("inf")
        total = 0.0
        for text, gold, st in examples:
            p = self.dist(text, st)
            total -= math.log(max(p.get(gold, 1e-12), 1e-12))
        return total / float(len(examples))

    def accuracy(self, examples):
        if not examples:
            return 0.0
        hit = sum(1 for t, g, s in examples if self.predict(t, s)[0] == g)
        return hit / float(len(examples))


# --------------------------------------------------------------------------
# response plans.  Every string a user can see is written here.
# --------------------------------------------------------------------------
#
# A plan is (skeleton, needs) where `needs` names the slots the skeleton uses:
#   {topic}   the current dialogue subject, when there is one
#   {echo}    a short quotation of the user's own wording
# A skeleton whose slots cannot be filled is skipped, so a plan can never
# render with an empty hole in it.

PLANS = {
    "greeting": [
        ("Hey. What do you want to get into?", []),
        ("Hi there. What's on your mind?", []),
        ("Hey — ask me anything, or just talk.", []),
        ("Good to see you. Where do you want to start?", []),
    ],
    "farewell": [
        ("Catch you later.", []),
        ("Alright — I'll be here when you come back.", []),
        ("See you. Come back with a harder question.", []),
    ],
    "thanks": [
        ("Any time.", []),
        ("Happy to. What else?", []),
        ("No problem — want to keep going on {topic}?", ["topic"]),
        ("Sure thing.", []),
    ],
    "howareyou": [
        ("Running fine. What are you working on?", []),
        ("Good — no complaints. What about you?", []),
        ("All good here. What brought you over?", []),
    ],
    "smalltalk": [
        ("Fair enough. Where do you want to take it?", []),
        ("Right. Anything you want to dig into?", []),
        ("Got it. What's next?", []),
    ],
    "reaction_positive": [
        ("Right? There's more to it if you want it.", []),
        ("Glad that landed. Want the next layer of {topic}?", ["topic"]),
        ("Yeah, it's a good one.", []),
        ("It is. Want me to keep pulling on that thread?", []),
    ],
    "reaction_negative": [
        ("Fair. Want me to come at it a different way?", []),
        ("Understood — tell me what would actually help.", []),
        ("That's useful to know. What were you after instead?", []),
    ],
    "agreement": [
        ("Good. What next?", []),
        ("Alright. Want to go deeper on {topic}?", ["topic"]),
        ("Cool — carry on whenever.", []),
    ],
    "correction": [
        ("My mistake — say it again the way you meant it and I'll redo it.", []),
        ("Got it, I read that wrong. What did you actually want?", []),
        ("Fair — rephrase it and I'll take another pass.", []),
    ],
    "continuation": [
        ("More on {topic}, coming up — ask me the specific part you want.", ["topic"]),
        ("Sure. Which part of {topic} do you want expanded?", ["topic"]),
        ("Happy to keep going — what angle?", []),
    ],
    "meta_identity": [
        ("I'm the assistant built into this page. Everything I do runs in "
         "your browser — no server, no outside model.", []),
        ("A local model on this page, plus a few public databases I look "
         "things up in. That's the whole stack.", []),
    ],
    "meta_style": [
        ("Noted — I'll keep it looser.", []),
        ("Fair. I'll drop the formality.", []),
        ("Point taken. Plainer from here.", []),
    ],
    "emotional_share": [
        ("That sounds like a lot. What happened?", []),
        ("I'm listening — go on.", []),
        ("Rough one. Want to talk it through or get distracted?", []),
    ],
    "request_chat": [
        ("Yeah, go ahead. What's on your mind?", []),
        ("Sure. Pick a subject and I'll follow.", []),
    ],
    "confusion": [
        ("Let me try that again — which part lost you?", []),
        ("Fair, that was unclear. Ask me the narrower version.", []),
    ],
    "acknowledge": [
        ("Understood.", []),
        ("Right.", []),
        ("Noted.", []),
    ],
    "boredom": [
        ("Then give me something to chew on — a topic, a project, anything.", []),
        ("We can fix that. Pick a subject and I'll dig.", []),
    ],
    "opinion_request": [
        ("I don't have preferences, but I can lay out the trade-offs — "
         "on what?", []),
        ("I'd rather give you the evidence than an opinion. What's the "
         "subject?", []),
        ("No taste to speak of. Ask me about {topic} and I'll give you the "
         "substance.", ["topic"]),
    ],
}

FALLBACK = "I'm here. What's on your mind?"


def render(act, state=None, seed=0, avoid=None):
    """Pick and fill a plan.  Returns ``(text, plan_index)``.

    Deterministic given ``seed`` so a benchmark is reproducible, and it refuses
    to repeat the plan it used last turn.
    """
    plans = PLANS.get(act) or PLANS.get("smalltalk") or []
    topic = ((state or {}).get("subject") or "").strip()
    usable = []
    for i, (skel, needs) in enumerate(plans):
        if "topic" in needs and not topic:
            continue
        usable.append((i, skel))
    if not usable:
        return FALLBACK, -1
    start = seed % len(usable)
    for k in range(len(usable)):
        i, skel = usable[(start + k) % len(usable)]
        if avoid is not None and i == avoid and len(usable) > 1:
            continue
        return skel.replace("{topic}", topic), i
    i, skel = usable[start]
    return skel.replace("{topic}", topic), i


# --------------------------------------------------------------------------
# seed corpus for the act model (labels, not replies)
# --------------------------------------------------------------------------

SEED = [
    ("hey", "greeting"), ("hi", "greeting"), ("hello", "greeting"),
    ("yo", "greeting"), ("sup", "greeting"), ("hey there", "greeting"),
    ("good morning", "greeting"), ("good evening", "greeting"),
    ("bye", "farewell"), ("goodbye", "farewell"), ("see you later", "farewell"),
    ("gtg", "farewell"), ("good night", "farewell"),
    ("thanks", "thanks"), ("thank you", "thanks"), ("ty", "thanks"),
    ("appreciate it", "thanks"), ("cheers", "thanks"), ("thanks a lot", "thanks"),
    ("how are you", "howareyou"), ("how's it going", "howareyou"),
    ("how have you been", "howareyou"), ("you doing ok", "howareyou"),
    ("what's up", "howareyou"),
    ("that's actually pretty cool", "reaction_positive"),
    ("nice", "reaction_positive"), ("cool", "reaction_positive"),
    ("that's wild", "reaction_positive"), ("interesting", "reaction_positive"),
    ("awesome", "reaction_positive"), ("lol", "reaction_positive"),
    ("haha", "reaction_positive"), ("lmao", "reaction_positive"),
    ("that's boring", "reaction_negative"), ("that sucks", "reaction_negative"),
    ("this is annoying", "reaction_negative"), ("useless", "reaction_negative"),
    ("yeah", "agreement"), ("yep", "agreement"), ("sure", "agreement"),
    ("ok", "agreement"), ("okay", "agreement"), ("alright", "agreement"),
    ("makes sense", "agreement"), ("fair enough", "agreement"),
    ("got it", "agreement"), ("i see", "agreement"), ("true", "agreement"),
    ("that's not what i meant", "correction"), ("no i meant something else", "correction"),
    ("nah", "correction"), ("nope", "correction"), ("you're wrong", "correction"),
    ("that's incorrect", "correction"),
    ("tell me more", "continuation"), ("continue", "continuation"),
    ("go on", "continuation"), ("keep going", "continuation"),
    ("expand on that", "continuation"), ("what else", "continuation"),
    ("elaborate", "continuation"),
    ("are you an ai", "meta_identity"), ("what are you", "meta_identity"),
    ("who made you", "meta_identity"), ("are you a bot", "meta_identity"),
    ("what's your name", "meta_identity"), ("are you real", "meta_identity"),
    ("why do you sound so formal", "meta_style"),
    ("you sound like a robot", "meta_style"),
    ("can you talk normally", "meta_style"),
    ("your tone is weird", "meta_style"),
    ("i had a weird day", "emotional_share"),
    ("i'm tired", "emotional_share"), ("i feel weird today", "emotional_share"),
    ("my day was rough", "emotional_share"), ("i'm stressed", "emotional_share"),
    ("can we talk", "request_chat"), ("let's talk", "request_chat"),
    ("talk to me", "request_chat"), ("wanna chat", "request_chat"),
    ("wait what", "confusion"), ("huh", "confusion"),
    ("i don't get it", "confusion"), ("that makes no sense", "confusion"),
    ("noted", "acknowledge"), ("right", "acknowledge"), ("mhm", "acknowledge"),
    ("i'm bored", "boredom"), ("nothing to do", "boredom"), ("bored", "boredom"),
    ("what do you think about that", "opinion_request"),
    ("do you like it", "opinion_request"),
    ("what's your favorite thing", "opinion_request"),
    ("do you prefer one", "opinion_request"),
]

SEED_HOLDOUT = [
    ("heya", "greeting"), ("catch you later", "farewell"),
    ("much appreciated", "thanks"), ("how you holding up", "howareyou"),
    ("that's dope", "reaction_positive"), ("this is lame", "reaction_negative"),
    ("exactly", "agreement"), ("no that's off", "correction"),
    ("and then", "continuation"), ("are you human", "meta_identity"),
    ("you talk so stiff", "meta_style"), ("i'm feeling off", "emotional_share"),
    ("just chat with me", "request_chat"), ("confused", "confusion"),
    ("nothing to do today", "boredom"), ("your opinion", "opinion_request"),
]


def train_act_model(seed=SEED, holdout=SEED_HOLDOUT, extra=()):
    """Fit the act model.  ``extra`` is (text, act, state) from a harvest."""
    rows = [(t, a, None) for t, a in seed] + list(extra)
    dev = [(t, a, None) for t, a in holdout]
    m = ActModel()
    stats = m.train(rows, dev=dev)
    stats["holdout_accuracy"] = round(m.accuracy(dev), 4)
    stats["train_accuracy"] = round(m.accuracy(rows), 4)
    return m, stats


# --------------------------------------------------------------------------
# harvest -> statistics.  Counts only; no text leaves this function.
# --------------------------------------------------------------------------

def statistics_from_turns(turns):
    """Reduce harvested (prompt, reply) turns to distributions.

    Every value returned is a count, a rate, or a bucket index.  There is no
    code path by which a harvested string reaches the return value, which is
    what makes it safe to ship the result to a browser under any licence and
    impossible for the assistant to quote its training data.
    """
    model, _ = train_act_model()
    trans = {}
    length_by_act = {}
    question_rate = {}
    sentences_by_act = {}
    hedge_rate = {}
    first_person = {}
    n = 0
    hedges = re.compile(
        r"\b(?:might|maybe|perhaps|probably|generally|typically|often|"
        r"i think|it depends|in general|usually)\b", re.I)
    for t in turns:
        prompt, reply = str(t.get("prompt") or ""), str(t.get("reply") or "")
        if not prompt or not reply:
            continue
        n += 1
        pact = model.predict(prompt)[0]
        ract = model.predict(reply)[0]
        key = pact + ">" + ract
        trans[key] = trans.get(key, 0) + 1
        words = len(re.findall(r"[A-Za-z0-9']+", reply))
        b = ("0" if words <= 8 else "1" if words <= 20 else
             "2" if words <= 45 else "3" if words <= 100 else "4")
        length_by_act.setdefault(pact, {})
        length_by_act[pact][b] = length_by_act[pact].get(b, 0) + 1
        sents = len([s for s in re.split(r"[.!?]+", reply) if s.strip()])
        sb = str(min(sents, 5))
        sentences_by_act.setdefault(pact, {})
        sentences_by_act[pact][sb] = sentences_by_act[pact].get(sb, 0) + 1
        question_rate[pact] = question_rate.get(pact, 0) + (1 if "?" in reply else 0)
        hedge_rate[pact] = hedge_rate.get(pact, 0) + (1 if hedges.search(reply) else 0)
        first_person[pact] = first_person.get(pact, 0) + \
            (1 if re.search(r"\bI\b", reply) else 0)
    totals = {}
    for t in turns:
        p = str(t.get("prompt") or "")
        if p:
            totals[model.predict(p)[0]] = totals.get(model.predict(p)[0], 0) + 1
    return {
        "turns": n,
        "act_transitions": trans,
        "reply_length_buckets": length_by_act,
        "reply_sentence_counts": sentences_by_act,
        "question_back_rate": {k: round(v / float(max(1, totals.get(k, 1))), 4)
                               for k, v in question_rate.items()},
        "hedge_rate": {k: round(v / float(max(1, totals.get(k, 1))), 4)
                       for k, v in hedge_rate.items()},
        "first_person_rate": {k: round(v / float(max(1, totals.get(k, 1))), 4)
                              for k, v in first_person.items()},
        "act_totals": totals,
        "built": int(time.time()),
    }


def load_statistics(path):
    if not path or not os.path.exists(path):
        return None
    try:
        import json
        with open(path) as fh:
            d = json.load(fh)
        return d.get("statistics") if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None
