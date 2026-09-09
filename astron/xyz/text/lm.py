"""The text-side conditional model.  Same architecture as GABRIEL, own vocabulary.

``gabriel/lm.py`` is a sparse log-linear autoregressive model over ARC program
tokens.  This is the same mathematics -- sparse feature rows, an exact softmax,
AdaGrad SGD, a held-out gate -- over a vocabulary of *text query signatures* and
*sense-hypothesis families*.  Nothing is imported from ``gabriel``: an ARC
program vocabulary cannot score English and English cannot score ARC programs,
and pretending otherwise would be the fake integration this package exists to
avoid.

Two heads, because the browser asks two different questions.

**Family head** -- ``P(family | query signature)``::

    P(f | h) = softmax_f  sum_{k in F(h)} w[k][f]

    F(h) = { b, s|<sig> for each signature token, c|<sig-a>|<sig-b> for a few
             informative pairs }

The softmax is taken over ``families.grammatical_families(query)`` only.  That
is constrained decoding, and it is what makes the model unable to route "what
is photosynthesis" to ``conversational``: that family is not in the candidate
set, so no weight can select it.

**Candidate head** -- ``P(intended sense | query, candidate)``, a binary
log-linear scorer over conjunctions of query-signature tokens and candidate
signature tokens.  It runs strictly **after** ``families.prune`` has thrown
away everything the identity ladder rejects, so its weights order valid
hypotheses and can never validate an invalid one.

Both heads export to a single JSON artifact the browser loads same-origin.  The
browser implements this exact arithmetic in about sixty lines; ``bench.py
--conformance`` checks the two agree to 1e-9 on a fixed query set.
"""

import json
import math
import os
import random
import time

from . import families as FAM
from . import signatures as S

# Signature pairs worth conjoining.  Chosen because each pair encodes a
# decision the marginals cannot: "a lowercase single token in the *whatis* form"
# is a concept lookup, while "a Title Case three-token phrase in the same form"
# is a named work.
PAIR_GROUPS = (("form", "len"), ("form", "case"), ("form", "convo"),
               ("form", "ctx"), ("case", "len"), ("ctx", "pron"),
               ("shift", "ctx"), ("cont", "ctx"))


def _group(tok):
    return tok.split(":", 1)[0]


def query_features(sigs):
    """Feature keys for the family head."""
    feats = ["b"]
    by_group = {}
    for s in sigs:
        feats.append("s|" + s)
        by_group.setdefault(_group(s), []).append(s)
    for ga, gb in PAIR_GROUPS:
        for a in by_group.get(ga, ()):
            for b in by_group.get(gb, ()):
                feats.append("c|" + a + "|" + b)
    return feats


# --------------------------------------------------------------------------
# candidate signatures
# --------------------------------------------------------------------------

def _bucket(v, edges):
    for i, e in enumerate(edges):
        if v <= e:
            return i
    return len(edges)


def candidate_signature(asked, cand):
    """Categorical description of one retrieved candidate.

    Deliberately about *evidence shape*, never about the topic: "a rank-1
    encyclopedia hit whose title is the asked phrase plus one token, with
    moderate coherence" is the same signature whether the topic is high school
    or jaguars.
    """
    title = str(cand.get("title") or "")
    contained, extras = FAM.ordered_containment(asked, title)
    base, qual = FAM.split_qualifier(title)
    cov = FAM.coverage(asked, (cand.get("text") or "") + " " + title)
    coh = float(cand.get("coherence") or 0.0)
    conc = float(cand.get("concentration") or 0.0)
    sig = [
        "tier:" + FAM.TIER_NAMES.get(int(cand.get("tier", 0)), "fuzzy"),
        "exp:" + ("no" if not contained else str(min(extras, 3))),
        "cov:" + str(_bucket(cov, [0.0, 0.34, 0.67, 0.99])),
        "coh:" + str(_bucket(coh, [0.2, 0.4, 0.6, 0.8])),
        "conc:" + str(_bucket(conc, [0.2, 0.5, 0.8])),
        "src:" + str(cand.get("source") or "unknown"),
        "rank:" + str(_bucket(int(cand.get("rank") or 9), [0, 1, 2, 4])),
        "qual:" + ("yes" if qual else "no"),
        "dis:" + ("yes" if cand.get("disambiguation") else "no"),
        "read:" + ("yes" if cand.get("full_text") else "no"),
        "dup:" + ("yes" if FAM.duplicate_mismatch(asked, title) else "no"),
        "tlen:" + str(_bucket(len(FAM.identity_tokens(base)), [1, 2, 3, 5])),
    ]
    return tuple(sig)


def candidate_features(qsigs, csigs):
    """Conjunctions of query shape and candidate shape."""
    feats = ["b"]
    for c in csigs:
        feats.append("d|" + c)
    qform = [s for s in qsigs if s.startswith(("form:", "len:", "case:", "dup:"))]
    for q in qform:
        for c in csigs:
            feats.append("x|" + q + "|" + c)
    return feats


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

class TextLM:
    """Two sparse log-linear heads sharing one persistence format."""

    def __init__(self, data=None):
        d = data or {}
        self.family_w = {k: dict(v) for k, v in (d.get("family_weights") or {}).items()}
        self.cand_w = {k: float(v) for k, v in (d.get("candidate_weights") or {}).items()}
        self.families = list(d.get("families") or FAM.FAMILIES)
        self.meta = dict(d.get("meta") or {})
        self.provenance = dict(d.get("provenance") or {})

    # -- persistence ---------------------------------------------------
    def to_dict(self, prune=1e-4):
        fw = {}
        for f, row in self.family_w.items():
            r = {t: round(v, 5) for t, v in row.items() if abs(v) >= prune}
            if r:
                fw[f] = r
        cw = {k: round(v, 5) for k, v in self.cand_w.items() if abs(v) >= prune}
        return {
            "kind": "astron-text-lm",
            "version": 1,
            "families": list(self.families),
            "family_weights": fw,
            "candidate_weights": cw,
            "meta": self.meta,
            "provenance": self.provenance,
        }

    def save(self, path):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.to_dict(), fh, sort_keys=True, separators=(",", ":"))
        os.replace(tmp, path)

    @staticmethod
    def load(path):
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                return TextLM(json.load(fh))
        except (OSError, ValueError):
            return None

    def is_trained(self):
        return bool(self.family_w) or bool(self.cand_w)

    # -- family head ---------------------------------------------------
    def family_logits(self, sigs, allowed):
        feats = query_features(sigs)
        acc = dict.fromkeys(allowed, 0.0)
        for f in feats:
            row = self.family_w.get(f)
            if not row:
                continue
            for fam in acc:
                v = row.get(fam)
                if v:
                    acc[fam] += v
        return acc

    @staticmethod
    def softmax(logits):
        if not logits:
            return {}
        peak = max(logits.values())
        exp = {k: math.exp(v - peak) for k, v in logits.items()}
        z = sum(exp.values()) or 1.0
        return {k: v / z for k, v in exp.items()}

    def family_dist(self, raw, state=None, site_lexicon=()):
        """``P(family | query)``, constrained to the grammatical candidates."""
        allowed = FAM.grammatical_families(raw, site_lexicon=site_lexicon, state=state)
        sigs = S.signatures(raw, state=state, site_lexicon=site_lexicon)
        if not self.family_w:
            # Untrained: a uniform prior over the grammar is still a correct,
            # if uninformative, answer.  Degraded, never wrong.
            return {f: 1.0 / len(allowed) for f in allowed}
        return self.softmax(self.family_logits(sigs, allowed))

    # -- candidate head ------------------------------------------------
    def candidate_score(self, qsigs, csigs):
        z = 0.0
        for k in candidate_features(qsigs, csigs):
            v = self.cand_w.get(k)
            if v:
                z += v
        return z

    def candidate_prob(self, qsigs, csigs):
        z = max(-14.0, min(14.0, self.candidate_score(qsigs, csigs)))
        return 1.0 / (1.0 + math.exp(-z))

    # -- training ------------------------------------------------------
    def train_families(self, examples, epochs=14, lr=0.5, l2=2e-6, seed=11,
                       dev=None, patience=3, verbose=False):
        """Cross-entropy SGD with AdaGrad, exact softmax over the grammar.

        The corpus is small enough (a few thousand signature/family pairs) that
        the exact softmax is affordable and the sampled approximation GABRIEL
        needs for a 282-token vocabulary is unnecessary here.  Reporting a real
        loss instead of a sampled one is worth more than the speed.
        """
        if not examples:
            return {"examples": 0, "stopped": "empty"}
        rng = random.Random(seed)
        acc_g = {}
        order = list(range(len(examples)))
        best = (float("inf"), None, 0)
        stale = 0
        t0 = time.time()
        loss_sum, n_seen, stopped = 0.0, 0, None
        for ep in range(epochs):
            rng.shuffle(order)
            loss_sum, n_seen = 0.0, 0
            for ei in order:
                ex = examples[ei]
                allowed = list(ex["allowed"])
                if ex["family"] not in allowed:
                    continue
                feats = query_features(ex["sigs"])
                p = self.softmax(self.family_logits(ex["sigs"], allowed))
                w = float(ex.get("weight", 1.0))
                loss_sum -= w * math.log(max(p.get(ex["family"], 1e-12), 1e-12))
                n_seen += 1
                for fam in allowed:
                    g = w * (p[fam] - (1.0 if fam == ex["family"] else 0.0))
                    if g == 0.0:
                        continue
                    for f in feats:
                        row = self.family_w.setdefault(f, {})
                        ga = acc_g.setdefault(f, {})
                        gg = g + l2 * row.get(fam, 0.0)
                        ga[fam] = ga.get(fam, 0.0) + gg * gg
                        row[fam] = row.get(fam, 0.0) - lr * gg / math.sqrt(ga[fam] + 1e-8)
            dev_loss = self.family_loss(dev) if dev else None
            if verbose:
                print("  fam epoch %d loss/ex=%.4f dev=%s" %
                      (ep + 1, loss_sum / max(1, n_seen),
                       "-" if dev_loss is None else round(dev_loss, 4)), flush=True)
            if dev_loss is not None:
                if dev_loss < best[0] - 1e-9:
                    best = (dev_loss,
                            {f: dict(r) for f, r in self.family_w.items()}, ep + 1)
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        stopped = "early stop"
                        break
        if best[1] is not None:
            self.family_w = best[1]
        return {"examples": len(examples), "epochs_run": ep + 1,
                "loss_per_example": round(loss_sum / max(1, n_seen), 4),
                "best_dev_loss": None if best[1] is None else round(best[0], 4),
                "best_epoch": best[2], "stopped": stopped or "epochs",
                "features": len(self.family_w),
                "seconds": round(time.time() - t0, 2)}

    def family_loss(self, examples):
        if not examples:
            return float("inf")
        total, n = 0.0, 0
        for ex in examples:
            allowed = list(ex["allowed"])
            if ex["family"] not in allowed:
                continue
            p = self.softmax(self.family_logits(ex["sigs"], allowed))
            total -= math.log(max(p.get(ex["family"], 1e-12), 1e-12))
            n += 1
        return total / max(1, n)

    def family_accuracy(self, examples):
        if not examples:
            return 0.0
        hit, n = 0, 0
        for ex in examples:
            allowed = list(ex["allowed"])
            if not allowed:
                continue
            p = self.softmax(self.family_logits(ex["sigs"], allowed))
            pick = max(p.items(), key=lambda kv: (kv[1], kv[0]))[0]
            hit += 1 if pick == ex["family"] else 0
            n += 1
        return hit / float(max(1, n))

    def train_candidates(self, examples, epochs=20, lr=0.4, l2=2e-6, seed=13,
                         dev=None, patience=3, verbose=False):
        """Binary logistic SGD over query x candidate conjunctions."""
        if not examples:
            return {"examples": 0, "stopped": "empty"}
        rng = random.Random(seed)
        acc_g = {}
        order = list(range(len(examples)))
        best = (float("inf"), None, 0)
        stale = 0
        t0 = time.time()
        loss_sum, n_seen, stopped = 0.0, 0, None
        for ep in range(epochs):
            rng.shuffle(order)
            loss_sum, n_seen = 0.0, 0
            for ei in order:
                ex = examples[ei]
                feats = candidate_features(ex["qsigs"], ex["csigs"])
                z = 0.0
                for k in feats:
                    z += self.cand_w.get(k, 0.0)
                z = max(-14.0, min(14.0, z))
                p = 1.0 / (1.0 + math.exp(-z))
                y = 1.0 if ex["label"] else 0.0
                loss_sum -= (y * math.log(max(p, 1e-12)) +
                             (1 - y) * math.log(max(1 - p, 1e-12)))
                n_seen += 1
                g = p - y
                if g == 0.0:
                    continue
                for k in feats:
                    gg = g + l2 * self.cand_w.get(k, 0.0)
                    acc_g[k] = acc_g.get(k, 0.0) + gg * gg
                    self.cand_w[k] = self.cand_w.get(k, 0.0) - \
                        lr * gg / math.sqrt(acc_g[k] + 1e-8)
            dev_loss = self.candidate_loss(dev) if dev else None
            if verbose:
                print("  cand epoch %d loss/ex=%.4f dev=%s" %
                      (ep + 1, loss_sum / max(1, n_seen),
                       "-" if dev_loss is None else round(dev_loss, 4)), flush=True)
            if dev_loss is not None:
                if dev_loss < best[0] - 1e-9:
                    best = (dev_loss, dict(self.cand_w), ep + 1)
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        stopped = "early stop"
                        break
        if best[1] is not None:
            self.cand_w = best[1]
        return {"examples": len(examples), "epochs_run": ep + 1,
                "loss_per_example": round(loss_sum / max(1, n_seen), 4),
                "best_dev_loss": None if best[1] is None else round(best[0], 4),
                "best_epoch": best[2], "stopped": stopped or "epochs",
                "features": len(self.cand_w),
                "seconds": round(time.time() - t0, 2)}

    def candidate_loss(self, examples):
        if not examples:
            return float("inf")
        total = 0.0
        for ex in examples:
            p = self.candidate_prob(ex["qsigs"], ex["csigs"])
            y = 1.0 if ex["label"] else 0.0
            total -= (y * math.log(max(p, 1e-12)) +
                      (1 - y) * math.log(max(1 - p, 1e-12)))
        return total / float(len(examples))


# --------------------------------------------------------------------------
# resolution: propose, verify, then rank.  In that order.
# --------------------------------------------------------------------------

def resolve(lm, raw, candidates, state=None, site_lexicon=()):
    """The full sense-resolution pass, as the browser runs it.

    1. the grammar proposes families;
    2. the model puts a distribution over them (constrained);
    3. the identity ladder prunes candidates -- hard, unlearned;
    4. the learned candidate head orders what survived;
    5. the ordering is stable *within* tier only.

    Returns a plan dict.  It never returns a candidate the ladder rejected, and
    it never promotes one across a tier boundary, whatever the weights say.
    """
    asked, rel, _parse = S.split_subject_relation(raw)
    sigs = S.signatures(raw, state=state, site_lexicon=site_lexicon)
    fam_p = lm.family_dist(raw, state=state, site_lexicon=site_lexicon) if lm else {}
    family = max(fam_p.items(), key=lambda kv: (kv[1], kv[0]))[0] if fam_p else "definition"

    focused = family in ("definition", "identity") and bool(asked)
    kept, best_tier, notes = FAM.prune(asked, candidates, focused=focused)

    ranked = []
    for c in kept:
        csig = candidate_signature(asked, c)
        c = dict(c)
        c["signature"] = list(csig)
        c["score"] = lm.candidate_score(sigs, csig) if lm else 0.0
        c["prob"] = lm.candidate_prob(sigs, csig) if lm else 0.5
        ranked.append(c)
    # tier first, always; the learned score orders inside a tier and nowhere else
    ranked.sort(key=lambda c: (-int(c["tier"]), -float(c["score"])))

    if focused and ranked and int(ranked[0]["tier"]) <= FAM.TIER_EXPANDED:
        family = "clarification_required"
    elif not ranked:
        family = "clarification_required"
    elif ranked[0].get("disambiguation"):
        family = "ambiguous"

    return {
        "raw": raw, "asked": asked, "relation": rel,
        "signatures": list(sigs), "families": fam_p, "family": family,
        "best_tier": best_tier, "candidates": ranked, "notes": notes,
    }
