"""A language model over the engine's program language.

Trained here, from scratch, in pure Python, on the programs this engine has
already written.  No external model is consulted at any point, and nothing in
this file opens a socket.

Model
-----

A sparse log-linear (maximum-entropy) autoregressive language model::

    P(t | h) = softmax_t  sum_{f in F(h)} w[f][t]

``F(h)`` is the set of features active in the history ``h``:

    b                bias -- the unigram distribution
    1|x              the previous token
    2|x|y            the previous two tokens
    s|sig            one per task signature in the prompt (all active at once)
    f|family         the solver family, once the model has emitted it
    p|n              position bucket

The prompt is the task's signatures, so the ``s|...`` features are what make
the model conditional on *this task* rather than a generic program prior.  The
first token the model emits is the solver family, which is why
``P(family | task)`` is something the model produces directly rather than a
second head bolted onto the side.

Training is stochastic gradient descent on cross-entropy with AdaGrad, using a
sampled softmax (negatives drawn from the unigram distribution raised to 0.75)
so that a training pass fits inside a cron tick.  Evaluation uses the exact
softmax, so a reported perplexity is real perplexity and not the sampled
approximation of it.

Decoding is *constrained*: :meth:`beam_chains` only scores continuations that
correspond to operators the engine actually has for the task in hand, so every
sequence the model produces is a well-formed program the enumerator can build.
The engine then checks it against the training pairs like any other hypothesis.
The model is a proposal distribution; it is never evidence.
"""

import bisect
import json
import math
import os
import random
import time

from . import tokens as T

POS_BUCKETS = 6


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def sig_features(sigs):
    return ["s|" + s for s in sigs]


def features(sig_feats, hist, pos, family):
    """Active feature keys for the next-token decision."""
    f = list(sig_feats)
    f.append("b")
    f.append("p|%d" % min(pos, POS_BUCKETS))
    f.append("1|" + (hist[-1] if hist else T.BOS))
    if len(hist) >= 2:
        f.append("2|" + hist[-2] + "|" + hist[-1])
    if family:
        f.append("f|" + family)
    return f


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class GabrielLM:
    """Sparse log-linear language model over program tokens."""

    def __init__(self, data=None):
        d = data or {}
        self.W = {k: dict(v) for k, v in d.get("weights", {}).items()}
        self.vocab = list(d.get("vocab", []))
        self.unigram = dict(d.get("unigram", {}))
        self.meta = dict(d.get("meta", {}))
        self._neg_toks, self._neg_cum = [], []
        self._rebuild_sampler()

    # -- persistence ---------------------------------------------------
    def to_dict(self, prune=1e-4):
        w = {}
        for f, row in self.W.items():
            r = {t: round(v, 5) for t, v in row.items() if abs(v) >= prune}
            if r:
                w[f] = r
        return {"weights": w, "vocab": self.vocab, "unigram": self.unigram,
                "meta": self.meta}

    def save(self, path):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.to_dict(), fh, sort_keys=True)
        os.replace(tmp, path)

    @staticmethod
    def load(path):
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                return GabrielLM(json.load(fh))
        except (OSError, ValueError):
            return None

    def is_trained(self):
        return bool(self.vocab) and bool(self.W)

    # -- scoring -------------------------------------------------------
    def _logits(self, feats, cands):
        """Unnormalised scores for ``cands`` under the active features.

        A candidate outside the vocabulary is scored as ``<unk>`` and pays
        ``log n`` for the ``n`` unknown candidates sharing that mass.  Scoring
        it as a bare zero instead would let any operator the model has never
        seen outrank every operator it has learned to distrust.
        """
        # `unigram` is what this model knows; `vocab` is the support it is
        # being normalised over.  They are the same after training and differ
        # only when one model is evaluated on another's vocabulary -- which is
        # precisely when the distinction matters.
        known = self.unigram
        keys = {c: (c if c in known else T.UNK) for c in cands}
        n_oov = sum(1 for c in cands if c not in known)
        penalty = math.log(n_oov) if n_oov > 1 else 0.0
        acc = dict.fromkeys(set(keys.values()), 0.0)
        get = self.W.get
        for f in feats:
            row = get(f)
            if not row:
                continue
            for k in acc:
                v = row.get(k)
                if v:
                    acc[k] += v
        return {c: acc[keys[c]] - (penalty if c not in known else 0.0)
                for c in cands}

    @staticmethod
    def _softmax(logits):
        if not logits:
            return {}
        peak = max(logits.values())
        exp = {t: math.exp(v - peak) for t, v in logits.items()}
        z = sum(exp.values()) or 1.0
        return {t: v / z for t, v in exp.items()}

    def dist(self, feats, cands):
        """Normalised distribution over a candidate set (constrained decode)."""
        return self._softmax(self._logits(feats, cands))

    def _walk(self, sigs, body):
        """Exact per-token log-probabilities along one sequence."""
        sf = sig_features(sigs)
        hist, fam, out = [], None, []
        for pos, tok in enumerate(body):
            p = self.dist(features(sf, hist, pos, fam), self.vocab)
            key = tok if tok in p else T.UNK
            out.append(math.log(max(p.get(key, 1e-12), 1e-12)))
            if T.is_fam(tok):
                fam = T.fam_name(tok)
            hist.append(key)
        return out

    def logprob(self, sigs, family, program):
        """Exact log P(program | task signatures), full-vocabulary softmax."""
        if not self.is_trained():
            return 0.0
        body = ([T.fam_token(family)] if family else []) + \
            T.tokenize(program) + [T.EOS]
        return sum(self._walk(sigs, body))

    def perplexity(self, examples):
        """Token-level perplexity on held-out examples (exact softmax)."""
        n, total = 0, 0.0
        for ex in examples:
            lps = self._walk(ex["sigs"], ex["body"])
            total -= sum(lps)
            n += len(lps)
        return math.exp(total / n) if n else float("inf")

    # -- strapping point 1: which family to believe ---------------------
    def family_dist(self, sigs):
        """``P(solver family | task signature)`` -- position 0 of the model."""
        if not self.is_trained():
            return {}
        fams = [t for t in self.vocab if T.is_fam(t)]
        if not fams:
            return {}
        p = self.dist(features(sig_features(sigs), [], 0, None), fams)
        return {T.fam_name(t): v for t, v in p.items()}

    # -- strapping point 2: which operator to try next -------------------
    def op_mass(self, sigs, op_names, depth=2, beam=4):
        """``operator -> probability mass`` the model puts on it for this task.

        The walk is constrained to the operator grammar, so the distribution is
        over operators the engine has, not over raw tokens -- marginalising raw
        tokens would mostly measure how often ``(`` follows ``crop``.  At each
        slot the model's distribution over operator heads is read in full (so
        every operator gets a probability, not just the beam), and each head's
        mass is split among the operators sharing it by walking the trie.
        """
        if not self.is_trained() or not op_names:
            return {}
        sf = sig_features(sigs)
        trie, heads = _op_trie(op_names)
        cands = sorted(set(heads) | {"$"})
        fams = sorted(self.family_dist(sigs).items(), key=lambda kv: -kv[1])[:3]
        states = [([T.fam_token(f)], f, p) for f, p in fams] or [([], None, 1.0)]
        states = [(h, f, w) for h, f, w in states
                  if not h or h[0] in self.unigram]
        mass = {}
        for _ in range(max(1, depth)):
            nxt = []
            for hist, fam, w in states:
                if w < 1e-6:
                    continue
                p = self.dist(features(sf, hist, len(hist), fam), cands)
                for head, ph in p.items():
                    if head == "$" or ph * w < 1e-7:
                        continue
                    node = trie.get(head)
                    if node is None:
                        continue
                    done = []
                    self._op_completions(sf, hist + [head], node, fam, 1.0, done)
                    for op, po in done:
                        mass[op] = mass.get(op, 0.0) + w * ph * po
                for head, ph in sorted(p.items(), key=lambda kv: -kv[1])[:beam]:
                    if head == "$":
                        continue
                    nxt.append((hist + [head, "("], fam, w * ph))
            states = nxt
            if not states:
                break
        return mass

    def _op_completions(self, sf, hist, node, family, prob, out, budget=6):
        """Split a head token's mass among the operators that start with it."""
        name, kids = node
        cands = sorted(kids)
        if name is not None:
            cands = cands + ["("]
        if not cands or budget <= 0:
            return
        p = self.dist(features(sf, hist, len(hist), family), cands)
        for tok, pv in p.items():
            if pv * prob < 1e-7:
                continue
            if tok == "(":
                out.append((name, prob * pv))
            else:
                self._op_completions(sf, hist + [tok], kids[tok], family,
                                     prob * pv, out, budget - 1)

    def op_bias(self, sigs, op_names, scale=1.2):
        """Search-order bonus per operator, in the scale ``enum_core`` expects.

        ``enum_core.search`` subtracts this from an operator's cost when it
        orders the library, and adds it to a node's beam bonus.  Bonuses are
        measured against the median operator and squashed into ``[0, scale]``,
        so a confident model can promote what it believes in but can never
        silence anything: the floor is zero, not a penalty.
        """
        mass = self.op_mass(sigs, op_names)
        if not mass:
            return {}
        vals = sorted(mass.values())
        ref = vals[len(vals) // 2] or 1e-9
        raw = {op: math.log(v / ref) for op, v in mass.items() if v > 0.0}
        hi = max(raw.values()) if raw else 0.0
        if hi <= 0.0:
            return {}
        return {op: round(scale * v / hi, 4)
                for op, v in raw.items() if v > 0.0}

    # -- strapping point 3: which programs to write ----------------------
    def beam_chains(self, sigs, allowed, max_len=6, beam=10, limit=48,
                    family="enumerate", deadline=None):
        """Decode operator chains under a grammar of operators that exist here.

        ``allowed`` is the set of operator names available for this task.  At
        every slot the model chooses between continuing with one of those
        operators and closing the chain with ``$``; nothing else can be
        emitted, so every result is a program the enumerator can build.

        Returns ``[(ops_in_application_order, logprob), ...]``, best first.
        The score is the model's log-probability of the whole emitted sequence
        -- not a per-token mean, which would reward length.  The caller adds
        the usual description-length cost on top, so a long chain still has to
        earn its extra operators.  Application order is the reverse of emission
        order, because ``crop(rot90($))`` names the outermost operator first.
        """
        if not self.is_trained() or not allowed:
            return []
        sf = sig_features(sigs)
        trie, heads = _op_trie(allowed)
        cands0 = sorted(set(heads) | {"$"})
        famtok = T.fam_token(family)
        beams = [([famtok] if famtok in self.vocab else [], [], 0.0)]
        done = []
        for _depth in range(max_len):
            if deadline is not None and time.time() > deadline:
                break
            nxt = []
            for hist, ops, lp in beams:
                p = self.dist(features(sf, hist, len(hist), family), cands0)
                for tok, pv in sorted(p.items(), key=lambda kv: -kv[1])[:beam]:
                    step = lp + math.log(max(pv, 1e-12))
                    if tok == "$":
                        if ops:
                            done.append((list(reversed(ops)), step))
                        continue
                    got = self._extend_op(sf, hist, tok, trie, family, step)
                    if got is not None:
                        h2, opname, lp2 = got
                        nxt.append((h2, ops + [opname], lp2))
            if not nxt:
                break
            nxt.sort(key=lambda r: -r[2])
            beams = nxt[:beam]
        done.sort(key=lambda r: -r[1])
        seen, out = set(), []
        for ops, score in done:
            key = tuple(ops)
            if key in seen:
                continue
            seen.add(key)
            out.append((ops, score))
            if len(out) >= limit:
                break
        return out

    def _extend_op(self, sf, hist, head, trie, family, lp):
        """Complete one operator name from its head token, staying in the trie.

        Returns ``(history_after_open_paren, op_name, logprob)``.  At a node
        that is both a complete name and a prefix of longer ones -- ``crop``
        and ``crop#3`` -- the model decides between ``(`` and the argument
        tokens, which is exactly the choice the token language presents.
        """
        node = trie.get(head)
        if node is None:
            return None
        hist = hist + [head]
        for _ in range(8):
            name, kids = node[0], node[1]
            cands = sorted(kids)
            if name is not None:
                cands.append("(")
            if not cands:
                return None
            p = self.dist(features(sf, hist, len(hist), family), cands)
            tok = max(p.items(), key=lambda kv: (kv[1], kv[0]))[0]
            lp += math.log(max(p[tok], 1e-12))
            hist = hist + [tok]
            if tok == "(":
                return hist, name, lp
            node = kids[tok]
        return None

    # -- training ------------------------------------------------------
    def _rebuild_sampler(self):
        self._neg_toks, self._neg_cum, acc = [], [], 0.0
        for t, c in sorted(self.unigram.items()):
            acc += max(float(c), 1.0) ** 0.75
            self._neg_toks.append(t)
            self._neg_cum.append(acc)

    def _sample_neg(self, rng, k, avoid):
        if not self._neg_cum:
            return set()
        total = self._neg_cum[-1]
        out = set()
        for _ in range(k * 3):
            if len(out) >= k:
                break
            i = bisect.bisect_left(self._neg_cum, rng.random() * total)
            if i >= len(self._neg_toks):
                continue
            t = self._neg_toks[i]
            if t != avoid:
                out.add(t)
        return out

    def build_vocab(self, examples, max_vocab=6000):
        counts = {}
        for ex in examples:
            w = ex.get("weight", 1)
            for t in ex["body"]:
                counts[t] = counts.get(t, 0) + w
        for t in (T.UNK, T.EOS, "$", "(", ")"):
            counts[t] = counts.get(t, 0) + 1
        keep = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_vocab]
        body = {t: float(c) for t, c in keep if not t.startswith(T.SIG)}
        self.vocab = sorted(body)
        self.unigram = body
        self._rebuild_sampler()

    def train(self, examples, epochs=6, lr=0.35, l2=1e-6, negatives=12,
              seed=17, max_seconds=None, verbose=False, dev=None, patience=2):
        """SGD with AdaGrad and a sampled softmax.  Returns training stats.

        With a ``dev`` slice this early-stops: perplexity is measured after
        every epoch, the best epoch's weights are kept, and training gives up
        after ``patience`` epochs without an improvement.  Training loss on a
        corpus this size keeps falling long after the model has stopped
        learning anything transferable, so the epoch count is a decision to be
        measured, not a constant to be guessed.
        """
        if not examples:
            return {"tokens": 0, "loss_per_token": 0.0, "seconds": 0.0,
                    "stopped": "empty", "features": len(self.W),
                    "vocab": len(self.vocab)}
        if not self.vocab:
            self.build_vocab(examples)
        rng = random.Random(seed)
        grad_acc = {}
        t0 = time.time()
        seen_tokens, loss_sum, stopped = 0, 0.0, None
        order = list(range(len(examples)))
        best = (float("inf"), None, 0)      # (dev ppl, weights, epoch)
        stale = 0
        for ep in range(epochs):
            rng.shuffle(order)
            for n, ei in enumerate(order):
                if max_seconds is not None and not (n & 15) and \
                        time.time() - t0 > max_seconds:
                    stopped = "time"
                    break
                ex = examples[ei]
                w = float(ex.get("weight", 1.0))
                sf = sig_features(ex["sigs"])
                hist, fam = [], None
                for pos, tok in enumerate(ex["body"]):
                    target = tok if tok in self.unigram else T.UNK
                    feats = features(sf, hist, pos, fam)
                    cands = [target] + sorted(
                        self._sample_neg(rng, negatives, target))
                    p = self._softmax(self._logits(feats, cands))
                    loss_sum -= w * math.log(max(p.get(target, 1e-12), 1e-12))
                    seen_tokens += 1
                    for c in cands:
                        g = w * (p[c] - (1.0 if c == target else 0.0))
                        if g == 0.0:
                            continue
                        for f in feats:
                            row = self.W.setdefault(f, {})
                            acc = grad_acc.setdefault(f, {})
                            gg = g + l2 * row.get(c, 0.0)
                            acc[c] = acc.get(c, 0.0) + gg * gg
                            row[c] = row.get(c, 0.0) - \
                                lr * gg / math.sqrt(acc[c] + 1e-8)
                    if T.is_fam(tok):
                        fam = T.fam_name(tok)
                    hist.append(target)
            dev_ppl = self.perplexity(dev) if dev else None
            if verbose:
                print("  epoch %d  tokens=%d  loss/tok=%.4f  dev_ppl=%s  %.1fs" %
                      (ep + 1, seen_tokens, loss_sum / max(1, seen_tokens),
                       "-" if dev_ppl is None else round(dev_ppl, 3),
                       time.time() - t0), flush=True)
            if dev_ppl is not None:
                if dev_ppl < best[0] - 1e-9:
                    best = (dev_ppl, {f: dict(r) for f, r in self.W.items()},
                            ep + 1)
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        stopped = stopped or "early stop"
            if stopped:
                break
        if best[1] is not None and best[2] != epochs:
            self.W = best[1]                # keep the epoch that generalised
        self.meta["steps"] = int(self.meta.get("steps", 0)) + seen_tokens
        self.meta["features"] = len(self.W)
        self.meta["vocab"] = len(self.vocab)
        return {"tokens": seen_tokens,
                "loss_per_token": round(loss_sum / max(1, seen_tokens), 4),
                "seconds": round(time.time() - t0, 1),
                "stopped": stopped or "epochs",
                "best_epoch": best[2] or epochs,
                "best_dev_perplexity": (None if best[1] is None
                                        else round(best[0], 4)),
                "features": len(self.W), "vocab": len(self.vocab)}


# --------------------------------------------------------------------------
# operator trie: what keeps constrained decoding well-formed
# --------------------------------------------------------------------------

def _op_trie(names):
    """Prefix trie over tokenised operator names.

    A node is ``[complete_name_or_None, {token: child}]``, so decoding can ask
    "which tokens may legally follow" at every slot.
    """
    root, heads = {}, set()
    for name in names:
        toks = T.op_tokens(name)
        if not toks:
            continue
        heads.add(toks[0])
        node = root.setdefault(toks[0], [None, {}])
        for t in toks[1:]:
            node = node[1].setdefault(t, [None, {}])
        node[0] = name
    return root, heads
