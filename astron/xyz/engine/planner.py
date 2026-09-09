"""A learned planner over reasoning actions.

The engine is compute bound: on the bundled corpora the median task consumes
three quarters of its budget, so *which* generator gets the next second is a
real decision with a measurable cost.  Until now that decision was a fixed
list.  This module replaces it with a model that is trained on what actually
worked, and that keeps updating as the search reports back.

What the model is
-----------------

A sparse multinomial log-linear model over an action vocabulary.  Actions are
the concrete choices the solver can make -- run a particular hypothesis family,
deepen the enumerator, or stop generating:

    P(a | s) = softmax_a( w_a . phi(s) )

``phi(s)`` is the reasoning state, and it is not only a description of the task.
It carries what has already been tried and what came back:

* task descriptors  -- shape law, palette change, separator lines, symmetry,
  object counts, grid size (the same signatures ``engine/learn.py`` uses);
* search history    -- which families have already run and produced nothing,
  how many hypotheses currently fit, how much budget is left;
* retrieved memory  -- the families that solved the most similar *previously
  solved* tasks, weighted by similarity.

The retrieval block is deliberately a *feature block*, not a shortcut.  Nothing
in this file can answer a task by looking one up; retrieval only shifts the
logits, and the model can learn to distrust it.  The equation is

    logits = W . phi_task(s) + W . phi_history(s) + W . phi_memory(s)
    P      = softmax(logits)

so dataset knowledge reaches the action distribution *through* the model's
parameters rather than around them.

What the planner may and may not decide
---------------------------------------

It chooses order and budget.  It never decides whether a hypothesis is correct:
every program the search returns is still executed against every training pair
and discarded unless it reproduces all of them.  A badly trained planner costs
time; it cannot make the engine accept a wrong answer.
"""

import json
import math
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLANNER_PATH = os.path.join(ROOT, "policy", "arc_planner.json")

# reasoning actions the search engine can actually take
FAMILIES = ("geometry", "colormap", "partition", "symmetry", "tiling", "blocks",
            "select", "regions", "counting", "cellwise", "objects_map",
            "motion", "substitute", "sequence", "paint", "patterns", "analogy",
            "compose", "panelabs", "panelwise", "objwise", "rewrite", "cascade",
            "enumerate_dsl", "objproc", "assemble", "refine", "conditional", "paneltable", "selfstamp", "tally", "extend", "locate", "objchain")
FAMILIES = FAMILIES + ("relproc",)
CONTROL = ("deepen", "stop")
ACTIONS = FAMILIES + CONTROL


def _sig_features(sigs):
    return ["t:" + s for s in sigs]


def _history_features(ran, n_fit, frac_left, step):
    f = ["h:ran:" + m for m in ran]
    f.append("h:nran:%d" % min(len(ran), 12))
    f.append("h:fit:" + ("none" if n_fit == 0 else
                         ("few" if n_fit < 5 else "many")))
    f.append("h:left:" + ("hi" if frac_left > 0.66 else
                          ("mid" if frac_left > 0.33 else "lo")))
    f.append("h:step:%d" % min(step, 10))
    return f


class Memory:
    """Signature -> families that solved tasks with that signature.

    Stores *reasoning outcomes*, never task content and never an answer grid.
    """

    def __init__(self, data=None):
        self.rows = [tuple(r) for r in (data or [])]   # [(sigs, family, task_id)]

    def add(self, sigs, family, task_id=""):
        self.rows.append((tuple(sigs), family, task_id))

    def features(self, sigs, k=12, exclude=None):
        """Retrieved evidence as feature strings.

        ``exclude`` drops the querying task's own contribution, which is what
        keeps a training example from retrieving its own answer.
        """
        if not self.rows:
            return []
        q = set(sigs)
        scored = []
        for row in self.rows:
            s, fam = row[0], row[1]
            tid = row[2] if len(row) > 2 else ""
            if exclude is not None and tid == exclude:
                continue
            inter = len(q & set(s))
            if not inter:
                continue
            jac = inter / float(len(q | set(s)))
            scored.append((jac, fam))
        if not scored:
            return []
        scored.sort(reverse=True)
        top = scored[:k]
        agg = defaultdict(float)
        tot = sum(j for j, _f in top) or 1.0
        for j, fam in top:
            agg[fam] += j / tot
        out = []
        for fam, v in agg.items():
            # three coarse buckets: a real-valued feature in a sparse model is
            # a scaling trap, and the bucket boundaries are what the weights
            # are actually fitted against
            b = "hi" if v > 0.5 else ("mid" if v > 0.2 else "lo")
            out.append("m:%s:%s" % (fam, b))
        return out


class Planner:
    def __init__(self, data=None):
        d = data or {}
        self.w = defaultdict(dict)
        for feat, row in d.get("w", {}).items():
            self.w[feat] = dict(row)
        self.memory = Memory(d.get("memory", []))
        self.meta = dict(d.get("meta", {}))
        self.trained = bool(self.w)

    # -- inference ------------------------------------------------------
    def features(self, sigs, ran=(), n_fit=0, frac_left=1.0, step=0,
                 exclude=None):
        return (_sig_features(sigs)
                + _history_features(tuple(ran), n_fit, frac_left, step)
                + self.memory.features(sigs, exclude=exclude)
                + ["bias"])

    def logits(self, feats):
        z = dict.fromkeys(ACTIONS, 0.0)
        has_relational_weights = False
        for f in feats:
            row = self.w.get(f)
            if not row:
                continue
            for a, v in row.items():
                if a in z:
                    z[a] += v
                    has_relational_weights |= a == "relproc"
        # Existing policies predate this peer-parameterized object algebra.
        # Transfer the object-induction scheduling prior until it is trained.
        if not has_relational_weights:
            z["relproc"] = z["objproc"]
        return z

    def distribution(self, sigs, ran=(), n_fit=0, frac_left=1.0, step=0,
                     available=None):
        feats = self.features(sigs, ran, n_fit, frac_left, step)
        z = self.logits(feats)
        keys = [a for a in ACTIONS if available is None or a in available]
        if not keys:
            return {}
        peak = max(z[a] for a in keys)
        ex = {a: math.exp(z[a] - peak) for a in keys}
        tot = sum(ex.values()) or 1.0
        return {a: v / tot for a, v in ex.items()}

    # -- training -------------------------------------------------------
    def fit(self, examples, epochs=14, lr=0.35, l2=2e-4, seed=11):
        """Multiclass logistic regression by SGD over sparse binary features."""
        rng = random.Random(seed)
        data = list(examples)
        if not data:
            return self
        w = defaultdict(lambda: defaultdict(float))
        for feat, row in self.w.items():
            for a, v in row.items():
                w[feat][a] = v
        for ep in range(epochs):
            rng.shuffle(data)
            rate = lr / (1.0 + 0.6 * ep)
            for feats, target, avail in data:
                keys = [a for a in ACTIONS if a in avail] if avail else list(ACTIONS)
                if target not in keys:
                    continue
                z = {a: 0.0 for a in keys}
                for f in feats:
                    row = w.get(f)
                    if not row:
                        continue
                    for a in keys:
                        v = row.get(a)
                        if v:
                            z[a] += v
                peak = max(z.values())
                ex = {a: math.exp(z[a] - peak) for a in keys}
                tot = sum(ex.values()) or 1.0
                for f in feats:
                    row = w[f]
                    for a in keys:
                        g = ex[a] / tot - (1.0 if a == target else 0.0)
                        row[a] = row[a] * (1.0 - rate * l2) - rate * g
        self.w = defaultdict(dict)
        for f, row in w.items():
            keep = {a: round(v, 5) for a, v in row.items() if abs(v) > 1e-3}
            if keep:
                self.w[f] = keep
        self.trained = True
        return self

    def accuracy(self, examples):
        ok = n = 0
        top3 = 0
        for feats, target, avail in examples:
            keys = [a for a in ACTIONS if a in avail] if avail else list(ACTIONS)
            if target not in keys:
                continue
            z = self.logits(feats)
            order = sorted(keys, key=lambda a: -z[a])
            n += 1
            ok += int(order[0] == target)
            top3 += int(target in order[:3])
        return {"n": n, "top1": round(ok / float(n), 4) if n else 0.0,
                "top3": round(top3 / float(n), 4) if n else 0.0}

    # -- persistence ----------------------------------------------------
    def to_dict(self):
        return {"w": {k: v for k, v in self.w.items()},
                "memory": [[list(r[0]), r[1], r[2] if len(r) > 2 else ""]
                           for r in self.memory.rows],
                "meta": self.meta}

    def save(self, path=PLANNER_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, sort_keys=True)

    @staticmethod
    def load(path=PLANNER_PATH):
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                return Planner(json.load(fh))
        except Exception:
            return None


ACTIVE = None


def activate(planner):
    global ACTIVE
    ACTIVE = planner


def active():
    return ACTIVE
