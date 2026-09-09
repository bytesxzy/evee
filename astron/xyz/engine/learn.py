"""Self-improvement: the engine learns from the tasks it has already solved.

Three mechanisms, all offline, all local -- no external model is consulted, and
nothing here ever sees a test output.

1. **Conditioned solver priors.**  Each task is reduced to a handful of
   categorical signatures (does the shape change? does the palette grow? is
   there a separator?).  From the record of which solver family produced the
   accepted program, we fit a bias per (signature, family).  At solve time the
   biases of the task's own signatures are summed into the ranking prior, so
   the family that historically explains tasks *like this one* is believed
   first.  This changes both ranking and, through module ordering, where the
   time goes.

2. **Library learning (abstraction mining).**  Accepted enumerator programs are
   parsed back into operator chains; frequent contiguous sub-chains are
   compiled into single named operators and installed into the DSL.  A depth-4
   search over a library containing 3-op abstractions reaches 12-op programs.
   This is the mechanism that makes the search space grow with experience
   rather than stay fixed.

3. **Operator bias.**  Operators that appear in accepted programs are ordered
   earlier in enumeration, which matters because the enumerator is truncated
   by a state cap: what gets explored first is what gets explored at all.

Every proposed policy must pass a paired evaluation before it is adopted; see
``bench/evolve.py``.  Nothing is trusted because it sounds like an improvement.
"""

import json
import math
import os
import re
from collections import Counter, defaultdict

from . import enum_core, portfolio
from . import grid as G

POLICY_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "policy", "policy.json")


# --------------------------------------------------------------------------
# task signatures
# --------------------------------------------------------------------------

def signatures(ctx):
    """Cheap categorical descriptors of a task, computed from train only."""
    sig = []
    same = ctx.same_shape
    sig.append("shape:same" if same else "shape:diff")
    if not same:
        bigger = all(G.area(b) > G.area(a) for a, b in ctx.train)
        smaller = all(G.area(b) < G.area(a) for a, b in ctx.train)
        sig.append("size:up" if bigger else ("size:down" if smaller else "size:mix"))
        if ctx.shape_ratio:
            sig.append("ratio:%dx%d" % ctx.shape_ratio)
        if ctx.inv_shape_ratio:
            sig.append("iratio:%dx%d" % ctx.inv_shape_ratio)
        if ctx.const_out_shape:
            sig.append("outshape:const")
    ip, op = ctx.in_palette, ctx.out_palette
    if op - ip:
        sig.append("pal:new")
    if ip - op:
        sig.append("pal:drop")
    if op == ip:
        sig.append("pal:same")
    sig.append("ntrain:%d" % min(len(ctx.train), 5))
    sig.append("bg:%d" % ctx.bg)
    a0 = ctx.train[0][0]
    h, w = G.dims(a0)
    sig.append("in:%s" % ("small" if h * w <= 64 else
                          ("mid" if h * w <= 400 else "big")))
    npal = len(ctx.in_palette)
    sig.append("ncol:%s" % ("low" if npal <= 3 else ("mid" if npal <= 5 else "high")))
    try:
        from .solvers.partition import _sep_color_candidates
        if _sep_color_candidates(ctx):
            sig.append("sep:yes")
    except Exception:
        pass
    if all(G.symmetries(a) for a in ctx.inputs):
        sig.append("sym:in")
    return tuple(sig)


# --------------------------------------------------------------------------
# program parsing
# --------------------------------------------------------------------------

_NAME_RE = re.compile(r"^([A-Za-z0-9_#\-]+)\((.*)\)$")


def parse_chain(name):
    """``"crop(rot90($))"`` -> ``["rot90", "crop"]`` (application order)."""
    ops = []
    s = name
    while s != "$":
        m = _NAME_RE.match(s)
        if not m:
            return None
        head, inner = m.group(1), m.group(2)
        if "," in inner:            # binary node: not a linear chain
            return None
        ops.append(head)
        s = inner
    ops.reverse()
    return ops


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------

class Policy:
    def __init__(self, data=None):
        d = data or {}
        self.solver_prior = dict(d.get("solver_prior", {}))
        self.feature_bias = {k: dict(v) for k, v in d.get("feature_bias", {}).items()}
        self.abstractions = list(d.get("abstractions", []))
        self.op_bias = dict(d.get("op_bias", {}))
        self.module_order = list(d.get("module_order", []))
        self.module_skip = {k: list(v) for k, v in d.get("module_skip", {}).items()}
        self.meta = dict(d.get("meta", {}))

    def to_dict(self):
        return {"solver_prior": self.solver_prior,
                "feature_bias": self.feature_bias,
                "abstractions": self.abstractions,
                "op_bias": self.op_bias,
                "module_order": self.module_order,
                "module_skip": self.module_skip,
                "meta": self.meta}

    def save(self, path=POLICY_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1, sort_keys=True)

    @staticmethod
    def load(path=POLICY_PATH):
        if not os.path.exists(path):
            return Policy()
        with open(path) as fh:
            return Policy(json.load(fh))

    # -- application --------------------------------------------------
    def bias_for(self, sigs):
        """Prior offset per solver family for a task with these signatures."""
        out = defaultdict(float)
        for s in sigs:
            for fam, v in self.feature_bias.get(s, {}).items():
                out[fam] += v
        for fam, v in self.solver_prior.items():
            out[fam] += v
        # clamp so a learned bias can reorder but never silence a family
        return {k: max(-3.0, min(3.0, v)) for k, v in out.items()}

    def skips_for(self, sigs, min_votes=2):
        """Families to leave out entirely for a task with these signatures."""
        votes = Counter()
        for s in sigs:
            for f in self.module_skip.get(s, ()):
                votes[f] += 1
        alive = set()
        for s in sigs:
            for f in self.feature_bias.get(s, {}):
                alive.add(f)
        return {f for f, v in votes.items() if v >= min_votes and f not in alive}

    def install(self):
        """Install abstractions and operator bias into the enumerator."""
        enum_core.clear_learned()
        for ab in self.abstractions:
            _install_abstraction(ab)
        enum_core.OP_BIAS = dict(self.op_bias)


def _install_abstraction(ab):
    names = tuple(ab["ops"])
    cost = float(ab.get("cost", 1.0))

    def factory(ctx, names=names):
        base = {n: f for n, _c, f in enum_core.base_unary_ops(ctx)}
        fns = [base.get(n) for n in names]
        if any(f is None for f in fns):
            return None

        def run(g, fns=fns):
            for f in fns:
                g = f(g)
                if g is None:
                    return None
            return g
        return run
    enum_core.add_learned_op(ab["name"], cost, factory)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def fit(records, prev=None, min_count=2, bias_weight=1.2,
        abs_min_count=3, max_abstractions=24):
    """Fit a policy from experience records.

    A record is ``{"sigs": [...], "solved": 0/1, "solver": str,
    "program": str, "time": float}``.
    """
    pol = Policy(prev.to_dict() if prev else None)

    # --- feature-conditioned family bias --------------------------------
    tot = Counter()
    hit = defaultdict(Counter)
    for r in records:
        for s in r["sigs"]:
            tot[s] += 1
            if r["solved"] and r.get("solver"):
                hit[s][r["solver"]] += 1
    fb = {}
    for s, n in tot.items():
        if n < min_count:
            continue
        fams = hit.get(s)
        if not fams:
            continue
        total_hits = sum(fams.values())
        row = {}
        for fam, c in fams.items():
            # log-odds of this family explaining a task with this signature,
            # relative to a uniform prior over the families ever seen
            p = (c + 0.5) / (total_hits + 0.5 * max(1, len(fams)))
            row[fam] = round(-bias_weight * math.log(max(p, 1e-6) * len(fams)), 4)
        fb[s] = row
    pol.feature_bias = fb

    # --- learned skips: reclaim time from families that never pay off ------
    # A family that has produced nothing across many tasks carrying a given
    # signature is not merely unlikely there -- it is spending budget that the
    # productive families could use.  Skipping is gated like everything else,
    # so a skip that costs a solve is rejected by the paired test.
    fams = set()
    for r in records:
        if r.get("solver"):
            fams.add(r["solver"])
    skips = {}
    for s, n in tot.items():
        if n < 12:
            continue
        dead = sorted(f for f in fams if hit.get(s, {}).get(f, 0) == 0)
        if dead:
            skips[s] = dead[:6]
    pol.module_skip = skips

    # --- global family yield --------------------------------------------
    solved_by = Counter(r["solver"] for r in records if r["solved"] and r.get("solver"))
    n_solved = sum(solved_by.values()) or 1
    pol.solver_prior = {fam: round(-0.6 * math.log(1 + 8.0 * c / n_solved), 4)
                        for fam, c in solved_by.items()}

    # --- operator statistics and abstraction mining ----------------------
    op_counts = Counter()
    grams = Counter()
    for r in records:
        if not r["solved"]:
            continue
        progs = list(r.get("programs") or [])
        if r.get("program"):
            progs.append(r["program"])
        seen_here = set()
        for prog in progs:
            chain = parse_chain(prog)
            if not chain:
                continue
            key = tuple(chain)
            if key in seen_here:
                continue          # one vote per task, not per variant
            seen_here.add(key)
            op_counts.update(chain)
            for n in (2, 3):
                for i in range(len(chain) - n + 1):
                    grams[tuple(chain[i:i + n])] += 1
    pol.op_bias = {k: round(0.35 * math.log(1 + v), 4)
                   for k, v in op_counts.items() if v >= 2}

    existing = {tuple(a["ops"]) for a in pol.abstractions}
    new = []
    for gram, c in grams.most_common():
        if c < abs_min_count or gram in existing:
            continue
        new.append({"name": "abs_" + "_".join(gram).replace("#", "c"),
                    "ops": list(gram),
                    "cost": round(1.0 + 0.35 * len(gram), 3),
                    "support": c})
        if len(pol.abstractions) + len(new) >= max_abstractions:
            break
    pol.abstractions = (pol.abstractions + new)[:max_abstractions]

    # --- module ordering by measured yield per second --------------------
    cost_by = Counter()
    for r in records:
        if r.get("solver"):
            cost_by[r["solver"]] += r.get("time", 0.0)
    order = sorted(solved_by, key=lambda f: -(solved_by[f] / (1.0 + cost_by[f])))
    pol.module_order = order

    pol.meta = {"n_skip_signatures": len(pol.module_skip),
                "n_records": len(records),
                "n_solved": sum(1 for r in records if r["solved"]),
                "n_abstractions": len(pol.abstractions),
                "n_signatures": len(fb)}
    return pol


# --------------------------------------------------------------------------
# runtime hook
# --------------------------------------------------------------------------

_ACTIVE = {"policy": None}


def activate(policy):
    """Make ``policy`` the one in force, or clear the engine back to stock.

    Deactivating has to undo the DSL extension too: a worker process that has
    already installed abstractions would otherwise keep searching with them
    while reporting itself as the unmodified baseline, quietly contaminating
    the paired comparison the whole loop rests on.
    """
    _ACTIVE["policy"] = policy
    if policy is not None:
        policy.install()
    else:
        enum_core.clear_learned()
        enum_core.OP_BIAS = {}
    portfolio.POLICY = policy


def active():
    return _ACTIVE["policy"]
