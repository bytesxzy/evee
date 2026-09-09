"""Strap the language model onto the engine, without touching the engine.

``engine/learn.py`` already defines the extension point: a ``Policy`` supplies
a per-family ranking prior, an operator bias for the enumerator, a module
order, and a set of families to skip.  :class:`GabrielPolicy` is that same
interface with the language model behind it, so ``learn.activate()`` installs
it exactly as it installs a fitted policy, and no line of ``engine/`` or
``bench/`` changes.

Two details are worth stating plainly, because they are what makes the model
conditional on the task rather than a global constant:

* ``portfolio.solve`` calls ``POLICY.bias_for(sigs)`` and then reads
  ``POLICY.op_bias``, in that order, for every task.  ``bias_for`` records the
  signatures it was given, and ``op_bias`` is a property that answers for
  *those* signatures.  If that order ever changed, the property would fall back
  to the fitted static bias -- degraded, never wrong.
* The operator names the model must score are the enumerator's, so they are
  harvested from ``enum_core.base_unary_ops`` against a synthetic context that
  exercises the whole palette.  Guessing the names would silently mismatch.
"""

import json
import math
import os

from engine import enum_core, learn, portfolio
from engine.task import Ctx

from . import proposer
from .lm import GabrielLM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LM_PATH = os.path.join(ROOT, "policy", "gabriel_lm.json")
POLICY_PATH = os.path.join(ROOT, "policy", "policy.json")

_OPS = None


def op_vocabulary():
    """Every operator name the enumerator can offer, palette included."""
    global _OPS
    if _OPS is None:
        cells = [[c for c in range(5)], [c for c in range(5, 10)]]
        other = [[9 - c for c in range(5)], [9 - c for c in range(5, 10)]]
        try:
            ctx = Ctx([(cells, other)], [cells])
            _OPS = sorted({n for n, _c, _f in enum_core.base_unary_ops(ctx)})
        except Exception:
            _OPS = []
    return _OPS


class GabrielPolicy(learn.Policy):
    """A fitted policy with the language model's opinion mixed in."""

    _lm = None
    _sigs = ()
    _fam_weight = 1.5          # log-odds -> engine prior units
    _op_weight = 0.7           # model bias -> enumerator bias units
    _op_clamp = 2.0

    def __init__(self, data=None, lm=None, fam_weight=None, op_weight=None):
        self._static_op_bias = {}
        self._cache_key = None
        self._cache = {}
        super().__init__(data)
        self._lm = lm if (lm is not None and lm.is_trained()) else None
        if fam_weight is not None:
            self._fam_weight = float(fam_weight)
        if op_weight is not None:
            self._op_weight = float(op_weight)

    # -- op_bias becomes a question about *this* task -------------------
    @property
    def op_bias(self):
        if self._lm is None or not self._sigs:
            return dict(self._static_op_bias)
        if self._cache_key != self._sigs:
            merged = dict(self._static_op_bias)
            try:
                lm_bias = self._lm.op_bias(self._sigs, op_vocabulary())
            except Exception:
                lm_bias = {}
            for op, v in lm_bias.items():
                merged[op] = round(min(self._op_clamp,
                                       merged.get(op, 0.0) + self._op_weight * v), 4)
            self._cache_key, self._cache = self._sigs, merged
        return dict(self._cache)

    @op_bias.setter
    def op_bias(self, value):
        self._static_op_bias = dict(value or {})

    def bias_for(self, sigs):
        """Fitted family prior, plus the model's ``P(family | task)``."""
        self._sigs = tuple(sigs)
        out = dict(super().bias_for(sigs))
        if self._lm is None:
            return out
        try:
            fam = self._lm.family_dist(sigs)
        except Exception:
            return out
        if not fam:
            return out
        n = max(1, len(fam))
        for f, p in fam.items():
            shift = -self._fam_weight * math.log(max(p, 1e-6) * n)
            out[f] = max(-3.0, min(3.0, out.get(f, 0.0) + shift))
        return out

    def to_dict(self):
        d = super().to_dict()
        d["op_bias"] = dict(self._static_op_bias)
        return d


def bind(policy_path=POLICY_PATH, lm_path=LM_PATH, proposals=True):
    """Activate GABRIEL in this process.  Returns ``(policy, lm)``.

    Safe to call in a worker-pool initializer; safe to call twice.
    """
    lm = GabrielLM.load(lm_path)
    data = None
    if policy_path and os.path.exists(policy_path):
        try:
            with open(policy_path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = None
    pol = GabrielPolicy(data, lm=lm)
    learn.activate(pol)
    if proposals and pol._lm is not None:
        portfolio._load_default()          # never register ahead of the defaults
        proposer.LM = pol._lm
        portfolio.register(proposer)
    return pol, pol._lm


def unbind():
    """Return the process to the stock engine."""
    learn.activate(None)
    proposer.LM = None
    if proposer in portfolio._REGISTRY:
        portfolio._REGISTRY.remove(proposer)
