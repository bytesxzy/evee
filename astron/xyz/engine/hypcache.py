"""Per-solve reuse of completed specialist generation.

Partial, expired, failed, and abandoned streams are never cached. Cache lifetime
is one top-level solve; transformed and leave-one-out contexts have distinct keys.
"""
from contextvars import ContextVar
from functools import wraps

ENABLED = True
_CURRENT = ContextVar("arc_hypothesis_cache", default=None)
_CACHEABLE = frozenset(("geometry", "colormap", "partition", "symmetry", "tiling",
    "blocks", "select", "regions", "counting", "cellwise", "objects_map",
    "objproc", "relproc", "motion", "substitute", "sequence", "paint", "patterns",
    "analogy", "assemble", "paneltable", "selfstamp", "tally", "extend", "locate"))


def scoped(fn):
    @wraps(fn)
    def call(*args, **kwargs):
        state = {"items": {}, "hits": 0, "misses": 0, "stored_hyps": 0} if ENABLED else None
        token = _CURRENT.set(state)
        try:
            result = fn(*args, **kwargs)
            if state is not None and hasattr(result, "diagnostics"):
                result.diagnostics["hypothesis_cache"] = {
                    k: v for k, v in state.items() if k != "items"}
                result.diagnostics["hypothesis_cache"]["entries"] = len(state["items"])
            return result
        finally:
            _CURRENT.reset(token)
    return call


def generate(module, ctx):
    state = _CURRENT.get()
    name = getattr(module, "__name__", "")
    if state is None or name.rsplit(".", 1)[-1] not in _CACHEABLE:
        yield from module.generate(ctx)
        return
    prior = tuple(sorted(getattr(ctx, "op_prior", {}).items()))
    key = (module, ctx.train, ctx.test_inputs, prior, ctx.budget)
    cached = state["items"].get(key)
    if cached is not None:
        state["hits"] += 1
        yield from cached
        return
    state["misses"] += 1
    complete = []
    for hyp in module.generate(ctx):
        complete.append(hyp)
        yield hyp
    if not ctx.timed_out() and len(state["items"]) < 96 and state["stored_hyps"] + len(complete) <= 12000:
        state["items"][key] = tuple(complete)
        state["stored_hyps"] += len(complete)

