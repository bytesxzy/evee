"""Validate and rank independent program generators without retraining.

Description length supplies prior weights. Bounded leave-one-out refits provide
additional evidence for the same rule and test behaviour, never an unrelated
rule in its family. Ensemble weights are evidence, not calibrated probabilities.
"""
import heapq
import importlib.util
import math
from pathlib import Path
import sys
import time
from . import grid as G
from .task import Ctx
from . import hypcache

SOLVER_PRIOR = {
    "geometry": 0.0, "cellwise": 1.0, "partition": 0.0, "symmetry": 0.0,
    "objects": 1.5, "tiling": 0.5, "colormap": 0.0, "select": 1.0,
    "compose": 2.5, "enumerate": 3.0, "sequence": 1.0,
}
_REGISTRY = []
POLICY = None


def register(module):
    if module not in _REGISTRY:
        _REGISTRY.append(module)


def _load_default():
    if _REGISTRY:
        return
    from .solvers import (analogy, assemble, blocks, cascade, cellwise,
                          colormap, compose, conditional, counting,
                          enumerate_dsl, extend, geometry, locate, motion, objchain,
                          objects_map, objproc,
                          objwise, paint, panelabs, panelwise, paneltable,
                          partition, patterns, refine, regions, relproc, rewrite, select,
                          selfstamp, sequence, substitute, symmetry, tally, tiling)
    for m in (geometry, colormap, partition, paneltable, symmetry, tiling,
              blocks, selfstamp, extend, select, locate, regions, counting, cellwise,
              objects_map, objproc, relproc,
              tally, motion, substitute, sequence, paint, patterns, assemble, analogy,
              compose, panelabs, panelwise, objwise, objchain, rewrite, cascade, refine,
              conditional, enumerate_dsl):
        register(m)


_ARCH_MATH = None
_ARCH_MATH_TRIED = False


def _load_arch_math_sidecar():
    """Find the top-level c4_arch_math addon without changing _REGISTRY.

    The sidecar is deliberately not a normal registered module: registering it
    would change the built-in portfolio's per-module time shares.  Instead, the
    solver below uses it only as a last-chance fallback after the stock modules
    have had their original budget.  Import failures are non-fatal.
    """
    global _ARCH_MATH, _ARCH_MATH_TRIED
    if _ARCH_MATH_TRIED:
        return _ARCH_MATH
    _ARCH_MATH_TRIED = True
    try:
        here = Path(__file__).resolve()
        target = next((parent / "c4_arch_math.py" for parent in here.parents
                       if (parent / "c4_arch_math.py").is_file()), None)
        if target is None:
            return None
        name = "cell4_arch_math_sidecar"
        spec = importlib.util.spec_from_file_location(name, target)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        # dataclasses and some import helpers expect the module to be registered
        # while its body executes.
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        if not callable(getattr(mod, "generate", None)):
            return None
        _ARCH_MATH = mod
    except Exception:
        _ARCH_MATH = None
    return _ARCH_MATH


class Result:
    __slots__ = ("predictions", "hyps", "elapsed", "n_hyps", "n_fit", "solver",
                 "chosen", "diagnostics")

    def __init__(self):
        self.predictions, self.hyps, self.chosen = [], [], []
        self.elapsed = 0.0
        self.n_hyps = self.n_fit = 0
        self.solver = None
        self.diagnostics = {"modules": [], "loo": [], "predictions": []}


def _prediction(hyp, grid):
    """Keep custom generators behind the same output boundary as built-ins."""
    try:
        out = hyp.apply(grid)
        if out is None:
            return None
        out = tuple(tuple(row) for row in out)
        return out if G.is_grid(out) else None
    except Exception:
        return None


def _candidates(mod, ctx, stats, validation_deadline=None):
    """Stream validation, isolating eager and lazy generator failures."""
    stats.update(generated=0, fitted=0, invalid=0, status="complete")
    started = time.time()
    validation_deadline = (ctx.deadline if validation_deadline is None
                           else validation_deadline)

    def expired():
        return validation_deadline is not None and time.time() > validation_deadline

    try:
        candidates = iter(hypcache.generate(mod, ctx))
        while True:
            if expired():
                stats["status"] = "timed_out"
                break
            try:
                hyp = next(candidates)
            except StopIteration:
                break
            stats["generated"] += 1
            try:
                cost = float(hyp.cost)
                if not math.isfinite(cost) or not isinstance(hyp.name, str) or not isinstance(hyp.solver, str):
                    raise ValueError("invalid hypothesis metadata")
            except Exception:
                stats["invalid"] += 1
                continue
            fits = True
            for inp, out in ctx.train:
                if expired():
                    stats["status"] = "timed_out"
                    fits = False
                    break
                if _prediction(hyp, inp) != out:
                    fits = False
                    break
            if fits:
                stats["fitted"] += 1
                yield hyp, cost
    except Exception as exc:
        stats["status"] = "error"
        stats["error"] = type(exc).__name__ + ": " + str(exc)[:160]
    finally:
        stats["generation_timed_out"] = ctx.timed_out()
        stats["elapsed"] = round(time.time() - started, 4)


def _loo_evidence(mod, ctx, targets):
    """Refit once per fold for all (solver, name, test-predictions) keys.

    Incomplete folds are missing evidence. Matching-name refits split credit
    when ambiguous; one lucky rule among many cannot earn full credit.
    """
    evidence = {key: [0.0, 0] for key in targets}
    if len(ctx.train) < 3:
        return evidence
    identities = {key[:2] for key in targets}
    for i, (held_in, held_out) in enumerate(ctx.train):
        if ctx.timed_out():
            break
        sub = ctx.train[:i] + ctx.train[i + 1:]
        started = time.time()
        fold_end = min(ctx.deadline or float("inf"), started + 0.7)
        sub_ctx = Ctx(sub, [held_in], deadline=started + (fold_end - started) * 0.9)
        if hasattr(ctx, "op_prior"):
            sub_ctx.op_prior = ctx.op_prior
        stats, observations = {}, {}
        for hyp, _cost in _candidates(mod, sub_ctx, stats, fold_end):
            identity = (hyp.solver, hyp.name)
            if identity not in identities:
                continue
            signature = tuple(_prediction(hyp, g) for g in ctx.test_inputs)
            observation = (_prediction(hyp, held_in), signature)
            observations.setdefault(identity, set()).add(observation)
        if stats["status"] != "complete" or sub_ctx.timed_out():
            continue
        for key, counts in evidence.items():
            variants = observations.get(key[:2], set())
            counts[1] += 1
            if variants:
                counts[0] += sum(pred == held_out and sig == key[2]
                                 for pred, sig in variants) / float(len(variants))
    return evidence


def _loo_bonus(mod, ctx, hyp):
    """Compatibility helper: evidence for one concrete rule."""
    key = (hyp.solver, hyp.name,
           tuple(_prediction(hyp, g) for g in ctx.test_inputs))
    wins, trials = _loo_evidence(mod, ctx, [key])[key]
    return wins / trials if trials else None



# --------------------------------------------------------------------------
# generation scheduling
# --------------------------------------------------------------------------

def _module_key(mod):
    return getattr(mod, "__name__", "").rsplit(".", 1)[-1]


def _harvest(mod, ctx, module_end, bias, reservoir, order, res):
    """Run one generator for its slice and fold what fits into the reservoir."""
    now = time.time()
    # Eager search generators commonly return at their own deadline. Keep a
    # separate validation slice so their useful results are not discarded.
    ctx.deadline = now + max(0.0, module_end - now) * 0.9
    fam = getattr(mod, "SOLVER", "")
    try:
        prior = float(SOLVER_PRIOR.get(fam, 2.0)) + float(bias.get(fam, 0.0))
        if not math.isfinite(prior):
            raise ValueError("nonfinite prior")
    except (TypeError, ValueError, OverflowError):
        prior = 2.0
    stats = {"solver": fam, "module": _module_key(mod)}
    for hyp, cost in _candidates(mod, ctx, stats, module_end):
        score = cost + prior
        if not math.isfinite(score):
            stats["invalid"] += 1
            continue
        item = (-score, -order, hyp, mod)
        order += 1
        if len(reservoir) < 600:
            heapq.heappush(reservoir, item)
        elif item[:2] > reservoir[0][:2]:
            heapq.heapreplace(reservoir, item)
    res.n_hyps += stats["generated"]
    res.diagnostics["modules"].append(stats)
    return order


def _planner_for(ctx, res):
    """The active planner plus this task's signatures, or ``None``."""
    try:
        from . import planner as PL
        pl = PL.active()
        if pl is None or not pl.trained:
            return None
        from .learn import signatures
        return pl, tuple(signatures(ctx))
    except Exception as exc:
        res.diagnostics["planner_error"] = type(exc).__name__
        return None


def _planned_generation(ctx, res, plan, phase1, phase2, bias, reservoir,
                        order, t0, generation_end):
    """Choose the next generator, and its share, from P(action | state).

    The loop is closed: after each generator reports back, the reasoning state
    changes -- which families have now failed, how many hypotheses fit, how much
    budget is left -- and the distribution is recomputed against the new state.
    Every family still runs; the planner moves time between them rather than
    silencing any, so a mistrained planner costs ordering, not coverage.
    """
    pl, sigs = plan
    trace = []
    ran = []
    deepened = False
    for group in (phase1, phase2):
        if not group:
            continue
        pool = {_module_key(m): m for m in group}
        while pool:
            now = time.time()
            if now >= generation_end:
                break
            span = max(1e-6, generation_end - t0)
            frac_left = max(0.0, generation_end - now) / span
            avail = set(pool)
            if group is phase2 and not deepened and "enumerate_dsl" in ran:
                avail.add("deepen")
            try:
                dist = pl.distribution(sigs, ran, len(reservoir), frac_left,
                                       len(ran), available=avail)
            except Exception:
                dist = {}
            if not dist:
                key = sorted(pool)[0]
                p = 1.0 / len(pool)
            else:
                key = max(dist, key=lambda a: (dist[a], a))
                p = dist[key]
            remaining_time = max(0.0, generation_end - now)
            if key == "deepen":
                deepened = True
                slice_ = min(remaining_time * 0.5,
                             max(0.5, p * remaining_time * 1.5))
                order = _deepen(ctx, res, bias, reservoir, order, now + slice_)
                trace.append({"action": "deepen", "p": round(p, 4),
                              "slice": round(slice_, 3),
                              "fit_after": len(reservoir)})
                continue
            mod = pool.pop(key)
            n_left = len(pool)
            floor = min(0.12, remaining_time / max(1, n_left + 1))
            want = p * remaining_time * 1.7
            cap = remaining_time - floor * n_left
            slice_ = max(floor, min(want, max(floor, cap)))
            before = len(reservoir)
            order = _harvest(mod, ctx, now + slice_, bias, reservoir, order, res)
            ran.append(key)
            trace.append({"action": key, "p": round(p, 4),
                          "slice": round(slice_, 3),
                          "fit_gain": len(reservoir) - before})
    res.diagnostics["plan"] = trace
    return order


class _DeepEnumerator:
    """A deeper pass of the general enumerator, available as a plan action."""

    SOLVER = "enumerate"
    __name__ = "deepen"

    @staticmethod
    def generate(ctx):
        from . import enum_core
        from .task import Hyp
        found = enum_core.search(ctx, depth=5, max_states=2600,
                                 deadline=ctx.deadline, level="full",
                                 use_binary=True,
                                 prior=getattr(ctx, "op_prior", None))
        return [Hyp("deep:" + n, f, 3.4 + cst, "enumerate")
                for n, cst, f in found]


def _deepen(ctx, res, bias, reservoir, order, module_end):
    return _harvest(_DeepEnumerator, ctx, module_end, bias, reservoir, order, res)


def _shape_options(ctx, tg):
    """Keep all demonstrated shape laws when training does not distinguish them."""
    if not ctx.train:
        return set()
    th, tw = G.dims(tg)
    shapes = set()
    if ctx.const_out_shape:
        shapes.add(ctx.const_out_shape)
    if ctx.same_shape:
        shapes.add((th, tw))
    if ctx.shape_ratio:
        y, x = ctx.shape_ratio
        shapes.add((th * y, tw * x))
    if ctx.inv_shape_ratio:
        y, x = ctx.inv_shape_ratio
        if not (th % y or tw % x):
            shapes.add((th // y, tw // x))
    if all(G.dims(b) == G.dims(a)[::-1] for a, b in ctx.train):
        shapes.add((tw, th))
    return shapes


@hypcache.scoped
def solve(train, test_inputs, time_budget=30.0, k=2, loo=True,
          modules=None, collect_all=False):
    """Return up to k distinct grids per input, plus bounded diagnostics.

    Budgets are cooperative: arbitrary solver code cannot be preempted, but
    generation, validation and refits all check their allocated deadlines.
    """
    time_budget = float(time_budget)
    if not math.isfinite(time_budget) or time_budget < 0:
        raise ValueError("time_budget must be finite and nonnegative")
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if modules is None:
        _load_default()
    mods = list(_REGISTRY if modules is None else modules)
    t0 = time.time()
    deadline = t0 + time_budget
    ctx = Ctx(train, test_inputs, deadline=deadline)
    res, bias = Result(), {}
    if POLICY is not None:
        try:
            from .learn import signatures
            sigs = signatures(ctx)
            bias = POLICY.bias_for(sigs)
            ctx.op_prior = POLICY.op_bias
            if POLICY.module_order:
                rank = {f: i for i, f in enumerate(POLICY.module_order)}
                mods = sorted(mods, key=lambda m: rank.get(getattr(m, "SOLVER", ""), 99))
            skip = POLICY.skips_for(sigs)
            kept = [m for m in mods if getattr(m, "SOLVER", "") not in skip]
            if kept:
                mods = kept
        except Exception as exc:
            bias = {}
            res.diagnostics["policy_error"] = type(exc).__name__

    # Cheap modules donate unused time to later search. Reserve enough budget
    # to evaluate test predictions and to refit rather than silently skip LOO.
    reserve = min(3.0, time_budget * 0.15) if loo and len(ctx.train) >= 3 else 0.0
    generation_end = deadline - reserve - min(0.2, time_budget * 0.03)
    phase1 = [m for m in mods if getattr(m, "PHASE", 1) != 2]
    phase2 = [m for m in mods if getattr(m, "PHASE", 1) == 2]
    reservoir, order = [], 0
    plan = _planner_for(ctx, res)
    if plan is not None:
        order = _planned_generation(ctx, res, plan, phase1, phase2, bias,
                                    reservoir, order, t0, generation_end)
    else:
        p1_end = t0 + (generation_end - t0) * (0.45 if phase2 else 1.0)
        share1 = (p1_end - t0) / max(1, len(phase1))
        for mi, mod in enumerate(phase1 + phase2):
            now = time.time()
            if now >= generation_end:
                break
            if mi < len(phase1):
                module_end = min(generation_end, max(now + share1 * 0.5,
                                                     t0 + share1 * (mi + 1)))
            else:
                remaining = len(phase1) + len(phase2) - mi
                module_end = min(generation_end,
                                 now + (generation_end - now) / remaining)
            order = _harvest(mod, ctx, module_end, bias, reservoir, order, res)

    # Regression-safe architectural math fallback.  It cannot alter a task for
    # which the stock engine already found any fitted hypothesis, and it only
    # consumes otherwise-unused generation time.
    if not reservoir and time.time() < generation_end:
        mod = _load_arch_math_sidecar()
        now = time.time()
        if mod is not None and now < generation_end:
            module_end = generation_end
            ctx.deadline = now + (module_end - now) * 0.9
            fam = getattr(mod, "SOLVER", "arch_math")
            stats = {"solver": fam, "mode": "fallback_only"}
            try:
                prior = float(SOLVER_PRIOR.get(fam, 2.0)) + float(bias.get(fam, 0.0))
                if not math.isfinite(prior):
                    raise ValueError("nonfinite prior")
            except (TypeError, ValueError, OverflowError):
                prior = 2.0
            for hyp, cost in _candidates(mod, ctx, stats, module_end):
                score = cost + prior
                if not math.isfinite(score):
                    stats["invalid"] += 1
                    continue
                item = (-score, -order, hyp, mod)
                order += 1
                if len(reservoir) < 600:
                    heapq.heappush(reservoir, item)
                elif item[:2] > reservoir[0][:2]:
                    heapq.heapreplace(reservoir, item)
            res.n_hyps += stats["generated"]
            res.diagnostics.setdefault("sidecars", []).append(stats)

    fitted = sorted([[-s, -o, h, m] for s, o, h, m in reservoir],
                    key=lambda r: (r[0], r[1]))
    res.n_fit = len(fitted)
    res.diagnostics["total_fitted"] = order
    ctx.deadline = deadline

    # Cache predictions once. Deduplication below prevents aliases from
    # crowding distinct behaviours out of the finite voting pool.
    signatures = {}
    for _score, idx, hyp, _mod in fitted:
        if signatures and ctx.timed_out():
            break
        signatures[idx] = tuple(_prediction(hyp, g) for g in ctx.test_inputs)
    fitted = [rec for rec in fitted if rec[1] in signatures]
    if loo and len(ctx.train) >= 3:
        groups, selected = {}, set()
        for _score, idx, hyp, mod in fitted:
            key = (hyp.solver, hyp.name, signatures[idx])
            full_key = (id(mod), key)
            if full_key not in selected and len(selected) < 12:
                selected.add(full_key)
                groups.setdefault(id(mod), (mod, []))[1].append(key)
        adjustments = {}
        for mod_id, (mod, keys) in groups.items():
            if ctx.timed_out():
                break
            for key, (wins, trials) in _loo_evidence(mod, ctx, keys).items():
                # Partial cross-validation carries proportionally less weight.
                adjustment = ((1.5 - 4.5 * wins / trials) * trials / len(ctx.train)
                              if trials else 0.0)
                adjustments[(mod_id, key)] = adjustment
                res.diagnostics["loo"].append({"solver": key[0], "name": key[1],
                    "wins": round(wins, 4), "trials": trials,
                    "folds": len(ctx.train), "adjustment": round(adjustment, 4)})
        for rec in fitted:
            key = (rec[2].solver, rec[2].name, signatures[rec[1]])
            rec[0] += adjustments.get((id(rec[3]), key), 0.0)
    fitted.sort(key=lambda r: (r[0], r[1]))
    res.hyps = [(r[2].solver + ":" + r[2].name, round(r[0], 2)) for r in fitted[:8]]

    pool, seen_behaviours = [], set()
    for rec in fitted:
        signature = signatures[rec[1]]
        key = (rec[2].solver, signature)
        if key not in seen_behaviours and any(g is not None for g in signature):
            seen_behaviours.add(key)
            pool.append(rec)
            if len(pool) >= 150:
                break
    res.diagnostics["voting_hypotheses"] = len(pool)
    preserves_colors = all(G.palette(b) <= G.palette(a) for a, b in ctx.train)
    for ti, tg in enumerate(ctx.test_inputs):
        shapes = _shape_options(ctx, tg)
        allowed = G.palette(tg) | ctx.out_palette if preserves_colors else None
        best, first, author = {}, {}, {}
        for rank, (score, idx, hyp, _mod) in enumerate(pool):
            g = signatures[idx][ti]
            if g is None:
                continue
            author.setdefault(g, (hyp.solver, hyp.name))
            family = best.setdefault(g, {})
            family[hyp.solver] = min(score, family.get(hyp.solver, float("inf")))
            first.setdefault(g, rank)
        scored = []
        for g, families in best.items():
            logits = [-s / 2.0 for s in families.values()]
            peak = max(logits)
            weight = peak + math.log(sum(math.exp(v - peak) for v in logits))
            violations = int(bool(shapes) and G.dims(g) not in shapes)
            violations += int(allowed is not None and not G.palette(g) <= allowed)
            weight += violations * math.log(0.25)
            scored.append((-weight, first[g], g, violations, len(families)))
        scored.sort(key=lambda row: row[:2])
        predictions = [row[2] for row in scored]
        res.predictions.append(predictions if collect_all else predictions[:k])
        res.chosen.append(author.get(predictions[0]) if predictions else None)
        res.diagnostics["predictions"].append({"distinct": len(scored),
            "shape_options": sorted(shapes),
            "top_support": scored[0][4] if scored else 0,
            "top_violations": scored[0][3] if scored else 0,
            "log_weight_margin": round(scored[1][0] - scored[0][0], 4)
                                 if len(scored) > 1 else None})
    res.solver = next((chosen[0] for chosen in res.chosen if chosen), None)
    res.elapsed = time.time() - t0
    res.diagnostics["timed_out"] = res.elapsed > time_budget
    res.diagnostics["unrun_modules"] = len(mods) - len(res.diagnostics["modules"])
    return res
