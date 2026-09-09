"""Regression tests for inference evidence, search isolation and voting."""
import os
import sys
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import grid as G, portfolio
from engine.task import Ctx, Hyp


def module(family, generate, phase=1):
    return types.SimpleNamespace(SOLVER=family, generate=generate, PHASE=phase)


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.old_policy = portfolio.POLICY
        portfolio.POLICY = None
        self.train = [([[1]], [[1]]), ([[2]], [[2]]), ([[3]], [[3]])]
        self.tests = [[[4]]]

    def tearDown(self):
        portfolio.POLICY = self.old_policy

    def solve(self, modules, **kwargs):
        opts = dict(time_budget=2.0, loo=False, modules=modules)
        opts.update(kwargs)
        return portfolio.solve(self.train, self.tests, **opts)

    def test_loo_does_not_credit_an_unrelated_family_rule(self):
        def generate(ctx):
            table = dict(ctx.train)
            return [Hyp("memorizer", lambda g: table.get(g, ((9,),)), 0, "mixed"),
                    Hyp("identity", lambda g: g, 1, "mixed")]
        result = self.solve([module("mixed", generate)], loo=True)
        self.assertEqual(result.predictions[0][0], ((4,),))
        evidence = {item["name"]: item for item in result.diagnostics["loo"]}
        self.assertEqual(evidence["memorizer"]["wins"], 0)
        self.assertEqual(evidence["identity"]["wins"], 3)
        self.assertGreater(evidence["memorizer"]["adjustment"], 0)
        self.assertLess(evidence["identity"]["adjustment"], 0)

    def test_loo_requires_the_same_test_behaviour(self):
        def generate(ctx):
            prediction = ((9,),) if len(ctx.train) == 3 else ((4,),)
            return [Hyp("changes_after_refit", lambda g: prediction if g == ((4,),) else g,
                        0, "refit"), Hyp("identity", lambda g: g, 1, "refit")]
        result = self.solve([module("refit", generate)], loo=True)
        self.assertEqual(result.predictions[0][0], ((4,),))
        evidence = {item["name"]: item for item in result.diagnostics["loo"]}
        self.assertEqual(evidence["changes_after_refit"]["wins"], 0)
        self.assertEqual(evidence["identity"]["wins"], 3)

    def test_loo_ambiguous_refits_share_evidence_and_ignore_duplicates(self):
        def generate(ctx):
            table = dict(ctx.train)
            exact = Hyp("same_name", lambda g: g, 1, "ambiguous")
            fitted = Hyp("same_name", lambda g: table.get(g, ((4,),) if g == ((4,),) else ((9,),)),
                         1, "ambiguous")
            return [exact] * 8 + [fitted]
        ctx = Ctx(self.train, self.tests, deadline=time.time() + 2)
        hyp = Hyp("same_name", lambda g: g, 1, "ambiguous")
        self.assertEqual(portfolio._loo_bonus(module("ambiguous", generate), ctx, hyp), 0.5)

    def test_failed_refit_is_missing_evidence(self):
        def generate(ctx):
            if len(ctx.train) < 3:
                raise RuntimeError("refit unavailable")
            return [Hyp("identity", lambda g: g, 1, "broken_refit")]
        result = self.solve([module("broken_refit", generate)], loo=True)
        item = result.diagnostics["loo"][0]
        self.assertEqual(item["trials"], 0)
        self.assertEqual(item["adjustment"], 0)

    def test_equal_training_sizes_leave_output_shape_ambiguous(self):
        train = [([[1, 1], [1, 1]], [[1, 1], [1, 1]]),
                 ([[2, 2], [2, 2]], [[2, 2], [2, 2]])]
        test = [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
        def generate(ctx):
            return [Hyp("relative", lambda g: g, 0, "shape"),
                    Hyp("constant", lambda g: G.const_grid(2, 2, g[0][0]), 0.5, "shape")]
        result = portfolio.solve(train, [test], modules=[module("shape", generate)],
                                 time_budget=1, loo=False)
        self.assertEqual(result.predictions[0][0], G.from_list(test))
        self.assertEqual(result.diagnostics["predictions"][0]["shape_options"], [(2, 2), (3, 3)])
        self.assertEqual(result.diagnostics["predictions"][0]["top_violations"], 0)

    def test_transposition_is_a_shape_option_for_square_demonstrations(self):
        ctx = Ctx([([[1, 2], [3, 4]], [[1, 3], [2, 4]])], [[[1, 2, 3], [4, 5, 6]]])
        options = portfolio._shape_options(ctx, ctx.test_inputs[0])
        self.assertIn((3, 2), options)

    def test_lazy_generator_failure_preserves_yielded_hypotheses(self):
        def broken(ctx):
            yield Hyp("identity", lambda g: g, 0, "lazy")
            raise RuntimeError("broken iterator")
        result = self.solve([module("lazy", broken), module("empty", lambda ctx: [])])
        self.assertEqual(result.predictions[0][0], ((4,),))
        self.assertEqual(result.diagnostics["modules"][0]["status"], "error")
        self.assertEqual(result.diagnostics["modules"][1]["status"], "complete")

    def test_bad_candidate_metadata_and_outputs_are_isolated(self):
        bad = types.SimpleNamespace(cost=0, name="bad", solver="safe", apply=lambda g: ((10,),))
        malformed = types.SimpleNamespace(cost=0, name="malformed", solver="safe", apply=lambda g: (0,))
        def generate(ctx):
            return [None, Hyp("nan", lambda g: g, float("nan"), "safe"), bad, malformed,
                    Hyp("valid", lambda g: g, 1, "safe")]
        result = self.solve([module("safe", generate)])
        self.assertEqual(result.n_fit, 1)
        self.assertEqual(result.predictions[0][0], ((4,),))
        self.assertEqual(result.diagnostics["modules"][0]["invalid"], 2)

    def test_eager_search_results_survive_the_generation_deadline(self):
        clock = [100.0]
        def generate(ctx):
            clock[0] = ctx.deadline + 0.001
            return [Hyp("found_at_deadline", lambda g: g, 0, "eager")]
        with patch("engine.portfolio.time.time", side_effect=lambda: clock[0]):
            result = self.solve([module("eager", generate)], time_budget=10)
        self.assertEqual(result.predictions[0][0], ((4,),))
        self.assertTrue(result.diagnostics["modules"][0]["generation_timed_out"])
        self.assertFalse(result.diagnostics["timed_out"])

    def test_timed_out_refits_are_neutral(self):
        clock = [100.0]
        def generate(ctx):
            clock[0] = ctx.deadline + 0.001
            return [Hyp("identity", lambda g: g, 0, "slow")]
        with patch("engine.portfolio.time.time", side_effect=lambda: clock[0]):
            ctx = Ctx(self.train, self.tests, deadline=110.0)
            hyp = Hyp("identity", lambda g: g, 0, "slow")
            self.assertIsNone(portfolio._loo_bonus(module("slow", generate), ctx, hyp))

    def test_late_cheaper_rules_survive_candidate_capacity(self):
        def generate(ctx):
            for i in range(605):
                yield Hyp("expensive_%d" % i,
                          lambda g: g if g != ((4,),) else ((9,),), 10, "stream")
            yield Hyp("late_identity", lambda g: g, 0, "stream")
        result = self.solve([module("stream", generate)])
        self.assertEqual(result.predictions[0][0], ((4,),))
        self.assertEqual(result.n_fit, 600)
        self.assertEqual(result.diagnostics["total_fitted"], 606)

    def test_aliases_cannot_crowd_out_distinct_voting_behaviours(self):
        def aliases(ctx):
            return [Hyp("alias_%d" % i, G.flip_h, 0, "aliases") for i in range(180)] + [
                Hyp("identity", lambda g: g, 1, "aliases")]
        other = module("independent", lambda ctx: [Hyp("identity", lambda g: g, 1, "independent")])
        result = portfolio.solve([([[1, 1]], [[1, 1]])], [[[2, 9]]],
                                 modules=[module("aliases", aliases), other],
                                 time_budget=2, loo=False)
        self.assertEqual(result.predictions[0][0], ((2, 9),))
        self.assertEqual(result.diagnostics["voting_hypotheses"], 3)
        self.assertEqual(result.diagnostics["predictions"][0]["top_support"], 2)
        self.assertEqual(result.solver, result.chosen[0][0])
        self.assertEqual(result.chosen[0], ("aliases", "identity"))

    def test_log_space_voting_handles_extreme_finite_costs(self):
        mods = [module("huge", lambda ctx: [Hyp("identity", lambda g: g, -1e300, "huge")])]
        result = self.solve(mods)
        self.assertEqual(result.predictions[0][0], ((4,),))

    def test_zero_budget_preserves_test_result_cardinality(self):
        result = self.solve([module("unused", lambda ctx: self.fail("must not execute"))], time_budget=0)
        self.assertEqual(result.predictions, [[]])
        self.assertEqual(result.chosen, [None])
        self.assertEqual(result.diagnostics["unrun_modules"], 1)

    def test_invalid_public_budget_and_k_fail_clearly(self):
        for budget in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.solve([], time_budget=budget)
        for k in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                self.solve([], k=k)

    def test_test_input_iterators_are_consumed_once(self):
        result = portfolio.solve(self.train, iter(self.tests), time_budget=1, loo=False,
                                 modules=[module("identity", lambda ctx: [Hyp("id", lambda g: g, 0, "identity")])])
        self.assertEqual(result.predictions, [[((4,),)]])


if __name__ == "__main__":
    unittest.main()
