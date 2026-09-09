"""Behavioral regressions for bounded compositional synthesis."""

import os
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import enum_core as E, grid as G
from engine.task import Ctx


def context(inputs, outputs, tests=()):
    return Ctx(list(zip(inputs, outputs)), tests)


class SearchTests(unittest.TestCase):
    def run_search(self, ctx, ops, **kwargs):
        with patch.object(E, "unary_ops", return_value=ops):
            return E.search(ctx, level="small", max_states=40,
                            use_binary=kwargs.pop("use_binary", False), **kwargs)

    def test_identity_is_a_zero_depth_program(self):
        a, t = ((1, 2),), ((3, 4),)
        found = self.run_search(context([a], [a], [t]), [], depth=0)
        self.assertEqual(found[0][0:2], ("$", 0.0))
        self.assertEqual(found[0][2](t), t)

    def test_equivalent_cheaper_program_replaces_prior_favorite(self):
        a = ((1, 2, 3), (4, 5, 6))
        ops = [("expensive", 9.0, G.transpose), ("cheap", 1.0, G.transpose)]
        found = self.run_search(context([a], [G.transpose(a)], [a]), ops,
                                depth=1, prior={"expensive": 100})
        self.assertEqual(found[0][0:2], ("cheap($)", 1.0))

    def test_cost_and_remaining_depth_both_matter(self):
        start, middle, detour = ((0, 0),), ((1, 2),), ((3,),)
        goal = ((4, 5, 4, 5),)

        def edge(source, dest):
            return lambda g: dest if g == source else None

        ops = [("direct", 8, edge(start, middle)),
               ("step1", 1, edge(start, detour)),
               ("step2", 1, edge(detour, middle)),
               ("finish", 1, edge(middle, goal))]
        ctx = context([start], [goal], [start])
        shallow = self.run_search(ctx, ops, depth=2)
        deeper = self.run_search(ctx, ops, depth=3)
        self.assertEqual(shallow[0][1], 9)
        self.assertEqual(deeper[0][1], 3)
        self.assertEqual(deeper[0][2](start), goal)

    def test_same_training_behavior_keeps_distinct_test_predictions(self):
        a, b, t = ((0, 0),), ((1, 2),), ((5, 5),)
        ops = [("a", 1, lambda g: b if g == a else ((3, 4),)),
               ("b", 1, lambda g: b if g == a else ((4, 3),))]
        found = self.run_search(context([a], [b], [t]), ops, depth=1)
        self.assertEqual({fn(t) for _, _, fn in found},
                         {((3, 4),), ((4, 3),)})

    def test_target_fitted_recolor_composes_after_crop(self):
        a = ((0, 0, 0, 0), (0, 1, 2, 0), (0, 2, 1, 0), (0, 0, 0, 0))
        b = ((4, 3), (3, 4))
        t = ((0, 0, 0, 0), (0, 1, 7, 0), (0, 2, 1, 0), (0, 0, 0, 0))
        ops = [("crop", 1.2, lambda g: G.crop_to_content(g, 0))]
        ctx = context([a], [b], [t])
        self.assertFalse(self.run_search(ctx, ops, depth=1))
        found = self.run_search(ctx, ops, depth=2)
        self.assertTrue(found)
        self.assertEqual(found[0][2](t), ((4, 7), (3, 4)))
        self.assertIn("cmap", found[0][0])

    def test_recolor_must_agree_across_every_training_pair(self):
        ctx = context([((1, 1),), ((1, 1),)],
                      [((2, 2),), ((3, 3),)], [((1, 1),)])
        self.assertFalse(self.run_search(ctx, [], depth=1))

    def test_recolor_does_not_memorize_unique_color_cells(self):
        ctx = context([((1, 2, 3, 4),)], [((4, 3, 2, 1),)])
        self.assertFalse(self.run_search(ctx, [], depth=1))

    def test_binary_self_concatenation(self):
        a, t = ((1, 0, 2),), ((3, 0, 4),)
        ctx = context([a], [G.hconcat(a, a)], [t])
        found = self.run_search(ctx, [], depth=1, use_binary=True)
        self.assertEqual(found[0][0], "hcat($,$)")
        self.assertEqual(found[0][2](t), G.hconcat(t, t))

    def test_binary_result_can_be_recolored(self):
        a, t = ((1, 1, 0),), ((1, 0, 1),)
        output = ((2, 2, 0, 2, 2, 0),)
        found = self.run_search(context([a], [output], [t]), [],
                                depth=2, use_binary=True)
        self.assertTrue(found)
        self.assertEqual(found[0][2](t), ((2, 0, 2, 2, 0, 2),))

    def test_binary_logical_ops_use_inferred_nonzero_background(self):
        a = ((1, 9, 9, 9, 2, 9), (9, 9, 9, 9, 9, 9))
        target = ((1, 2, 9), (9, 9, 9))
        # Neither half nor a recoloring alone fits; overlay must combine
        # disjoint foreground cells on a field whose background is 9.
        ops = [("left", 1, lambda g: G.half(g, "left")),
               ("right", 1, lambda g: G.half(g, "right"))]
        self.assertEqual(context([a], [target]).bg, 9)
        with patch.object(E, "binary_ops", side_effect=lambda bg: (
                ("or", 2, lambda x, y: E._log(x, y, "or", bg)),)):
            found = self.run_search(context([a], [target], [a]), ops,
                                    depth=2, use_binary=True)
        self.assertTrue(found)
        self.assertEqual(found[0][2](a), target)

    def test_narrow_beam_considers_later_target_near_candidates(self):
        start, near, far = ((0, 0),), ((1, 1, 1),), ((2,),)
        goal = ((3, 4, 3),)
        ops = [("aa_far", 1, lambda g: far if g == start else None),
               ("zz_near", 1, lambda g: near if g == start else None),
               ("finish", 1, lambda g: goal if g == near else None)]
        with patch.object(E, "unary_ops", return_value=ops):
            found = E.search(context([start], [goal]), depth=2,
                             max_states=1, level="small", use_binary=False)
        self.assertTrue(found)
        self.assertEqual(found[0][0], "finish(zz_near($))")

    def test_seed_counts_toward_the_depth_limit(self):
        start, middle, goal = ((0, 0),), ((1, 2, 1),), ((3, 4, 3, 4),)
        seeds = [("seed", 1, lambda g: middle if g == start else None)]
        ops = [("finish", 1, lambda g: goal if g == middle else None)]
        with patch.object(E, "unary_ops", return_value=ops), \
                patch.object(E, "seed_ops", return_value=seeds):
            self.assertFalse(E.search(context([start], [goal]), depth=1,
                                      level="full", use_binary=False))
            found = E.search(context([start], [goal]), depth=2,
                             level="full", use_binary=False)
        self.assertTrue(found)
        self.assertEqual(found[0][0], "finish(seed($))")

    def test_partial_and_mutable_operators_do_not_crash_search(self):
        a, t = ((0, 0),), ((3, 3),)

        def partial(g):
            if g == t:
                raise ValueError("undefined on test input")
            return ((1, 2),)

        ops = [("partial", 1, partial), ("mutable", 1, lambda g: ([1, 2],)),
               ("ragged", 1, lambda g: ((1, 2), (3,)))]
        self.assertFalse(self.run_search(context([a], [((1, 2),)], [t]),
                                         ops, depth=1))

    def test_expired_context_deadline_prevents_operator_work(self):
        a = ((1, 2),)
        ctx = context([a], [((2, 1),)], [a])
        ctx.deadline = time.time() - 1
        with patch.object(E, "unary_ops", side_effect=AssertionError("must not run")):
            self.assertEqual(E.search(ctx, depth=2), [])

    def test_expired_deadline_prevents_seed_work(self):
        a = ((1, 2),)
        with patch.object(E, "seed_ops", side_effect=AssertionError("must not run")):
            self.assertEqual(E.search(context([a], [((2, 1),)]),
                                      deadline=time.time() - 1), [])


if __name__ == "__main__":
    unittest.main()
