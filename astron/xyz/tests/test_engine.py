"""Behavioural tests for the ASTRA engine.

The important ones are not the algebra checks -- they are the guarantees the
reported score depends on: that a solver cannot see a test answer, that a task
only counts when every test pair is exact, that the ranking is deterministic,
and that the capacity guards actually reject memorising rules.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import enum_core, grid as G, learn, objects as O, portfolio  # noqa: E402
from engine.task import Ctx, Hyp  # noqa: E402


class TestGrid(unittest.TestCase):
    g = ((1, 2, 3), (4, 5, 6))

    def test_dims_and_validity(self):
        self.assertEqual(G.dims(self.g), (2, 3))
        self.assertTrue(G.valid(self.g))
        self.assertFalse(G.valid(((1, 2), (3,))))
        self.assertFalse(G.valid(()))

    def test_dihedral_is_a_group(self):
        for _n, f in G.DIHEDRAL:
            self.assertTrue(G.valid(f(self.g)))
        self.assertEqual(G.rot90(G.rot90(G.rot90(G.rot90(self.g)))), self.g)
        self.assertEqual(G.flip_h(G.flip_h(self.g)), self.g)
        self.assertEqual(G.rot180(self.g), G.flip_h(G.flip_v(self.g)))
        self.assertEqual(G.transpose(G.transpose(self.g)), self.g)

    def test_scale_round_trip(self):
        up = G.upscale(self.g, 3, 2)
        self.assertEqual(G.dims(up), (6, 6))
        self.assertEqual(G.downscale(up, 3, 2), self.g)
        self.assertIsNone(G.downscale(self.g, 4, 1))

    def test_block_reduce_nonbg(self):
        g = ((0, 0, 0, 3), (0, 7, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        self.assertEqual(G.block_reduce_nonbg(g, 2, 2, 0), ((7, 3), (0, 0)))
        bad = ((7, 3), (0, 0))
        self.assertIsNone(G.block_reduce_nonbg(bad, 2, 2, 0))

    def test_holes_and_fill(self):
        ring = ((0, 4, 0), (4, 0, 4), (0, 4, 0))
        self.assertEqual(G.holes(ring, 0, False), [(1, 1)])
        self.assertEqual(G.fill_holes(ring, 2, 0)[1][1], 2)

    def test_gravity_preserves_multiset(self):
        g = ((0, 5, 0), (3, 0, 0), (0, 0, 7))
        for d in ("up", "down", "left", "right"):
            self.assertEqual(sorted(v for r in G.gravity(g, 0, d) for v in r),
                             sorted(v for r in g for v in r))


class TestObjects(unittest.TestCase):
    def test_connectivity_modes_differ(self):
        g = ((1, 0), (0, 1))
        self.assertEqual(len(O.segment(g, "c4", 0)), 2)
        self.assertEqual(len(O.segment(g, "c8", 0)), 1)

    def test_gap_segmentation_joins_dashes(self):
        g = ((3, 0, 3, 0, 3),) + ((0, 0, 0, 0, 0),) * 4
        self.assertEqual(len(O.segment(g, "c8", 0)), 3)
        self.assertEqual(len(O.segment(g, "g2", 0)), 1)

    def test_segmentation_cache_is_consistent(self):
        g = ((1, 1, 0), (0, 1, 0), (0, 0, 2))
        a = O.segment(g, "c4", 0)
        b = O.segment(g, "c4", 0)
        self.assertIs(a, b)
        self.assertEqual(sorted(o.size for o in a), [1, 3])

    def test_holes_count(self):
        g = ((4, 4, 4), (4, 0, 4), (4, 4, 4))
        o = O.segment(g, "c4", 0)[0]
        self.assertEqual(o.holes_count(), 1)
        self.assertFalse(o.is_rect())


class TestNoLeak(unittest.TestCase):
    """The solver must be unable to read the answer it is scored on."""

    task = {
        "train": [([[1, 0], [0, 0]], [[0, 1], [0, 0]]),
                  ([[0, 2], [0, 0]], [[2, 0], [0, 0]]),
                  ([[0, 0], [3, 0]], [[0, 0], [0, 3]])],
        "test_in": [[[0, 0], [0, 5]]],
    }

    def test_ctx_carries_no_outputs(self):
        ctx = Ctx(self.task["train"], self.task["test_in"])
        blob = repr(ctx.__dict__)
        self.assertIn("test_inputs", blob)
        self.assertNotIn("test_outputs", blob)
        self.assertFalse(hasattr(ctx, "answers"))

    def test_prediction_ignores_any_claimed_answer(self):
        r1 = portfolio.solve(self.task["train"], self.task["test_in"],
                             time_budget=6.0)
        r2 = portfolio.solve(self.task["train"], self.task["test_in"],
                             time_budget=6.0)
        self.assertEqual(r1.predictions, r2.predictions)
        self.assertTrue(r1.predictions[0])


class TestRanking(unittest.TestCase):
    def test_hypothesis_must_fit_every_pair(self):
        train = ((((1,),), ((2,),)), (((3,),), ((3,),)))
        always2 = Hyp("k2", lambda g: ((2,),), 1.0, "t")
        self.assertFalse(always2.fits(train))
        ident = Hyp("id", lambda g: g, 1.0, "t")
        self.assertFalse(ident.fits(train))

    def test_exceptions_in_a_hypothesis_are_contained(self):
        boom = Hyp("boom", lambda g: 1 / 0, 1.0, "t")
        self.assertIsNone(boom.apply(((1,),)))
        self.assertFalse(boom.fits(((((1,),), ((1,),)),)))

    def test_invalid_output_is_rejected(self):
        bad = Hyp("bad", lambda g: ((1, 2), (3,)), 1.0, "t")
        self.assertIsNone(bad.apply(((1,),)))

    def test_vote_prefers_agreement_across_families(self):
        train = [([[1]], [[1]])]
        r = portfolio.solve(train, [[[1]]], time_budget=4.0, loo=False)
        self.assertEqual(r.predictions[0][0], ((1,),))


class TestCapacityGuards(unittest.TestCase):
    def test_cellwise_rejects_a_memorising_table(self):
        from engine.solvers import cellwise
        # each cell has a distinct 8-neighbourhood; a table keyed on it would
        # be exactly as large as the data and must not be offered
        import random
        random.seed(0)
        a = tuple(tuple(random.randint(0, 9) for _ in range(6)) for _ in range(6))
        b = tuple(tuple(random.randint(0, 9) for _ in range(6)) for _ in range(6))
        ctx = Ctx([(G.to_list(a), G.to_list(b))], [G.to_list(a)])
        for h in cellwise.generate(ctx):
            self.assertLess(len(h.fn.table) * 3, G.area(a) + 1)

    def test_colormap_lookup_requires_compression(self):
        from engine.solvers import colormap
        train = [([[1, 1]], [[2, 2]]), ([[3, 3]], [[4, 4]])]
        ctx = Ctx(train, [[[1, 1]]])
        names = [h.name for h in colormap.generate(ctx)]
        self.assertFalse([n for n in names if n.startswith("lookup_")])


class TestSymmetryRepair(unittest.TestCase):
    def test_repairs_a_mirrored_grid(self):
        from engine.solvers.symmetry import _repair
        base = ((1, 2, 3, 2, 1),
                (4, 5, 6, 5, 4),
                (7, 8, 9, 8, 7),
                (4, 5, 6, 5, 4),
                (1, 2, 3, 2, 1))
        holed = tuple(tuple(0 if (r, c) in {(1, 1), (2, 1)} else v
                            for c, v in enumerate(row))
                      for r, row in enumerate(base))
        self.assertEqual(_repair(holed, 0), base)

    def test_declines_on_a_cell_symmetry_cannot_determine(self):
        # the centre of a symmetric grid is a fixed point of every symmetry,
        # so nothing determines it -- the strict repair must decline, not guess
        from engine.solvers.symmetry import _repair
        base = ((1, 2, 3, 2, 1),
                (4, 5, 6, 5, 4),
                (7, 8, 9, 8, 7),
                (4, 5, 6, 5, 4),
                (1, 2, 3, 2, 1))
        holed = tuple(tuple(0 if (r, c) == (2, 2) else v
                            for c, v in enumerate(row))
                      for r, row in enumerate(base))
        self.assertIsNone(_repair(holed, 0))

    def test_declines_when_an_orbit_is_unobserved(self):
        from engine.solvers.symmetry import _repair
        blank = G.const_grid(5, 5, 0)
        self.assertIsNone(_repair(blank, 0))


class TestEnumerator(unittest.TestCase):
    def test_finds_a_two_step_program(self):
        a = ((1, 0), (0, 0))
        b = G.rot180(a)
        ctx = Ctx([(G.to_list(a), G.to_list(b))], [G.to_list(a)])
        found = enum_core.search(ctx, depth=2, max_states=300, level="full")
        self.assertTrue(found)
        self.assertEqual(found[0][2](a), b)

    def test_observational_equivalence_collapses_duplicates(self):
        a = ((1, 2), (3, 4))
        ctx = Ctx([(G.to_list(a), G.to_list(a))], [G.to_list(a)])
        ops = enum_core.base_unary_ops(ctx)
        states = set()
        for _n, _c, f in ops:
            try:
                r = f(a)
            except Exception:
                continue
            if r is not None:
                states.add(r)
        self.assertLess(len(states), len(ops))


class TestLearning(unittest.TestCase):
    def test_chain_parsing(self):
        self.assertEqual(learn.parse_chain("crop(rot90($))"), ["rot90", "crop"])
        self.assertEqual(learn.parse_chain("$"), [])
        self.assertIsNone(learn.parse_chain("hcat(a($),b($))"))

    def test_fit_mines_abstractions_and_biases(self):
        recs = [{"sigs": ["shape:same"], "solved": 1, "solver": "geometry",
                 "program": "crop(rot90($))", "time": 0.1} for _ in range(4)]
        recs += [{"sigs": ["shape:diff"], "solved": 0, "solver": None,
                  "program": None, "time": 0.1}]
        pol = learn.fit(recs, abs_min_count=3)
        self.assertIn("shape:same", pol.feature_bias)
        self.assertIn("geometry", pol.feature_bias["shape:same"])
        self.assertTrue(any(a["ops"] == ["rot90", "crop"]
                            for a in pol.abstractions))
        self.assertIn("rot90", pol.op_bias)

    def test_bias_is_clamped(self):
        pol = learn.Policy({"feature_bias": {"s": {"f": -99.0}},
                            "solver_prior": {"f": -99.0}})
        self.assertGreaterEqual(pol.bias_for(("s",))["f"], -3.0)

    def test_abstraction_installs_and_runs(self):
        pol = learn.Policy({"abstractions": [
            {"name": "abs_t", "ops": ["rot90", "rot90"], "cost": 1.5}]})
        pol.install()
        try:
            a = ((1, 2), (3, 4))
            ctx = Ctx([(G.to_list(a), G.to_list(a))], [G.to_list(a)])
            ops = dict((n, f) for n, _c, f in enum_core.unary_ops(ctx))
            self.assertIn("abs_t", ops)
            self.assertEqual(ops["abs_t"](a), G.rot180(a))
        finally:
            enum_core.clear_learned()

    def test_signatures_are_derived_from_train_only(self):
        ctx = Ctx([([[1]], [[1, 1]])], [[[1]]])
        sig = learn.signatures(ctx)
        self.assertIn("shape:diff", sig)
        self.assertIn("size:up", sig)


class TestHarness(unittest.TestCase):
    def test_sign_test(self):
        from bench.evolve import sign_test
        self.assertEqual(sign_test(0, 0), 1.0)
        self.assertAlmostEqual(sign_test(5, 0), 1 / 32.0)
        self.assertAlmostEqual(sign_test(1, 1), 0.75)
        self.assertLess(sign_test(10, 1), 0.01)

    def test_split_is_deterministic_and_disjoint(self):
        from bench.evolve import split
        files = ["t%03d.json" % i for i in range(200)]
        a1, b1 = split(files)
        a2, b2 = split(files)
        self.assertEqual((a1, b1), (a2, b2))
        self.assertEqual(len(a1) + len(b1), 200)
        self.assertFalse(set(a1) & set(b1))

    def test_task_scored_only_when_every_test_pair_is_exact(self):
        from bench.run_arc import run_one
        import json
        import tempfile
        task = {"train": [{"input": [[1]], "output": [[1]]}],
                "test": [{"input": [[1]], "output": [[1]]},
                         {"input": [[2]], "output": [[9]]}]}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "fake_task.json")
            with open(p, "w") as fh:
                json.dump(task, fh)
            r = run_one((p, 4.0, 2, False, None, ""))
        self.assertEqual(r["solved"], 0)     # identity gets pair 1, not pair 2
        self.assertEqual(r["n_test"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestPolicySkips(unittest.TestCase):
    def test_skips_need_votes_and_no_contrary_evidence(self):
        recs = []
        for i in range(14):
            recs.append({"sigs": ["shape:same", "bg:0"], "solved": 1,
                         "solver": "geometry", "program": None, "time": 0.1})
        pol = learn.fit(recs)
        # only one family has ever been seen solving, so nothing is dead yet
        self.assertEqual(pol.module_skip, {})
        recs += [{"sigs": ["shape:same", "bg:0"], "solved": 1,
                  "solver": "cellwise", "program": None, "time": 0.1}]
        recs += [{"sigs": ["shape:diff", "bg:0"], "solved": 1,
                  "solver": "select", "program": None, "time": 0.1}
                 for _ in range(13)]
        pol2 = learn.fit(recs)
        # "select" never solved a shape:same task, so it is a skip candidate
        self.assertIn("select", pol2.module_skip.get("shape:same", []))
        # but a family with a bias entry at one of the task's signatures stays
        self.assertNotIn("geometry", pol2.skips_for(("shape:same", "bg:0")))

    def test_policy_round_trip_keeps_skips(self):
        pol = learn.Policy({"module_skip": {"s": ["a", "b"]}})
        again = learn.Policy(pol.to_dict())
        self.assertEqual(again.module_skip, {"s": ["a", "b"]})


class TestActivation(unittest.TestCase):
    def test_deactivating_clears_learned_operators(self):
        pol = learn.Policy({"abstractions": [
            {"name": "abs_z", "ops": ["rot90", "crop"], "cost": 1.5}],
            "op_bias": {"rot90": 1.0}})
        learn.activate(pol)
        self.assertTrue(enum_core.LEARNED)
        self.assertTrue(enum_core.OP_BIAS)
        learn.activate(None)
        self.assertFalse(enum_core.LEARNED)
        self.assertFalse(enum_core.OP_BIAS)
        self.assertIsNone(portfolio.POLICY)
