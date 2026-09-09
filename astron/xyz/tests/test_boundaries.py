"""The external grid and hypothesis boundaries must not alter evidence."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import grid as G
from engine.task import Ctx, Hyp


class GridBoundaries(unittest.TestCase):
    def test_rejects_invalid_cells_and_shapes(self):
        for value in ([[1.5]], [[True]], [["2"]], [[-1]], [[10]],
                      [[None]], [[1], [2, 3]], [], [[]], (None,),
                      [[0] * 61], [[{}]]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Ctx([(value, [[1]])], [])
                self.assertIsNone(Hyp("bad", lambda g, v=value: v).apply(((0,),)))

    def test_predictions_are_hashable_even_with_mixed_row_containers(self):
        h = Hyp("mixed", lambda g: ([1, 2], (3, 4)))
        result = h.apply(((0,),))
        self.assertEqual(result, ((1, 2), (3, 4)))
        self.assertEqual({result: 1}[result], 1)

    def test_context_copies_mutable_evidence(self):
        source = [[1]]
        ctx = Ctx([(source, [[2]])], [source])
        source[0][0] = 9
        self.assertEqual(ctx.inputs, (((1,),),))
        self.assertEqual(ctx.test_inputs, (((1,),),))

    def test_empty_training_is_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "training"):
            Ctx([], [[[1]]])

    def test_internal_masks_remain_structurally_valid(self):
        self.assertTrue(G.valid(((None, 1),)))
        self.assertFalse(G.is_grid(((None, 1),)))
        self.assertFalse(G.valid((1,)))
        self.assertFalse(G.valid(([1],)))


if __name__ == "__main__":
    unittest.main()
