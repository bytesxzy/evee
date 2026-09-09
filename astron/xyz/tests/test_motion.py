"""Movement reasoning must transfer positions and resolve identical copies."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.task import Ctx
from engine.solvers import motion


def scene(height, width, cells):
    rows = [[0] * width for _ in range(height)]
    for r, c, color in cells:
        rows[r][c] = color
    return tuple(tuple(row) for row in rows)


def square(row, col, color=2):
    return [(row + r, col + c, color) for r in range(2) for c in range(2)]


class ObjectMotion(unittest.TestCase):
    def test_repeated_objects_use_joint_translation_not_greedy_matching(self):
        # The first source's nearest target belongs to the SECOND source.
        a = scene(1, 10, [(0, 5, 3), (0, 8, 3)])
        b = scene(1, 10, [(0, 1, 3), (0, 4, 3)])
        ctx = Ctx([(a, b)], [])
        table = motion._fit(ctx, "c4", "all", 0)
        self.assertEqual(table, {0: (0, -4)})
        self.assertEqual(motion._apply(a, "c4", "all", table, 0), b)

    def test_alignment_generalizes_to_new_distance_size_and_color(self):
        a1 = scene(8, 10, square(4, 1) + [(1, 7, 3)])
        b1 = scene(8, 10, square(4, 1) + [(4, 7, 3)])
        a2 = scene(8, 10, square(2, 4) + [(6, 1, 4), (0, 8, 4)])
        b2 = scene(8, 10, square(2, 4) + [(2, 1, 4), (2, 8, 4)])
        test = scene(11, 9, square(7, 3, 6) + [(1, 0, 8)])
        answer = scene(11, 9, square(7, 3, 6) + [(7, 0, 8)])
        ctx = Ctx([(a1, b1), (a2, b2)], [test])
        found = motion.generate(ctx)
        rules = [h for h in found if h.name.startswith("align_") and h.fits(ctx.train)]
        self.assertTrue(rules)
        self.assertTrue(any(h.apply(test) == answer for h in rules))

    def test_ambiguous_anchor_is_rejected(self):
        g = scene(7, 9, square(1, 1) + square(4, 5, 4))
        self.assertIsNone(motion._apply_relative(g, "c4", 0, "largest", "near", "keep"))

    def test_overlap_is_rejected_even_when_colors_match(self):
        g = scene(6, 6, square(3, 2) + [(0, 2, 2)])
        self.assertIsNone(motion._apply_relative(g, "c4", 0, "largest", "near", "keep"))

    def test_half_cell_center_is_ambiguous(self):
        g = scene(8, 8, square(3, 3) + [(0, 0, 7)])
        self.assertIsNone(motion._apply_relative(g, "c4", 0, "largest", "after", "center"))

    def test_table_cannot_destroy_colliding_objects(self):
        g = scene(1, 5, [(0, 0, 2), (0, 4, 3)])
        self.assertIsNone(motion._apply(g, "c4", "color", {2: (0, 2), 3: (0, -2)}, 0))


if __name__ == "__main__":
    unittest.main()
