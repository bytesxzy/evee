import time
import unittest
from engine import grid as G
from engine.task import Ctx
from engine.solvers import relproc as R


def attraction(color, anchor_col, row, marker_col, height=6, width=13):
    g = [[0] * width for _ in range(height)]
    for r in range(height):
        g[r][anchor_col] = color
    g[row][marker_col] = color
    out = [x[:] for x in g]
    out[row][marker_col] = 0
    out[row][anchor_col + (-1 if marker_col < anchor_col else 1)] = color
    return G.as_grid(g), G.as_grid(out)


class RelationalPrograms(unittest.TestCase):
    def test_induced_relation_transfers_to_new_color_and_direction(self):
        train = [attraction(2, 8, 1, 1), attraction(3, 9, 3, 2),
                 attraction(4, 7, 4, 1)]
        inp, expected = attraction(9, 4, 2, 10, height=7)
        ctx = Ctx(train, [inp], deadline=time.time() + 5)
        hypotheses = R.generate(ctx)
        self.assertTrue(hypotheses)
        self.assertTrue(all(all(h.apply(a) == b for a, b in train) for h in hypotheses))
        self.assertTrue(any(h.apply(inp) == expected for h in hypotheses))

    def test_ambiguous_geometric_anchor_is_rejected(self):
        g = G.as_grid([[0, 2, 0, 3, 0, 2, 0]])
        scene = R.Scene(g, "c4", 0)
        center = next(o for o in scene.objs if o.color == 3)
        self.assertIsNone(R._peer(scene, center, "diff"))
        self.assertIsNone(R._writes(scene, center, ("touch", "diff")))
        self.assertEqual(R._peer(scene, center, "diff", color_only=True).color, 2)

    def test_conflicting_draws_are_rejected(self):
        g = G.as_grid([[2, 0, 0, 3]])
        rule = R.Rule("c4", 0, ("color",),
                      {(2,): ("connect", "near", "straight", 4),
                       (3,): ("connect", "near", "straight", 5)})
        self.assertIsNone(rule(g))

    def test_missing_feature_value_abstains(self):
        g, _ = attraction(9, 4, 2, 10)
        rule = R.Rule("c4", 0, ("color",), {(2,): ("keep",)})
        self.assertIsNone(rule(g))

    def test_no_stencil_or_absolute_displacement_vocabulary(self):
        self.assertFalse(set(R._VERBS) & {"patch", "move", "stencil"})
        g, _ = attraction(3, 8, 2, 1)
        scene = R.Scene(g, "c4", 0)
        marker = next(o for o in scene.objs if o.size == 1)
        writes = R._writes(scene, marker, ("touch", "samebig"))
        self.assertEqual(writes, {(2, 1): 0, (2, 7): 3})

    def test_template_scale_is_inferred_from_anchor_not_a_fixed_list(self):
        g = [[0] * 24 for _ in range(24)]
        g[1][1], g[1][2], g[2][1], g[2][2] = 3, 3, 3, 1
        for r in range(10, 14):
            for c in range(14, 18):
                g[r][c] = 4
        for r in range(14, 18):
            for c in range(14, 18):
                g[r][c] = 1
        scene = R.Scene(G.as_grid(g), "m4", 0)
        receiver = max(scene.objs, key=lambda o: o.size)
        expected = {(r, c): (1 if r >= 14 and c >= 14 else 4)
                    for r in range(10, 18) for c in range(10, 18)}
        self.assertEqual(R._writes(scene, receiver, ("complete", "diff")), expected)

    def test_equivalent_peer_selectors_reuse_pure_writes(self):
        g, _ = attraction(3, 8, 2, 1)
        scene = R.Scene(g, "c4", 0)
        marker = next(o for o in scene.objs if o.size == 1)
        a = R._writes(scene, marker, ("touch", "samebig"))
        b = R._writes(scene, marker, ("touch", "larger"))
        self.assertIs(a, b)


if __name__ == "__main__":
    unittest.main()
