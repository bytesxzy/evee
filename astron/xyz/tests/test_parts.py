import time
import unittest
from engine import grid as G
from engine.task import Ctx
from engine.solvers import parts as P


class JointParts(unittest.TestCase):
    def test_touching_parts_are_jointly_decomposed(self):
        cells = {(0, c) for c in range(5)} | {(1, 0), (1, 1)}
        vocabulary = ((((1, 1), (1, 1)), 8), (((1, 1, 1),), 2))
        expected = {rc: (8 if rc[1] < 2 else 2) for rc in cells}
        self.assertEqual(P._cover(cells, vocabulary), expected)

    def test_ambiguous_colors_abstain(self):
        self.assertIsNone(P._cover({(0, 0), (0, 1)},
                                  ((((1, 1),), 2), (((1, 1),), 8))))

    def test_exhausted_search_does_not_claim_uniqueness(self):
        self.assertIsNone(P._cover({(0, 0), (0, 1)}, ((((1, 1),), 2),), max_states=1))

    def test_parts_seen_in_only_one_demonstration_are_rejected(self):
        ctx = Ctx([([[5,5,5,5,5],[5,5,0,0,0]],
                    [[8,8,2,2,2],[8,8,0,0,0]])], [], deadline=time.time()+1)
        self.assertIsNone(P._vocabulary(ctx, 0))

    def test_induction_verifies_every_demonstration(self):
        a = G.as_grid([[5,5,5,5,5],[5,5,0,0,0]])
        b = G.as_grid([[8,8,2,2,2],[8,8,0,0,0]])
        ctx = Ctx([(a,b),(G.transpose(a),G.transpose(b))], [a],
                  deadline=time.time()+1)
        rules = P.generate(ctx)
        self.assertTrue(rules)
        self.assertTrue(all(all(h.apply(x)==y for x,y in ctx.train) for h in rules))


if __name__ == "__main__":
    unittest.main()

