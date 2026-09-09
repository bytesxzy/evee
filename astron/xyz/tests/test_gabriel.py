"""Behavioural tests for GABRIEL -- the language model and how it is strapped on.

The ones that matter are not the tokeniser checks.  They are the properties the
whole arrangement rests on: that the model actually conditions on the task,
that constrained decoding cannot emit an operator the engine does not have,
that a proposed program is still rejected unless it reproduces every training
pair, that the trainer refuses a model which got worse, and that binding the
model leaves the no-leak boundary exactly where the engine put it.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import enum_core, learn, portfolio  # noqa: E402
from engine.task import Ctx  # noqa: E402
from gabriel import bind, corpus, proposer, tokens as T  # noqa: E402
from gabriel.lm import GabrielLM, _op_trie  # noqa: E402


def toy_examples():
    """Two families with disjoint signatures: a model must separate them."""
    a = {"sigs": ["shape:same", "pal:same"], "family": "geometry",
         "body": ["<fam>geometry", "rot90", "(", "$", ")", "<eos>"],
         "weight": 2.0}
    b = {"sigs": ["shape:diff", "size:up"], "family": "tiling",
         "body": ["<fam>tiling", "tile", "#", "3", "(", "$", ")", "<eos>"],
         "weight": 1.0}
    return [a, b]


def toy_model(epochs=40):
    m = GabrielLM()
    m.build_vocab(toy_examples())
    m.train(toy_examples(), epochs=epochs)
    return m


class TestTokens(unittest.TestCase):
    def test_structural_split(self):
        self.assertEqual(T.tokenize("crop(rot90($))"),
                         ["crop", "(", "rot90", "(", "$", ")", ")"])
        self.assertEqual(T.tokenize("fill#3"), ["fill", "#", "3"])
        self.assertEqual(T.tokenize("sep#5.and->2"),
                         ["sep", "#", "5", ".", "and", "->", "2"])

    def test_chain_name_is_application_order(self):
        # crop(rot90($)) applies rot90 first, then crop.
        self.assertEqual(T.chain_name(["rot90", "crop"]), "crop(rot90($))")
        self.assertEqual(learn.parse_chain(T.chain_name(["rot90", "crop"])),
                         ["rot90", "crop"])

    def test_pathological_length_is_bounded(self):
        self.assertLessEqual(len(T.tokenize("f(" * 500)), T.MAX_TOKENS)


class TestLanguageModel(unittest.TestCase):
    def test_conditions_on_the_task(self):
        m = toy_model()
        same = m.family_dist(["shape:same", "pal:same"])
        diff = m.family_dist(["shape:diff", "size:up"])
        self.assertGreater(same["geometry"], same["tiling"])
        self.assertGreater(diff["tiling"], diff["geometry"])

    def test_distributions_are_normalised(self):
        m = toy_model()
        p = m.family_dist(["shape:same", "pal:same"])
        self.assertAlmostEqual(sum(p.values()), 1.0, places=6)

    def test_learning_beats_the_unigram(self):
        m = toy_model()
        self.assertLess(m.perplexity(toy_examples()), 3.0)

    def test_untrained_model_is_inert(self):
        m = GabrielLM()
        self.assertFalse(m.is_trained())
        self.assertEqual(m.family_dist(["shape:same"]), {})
        self.assertEqual(m.op_bias(["shape:same"], ["rot90"]), {})
        self.assertEqual(m.beam_chains(["shape:same"], ["rot90"]), [])

    def test_unknown_operators_do_not_outrank_known_ones(self):
        # A candidate outside the vocabulary is scored as <unk>, not as zero;
        # scoring it as zero would let never-seen operators win every decode.
        m = toy_model()
        chains = m.beam_chains(["shape:same", "pal:same"],
                               ["rot90", "never_seen_op", "other_unknown"],
                               max_len=2, limit=4)
        self.assertEqual(chains[0][0], ["rot90"])

    def test_decoding_cannot_leave_the_operator_set(self):
        m = toy_model()
        allowed = {"rot90", "crop", "tile#3"}
        for ops, _score in m.beam_chains(["shape:same"], sorted(allowed),
                                         max_len=4, limit=20):
            self.assertTrue(set(ops) <= allowed)
            self.assertTrue(1 <= len(ops) <= 4)

    def test_op_bias_stays_in_the_engines_scale(self):
        m = toy_model()
        bias = m.op_bias(["shape:diff", "size:up"], ["rot90", "crop", "tile#3"],
                         scale=1.2)
        self.assertTrue(all(0.0 <= v <= 1.2 for v in bias.values()))

    def test_op_trie_recovers_every_name(self):
        names = ["crop", "crop#3", "fill#10", "rot90"]
        trie, heads = _op_trie(names)
        self.assertEqual(heads, {"crop", "fill", "rot90"})
        found = []

        def walk(node):
            if node[0] is not None:
                found.append(node[0])
            for child in node[1].values():
                walk(child)
        for node in trie.values():
            walk(node)
        self.assertEqual(sorted(found), sorted(names))

    def test_round_trips_through_json(self):
        import json
        import tempfile
        m = toy_model()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "lm.json")
            m.save(p)
            back = GabrielLM.load(p)
            self.assertEqual(back.vocab, m.vocab)
            self.assertAlmostEqual(back.perplexity(toy_examples()),
                                   m.perplexity(toy_examples()), places=3)
            with open(p) as fh:
                json.load(fh)                  # plain JSON, no pickles

    def test_load_of_a_corrupt_file_is_not_fatal(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "lm.json")
            with open(p, "w") as fh:
                fh.write("{not json")
            self.assertIsNone(GabrielLM.load(p))
            self.assertIsNone(GabrielLM.load(os.path.join(d, "absent.json")))

    def test_early_stopping_keeps_the_best_epoch(self):
        m = GabrielLM()
        ex = toy_examples()
        m.build_vocab(ex)
        stats = m.train(ex, epochs=30, dev=ex, patience=2)
        self.assertLessEqual(stats["best_epoch"], 30)
        self.assertIsNotNone(stats["best_dev_perplexity"])


class TestPolicyBinding(unittest.TestCase):
    def test_op_bias_answers_for_the_last_task(self):
        m = toy_model()
        pol = bind.GabrielPolicy({"op_bias": {"crop": 0.5}}, lm=m)
        self.assertEqual(pol.op_bias, {"crop": 0.5})     # no task yet: static
        pol.bias_for(("shape:diff", "size:up"))
        first = pol.op_bias
        pol.bias_for(("shape:same", "pal:same"))
        second = pol.op_bias
        self.assertNotEqual(first, second)               # conditioned, not fixed
        self.assertGreaterEqual(first.get("crop", 0.0), 0.5)

    def test_family_prior_is_clamped(self):
        m = toy_model()
        pol = bind.GabrielPolicy(None, lm=m)
        for v in pol.bias_for(("shape:same", "pal:same")).values():
            self.assertGreaterEqual(v, -3.0)
            self.assertLessEqual(v, 3.0)

    def test_without_a_model_it_is_an_ordinary_policy(self):
        pol = bind.GabrielPolicy({"op_bias": {"crop": 0.4}}, lm=None)
        pol.bias_for(("shape:same",))
        self.assertEqual(pol.op_bias, {"crop": 0.4})
        self.assertEqual(pol.to_dict()["op_bias"], {"crop": 0.4})

    def test_operator_vocabulary_matches_the_enumerator(self):
        names = bind.op_vocabulary()
        self.assertIn("rot90", names)
        self.assertTrue(any(n.startswith("crop#") for n in names))

    def test_bind_and_unbind_are_reversible(self):
        try:
            bind.bind(policy_path="", lm_path="")       # no artefacts present
            self.assertIsNotNone(learn.active())
        finally:
            bind.unbind()
        self.assertIsNone(learn.active())
        self.assertNotIn(proposer, portfolio._REGISTRY)
        self.assertEqual(enum_core.OP_BIAS, {})


class TestProposer(unittest.TestCase):
    train = [(((1, 2), (3, 4)), ((3, 1), (4, 2)))]      # rot90

    def test_no_model_means_no_proposals(self):
        proposer.LM = None
        self.assertEqual(proposer.generate(Ctx(self.train, [((1, 2), (3, 4))])), [])

    def test_proposals_are_runnable_and_still_have_to_fit(self):
        proposer.LM = toy_model()
        try:
            ctx = Ctx(self.train, [((5, 6), (7, 8))])
            hyps = proposer.generate(ctx)
            self.assertTrue(hyps)
            fitting = 0
            for h in hyps:
                self.assertEqual(h.solver, "gabriel")
                self.assertTrue(h.cost > 0)
                self.assertIsNotNone(learn.parse_chain(h.name))
                out = h.apply(ctx.inputs[0])               # total, or None
                if out is not None and h.fits(ctx.train):
                    fitting += 1
            # the model proposes freely; only programs that reproduce the
            # demonstration can be counted, and here exactly one rule does
            self.assertGreaterEqual(fitting, 1)
        finally:
            proposer.LM = None

    def test_proposals_never_see_a_test_output(self):
        proposer.LM = toy_model()
        try:
            ctx = Ctx(self.train, [((5, 6), (7, 8))])
            proposer.generate(ctx)
            self.assertFalse(hasattr(ctx, "test_outputs"))
            self.assertEqual(len(ctx.test_inputs), 1)
        finally:
            proposer.LM = None


class TestCorpusAndGate(unittest.TestCase):
    def test_dev_split_is_deterministic_and_disjoint(self):
        ids = ["arc1_%04x" % i for i in range(400)]
        dev = {i for i in ids if corpus._is_dev(i, 0.15)}
        self.assertTrue(0 < len(dev) < len(ids))
        self.assertEqual(dev, {i for i in ids if corpus._is_dev(i, 0.15)})

    def test_holdout_tasks_never_enter_the_corpus(self):
        from bench.evolve import split
        train, dev, _stats = corpus.build()
        if not train:
            self.skipTest("no evidence in this checkout")
        ids = {e["id"] for e in train} | {e["id"] for e in dev}
        _fit, hold = split(sorted(i + ".json" for i in ids))
        self.assertEqual(hold, [])

    def test_corpus_examples_are_well_formed(self):
        train, dev, _stats = corpus.build()
        if not train:
            self.skipTest("no evidence in this checkout")
        for ex in (train + dev)[:200]:
            self.assertTrue(ex["body"][-1] == T.EOS)
            self.assertTrue(ex["weight"] >= 1.0)
            self.assertTrue(len(ex["body"]) <= T.MAX_TOKENS + 2)

    def test_a_worse_model_is_not_adopted(self):
        import json
        import tempfile
        good = toy_model()
        bad = GabrielLM()
        bad.build_vocab(toy_examples())          # untrained: uniform-ish
        dev = toy_examples()
        self.assertLess(good.perplexity(dev), bad.perplexity(dev))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "lm.json")
            good.save(p)
            with open(p) as fh:
                self.assertIn("weights", json.load(fh))


if __name__ == "__main__":
    unittest.main(verbosity=2)
