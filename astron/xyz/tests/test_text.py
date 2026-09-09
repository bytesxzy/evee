"""Tests for the text sibling.

Mirrors the discipline of ``tests/test_gabriel.py``: the properties that must
hold are asserted, not described.  Nothing here touches the network, and
nothing here imports ``gabriel`` or ``engine`` -- the two vocabularies are
separate by construction and a test that crossed them would be asserting the
opposite of the design.
"""

import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from text import bench as B          # noqa: E402
from text import conversation as CV  # noqa: E402
from text import corpus as C         # noqa: E402
from text import export as EX        # noqa: E402
from text import families as FAM     # noqa: E402
from text import lm as LM            # noqa: E402
from text import signatures as S     # noqa: E402
from text import train as TRAIN      # noqa: E402


class TestIdentity(unittest.TestCase):
    """A token set is allowed for coverage; it is not sufficient for identity."""

    def test_multiplicity_is_preserved(self):
        self.assertNotEqual(FAM.identity_key("high school"),
                            FAM.identity_key("High School High"))
        self.assertTrue(FAM.duplicate_mismatch("high school", "High School High"))

    def test_stop_words_are_part_of_a_name(self):
        self.assertNotEqual(FAM.identity_key("matrix"), FAM.identity_key("the matrix"))

    def test_order_matters(self):
        self.assertNotEqual(FAM.identity_key("rock math"), FAM.identity_key("math rock"))

    def test_qualifier_is_not_part_of_the_name(self):
        self.assertEqual(FAM.identity_key("Matrix (mathematics)"),
                         FAM.identity_key("matrix"))
        base, qual = FAM.split_qualifier("Python (programming language)")
        self.assertEqual(base, "Python")
        self.assertEqual(qual, "programming language")

    def test_ordered_containment_counts_extras(self):
        self.assertEqual(FAM.ordered_containment("high school", "High School High"),
                         (True, 1))
        self.assertEqual(FAM.ordered_containment("school", "School of Rock"),
                         (True, 2))
        self.assertEqual(FAM.ordered_containment("math", "MathJax")[0], False)


class TestLadder(unittest.TestCase):

    def test_every_adversarial_pair(self):
        fails = B.run_ladder(verbose=False)
        self.assertEqual(fails, [], "ladder failures: %r" % (fails,))

    def test_verified_redirect_outranks_an_expansion(self):
        kept, best, _ = FAM.prune("high school", [
            {"title": "High School High"},
            {"title": "High School Musical"},
            {"title": "Secondary school", "verified_title": True,
             "alias": "high school"}])
        self.assertEqual(best, FAM.TIER_VERIFIED_TITLE)
        self.assertEqual([c["title"] for c in kept], ["Secondary school"])

    def test_specificity_works_both_ways(self):
        t, _ = FAM.tier("high school musical", {"title": "High School Musical"})
        self.assertGreaterEqual(t, FAM.TIER_EXACT_PHRASE)
        t, _ = FAM.tier("math rock", {"title": "Math rock"})
        self.assertGreaterEqual(t, FAM.TIER_EXACT_PHRASE)

    def test_an_expansion_never_survives_a_better_tier(self):
        kept, _b, _n = FAM.prune("apple", [
            {"title": "Apple Inc."}, {"title": "Apple"}])
        self.assertNotIn("Apple Inc.", [c["title"] for c in kept])


class TestSignatures(unittest.TestCase):

    def test_signatures_are_topic_blind(self):
        a = S.signatures("what is photosynthesis")
        b = S.signatures("what is cryptography")
        self.assertEqual(a, b)

    def test_case_shape_distinguishes_a_name(self):
        self.assertIn("case:lower", S.signatures("what is high school"))
        self.assertIn("case:title", S.signatures("What is High School Musical"))

    def test_duplicate_token_signal(self):
        self.assertIn("dup:yes", S.signatures("what is high school high"))
        self.assertIn("dup:no", S.signatures("what is high school"))

    def test_topic_shift_is_detected_and_stripped(self):
        self.assertIn("shift:yes", S.signatures("anyway what is photosynthesis"))
        self.assertEqual(S.split_subject_relation("anyway what is photosynthesis")[0],
                         "photosynthesis")

    def test_contraction_is_a_copula_not_a_possessive(self):
        subj, rel, parse = S.split_subject_relation("what's photosynthesis")
        self.assertEqual(subj, "photosynthesis")
        self.assertEqual(rel, "")
        self.assertEqual(parse, "bare")

    def test_a_real_possessive_still_parses(self):
        subj, rel, parse = S.split_subject_relation("France's capital")
        self.assertEqual((subj, rel, parse), ("France", "capital", "possessive"))

    def test_empty_and_strange_input(self):
        for q in ("", "   ", "???", "a", "\n\t", "😀"):
            self.assertIsInstance(S.signatures(q), tuple)
            self.assertIsInstance(S.split_subject_relation(q), tuple)


class TestGrammar(unittest.TestCase):
    """Constrained decoding: the model may only choose among these."""

    def test_a_research_form_can_never_route_conversationally(self):
        for q in ("what is photosynthesis", "who is Albert Einstein",
                  "capital of France", "how does a turbine work",
                  "define entropy", "explain quantum mechanics"):
            self.assertNotIn("conversational", FAM.grammatical_families(q),
                             "%r admitted conversational" % q)

    def test_a_social_turn_admits_conversation(self):
        for q in ("hey", "lol", "how are you", "that's cool", "wait what",
                  "why do you sound so formal", "nah bro"):
            self.assertIn("conversational", FAM.grammatical_families(q),
                          "%r did not admit conversational" % q)

    def test_the_corpus_never_labels_outside_the_grammar(self):
        _tr, _dv, _ho, stats = C.build_family_corpus()
        self.assertEqual(stats["grammar_misses"], 0)


class TestModel(unittest.TestCase):

    def test_untrained_model_is_uniform_not_wrong(self):
        m = LM.TextLM()
        d = m.family_dist("what is photosynthesis")
        self.assertTrue(d)
        self.assertAlmostEqual(sum(d.values()), 1.0, places=6)

    def test_fit_beats_the_grammar_alone_on_the_holdout(self):
        model, report = TRAIN.fit()
        self.assertGreaterEqual(report["holdout"]["family_accuracy"],
                                report["baseline"]["family_holdout_accuracy_uniform"])

    def test_the_learned_head_never_sees_a_rejected_candidate(self):
        """The ladder runs first; pruned candidates cannot be scored back in."""
        model, _r = TRAIN.fit()
        cands = [{"title": "High School High", "coherence": 0.9, "rank": 1,
                  "source": "wikipedia"},
                 {"title": "Secondary school", "verified_title": True,
                  "alias": "high school", "coherence": 0.9, "rank": 0,
                  "source": "wikipedia"}]
        plan = LM.resolve(model, "what is high school", cands)
        self.assertEqual([c["title"] for c in plan["candidates"]],
                         ["Secondary school"])

    def test_ranking_never_crosses_a_tier(self):
        model, _r = TRAIN.fit()
        cands = [{"title": "High School High", "coherence": 1.0, "rank": 0,
                  "source": "wikipedia", "full_text": True},
                 {"title": "High School Musical", "coherence": 1.0, "rank": 0,
                  "source": "wikipedia", "full_text": True}]
        plan = LM.resolve(model, "what is high school musical", cands)
        tiers = [c["tier"] for c in plan["candidates"]]
        self.assertEqual(tiers, sorted(tiers, reverse=True))
        self.assertEqual(plan["candidates"][0]["title"], "High School Musical")

    def test_saved_model_round_trips(self):
        model, _r = TRAIN.fit()
        d = model.to_dict()
        again = LM.TextLM(d)
        q = "what is photosynthesis"
        self.assertEqual(sorted(model.family_dist(q)), sorted(again.family_dist(q)))


class TestCorpusDiscipline(unittest.TestCase):

    def test_splits_are_disjoint_by_template(self):
        train, dev, hold, _s = C.build_family_corpus()
        tt = {e["template"] for e in train}
        dt = {e["template"] for e in dev}
        ht = {e["template"] for e in hold}
        self.assertFalse(tt & dt)
        self.assertFalse(tt & ht)
        self.assertFalse(dt & ht)

    def test_candidate_corpus_avoids_the_benchmark_holdout(self):
        self.assertEqual(B.run_disjointness(), [])

    def test_the_runtime_does_not_contain_the_benchmark(self):
        paths = [TRAIN.MODEL_PATH, EX.PACK_PATH,
                 os.path.join(ROOT, "text", "families.py"),
                 os.path.join(ROOT, "text", "signatures.py"),
                 os.path.join(ROOT, "text", "lm.py")]
        self.assertEqual(B.run_audit(paths), [])


class TestConversation(unittest.TestCase):

    def test_act_model_generalises(self):
        _m, stats = CV.train_act_model()
        self.assertGreaterEqual(stats["holdout_accuracy"], 0.6)

    def test_no_dataset_text_is_shipped(self):
        """Every string a user can see originates in this repository."""
        payload = json.loads(open(EX.PACK_PATH).read()) if os.path.exists(EX.PACK_PATH) else None
        if payload is None:
            self.skipTest("pack not exported")
        self.assertFalse(payload["provenance"]["verbatim_text_from_datasets"])
        for act, plans in payload["plans"].items():
            for skel, _needs in plans:
                self.assertIn(skel, [p[0] for p in CV.PLANS[act]])

    def test_statistics_are_counts_only(self):
        stats = CV.statistics_from_turns([
            {"prompt": "hey", "reply": "Hello there, how can I help?"},
            {"prompt": "thanks", "reply": "Any time."}])
        blob = json.dumps(stats)
        self.assertNotIn("Hello there", blob)
        self.assertNotIn("Any time", blob)

    def test_a_plan_never_renders_an_empty_slot(self):
        for act in CV.ACTS:
            text, _i = CV.render(act, {"subject": ""}, seed=0)
            self.assertNotIn("{topic}", text)
            self.assertTrue(text.strip())


class TestExport(unittest.TestCase):

    def test_artifacts_exist_and_are_small(self):
        for path in (TRAIN.MODEL_PATH, EX.PACK_PATH):
            self.assertTrue(os.path.exists(path), path)
            self.assertLess(os.path.getsize(path), 2 * 1024 * 1024, path)

    def test_model_declares_the_contract_the_browser_needs(self):
        d = json.load(open(TRAIN.MODEL_PATH))
        self.assertEqual(d["kind"], "astron-text-lm")
        for key in ("family_weights", "candidate_weights", "families", "stop",
                    "contract", "provenance"):
            self.assertIn(key, d)
        self.assertEqual([list(p) for p in LM.PAIR_GROUPS],
                         d["contract"]["pair_groups"])

    def test_conformance_vectors_are_producible(self):
        vec = EX.conformance_vectors()
        self.assertTrue(vec["queries"])
        self.assertTrue(vec["tiers"])
        for row in vec["queries"]:
            self.assertIn("signatures", row)
            self.assertIn("families", row)


class TestIsolation(unittest.TestCase):
    """The ARC side must not be able to notice this package exists."""

    def test_no_arc_imports(self):
        import text
        base = os.path.dirname(os.path.abspath(text.__file__))
        for name in os.listdir(base):
            if not name.endswith(".py"):
                continue
            src = open(os.path.join(base, name)).read()
            self.assertNotIn("from engine", src, name)
            self.assertNotIn("from gabriel", src, name)
            self.assertNotIn("import engine", src, name)
            self.assertNotIn("import gabriel", src, name)

    def test_only_the_harvester_touches_the_network(self):
        import text
        base = os.path.dirname(os.path.abspath(text.__file__))
        for name in os.listdir(base):
            if not name.endswith(".py") or name == "harvest_oasst.py":
                continue
            src = open(os.path.join(base, name)).read()
            # Strip docstrings: a module that *documents* not opening a socket
            # is not opening one.
            code = re.sub(r'"""(?:.|\n)*?"""', " ", src)
            code = re.sub(r"(?m)^\s*#.*$", " ", code)
            for forbidden in ("urllib.request", "http.client", "import socket",
                              "socket.socket", "requests.get", "requests.post"):
                self.assertNotIn(forbidden, code, "%s uses %s" % (name, forbidden))


if __name__ == "__main__":
    unittest.main()
