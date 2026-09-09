"""Tests of scoring and isolation, independent of solver timing."""
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import regression as R
from bench.run_arc import score_predictions


class TestRegression(unittest.TestCase):
    task = {"train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[9]]}]}

    def test_selection_balances_groups_and_ignores_input_order(self):
        names = ["arc1_%d.json" % n for n in range(10)] + ["arc2_%d.json" % n for n in range(10)]
        selected = R.select_tasks(names, 6)
        self.assertEqual(selected, R.select_tasks(list(reversed(names)), 6))
        self.assertEqual(sum(n.startswith("arc1") for n in selected), 3)
        self.assertEqual(len(set(selected)), 6)
        self.assertEqual(len(R.select_tasks(names, 0)), 20)

    def test_answers_never_enter_worker_request(self):
        request = R.solver_request(self.task, 1, 2, True)
        self.assertEqual(request["test_inputs"], [[[3]]])
        self.assertNotIn("9", json.dumps(request))
        altered = dict(self.task, test=[{"input": [[3]], "output": [[8]]}])
        self.assertEqual(request, R.solver_request(altered, 1, 2, True))

    def test_empty_or_truncated_predictions_cannot_solve(self):
        self.assertEqual(R.score([], []), (0, 0))
        self.assertEqual(R.score([], [[[1]]]), (0, 0))
        self.assertEqual(R.score([[[[1]]]], [[[1]], [[2]]]), (0, 0))

    def test_every_test_pair_and_only_first_two_guesses_count(self):
        self.assertEqual(R.score([[[[1]]], [[[0]], [[2]]]], [[[1]], [[2]]]), (0, 1))
        self.assertEqual(R.score([[[[0]], [[2]], [[1]]]], [[[1]]]), (0, 0))

    def test_regular_harness_rejects_empty_or_incomplete_results(self):
        self.assertEqual(score_predictions([], []), (False, False))
        self.assertEqual(score_predictions([], [[[1]]]), (False, False))
        self.assertEqual(score_predictions([[[[1]]]], [[[1]], [[2]]]), (False, False))

    def test_regular_harness_top2_really_means_two(self):
        self.assertEqual(score_predictions([[[[0]], [[2]], [[1]]]], [[[1]]]), (False, False))
        self.assertEqual(score_predictions([[[[0]], [[1]]]], [[[1]]]), (False, True))

    @patch.object(R.subprocess, "run")
    def test_subprocess_has_timeout_and_deterministic_hash_seed(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, json.dumps({"predictions": [[[[9]]]]}), "")
        result = R.run_task(R.ROOT, self.task, 1, 2, True, 3)
        self.assertEqual((result["solved"], result["solved2"]), (1, 1))
        self.assertEqual(run.call_args.kwargs["timeout"], 4)
        self.assertEqual(run.call_args.kwargs["env"]["PYTHONHASHSEED"], "0")
        self.assertNotIn("9", run.call_args.kwargs["input"])

    @patch.object(R.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 4))
    def test_timed_out_worker_is_unsolved(self, run):
        result = R.run_task(R.ROOT, self.task, 1, 2, True, 3)
        self.assertEqual(result["solved2"], 0)
        self.assertIn("timeout", result["error"])

    def test_comparison_rejects_mismatched_tasks_settings_and_content(self):
        report = {"budget": 1, "k": 2, "loo": True, "grace": 3,
                  "hash_seed": 0, "policy": None, "engine_sha256": "engine",
                  "per_task": [{"id": "a", "sha256": "data", "solved": 0, "solved2": 0}]}
        self.assertEqual(R.compare(report, report)["solved"]["wins"], [])
        with self.assertRaises(ValueError):
            R.compare(report, dict(report, budget=2))
        with self.assertRaises(ValueError):
            R.compare(report, dict(report, per_task=[]))
        with self.assertRaises(ValueError):
            R.compare(report, dict(report, per_task=[dict(report["per_task"][0], sha256="changed")]))


if __name__ == "__main__":
    unittest.main()
