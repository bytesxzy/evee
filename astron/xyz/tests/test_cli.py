import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.__main__ import solve_task
from engine.portfolio import Result


class InferenceCLI(unittest.TestCase):
    def test_answers_are_never_forwarded_and_attempts_are_json_ready(self):
        task = {"train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]], "output": [[9]]}]}
        result = Result()
        result.predictions = [[((4,),), ((5,),)]]
        with patch("engine.__main__.portfolio.solve", return_value=result) as solve:
            output = solve_task(task, budget=1.0)
        solve.assert_called_once_with([([[1]], [[2]])], [[[3]]],
                                      time_budget=1.0, k=2, loo=True)
        self.assertEqual(output["predictions"], [{"attempt_1": [[4]], "attempt_2": [[5]]}])

    def test_hidden_test_output_is_optional(self):
        task = {"train": [{"input": [[1]], "output": [[1]]}],
                "test": [{"input": [[2]]}]}
        with patch("engine.__main__.portfolio.solve", return_value=Result()):
            self.assertEqual(solve_task(task)["predictions"], [])

    def test_empty_task_is_rejected(self):
        for task in ({}, {"train": [], "test": []}, []):
            with self.assertRaises(ValueError):
                solve_task(task)


if __name__ == "__main__":
    unittest.main()
