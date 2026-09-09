import time
import unittest
from types import SimpleNamespace
from engine import hypcache as C
from engine.task import Ctx


class HypothesisCache(unittest.TestCase):
    def module(self, fn):
        class Module:
            __name__ = "engine.solvers.geometry"
            generate = staticmethod(fn)
        Module.__name__ = "engine.solvers.geometry"
        return Module

    def context(self, value=1):
        return Ctx([([[value]], [[value]])], [[[value]]], deadline=time.time()+5)

    def test_completed_results_reused_only_within_one_solve(self):
        calls = []
        mod = self.module(lambda ctx: calls.append(ctx) or [1, 2])
        @C.scoped
        def run():
            self.assertEqual(list(C.generate(mod, self.context())), [1, 2])
            self.assertEqual(list(C.generate(mod, self.context())), [1, 2])
            return SimpleNamespace(diagnostics={})
        result = run()
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.diagnostics["hypothesis_cache"]["hits"], 1)
        run()
        self.assertEqual(len(calls), 2)

    def test_context_and_operator_prior_are_part_of_key(self):
        calls = []
        mod = self.module(lambda ctx: calls.append(ctx) or [])
        @C.scoped
        def run():
            a, b, c = self.context(), self.context(2), self.context()
            c.op_prior = {"rotate": 1.0}
            for ctx in (a, b, c):
                list(C.generate(mod, ctx))
        run()
        self.assertEqual(len(calls), 3)

    def test_expired_or_abandoned_stream_is_not_reused(self):
        calls = []
        mod = self.module(lambda ctx: calls.append(ctx) or [1, 2])
        @C.scoped
        def run():
            expired = self.context()
            expired.deadline = time.time()-1
            list(C.generate(mod, expired))
            stream = C.generate(mod, self.context())
            next(stream)
            stream.close()
            list(C.generate(mod, self.context()))
        run()
        self.assertEqual(len(calls), 3)

    def test_failed_stream_is_not_cached(self):
        calls = []
        def failing(ctx):
            calls.append(ctx)
            yield 1
            raise ValueError("failed")
        mod = self.module(failing)
        @C.scoped
        def run():
            for _ in range(2):
                with self.assertRaises(ValueError):
                    list(C.generate(mod, self.context()))
        run()
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()

