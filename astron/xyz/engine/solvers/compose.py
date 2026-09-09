"""Fast shallow composition pass (depth 2, small op set).

Runs before the full enumerator so that short programs are found -- and ranked
-- cheaply.  Anything it finds, the deep enumerator would also find, but later
and at higher cost, which is exactly the ordering we want.
"""

import time

from .. import enum_core
from ..task import Hyp

SOLVER = "compose"
PHASE = 2


def generate(ctx):
    dl = ctx.deadline or (time.time() + 5.0)
    dl = min(dl, time.time() + 4.0)
    found = enum_core.search(ctx, depth=2, max_states=600, deadline=dl,
                             level="full", use_binary=False)
    return [Hyp(n, f, 2.0 + c, SOLVER) for n, c, f in found]
