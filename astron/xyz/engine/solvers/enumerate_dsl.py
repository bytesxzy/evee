"""Deep enumeration -- the general fallback when no specialist fires.

Depth 4 with observational-equivalence pruning plus a binary combination layer.
Consumes whatever time budget the orchestrator has left, which is why it is
registered last.
"""

import time

from .. import enum_core
from ..task import Hyp

SOLVER = "enumerate"
PHASE = 2


def generate(ctx):
    dl = ctx.deadline or (time.time() + 10.0)
    found = enum_core.search(ctx, depth=4, max_states=1500, deadline=dl,
                             level="full", use_binary=True,
                             prior=getattr(ctx, "op_prior", None))
    return [Hyp(n, f, 3.0 + c, SOLVER) for n, c, f in found]
