"""GABRIEL -- a language model strapped to a symbolic ARC reasoner.

GABRIEL is the whole system: the ASTRA symbolic engine (``engine/``) with a
language model over the engine's *own* program language bolted onto every
place where the engine has to make a choice.

Nothing here calls an external service.  There is no API key, no remote model,
no network access at any point.  The language model is trained from scratch, in
this process, out of the programs the engine itself has synthesised, and it is
a few hundred kilobytes of JSON.  This is a hard requirement, not a style
preference: consulting an outside model would invalidate every ARC number the
repository reports.

The three places the model is strapped in
----------------------------------------

1. **Which family to believe.**  The model emits the solver family as the first
   token after the task's signature prompt, so ``P(family | signature)`` is a
   real distribution the model produces, and it is fed into the engine's
   ranking prior through :class:`gabriel.bind.GabrielPolicy` -- the same
   ``Policy`` interface ``engine/learn.py`` already defines.

2. **Which operator to try next.**  The model's expected token distribution for
   a task becomes ``op_bias``, which the enumerator uses both to order its
   operator list and to keep nodes in its beam.  In a search truncated by a
   state cap, what is explored first is what is explored at all.

3. **Which programs to write.**  :mod:`gabriel.proposer` decodes whole operator
   chains from the model under a grammar constrained to operators that exist
   for this task, and hands them to the engine as ordinary hypotheses.  The
   engine then does what it always does: rejects any program that does not
   reproduce every training pair.  The model proposes; the verifier disposes.

The model never sees a test output.  It is trained on program text and on task
signatures computed from training pairs only, and at solve time it is handed
the same :class:`engine.task.Ctx` every other solver gets.
"""

__all__ = ["NAME", "VERSION", "ENGINE"]

NAME = "GABRIEL"
VERSION = "1.0"
ENGINE = "ASTRA 3.0"
