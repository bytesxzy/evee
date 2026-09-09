"""ASTRON's text sibling.

GABRIEL is a sparse log-linear model over the ARC *program* language.  This
package is its sibling over natural-language *queries*: the same architecture --
signatures, a conditional log-linear scorer, competing hypothesis families,
constrained decoding, verification before acceptance, a held-out gate -- applied
to the question "what concept did the person actually ask about?".

It shares no vocabulary with GABRIEL and imports nothing from ``gabriel`` or
``engine``.  ARC tokens are ARC tokens; English is English.  What is shared is
the *discipline*:

    model proposes  ->  verifier disposes

Nothing in this package opens a socket at inference time.  ``harvest_oasst.py``
is the only module that touches the network, it is offline-only, and its output
is a set of fitted parameters -- never corpus text.
"""

__all__ = ["signatures", "families", "lm", "corpus", "train", "export", "bench"]
