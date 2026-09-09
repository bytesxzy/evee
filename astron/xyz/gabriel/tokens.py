"""The token language.

The engine names every hypothesis it builds.  Those names are not free text --
they are a small, regular language written by the solvers themselves::

    crop(rot90($))                  an enumerator chain
    T@fractal#0                     a transposed fractal rule, colour 0
    paint#1($)>>c4@0.by_size_rank   a paint composed with an object ranking
    sep#5.and->2                    a separator rule with a boolean merge

Tokenising that language is what makes a language model over it possible.  The
scanner below is deliberately structural rather than clever: identifiers,
integers, and the punctuation the solvers use as combinators.  ``#`` stays
attached to nothing -- ``crop#3`` becomes ``crop`` ``#`` ``3`` -- so the model
can learn that ``crop`` takes a colour argument without having to see every
colour separately.

A training sequence is a *prompt* followed by a *program*:

    <sig>shape:same  <sig>pal:same ... <bos> <fam>geometry  rot90 ( $ ) <eos>

The prompt tokens are the task signatures from :func:`engine.learn.signatures`,
which are computed from training pairs only.  The family is emitted as the
first real token, which is what makes ``P(family | signature)`` fall out of the
model directly instead of needing a second head.
"""

import re

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SIG = "<sig>"          # prefix for prompt tokens
FAM = "<fam>"          # prefix for solver-family tokens

_SCAN = re.compile(r"""
    (?P<sym>->|>>|<>|[()\[\],@~#.$+\-*/|>:^=!&])
  | (?P<num>\d+)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<ws>\s+)
""", re.X)

MAX_TOKENS = 48        # programs longer than this are pathological; drop them


def tokenize(program):
    """Split a hypothesis name into tokens.  Unrecognised bytes are dropped."""
    out = []
    pos, n = 0, len(program)
    while pos < n:
        m = _SCAN.match(program, pos)
        if not m:
            pos += 1
            continue
        pos = m.end()
        if m.lastgroup == "ws":
            continue
        out.append(m.group())
        if len(out) > MAX_TOKENS:
            return out[:MAX_TOKENS]
    return out


def sig_token(s):
    return SIG + s


def fam_token(f):
    return FAM + f


def is_fam(tok):
    return tok.startswith(FAM)


def fam_name(tok):
    return tok[len(FAM):]


def sequence(sigs, family, program):
    """(prompt tokens, target tokens) for one training example."""
    prompt = [sig_token(s) for s in sigs]
    body = [fam_token(family)] if family else []
    body += tokenize(program) + [EOS]
    return prompt, body


# --------------------------------------------------------------------------
# chains
# --------------------------------------------------------------------------

def chain_tokens(ops):
    """Token form of an operator chain, in the notation the enumerator uses.

    ``["rot90", "crop"]`` -> tokens of ``crop(rot90($))``.  Application order
    in, source order out, matching :func:`engine.learn.parse_chain`.
    """
    return tokenize(chain_name(ops))


def chain_name(ops):
    name = "$"
    for op in ops:
        name = "%s(%s)" % (op, name)
    return name


def op_tokens(op):
    """Tokens an operator name contributes, e.g. ``fill#3`` -> fill, #, 3."""
    return tokenize(op)
