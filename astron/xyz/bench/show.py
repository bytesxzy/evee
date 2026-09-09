"""Render a task as text, for inspecting failures."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "arc")
SYM = ".123456789"


def render(g):
    return [" ".join(SYM[v] if v < 10 else "?" for v in row) for row in g]


def show(tid, with_test=True):
    with open(os.path.join(DATA, tid + ".json")) as fh:
        d = json.load(fh)
    print("===", tid)
    pairs = [("train", p) for p in d["train"]]
    if with_test:
        pairs += [("test", p) for p in d["test"]]
    for kind, p in pairs:
        a, b = render(p["input"]), render(p["output"])
        wa = max(len(x) for x in a)
        n = max(len(a), len(b))
        print("-- %s  %dx%d -> %dx%d" % (kind, len(p["input"]), len(p["input"][0]),
                                         len(p["output"]), len(p["output"][0])))
        for i in range(n):
            l = a[i] if i < len(a) else ""
            r = b[i] if i < len(b) else ""
            print("  %-*s   |  %s" % (wa, l, r))


if __name__ == "__main__":
    for t in sys.argv[1:]:
        show(t)
