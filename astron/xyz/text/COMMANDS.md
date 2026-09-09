# Exact commands used

Every number in [RESULTS.md](RESULTS.md) came from one of these, run in this
checkout, on a 4 vCPU Linux box with Python 3.11.15 and Node 22.22.2.  Nothing
is estimated and nothing is carried over from another machine.

Paths below assume the repository root (`robots_topic_coherent.html` and
`astron/` are siblings).

---

## ASTRA / GABRIEL — the existing suite still passes

`astron/xyz/tests/` has no `__init__.py`, so `unittest discover` cannot load it
(see [BUGS.md](BUGS.md) #21).  The invocation the project's own README uses is:

```bash
cd astron/xyz
python3 -m unittest tests.test_gabriel tests.test_engine tests.test_search \
                    tests.test_boundaries tests.test_motion tests.test_portfolio \
                    tests.test_cli tests.test_regression
```

## The text sibling's own tests

```bash
cd astron/xyz
python3 -m unittest tests.test_text -v
```

## Fitting and exporting the browser artifacts

```bash
cd astron/xyz
python3 -m text.train --dry-run     # fit + gate report, writes nothing
python3 -m text.train               # fit, gate, write policy/ if adopted
python3 -m text.export              # refit + write both artifacts
```

With an OASST1 harvest (needs network — see [DATA.md](DATA.md) §5):

```bash
cd astron/xyz
python3 -m text.harvest_oasst --max-rows 4000 --out text/data/oasst_stats.json
python3 -m text.export --stats text/data/oasst_stats.json
```

## Gates: the identity ladder, the leak audit, the conformance vectors

```bash
cd astron/xyz
python3 -m text.bench                       # all three
python3 -m text.bench --ladder              # 33 adversarial pairs
python3 -m text.bench --audit               # runtime must not contain the benchmark
python3 -m text.bench --conformance         # write text/browser/conformance.json
```

## Browser runtime tests

```bash
node astron/xyz/text/browser/runtime_tests.js robots_topic_coherent.html
```

Drives a real DOM, a real `fetch`, real timers and a real `AbortController`
against fixture responses, including every failure path: no artifact, malformed
artifact, Wikipedia down, title resolver down, a source that hangs, a corrupt
cache, repeated requests, empty and strange input, and 400 upvotes pushed onto
the wrong sense.

## Python ↔ JavaScript conformance

```bash
cd astron/xyz && python3 -m text.bench --conformance && cd ../..
node astron/xyz/text/browser/conformance_check.js \
     robots_topic_coherent.html \
     astron/xyz/text/browser/conformance.json
```

## The adversarial sense benchmark, baseline vs revision

```bash
# tuning split -- used while implementing
node astron/xyz/text/browser/bench.js \
     --a robots_topic_coherent.baseline.html \
     --b robots_topic_coherent.html \
     --split tune

# held-out split -- run once, after the implementation was frozen
node astron/xyz/text/browser/bench.js \
     --a robots_topic_coherent.baseline.html \
     --b robots_topic_coherent.html \
     --split hold

# everything, with a per-case JSON dump and a win/loss list
node astron/xyz/text/browser/bench.js \
     --a robots_topic_coherent.baseline.html \
     --b robots_topic_coherent.html \
     --split all --json bench_all.json --verbose
```

## ARC, 550 public tasks, before and after the counting solver

```bash
cd astron/xyz

# baseline: with engine/solvers/counting.py NOT registered in portfolio.py
python3 bench/run_arc.py --out evidence/baseline_arc_full.json \
        --budget 20 --jobs 4 --quiet

# after: with it registered
python3 bench/run_arc.py --out evidence/counting_arc_full.json \
        --budget 20 --jobs 4 --quiet
```

Paired comparison and sign test:

```bash
cd astron/xyz
python3 - <<'PY'
import json
from math import comb
b = json.load(open('evidence/baseline_arc_full.json'))
c = json.load(open('evidence/counting_arc_full.json'))
bb = {r['id']: r['solved'] for r in b['per_task']}
cc = {r['id']: r['solved'] for r in c['per_task']}
w = sorted(i for i in bb if cc.get(i, 0) > bb[i])
l = sorted(i for i in bb if cc.get(i, 0) < bb[i])
n = len(w) + len(l)
p = sum(comb(n, i) for i in range(len(w), n + 1)) / 2.0 ** n if n else 1.0
print('solved %d -> %d   wins %d  losses %d  p=%.4g' %
      (b['solved'], c['solved'], len(w), len(l), p))
print('wins:', w)
print('losses:', l)
PY
```

## Everything, in one pass

```bash
cd astron/xyz && \
  python3 -m unittest tests.test_gabriel tests.test_engine tests.test_search \
                      tests.test_boundaries tests.test_motion tests.test_portfolio \
                      tests.test_cli tests.test_regression && \
  python3 -m unittest tests.test_text && \
  python3 -m text.export && \
  python3 -m text.bench && \
  cd ../.. && \
  node astron/xyz/text/browser/conformance_check.js robots_topic_coherent.html \
       astron/xyz/text/browser/conformance.json && \
  node astron/xyz/text/browser/runtime_tests.js robots_topic_coherent.html
```
