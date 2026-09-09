"""Bounded, keyless harvest of OpenAssistant/oasst1 via the HF Dataset Viewer.

This is the **only** module in ASTRON that opens a socket, it runs offline (a
cron tick or a one-off command), and the browser never calls it.  What it
produces is a set of *fitted parameters and counts* -- never corpus text.  That
is both the anti-copy mechanism and the licensing answer: nothing from the
dataset is redistributed, so no assistant reply can ever be pasted back at a
user, by accident or otherwise.

Why OASST1
----------

The previous conversation path retrieved from a small canned YAML corpus and
returned the nearest reply verbatim.  OASST1 is human-written assistant-style
dialogue *with human quality annotations*, which is the part that matters here:
the annotations let a harvest keep only turns humans judged good, and the tree
structure (``message_id`` / ``parent_id`` / ``message_tree_id``) lets a harvest
keep conversational *transitions* rather than a bag of unrelated lines.

Discovery, not guessing
-----------------------

The config and split names are **discovered** from ``/splits`` at run time.
Hard-coding ``default``/``train`` would break silently the first time the
dataset was re-published under different names, and would also hide the case
where the dataset has moved.

Licence check, not licence assumption
-------------------------------------

``--require-license`` (on by default) reads the dataset's declared licence from
the Hub API and refuses to harvest unless it is in :data:`SAFE_LICENSES`.  At
the time of writing OASST1 declares Apache-2.0, which is commercially usable --
but a claim in a comment is not a check, and this product is B2B.  The same
guard is why DailyDialog is not wired in: the commonly distributed copies carry
CC BY-NC-SA terms, which is a licensing problem you cannot fix downstream.

What is kept
------------

Per the quality metadata OASST1 ships:

* ``lang == "en"``
* ``deleted == false``
* ``review_result == true`` when the field is present
* tree state in :data:`READY_STATES`
* ``rank == 0`` (the top-ranked sibling) where a ranking exists
* label thresholds: low ``toxicity``/``spam``, adequate ``quality``

Usage::

    python3 -m text.harvest_oasst --max-rows 4000 --out text/data/oasst_stats.json
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from . import conversation as CONV

DATASET = "OpenAssistant/oasst1"
VIEWER = "https://datasets-server.huggingface.co"
HUB = "https://huggingface.co/api/datasets"

# Licences this product may build a commercial corpus from.  A dataset outside
# this set is refused, not warned about.
SAFE_LICENSES = frozenset((
    "apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause",
    "cc0-1.0", "cc-by-4.0", "cc-by-3.0", "odc-by", "unlicense",
))

READY_STATES = frozenset(("ready_for_export", "prompt_lottery_waiting",
                          "ranking", "aborted_low_grade"))
GOOD_STATES = frozenset(("ready_for_export",))

USER_AGENT = "astron-text-harvest/1 (+offline; no redistribution)"


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read().decode("utf-8"))


# --------------------------------------------------------------------------
# discovery + licence
# --------------------------------------------------------------------------

def check_license(dataset=DATASET):
    """Return ``(license_id, allowed, raw)``.  Never assumes; always reads."""
    info = _get("%s/%s" % (HUB, urllib.parse.quote(dataset)))
    tags = info.get("tags") or []
    lic = (info.get("cardData") or {}).get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    if not lic:
        for t in tags:
            if isinstance(t, str) and t.startswith("license:"):
                lic = t.split(":", 1)[1]
                break
    lic = (lic or "").strip().lower()
    return lic, lic in SAFE_LICENSES, info


def discover_split(dataset=DATASET):
    """Ask the viewer which (config, split) pairs exist.  Do not guess."""
    j = _get("%s/splits?dataset=%s" % (VIEWER, urllib.parse.quote(dataset)))
    rows = j.get("splits") or []
    if not rows:
        raise RuntimeError("no splits reported for %s" % dataset)
    for want in ("train", "validation", "test"):
        for r in rows:
            if r.get("split") == want:
                return r.get("config"), r.get("split"), rows
    return rows[0].get("config"), rows[0].get("split"), rows


def fetch_rows(dataset, config, split, offset, length):
    url = ("%s/rows?dataset=%s&config=%s&split=%s&offset=%d&length=%d" %
           (VIEWER, urllib.parse.quote(dataset), urllib.parse.quote(config),
            urllib.parse.quote(split), offset, length))
    j = _get(url)
    return [r.get("row") or {} for r in (j.get("rows") or [])], j.get("num_rows_total")


# --------------------------------------------------------------------------
# quality filter
# --------------------------------------------------------------------------

def _label(row, name):
    """OASST1 ships labels as parallel ``labels.name`` / ``labels.value`` lists."""
    labels = row.get("labels")
    if isinstance(labels, dict):
        names = labels.get("name") or []
        values = labels.get("value") or []
        for n, v in zip(names, values):
            if n == name:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
    if isinstance(labels, list):
        for item in labels:
            if isinstance(item, dict) and item.get("name") == name:
                try:
                    return float(item.get("value"))
                except (TypeError, ValueError):
                    return None
    return None


def keep_row(row, strict=True):
    """Apply OASST1's own quality metadata.  Returns ``(keep, reason)``."""
    if row.get("lang") != "en":
        return False, "lang"
    if row.get("deleted"):
        return False, "deleted"
    rr = row.get("review_result")
    if rr is False:
        return False, "review_result"
    if strict and rr is None and (row.get("review_count") or 0) > 0:
        return False, "unreviewed"
    state = row.get("tree_state")
    if state and state not in READY_STATES:
        return False, "tree_state"
    if strict and state and state not in GOOD_STATES:
        return False, "tree_state_strict"
    tox = _label(row, "toxicity")
    if tox is not None and tox > 0.2:
        return False, "toxicity"
    spam = _label(row, "spam")
    if spam is not None and spam > 0.2:
        return False, "spam"
    qual = _label(row, "quality")
    if strict and qual is not None and qual < 0.5:
        return False, "quality"
    rank = row.get("rank")
    if row.get("role") == "assistant" and rank is not None and int(rank) != 0:
        return False, "not_top_ranked"
    text = str(row.get("text") or "").strip()
    if not text or len(text) > 4000:
        return False, "length"
    return True, "keep"


# --------------------------------------------------------------------------
# tree reconstruction
# --------------------------------------------------------------------------

def build_turns(rows):
    """Reconstruct ``(prompter utterance, assistant reply)`` adjacency.

    Conversation structure is the point of using this dataset, so the pairing
    goes through ``parent_id`` rather than through row order.  Rows arrive in
    pages and a parent may be in an earlier page than its child, so the index is
    built over everything harvested before any pairing happens.
    """
    by_id = {}
    for r in rows:
        mid = r.get("message_id")
        if mid:
            by_id[mid] = r
    turns = []
    for r in rows:
        if r.get("role") != "assistant":
            continue
        parent = by_id.get(r.get("parent_id"))
        if not parent or parent.get("role") != "prompter":
            continue
        grand = by_id.get(parent.get("parent_id"))
        turns.append({
            "prompt": str(parent.get("text") or ""),
            "reply": str(r.get("text") or ""),
            "prev_reply": str(grand.get("text") or "") if grand else "",
            "tree": r.get("message_tree_id"),
        })
    return turns


# --------------------------------------------------------------------------
# the harvest
# --------------------------------------------------------------------------

def harvest(max_rows=4000, page=100, dataset=DATASET, require_license=True,
            strict=True, verbose=True, sleep=0.2):
    """Fetch, filter, pair, and reduce to statistics.  Returns a report dict."""
    lic, allowed, _info = check_license(dataset)
    if require_license and not allowed:
        raise SystemExit(
            "refusing to harvest %s: declared license %r is not in the "
            "commercial-safe allowlist %s.  Pass --no-require-license only if "
            "you have separately cleared the terms." %
            (dataset, lic or "unknown", sorted(SAFE_LICENSES)))
    config, split, all_splits = discover_split(dataset)
    if verbose:
        print("dataset %s  license=%s  config=%s  split=%s" %
              (dataset, lic, config, split), file=sys.stderr)

    kept, dropped, offset, total = [], {}, 0, None
    while len(kept) < max_rows:
        want = min(page, max_rows - len(kept) + page)
        rows, total = fetch_rows(dataset, config, split, offset, min(100, want))
        if not rows:
            break
        offset += len(rows)
        for r in rows:
            ok, why = keep_row(r, strict=strict)
            if ok:
                kept.append(r)
            else:
                dropped[why] = dropped.get(why, 0) + 1
        if verbose and offset % 1000 < 100:
            print("  scanned %d, kept %d" % (offset, len(kept)), file=sys.stderr)
        if total and offset >= total:
            break
        if sleep:
            time.sleep(sleep)

    turns = build_turns(kept)
    stats = CONV.statistics_from_turns(turns)
    return {
        "dataset": dataset, "license": lic, "license_allowed": allowed,
        "config": config, "split": split,
        "splits_seen": [{"config": s.get("config"), "split": s.get("split")}
                        for s in all_splits],
        "scanned": offset, "kept": len(kept), "dropped": dropped,
        "turns": len(turns), "total_rows_reported": total,
        "harvested_at": int(time.time()),
        "statistics": stats,
        "note": ("Only derived statistics are stored.  No message text from "
                 "the dataset is written to this file or shipped to any "
                 "browser."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--max-rows", type=int, default=4000)
    ap.add_argument("--page", type=int, default=100)
    ap.add_argument("--loose", action="store_true",
                    help="keep rows the strict quality filter would drop")
    ap.add_argument("--no-require-license", dest="require_license",
                    action="store_false")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "oasst_stats.json"))
    a = ap.parse_args()
    try:
        report = harvest(max_rows=a.max_rows, page=a.page, dataset=a.dataset,
                         require_license=a.require_license, strict=not a.loose)
    except (urllib.error.URLError, OSError) as exc:
        # Graceful degradation is a rule, not an aspiration: a failed harvest
        # leaves the previous artifact exactly where it was.
        print(json.dumps({"ok": False, "error": repr(exc)[:300],
                          "note": "previous artifact left untouched"}, indent=2))
        return 1
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    summary = {k: v for k, v in report.items() if k != "statistics"}
    summary["statistics_keys"] = sorted(report["statistics"].keys())
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
