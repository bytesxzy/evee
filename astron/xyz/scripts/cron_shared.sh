#!/usr/bin/env bash
# One GABRIEL tick sized for cPanel shared hosting.  Minutes, not hours.
#
# scripts/cron_evolve.sh is written for a VPS you own: it runs a full evolve
# round, which takes 20-60 minutes across four cores.  On shared hosting that
# job gets killed, and it would be spending a machine you share with other
# people.  So this script does not run evolve at all by default.
#
# What it does instead is the half that actually accumulates across ticks:
# harvest a few tasks, then refit the language model and adopt it only if
# held-out perplexity fell.  One tick is roughly 30-90 seconds on one core.
# Evolve rounds, when you want them, belong on a machine you own -- copy
# policy/ and evidence/ back and forth, they are just JSON.
#
# Differences from the VPS script, all forced by the environment:
#   * one worker, always -- shared hosting caps concurrent processes;
#   * a mkdir lock instead of flock, which may not be installed;
#   * a 5 MB log cap instead of 50 MB, because disk quota is small;
#   * the interpreter is looked up, because the system python3 on a cPanel box
#     is often too old and the real one lives in a Setup Python App virtualenv.
#
# Environment:
#   GABRIEL_ROOT     install path   (default: the path below)
#   GABRIEL_PYTHON   interpreter    (default: newest python3.x on PATH)
#   GABRIEL_LIMIT    tasks per tick (default: 4)
#   GABRIEL_BUDGET   seconds/task   (default: 6)
#   GABRIEL_SECONDS  training cap   (default: 60)
#   GABRIEL_EVOLVE=1 run an evolve round anyway (expect it to be killed)

set -euo pipefail

ROOT="${GABRIEL_ROOT:-$HOME/astraarcengine-improved/astra}"
LOCK="${TMPDIR:-/tmp}/gabriel-$(id -u).lock"
LOG_MAX=$((5 * 1024 * 1024))
LIMIT="${GABRIEL_LIMIT:-4}"
BUDGET="${GABRIEL_BUDGET:-6}"
SECONDS_CAP="${GABRIEL_SECONDS:-60}"
RUN_EVOLVE="${GABRIEL_EVOLVE:-0}"

cd "$ROOT"
mkdir -p logs policy evidence
LOG="$ROOT/logs/evolve.log"

# ---- interpreter ------------------------------------------------------
# math.comb needs 3.8.  The system python3 on a cPanel box is often 3.6, so
# prefer an explicit one, then the newest versioned binary on PATH.
find_python() {
  if [ -n "${GABRIEL_PYTHON:-}" ]; then echo "$GABRIEL_PYTHON"; return; fi
  for v in 3.13 3.12 3.11 3.10 3.9 3.8; do
    if command -v "python$v" >/dev/null 2>&1; then command -v "python$v"; return; fi
  done
  command -v python3 || command -v python
}
PY="$(find_python || true)"

# ---- logging ----------------------------------------------------------
now() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say() { printf '%s %s\n' "$(now)" "$*" >> "$LOG"; }
stamp() { while IFS= read -r line; do printf '%s %s\n' "$(now)" "$line"; done; }

touch "$LOG"
size="$(wc -c < "$LOG" 2>/dev/null || echo 0)"
if [ "$size" -gt "$LOG_MAX" ]; then mv -f "$LOG" "$LOG.1"; touch "$LOG"; fi

# ---- one tick at a time ----------------------------------------------
# mkdir is atomic everywhere; flock is not always installed on shared hosting.
# A lock older than two hours is a crashed tick, not a running one.
if [ -d "$LOCK" ]; then
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +120 2>/dev/null)" ]; then
    say "[tick] clearing a stale lock at $LOCK"
    rmdir "$LOCK" 2>/dev/null || true
  else
    say "[tick] a previous tick still holds $LOCK -- skipping"
    exit 0
  fi
fi
mkdir "$LOCK" 2>/dev/null || { say "[tick] lost the race for $LOCK -- skipping"; exit 0; }
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

if [ -z "$PY" ]; then
  say "[tick] no python interpreter found -- set GABRIEL_PYTHON in the cron line"
  exit 0
fi

VER="$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "?")"
case "$VER" in
  3.8|3.9|3.1[0-9]) ;;
  *) say "[tick] $PY is Python $VER; 3.8 or newer is required -- set GABRIEL_PYTHON"
     exit 0 ;;
esac

say "[tick] start root=$ROOT python=$PY ($VER) limit=$LIMIT budget=$BUDGET"
START="$(date -u +%s)"

run_step() {
  local label="$1"; shift
  local rc=0
  say "[$label] $*"
  set +e
  nice -n 19 "$PY" "$@" 2>&1 | stamp >> "$LOG"
  rc="${PIPESTATUS[0]}"
  set -e
  say "[$label] exit=$rc"
  return 0
}

run_step harvest -m gabriel.harvest --limit "$LIMIT" --budget "$BUDGET" \
  --jobs 1 --keep 20
run_step train -m gabriel.train --max-seconds "$SECONDS_CAP" --quiet

if [ "$RUN_EVOLVE" = "1" ]; then
  say "[evolve] shared hosting will probably kill this; run it on your own machine"
  run_step evolve bench/evolve.py --rounds 1 --budget "$BUDGET" --jobs 1
  if [ -f "$ROOT/policy/lineage.jsonl" ]; then
    cat "$ROOT/policy/lineage.jsonl" >> "$ROOT/logs/lineage.history.jsonl"
  fi
fi

say "[tick] done in $(( $(date -u +%s) - START ))s"
exit 0
