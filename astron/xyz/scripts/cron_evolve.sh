#!/usr/bin/env bash
# One cron tick of GABRIEL training.  One round, then exit.
#
# This is not a daemon.  Cron starts it, it does one bounded pass, it stops.
# If a previous tick is still running the lock is not free, and this tick exits
# 0 without doing anything -- overlapping ticks are a no-op by design, because
# one evolve round takes considerably longer than the cron interval.
#
# A tick is, in order:
#   1. harvest  -- run a rotating slice of fit tasks, keep the programs
#   2. train    -- refit the language model, adopt it only if perplexity fell
#   3. evolve   -- one gated round of bench/evolve.py (the long part)
#
# The two short steps run first so that a long evolve round can never starve
# them.  Every step is skippable by environment variable; every step is
# bounded; nothing here deletes policy/policy.json or policy/lineage.jsonl.
#
# Environment:
#   ASTRA_ROOT             install path            (default: the path below)
#   ASTRA_JOBS             worker processes        (default: min(4, nproc))
#   ASTRA_BUDGET           evolve seconds/task     (default: 14)
#   ASTRA_EVOLVE=0         skip the evolve round
#   GABRIEL_TRAIN=0        skip the language-model steps
#   GABRIEL_HARVEST_LIMIT  tasks per harvest       (default: 24)
#   GABRIEL_HARVEST_BUDGET seconds/task            (default: 8)
#   GABRIEL_TRAIN_SECONDS  training cap            (default: 120)

set -euo pipefail

ROOT="${ASTRA_ROOT:-/root/astraarcengine-improved/astra}"
LOCK="/tmp/astra-evolve.lock"
LOG_MAX=$((50 * 1024 * 1024))

BUDGET="${ASTRA_BUDGET:-14}"
HARVEST_LIMIT="${GABRIEL_HARVEST_LIMIT:-24}"
HARVEST_BUDGET="${GABRIEL_HARVEST_BUDGET:-8}"
TRAIN_SECONDS="${GABRIEL_TRAIN_SECONDS:-120}"
RUN_EVOLVE="${ASTRA_EVOLVE:-1}"
RUN_GABRIEL="${GABRIEL_TRAIN:-1}"

cd "$ROOT"
mkdir -p logs policy evidence queue predictions
LOG="$ROOT/logs/evolve.log"

# ---- worker count -----------------------------------------------------
# Fits a 4 vCPU box; falls back on a 2 vCPU / 2 GB box rather than thrashing.
CPUS="$(nproc 2>/dev/null || echo 1)"
JOBS="${ASTRA_JOBS:-4}"
[ "$JOBS" -gt "$CPUS" ] && JOBS="$CPUS"
MEM_KB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
if [ "$MEM_KB" -gt 0 ] && [ "$MEM_KB" -lt 1572864 ]; then      # under 1.5 GB
  JOBS=1
elif [ "$MEM_KB" -gt 0 ] && [ "$MEM_KB" -lt 3145728 ]; then    # under 3 GB
  [ "$JOBS" -gt 2 ] && JOBS=2
fi
[ "$JOBS" -lt 1 ] && JOBS=1

# ---- logging ----------------------------------------------------------
rotate() {
  local size
  size="$(stat -c %s "$LOG" 2>/dev/null || echo 0)"
  if [ "$size" -gt "$LOG_MAX" ]; then
    mv -f "$LOG" "$LOG.1"
    : > "$LOG"
  fi
}

stamp() {
  # Timestamp every line.  Deliberately not `ts` (moreutils) so the script has
  # no dependency outside coreutils and the venv.
  while IFS= read -r line; do
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$line"
  done
}

say() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG"; }

rotate
touch "$LOG"

# ---- one tick at a time ----------------------------------------------
exec 9>"$LOCK"
if ! flock -n 9; then
  say "[tick] a previous round still holds $LOCK -- skipping"
  exit 0
fi

# shellcheck disable=SC1091
if [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
else
  say "[tick] no venv at $ROOT/.venv -- using system python3"
fi

PY="$(command -v python3)"
say "[tick] start root=$ROOT jobs=$JOBS budget=$BUDGET python=$PY"
START="$(date -u +%s)"

run_step() {
  # run_step <label> <args...>   -- never aborts the tick; logs the exit code
  local label="$1"; shift
  local rc=0
  say "[$label] $*"
  set +e
  nice -n 10 "$PY" "$@" 2>&1 | stamp >> "$LOG"
  rc="${PIPESTATUS[0]}"
  set -e
  say "[$label] exit=$rc"
  return 0
}

if [ "$RUN_GABRIEL" = "1" ]; then
  run_step harvest -m gabriel.harvest \
    --limit "$HARVEST_LIMIT" --budget "$HARVEST_BUDGET" --jobs "$JOBS"
  run_step train -m gabriel.train --max-seconds "$TRAIN_SECONDS" --quiet
fi

if [ "$RUN_EVOLVE" = "1" ]; then
  run_step evolve bench/evolve.py --rounds 1 --budget "$BUDGET" --jobs "$JOBS"
  # evolve.py rewrites policy/lineage.jsonl from scratch every run, so a tick's
  # rounds would otherwise be invisible by the next tick.  Archive, never
  # delete: policy/policy.json and policy/lineage.jsonl are left exactly as
  # evolve.py wrote them.
  if [ -f "$ROOT/policy/lineage.jsonl" ]; then
    cat "$ROOT/policy/lineage.jsonl" >> "$ROOT/logs/lineage.history.jsonl"
    say "[evolve] appended $(wc -l < "$ROOT/policy/lineage.jsonl") lineage row(s) to logs/lineage.history.jsonl"
  fi
fi

say "[tick] done in $(( $(date -u +%s) - START ))s"
exit 0
