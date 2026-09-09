#!/usr/bin/env bash
# Install GABRIEL's training cron on a Namecheap KVM VPS (Ubuntu 24.04, root).
#
# Idempotent: run it as many times as you like.  It installs OS packages if
# they are missing, creates the virtualenv if it is missing, creates the
# working directories if they are missing, and replaces *only* its own line in
# the crontab -- any other cron job you have stays exactly where it is.
#
#   bash scripts/install_namecheap.sh              # install and schedule
#   bash scripts/install_namecheap.sh --no-cron    # set up, schedule nothing
#   bash scripts/install_namecheap.sh --uninstall  # remove the cron line only
#
# This is for a VPS you administer (Namecheap's "user-responsible" KVM plans).
# It is not for Namecheap Shared/Stellar cPanel hosting: there is no Setup
# Python App step here, nothing is served from public_html, and no CGI is
# involved.  Nothing in this repository listens on a port.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
MARKER="# GABRIEL-EVOLVE (astra) -- managed by scripts/install_namecheap.sh"
INTERVAL="${GABRIEL_CRON_INTERVAL:-*/6}"
DO_CRON=1
UNINSTALL=0

for arg in "$@"; do
  case "$arg" in
    --no-cron)   DO_CRON=0 ;;
    --uninstall) UNINSTALL=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

info() { printf '\033[1m==>\033[0m %s\n' "$*"; }

crontab_without_ours() {
  crontab -l 2>/dev/null | grep -v -F "$MARKER" | grep -v -F "cron_evolve.sh" || true
}

if [ "$UNINSTALL" = "1" ]; then
  info "removing the GABRIEL cron line (nothing else is touched)"
  crontab_without_ours | crontab -
  crontab -l 2>/dev/null || true
  echo
  echo "Removed.  policy/, evidence/ and logs/ are untouched; delete them by hand"
  echo "if you actually want the training thrown away."
  exit 0
fi

# ---- 1. packages ------------------------------------------------------
# Stdlib only -- there is no pip install step anywhere in this repository, and
# python3-pip is here purely so the venv can bootstrap itself.
if [ "$(id -u)" = "0" ]; then
  info "installing OS packages (python3 python3-venv python3-pip unzip tmux)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq python3 python3-venv python3-pip unzip tmux
else
  info "not root: skipping apt-get (install python3-venv yourself if missing)"
fi

# ---- 2. virtualenv ----------------------------------------------------
if [ ! -x "$ROOT/.venv/bin/python3" ]; then
  info "creating virtualenv at $ROOT/.venv"
  python3 -m venv "$ROOT/.venv"
else
  info "virtualenv already present at $ROOT/.venv"
fi
"$ROOT/.venv/bin/python3" --version

# ---- 3. working directories -------------------------------------------
info "creating working directories"
mkdir -p "$ROOT/policy" "$ROOT/logs" "$ROOT/queue" "$ROOT/predictions" "$ROOT/evidence"
chmod +x "$ROOT/scripts/cron_evolve.sh"

# ---- 4. smoke test ----------------------------------------------------
info "checking the engine imports and the model loads"
( cd "$ROOT" && "$ROOT/.venv/bin/python3" - <<'PY'
import sys
sys.path.insert(0, ".")
from engine import portfolio, learn          # noqa: F401
from gabriel import bind, corpus             # noqa: F401
from gabriel.lm import GabrielLM
lm = GabrielLM.load("policy/gabriel_lm.json")
print("engine: ok")
print("gabriel: ok  language model: %s" %
      ("not trained yet -- the first cron tick will train one" if lm is None
       else "vocab=%d features=%d dev_perplexity=%s"
            % (len(lm.vocab), len(lm.W), lm.meta.get("dev_perplexity"))))
PY
)

# ---- 5. cron ----------------------------------------------------------
LINE="$INTERVAL * * * * $ROOT/scripts/cron_evolve.sh"
if [ "$DO_CRON" = "1" ]; then
  info "installing the cron line (replacing any previous GABRIEL/ASTRA one)"
  { crontab_without_ours; echo "$MARKER"; echo "$LINE"; } | crontab -
else
  info "--no-cron given; the line you would install is:"
  echo "    $LINE"
fi

cat <<EOF

$(info "installed")

  cron line        $LINE
  tick script      $ROOT/scripts/cron_evolve.sh
  lock             /tmp/astra-evolve.lock   (overlapping ticks exit 0)

How to check it is working
--------------------------

  crontab -l                                  # the schedule
  tail -f $ROOT/logs/evolve.log               # what a tick is doing
  tail -5 $ROOT/policy/gabriel_lineage.jsonl  # language-model rounds
  cat $ROOT/policy/lineage.jsonl              # the newest evolve round
  tail -3 $ROOT/logs/lineage.history.jsonl    # every evolve round, ever
  python3 -c "import json;print(json.load(open('$ROOT/evidence/evolution.json'))['lineage'][-1])"

Run one tick by hand instead of waiting for cron:

  ASTRA_ROOT=$ROOT $ROOT/scripts/cron_evolve.sh

Stop training:

  bash $ROOT/scripts/install_namecheap.sh --uninstall

Resource behaviour, so this stays inside a fair-use policy
----------------------------------------------------------

  * one tick at a time, enforced by flock -- ticks never pile up;
  * every python process runs at nice 10, so anything else on the box wins;
  * workers are capped at min(4, nproc), and dropped to 2 under 3 GB of RAM
    and to 1 under 1.5 GB;
  * logs rotate at 50 MB, harvested evidence is pruned to a fixed count, so
    disk use is bounded rather than monotonically growing;
  * no inbound service, no port is opened, and after this installer finishes
    the training loop makes no network requests at all.

EOF
