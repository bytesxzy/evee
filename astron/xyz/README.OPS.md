# Running GABRIEL's training on a Namecheap VPS

One cron tick = one bounded training round, then the process exits.  There is
no daemon, nothing listens on a port, and nothing outside this box is called.

## Install

Target: a Namecheap **KVM VPS** (user-responsible plan), Ubuntu 24.04, root over
SSH on port 22.  Quasar-class (4 vCPU / 6 GB) is comfortable; 2 vCPU / 2 GB
works, and the tick script drops its worker count on its own when it sees the
smaller box.  Namecheap **Shared/Stellar cPanel hosting is out of scope** --
there is no "Setup Python App" step here and nothing goes in `public_html`.

```bash
scp astraarcengineimproved.zip root@YOUR_VPS_IP:/root/
ssh root@YOUR_VPS_IP
cd /root && unzip -q astraarcengineimproved.zip
cd /root/astraarcengine-improved/astra
bash scripts/install_namecheap.sh
```

The installer is idempotent.  It installs `python3 python3-venv python3-pip
unzip tmux`, creates `.venv`, creates `policy/ logs/ queue/ predictions/`,
smoke-tests the imports, and installs one crontab line -- replacing only its
own, leaving the rest of your crontab alone.

The line it installs:

```cron
*/6 * * * * /root/astraarcengine-improved/astra/scripts/cron_evolve.sh
```

Every six minutes as you asked.  `scripts/crontab.line` explains why the
interval is not what bounds the work: the tick takes `/tmp/astra-evolve.lock`
with `flock -n`, so a tick that finds a round already running logs one line and
exits 0.  One evolve round takes roughly **20-60 minutes** on 4 vCPU, so most
six-minute ticks are deliberate no-ops and the schedule really means "start the
next round as soon as the last one finishes".  `*/15` is equally correct.

## What one tick does

```
1. harvest   python3 -m gabriel.harvest --limit 24 --budget 8 --jobs $JOBS
2. train     python3 -m gabriel.train --max-seconds 120 --quiet
3. evolve    python3 bench/evolve.py --rounds 1 --budget 14 --jobs $JOBS
```

All three run at `nice 10` so a web stack on the same VPS keeps its CPU, and
`$JOBS` is `min(4, nproc)`, dropped to 2 under 3 GB of RAM and to 1 under
1.5 GB.  The two short steps run first so a long evolve round cannot starve
them.  Skip either half with `ASTRA_EVOLVE=0` or `GABRIEL_TRAIN=0`.

## Where the training lives

| File | Written by | What it is |
| --- | --- | --- |
| `policy/policy.json` | `bench/evolve.py` | the symbolic policy in force: solver priors, operator bias, up to 24 mined abstractions |
| `policy/gabriel_lm.json` | `gabriel/train.py` | the language model in force |
| `policy/lineage.jsonl` | `bench/evolve.py` | rounds of the **most recent** evolve run only -- it is rewritten each run |
| `logs/lineage.history.jsonl` | the tick script | every evolve round ever, appended after each tick |
| `policy/gabriel_lineage.jsonl` | `gabriel/train.py` | every language-model round, appended, with the perplexity that decided it |
| `evidence/evolution.json` | `bench/evolve.py` | the last evolve run's summary |
| `evidence/gabriel_lm.json` | `gabriel/train.py` | the last training pass's summary |
| `evidence/gabriel-harvest-*.json` | `gabriel/harvest.py` | run records, which are the model's training data; pruned to 40 files |

The tick script never deletes `policy/policy.json` or `policy/lineage.jsonl`.
It copies lineage rows out before the next run overwrites them, which is the
only reason `logs/lineage.history.jsonl` exists.

## Confirming a round actually happened

```bash
crontab -l                                   # the schedule
tail -f logs/evolve.log                      # a tick, live
cat policy/lineage.jsonl                     # the newest evolve round
tail -3 logs/lineage.history.jsonl           # every evolve round
tail -3 policy/gabriel_lineage.jsonl         # every language-model round
python3 -c "import json;print(json.load(open('evidence/evolution.json'))['lineage'][-1])"
```

An evolve round prints one JSON line with `wins`, `losses`, `p` and
`accepted`.  `accepted: false` is a normal, healthy outcome -- it means the
candidate policy failed the sign test and the incumbent was kept.  A language
model round prints `adopted` with the dev perplexity that decided it.  Rounds
that change nothing are the loop working, not the loop broken.

## Shared hosting instead of a VPS

`scripts/cron_shared.sh` is the same tick sized for a cPanel account: one
worker, a `mkdir` lock instead of `flock`, a 5 MB log cap, an interpreter
lookup (the system `python3` on a cPanel box is often too old -- 3.8 is the
floor), and **no evolve round**, because a 20-60 minute job across four cores
is not something to run on a machine you share.  What it does run is the half
that accumulates across ticks anyway: harvest a few tasks, refit the language
model, adopt it only if perplexity fell.  About 30-90 seconds per tick.

    */15 * * * * /home/YOURUSER/astraarcengine-improved/astra/scripts/cron_shared.sh >/dev/null 2>&1

Evolve rounds belong on hardware you own.  `policy/` and `evidence/` are plain
JSON, so moving results between the two is a file copy.

## Stopping

```bash
bash scripts/install_namecheap.sh --uninstall   # removes only its cron line
```

Or `crontab -e` and delete the line by hand.  To stop a round that is already
running, kill it and drop the lock:

```bash
pkill -f 'bench/evolve.py' ; pkill -f 'gabriel\.' ; rm -f /tmp/astra-evolve.lock
```

Nothing in `policy/` or `evidence/` is removed by uninstalling; delete those by
hand if you genuinely want the training thrown away.

## Namecheap terms

Written for a KVM VPS on a user-responsible plan.  Namecheap's own AUP and
Terms are the authority; check them for your plan rather than trusting this
list.  What this workload does, so you can check it against them:

* **Cron frequency.** The shared-hosting rules (one run per five minutes at
  most, a small cap on concurrent jobs) apply to cPanel/Stellar, not to a VPS.
  The `*/6` line stays above that five-minute floor anyway, and `flock` means
  exactly one job is ever running.
* **Resource use.** `nice 10`, workers capped at `min(4, nproc)` and reduced on
  small boxes, one instance at a time.  It is a CPU-bound batch job that yields
  to everything else on the machine.
* **Disk.** Logs rotate at 50 MB keeping one old file; harvested evidence is
  pruned to 40 files.  Nothing grows without a bound.
* **Network.** After the installer's `apt-get`, the training loop makes no
  network requests at all.  No inbound service, no open port, no proxy, no P2P,
  no mining, no bulk mail, no scraping.
* **Data.** Public ARC task JSON that ships in `data/arc/`.  Nothing is fetched
  and nothing is redistributed.

## Two things this is not

**It is not an ARC-AGI-3 score.** ARC-AGI-3 is interactive -- an agent takes
actions in an environment over time.  This engine is static: it reads a task,
synthesises a program, and returns grids.  It cannot be scored on ARC-AGI-3 at
all, and no number produced here should be presented as one.

**The numbers in `RESULTS.md` are measured on the public task files bundled in
`data/arc/`,** not on any hidden evaluation set, and they are not official.
Training here shifts the engine's behaviour on the fit split of those same
public files; treat a change on the fit split as a hypothesis, and the holdout
split -- which is never trained on and never gates anything -- as the only
number worth quoting.
