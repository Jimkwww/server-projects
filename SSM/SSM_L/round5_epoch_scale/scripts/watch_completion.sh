#!/usr/bin/env bash
# Watcher v2: wait for the round5 finish script's sentinel file, then package + verify.
# Polls the FINISHED_V2.flag (written only by run_finish.sh after ALL steps), so it cannot
# fire on stale tmux scrollback.
ROOT=/home/user_wzk/wzk/projects/SSM/SSM_L
R5=$ROOT/round5_epoch_scale
FLAG="$R5/runs/downstream/FINISHED_V2.flag"
cd "$ROOT" || exit 1

log() { echo "[watcher $(date '+%F %T')] $*"; }

log "started. waiting for $FLAG ..."
i=0
while [ ! -f "$FLAG" ]; do
  i=$((i+1))
  if [ $((i % 10)) -eq 0 ]; then
    NF=$(find "$R5/runs/downstream" -name DONE.json -path '*finetune*' 2>/dev/null | wc -l)
    log "still waiting (finetune done: $NF/72) ..."
  fi
  sleep 180
done

log "finish sentinel seen."
sleep 15
log "running package_r5.py"
python "$R5/scripts/package_r5.py" 2>&1 | tail -3

P="$ROOT/deliverables/round5_epoch_scale_package.tar.gz"
log "verifying deliverables"
if [ -f "$P" ]; then
  log "package OK: $P ($(stat -c %s "$P") bytes)"
else
  log "ERROR: package missing at $P"
fi
log "reports:"
ls -1 "$R5/reports/" 2>/dev/null
log "figs count:"
ls -1 "$R5/figs/" 2>/dev/null | wc -l
log "watcher done"
