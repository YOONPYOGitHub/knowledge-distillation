#!/bin/bash
# usage: chain_after.sh <family> <gpus>
# Carries a family from Round 1 to the end: Round 2 by the frozen rule, final selection,
# then seeds 43 and 44 with the selected CE and KD settings. Safe to rerun after a restart:
# finished stages are skipped and an interrupted stage resumes from its checkpoints.
F=$1; G=$2
R=/root/runs/fineweb4way; L=$R/$F/logs
done_round() { [ -f "$R/$F/$1/selection.json" ]; }
wait_round() {  # wait while another session still runs this round, then require success
  until done_round "$1"; do
    pgrep -f "output-dir $R/$F/$1( |$)" >/dev/null || return 1
    sleep 120
  done
}
run() {  # name trials-file [extra args]
  local name=$1 trials=$2; shift 2
  done_round "$name" && return 0
  if pgrep -f "output-dir $R/$F/$name( |$)" >/dev/null; then wait_round "$name"; return; fi
  "$R/run_round.sh" "$F" "$G" "$name" "$trials" "$@"
}

if ! done_round round1; then
  wait_round round1 || { echo "[chain-after $F] round1 not finished and not running; resume it first" | tee -a $L/chain.status; exit 1; }
fi
[ -f "$R/$F/round2_trials.json" ] || python3 "$R/make_round2.py" "$F" | tee -a $L/chain.status || exit 1
run round2 "$R/$F/round2_trials.json" --seed 42 || { echo "[chain-after $F] round2 failed" | tee -a $L/chain.status; exit 1; }
[ -f "$R/$F/seed_trials.json" ] || python3 "$R/select_final.py" "$F" | tee -a $L/chain.status || exit 1
for S in 43 44; do
  run seed$S "$R/$F/seed_trials.json" --seed $S || { echo "[chain-after $F] seed$S failed" | tee -a $L/chain.status; exit 1; }
done
echo "[chain-after $F] training complete $(date -u +%FT%TZ)" | tee -a $L/chain.status
