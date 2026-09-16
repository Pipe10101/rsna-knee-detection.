#!/usr/bin/env bash
# Supervisor: wait for the in-flight arm to finish, then run the reg4-based queue.
#
# The old driver (ablate_slotknee.sh) was stopped between arms so it would not spend the
# next ~18 h on old-baseline architecture ablations (tokenpool/tb2/nomixer) that answer a
# question reg4 superseded.  Its arms are all resumable -- re-running that script later
# skips everything already on disk -- so nothing is lost, only re-ordered.
#
# Only ONE MPS training may run at a time, hence the wait.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG="logs/reg4_queue.log"

while pgrep -f "train_slotknee.py" > /dev/null 2>&1; do
    sleep 60
done
echo "$(date '+%F %T') in-flight arm finished; starting reg4 queue" >> "$LOG"
exec bash scripts/queue_reg4_arms.sh \
    cache/slots_P224_full/slots_P224 models/ablate_full 8 0 1 >> "$LOG" 2>&1
