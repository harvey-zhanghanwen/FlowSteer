#!/usr/bin/env bash
# Operational entry point for this existing HotpotQA run, not a new trainer.
# The single-GPU lifecycle and training algorithm remain in train_hotpotqa_grpo.py.
set -euo pipefail

training_gpu=${1:?Usage: resume_hotpotqa_on_idle_gpu.sh PHYSICAL_GPU}
if [[ ! "$training_gpu" =~ ^(0|[1-9][0-9]*)$ ]]; then
  printf 'Physical GPU must be a non-negative integer\n'
  exit 2
fi
cd /ssd1/iclr/1/.tmp/FlowSteer-hotpotqa-skillflow-ttb-250step-20260906
exec 9>artifacts/hotpotqa_dynamic_ledger_grpo_300step/training_process.lock
flock -n 9
/ssd1/iclr/gpf/venvs/skillflow/bin/python - "$training_gpu" <<'PY'
import sys
from src.interactive.config_loader import load_yaml
gpu = load_yaml('config/training_hotpotqa_dynamic_ledger_grpo.yaml')['gpu']
target = int(sys.argv[1])
assert {gpu[name] for name in ('learner_physical', 'rollout_physical',
    'gradient_replica_physical', 'supervisor_gpu_id')} == {target}
assert gpu['learner_device'] == gpu['gradient_replica_device'] == f'cuda:{target}'
PY
printf 'GPU%s resume queue pid=%s started=%s; optimizer is not running while waiting\n' "$training_gpu" "$$" "$(date -Is)"

# Resource admission only: do not adopt/stop another project's GPU process.
# Three idle snapshots and this local flock are not a cluster-wide reservation.
idle_checks=0
last_resource_state=''
queue_deadline=$((SECONDS + 86400))
while (( SECONDS < queue_deadline )); do
  if [ -f artifacts/hotpotqa_dynamic_ledger_grpo_300step/STOP_REQUESTED ]; then
    printf 'STOP_REQUESTED present; no training launched\n'
    exit 0
  fi
  gpu_free=$(nvidia-smi -i "$training_gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  gpu_apps=$(nvidia-smi -i "$training_gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
  gpu_free=${gpu_free//[[:space:]]/}
  if [[ ! "$gpu_free" =~ ^[0-9]+$ ]]; then
    printf 'Invalid GPU resource response; not launching\n'
    exit 2
  fi
  if [ -z "$gpu_apps" ] && (( gpu_free >= 78000 )); then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi
  resource_state="free_mib=$gpu_free compute_pids=$gpu_apps idle_checks=$idle_checks"
  if [ "$resource_state" != "$last_resource_state" ]; then
    printf '%s %s\n' "$(date -Is)" "$resource_state"
    last_resource_state=$resource_state
  fi
  if (( idle_checks >= 3 )); then break; fi
  sleep 2
done
if (( idle_checks < 3 )); then
  printf '24-hour GPU%s wait expired without launching training\n' "$training_gpu"
  exit 75
fi
if [ -n "$(ss -H -ltn '( sport = :8016 )')" ]; then
  printf 'Director port 8016 is occupied; no existing service changed\n'
  exit 2
fi

set -a
. /ssd1/iclr/1/FlowSteer/.env
set +a
# Keep the frozen worker on 8015, Director on its configured 8016.
unset FLOWSTEER_SUPERVISOR_PORT
# Use physical GPU numbering; SGLang manages visibility in its own child.
unset CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
# The standard SDK reads the existing credential file; no credential is copied.
export NETRC=/ssd1/iclr/1/.netrc
printf 'GPU%s idle; resuming existing checkpoint and W&B run at %s\n' "$training_gpu" "$(date -Is)"
exec /ssd1/iclr/gpf/venvs/skillflow/bin/python scripts/train_hotpotqa_grpo.py \
  --config config/training_hotpotqa_dynamic_ledger_grpo.yaml \
  --allow-md-grpo --resume --stop-after-optimizer-steps 300 \
  >>"artifacts/hotpotqa_dynamic_ledger_grpo_300step/long_training_console_20260908_gpu${training_gpu}_resume.log" 2>&1
