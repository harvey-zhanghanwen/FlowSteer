#!/usr/bin/env bash
# Operational entry point for this existing HotpotQA run, not a new trainer.
# The single-GPU lifecycle and training algorithm remain in train_hotpotqa_grpo.py.
set -euo pipefail

gpu_selection=${1:-auto}
if [[ "$gpu_selection" != auto && ! "$gpu_selection" =~ ^(0|[1-9][0-9]*)$ ]]; then
  printf 'GPU selection must be auto or a non-negative physical GPU index\n'
  exit 2
fi
approved_residual_pid=${HOTPOTQA_APPROVED_RESIDUAL_PID:-}
stop_after_steps=${HOTPOTQA_STOP_AFTER_STEPS:-250}
if [[ ! "$stop_after_steps" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Stop boundary must be a positive optimizer step count\n'
  exit 2
fi
if [[ -n "$approved_residual_pid" && ( "$gpu_selection" == auto || ! "$approved_residual_pid" =~ ^[1-9][0-9]*$ ) ]]; then
  printf 'Approved residual PID requires an explicitly selected GPU and positive PID\n'
  exit 2
fi
cd /ssd1/iclr/1/.tmp/FlowSteer-hotpotqa-skillflow-ttb-250step-20260906
exec 9>artifacts/hotpotqa_dynamic_ledger_grpo_300step/training_process.lock
flock -n 9
printf 'GPU selection=%s resume queue pid=%s started=%s; optimizer is not running while waiting\n' "$gpu_selection" "$$" "$(date -Is)"

# Optional operational admission for the explicitly approved small allocation.
# Do not stop it, infer ownership, or relax the existing free-memory threshold.
gpu_processes_admissible() {
  local target_gpu=$1 observed_apps observed_pid observed_memory observed_util
  observed_apps=$(nvidia-smi -i "$target_gpu" --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits) || return 2
  if [[ -z "$observed_apps" ]]; then return 0; fi
  [[ -n "$approved_residual_pid" && "$observed_apps" != *$'\n'* ]] || return 1
  IFS=, read -r observed_pid observed_memory <<< "$observed_apps"
  observed_pid=${observed_pid//[[:space:]]/}
  observed_memory=${observed_memory//[[:space:]]/}
  [[ "$observed_pid" == "$approved_residual_pid" && "$observed_memory" =~ ^[0-9]+$ ]] || return 1
  (( observed_memory <= 1024 )) || return 1
  observed_util=$(nvidia-smi -i "$target_gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits) || return 2
  [[ "$observed_util" == 0 ]]
}

# Resource admission only: do not adopt/stop another project's GPU process.
# Three idle snapshots and this local flock are not a cluster-wide reservation.
while true; do
idle_checks=0
training_gpu=''
last_resource_state=''
queue_deadline=$((SECONDS + 86400))
nvidia_query=(--query-gpu=index,memory.free --format=csv,noheader,nounits)
if [[ "$gpu_selection" != auto ]]; then
  nvidia_query+=(-i "$gpu_selection")
fi
while (( SECONDS < queue_deadline )); do
  if [ -f artifacts/hotpotqa_dynamic_ledger_grpo_300step/STOP_REQUESTED ]; then
    printf 'STOP_REQUESTED present; no training launched\n'
    exit 0
  fi
  gpu_snapshot=$(nvidia-smi "${nvidia_query[@]}")
  available_gpu=''
  while IFS=, read -r candidate_gpu candidate_free; do
    candidate_gpu=${candidate_gpu//[[:space:]]/}
    candidate_free=${candidate_free//[[:space:]]/}
    if [[ ! "$candidate_gpu" =~ ^[0-9]+$ || ! "$candidate_free" =~ ^[0-9]+$ ]]; then
      printf 'Invalid GPU resource response; not launching\n'
      exit 2
    fi
    if (( candidate_free < 78000 )); then continue; fi
    if gpu_processes_admissible "$candidate_gpu"; then
      available_gpu=$candidate_gpu
      break
    fi
  done <<< "$gpu_snapshot"
  if [ -n "$available_gpu" ] && [ "$available_gpu" = "$training_gpu" ]; then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
    training_gpu=$available_gpu
    if [ -n "$training_gpu" ]; then idle_checks=1; fi
  fi
  resource_state="selected_gpu=${training_gpu:-none} idle_checks=$idle_checks free_memory_mib=[$gpu_snapshot]"
  if [ "$resource_state" != "$last_resource_state" ]; then
    printf '%s %s\n' "$(date -Is)" "$resource_state"
    last_resource_state=$resource_state
  fi
  if (( idle_checks >= 3 )); then break; fi
  sleep 2
done
if (( idle_checks < 3 )); then
  printf 'GPU still busy after 24 hours; continuing resource wait without launching training\n'
  sleep 30
  continue
fi
if [ -n "$(ss -H -ltn '( sport = :8016 )')" ]; then
  printf 'Director port 8016 is occupied; waiting without changing the existing service\n'
  sleep 30
  continue
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
export HOTPOTQA_TRAIN_GPU=${training_gpu:?No idle GPU selected}
/ssd1/iclr/gpf/venvs/skillflow/bin/python - "$training_gpu" <<'PY'
import json
import sys
from pathlib import Path
from src.interactive.config_loader import load_model_registry, load_yaml, validate_agent_graph_config
config = load_yaml('config/training_hotpotqa_dynamic_ledger_grpo.yaml')
validate_agent_graph_config(config)
gpu = config['gpu']
target = int(sys.argv[1])
assert {int(gpu[name]) for name in ('learner_physical', 'rollout_physical',
    'gradient_replica_physical', 'supervisor_gpu_id')} == {target}
assert gpu['learner_device'] == gpu['gradient_replica_device'] == f'cuda:{target}'
state = json.loads(Path('artifacts/hotpotqa_dynamic_ledger_grpo_300step/run_state.json').read_text())
assert Path(state['optimizer_state_checkpoint']).is_file()
frozen = json.loads(Path(state['ledger_epoch_receipt']).read_text())['condition']['versions']['model_catalog']
registry = load_model_registry(config['agent_graph']['model_catalog_path'])
assert registry.catalog_id == frozen, 'Frozen worker catalog differs; no model launched'
print(f'GPU{target} checkpoint and catalog preflight passed; resuming step {state["optimizer_updates_completed"] + 1}')
PY
# Preflight imports take time: do not launch if another project took the GPU.
final_free=$(nvidia-smi -i "$training_gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
final_free=${final_free//[[:space:]]/}
if [[ ! "$final_free" =~ ^[0-9]+$ ]] || (( final_free < 78000 )) || ! gpu_processes_admissible "$training_gpu"; then
  printf 'Selected GPU became occupied during preflight; returning to resource wait\n'
  sleep 30
  continue
fi
printf 'GPU%s idle; resuming existing checkpoint and W&B run at %s\n' "$training_gpu" "$(date -Is)"
training_exit=0
/ssd1/iclr/gpf/venvs/skillflow/bin/python scripts/train_hotpotqa_grpo.py \
  --config config/training_hotpotqa_dynamic_ledger_grpo.yaml \
  --allow-md-grpo --resume --stop-after-optimizer-steps "$stop_after_steps" \
  >>"artifacts/hotpotqa_dynamic_ledger_grpo_300step/long_training_console_20260908_gpu${training_gpu}_resume.log" 2>&1 || training_exit=$?
if (( training_exit == 75 )); then
  printf 'Zero-update batch archived; waiting 30s before resource-gated resampling (no optimizer step counted)\n'
  sleep 30
  continue
fi
# Unknown failures and post-update recovery are not safe batch replays.
exit "$training_exit"
done
