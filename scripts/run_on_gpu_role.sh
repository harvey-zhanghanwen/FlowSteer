#!/usr/bin/env bash
# Execute one command on a validated physical GPU assigned to a FlowSteer role.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <learner|rollout|gradient> <command> [args...]" >&2
  exit 2
fi

role="$1"
shift

case "$role" in
  learner|train)
    gpu="${FLOWSTEER_LEARNER_GPU:-${FLOWSTEER_TRAIN_GPU:-3}}"
    ;;
  rollout|inference)
    gpu="${FLOWSTEER_ROLLOUT_GPU:-${FLOWSTEER_INFERENCE_GPU:-0}}"
    ;;
  gradient|backward|probe)
    gpu="${FLOWSTEER_GRADIENT_GPU:-${FLOWSTEER_PROBE_GPU:-5}}"
    ;;
  *)
    echo "unknown role: $role" >&2
    exit 2
    ;;
esac

if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "GPU index for role $role must be a non-negative integer" >&2
  exit 2
fi

if ! nvidia-smi --id="$gpu" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
  echo "GPU $gpu for role $role is not available" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$gpu"
exec "$@"
