#!/usr/bin/env bash
# Start the Qwen3.5-9B Flow-Director with SkillFlow's SGLang runtime.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${QWEN35_9B_MODEL_PATH:-Qwen/Qwen3.5-9B}"
python_bin="${FLOWSTEER_PYTHON_BIN:-python3}"
port="${FLOWSTEER_SUPERVISOR_PORT:-8005}"
context_length="${FLOWSTEER_SUPERVISOR_CONTEXT_LENGTH:-32768}"
mem_fraction="${FLOWSTEER_SUPERVISOR_MEM_FRACTION:-0.82}"
api_key="${SGLANG_API_KEY:-EMPTY}"

exec "$project_root/scripts/run_on_gpu_role.sh" rollout \
  "$python_bin" -m sglang.launch_server \
  --model-path "$model_path" \
  --served-model-name supervisor_theta \
  --host 127.0.0.1 \
  --port "$port" \
  --api-key "$api_key" \
  --mem-fraction-static "$mem_fraction" \
  --context-length "$context_length" \
  --enable-lora \
  --max-lora-rank 64 \
  --max-loras-per-batch 1 \
  --max-loaded-loras 2 \
  --lora-target-modules q_proj k_proj v_proj o_proj \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --schedule-policy lpm \
  --trust-remote-code \
  "$@"
