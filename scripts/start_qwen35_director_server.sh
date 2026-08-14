#!/usr/bin/env bash
# Start the Qwen3.5-9B Flow-Director vLLM endpoint on the inference GPU.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${QWEN35_9B_MODEL_PATH:-Qwen/Qwen3.5-9B}"
python_bin="${FLOWSTEER_PYTHON_BIN:-python3}"

exec "$project_root/scripts/run_on_gpu_role.sh" inference \
  "$python_bin" -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --served-model-name qwen3.5-9b-director \
  --host 127.0.0.1 \
  --port 8003 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 64 \
  --trust-remote-code \
  --dtype bfloat16
