#!/usr/bin/env bash
# Start the Qwen3.5-9B Flow-Director with SkillFlow's SGLang runtime.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_model_path="/home/test/SKILLEV/skillev-new-b2-temp/model/Qwen3.5-9B-modelscope"
default_tokenizer_path="/home/test/SKILLEV/skillev-new-b2-temp/tokenizer/Qwen3.5-9B"
model_path="${QWEN35_9B_MODEL_PATH:-$default_model_path}"
tokenizer_path="${QWEN35_9B_TOKENIZER_PATH:-$default_tokenizer_path}"
python_bin="${FLOWSTEER_PYTHON_BIN:-python3}"
port="${FLOWSTEER_SUPERVISOR_PORT:-8015}"
context_length="${FLOWSTEER_SUPERVISOR_CONTEXT_LENGTH:-32768}"
mem_fraction="${FLOWSTEER_SUPERVISOR_MEM_FRACTION:-0.82}"
api_key="${SGLANG_API_KEY:-EMPTY}"

# SGLang's Qwen3.5 activation kernel is JIT-compiled during warmup.  The
# deployed SkillFlow venv contains ninja, but invoking its Python by absolute
# path does not automatically put the venv's bin directory on PATH.
python_bin_dir="$(cd "$(dirname "$python_bin")" 2>/dev/null && pwd || true)"
if [[ -n "$python_bin_dir" && -x "$python_bin_dir/ninja" ]]; then
  export PATH="$python_bin_dir:$PATH"
fi
if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda-12.9/bin/nvcc ]]; then
  export CUDA_HOME=/usr/local/cuda-12.9
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
usage: $(basename "$0") [additional sglang.launch_server arguments]

Starts the Qwen3.5-9B Flow-Director on the rollout GPU (physical GPU 4 by
default). The local Qwen3.5 model and tokenizer can be overridden separately:

  QWEN35_9B_MODEL_PATH       default: $default_model_path
  QWEN35_9B_TOKENIZER_PATH   default: $default_tokenizer_path
  FLOWSTEER_PYTHON_BIN       default: python3
  FLOWSTEER_ROLLOUT_GPU      default: 4
  FLOWSTEER_SUPERVISOR_PORT  default: 8015

All remaining arguments are passed to sglang.launch_server.

This server enables dynamic LoRA loading. Publish an adapter with POST
/load_lora_adapter using {"lora_name": NAME, "lora_path": CHECKPOINT_PATH},
then select that registered NAME in native POST /generate with the request
field {"lora_path": NAME}. The similarly named field is an adapter selector
at generation time; it is not a new filesystem path.
EOF
  exit 0
fi

exec "$project_root/scripts/run_on_gpu_role.sh" rollout \
  "$python_bin" -m sglang.launch_server \
  --model-path "$model_path" \
  --tokenizer-path "$tokenizer_path" \
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
  --enable-multimodal \
  "$@"
