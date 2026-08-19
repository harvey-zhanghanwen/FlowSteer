# AIME 2026 Round 01 architecture backup

This backup contains the evaluation-only AIME adaptation of the existing
SkillFlow/FlowSteer runtime. It keeps the local Qwen3.5-9B Flow-Director,
progressive `ADD_SUBGRAPH` Canvas execution, strict INTEGER terminal evaluator,
Direct/AgentGraph paired reporting, trajectory receipts, and Stable Zero
checks. It does not contain training, optimizer updates, generated data, model
weights, credentials, or per-task runtime artifacts.

## Clean backup reference

This recovery snapshot is published on branch
`backup/aime2026-stable-zero-arch-clean-20260819`. Restore it in an isolated
directory so an existing dirty worktree is not changed:

```bash
git clone --single-branch --depth 1 \
  --branch backup/aime2026-stable-zero-arch-clean-20260819 \
  https://github.com/harvey-zhanghanwen/FlowSteer.git \
  FlowSteer-aime2026-restore
```

The tracked `artifacts/` tree is deliberately absent from this clean branch.
Restore the external Step-0 adapter and the dependencies listed below before
running model-backed evaluation.

## Runtime recovery prerequisites

The saved Round 01 configuration is evaluation-only, but it intentionally
binds inference to the frozen Step-0 adapter
`theta_jointqa_progressive_step_000000`.  The clean branch excludes that
62 MiB adapter.  Restore the exact three-file adapter directory from the
project backup branch before running Stable Zero or evaluation:

```bash
git fetch --depth 1 origin experiment/joint-qa-progressive-skill-rl-02
git restore --source FETCH_HEAD -- \
  artifacts/hotpotqa_multiagent_skill/policy_step_000000/theta
```

Set the local runtime paths without committing them.  The three source
variables below correspond directly to `config/datasets_aime2026.yaml`:

```bash
export FLOWSTEER_AIME_HISTORY_PATH=/path/to/aime_2000_2025.jsonl
export FLOWSTEER_AIME2025_PATH=/path/to/flowsteer/aime2025.jsonl
export FLOWSTEER_DATASETS_ROOT=/path/to/datasets
export QWEN35_9B_MODEL_PATH=/path/to/Qwen3.5-9B
export QWEN35_9B_TOKENIZER_PATH=/path/to/Qwen3.5-9B-tokenizer
export FLOWSTEER_PYTHON_BIN=/path/to/skillflow-runtime/bin/python
export FLOWSTEER_ROLLOUT_GPU=4
export FLOWSTEER_SUPERVISOR_PORT=8015
```

Keep port `8015` for this frozen configuration because the local Agent model
entry in `config/model_catalog_hotpotqa_deep_v6.yaml` uses that endpoint.  If
the Director selects a remote Agent, also provide `VECTOR_ENGINE_API_KEY`
through the local environment.  Do not add any credential to Git.

Start the SkillFlow SGLang Supervisor in a separate terminal:

```bash
bash scripts/start_qwen35_director_server.sh
```

The evaluation runner calls SGLang `/load_lora_adapter` and performs its
adapter canary automatically; no manual adapter-load request is required.
`requirements-qwen35-runtime.txt` describes the runtime dependencies, while
the local model, CUDA/SGLang environment, adapter, and raw datasets remain
external to this source backup.

## Population boundary

- Training candidate population: AIME 2000--2024, cycled internally to 512.
- Architecture development: all 30 AIME 2025 tasks.
- Final evaluation: all 30 official AIME 2026 tasks.
- AIME 2026 tasks are never used for architecture selection, posterior fitting,
  or Skill validation.

Materialize the ignored local data:

```bash
"$FLOWSTEER_PYTHON_BIN" scripts/prepare_aime2026_dataset.py
```

Validate the configuration without starting a model:

```bash
"$FLOWSTEER_PYTHON_BIN" \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/development_aime2026_round_01.yaml \
  --prepare-only
```

Run Stable Zero and the fixed final evaluation after restoring `.env`, the
local Qwen3.5-9B model/tokenizer, and the external Step-0 adapter:

```bash
set -a
source .env
set +a

"$FLOWSTEER_PYTHON_BIN" \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_aime2026_round_01.yaml \
  --canary-only

"$FLOWSTEER_PYTHON_BIN" \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_aime2026_round_01.yaml
```

## Skill boundary

The task-family Skill adapter reuses the existing Bayesian posterior, UCB,
EVSI, problem-cluster calibration, and lifecycle gate. Round 01 has no
randomized paired intervention or independent confirmatory evidence, so no
AIME candidate is published as `ACTIVE` and `skills.enabled` remains `false`.
Observed model/topology correlations from development are diagnostic only.

## Recorded Round 01 result

- Development (AIME 2025): AgentGraph 20/30 = 66.67% accuracy;
  Qwen3.5-9B Direct 0/30 = 0.00%.
- Fixed final (AIME 2026): AgentGraph 13/30 = 43.33% accuracy;
  Qwen3.5-9B Direct 1/30 = 3.33%.
- Stable Zero: 30/30 final tasks passed the full chain with valid evaluator
  receipts, explicit `FINISH`, saved output inbox, and verified Director turns.
- Terminal, evaluator, and operational failures: 0.
- Policy: `qwen35-9b-aime2026-round-01-step-000000`;
  Skill condition: memory-off; training/optimizer updates: none.

## External dependencies not stored in Git

- `data/aime2026_v1/`
- `.env`
- local Qwen3.5-9B model and tokenizer
- `artifacts/hotpotqa_multiagent_skill/policy_step_000000/theta`
- all Direct predictions, AgentGraph trajectories, evidence streams, and model
  responses under `artifacts/`
