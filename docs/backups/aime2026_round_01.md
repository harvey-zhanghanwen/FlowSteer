# AIME 2026 Round 01 architecture backup

This backup contains the evaluation-only AIME adaptation of the existing
SkillFlow/FlowSteer runtime. It keeps the local Qwen3.5-9B Flow-Director,
progressive `ADD_SUBGRAPH` Canvas execution, strict INTEGER terminal evaluator,
Direct/AgentGraph paired reporting, trajectory receipts, and Stable Zero
checks. It does not contain training, optimizer updates, generated data, model
weights, credentials, or per-task runtime artifacts.

## Population boundary

- Training candidate population: AIME 2000--2024, cycled internally to 512.
- Architecture development: all 30 AIME 2025 tasks.
- Final evaluation: all 30 official AIME 2026 tasks.
- AIME 2026 tasks are never used for architecture selection, posterior fitting,
  or Skill validation.

Materialize the ignored local data:

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python scripts/prepare_aime2026_dataset.py
```

Validate the configuration without starting a model:

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python \
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

/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_aime2026_round_01.yaml \
  --canary-only

/ssd1/iclr/gpf/venvs/skillflow/bin/python \
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
