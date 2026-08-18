# HealthBench Professional Round 01 architecture backup

This backup contains the evaluation-only HealthBench Professional adaptation
of the existing SkillFlow/FlowSteer runtime. It keeps the local Qwen3.5-9B
Flow-Director, progressive `ADD_SUBGRAPH` Canvas execution, free-text terminal
protocol, Direct/AgentGraph paired reporting, trajectory receipts, and Stable
Zero checks.

The public reference evaluator is compatible with OpenAI simple-evals: each
rubric item is graded independently and the reported primary metric is the
mean un-clipped `raw_score`. The evaluator-only GPT-4.1 registry is separate
from the AgentGraph model catalog, so the judge is not selectable by
Flow-Director. This is not the unavailable OpenAI private held-out evaluator
and must not be reported as an official leaderboard score.

## Clean backup reference

This recovery snapshot is published on branch
`backup/healthbench-professional-stable-zero-arch-clean-20260819`. Restore it
in an isolated directory so an existing dirty worktree is not changed:

```bash
git clone --single-branch --depth 1 \
  --branch backup/healthbench-professional-stable-zero-arch-clean-20260819 \
  https://github.com/harvey-zhanghanwen/FlowSteer.git \
  FlowSteer-healthbench-restore
```

The tracked `artifacts/` tree is deliberately absent from this clean branch.
Restore the external Step-0 adapter and the dependencies listed below before
running model-backed evaluation.

## Population boundary

- Public records: 525.
- Fixed held-out: 128 unique base tasks.
- Remaining training candidates: 397 unique base tasks, cycled only inside the
  training split to 512.
- Architecture development: the first 32 training tasks.
- Development and held-out base task IDs are disjoint.

Materialize the ignored local data:

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/prepare_agentgraph_datasets.py \
  --datasets healthbench_professional
```

Validate the configuration without starting a model:

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/development_healthbench_professional_round_01.yaml \
  --prepare-only
```

After restoring `.env`, the local Qwen3.5-9B model/tokenizer, and the external
Step-0 adapter, run:

```bash
set -a
source .env
set +a

/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_professional_round_01.yaml \
  --canary-only

/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_professional_round_01.yaml
```

## Skill boundary

The task-family Skill adapter reuses the existing Bayesian posterior, UCB,
EVSI, problem-cluster calibration, and lifecycle gate. Round 01 does not
publish a Skill without randomized paired effects and independent confirmation;
until those exist, `skills.enabled` remains `false`.

## Recorded development result

- Fixed development tasks: 32.
- Stable Zero: 32/32 passed the full chain with valid Direct and AgentGraph
  evaluator receipts, explicit `FINISH`, saved output inbox, and verified
  Director turns.
- AgentGraph mean rubric `raw_score`: 0.1533.
- Qwen3.5-9B Direct mean rubric `raw_score`: 0.1075.
- Mean paired delta: +0.0458; pair outcomes: 4 higher, 24 equal, 4 lower.
- Terminal, evaluator, and operational failures: 0.
- Policy: `qwen35-9b-healthbench-round-01-step-000000`;
  Skill condition: memory-off; training/optimizer updates: none.

The development topology distribution includes single-node, serial, fan-in,
and mixed graphs. These observations are not randomized causal evidence and
therefore do not change the Skill lifecycle state.

## Recorded held-out result

- Fixed held-out tasks: 128.
- Stable Zero: 128/128 passed with valid Direct and AgentGraph evaluator
  receipts, explicit `FINISH`, saved output inbox, and verified Director turns.
- AgentGraph mean rubric `raw_score`: 0.2075.
- Qwen3.5-9B Direct mean rubric `raw_score`: 0.1318.
- Mean paired delta: +0.0757; pair outcomes: 35 higher, 74 equal, 19 lower.
- Terminal, evaluator, and operational failures: 0.
- Policy: `qwen35-9b-healthbench-round-01-step-000000`;
  Skill condition: memory-off; training/optimizer updates: none.

This value is the public simple-evals-compatible mean rubric `raw_score`; it is
not an accuracy percentage and is not the unavailable private leaderboard
metric.

## Data excluded from Git

Credentials, conversations, rubrics, physician responses, judge responses,
per-task predictions, trajectories, model weights, generated data, and all
other runtime artifacts remain outside the backup.
