# AgentGraph seven-dataset alignment

This phase prepares data only.  It does not start the Qwen3.5-9B Director,
training, rollout, backward propagation, benchmark environments, or evaluators.

## Local inventory

The server already contained all seven requested sources, so no duplicate raw
download was needed.  The empty `/home/test/datasets/HotpotQA` directory and
the incomplete TriviaQA `.part` archive are not used.

| Dataset | Selected local source | Static readiness |
| --- | --- | --- |
| HotpotQA | `/home/test/datasets/HotpotQA_HF/distractor` | complete Parquet train/validation copy |
| TriviaQA | `/home/test/datasets/TriviaQA_HF/rc.nocontext` | complete selected closed-book Parquet copy |
| AIME family | `/home/test/datasets/AIME_2026`, the FlowSteer AIME 2025 development set, and SkillFlow's local historical-AIME pool | protocol-separated development, training, and untouched 2026 final-test populations |
| HealthBench Professional | `/home/test/datasets/HealthBench_Professional` | complete 525-record public release |
| WebShop | `/home/test/datasets/WebShop` | products and 12,087 human goals present; Lucene index still needs a later runtime build |
| ALFWorld | `/home/test/datasets/ALFWorld` | official text-game data/code present |
| SWE-bench | regular train/dev under `/home/test/datasets/SWE-bench`; Verified cache under `/ssd1/iclr/.private/skillflow-resources/swebench-verified` | 128 regular-dev instances and all pinned repositories are ready; official Docker evaluation remains externally unavailable |

## Unified record

Every output line overlays the design-note task contract on the upstream
compatibility fields:

```text
schema_version, task_id, question, ground_truth, split, metadata
source, dataset
answer, task_type, context, extra
env_type?, env_config?, code_files?
```

The first line is the strict `TaskRecord` input.  The second line supports the
legacy FlowSteer loader.  The remaining fields retain SkillFlow's dataset and
environment boundary.  The strict loader rehydrates the SkillFlow fields under
`TaskRecord.metadata.skillflow`.

Evaluator-only values are never concatenated into `question`:

- HotpotQA supporting facts;
- TriviaQA accepted aliases;
- AIME gold integer;
- HealthBench rubrics and physician reference response;
- WebShop/ALFWorld environment target;
- SWE-bench gold patch and test payload.

## Initial unified 128/512 alignment

The checked-in catalog implements the user's deterministic rule without a
shuffle:

1. consume the first 128 candidates as held-out validation;
2. consume the following 512 candidates as training;
3. if fewer than 512 training candidates remain, cycle only that post-held-out
   training pool from its first record;
4. never cycle or reuse a held-out base task in training.

Current aligned counts:

| Dataset | Held out | Train | Unique train bases | Train-only cycles |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA | 128 | 512 | 512 | 0 |
| TriviaQA | 128 | 512 | 512 | 0 |
| AIME family | 128 | 512 | 402 | 110 |
| HealthBench Professional | 128 | 512 | 397 | 115 |
| WebShop | 128 | 512 | 512 | 0 |
| ALFWorld | 128 | 512 | 512 | 0 |
| SWE-bench Verified | 128 | 512 | 372 | 140 |
| **Total** | **896** | **3,584** | **3,219** | **365** |

AIME 2026 has only 30 official tasks.  The initial unified alignment used the
following candidate sequence:

```text
30 official AIME 2026 + 98 historical AIME held out
402 remaining unique historical AIME used for training
110 additional records cycled only from those 402 training candidates
```

Each cycle record has its own `task_id`, while
`metadata.sampling.base_task_id` identifies the underlying task.  Validation
and training base-task sets have zero overlap for all seven datasets.

This 128/512 view is a custom project recipe, not an untouched official
leaderboard split.  Official benchmark evaluation must later use separately
frozen full test views and the corresponding environment/rubric/harness.

## Current protocol-separated evaluation views

The architecture-validation runs no longer interpret the initial generic view
as an official AIME or SWE-bench result.

### AIME family development-128

`config/datasets_aime_development128.yaml` defines the current fixed
architecture-development condition:

```text
development: 30 AIME 2025 tasks + 98 historical AIME tasks
training candidates: the next 512 historical tasks, with maximum year 2024
final test: all 30 official AIME 2026 tasks, untouched
```

The 128-task score must therefore be reported as **AIME-family
development-128**, not as AIME 2026 accuracy.  Training is disabled in the
current condition.

### SWE-bench regular-dev Coding Agent

`config/datasets_swebench.yaml` keeps the three populations disjoint:

```text
training: 512 regular-train instances
architecture development: all 128 regular-dev instances
final test: complete SWE-bench Verified population
```

For the development condition, all 128 task-pinned repositories and base
commits are available.  The SkillFlow-derived detached-worktree lifecycle and
repository tools are connected through `CodingExecutionAdapter`; formal
scoring is exclusively the pinned official SWE-bench Docker harness
`resolved_rate`.  If that harness is unavailable, the condition is reported
as **unmeasurable**, never as zero and never with local test success used as a
proxy metric.

## Commands

Prepare once (the command refuses to overwrite a non-empty output directory):

```bash
python scripts/prepare_agentgraph_datasets.py \
  --catalog config/datasets_agentgraph.yaml
```

Validate counts, schema, cycling location, AIME isolation, and cross-split base
IDs:

```bash
python scripts/validate_agentgraph_datasets.py \
  --data-dir data/agentgraph_v1
```

Exercise the AgentGraph loader without starting the Director or any executor:

```bash
python scripts/run_agentgraph.py \
  --dataset data/agentgraph_v1/train.jsonl \
  --expected-split train \
  --task-index 0 \
  --dry-load
```

Generated records live under `data/agentgraph_v1/` and remain git-ignored.
Only preparation/loading code, configuration, documentation, and the
non-content manifest are backed up to GitHub.
