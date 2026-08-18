# SWE-bench Verified Round 01 architecture backup

This backup describes the evaluation-only SWE-bench Verified adaptation of the
existing FlowSteer/AgentGraph runtime. It freezes the local official Verified
records, the SkillFlow SWE-bench evaluator, and the specified official Docker
harness. The only formal benchmark metric is the official harness resolved
rate. Text similarity, patch-shape heuristics, and all other proxy scores are
excluded from formal reporting.

## Publication identity

- Remote branch: `backup/swebench-verified-arch-blocked-clean-20260819`
- Exact commit: obtain after committing with
  `git rev-parse HEAD`; it is intentionally not self-referenced in this file.

The branch is an independently recoverable source/configuration snapshot. It
does not claim that a formal SWE-bench score was produced.

## Frozen data boundary

The local official SWE-bench Verified release contains 500 unique instances.
The deterministic project split is:

- Fixed held-out validation: the first 128 Verified instances.
- Remaining unique training candidates: 372 Verified instances.
- Project training view: 512 records, obtained by cycling only those 372
  remaining candidates.
- Architecture development: the first 32 records of the project training view.

The 128 held-out instances are disjoint from the 372 unique training
candidates. Cycling never enters the held-out split. The name `train` denotes
the development-side data partition here; Round 01 does not perform training.

Materialize only the SWE-bench records from the checked-in dataset catalog:

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/prepare_agentgraph_datasets.py \
  --datasets swe_bench
```

The data preparation step is expected to use:

- Verified dataset: `/ssd1/iclr/.private/skillflow-resources/swebench-verified`
- SkillFlow evaluator:
  `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/training/swe_bench_eval.py`
- Official harness:
  `/ssd1/iclr/.private/skillflow-resources/SWE-bench-harness-d83`

## Production configurations

- Development:
  `config/development_swebench_verified_round_01.yaml`
- Fixed held-out evaluation:
  `config/evaluation_swebench_verified_round_01.yaml`

Both configurations are evaluation-only, use one rollout per task, and keep
Docker evaluation concurrency at one. They require the official harness and
declare `resolved_rate` as the only official metric with proxy metrics
disabled.

## Prepare-only recovery gate

The prepare-only path freezes the selected tasks and writes a run manifest
without constructing the inference backend, contacting a model or API, or
touching Docker:

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/development_swebench_verified_round_01.yaml \
  --prepare-only

/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_swebench_verified_round_01.yaml \
  --prepare-only
```

The corresponding manifest locations are:

- `artifacts/swebench_verified_round_01/development/run_manifest.json`
- `artifacts/swebench_verified_round_01/evaluation/run_manifest.json`

A `prepared` manifest proves only configuration validation and frozen task
selection. It is not a Stable Zero receipt, an official harness result, or a
held-out score.

## Formal evaluation contract

For every Direct and AgentGraph prediction, the configured SkillFlow evaluator
must delegate to the specified official SWE-bench Docker harness. A valid
evaluation receipt must contain a boolean `resolved` result and explicitly show
that no proxy metric was used. The aggregate resolved rate is the arithmetic
mean of those per-instance boolean outcomes over the fixed denominator.

If the evaluator, Verified instance, harness import, Docker daemon, or official
receipt is unavailable, the task is invalid and excluded as an evaluator or
operational failure. The runtime must not manufacture a score or substitute a
text-based metric.

After all external runtime requirements have been restored, the intended
gates are:

```bash
/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/development_swebench_verified_round_01.yaml \
  --canary-only

/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/development_swebench_verified_round_01.yaml

/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_swebench_verified_round_01.yaml \
  --canary-only

/ssd1/iclr/gpf/venvs/skillflow/bin/python \
  scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_swebench_verified_round_01.yaml
```

These commands are recovery instructions only; they were not executed while
creating this document.

## Skill and training boundary

Evidence-gated Skill use is **OFF** for both conditions:

- `skills.enabled` remains `false`.
- No ACTIVE Skill is retrieved or injected.
- No Skill candidate is published or advanced through its lifecycle.

Round 01 also performs no training, GRPO, backward pass, optimizer update,
policy synchronization, exploration, Bayesian update, LoRA update, or LoRA
publication. The configured Step-0 adapter is a fixed evaluation dependency,
not a SWE-bench training output.

## Current blocker and measurement status

The Verified dataset, SkillFlow evaluator, specified harness checkout, and
Docker socket path are present. The non-container Docker daemon ping currently
fails closed with:

```text
docker.errors.DockerException
caused by PermissionError(13, "Permission denied")
```

Because the official harness cannot reach the Docker daemon, neither the
Stable Zero canary nor the development or fixed held-out official evaluation
has run. The current formal measurement status is **blocked / unmeasured**.
It must not be reported as zero resolved rate, zero accuracy, or a completed
benchmark result.

No development report or held-out report should be published until Docker
preflight succeeds and the official receipt chain is complete for every task
in the relevant fixed condition.

## External runtime requirements

Full execution additionally requires the configured local Qwen3.5-9B model and
tokenizer, the fixed Step-0 adapter, a running compatible Director service, the
configured executor model services, and a Docker daemon that can run the
official SWE-bench harness. None of these runtime requirements is implied by a
prepare-only manifest.
