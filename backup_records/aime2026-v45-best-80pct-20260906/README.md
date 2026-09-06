# AIME 2026 v45 historical-best backup

This backup records the highest completed, same-denominator AIME 2026
AgentGraph result available in this workspace on 2026-09-06.

## Evaluated condition

- Condition: `aime2026_runtime_v45_provider_protocol_admission`
- Config: `config/evaluation_aime2026_runtime_v45_provider_protocol_admission.yaml`
- Official test population: fixed 30 AIME 2026 tasks
- AgentGraph strict Accuracy: `24 / 30 = 80.00%`
- Evaluator-valid AgentGraph Accuracy: `24 / 27 = 88.89%`
- Qwen3.5-9B Direct: `19 / 30 = 63.33%`
- Explicit `FINISH`: `27 / 30`
- Terminal failures: `3 / 30`
- Collection/operational failures: `0`
- Training, GRPO, MACE, Bayesian update, and Skill injection: disabled

The later interface-only prompt ablation obtained `21 / 30 = 70.00%` and is
not the selected historical-best condition.

## Recovery entry point

Use the branch named `backup/aime2026-v45-best-80pct-20260906`. The executable
entry point is:

```text
scripts/evaluate_completion_benchmark_round.py
  --config config/evaluation_aime2026_runtime_v45_provider_protocol_admission.yaml
```

The final config selects:

- Director prompt `agentgraph.director.minimal-neutral.v15`;
- model catalog `config/model_catalog_aime2026_heterogeneous_thinking_v18.yaml`;
- target-blind integer Exact Match evaluator;
- free-text, role-neutral Agent contracts;
- explicit `FINISH` terminal semantics;
- disabled training and Skill paths.

## Evidence included

The Git history includes the final report, root-cause report, run manifest,
preflight receipt, selected task manifest, Direct predictions, paired results,
Wrong Demos, and the empty collection-failure receipt. Large lossless runtime
files are stored as gzip-compressed copies below this directory:

```text
evidence/agentgraph_trajectories.jsonl.gz
evidence/evidence_trajectories.jsonl.gz
evidence/rollout_checkpoints.jsonl.gz
evidence/snapshots.jsonl.gz
```

These are evidence copies; the active runtime continues to use the ignored
`artifacts/` tree.

## Provenance limitation

The formal run manifest records branch
`backup/aime2026-v3-best-v5-candidate-20260830` at commit `52e4e2a`, but the
worktree was dirty and the manifest recorded only branch/HEAD. The v45 config,
v18 model catalog, and v45 runtime changes were therefore not recoverable from
that commit alone.

This branch preserves the complete currently available v45-compatible source,
configuration, tests, and result evidence. Most runtime files predate the
formal run. `src/interactive/director.py` also contains a later additive
interface-only prompt-ablation option; the selected v45 config still resolves
the unchanged `minimal-neutral.v15` prompt. Consequently this backup is the
best recoverable v45-compatible snapshot, not a claim of a byte-identical clean
commit captured at evaluation start.

Raw `data/` files and model weights remain outside Git according to repository
policy. The selected-task manifest needed to identify the fixed 30-task
denominator is included with the evidence.
