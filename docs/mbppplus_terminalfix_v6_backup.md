# MBPP+ terminalfix v6 backup

This branch preserves the source, configuration, tests, and compact evaluation
evidence for the highest evaluator-complete MBPP+ AgentGraph result available at
the time of backup.

## Result and protocol

- Condition: `mbppplus_react_feedback_v11_runtimefix_thinking_terminalfix_v6`
- Dataset: MBPP+ v0.2.0 fixed test set, 100 tasks
- Evaluator: EvalPlus 0.3.1 complete-source protocol
- AgentGraph Base pass@1: 89/100 (89%)
- AgentGraph Plus pass@1: 81/100 (81%)
- Explicit FINISH: 92/100
- `max_rounds`: 8/100
- Direct Base pass@1: 83/100 (83%)
- Direct Plus pass@1: 71/100 (71%)
- Training, GRPO, optimizer updates, LoRA publication, MACE, Bayesian updates,
  and Skill evolution were disabled.

The run manifest records base commit
`6a024995164b6f86bd937502bfc8cfb19d51ea1d`. The formal run used the source
snapshot committed on this backup branch in addition to that base. The later
`interface_only_prompt_ablation_v1` change to `director.py` is intentionally
excluded because it was created after the v6 run; v6 used
`agentgraph.director.minimal-neutral-stepwise-react.v1` from the base commit.

## Included evidence

- v6 evaluation YAML
- Markdown and JSON evaluation reports
- run manifest and preflight receipt
- fixed selected-task manifest
- Direct predictions and paired results
- wrong demos and collection-failure ledger
- compact evidence snapshots

The EvalPlus cache is reproducible and is not included.

## Large local trajectories

The two complete trajectory files are retained in the original local worktree
but are not committed to ordinary Git because each exceeds 100 MB:

- `artifacts/mbppplus_react_feedback_v11_runtimefix_thinking_terminalfix_v6/evaluation/agentgraph_trajectories.jsonl`
- `artifacts/mbppplus_react_feedback_v11_runtimefix_thinking_terminalfix_v6/evaluation/evidence/trajectories.jsonl`

The compact committed evidence is sufficient to recover the evaluated code and
configuration and to verify the reported aggregate and paired results. The full
per-turn local trajectories remain necessary for detailed turn-level replay.

## Recovery entry point

After checking out this branch, use:

```bash
PYTHONPATH=. python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_mbppplus_react_feedback_v11_runtimefix_thinking_terminalfix_v6.yaml \
  --prepare-only
```

Running a new model evaluation is intentionally separate from restoring this
snapshot and requires the configured local Qwen3.5-9B SGLang endpoint.
