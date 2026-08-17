# HotpotQA Architecture Validation — hotpotqa_incremental_graph_v9_2_train16

Evaluation split: **train**; frozen architecture-development samples: **16**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

Metric scope: **official-compatible answer-only EM/F1**. Supporting-fact and joint metrics are unavailable because this run does not emit formal supporting-fact predictions.

AgentGraph explicit FINISH: **16/16**; natural max-round terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 16 | 16 | 75.00 | 90.91 |
| AgentGraph | 16 | 16 | 75.00 | 83.42 |

AgentGraph − Direct: **+0.00 EM**, **-7.50 F1**.

## Failure types

- `architecture_gain`: 2
- `architecture_regression_candidate`: 2
- `correct`: 7
- `executor_or_provider_failure`: 3
- `partial_or_overlong_answer`: 2

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
