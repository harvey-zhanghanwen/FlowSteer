# HotpotQA Architecture Validation — hotpotqa_incremental_graph_v9_6_regression8

Evaluation split: **train**; frozen architecture-development samples: **8**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

Metric scope: **official-compatible answer-only EM/F1**. Supporting-fact and joint metrics are unavailable because this run does not emit formal supporting-fact predictions.

AgentGraph explicit FINISH: **8/8**; natural max-round terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 8 | 8 | 12.50 | 45.83 |
| AgentGraph | 8 | 8 | 12.50 | 52.08 |

AgentGraph − Direct: **+0.00 EM**, **+6.25 F1**.

## Failure types

- `architecture_regression_candidate`: 1
- `correct`: 1
- `partial_or_overlong_answer`: 5
- `shared_reasoning_or_model_failure_candidate`: 1

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
