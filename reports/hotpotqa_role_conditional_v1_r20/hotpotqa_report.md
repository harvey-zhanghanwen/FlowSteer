# HotpotQA Architecture Validation — hotpotqa_role_conditional_v1_r20

Evaluation split: **validation**; fixed project validation samples: **128**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian update, or Skill publication ran. No Skill was injected.

Metric scope: **official-compatible answer-only EM/F1**. Supporting-fact and joint metrics are unavailable because this run does not emit formal supporting-fact predictions.

AgentGraph explicit FINISH: **60/128**; natural non-FINISH terminal failures: **67**; operational/evaluator failures: **1**.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 70.31 | 78.68 |
| AgentGraph | 127 | 127 | 38.28 | 41.22 |

AgentGraph − Direct: **-32.03 EM**, **-37.46 F1**.

## Failure types

- `agentgraph_operational_failure`: 1
- `architecture_gain`: 6
- `architecture_regression_candidate`: 1
- `correct`: 43
- `director_max_rounds`: 66
- `director_no_admissible_action`: 1
- `executor_or_provider_failure`: 6
- `partial_or_overlong_answer`: 2
- `shared_reasoning_or_model_failure_candidate`: 2

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
