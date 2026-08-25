# HotpotQA Architecture Validation — hotpotqa_role_conditional_v1_r6

Evaluation split: **validation**; fixed project validation samples: **128**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian update, or Skill publication ran. No Skill was injected.

Metric scope: **official-compatible answer-only EM/F1**. Supporting-fact and joint metrics are unavailable because this run does not emit formal supporting-fact predictions.

AgentGraph explicit FINISH: **52/128**; natural max-round terminal failures: **52**; operational/evaluator failures: **24**.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 70.31 | 78.68 |
| AgentGraph | 104 | 104 | 32.81 | 35.70 |

AgentGraph − Direct: **-37.50 EM**, **-42.98 F1**.

## Failure types

- `agentgraph_operational_failure`: 24
- `architecture_gain`: 6
- `correct`: 36
- `director_max_rounds`: 52
- `executor_or_provider_failure`: 4
- `partial_or_overlong_answer`: 3
- `shared_reasoning_or_model_failure_candidate`: 3

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
