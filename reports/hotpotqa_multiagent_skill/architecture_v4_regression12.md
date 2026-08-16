# HotpotQA Architecture Validation — hotpotqa_multiagent_v4_regression12

Fixed project-held-out samples: **12**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

AgentGraph explicit FINISH: **12/12**; natural max-round terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 12 | 12 | 58.33 | 70.56 |
| AgentGraph | 12 | 12 | 25.00 | 46.67 |

AgentGraph − Direct: **-33.33 EM**, **-23.89 F1**.

## Failure types

- `architecture_gain`: 1
- `architecture_regression_candidate`: 5
- `correct`: 2
- `partial_or_overlong_answer`: 2
- `shared_reasoning_or_model_failure_candidate`: 2

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
