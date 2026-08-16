# HotpotQA Architecture Validation — hotpotqa_multiagent_v3_dev128

Fixed project-held-out samples: **128**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

AgentGraph explicit FINISH: **127/128**; natural max-round terminal failures: **1**; operational/evaluator failures: **0**.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 72.66 | 82.08 |
| AgentGraph | 128 | 128 | 67.97 | 80.23 |

AgentGraph − Direct: **-4.69 EM**, **-1.85 F1**.

## Failure types

- `architecture_gain`: 10
- `architecture_regression_candidate`: 16
- `correct`: 77
- `director_max_rounds`: 1
- `partial_or_overlong_answer`: 17
- `shared_reasoning_or_model_failure_candidate`: 7

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
