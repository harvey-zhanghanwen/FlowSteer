# HotpotQA Architecture Validation — hotpotqa_interaction_v8_dev128

Fixed project-held-out samples: **128**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

AgentGraph explicit FINISH: **128/128**; natural max-round terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 72.66 | 82.08 |
| AgentGraph | 128 | 128 | 71.88 | 83.05 |

AgentGraph − Direct: **-0.78 EM**, **+0.97 F1**.

## Failure types

- `architecture_gain`: 11
- `architecture_regression_candidate`: 12
- `correct`: 81
- `partial_or_overlong_answer`: 15
- `shared_reasoning_or_model_failure_candidate`: 9

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
