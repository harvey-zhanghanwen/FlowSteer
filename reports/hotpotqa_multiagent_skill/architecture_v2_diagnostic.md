# HotpotQA Architecture Validation — hotpotqa_multiagent_v2_architecture_diagnostic

Fixed project-held-out samples: **14**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 14 | 14 | 42.86 | 48.57 |
| AgentGraph | 14 | 14 | 50.00 | 61.84 |

AgentGraph − Direct: **+7.14 EM**, **+13.27 F1**.

## Failure types

- `architecture_gain`: 3
- `architecture_regression_candidate`: 2
- `correct`: 4
- `partial_or_overlong_answer`: 1
- `shared_reasoning_or_model_failure_candidate`: 4

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
