# HotpotQA Architecture Validation — hotpotqa_multiagent_v1_architecture_diagnostic

Fixed project-held-out samples: **14**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 14 | 14 | 42.86 | 46.43 |
| AgentGraph | 14 | 14 | 35.71 | 52.86 |

AgentGraph − Direct: **-7.14 EM**, **+6.43 F1**.

## Failure types

- `architecture_gain`: 1
- `architecture_regression_candidate`: 2
- `correct`: 4
- `partial_or_overlong_answer`: 2
- `shared_reasoning_or_model_failure_candidate`: 5

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
