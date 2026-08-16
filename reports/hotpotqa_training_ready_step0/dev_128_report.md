# HotpotQA Architecture Validation — hotpotqa_training_ready_step0_dev128

Fixed project-held-out samples: **128**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 72.66 | 82.08 |
| AgentGraph | 128 | 128 | 73.44 | 81.62 |

AgentGraph − Direct: **+0.78 EM**, **-0.46 F1**.

## Failure types

- `architecture_gain`: 10
- `architecture_regression_candidate`: 9
- `correct`: 83
- `executor_or_provider_failure`: 1
- `partial_or_overlong_answer`: 14
- `shared_reasoning_or_model_failure_candidate`: 11

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
