# HotpotQA Architecture Validation — hotpotqa_training_ready_step0_untouched32

Fixed project-held-out samples: **32**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 32 | 32 | 78.12 | 87.77 |
| AgentGraph | 32 | 32 | 71.88 | 83.62 |

AgentGraph − Direct: **-6.25 EM**, **-4.15 F1**.

## Failure types

- `architecture_gain`: 2
- `architecture_regression_candidate`: 4
- `correct`: 21
- `partial_or_overlong_answer`: 5

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
