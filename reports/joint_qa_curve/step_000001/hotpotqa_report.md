# HotpotQA Architecture Validation — joint_qa_matched_control_step1_hotpotqa

Evaluation split: **validation**; fixed project validation samples: **32**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

Metric scope: **official-compatible answer-only EM/F1**. Supporting-fact and joint metrics are unavailable because this run does not emit formal supporting-fact predictions.

AgentGraph explicit FINISH: **31/32**; natural max-round terminal failures: **1**; operational/evaluator failures: **0**.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 32 | 32 | 68.75 | 78.18 |
| AgentGraph | 32 | 32 | 68.75 | 79.69 |

AgentGraph − Direct: **+0.00 EM**, **+1.51 F1**.

## Failure types

- `architecture_gain`: 2
- `architecture_regression_candidate`: 1
- `correct`: 20
- `director_max_rounds`: 1
- `partial_or_overlong_answer`: 5
- `shared_reasoning_or_model_failure_candidate`: 3

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
