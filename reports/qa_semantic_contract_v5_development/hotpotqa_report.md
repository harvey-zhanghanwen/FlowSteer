# HotpotQA Architecture Validation — hotpotqa_semantic_contract_v5_development

Evaluation split: **validation**; fixed project validation samples: **128**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian update, or Skill publication ran. No Skill was injected.

Metric scope: **official-compatible answer-only EM/F1**. Supporting-fact and joint metrics are unavailable because this run does not emit formal supporting-fact predictions.

AgentGraph explicit FINISH: **126/128**; natural max-round terminal failures: **2**; operational/evaluator failures: **0**.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 70.31 | 78.68 |
| AgentGraph | 128 | 128 | 73.44 | 83.40 |

AgentGraph − Direct: **+3.12 EM**, **+4.72 F1**.

## Failure types

- `architecture_gain`: 14
- `architecture_regression_candidate`: 9
- `correct`: 80
- `director_max_rounds`: 2
- `executor_or_provider_failure`: 2
- `partial_or_overlong_answer`: 14
- `shared_reasoning_or_model_failure_candidate`: 7

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
