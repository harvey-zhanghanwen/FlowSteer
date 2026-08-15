# HotpotQA Architecture Validation — Round 01

Fixed project-held-out samples: **128**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 72.66 | 82.08 |
| AgentGraph | 128 | 128 | 75.00 | 84.44 |

AgentGraph − Direct: **+2.34 EM**, **+2.36 F1**.

完整的 Workflow/通信核对、Wrong Demo 首错位置与根因分层见
[`diagnostic_report.md`](diagnostic_report.md)。

## Failure types

- `architecture_gain`: 10
- `architecture_regression_candidate`: 7
- `correct`: 86
- `partial_or_overlong_answer`: 16
- `shared_reasoning_or_model_failure_candidate`: 9

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
