# HotpotQA Architecture Validation — Round 01

Fixed project-held-out samples: **128**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 72.66 | 81.75 |
| AgentGraph | 128 | 128 | 68.75 | 79.53 |

AgentGraph − Direct: **-3.91 EM**, **-2.22 F1**.

Round-01 saved AgentGraph outputs rescored with the same official answer evaluator: **75.00 EM**, **83.95 F1**.

Terminal failures: **3**.

## Dynamic retrieval Tool

- Tool-invoked tasks: **128**
- Calls: **288** (`search`=144, `read`=144)
- Successful / failed calls: **288 / 0**
- Tasks with query rewriting: **10**
- Director Tool calls: **0**
- Worker retrieval Tool calls: **288**
- Retrieval artifact routed via AgentGraph relation: **True**
- QA-memory records / unique sources / cycled: **512 / 512 / 0**

## Failure types

- `architecture_gain`: 7
- `architecture_regression_candidate`: 10
- `correct`: 81
- `director_max_rounds`: 3
- `executor_or_provider_failure`: 1
- `partial_or_overlong_answer`: 15
- `shared_reasoning_or_model_failure_candidate`: 11

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
