# HotpotQA Architecture Validation — Round 01

Fixed project-held-out samples: **128**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 72.66 | 81.75 |
| AgentGraph | 128 | 128 | 1.56 | 1.56 |

AgentGraph − Direct: **-71.09 EM**, **-80.19 F1**.

Round-01 saved AgentGraph outputs rescored with the same official answer evaluator: **75.00 EM**, **83.95 F1**.

Terminal failures: **124**.

## Dynamic retrieval Tool

- Tool-invoked tasks: **72**
- Calls: **243** (`search`=81, `read`=162)
- Successful / failed calls: **243 / 0**
- Tasks with query rewriting: **7**
- Director Tool calls: **0**
- Worker retrieval Tool calls: **243**
- Retrieval artifact routed via AgentGraph relation: **True**
- QA-memory records / unique sources / cycled: **512 / 512 / 0**

## Failure types

- `architecture_regression_candidate`: 1
- `correct`: 1
- `director_max_rounds`: 124
- `executor_or_provider_failure`: 1
- `shared_reasoning_or_model_failure_candidate`: 1

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
