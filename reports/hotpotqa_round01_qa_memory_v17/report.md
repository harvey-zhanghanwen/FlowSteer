# HotpotQA Architecture Validation — Round 01

Fixed project-held-out samples: **128**. The model input uses all ten supplied passages. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 72.66 | 81.75 |
| AgentGraph | 127 | 127 | 68.75 | 78.62 |

AgentGraph − Direct: **-3.91 EM**, **-3.13 F1**.

Round-01 saved AgentGraph outputs rescored with the same official answer evaluator: **75.00 EM**, **83.95 F1**.

Terminal failures: **2**.

## Dynamic retrieval Tool

- Tool-invoked tasks: **127**
- Calls: **417** (`search`=139, `read`=278)
- Successful / failed calls: **417 / 0**
- Tasks with query rewriting: **7**
- Director Tool calls: **0**
- Worker retrieval Tool calls: **417**
- Retrieval artifact routed via AgentGraph relation: **True**
- QA-memory records / unique sources / cycled: **512 / 512 / 0**

## Failure types

- `agentgraph_operational_failure`: 1
- `architecture_gain`: 4
- `architecture_regression_candidate`: 9
- `correct`: 70
- `director_max_rounds`: 1
- `executor_or_provider_failure`: 23
- `partial_or_overlong_answer`: 9
- `shared_reasoning_or_model_failure_candidate`: 11

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
