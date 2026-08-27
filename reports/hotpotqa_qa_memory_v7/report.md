# HotpotQA Architecture Validation — Dynamic Embedding Retrieval

Fixed project-held-out samples: **128**. The Director receives the original question and control-plane Canvas receipts only. Tool-capable worker Agents dynamically search/read the global train-only QA-memory and route evidence through graph relations. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 72.66 | 81.75 |
| AgentGraph | 126 | 126 | 4.69 | 6.40 |

AgentGraph − Direct: **-67.97 EM**, **-75.35 F1**.

Round-01 saved AgentGraph outputs rescored with the same official answer evaluator: **75.00 EM**, **83.95 F1**.

Terminal failures: **88**.

## Dynamic retrieval Tool

- Tool-invoked tasks: **120**
- Calls: **1764** (`search`=882, `read`=882)
- Successful / failed calls: **1764 / 0**
- Tasks with query rewriting: **120**
- Director Tool calls: **0**
- Worker retrieval Tool calls: **1764**
- Retrieval artifact routed via AgentGraph relation: **True**
- QA-memory records / unique sources / cycled: **512 / 512 / 0**

## Failure types

- `agentgraph_operational_failure`: 2
- `architecture_regression_candidate`: 1
- `correct`: 1
- `director_max_rounds`: 86
- `executor_or_provider_failure`: 37
- `partial_or_overlong_answer`: 1

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
