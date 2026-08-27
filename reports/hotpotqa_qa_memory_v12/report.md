# HotpotQA Architecture Validation — Dynamic Embedding Retrieval

Fixed project-held-out samples: **128**. The Director receives the original question and control-plane Canvas receipts only. Tool-capable worker Agents dynamically search/read the global train-only QA-memory and route evidence through graph relations. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 72.66 | 81.75 |
| AgentGraph | 123 | 123 | 15.62 | 20.66 |

AgentGraph − Direct: **-57.03 EM**, **-61.09 F1**.

Round-01 saved AgentGraph outputs rescored with the same official answer evaluator: **75.00 EM**, **83.95 F1**.

Terminal failures: **25**.

## Dynamic retrieval Tool

- Tool-invoked tasks: **123**
- Calls: **266** (`search`=133, `read`=133)
- Successful / failed calls: **266 / 0**
- Tasks with query rewriting: **6**
- Director Tool calls: **0**
- Worker retrieval Tool calls: **266**
- Retrieval artifact routed via AgentGraph relation: **True**
- QA-memory records / unique sources / cycled: **512 / 512 / 0**

## Failure types

- `agentgraph_operational_failure`: 5
- `architecture_gain`: 3
- `architecture_regression_candidate`: 57
- `correct`: 17
- `director_max_rounds`: 20
- `partial_or_overlong_answer`: 5
- `shared_reasoning_or_model_failure_candidate`: 21

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
