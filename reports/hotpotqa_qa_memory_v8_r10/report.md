# HotpotQA Architecture Validation — Dynamic Embedding Retrieval

Fixed project-held-out samples: **128**. The Director receives the original question and control-plane Canvas receipts only. Tool-capable worker Agents dynamically search/read the global train-only QA-memory and route evidence through graph relations. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 72.66 | 81.75 |
| AgentGraph | 128 | 128 | 0.78 | 0.78 |

AgentGraph − Direct: **-71.88 EM**, **-80.97 F1**.

Round-01 saved AgentGraph outputs rescored with the same official answer evaluator: **75.00 EM**, **83.95 F1**.

Terminal failures: **127**.

## Dynamic retrieval Tool

- Tool-invoked tasks: **122**
- Calls: **389** (`search`=371, `read`=18)
- Successful / failed calls: **389 / 0**
- Tasks with query rewriting: **102**
- Director Tool calls: **0**
- Worker retrieval Tool calls: **389**
- Retrieval artifact routed via AgentGraph relation: **True**
- QA-memory records / unique sources / cycled: **512 / 512 / 0**

## Failure types

- `correct`: 1
- `director_max_rounds`: 1
- `knowledge_base_coverage_failure`: 119
- `retrieval_strategy_failure`: 7

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
