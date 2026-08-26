# HotpotQA Architecture Validation — Dynamic Embedding Retrieval

Fixed project-held-out samples: **128**. The Director and Agent Runtime receive only the original question; public passages are obtained dynamically through the task-scoped embedding search/read Tool. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Completed | Valid | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | 128 | 72.66 | 81.75 |
| AgentGraph | 128 | 128 | 28.91 | 33.97 |

AgentGraph − Direct: **-43.75 EM**, **-47.78 F1**.

Round-01 saved AgentGraph outputs rescored with the same official answer evaluator: **75.00 EM**, **83.95 F1**.

Terminal failures: **73**.

## Dynamic retrieval Tool

- Tool-invoked tasks: **60**
- Calls: **254** (`search`=125, `read`=129)
- Successful / failed calls: **250 / 4**
- Tasks with query rewriting: **57**

## Failure types

- `architecture_gain`: 3
- `correct`: 3
- `director_max_rounds`: 73
- `executor_or_provider_failure`: 48
- `partial_or_overlong_answer`: 1

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
