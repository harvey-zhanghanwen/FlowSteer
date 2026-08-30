# HotpotQA Architecture Validation — Dynamic Embedding Retrieval

Fixed evaluation samples: **128**. Evaluation scope: **in_database_transductive**. The Director receives the public task and control-plane receipts only. Tool-capable worker Agents dynamically search/read the full-dataset declarative-fact memory and route evidence through explicit graph relations. No training, backward pass, optimizer step, policy update, MACE, Bayesian, or Skill loop ran.

| Condition | Evaluator-valid | Explicit FINISH | Strict EM | Strict F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct (local) | 128 | N/A | 72.66 | 81.75 |
| AgentGraph | 128 | 127 | 79.69 | 89.55 |

AgentGraph − Direct: **+7.03 EM**, **+7.80 F1**.

Round-01 saved AgentGraph outputs rescored with the same official answer evaluator: **75.00 EM**, **83.95 F1**.

Terminal failures: **1** (`max_rounds`). Here, `agentgraph.completed=128` in
`report.json` denotes available/evaluator-valid trajectories, not 128 explicit
`FINISH` actions.

## Dynamic retrieval Tool

- Tool-invoked tasks: **128**
- Calls: **390** (`search`=130, `read`=260)
- Successful / failed calls: **390 / 0**
- Tasks with query rewriting: **2**
- Director Tool calls: **0**
- Worker retrieval Tool calls: **390**
- Retrieval artifact routed via AgentGraph relation: **True**
- Retrieval records / unique sources / cycled: **97852 / 97852 / 0**

## Failure types

- `architecture_gain`: 15
- `architecture_regression_candidate`: 5
- `correct`: 87
- `director_max_rounds`: 1
- `partial_or_overlong_answer`: 15
- `shared_reasoning_or_model_failure_candidate`: 5

## Interpretation boundary

The paper numbers supplied by the user are references, not paired results. Root-cause claims must be based on the saved Director actions, graph snapshots, Agent inputs/outputs, communication receipts, output-agent inbox, evaluator receipt, tokens, latency, and API attempts in this round's artifacts.
