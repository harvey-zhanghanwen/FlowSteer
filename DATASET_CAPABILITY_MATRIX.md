# Dataset Capability Matrix

Recorded from the source tree on 2026-08-20.  This matrix separates three
different claims:

- **implemented** means the Runtime/adapter and its unit boundary exist;
- **configured** means a frozen evaluation condition exists; and
- **live-validated** means that condition produced evaluator-valid trajectories.

An older result is not evidence that the new Tool/ReAct/Coding condition is
live-validated.  In particular, the saved HotpotQA, TriviaQA, AIME,
HealthBench and WebShop scores below predate the unified task-scoped Runtime.

## Shared architecture boundary

| Boundary | Current status | Source classification |
| --- | --- | --- |
| Progressive Canvas | `AgentWorkflowEnv.step` retains FlowSteer's one accepted atomic Canvas edit followed by execution of the current graph and execution feedback to the next Director turn. `ADD_SUBGRAPH` may add one component containing multiple Agents; `FINISH` remains explicit and is distinct from `max_rounds`. | **Direct reuse:** FlowSteer `InteractiveWorkflowEnv.step`, workflow state and trajectory loop. **Necessary adaptation:** free AgentGraph actions, quotient-DAG validation and finite two-Agent reciprocal blocks. |
| Unified Agent execution | `AgentRuntime` dispatches `reasoning`, `react`, and `coding` nodes through one scheduler and one model registry. Task-scoped Tool runtimes are selected by the explicit evaluation condition; only one stateful environment or repository owner is admitted in a graph. | **Necessary adaptation:** SkillFlow bounded execution inside the retained FlowSteer scheduler. |
| Agent communication | `CommunicationEnvelope` carries source/target Agent IDs, artifact type/body, graph revision, optional environment revision, dependency, and Tool receipts. Legacy text fields remain for trajectory compatibility. | **Project algorithm addition:** typed cross-Agent envelopes are not represented by upstream FlowSteer or SkillFlow. |
| Tool boundary | `ToolRegistry` admits immutable resource IDs with dataset scope, schema, side-effect, timeout and version metadata. `ToolReactExecutionAdapter` executes one `StructuredAction` per model turn and returns only the public observation. | **Direct reuse:** SkillFlow Tool/StructuredAction contracts. **Necessary adaptation:** asynchronous timeout and task-scoped registry. **Project algorithm addition:** `ToolCapability` and measured `ToolReceipt`. |
| Trajectory receipts | `rollout_collector.py` persists provider/model calls, token and latency metadata, ReAct traces, Tool receipts, environment reset/transition receipts, environment revision, evaluator replay trace, and Coding Agent receipts in the existing trajectory boundary. | **Direct reuse:** FlowSteer action masks/turn records and SkillFlow rollout artifacts. **Necessary adaptation:** JSON-safe heterogeneous execution metadata. |
| Native evaluator boundary | `task_evaluator.py` keeps gold answers, accepted aliases, rubrics, environment reward/`won`, and SWE-bench resolution outside model-visible Runtime state. Invalid evaluator receipts are excluded rather than replaced by proxy scores. | **Necessary adaptation:** benchmark-native evaluators are normalized behind one `evaluate_task` interface. |
| Model capability admission | `probe_model_capabilities.py` first requires an exact `/v1/models` ID and then probes Text, `StructuredAction`, and Coding-format compatibility without alias substitution or silent fallback. `model_catalog_multidataset_tool_v1.yaml` contains the remote models admitted by the saved canary; the current local Qwen3.5-9B service canary is still pending. | **Necessary adaptation:** provider discovery/capability receipt for the heterogeneous Executor catalog. The Flow-Director remains local Qwen3.5-9B. |
| Skill and training boundary | Skill schemas, lifecycle, retrieval and the project evidence gate exist, but every new multidataset Stable Zero configuration has Skills, GRPO, backward, optimizer updates and policy publication disabled. | **Direct reuse:** SkillFlow evidence/library primitives. **Project algorithm addition:** paired AgentGraph effect/posterior gate. **Not implemented/executed for this phase:** evidence-gated `ACTIVE` Skills and micro-training. |

## Seven-dataset Runtime matrix

| Dataset | Direct/Simple baseline | AgentGraph execution and Tool | Native evaluator / primary metric | Implementation and validation state |
| --- | --- | --- | --- | --- |
| HotpotQA | Local Qwen3.5-9B, one closed-context call over the supplied ten passages | Model-driven `search`/`read` through SkillFlow `RetrievalIndex`, bounded by `ToolReactExecutionAdapter`; the Director chooses the AgentGraph and Tool assignment | `hotpotqa.official.answer.v1`; normalized Exact Match and token F1 | Runtime wiring, frozen two-task canary condition and generic runner support are **implemented/configured**. Tool-enabled Stable Zero is **not yet live-validated**. Closed-context Direct and retrieval-enabled AgentGraph are different protocols and must be reported separately. |
| TriviaQA | Local Qwen3.5-9B, one question-only call | Model-driven `search`/`read` through the same frozen SkillFlow `RetrievalIndex`; the legacy deterministic question-query prefetch is not used by this condition | `triviaqa.official.answer.v1`; maximum normalized Exact Match/F1 over accepted aliases | Runtime wiring, frozen two-task canary condition and generic runner support are **implemented/configured**. Tool-enabled Stable Zero is **not yet live-validated**. Question-only Direct and retrieval-enabled AgentGraph are different protocols. |
| AIME 2026 | Local Qwen3.5-9B, one integer submission | Model-driven calculator and bounded child-process Python execution through `ToolReactExecutionAdapter`; Tool observations carry no reward | `skillflow.protocol-v10.static.integer.v1`; strict integer Exact Match/accuracy | Computation tools, runner wiring and frozen two-task canary condition are **implemented/configured**. Computation-enabled Stable Zero is **not yet live-validated**. |
| HealthBench Professional | Local Qwen3.5-9B, one healthcare response | Model-driven search over the frozen SkillFlow MedRAG textbooks BM25 corpus; corpus identity/revision/row count are checked at open time and rubrics are not Tool inputs | `openai.simple-evals.healthbench.v1`; rubric mean raw score with the configured reference judge | MedRAG Tool, runner wiring and frozen two-task canary condition are **implemented/configured**. MedRAG-enabled Stable Zero is **not yet live-validated**. |
| WebShop | Single local Qwen3.5-9B ReAct policy under the same environment and step budget | Request-scoped SkillFlow RAGEN episode; only admissible `search[...]`/`click[...]` actions and public observations are model-visible; evaluator replays the recorded transition trace | `skillflow.ragen_adapter.v2`; official environment return/success | Environment adapter, replay boundary, runner wiring and frozen two-task canary condition are **implemented/configured**. The new condition is **not yet live-validated**. |
| ALFWorld | Single local Qwen3.5-9B ReAct policy under the same game and step budget | Request-scoped SkillFlow RAGEN episode with admissible simulator actions, public observations and evaluator-locked task identity | `skillflow.ragen_adapter.v2`; terminal `won`/success | Environment adapter, replay boundary, runner wiring and frozen two-task canary condition are **implemented/configured**. No evaluator-valid live Stable Zero episode exists yet. |
| SWE-bench Verified | One bounded Coding Agent under the same repository, tools and test budget | Detached task/base-commit worktree; repository-relative list/search/view/exact-edit/diff/test tools; iterative Coding Agent completion requires a real edit, test call and inspected changed workspace diff | `swebench.harness.v1`; official Docker harness `resolved` | Worktree lifecycle, repository Tool registry, Coding Agent, runner wiring and frozen two-task condition are **implemented/configured**. The condition is only `prepared`: no live Coding trajectory or official-harness `resolved` receipt exists. |

## Historical results kept separate from the new Runtime

| Dataset | Historical fixed evaluation | Saved result | Why it is not a new-condition result |
| --- | --- | --- | --- |
| HotpotQA | 128/128 evaluator-valid | Direct EM 72.66%, F1 82.08%; AgentGraph EM 75.00%, F1 84.44% | Closed-context architecture run before model-driven QA Tool wiring |
| TriviaQA | 128/128 evaluator-valid | Direct EM 51.56%, F1 57.90%; AgentGraph EM 52.34%, F1 61.80% | Used the legacy deterministic retrieval-prefetch boundary |
| AIME 2026 | 30/30 evaluator-valid official 2026 tasks | Direct 1/30 (3.33%); AgentGraph 13/30 (43.33%) | Run before calculator/Python Tool wiring |
| HealthBench Professional | 128/128 evaluator-valid | Direct mean raw score 0.1318; AgentGraph 0.2075 | Run before the frozen MedRAG Tool boundary |
| WebShop | 126/128 evaluator-valid | Direct success 24.22%; AgentGraph strict success 22.66% | Legacy evaluator-owned interaction; two operational failures remain |
| ALFWorld | No evaluator-valid saved run | Not available | No live episode |
| SWE-bench Verified | No official resolved run | Not available | Official harness result unavailable |

## Protocol boundaries and remaining work

1. The deterministic 128-held-out/512-training views are project splits, not
   automatically official leaderboard evaluations. AIME's separate 30-task
   official-2026 view remains distinct.
2. QA Direct and retrieval-enabled AgentGraph conditions are not
   protocol-equivalent. Report each protocol independently; do not interpret
   their score difference as a paired architecture effect.
3. WebShop and ALFWorld receive success only from the RAGEN environment.
   Runtime traces are replayed by the evaluator; model-visible observations do
   not contain reward or `won`.
4. SWE-bench receives `resolved` only from the official harness. A generated
   diff, model judgement, or local proxy test is not a resolved instance.
5. All seven fixed conditions have `prepare-only` manifests and each declares
   a two-task canary. Run the actual canaries before claiming a unified Runtime
   Stable Zero result; for SWE-bench, the official Docker harness must also
   return a valid `resolved` receipt.
6. `ACTIVE Skill = 0` for this multidataset phase. The evidence gate, Skill
   activation, micro-training, optimizer update and policy synchronization
   have not been executed and must not be reported as completed.
