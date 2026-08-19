# Dataset Capability Matrix

Recorded from the source tree and receipt-backed Stable Zero runs on
2026-08-20. This matrix separates three
different claims:

- **implemented** means the Runtime/adapter and its unit boundary exist;
- **configured** means a frozen evaluation condition exists; and
- **live-validated** means that condition produced evaluator-valid trajectories.

Historical results remain separate from the new Tool/ReAct/Coding conditions.
The current live-validation claims below refer only to the fixed two-task
Stable Zero receipts and are not formal benchmark estimates.

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
| Skill and training boundary | Skill schemas, lifecycle, retrieval and the project evidence gate exist. The latest independent paired evidence produced two `CANDIDATE` Skills and zero `ACTIVE` Skills, so every multidataset Stable Zero configuration keeps Skill injection, GRPO, backward, optimizer updates and policy publication disabled. | **Direct reuse:** SkillFlow evidence/library primitives. **Project algorithm addition:** paired AgentGraph effect/posterior gate. **Not executed for this phase:** Skill injection and micro-training because the evidence gate did not approve an `ACTIVE` Skill. |

## Seven-dataset Runtime matrix

| Dataset | Direct/Simple baseline | AgentGraph execution and Tool | Native evaluator / primary metric | Implementation and validation state |
| --- | --- | --- | --- | --- |
| HotpotQA | Local Qwen3.5-9B, one closed-context call over the supplied ten passages | Model-driven `search`/`read` through SkillFlow `RetrievalIndex`, bounded by `ToolReactExecutionAdapter`; the Director chooses the AgentGraph and Tool assignment | `hotpotqa.official.answer.v1`; normalized Exact Match and token F1 | **Live-validated, 2/2 Stable Zero:** Direct EM/F1 100%/100%; AgentGraph 100%/100%; explicit FINISH and evaluator receipts 2/2. Optional retrieval Tool calls were 0, so Tool use itself is not validated. The two arms remain protocol-separated. |
| TriviaQA | Local Qwen3.5-9B, one question-only call | Model-driven `search`/`read` through the same frozen SkillFlow `RetrievalIndex`; the legacy deterministic question-query prefetch is not used by this condition | `triviaqa.official.answer.v1`; maximum normalized Exact Match/F1 over accepted aliases | **Live-validated, 2/2 Stable Zero:** Direct EM/F1 50%/50%; AgentGraph 50%/92.86%; explicit FINISH and evaluator receipts 2/2. Optional retrieval Tool calls were 0; one AgentGraph answer failed EM at the terminal Format boundary. The two arms remain protocol-separated. |
| AIME 2026 | Local Qwen3.5-9B, one integer submission | Model-driven calculator and bounded child-process Python execution through `ToolReactExecutionAdapter`; Tool observations carry no reward | `skillflow.protocol-v10.static.integer.v1`; strict integer Exact Match/accuracy | **Live-validated, 2/2 Stable Zero:** Direct accuracy 50%; AgentGraph 100%; explicit FINISH and evaluator receipts 2/2. Optional computation Tool calls were 0, so computation Tool use itself is not validated. |
| HealthBench Professional | Local Qwen3.5-9B, one healthcare response | Model-driven search over the frozen SkillFlow MedRAG textbooks BM25 corpus; corpus identity/revision/row count are checked at open time and rubrics are not Tool inputs | `openai.simple-evals.healthbench.v1`; rubric mean raw score with the configured reference judge | **Live-validated, 2/2 Stable Zero:** Direct and AgentGraph mean raw_score are both 0.20; explicit FINISH and evaluator receipts 2/2. One MedRAG Tool call returned `ValueError`, so successful MedRAG use is not validated. |
| WebShop | One ReAct policy under the same environment and step budget | Request-scoped SkillFlow RAGEN episode; only admissible `search[...]`/`click[...]` actions and public observations are model-visible; evaluator replays the recorded transition trace | `skillflow.ragen_adapter.v2`; official environment return/success | **Live-validated, v2 2/2 Stable Zero:** Direct and AgentGraph success are both 100%. The v1 JSON/native-action mismatch was corrected by making the native environment action grammar executor-authoritative; v2 recorded 6/6 valid environment transitions. |
| ALFWorld | One ReAct policy under the same game and step budget | Request-scoped SkillFlow RAGEN episode with admissible simulator actions, public observations and evaluator-locked task identity | `skillflow.ragen_adapter.v2`; terminal `won`/success | **Live-validated, v2 2/2 Stable Zero:** Direct success 50%; AgentGraph success 100%. Interactive FINISH now requires exactly one ReAct environment actor; v2 recorded 10/10 valid environment transitions. |
| SWE-bench Verified | One bounded Coding Agent under the same repository, tools and test budget | Detached task/base-commit worktree; repository-relative list/search/view/exact-edit/diff/test tools; iterative Coding Agent completion requires a real edit, test call and inspected changed workspace diff | `swebench.harness.v1`; official Docker harness `resolved` | Worktree lifecycle, repository Tool registry, Coding Agent and frozen condition are **implemented/configured**. Two fixed `astropy/astropy` worktree create/cleanup canaries passed, but the official Docker harness failed runtime preflight. No model/API/Coding trajectory ran and `resolved_rate` is **unmeasurable**, not zero. |

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
5. Six fixed conditions produced two-task evaluator-valid Stable Zero
   receipts. SWE-bench stopped fail-closed at official Docker harness preflight;
   it has no Coding trajectory or valid `resolved` receipt and therefore keeps
   `ALL_DATASETS_STABLE_ZERO_COMPLETE = NO`.
6. `ACTIVE Skill = 0` for this multidataset phase. The evidence gate did run:
   HotpotQA and TriviaQA candidates were rejected by the calibrated
   lower-bound/harm criteria. Consequently Skill injection, micro-training,
   optimizer update and policy synchronization were not executed and must not
   be reported as completed.
