# Dataset Capability Matrix

Recorded from the source tree and existing receipts on 2026-08-20. This
matrix separates four different claims:

- **implemented** means the Runtime/adapter and its unit boundary exist;
- **configured** means a frozen evaluation condition exists; and
- **historical diagnostic** means a prior condition produced receipts but is
  not evidence for the current clean development Stable Zero protocol; and
- **live-validated** means the current frozen condition produced
  evaluator-valid trajectories.

Historical results remain separate from the new Tool/ReAct/Coding conditions.
For fixed-action Tools, the per-action exact JSON Schema is implemented and
covered at the interface boundary. The current HotpotQA and TriviaQA v3
natural-action canaries each persisted one successful retrieval `ToolReceipt`,
but both reused two records from an architecture-development block that had
already been exercised by prior diagnostics; they are exposed development
canaries, not held-out benchmark evaluations;
AIME-2025 development and HealthBench v2 completed Stable Zero without
naturally selecting their optional Tools. Separate forced probes are marked
diagnostic-only and excluded from benchmark, GRPO and Skill evidence. Two-task
canaries and historical small-sample results are not formal benchmark estimates.

## Shared architecture boundary

| Boundary | Current status | Source classification |
| --- | --- | --- |
| Progressive Canvas | `AgentWorkflowEnv.step` retains FlowSteer's one accepted atomic Canvas edit followed by execution of the current graph and execution feedback to the next Director turn. `ADD_SUBGRAPH` may add one component containing multiple Agents; `FINISH` remains explicit and is distinct from `max_rounds`. | **Direct reuse:** FlowSteer `InteractiveWorkflowEnv.step`, workflow state and trajectory loop. **Necessary adaptation:** free AgentGraph actions, quotient-DAG validation and finite two-Agent reciprocal blocks. |
| Unified Agent execution | `AgentRuntime` dispatches `reasoning`, `react`, and `coding` nodes through one scheduler and one model registry. Task-scoped Tool runtimes are selected by the explicit evaluation condition; only one stateful environment or repository owner is admitted in a graph. | **Necessary adaptation:** SkillFlow bounded execution inside the retained FlowSteer scheduler. |
| Agent communication | `CommunicationEnvelope` carries source/target Agent IDs, artifact type/body, graph revision, optional environment revision, dependency, and Tool receipts. Legacy text fields remain for trajectory compatibility. | **Project algorithm addition:** typed cross-Agent envelopes are not represented by upstream FlowSteer or SkillFlow. |
| Tool boundary | `ToolRegistry` admits immutable resource IDs with dataset scope, side-effect, timeout and version metadata. `ToolCapability.action_schemas` stores an exact JSON Schema for each fixed Tool action; dynamic environments expose state-dependent admissible actions through their native action grammar. `ToolReactExecutionAdapter` executes one `StructuredAction` per model turn, carries the executed action plus public observation into continuation state, and bounds each action generation with SkillFlow's `max_action_tokens` boundary. HotpotQA and TriviaQA v3 each produced one successful natural retrieval `ToolReceipt`; AIME and HealthBench natural adoption remains absent. | **Direct reuse:** SkillFlow Tool/StructuredAction contracts, public action history and action-token budget. **Necessary adaptation:** per-action schema dispatch, asynchronous timeout and task-scoped registry. **Project algorithm addition:** `ToolCapability` and measured `ToolReceipt`. |
| Trajectory receipts | `rollout_collector.py` persists provider/model calls, token and latency metadata, ReAct traces, Tool receipts, environment reset/transition receipts, environment revision, evaluator replay trace, and Coding Agent receipts in the existing trajectory boundary. | **Direct reuse:** FlowSteer action masks/turn records and SkillFlow rollout artifacts. **Necessary adaptation:** JSON-safe heterogeneous execution metadata. |
| Native evaluator boundary | `task_evaluator.py` keeps gold answers, accepted aliases, rubrics, environment reward/`won`, and SWE-bench resolution outside model-visible Runtime state. Invalid evaluator receipts are excluded rather than replaced by proxy scores. | **Necessary adaptation:** benchmark-native evaluators are normalized behind one `evaluate_task` interface. |
| Runtime dataset registry | Historical conditions continue to use their receipt-bound explicit JSONL paths and frozen `selected_tasks`. Future conditions may opt in to `datasets_runtime_v2.yaml`; the runner then fail-closes on dataset key, split paths, task schema, preparation catalog/manifest schema, available provenance, and SWE-bench evaluator-source drift before task selection. | **Necessary adaptation:** the original generic seven-source catalog was superseded by dataset-specific preparation protocols. The versioned registry binds those existing catalogs/manifests without rewriting historical configurations or artifacts. |
| Model capability admission | `probe_model_capabilities.py` first requires an exact `/v1/models` ID and then probes Text, `StructuredAction`, and Coding-format compatibility without alias substitution or silent fallback. Historical Stable Zero trajectories remain receipt-bound to immutable `model_catalog_multidataset_tool_v1.yaml`. `model_catalog_multidataset_tool_v2.yaml` preserves the same exact entries and adds only the passed local Qwen3.5-9B non-thinking canary metadata for future conditions; it is not retroactively assigned to v1 trajectories. The WebShop v4 receipt records heterogeneous Executor use by `deepseek-v4-flash` and local Qwen3.5-9B; this is execution evidence for those two catalog entries only, not for the entire model pool. | **Necessary adaptation:** provider discovery/capability receipt for the heterogeneous Executor catalog. The Flow-Director remains local Qwen3.5-9B. Catalog precedence is SkillFlow's explicit provider/model and Qwen chat-template boundary, then the retained FlowSteer Runtime receipt boundary, then this minimal versioned admission adapter. |
| Skill and training boundary | Skill schemas, lifecycle, retrieval and the project evidence gate exist. ACTIVE-Skill retrieval applies SkillFlow's exact `required_tools ⊆ available_tools` applicability condition to task-scoped, available, dataset-compatible Tool IDs; natural retrieval and forced-probe matching share that predicate. Turn/trajectory records independently persist the version-compatible ACTIVE library and ranked Director-visible H0 retrieval/visibility. Executor-side `ActionKind.SKILL` remains fail-closed because neither SkillFlow's generic Skill document nor the current project Skill record supplies a versioned executable `resource_id/name/arguments` schema; that schema is not guessed. The latest independent paired evidence produced two `CANDIDATE` Skills and zero `ACTIVE` Skills; therefore the current multidataset conditions perform no Skill injection, GRPO, backward, optimizer update, or policy publication. Two earlier joint-QA bounded GRPO runs each performed one real LoRA optimizer update and policy synchronization, but their fixed held-out macro metrics did not improve over Step 0. | **Direct reuse:** SkillFlow evidence/library primitives, `SkillApplicability.required_tools` / `TaskRetrievalFeatures.available_tools`, `CanonicalInitialContextAssembler` ACTIVE/H0 identities, and canonical Skill-invocation admission. **Necessary adaptation:** projection from task-scoped `ToolRegistry` and `SkillStore` into FlowSteer Turn/Trajectory receipts. **Project algorithm addition:** paired AgentGraph effect/posterior gate. **Current phase:** Director-visible applicability/retrieval receipts are interface-tested, but there is no ACTIVE Skill, versioned Executor invocation schema, admitted invocation receipt, Skill effect, or current-condition optimizer update. **Prior core evidence:** the Director LoRA update/sync path is executable, but it does not validate learning for the new Tool/Environment/Coding action domains. |

## Seven-dataset Runtime matrix

| Dataset | Task Type | Evaluator | External Tool | ReAct | Coding Agent | Skill Type | 当前验证状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HotpotQA | Multi-hop, open-domain question answering | `hotpotqa.official.answer.v1`; normalized Exact Match and token F1 | Frozen Wikipedia retrieval index: `search`, `read` | Yes. Fixed actions use per-action exact JSON Schema. | No | Retrieval, multi-hop reasoning, answer formatting. `ACTIVE`: none. | v3 exposed development canary: 2/2 Stable Zero; Direct and AgentGraph EM/F1 are both 100%/100%. One natural `qa-retrieval.search` receipt was persisted. The v4 Tool-availability OFF/ON diagnostic is now live-complete on the same two exposed development tasks: 2 randomized pairs, 4 evaluator-valid trajectories, both arms EM/F1 100%/100%, mean availability-assignment ITT ΔEM=0 and ΔF1=0. OFF exposes no Tool catalog; ON exposes exact `qa-retrieval.search/read`, but both ON trajectories made zero Tool calls and produced zero ToolReceipt. Therefore v4 validates treatment exposure and paired runtime only—not Tool usefulness, Skill evidence, held-out accuracy, or a benchmark estimate. Direct reuse is exactly `2`, new Direct calls are `0`; `joint_qa_v2/test.jsonl` remains untouched. |
| TriviaQA | Open-domain factual question answering with accepted aliases | `triviaqa.official.answer.v1`; maximum normalized Exact Match and token F1 over accepted aliases | Frozen Wikipedia retrieval index: `search`, `read` | Yes. Fixed actions use per-action exact JSON Schema. | No | Retrieval, entity disambiguation, answer normalization. `ACTIVE`: none. | v3 exposed development canary: 2/2 Stable Zero; Direct EM/F1 50%/50%, AgentGraph EM/F1 100%/100%. These two records were reused during architecture development, so the values are not held-out metrics and the protocol-separated delta is not a causal architecture estimate. One successful natural retrieval `ToolReceipt` was persisted. `joint_qa_v2` now isolates development, train, Skill confirmation and test; its TriviaQA test partition has not been run. |
| AIME 2026 target | Competition mathematics with an integer answer | `skillflow.protocol-v10.static.integer.v1`; strict integer Exact Match/accuracy | Calculator and bounded `python_exec` | Yes. Fixed actions use per-action exact JSON Schema. | No; calculator and Python execution are Tool actions, not a Coding Agent. | Mathematical reasoning, calculation, verification, integer-answer formatting. `ACTIVE`: none. | Development canary actually uses two AIME 2025 tasks: 2/2 Stable Zero; Direct 0/2, AgentGraph 1/2 (50% accuracy). No computation Tool receipt was produced. The separate AIME 2026 official-test run remains evaluation-only and is not adaptation evidence. |
| HealthBench Professional | Open-ended clinical response generation | `openai.simple-evals.healthbench.v1`; public simple-evals-compatible reference-judge protocol and mean `raw_score` | Frozen MedRAG BM25 textbook index: `search` | Yes. Fixed actions use per-action exact JSON Schema. | No | Clinical evidence retrieval, clinical reasoning, safety, response completeness. `ACTIVE`: none. | v2 validation canary: 2/2 Stable Zero; Direct and AgentGraph mean `raw_score` are both 0.2. The reference response and rubric are evaluator-only. No MedRAG receipt was produced, and this reference-judge diagnostic is not an official private HealthBench evaluation or a benchmark estimate. |
| WebShop | Goal-conditioned interactive web shopping | `skillflow.ragen_adapter.v2`; native episode return/success | Native WebShop environment with state-dependent `search[...]` and `click[...]` actions | Yes. The environment supplies dynamic admissible actions through its native action grammar. | No | Goal-conditioned search, attribute filtering, action selection. `ACTIVE`: none. | Native validation v4 Stable Zero: 2/2 full-chain and explicit FINISH; Direct 1/2 success, AgentGraph 1/2 success. AgentGraph used `deepseek-v4-flash` and local Qwen Executors and reduced successful-task steps from 9 to 6. The v2 native-test run remains excluded; v3 is retained only as the pre-`max_action_tokens` failure diagnostic. |
| ALFWorld | Embodied text-environment task completion | `skillflow.ragen_adapter.v2`; native terminal `won`/success | Native ALFWorld environment with state-dependent admissible actions | Yes. The environment supplies dynamic admissible actions through its native action grammar. | No | Task planning, object-state tracking, action selection. `ACTIVE`: none. | v2 canary on two project-held-out tasks sampled from the native training population: 2/2 Stable Zero; Direct success 1/2 (50%), AgentGraph success 2/2 (100%). Native transitions are evaluator-valid, but this is not the official ALFWorld validation/test protocol or a benchmark estimate. |
| SWE-bench (Verified reserved for final evaluation) | Repository-level software issue resolution | `swebench.harness.v1`; official Docker harness `resolved`/resolved rate | Detached repository worktree with list, regex search, view, exact-edit, diff and test Tools | Yes, within the bounded Coding Agent Tool loop. Fixed actions use per-action exact JSON Schema. | Partial | The current surface covers bounded inspect/edit/test/re-edit/diff/finalize, but not the full requested symbol/reference, general command, multi-file apply-patch or create/delete/move contract. `ACTIVE`: none. | v2 split isolation is materialized: regular SWE-bench dev provides 128 validation tasks, regular train provides 512 training tasks, and all 500 Verified tasks are reserved for final evaluation; instance IDs are pairwise disjoint. Completion now requires a fresh `run_tests` and changed `diff` after the last changed edit, preventing stale patch submission, and the condition budget is 9 turns/8 Tool calls. Runtime preflight remains blocked by Docker harness availability; the fixed canary's `sqlfluff` repository mirror is also absent. Therefore no live Coding trajectory or official `resolved` receipt exists, and the metric remains unmeasurable rather than 0. |

## Historical results kept separate from the new Runtime

| Dataset | Historical fixed evaluation | Saved result | Why it is not a new-condition result |
| --- | --- | --- | --- |
| HotpotQA | 128/128 evaluator-valid | Direct EM 72.66%, F1 82.08%; AgentGraph EM 75.00%, F1 84.44% | Exposed development architecture run before model-driven QA Tool wiring; not the protected `joint_qa_v2` test partition |
| TriviaQA | 128/128 evaluator-valid | Direct EM 51.56%, F1 57.90%; AgentGraph EM 52.34%, F1 61.80% | Exposed development run using the legacy deterministic retrieval-prefetch boundary; not the protected `joint_qa_v2` test partition |
| AIME 2026 | 30/30 evaluator-valid official 2026 test tasks | Direct 1/30 (3.33%); AgentGraph 13/30 (43.33%) | **Official-test diagnostic**, not development Stable Zero; run before calculator/Python Tool wiring |
| HealthBench Professional | 128/128 evaluator-valid | Direct mean raw score 0.1318; AgentGraph 0.2075 | Run before the frozen MedRAG Tool boundary |
| WebShop | 126/128 evaluator-valid | Direct success 24.22%; AgentGraph strict success 22.66% | Legacy evaluator-owned interaction; two operational failures remain |
| ALFWorld | Pre-v2 condition | Not available | The current two-task native-environment v2 canary is reported in the Runtime matrix, not treated as a formal historical benchmark. |
| SWE-bench Verified | No official resolved run | Not available | Official harness result unavailable |

## Protocol boundaries and remaining work

1. The deterministic 128-validation/512-training views are project splits, not
   automatically official leaderboard evaluations. For HotpotQA and TriviaQA,
   `joint_qa_v2` now separates development `[0:128]`, train `[128:640]`,
   independent Skill confirmation `[672:736]`, and test `[736:864]` after a
   quarantined diagnostic block. The old v3 canaries and historical 128-task
   runs belong to the exposed development protocol; the protected test
   partitions have not been run. The HotpotQA v4 configuration explicitly
   declares `stage=development` and `required_partition=development`, so its
   canary cannot be presented as a final held-out result. The current AIME
   development canary uses AIME 2025 tasks and completed 2/2 Stable Zero; it
   does not estimate AIME 2026 performance. The preserved AIME 2026 run is an
   **official-test diagnostic** and is excluded from adaptation evidence.
2. QA Direct and retrieval-enabled AgentGraph conditions are not
   protocol-equivalent. Report each protocol independently; do not interpret
   their score difference as a paired architecture effect.
3. HealthBench follows a public simple-evals-compatible reference-judge
   protocol, not the private official HealthBench evaluation service. The
   physician reference response and rubric items are supplied only to the
   evaluator judge; they are excluded from model-visible Runtime state and
   from MedRAG Tool inputs. The v2 two-task mean `raw_score` of 0.2 for both
   Direct and AgentGraph is a diagnostic only; neither trajectory used MedRAG.
4. WebShop and ALFWorld receive success only from the RAGEN environment.
   Runtime traces are replayed by the evaluator; model-visible observations do
   not contain reward or `won`. WebShop v4 completed 2/2 native-validation
   Stable Zero with Direct and AgentGraph each at 50% success and with
   heterogeneous Executors. WebShop v2 used native test goals and remains
   excluded; v3 is retained only as a pre-`max_action_tokens` failure
   diagnostic. ALFWorld v2 uses project-held-out native-training tasks, not an
   official validation/test split.
5. SWE-bench receives `resolved` only from the official harness. A generated
   diff, model judgement, or local proxy test is not a resolved instance. The
   v2 dataset boundary now isolates regular dev (validation), regular train
   (training) and the complete Verified split (final evaluation). Docker
   permission/harness preflight still fails, so there is no live Coding
   trajectory or valid `resolved` receipt and
   `ALL_DATASETS_STABLE_ZERO_COMPLETE = NO`.
6. Per-action exact JSON Schema support for fixed Tools is implemented and
   interface-tested. HotpotQA and TriviaQA v3 each produced one successful
   natural retrieval `ToolReceipt`; AIME-2025 development and HealthBench v2
   produced none. Diagnostic-only forced probes passed end to end for TriviaQA
   and HealthBench. HotpotQA's forced probe passed schema/backend dispatch but
   failed bounded termination after repeated `read`; AIME executed both
   computation backends but later emitted invalid JSON and exhausted its turn
   budget. These forced results are not benchmark or Skill evidence. WebShop
   and ALFWorld separately executed state-dependent native environment actions.
7. `ACTIVE Skill = 0` for this multidataset phase. The evidence gate did run:
   HotpotQA and TriviaQA candidates were rejected by the calibrated
   lower-bound/harm criteria. Consequently Skill injection, micro-training,
   optimizer update and policy synchronization were not executed in the new
   unified Runtime conditions and must not be reported as current-condition
   training. Separately, the earlier joint-QA core has two receipt-valid
   one-update LoRA runs with successful policy synchronization and post-update
   canaries. Their Step 0→2 fixed held-out macro EM/F1 changed from
   56.25%/68.58% to 54.69%/65.31%, so they establish an executable learning
   path but not a positive performance trend. Across the current seven-dataset
   Stable Zero phase, `ACTIVE Skill = 0` and `optimizer_updates = 0`.
8. Tool-aware Skill applicability is now wired at the architecture boundary:
   only exact Tool IDs that are available and compatible with the current
   dataset satisfy a Skill's sorted, unique `required_tools` condition. This is
   the SkillFlow retrieval predicate, not a new Skill policy. Because there are
   no ACTIVE Skills and Executor-side Skill invocation remains rejected, it
   establishes retrieval readiness only; it provides no Skill-use or accuracy
   evidence.
