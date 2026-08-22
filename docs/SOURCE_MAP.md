# Architecture source map

This map records where the architecture comes from.  The paper PDFs and the
project design note are design inputs, not executable instructions.

## FlowSteer code retained as the Canvas foundation

| Project boundary | Upstream FlowSteer reference | Local status |
| --- | --- | --- |
| Atomic action protocol | `src/interactive/action_parser.py` | Retained for the legacy Operator path; `agent_action_parser.py` is the free-AgentGraph adaptation required by the design note. |
| Mutable workflow state | `src/interactive/workflow_graph.py` | Retained for the legacy path; `agent_graph.py` extends the state to arbitrary model-labelled Agent nodes and two-bit relations. |
| Multi-turn Canvas | `src/interactive/workflow_env.py::InteractiveWorkflowEnv.step` | Retained; `agent_workflow_env.py` keeps the same reset/step/feedback boundary and canonical per-turn history for AgentGraph actions. Snapshot/restore/fork preserve that visible history, while runtime objects remain process-local. |
| Trajectory and action-token records | `src/interactive/workflow_builder.py` | Retained and supplemented by `records.py` for the new path. |
| Executor boundary | `src/aflow_executor.py::AFlowExecutor.execute_workflow` and `scripts/async_llm.py` | Preserved as the legacy executor; the AgentGraph path uses the same OpenAI-compatible service boundary through `openai_gateway.py`. |
| One-action Director loop | `train_interactive.py` and the FlowSteer paper's progressive Canvas loop | Preserved in `director.py`; the initial prompt is deliberately shorter and has no workflow templates. Maximum-round termination is returned explicitly and is never presented as `finish`. |
| Evidence-driven terminal choice | `src/interactive/prompt_templates.py` (finish when the current result satisfies the task) and `src/interactive/workflow_env.py` (execution result returned as next-step feedback) | `director.py` keeps explicit `finish` and `max_rounds` distinct, but now says continuation must identify a missing evidence hop, conflict, format/runtime error, or task mismatch. Step0-v1 removes the sampled per-turn preferred-model hint entirely; no answer-presence auto-finish or fixed workflow is added. |
| Progressive execution cache | `src/interactive/workflow_env.py` (`execute_each_step`, `last_execution_result`) | Adapted as a revision-local result in `agent_workflow_env.py`. `finish` may reuse it, but a non-FINISH no-op is rejected; `rollout_collector.py` marks valid reuse and does not serialize old Agent calls as new executions. |
| AgentGraph search-space bounds | Project design note sections 3 and 4 plus `config/*agentgraph*.yaml` | The declared `max_agents` is consumed by the Canvas; the two-Agent reciprocal-block limit, unique output/reachability flags, seeded Executor selection, six actions, and progressive execution mode are validated against the fixed runtime semantics rather than left as descriptive YAML. |

The Qwen3-8B defaults, vLLM Director launcher, predefined Operator search
space, structural reward, and legacy training loop are not reused by the new
Qwen3.5 path.

## SkillFlow code used for Qwen3.5 runtime structure

| Project boundary | SkillFlow reference | Local status |
| --- | --- | --- |
| Qwen3.5-9B Supervisor default | `configs/skillflow.yaml` | Mirrored in `config/training_agent_graph.yaml`. |
| SGLang Supervisor child | `training/sglang_manager.py::SGLangSupervisorManager` | Adapted in `src/interactive/sglang_manager.py`; import remains side-effect free. |
| Standalone SGLang launch arguments | `scripts/restart_sglang.sh` | Adapted in `scripts/start_qwen35_director_server.sh`; only one rollout service is declared. |
| Forward/backward LoRA profile | `training/gflownet_trainer.py::setup` | The original architecture-only configuration mirrored theta rank 64 and phi rank 16.  Architecture-v6 later activates only the single Director `theta` adapter through the one-pass GRPO adaptation documented below; SkillFlow's backward policy and Z head are not claimed. |
| Three-role GPU topology | `device`, `supervisor_gpu_id`, and `extra_device` in SkillFlow | The declared topology maps learner, rollout Supervisor, and gradient replica separately.  Formal Step 1 used physical GPUs 3/4/5.  When an unrelated service later occupied physical GPU 5, Step 2/3 moved only the replica to the free physical GPU 7 and recorded that runtime adaptation; the external service was not touched. |
| Split micro-batch backward | `GFlowNetTrainer._batched_logprob_backward` | The initial architecture config was inactive.  Architecture-v6 later uses the existing local one-pass action-masked learner with a fixed two-way group partition and OOM backoff.  This is not SkillFlow's TTB backward algorithm, and the one-group micro runs placed no effective work on the second partition. |
| Skill injection after bootstrap | `GenericTaskEnvironment` and `SkillWorkspace` | The Director prompt omits the Skill field when the validated Skill list is empty. |
| Bounded visible interaction history | `training/react_prompts.py` (`action_history`, `history_length`) and `training/environment.py::_build_react_prompt` | The Director receives only the configured recent Canvas-history window. Entries contain canonical action, acceptance/terminal state, graph revision, compact feedback, and whether execution was reused; no role template or unvalidated Skill is injected. |
| Dataset preparation fields | `data/prepare_v3.py` | Adapted in `scripts/prepare_agentgraph_datasets.py`: retains `question`, `answer`, `task_type`, `context`, `extra`, and environment fields while adding the design-note `TaskRecord` keys. |
| WebShop/ALFWorld task handles | `src/ragen_adapter.py` | The aligned records preserve `env_type` and `env_config`; runtime installation is reported separately from static dataset readiness. |
| SWE-bench evaluator handle | `training/swebench_client.py` | The aligned records retain the Verified instance ID and harness payload; no repository checkout or tests are run during preparation. |
| JSONL loading boundary | FlowSteer `train_interactive.py::load_dataset` and `eval_only.py::load_dataset` | `src/interactive/task_dataset.py` retains streaming JSONL while enforcing the design-note schema and split isolation; `scripts/run_agentgraph.py --dry-load` exercises it without a model call. |
| Exact generation seed | `contracts/scientific_sampling.py::ScientificSamplingCoordinate`, `rollout/types.py::derive_generation_seed`, and `rollout/engine.py` | `scientific_sampling.py` directly ports the dependency-light SkillFlow coordinate and seed derivation. The native Director sends the derived value as SGLang `sampling_seed`; the trajectory saves one coordinate receipt and every turn saves and verifies the exact generation seed. |
| Existing adapter inference readiness | `training/external_sglang.py::publish_external_adapter` and `runtime/sglang_gateway.py` | `policy_sync.py::ensure_loaded_adapter` reuses only model-list, load, verification, and canary for evaluation. It neither trains nor publishes a new policy. |
| Multi-hop one-call contract | `training/task_prompts.py::MULTI_HOP_QA` | The paired local Direct path uses the upstream brief multi-hop contract through the existing Agent gateway; it bypasses Director, Canvas, and AgentGraph. |
| Intermediate observation vs terminal answer | `training/environment.py::step`, `training/task_prompts.py::MULTI_HOP_QA`, and `training/batch_inference.py` | `AgentRuntime` derives one Output identity from the already-validated `graph.output_agent_id`. `openai_gateway.py` gives non-Output nodes an intermediate-artifact boundary and gives only the Output node the concise `<answer>` terminal contract. SkillFlow's fixed tools and mandatory tool-use policy are not copied. |

SkillFlow's TTB objective, backward policy training, partition head, benchmark
environment, and local multi-executor launcher are not copied.  The historical
architecture phase kept GRPO disabled; the later architecture-v6 Hotpot micro
phase explicitly activated the project's terminal-only, action-masked one-pass
GRPO adaptation for three bounded updates.

## Project-specific algorithm modules

The following are additions required by the attached design note rather than
claims of upstream FlowSteer or SkillFlow functionality:

- free-text Agent contracts, per-node model selection, and finite two-stage
  bidirectional Agent execution;
- the MACE feature/bandit baseline and joint Bayesian posterior primitives;
- same-prefix paired-probe and EVSI primitives;
- version-bound Skill evidence schemas, lifecycle, and persistence records.

Those modules are isolated from the runtime reward path.  MACE/Bayesian
exploration and production Skills stayed disabled throughout the architecture-v6
run.  Only the explicitly versioned Hotpot micro configs enable the local
terminal-only GRPO learner; the architecture/evaluation configs remain
inference-only.

## Dataset-specific adaptations required by the requested benchmark list

The public SkillFlow data release cannot be copied verbatim: it contains
MedQA, not HealthBench Professional, and its checked-in `prepare_v3.py` still
generates synthetic ALFWorld prompts.  The local adapter therefore makes only
the compatibility changes required for this project:

- real playable ALFWorld task descriptions and native train/seen/unseen splits;
- the official WebShop baseline split ranges and full-product environment
  paths;
- the user-specified deterministic 128-held-out/512-train view for every
  benchmark, with training-only cycling when a source pool is short;
- AIME's explicit 30 official-2026 + 98 historical held-out composition,
  followed by 402 unique historical training candidates and 110 training-only
  cycle records;
- SWE-bench Verified in source order (128 held out, 372 unique training
  candidates, then 140 training-only cycle records); and
- the design-note `task_id/question/ground_truth/split/metadata` contract on
  every record, alongside upstream FlowSteer and SkillFlow aliases.

Evaluator-only material (rubrics, gold patches, accepted aliases, supporting
facts, and environment targets) is stored under `metadata.evaluator_payload`.
The AgentGraph input loader constructs the Director task only from `question`.

This 128/512 view is a project training/validation recipe, not an untouched
official benchmark score split.  The manifest retains native split and base
task IDs so a future official-evaluation view can remain separate.

For HotpotQA, the previous preparation path truncated every passage to 300
characters even though the canonical record retained full context.  SkillFlow's
retrieval/public-context boundary keeps the evidence intact.  The converter and
runtime loader now render all ten supplied passages; the loader reconstructs
older aligned JSONL from its answer-free top-level `context` without reading
supporting facts or ground truth.

## HotpotQA Round-01 evaluation-only driver

`scripts/evaluate_hotpotqa_round.py` is a necessary project adapter rather than
a new workflow architecture.  It reuses `LiveSmokeBackend.collect`, the exact
receipt collector, `AgentRuntime`, `OpenAICompatibleGateway`, `EvidenceStore`,
and `evaluate_task`.  Its own responsibilities are limited to freezing the 128
HotpotQA validation tasks, running a paired one-call Direct condition, atomic
checkpoint/resume, strict-denominator aggregation, Wrong Demo materialization,
and reporting.  It never calls trainer, optimizer, backward, policy publish,
MACE, Bayesian, or Skill code.

## HotpotQA Training-ready Step 0 adaptations

The Step-0 protocol repair is driven by the saved Round-01 behavior: all 15
multi-Agent upstream nodes emitted a task-level answer, 12/15 downstream
outputs copied that answer verbatim, and four workflows edited again after a
valid Output candidate.  The local changes therefore remain protocol and
observability adaptations rather than method changes:

- `agent_runtime.py` derives `is_output_agent` from the existing unique Output
  node; it does not extend the node/action/search-space schema.
- `openai_gateway.py` separates intermediate artifacts from the unique
  terminal answer and applies diagnostic masking only while rendering a model
  prompt.  Canonical upstream and peer messages remain intact in receipts.
- `agent_workflow_env.py` returns compact answer-format facts and a bounded
  Output inbox preview as FlowSteer-style progressive execution feedback.  It
  never sees ground truth or evaluator correctness.
- `director.py` adds a general issue-driven finish/continue rule and removes
  the fresh weighted model suggestion after a successful execution candidate.
  Agent count, free-text contract, model, relation, Output, and finish remain
  Director choices.
- `diagnose_hotpotqa_communication.py` replays frozen final multi-Agent graphs
  under `normal` and `upstream_masked`.  It bypasses Director, Direct baseline,
  training, MACE/Bayesian, and Skills; its typed records are always
  `diagnostic_only=true` and `grpo_eligible=false`.
- `freeze_hotpotqa_step0_untouched.py` reuses the existing Hotpot converter and
  retagging functions to freeze raw candidates 640--671, after the existing
  128 held-out and 512 training candidates.  This confirmation slice is not a
  new split algorithm and must not be used for prompt tuning.

`evaluation_hotpotqa_training_ready_step0.yaml` binds the development baseline
to `training_ready_step0 / step_000000`, the unchanged existing adapter, prompt
and tool versions, evaluator, catalog, split, seed, and source revision captured
by the run manifest.  GRPO, backward, optimizer updates, policy publication,
exploration, and Skills remain disabled.

## HotpotQA Multi-Agent Step0-v1 adaptations

This version is driven by the saved development evidence (93 singleton graphs,
35 two-node chains, no 3+ node graph, and no positive communication-ablation
effect).  It does not add a topology template, role enumeration, minimum Agent
count, or structural reward.

| Current module | Reference source | Reused boundary | Incompatibility and minimal adaptation |
| --- | --- | --- | --- |
| `director.py` contract and relation guidance | FlowSteer `workflow_graph.py::WorkflowNode.custom_prompt`, `workflow_env.py::_handle_prompt_input`; SkillFlow `src/executor/m_exec.py::MExec.execute` | Natural-language node instruction and one-action Canvas loop | Neither upstream has a free heterogeneous AgentGraph contract. The existing free string is retained; the neutral prompt only asks it to state objective, input/dependency, artifact, and completion. No typed role or workflow recipe is introduced. |
| `director.py` model catalog view | Existing `ModelSpec.metadata` and SkillFlow provider/model boundary | Exact configured model IDs and bounded metadata | The old prompt exposed a sampled preferred model that matched 92.1% of saved `add_agent` choices. Step0-v1 removes that hint and exposes only canary-backed family/profile facts; model selection remains a Director action. |
| `agent_workflow_env.py` terminal gate | FlowSteer `workflow_env.py::_check_finish_constraints` and `_step_internal`; SkillFlow `training/environment.py::GenericTaskEnvironment.step` terminal-action rejection | Environment-side rejection of an invalid terminal action | Upstream constraints are Operator/tool specific. Step0-v1 adds only a configurable QA protocol: FINISH is rejected unless the latest Output is one non-empty, exact `<answer>...</answer>` wrapper. Other datasets keep protocol `none`. |
| `agent_runtime.py`, `openai_gateway.py`, `rollout_collector.py` communication envelope | SkillFlow trajectory action/instruction/observation boundary and the existing project upstream receipt | Free artifact body, source/target routing, graph revision, rendered-message receipt | Neither upstream implements peer-Agent envelopes. The project adaptation adds only facts already known by the runtime: source, target, generic artifact/candidate type, target contract as request/dependency, and graph revision. It does not invent confidence or evidence references and does not force JSON reasoning. |
| `AgentGraph.topology_statistics` | FlowSteer `WorkflowGraph.get_statistics` | Read-only shape telemetry in Canvas state | FlowSteer's fixed Operator AST cannot represent model-labelled pairwise AgentGraph edges. A thin calculation reports observed nodes, edges, quotient depth/width, fan-in/out and reciprocal pairs; nothing consumes it as reward or validity. |
| `scripts/discover_models.py` and `model_catalog_hotpotqa_multiagent_v1.yaml` | Existing OpenAI-compatible gateway and model registry | Provider `/v1/models`, exact model names, normal Output protocol request | Discovery now persists a non-secret list receipt and one canary per proposed exact ID. Only passed IDs enter the Hotpot catalog; Gemini was absent and Grok canaries returned HTTP 429, so neither is added. |
| `evaluate_hotpotqa_round.py` task-ID selection | Existing Round-01 fixed sequential selector | Same loader, evaluator, strict denominator, resume and receipts | Architecture diagnostics require an explicit development-only slice. The adapter accepts a unique ordered task-ID list and otherwise preserves the old sequential behavior. |

The existing `theta_smoke_step_000001` adapter had one real optimizer update
whose only non-zero advantage came from TriviaQA.  It is therefore labelled a
warm-start diagnostic policy here, not a formal untrained HotpotQA Step0.
Formal `policy_step_000000` must be materialized separately from base
Qwen3.5-9B plus a deterministically initialized LoRA before Step0-to-StepN
training begins.

## HotpotQA Multi-Agent architecture-v2 diagnostic adaptations

Architecture-v2 is a versioned development hypothesis derived from the saved
v1 trajectories.  It is not an upstream Skill, a topology template, or a
training result.

| Current module | Reference source | Reused boundary | Incompatibility and minimal adaptation |
| --- | --- | --- | --- |
| `director.py` catalog presentation | Existing `ModelRegistry` plus SkillFlow's explicit provider/model boundary | The same frozen exact model IDs and metadata are shown on every turn | v1 placed one sorted family first and selected that family for 14/15 nodes.  v2 changes only presentation order with a deterministic seed; it does not select a model or alter priors.  The diagnostic exposed that this order seed must be separated from rollout sampling before grouped training so same-task/same-condition rollouts see the same prompt. |
| `director.py` dependency-coverage sentence | FlowSteer progressive Canvas state/feedback and the user design note's free contract fields | One atomic edit per turn, free-text contracts, and unrestricted legal AgentGraph topology remain unchanged | v1 produced 13 singleton graphs on the fixed 14-task diagnostic.  v2 asks the Director to check visible evidence dependencies and name consumed upstream artifacts, without naming roles, requiring an Agent count, or rewarding complexity.  This is a project hypothesis supported by diagnostic evidence, not direct upstream code or an ACTIVE Skill. |
| `openai_gateway.py` Output span clarification | SkillFlow's terminal answer versus intermediate observation boundary and the existing project Output-only protocol | Only the graph Output Agent may emit the task answer | Some v1 outputs placed structured reports inside the answer wrapper.  v2 adds only that the span itself is not JSON, a key-value report, or an explanation. |
| `evaluation_hotpotqa_multiagent_v2_diagnostic.yaml` | Existing Hotpot round driver and v1 fixed task-ID diagnostic | Same 14 development task IDs, evaluator, warm-start policy, frozen catalog, full-context input, and strict denominator | The Director action budget is raised from 256 to 512 because v1 saved truncated/malformed actions.  Direct predictions are copied from the already completed Round-01 file; no new Direct request is made. |

The v2 fixed-task run is evaluation-only.  Its saved behavior is tied to the
warm-start `theta_smoke_step_000001` adapter and cannot be labelled formal
HotpotQA `policy_step_000000`.

## HotpotQA Multi-Agent architecture-v3 compatibility fixes

V3 changes no action, graph, role, routing, reward, or Skill method.  It fixes
only execution-condition and receipt defects proven by v2:

| Current module | Reference source | Reused boundary | Minimal compatibility fix |
| --- | --- | --- | --- |
| `director.py`, `train_agentgraph_smoke.py` | FlowSteer same-task rollout grouping and exact prompt receipts | Rollout sampling seed still varies exactly as before | Catalog presentation now receives a separate task/condition-stable seed.  Same-task/same-condition rollouts therefore see one catalog order while retaining different sampling seeds. |
| `agent_workflow_env.py` | FlowSteer environment-side FINISH constraints | FINISH remains an environment hard gate, not a reward | Feedback and acceptance share one parser requiring exactly one opening tag, one closing tag, a full wrapper, and non-empty content.  Multiple, nested, and empty wrappers are rejected. |
| `evaluate_hotpotqa_round.py` | Existing resumable Round-01 evaluation driver | Direct records are still copied without another provider call | Copied records receive a reuse receipt, and final manifest materialization preserves the source plus reused/newly-collected counts. |
| `evaluation_hotpotqa_multiagent_v3_dev128.yaml` | Existing fixed 128-task Hotpot development view | Same task order, evaluator, full context, frozen catalog, policy, adapter, and Direct baseline | The config removes unused `max_context_tokens` and `live_adapter_name` claims; actual serving context and loaded adapter remain preflight receipts. |

This remains a warm-start architecture-development evaluation.  GRPO,
backward, optimizer work, policy publication, MACE, Bayesian updates, and
Skills are disabled.

## HotpotQA Multi-Agent architecture-v4 no-op feedback repair

V3 saved one 20-turn failure in which the Director repeatedly selected the
already-selected Output Agent.  `AgentGraph` correctly left its revision
unchanged, but the Canvas labelled the edit accepted and replayed the cached
invalid Output.  FlowSteer's existing
`workflow_env.py::_format_pending_modify_prompt_request` explicitly requires a
different prompt after failure to prevent the same execution loop.  V4 applies
that boundary to the free AgentGraph Canvas: any non-FINISH action that changes
no graph revision is rejected with a compact instruction to change the Agent
contract/model or graph before expecting another execution.  Terminal-format
feedback carries the same hint.  No action, role, topology, reward, evaluator,
or model-selection rule is added.

The v4 regression config supplies an explicit catalog-order namespace equal to
the v3 development namespace.  This reuses the existing task-stable shuffle
while preventing an unrelated condition label from changing model
presentation during the targeted v3-to-v4 comparison.  Sampling seeds,
condition ID, prompt/tool version, and trajectory versions remain separately
recorded.

## HotpotQA architecture-v5 scientific sampling compatibility

The v4 regression exposed a measurement defect rather than a new workflow
hypothesis: the evaluation runner passed the task's selected-list position as
`rollout_index`, and the Director used `experiment.seed + rollout_index +
round_index`.  The same task therefore received a different policy sample when
the 128-task list was reduced to a 12-task subset.  Catalog order had already
been separated, so this defect affected generation only.

| Current module | SkillFlow source | Reused boundary | Minimal adaptation |
| --- | --- | --- | --- |
| `scientific_sampling.py` | `src/skillev/contracts/canonical.py::normalize_json`, `src/skillev/contracts/scientific_sampling.py::{ScientificSamplingCoordinate,scientific_sampling_schedule_hash}`, and `src/skillev/rollout/types.py::{GenerationPhase,derive_generation_seed}` | Canonical sampling identity, coordinate fields, artifact-namespace exclusion, phase/turn seed derivation | Direct dependency-light port of only the sampling-required functions so this repository does not require a second checkout at import time. |
| `director.py` | `src/skillev/rollout/engine.py` per-step calls to `derive_generation_seed` | One fixed coordinate per rollout and one derived action seed per turn | `AgentGraphOrchestrator.generation_seed` is the single caller used by both inference and exact-receipt collection.  Legacy construction remains available only for non-training compatibility tests. |
| `train_agentgraph_smoke.py` | `src/skillev/training/runtime_components.py::RolloutBatchCollector.collect` | Base schedule, purpose, task identity, rollout position, and optimizer-step/anchor coordinate | The local free-AgentGraph collector has one Director action generation rather than SkillFlow's separate reasoning/action calls, so it fixes phase to `action`.  `sequence_position` is the ordinal within the current task; it never uses selected-list order. |
| `records.py`, `rollout_collector.py` | SkillFlow `RolloutRequest.sampling_coordinate` and exact generation request | Coordinate is stored once; per-turn seed is stored beside the exact token/log-prob receipt | New trajectories use schema `flowsteer.agentgraph.v2`.  GRPO eligibility now requires that the schedule, task, and every turn seed can be reconstructed from the saved receipt. |
| `evaluate_hotpotqa_round.py`, `evaluate_agentgraph_architecture.py` | SkillFlow's result-affecting-coordinate rule | Artifact/run/list position must not select model randomness | Single-rollout evaluation passes task-local ordinal `0` for every task.  Full-list, reordered-list, and single-task subset execution therefore share the same task coordinate. |

This compatibility change does not alter the Director prompt, Canvas action
space, model catalog, Agent runtime, evaluator, reward, MACE/Bayesian state, or
Skill visibility.  It is required before a controlled architecture A/B or
formal Step0-to-StepN comparison can be interpreted.

## HotpotQA formal Step-0 static training preconditions

These modules close the four static compatibility gaps listed by the v5
report.  They do not execute a rollout, optimizer, policy publication, or
Skill evolution loop.  The upstream implementation remains the source of the
transactional boundaries; the local code is limited to this repository's
single-`theta` AgentGraph policy and already-aligned HotpotQA records.

| Current module | Classification | Reference source | Reused boundary | Incompatibility and minimal adaptation |
| --- | --- | --- | --- | --- |
| `hotpot_step0.py`, `materialize_hotpotqa_step0.py` | Necessary adaptation | SkillFlow `scripts/build_gate_4c_initial_policy.py::main`; `src/skillev/policy/hf_backbone.py::{bind_initial_trainable_state,save_checkpoint}`; existing `smoke_trainer.py::Qwen35OnePassSmokeTrainer._load_models` | Fixed initialization seed, deterministic algorithms, bind-before-save, fresh output, and immutable initial-policy receipt | SkillFlow persists forward/backward adapters and a Z head.  This Director has one SGLang-facing PEFT adapter named `theta`, so the adapter reuses the existing Qwen3.5 multimodal/PEFT loader and saves only untouched `theta` tensors.  Its default command is a no-model, no-write preflight; materialization is explicit and was not run in this phase. |
| `hotpot_training_schedule.py`, `freeze_hotpot_training_schedule.py` | Necessary adaptation | SkillFlow `packages/private-evaluation/.../curriculum.py::{PrivateFrozenTaskSequence,frozen_sequence_from_task_ids}`; `src/skillev/benchmarks/static.py::OrderedBenchmarkTaskProvider`; `src/skillev/runtime/execution_state.py::OrderedTaskCursorState`; `src/skillev/runtime/attempt_run_plan.py::AttemptRunProgress` | Immutable task order, exact cursor, write-once artifacts, and one-step-at-a-time progress | SkillFlow's private provider is not this project's aligned JSONL loader and does not carry the local grouped-rollout ordinal.  The adapter binds existing HotpotQA train positions and task-local rollout ordinals, rejects validation/test membership, and never re-splits data.  It does not collect or train. |
| `smoke_trainer.py`, `train_agentgraph_smoke.py` exact-continuation flag | Necessary adaptation | SkillFlow `src/skillev/training/checkpoint.py::FilesystemTrainingCheckpointStore::{save,restore}` and `src/skillev/training/runtime_components.py::TTBOptimizerKernel.restore_policy_optimizer_exact` | Save policy plus optimizer continuation state and require the immediately preceding runtime identity before the next update | The local one-pass learner already has a PEFT checkpoint layout rather than SkillFlow's forward/backward/Z checkpoint.  Formal mode now saves AdamW state from Step 1, requires it for Step 2+, and binds adapter/optimizer metadata to the immediately preceding policy.  Legacy smoke behavior remains available only when the formal flag is false.  No optimizer step was run for this change. |
| `policy_sync.py`, `train_agentgraph_smoke.py::LiveSmokeBackend.publish` | Necessary adaptation | SkillFlow `src/skillev/runtime/sglang_gateway.py::{_swap_supervisor_adapter_sync,_validate_adapter}` and `training/external_sglang.py::publish_external_adapter` | Pause admission, drain in-flight calls, load, model-list verification, canary, generation-route switch, old-adapter unload, rollback, and resume | SkillFlow owns the active generation route inside its gateway; this repository stores it in `SGLangReceiptDirectorClient`.  Two small callbacks move route switch/rollback into the same publisher transaction.  `activate_existing_policy` applies the same boundary to an already-materialized untrained Step 0 while recording `training_performed=false` and `policy_published=false`. |

There is no project-algorithm addition in these four modules.  AgentGraph,
MACE, Bayesian exploration, and Skill lifecycle remain the project/design-note
layer described above.  This paragraph records the static precondition phase;
the later explicitly authorized architecture-v6 run materialized Step 0 and
executed the bounded Step 1--3 rollout/update/publish chain documented below.

## HotpotQA architecture-v6 deep/multi-model/Skill phase

This later phase is explicitly authorized by the user's deep HotpotQA task.
It retains FlowSteer's atomic Canvas and SkillFlow's Qwen3.5/SGLang/checkpoint
transactions. It adds no graph-shape reward, macro workflow action, fixed
role set, topology quota, or model-family routing rule.

| Current module | Classification | Reference source | Reused boundary | Incompatibility and minimal adaptation |
| --- | --- | --- | --- | --- |
| `model_catalog_hotpotqa_deep_v6.yaml`, `test_model_catalog_v6.py` | Necessary configuration adaptation | Existing `ModelRegistry`, `AgentGraphOrchestrator.build_prompt`, `scripts/discover_models.py`, and `OpenAICompatibleGateway` | Exact provider IDs, the same Output contract, bounded routing metadata, equal priors, and one canary receipt per newly proposed model | The provider list changes over time and neither upstream ships this account's catalog. Ten receipt-backed arms are versioned with neutral provider/capability facts; no router, role recipe, model reward, or Director replacement is added. |
| `AgentGraph.topology_statistics`, `construction_progress`, `effective_dependency_statistics`, `graph_diagnostics.py` | FlowSteer-derived diagnostic adaptation | FlowSteer `WorkflowGraph.get_statistics`, progressive `WorkflowEnv.step` feedback, and the existing reciprocal-component validator | Read-only structural statistics and one-edit-at-a-time Canvas state | FlowSteer's fixed Operator graph cannot express arbitrary model-labelled relations. The adapter contracts finite reciprocal pairs, computes a lower-bound remaining atomic-edit cost, and grades dependency evidence as unverified/weak/verified. The later observation-boundary correction keeps that cost offline because exposing the shortest terminal path was not behaviorally neutral. Runtime delivery is at most weak; verified requires an explicit independent paired intervention. |
| `director.py` architecture-v6 text/state | Necessary minimal adaptation | FlowSteer's one-action Director loop and current AgentGraph legal actions | Short neutral system prompt, current state, feedback, and model boundary | Historical v3/v4 evidence showed policy stopping after the first complete singleton, not a max-round limit. One sentence names optional abstract relation shapes and one state object reports construction progress; there is no Hotpot workflow, role enum, minimum Agent count, or complexity preference. |
| `skills/pipeline.py`, `skills/validator.py` | Project algorithm integration required by the user MD | SkillFlow `evidence/schema.py`, `evidence/store.py`, `runtime/skill_library.py`, and `evolution/retriever.py`; existing `TrajectoryRecord`, `ProbeRecord`, `EvidenceStore`, lifecycle/store/retriever | Split evidence, immutable ACTIVE library, task-conditioned retrieval, and append-only evidence receipts | SkillFlow does not implement this project's paired AgentGraph effect gate. The thin orchestration layer enforces discovery/validation problem isolation, forced-probe GRPO exclusion, runtime/executor/version binding, delayed activation, and deterministic rejectable prompt-prior rendering. It does not mutate Canvas. |
| `train_agentgraph_smoke.py` Hotpot micro scope | Necessary runner adaptation | Existing one-pass FlowSteer rollout/trajectory path; SkillFlow frozen schedule/cursor, exact optimizer continuation, publish, and canary sequence | Same collector, terminal evaluator reward, action-masked one-pass learner, one optimizer update, atomic adapter sync, and post-update canary | The legacy runner selected seven datasets by source. Formal Hotpot mode resolves exactly one predeclared schedule step and its frozen rollout ordinals, then commits a new write-once cursor only after update, sync, and canary. The legacy 7x2 path remains unchanged. |
| `evaluate_hotpotqa_round.py` v6 receipts | Necessary evaluation/reporting adaptation | Existing fixed held-out runner and `graph_diagnostics.py` | Strict EM/F1, saved full trajectory, resumability, and fixed Direct comparator | The Director condition may use a new sampling seed while the Local Direct comparator must retain its original seed. `direct_generation_seed` freezes that comparator independently, and the manifest takes its declared server context from the Director config. Graph diagnostics are added without changing evaluator reward. |

Formal Step 0 has now been materialized at policy
`qwen35-9b-hotpot-step-000000` / adapter `theta_hotpot_step_000000` with the
fixed seed and zero optimizer updates.  Its adapter presence and canary plus a
two-task Stable Zero chain are recorded; the surviving Step-0 receipt is not a
full pause/drain/route-switch activation transaction and must not be described
as one.  The subsequent Step 1--3 policy chain performed one real optimizer
update per step, saved/restored optimizer state, published each new adapter,
switched the Director route, and passed the post-update canary.

## HotpotQA architecture-v6 exact-resume and micro-training runtime repairs

These changes were made only to complete the already-authorized, frozen
Step 0--3 experiment without recollecting successful paid behavior rollouts.
They do not change the Director prompt, AgentGraph action language, terminal
reward, model catalog, or graph-shape distribution.

| Current module | Classification | Reference source | Reused boundary | Incompatibility and minimal adaptation |
| --- | --- | --- | --- | --- |
| `records.py::{TaskRecord,ExecutionRecord,TurnRecord,EvaluationRecord,TrajectoryRecord}.from_dict` | Necessary persistence adaptation | SkillFlow immutable rollout/request records and exact receipt validation | Reload persisted behavior through the same typed invariants and rederive admission fields | The local dataclasses previously serialized but could not reconstruct a frozen batch.  Deserialization now rejects unknown/tampered derived state rather than trusting JSON. |
| `train_agentgraph_smoke.py --resume-initial-rollouts` | SkillFlow lifecycle reuse plus necessary runner adaptation | SkillFlow frozen schedule/cursor, attempt progress, exact batch identity, checkpoint-before-resume, and write-once continuation boundaries | Require the exact task/version/condition/group/rollout/coordinate/adapter/server/evaluator/evidence batch and prove zero checkpoint, optimizer, publish, canary, and cursor persistence before resume | This repository has a single-theta AgentGraph batch and local artifact layout rather than SkillFlow's TTB attempt runtime.  Resume is therefore restricted to frozen Hotpot micro steps and never recollects any already-successful initial rollout. |
| `smoke_trainer.py` provider-logprob diagnostic | Direct scientific boundary reuse | SkillFlow `rollout/generator.py` excludes serving policy scores; `policy/hf_backbone.py` teacher-forces exact sampled action token IDs; FlowSteer masks policy/action tokens | Original sampled IDs, action mask, route/version/evaluator receipts, and learner teacher-forcing remain the optimization inputs | Provider/SGLang logprob numerical drift is retained as a mean/p95/max diagnostic but no longer rejects an exact group merely for exceeding a tolerance.  Presence, shape, and finiteness are still receipt/admission requirements; non-finite learner scores remain fail-closed. |
| `evaluate_hotpotqa_round.py` fixed Direct reuse ordering | Necessary evaluation repair | Existing fixed comparator and resume boundary in the same runner | Reuse the declared Step-0 Direct records without another model call | A stale destination canary was previously considered before the authoritative fixed source.  The declared source is now ordered first, while successful records are still never overwritten during normal resume. |
| `diagnostic_hotpotqa_deep_v6_step3_communication.yaml` and saved diagnostic receipts | Existing diagnostic path, no new method | Existing `diagnose_hotpotqa_communication.py` normal/upstream-masked replay | Frozen final graphs, same tasks/models, no Director/direct/training, `diagnostic_only=true`, `grpo_eligible=false` | Only five Step-3 multi-Agent graphs had executable Output paths; all ten arms completed.  No score or answer change was observed, so transport is proven but causal communication utility is not. |

### Remaining upstream-compatibility gaps after Step 3

- SkillFlow pins `flash-linear-attention==0.5.2` and verifies the Qwen3.5
  gated-delta kernel.  The local loader does not call that enforcement helper,
  so FLA compatibility is **not proven by project code** even though the
  training environment supplied the dependency.
- One Step-2 trajectory began with a legal malformed sampled action
  (`executed_prefix_tokens=0`) and later finished with reward 1.  It was kept in
  the frozen artifact but the current all-turn admission rule excluded the
  whole trajectory.  This differs from SkillFlow's "invalid actions remain
  data" boundary and FlowSteer's model-response masking; it remains a known
  adaptation gap rather than being silently relabelled as direct reuse.
- Provider logprob *values* do not enter the loss or threshold rejection, but
  their receipt presence and finiteness still participate in local admission.
- The Skill orchestration code and tests are wired, but the formal Step 0--3
  runs produced no paired probes, validated candidates, ACTIVE Skills, or
  retrieval gain.  Production Skill end-to-end readiness is therefore not
  claimed.

## Post-Step-3 Director observation-boundary correction

The fixed Step-3 receipts exposed an observation-protocol bias rather than an
AgentGraph expressivity failure.  Of 127 format-valid progressive executions,
126 were followed immediately by `finish`; 119/128 trajectories stopped after
their first execution.  Before this correction the next Director prompt
simultaneously labelled that provisional value `final_answer`, repeated it in
both current feedback and the history tail, reported a one-action shortest path
to termination, and told the policy to prefer `finish`.

| Current module | Classification | Reference source | Reused boundary | Minimal correction |
| --- | --- | --- | --- | --- |
| `director.py::AgentGraphOrchestrator.build_prompt` | Necessary SkillFlow compatibility correction | SkillFlow `training/react_prompts.py` and `training/environment.py::_build_react_prompt` | Bounded prior observation/action history plus one distinct current observation | Reconstruct observation-before-action history, keep the latest Canvas feedback only in the current-observation field, and remove Director-visible shortest-to-finish accounting. `AgentGraph.construction_progress` remains available for offline diagnostics. |
| `agent_workflow_env.py::_accepted_feedback` | Necessary FlowSteer terminal-semantics correction | FlowSteer `workflow_env.py::_format_feedback` exposes an execution result; the user MD §4.1/§4.2 reserves the terminal answer/reward for explicit `finish` | Progressive execution remains immediate, compact, and revision-local | Rename the Director-visible provisional `final_answer` to neutral `output`, and `answer_protocol` to `output_format`; internal runtime and terminal trajectory fields are unchanged. |
| `director.py::DIRECTOR_SYSTEM_PROMPT` | Necessary neutral search-space correction | FlowSteer one-edit Canvas; user MD §3.4 and §4.2 forbid structural reward or fixed topology | Same six atomic actions, free contracts, model catalog, and task-only terminal evaluator | Remove asymmetric `prefer finish` / `not to make the graph larger` wording. Structural completeness and output format are now explicitly necessary but not sufficient for task adequacy; neither singleton nor deeper graphs are preferred by size. |

This correction adds no minimum Agent count, graph-depth reward, topology
quota, role template, forced relation, exploration bonus, or evaluator change.
It has passed no-model interface regression tests only.  A future controlled
rollout must use a new prompt/condition version and the same fixed tasks before
any change in graph-depth behavior can be claimed.

## Joint-QA component-level progressive Canvas phase

The HotpotQA/TriviaQA joint phase changes the Canvas transaction boundary, not
the AgentGraph runtime or task reward.  FlowSteer's legacy implementation
already lets one structure-level `ADD` action create several child Operators;
`InteractiveWorkflowEnv` waits until that structure is fully configured and
then executes the workflow once.  A free AgentGraph has no fixed
parallel/conditional/loop Operator AST, so the smallest compatible adaptation
is an explicit subgraph transaction.

| Current module | Classification | Reference source | Reused boundary | Minimal adaptation |
| --- | --- | --- | --- | --- |
| `agent_action_parser.py::ADD_SUBGRAPH` | Necessary adaptation | FlowSteer `action_parser.py::_parse_add_parallel/_parse_add_conditional/_parse_add_loop` | One Director action can describe one multi-node structure | The action carries one to three free Agents, their two-bit relations, and an optional Output identity because the fixed Operator DSL cannot represent model-labelled Agent nodes. |
| `agent_workflow_env.py::_apply_mutation` | Direct reuse plus necessary adaptation | FlowSteer `workflow_env.py::_handle_add/_handle_prompt_input`; existing `AgentGraph` scalar mutations | Mutate a candidate Canvas, validate the completed structure, execute once, return one feedback observation | Existing `add_agent`, `set_relation`, `set_output`, dirty closure, validation, rollback and Runtime are reused inside one transaction.  A failed internal mutation never commits the candidate graph. |
| `agent_runtime.py` | Direct reuse | Existing quotient-DAG scheduler and finite reciprocal block | Parallel ready blocks, fan-in/fan-out routing, and two-Agent draft/revision exchange | No change.  A subgraph transaction still produces the normal per-Agent call and communication receipts inside one Runtime invocation. |
| `director.py` | Necessary prompt/action-schema adaptation | FlowSteer progressive Director loop; project design note's neutral search-space requirement | One sampled action, execution feedback, next Canvas observation | The concise prompt lists the legal subgraph fields and execution boundary.  It does not prescribe fan-in, chains, roles, Agent count, model family, or an unvalidated Skill. |
| `config_loader.py` | Compatibility adaptation | Existing versioned AgentGraph configuration | Fail-closed action search space | Both the legacy six-scalar-action profile and the new `add_subgraph` profile remain readable; new trajectories use a new prompt/tool/condition version and cannot be grouped with legacy rollouts. |

The joint Executor catalog directly reuses
`model_catalog_hotpotqa_deep_v6.yaml`: one local Qwen3.5-9B arm and nine
previously canary-backed remote exact model IDs, all with equal numeric priors.
A fresh `/v1/models` list check is a run precondition; it does not replace the
local Qwen3.5-9B Flow-Director or add a role-to-model routing template.

The experiment-specific partition adapter
`scripts/prepare_joint_qa_partitions.py` directly reuses the existing HotpotQA
and TriviaQA converters and record retagging.  It freezes each canonical
candidate stream as development `[0:128]`, train `[128:640]`, quarantine
`[640:672]`, independent Skill confirmation `[672:736]`, and final test
`[736:864]`.  The quarantine accounts for prior diagnostic exposure.  Full
ordered task IDs are persisted in the generated manifest; confirmation and
test are admitted to neither optimizer schedules nor Skill discovery.

The earlier HotpotQA and TriviaQA v3 Tool/ReAct canaries reused two records
from the repeatedly exercised development block.  Their saved 2/2 outcomes
are Stable Zero execution/evaluator checks, not held-out benchmark estimates.
`config/joint_qa_partitions_v2.yaml` and
`scripts/prepare_joint_qa_partitions.py::{_take_candidate_prefix,_partition_record,prepare}`
materialize separate development, train, quarantine, independent Skill
confirmation and test records.  The test partition has not been passed to a
model or evaluator.  This keeps SkillFlow's task/held-out separation first,
FlowSteer's frozen trajectory/evaluator boundary second, and adds only the
dataset-specific partition tags required by the project.

This phase does not treat topology depth, reciprocal communication, model
diversity, or Skill visibility as reward.  GRPO remains terminal token F1 only;
forced paired interventions remain ineligible for GRPO.  MACE-style UCB/EVSI,
the joint Bayesian posterior and evidence-gated Skill lifecycle remain the
project-algorithm layer required by the design note and must be evaluated in a
separate frozen exploration epoch before any Skill becomes ACTIVE.

## Joint-QA frozen Skill epoch and one-update training boundary

This layer connects existing exploration, Skill, and learner components; it
does not add a second rollout engine, optimizer, Skill store, or policy
publisher.  Skill effects are explicitly the intent-to-treat effect of making
a rejectable prompt prior visible during a complete trajectory from an empty
Canvas.  They are not labelled as local topology effects or verified Director
adoption.

| Current module | Classification | Reference source | Reused boundary | Minimal adaptation |
| --- | --- | --- | --- | --- |
| `exploration/skill_experiment.py::rank_probes_by_evsi` | Project-algorithm wiring required by the design note | Existing `exploration/evsi.py::{make_common_random_numbers,particle_evsi_many}` and joint `BayesianLinearPosterior` | MACE-style posterior UCB supplies a top-K candidate set; particle EVSI with common random numbers selects the next paid paired intervention | The existing numerical EVSI primitive previously had no joint-QA caller.  The adapter only maps the fixed candidate/dataset feature differences into that primitive and records the deterministic ranking. |
| `run_joint_qa_mace_skill.py` | Thin experiment adapter | Existing paired-probe runner, `JointQAPosteriorScheduler`, `SkillEvidencePipeline`, and SkillFlow retrieval observation adapter | Balanced cold start, UCB prefilter, EVSI selection, randomized paired-arm order, independent confirmation, deterministic gate, and delayed next-epoch activation | Old Step-2 paths and a fixed fan-in prior were incompatible with the new `add_subgraph` action profile.  The runner uses disjoint train evidence plus a versioned confirmation partition (`skill_confirmation` in rounds 0–2; adaptive development `[32:52]` in round 3; fresh canonical source `[864:904]` in round 4) and bounded, rejectable failure-derived priors without mutating Canvas. |
| `skills/pipeline.py::retrieval_snapshot` | SkillFlow lifecycle adaptation | SkillFlow frozen ACTIVE Skill library per rollout/training epoch | Every Director turn in one batch reads the same immutable Skill records | The local pipeline formerly reread its JSON store on every turn.  An optional typed snapshot freezes visibility while leaving the existing store and retriever unchanged. |
| `skills/pipeline.py::{PromptSkillPrior.to_dict,render_validated_skill}` | SkillFlow model-visible context adaptation | SkillFlow `TaskConditionedSkillRetriever.retrieve` and `render_retrieved_skill_block` expose Skill identity plus authored content, while applicability/evidence remain in the frozen library and receipts | ACTIVE-only retrieval, full SkillStore record, exact version binding, and rejectable prompt-prior semantics | The previous Director observation repeated full condition/action/evidence metadata and the instruction.  The model-visible projection now contains Skill ID/version, one applicability line, and one instruction; complete condition/action/evidence remain persisted and unchanged. |
| `train_agentgraph_smoke.py` Skill-on joint micro boundary | Necessary SkillFlow training adaptation | Existing SkillFlow-style exact group admission, one-pass LoRA learner, frozen schedule/cursor, adapter publication and updated-policy canary | Terminal F1-only GRPO, one real `optimizer.step`, exact policy/version receipt, pause/drain/load/canary route switch | Skill-on is admitted only for a frozen store containing version-compatible ACTIVE Skills covering both datasets.  The manifest records posterior/library versions and first-turn Skill visibility; forced probes remain disabled.  Joint schedule resolution additionally includes the independent Skill-confirmation path in its held-out union. |
| `materialize_joint_qa_progressive_skill_training.py`, `freeze_joint_qa_training_schedule.py` | Necessary experiment materialization adaptation | Existing `freeze_joint_qa_training_schedule`, write-once cursor, `SkillStore`, YAML loader, and `validate_smoke_bounds` | Evidence gate is checked before a fixed train-only schedule and resolved Skill-on config can exist | The adapter selects one predeclared unused train position per dataset, binds exact ACTIVE Skill IDs/library/posterior/policy versions, and writes no rollout or model state.  The freeze CLI now forwards the optional Skill-confirmation path into the existing held-out-union check. |
| `materialize_joint_qa_additional_confirmation.py` | Necessary data-partition adaptation | Existing `prepare_agentgraph_datasets.CONVERTERS` and `prepare_joint_qa_partitions::{_take_candidate_prefix,_partition_record}` | Canonical sequential conversion, TaskRecord retagging, write-once publication, and problem-ID isolation | The original 64-item confirmation block is exhausted and the development block was exposed by prior architecture evaluation.  The adapter freezes canonical positions `[864:904]` after the untouched test block as validation-only data and emits a combined held-out union; it does not add a converter or change train/development/test records. |
| Progressive Step-0 and Skill-on Step-1 YAML files | Necessary configuration adaptation | Existing HotpotQA/TriviaQA evaluators, ten-arm v6 Executor catalog, formal zero-update LoRA, and joint-QA smoke trainer | Fixed development sample, local Qwen3.5-9B Director, `add_subgraph` search space, two train tasks x eight rollouts, one update, and two canaries | The training YAML remains a fail-closed template until evidence-gated ACTIVE Skill IDs and their exact library/posterior versions are materialized.  No placeholder is reported as an executed run. |

The SkillStore is frozen before natural GRPO collection.  A successful LoRA
update creates a new policy version, after which the prior ACTIVE Skills are
version-incompatible and must be suspended or independently revalidated before
another training epoch.  The fixed reported development block `[0:32]` and all
final-test tasks are never used for posterior fitting, EVSI, Skill confirmation,
or optimizer data.  Round 3 uses the disjoint development block `[32:52]` only
as adaptive Skill confirmation and permanently excludes it from reported
development metrics.

The first progressive Step-0 stream showed that component transactions removed
the singleton collapse but still produced only serial depth-2/3 graphs.  The
fresh Skill epoch therefore replaces the obsolete transaction-construction
prior with a rejectable dependency-aligned topology prior: parallel branches
are suggested only for independent evidence, finite reciprocal revision only
for a draft/critique dependency, and serial dependencies otherwise.  This is a
paired Skill candidate, not a base Director template, topology quota, reward,
or direct Canvas mutation; it can become ACTIVE only through the unchanged
independent evidence gate.

That first frozen evidence epoch rejected both candidates under the registered
gate, so `run_joint_qa_mace_skill.py --round 1` is only a versioned repeat of
the same caller chain: `JointQAPosteriorScheduler` -> randomized paired
intervention -> `SkillEvidencePipeline` -> deterministic gate.  It consumes
new train positions `[4:7]`, natural-candidate position `7`, and independent
Skill-confirmation positions `[20:40]`; the initial epoch and the formal GRPO
tasks remain excluded.  Its two replacement prompt priors come directly from
persisted answer-type/span and subject-relation failure cases.  One candidate
mentions conditional evidence fan-in, but terminal F1 remains only an
intent-to-treat Skill outcome; non-chain topology adoption requires a separate
receipt-based acceptance check and receives no reward.

The second frozen evidence epoch also rejected both selected candidates:
HotpotQA showed negative paired transfer from unconditional answer-span
handling, while TriviaQA's small positive mean did not clear the calibrated
lower bound.  `run_joint_qa_mace_skill.py --round 2` therefore remains the same
upstream caller chain and changes only immutable experiment coordinates and
the failure-derived prompt priors.  It consumes train positions `[13:16]`,
natural-candidate position `16`, and Skill-confirmation positions `[40:60]`,
while excluding both prior evidence rounds and the formal GRPO task positions.
The two priors condition semantic verification on subject/relation/qualifier or
answer-type ambiguity, and condition parallel evidence fan-in on actual
subproblem independence.  Their topology adoption receipts remain diagnostic
only and never enter terminal F1, GRPO reward, posterior observations, or the
Skill evidence gate.

Round 2 then established positive, zero-harm mean effects for semantic
grounding on both datasets, but its problem-cluster bootstrap lower bounds did
not clear the unchanged publication threshold.  Round 3 therefore combined
that observed answer contract with the FlowSteer
component boundary in one rejectable prior: conditionally independent evidence
branches and their semantic fan-in execute as one `ADD_SUBGRAPH` transaction,
and the Format Agent is added only after Canvas feedback.  Serial and
single-hop tasks retain the smallest directed semantic component.  The round
uses train positions `[17:20]`, natural-candidate position `20`, and the
progressive-runner-unused development positions `[32:52]`; the fixed
development evaluation `[0:32]`, all prior evidence, formal GRPO positions,
and final test remain excluded.  Those positions had appeared in an earlier
128-task architecture evaluation, so round 3 is adaptive revalidation rather
than independent confirmation.  Its negative result is retained, but it cannot
publish a Skill.  This is a refinement of immutable experiment coordinates and
prompt-prior content over the same runner, posterior, paired intervention, and
gate—not a new exploration or Skill architecture.

Round 3 selected the answer-handoff candidate on both datasets, but its paired
effects were negative and the unchanged gate rejected publication.  Its long
prompt prior also reduced depth-three final graphs and did not produce a
committed fan-in topology.  Round 4 therefore leaves the runtime and action
space unchanged and separates the prior into (1) conditional fan-in with
deferred Format and (2) exact-answer handoff.  This follows SkillFlow's rule
that a Skill is a rejectable prompt prior rather than a direct Canvas edit.
Discovery uses previously unused train positions `[48:51]` plus natural
position `51`.  Confirmation is materialized by the existing converters from
canonical source positions `[864:904]`, after the complete frozen test block;
these 40 fresh problems per dataset are validation-only and are included in
the optimizer schedule's held-out union.

`materialize_joint_qa_progressive_evaluations.py` and
`materialize_joint_qa_progressive_skill_training.py` derive the common delayed
visibility epoch from the two matched ACTIVE Skill records instead of assuming
epoch 2.  The training task positions are read from the existing frozen-schedule
configuration (HotpotQA position 9 and TriviaQA position 12), keeping the
materializer a thin adapter over `freeze_joint_qa_training_schedule` rather
than adding another sampler.

The progressive Skill-on micro-training boundary reuses the existing
`SkillLifecycleManager.audit` transition after a successful LoRA policy
publication.  The runner changes only the policy coordinate, persists every
affected `ACTIVE -> SUSPENDED` transition in the configured `SkillStore`, and
records the lifecycle receipt before post-update canaries.  This is the
necessary runner integration for the design document's policy-drift rule; it
does not add a second lifecycle implementation.  The same boundary now derives
the manifest's post-update canary bound from `policy_sync.post_update_canary_count`
instead of reporting a stale constant.

## Intermediate Canvas terminal-feedback alignment and Skill epoch 5

Round 4 supplied the causal evidence for one narrower architecture correction.
The `conditional_fan_in_deferred_format` prompt prior was executable end to
end, but most treatment trajectories either selected an Output prematurely or
reached `max_rounds`.  The Director observation was applying complete-graph and
Format checks to every intermediate Canvas, while both FlowSteer and the local
Runtime reserve those checks for the `FINISH` boundary.

| Current module | Classification | Reference source | Reused boundary | Minimal adaptation |
| --- | --- | --- | --- | --- |
| `director.py::AgentGraphOrchestrator._canvas_observation` | Necessary FlowSteer compatibility correction | FlowSteer `workflow_env.py::_step_internal`: `_check_finish_constraints` runs only in the `FINISH` branch; local `AgentWorkflowEnv.step` validates ordinary mutations and progressive execution with `require_complete=False` | Every accepted subgraph transaction still executes once and returns the same Canvas feedback; `FINISH` still enforces complete graph, Format Agent and exact-answer protocol | Intermediate observations use partial structural validation. `terminal_format_issue` is shown only after an Output has been selected. A premature `FINISH` remains rejected and its exact terminal issue is returned through `canvas_feedback`. |
| `run_joint_qa_mace_skill.py` epoch 5 | Same MACE/Bayesian/Skill caller chain with new immutable coordinates | Existing `JointQAPosteriorScheduler`, randomized paired intervention, `SkillEvidencePipeline`, delayed activation and the two round-4 atomic hypotheses | No topology reward, direct Canvas edit, fixed Agent count, or relaxed gate | The obsolete round-4 workaround text is removed after the observation correction. Discovery uses train positions `[52:55]`, the natural candidate uses position `55`, and confirmation uses a fresh validation-only canonical block `[904:944]`. The changed observation protocol receives a new prompt version so its trajectories cannot group with earlier behavior. |
| `joint_qa_round5_confirmation.yaml` | Existing additional-confirmation adapter reuse | `materialize_joint_qa_additional_confirmation.py`, which directly reuses the existing dataset converters and partition record adapter | Write-once sequential conversion and exclusion from GRPO, development metrics and final test | Publishes 40 new tasks per dataset as `skill_confirmation_round5`; the fixed test block and every earlier evidence task remain untouched. |

Round 4 itself remains a valid rejection result: HotpotQA had paired mean
`delta F1=-0.1923` with calibrated interval `[-0.3423,-0.0500]`; TriviaQA had
`delta F1=+0.0375` but interval `[-0.0875,+0.1650]` and failed the unchanged
harm gate. Neither candidate was activated or admitted to GRPO. Epoch 5 is
therefore a new-policy revalidation after a source-mapped observation fix, not
post-hoc activation of the rejected evidence.

Epoch 5 was then aborted after 4/174 trajectories because its treatment arm
exposed a narrower JSON compatibility fault. The Director represented an
intentionally absent Output as `"output_agent_id": null`, while the strict
parser accepted only an omitted field. Nine such rejections occurred in
candidate arms and none in incumbent arms, so those trajectories are retained
only as diagnostics and are excluded from the posterior, Skill gate and GRPO.

| Current module | Classification | Reference source | Reused boundary | Minimal adaptation |
| --- | --- | --- | --- | --- |
| `agent_action_parser.py::AgentActionParser._build_action` (`ADD_SUBGRAPH`) | Necessary JSON-schema compatibility adaptation | Existing `AgentAction.output_agent_id: Optional[str]`, FlowSteer structure-level ADD before terminal Output selection, and the project action schema where `output_agent_id` is optional | Omitted Output and explicit JSON null both denote an output-free intermediate component; a non-null value must still be a non-empty Agent ID | Normalize only `ADD_SUBGRAPH.output_agent_id=null` to `None`. Empty strings and non-string, non-null values remain invalid; no other optional field is relaxed. |
| `run_joint_qa_mace_skill.py` epoch 6 | Same frozen evidence caller chain with a new tool identity | Existing epoch-5 candidate hypotheses, paired-arm randomization, posterior scheduler and unchanged Skill gate | The four contaminated epoch-5 trajectories and its reserved tasks are never resumed or pooled | Use train discovery `[56:59]`, natural position `59`, fresh canonical confirmation `[944:984]`, and a new tool version reflecting nullable optional Output semantics. |

This adaptation does not add an Agent, relation, topology prior, repair pass or
reward. It only makes the parser agree with the already-declared optional
field and with the Runtime's existing support for `output_agent_id=None`.

## Multidataset Tool, ReAct, and Coding architecture sources

This section records the source decision and the implementation state of the
multidataset Tool/ReAct/Coding boundary.  A listed upstream reference is an
implementation source, not an instruction to replace FlowSteer's Director or
progressive Canvas.  Status is classified as **direct reuse**, **necessary
adaptation**, **project algorithm addition**, or **not implemented/executed**;
the last category must not be reported as completed capability.
Source selection follows **SkillFlow > FlowSteer > necessary project
adaptation**: SkillFlow is the first source for Qwen/Supervisor, Tool, Skill
and bounded execution contracts; FlowSteer is retained for progressive Canvas,
action-mask, trajectory and evaluator timing; local code is limited to the
incompatibility stated in each row.

| Classification | Current multidataset scope |
| --- | --- |
| **Direct reuse** | FlowSteer progressive Canvas `edit -> execute -> feedback` semantics, explicit `FINISH`, action masks and trajectory records; SkillFlow `StructuredAction`, Tool registry contracts, model-visible `public_context.tools` / `SUPERVISOR_TOOLS` exact action-name and argument-schema surfaces, `RetrievalIndex`, bounded computation executors, RAGEN environments, MedRAG BM25 assets, SWE-bench worktree/repository primitives and evidence-library contracts. |
| **Necessary adaptation** | Free AgentGraph validation/scheduling, task-scoped `reasoning|react|coding` dispatch, fixed-name `ToolCapability.action_schemas` admission, public action/observation continuation state, per-action decoding budget, SQLite thread affinity, evaluator-locked RAGEN session/replay, dynamic native environment action grammar, required interactive environment actor at FINISH, frozen MedRAG resource binding, separated SWE worktree ownership, unified native-evaluator receipts, and exact model-capability admission. |
| **Project algorithm addition** | Typed `CommunicationEnvelope`, `ToolCapability`, measured `ToolReceipt`, the existing same-prefix paired AgentGraph posterior/evidence gate, and the receipt-only multidataset report renderer. |
| **Not implemented/executed** | A live SWE-bench Coding trajectory and valid official-harness `resolved` receipt, evidence-gated multidataset `ACTIVE` Skills, Skill injection, and micro-training for the new unified Tool/Environment/Coding conditions. Six other fixed conditions produced evaluator-valid two-task Stable Zero receipts. The prior joint-QA core separately has two receipt-valid one-update LoRA runs with successful policy synchronization; those receipts do not establish learning for the new action domains. |

| current_file | upstream_reference | reused_logic | required_adaptation | why_direct_reuse_failed | dataset_scope |
| --- | --- | --- | --- | --- | --- |
| `src/interactive/agent_runtime.py::{AgentRuntime.execute,_build_plan,_execute_block,_upstream,_request,_invoke}`; `agent_workflow_env.py::AgentWorkflowEnv.step`; `agent_action_parser.py::AgentActionParser`; `agent_graph.py::AgentGraph` | FlowSteer `src/interactive/workflow_env.py::{InteractiveWorkflowEnv.step,_step_internal,_execute_workflow,_format_feedback}`, `workflow_graph.py::WorkflowGraph`, `workflow_builder.py::{InteractiveWorkflowBuilder.run_loop_async,TurnRecord,Trajectory,create_action_mask}`; SkillFlow `src/skillev/runtime/bounded_agent.py::BoundedAgent.execute_turn` and `runtime/execution.py::{RolloutEnvironmentSession,EnvironmentObservation}` | One atomic edit per Director turn, transactional Canvas mutation/validation, execution after an accepted edit, execution feedback, explicit `FINISH`, action masks, trajectory records, bounded public observation/action execution | Keep the existing quotient-DAG and finite reciprocal scheduler; dispatch each node through a bounded execution adapter selected by its declared execution mode | FlowSteer uses a fixed Operator AST and SkillFlow's bounded agent is a single-Supervisor sequential environment loop; neither can replace a heterogeneous multi-model AgentGraph scheduler | All seven datasets |
| `src/interactive/director.py::DIRECTOR_SYSTEM_PROMPT` | FlowSteer one-action Director loop, free Canvas feedback, and the current AgentGraph action schema; SkillFlow's separation of Tool `resource_id` from action `name` | One action per turn, short neutral prompt, current state/feedback, model catalog and Tool catalog remain unchanged | The v2 HotpotQA failure receipt showed that the free-schema Director put top-level `output_agent_id` inside an Agent object and used action names (`search`/`read`) as `allowed_tools`. V3 added only those two wire-schema clarifications. The later fixed-development v4 wrong-demo audit found accepted contracts that changed the task relation or qualifier, so v5 adds one neutral contract-fidelity boundary and states that independent evidence may merge before the separate Format sink. It still adds no named role recipe, required Agent count, topology preference, benchmark answer, or Skill | FlowSteer's fixed Operator grammar has no heterogeneous Tool resource identifier, and neither upstream prompt represents free-text semantic contracts selected by a heterogeneous model catalog. The failure-driven wording is therefore a minimal project adaptation, not an upstream Skill or fixed workflow template | All Tool-capable datasets; HotpotQA semantic-contract v5 is the first configured consumer of the v5 prompt |
| `src/interactive/agent_runtime.py::AgentRuntime.validate_execution_contracts`; `agent_workflow_env.py::AgentWorkflowEnv.step`; `openai_gateway.py::build_agent_messages` | FlowSteer `InteractiveWorkflowEnv._step_internal` transactional edit/validation/execution/feedback boundary and extraction-only Format prompt; SkillFlow `BoundedAgent.execute_turn`, Tool resource/action admission, and evidence-grounded concise QA contracts | The existing Runtime remains the sole authority for `reasoning|react|coding` and Tool compatibility; an accepted Canvas transaction still executes exactly once. The Format Agent still extracts one completed semantic solution and remains separate from task solving | **Necessary compatibility adaptation:** call the existing Runtime validator on the candidate graph before committing the Canvas revision, so an invalid `reasoning + allowed_tools` declaration is returned as immediate edit feedback without execution or persistence. Intermediate contracts preserve relation, qualifiers, comparison criterion and answer type; explicit verification independently reconstructs evidence. Format fails closed with an empty wrapper for unresolved uncertainty/candidates instead of converting unknown to `no`, and preserves evidence-aligned units/qualifiers | FlowSteer uses a fixed Operator AST and SkillFlow uses one bounded Supervisor, so neither has a heterogeneous `execution_mode + allowed_tools` graph mutation to validate. Their prompts also do not cover this project's free-text artifact handoff ambiguity. The semantic and Format clauses are failure-driven project adaptations, not copied ACTIVE Skills; old restored snapshots with several pre-existing invalid declarations may require whole-subgraph repair because candidate validation checks every node | Canvas execution validation applies to all datasets; semantic/Format v5 is evaluated on the fixed HotpotQA development-128 condition |
| `src/interactive/tool_runtime.py::{StructuredAction,ToolRequest,ToolResult,ToolBackend,ToolRegistration,ToolRegistry,ToolCapability,ToolReceipt}`; `react_execution.py::{ReactExecutionError,ToolReactExecutionAdapter._contract,ToolReactExecutionAdapter.execute}` | SkillFlow `src/skillev/runtime/tools.py::{ToolRequest,ToolResult,ToolBackend,ToolRegistration,ToolRegistry}`, `runtime/contracts.py::{StructuredAction,ActionKind,BudgetVector}`, `runtime/bounded_agent.py::{execute_turn,_execute_action,_reserve}`, `rollout/context.py::{CanonicalInitialContextAssembler,_ACTION_GUIDANCE}`, public action history, `RolloutDecoding.max_action_tokens`, benchmark `to_rollout_task` / `to_retrieval_rollout_task` `public_context.tools` maps, and deployed `training/batch_inference.py::SUPERVISOR_TOOLS` function `name` / `parameters` schemas | Immutable resource-ID registry, normalized request/result, exact five-field `StructuredAction`, operation name separate from `resource_id`, model-visible exact action names/argument schemas, public action plus Tool observation on continuation, bounded action decoding, explicit completion, and one action/observation turn are direct reuse | `ToolCapability.action_schemas` projects SkillFlow's per-task action schema into the heterogeneous AgentGraph Tool catalog; the ReAct contract renders each fixed schema and explicitly rejects the legacy two-field `ToolRequest` wire shape. Each subcall propagates `max_action_tokens`; successful action identity is returned with the public observation; turn-exhaustion exceptions preserve partial trace/receipts. The pre-dispatch exact-name check admits only names published for that resource, while the concrete backend remains authoritative for arguments and semantics. Generic Tool/ReAct fails closed for resources without fixed schemas. Async timeout and measured `ToolReceipt` remain project additions | SkillFlow's `ToolRegistration` has no dataset/schema/side-effect/version metadata or multi-Agent asynchronous receipt; its exact domain, action history and token budget live in separate rollout structures. FlowSteer's heterogeneous runtime needs a thin task-scoped projection but does not replace SkillFlow's `StructuredAction` or backend contract | HotpotQA, TriviaQA, AIME-2025 development, HealthBench Professional, and SWE-bench fixed-name Tools; the common registry also carries the two environment resources |
| `src/interactive/qa_retrieval.py::{SkillFlowQARetriever,QARetrievalReceipt,augment_task_with_retrieval}`; `qa_tool_adapter.py::{QARetrievalToolBackend,QARetrievalReactExecutionAdapter,open_qa_tool_registry}`; `react_execution.py::ToolReactExecutionAdapter` | SkillFlow `src/skillev/benchmarks/retrieval.py::{RetrievalIndex.open,RetrievalIndex.search,RetrievalIndex.read,QARetrievalEnvironment.execute,QARetrievalEnvironment.validate_completion}` | Read-only versioned `RetrievalIndex`, answer-free `search`/`read` actions under the single `qa-retrieval` resource, public passage IDs/text, corpus/index identity and explicit Tool/complete actions are direct reuse | Necessary adaptation keeps SQLite on one thread and persists query, top-k, passage IDs, observations, latency and corpus version. A `read` is admitted only for a canonical `passage_id` returned by a successful search observation in the same bounded execution; titles and document IDs fail closed. Exact repeated Tool actions are returned as public duplicate observations without redispatch. A measured cold query took about 16 seconds, so the configured timeout remains longer than the earlier 10-second limit | The historical deterministic-prefetch wrapper remains only for legacy results; it is not used by the model-driven Tool condition. The canonical-ID admission is a failure-driven AgentGraph adapter because the upstream environment accepts any syntactically valid opaque ID and reports `passage_not_found`, while the free Canvas previously allowed a read-only resource assignment. Closed-context/question-only Direct and retrieval-enabled AgentGraph scores remain separate protocols | HotpotQA and TriviaQA |
| `src/interactive/computation_tools.py::{AIMEComputationToolBackend,create_aime_computation_registry}`; `react_execution.py::ToolReactExecutionAdapter` | SkillFlow `training/tools.py::{execute_tool,_calculator,_python_exec,_exec_code_in_process}` | Calculator namespace, Python child-process execution and hard timeout are direct reuse | Necessary adaptation exposes both functions as AIME-scoped Tool resources and omits SkillFlow's Tool success/failure reward. The AIME-2025 two-task natural Stable Zero selected no computation Tool. A diagnostic-only forced probe produced successful `python_exec` and `calculator` receipts, then emitted invalid JSON and exhausted its ReAct budget; backend capability is established, model/termination compliance is not | The upstream functions are coupled to SkillFlow's training environment and reward path; only their public observation-producing execution is reused | AIME-2025 development for the AIME 2026 target adapter |
| `src/interactive/healthbench_tool_adapter.py::{FrozenMedRAGBM25Corpus,HealthBenchMedRAGSearchToolBackend,open_healthbench_medrag_tool_registry}`; `react_execution.py::ToolReactExecutionAdapter` | SkillFlow MedRAG textbooks runtime and its formal preflight/BM25 scoring boundary | Frozen textbook corpus, source revision/row-count binding, tokenization, BM25 constants, top-3 result projection and public snippet text are direct reuse | Necessary adaptation registers one HealthBench-only read-only resource and keeps evaluator rubrics, reference answer and judge state outside Tool inputs. The two-task natural Stable Zero selected no MedRAG Tool; a diagnostic-only `search→complete` forced probe passed exact schema, backend and model termination. That probe is excluded from benchmark and Skill evidence | SkillFlow's released task list does not provide a HealthBench AgentGraph Tool boundary; this adapter exposes only the existing frozen MedRAG corpus | HealthBench Professional |
| `src/interactive/environment_execution.py::{EnvironmentExecutionAdapter,RAGENEnvironmentSession,RAGENEnvironmentSessionFactory,evaluator_locked_ragen_session_factory}`; `agent_workflow_env.py::{required_tool_id,required_tool_issue}`; `task_evaluator.py::_evaluate_environment` | SkillFlow deployed `src/ragen_adapter.py::{RAGENAdapter.reset,RAGENAdapter.step,ALFWorldEnv.reset,ALFWorldEnv.step,WebShopEnv.reset,WebShopEnv.step}`; SkillFlow `runtime/bounded_agent.py::BoundedAgent.execute_turn`, `runtime/execution.py::{RolloutEnvironmentSession,EnvironmentObservation}`, `RolloutDecoding.max_action_tokens`, benchmark `public_context.tools`, and per-step admissible-action observations | Real reset, current admissible native actions, environment step, public observation, bounded turns, bounded action decoding and failure observations are direct reuse | Necessary adaptation creates one request-scoped AgentRuntime resource, records typed environment revisions/Tool receipts/evaluator replay trace, and excludes reward/`won` from model-visible state. These resources deliberately expose an empty static `action_schemas` map: every provider turn receives the live environment's admissible native actions, which are the sole action-space authority. The generic fixed-name Tool/ReAct adapter therefore cannot dispatch this resource. The executor's native action grammar takes precedence over free Canvas formatting, interactive FINISH requires exactly one ReAct Agent owning the condition's environment Tool, and the SkillFlow action-token bound is propagated to both Direct and AgentGraph environment-policy calls so long WebShop histories fit the local Qwen context. These changes were derived from preserved v1/v3 failure receipts and validated on fixed development tasks in v2/v4 | RAGEN has no AgentGraph or typed receipt boundary, a static action-name catalog cannot represent environment-state-dependent native actions, and the generic Executor completion limit can exceed the smaller local serving context after several long observations | ALFWorld and WebShop |
| `scripts/evaluate_hotpotqa_round.py::_collect_graph`; `scripts/evaluate_completion_benchmark_round.py` optional `task_timeout_seconds` and per-result failure checkpoint | FlowSteer `src/aflow_executor.py::AFlowExecutor.execute_workflow` configurable `asyncio.wait_for` wall-clock boundary (default 300 seconds); project `scripts/evaluate_agentgraph_architecture.py` per-task checkpoint boundary | A bounded workflow attempt and task-by-task checkpoint are reused | **Necessary interactive-evaluation adaptation:** the shared collector may apply the same configurable wall-clock boundary to one complete Canvas rollout. A deadline is recorded as an operational collection failure, never converted into native ALFWorld/WebShop task failure; successful exact-match trajectory receipts remain resumable. Failure JSONL and manifest are checkpointed after every completed attempt so interruption does not erase timeout evidence | FlowSteer's legacy executor bounds one generated workflow, while the progressive Canvas collector can re-execute multiple environment episodes inside one task and otherwise has no whole-task wall-clock bound | ALFWorld fixed-128 evaluation; optional for other completion benchmarks |
| `src/interactive/swe_worktree.py::{SWEbenchRepositoryIdentity,PreparedSWEbenchWorktree,prepare_swebench_worktree_for_task}`; `coding_tools.py::{RepositoryToolBackend,create_swebench_repository_registry}`; `coding_execution.py::CodingExecutionAdapter`; `swebench_adapter.py::OfficialSWEbenchHarness`; `config/evaluation_swebench_regular_dev_coding_agent_v2.yaml` | SkillFlow `training/environment.py::{GenericTaskEnvironment._setup_swe_repo,cleanup,_resolve_repo_path,_handle_bash,_handle_list_files,_handle_search_code,_handle_view_file,_handle_str_replace_editor,_generate_filemap,_generate_workspace_diff,_handle_run_tests,_run_tests_in_swe_env}` and `training/swe_bench_eval.py::evaluate_patch`; official Codex CLI `--codex-run-as-apply-patch` / `codex-rs/apply-patch` | Base-commit detached worktree lifecycle, repository-relative list/search/view, `bash`, create/replace/insert/undo, long-file AST file maps, workspace diff, targeted test execution, Tool result returned to the next model turn, and official harness evaluation are direct reuse/thin ports. `apply_patch` is executed by the official Codex CLI rather than a project parser | Necessary adaptation separates worktree ownership from SkillFlow's monolithic episode, converts Tool outputs into structured receipts, and places ephemeral v2 worktrees inside the Codex workspace required by the installed CLI. All four regular-dev repository mirrors and 128 base commits are available; a task-pinned no-model Tool canary passed. The official Docker harness still fails preflight, so execution remains fail-closed before model Coding trajectories and no proxy `resolved` score is emitted | AST file maps plus textual search do not constitute an LSP symbol/reference engine. The official harness remains the only source of `resolved` | SWE-bench regular dev; Verified reserved for final evaluation |
| `src/interactive/agent_runtime.py::{UpstreamMessage,CommunicationEnvelope,AgentExecutionAdapter}`; `agent_graph.py::{AgentNode,AgentExecutionMode}`; `records.py::{ExecutionRecord,TurnRecord,TrajectoryRecord}`; `rollout_collector.py::{_execution_record,_runtime_summary}` | FlowSteer `workflow_builder.py::{TurnRecord,Trajectory,create_action_mask}`; SkillFlow `src/skillev/rollout/artifact.py::{CompletedStepDraft,materialize_trajectory_step,RolloutManifest,RolloutArtifact,finalize_trajectory_record}` and `runtime/execution.py::EnvironmentObservation` | Sampled action IDs/masks, graph snapshot/revision, per-call execution receipt, complete action/observation step commit and terminal evaluator receipt are direct reuse | `CommunicationEnvelope` and heterogeneous `reasoning|react|coding` dispatch are project additions. JSON-safe persistence now includes model calls, Tool receipts, ReAct traces, environment reset/transitions/revisions/evaluator replay, Coding receipts, tokens and latency while retaining legacy text fields | FlowSteer has no tool-level receipts, and SkillFlow's trajectory is single-agent sequential; neither represents typed artifacts across AgentGraph edges | All seven datasets |
| `src/interactive/task_evaluator.py::evaluate_task`; `aime2026_adapter.py`; `swebench_adapter.py::OfficialSWEbenchHarness`; `scripts/evaluate_completion_benchmark_round.py::{validate_completion_benchmark_config,_select_tasks,_direct_one}` | HotpotQA official answer normalization; TriviaQA accepted-answer normalization; SkillFlow Protocol 10 integer evaluation and deployed RAGEN adapter; OpenAI simple-evals HealthBench rubric contract; official SWE-bench harness; local `scripts/run_joint_qa_mace_skill.py::_require_partition`; FlowSteer task-only Runtime input and terminal evaluator timing as retained by `scripts/train_agentgraph_smoke.py::_workflow_problem` | Each benchmark's native answer/terminal/harness contract and primary metric are retained. Model generation receives `TaskRecord.question`; ground truth and evaluator payload remain evaluator-only | Necessary adaptation normalizes evaluator receipts behind one async boundary and replays only the recorded RAGEN trajectory. The completion runner now validates `stage`, requires `split=test` for `final_evaluation`, and makes `_select_tasks` check `required_partition` on both the source selection and any frozen selection before execution. `development_hotpotqa_tool_react_stable_zero_v4.yaml` binds `stage=development` and `required_partition=development`; its two-task canary is therefore labelled a chain/evaluator smoke check rather than benchmark accuracy. Gold answers, aliases, rubrics, environment reward/`won` and SWE-bench resolution remain evaluator-only | Upstream runners do not know the project's `joint_qa_partition` metadata. The check is the same fail-closed comparison already used by `_require_partition`, not a new sampler or evaluator. The protected `joint_qa_v2` test partition remains unexecuted | All seven datasets; explicit v4 partition guard currently configured for HotpotQA development |
| `scripts/report_multidataset_stable_zero.py`; `MULTIDATASET_AGENT_ARCHITECTURE_STABLEZERO_REPORT.md`; `reports/multidataset_stablezero/*.md` | Existing FlowSteer trajectory/turn/action-mask records, SkillFlow rollout manifests, and this project's unified paired-result/evaluator receipt schemas | No new execution semantics are introduced; the renderer reads only persisted manifests, paired results, trajectories, full recorded `CommunicationEnvelope` values, progressive Canvas feedback/revisions, public `StructuredAction`/observation traces, actual Director/Executor model-call telemetry, Tool/environment receipts, capability-canary receipts, evidence-gate publications, and prior bounded-training/sync receipts | **Project extension:** produces Chinese receipt-backed per-dataset and consolidated reports without model, environment, evaluator, Skill publication or training calls. Missing official evidence remains explicitly unmeasurable, prior joint-QA micro-training is version-separated from the new unified Runtime, and hidden chain-of-thought is neither read nor reconstructed | Neither upstream repository includes a renderer for this project's seven heterogeneous conditions and typed receipt schema | All seven datasets |
| `scripts/probe_model_capabilities.py`; `config/model_catalog_multidataset_tool_v1.yaml`; `config/model_catalog_multidataset_tool_v2.yaml`; `config/development_hotpotqa_tool_react_stable_zero_v4.yaml` | Existing exact `/v1/models` discovery, SkillFlow/Qwen chat-template configuration, the OpenAI-compatible Agent gateway, the immutable v1 catalog, and the saved local Qwen3.5-9B non-thinking canary | Exact provider/model IDs, catalog order/weights and gateway message contract are reused. Historical trajectories continue to identify v1 in their receipts | Necessary adaptation admits a model only after explicit Text, `StructuredAction`, and Coding-format canaries, with no alias substitution or silent fallback. V2 copies the exact v1 entries and adds the passed local `supervisor_theta` canary source plus frozen `chat_template_enable_thinking=false` metadata; only new conditions such as HotpotQA v4 reference v2. No historical receipt is rewritten or regrouped | Listing a model does not establish its ReAct/Coding compatibility, and a Qwen reasoning-only response cannot be treated as message content. Mutating v1 would mix catalog coordinates across persisted trajectories, so versioned future-only admission is the minimal compatibility change. The Flow-Director remains local Qwen3.5-9B and is not substituted with an API model | Future heterogeneous Executor conditions for all seven datasets; HotpotQA v4 development is the first configured consumer |
| `src/interactive/skills/{pipeline.py,validator.py,lifecycle.py,retrieval.py,schema.py}`; `scripts/train_agentgraph_smoke.py::{LiveSmokeBackend._skill_query,LiveSmokeBackend._forced_probe_condition_matches}` | SkillFlow `src/skillev/evidence/schema.py::{EvidenceRecord,may_update_posterior}`, `evidence/store.py::{EvidenceStore,SplitEvidenceStores}`, `runtime/skill_library.py::{SkillLibrary,SkillLibraryState,active_documents}`, `runtime/skills.py::SkillApplicability`, `evolution/retriever.py::{TaskRetrievalFeatures,TaskConditionedSkillRetriever.retrieve,applicability_matches,RetrievalDecision}` | Physical evidence split, immutable/versioned ACTIVE library, eligible evidence only, lifecycle and retrieval receipt are reused. In particular, the exact SkillFlow applicability predicate `required_tools ⊆ available_tools` is applied before an ACTIVE Skill can become a Director-visible prompt prior | The same-prefix randomized paired AgentGraph intervention and posterior evidence gate are project algorithm additions. The task-scoped adapter projects only `availability=true`, dataset-compatible exact `tool_id` values from the existing `ToolRegistry` into `SkillQuery.available_tools`; natural retrieval and forced-probe matching call the same fail-closed predicate. Compatibility remains bound to model, policy, Tool and evaluator versions | SkillFlow's retriever consumes `RolloutTask.available_tools`, while this runtime owns Tools through `AgentRuntime.tool_registry`; this field projection is the only required interface adaptation. For the current seven-dataset phase, evidence-gated `ACTIVE` Skills, Executor-side Skill invocation and Skill-on micro-training have **not been executed**. `ActionKind.SKILL` remains rejected because the local runtime does not implement Executor-side Skill execution; the structured exposure/invocation receipt admission surface is recorded separately below. Earlier joint-QA optimizer/sync receipts validate only the common Director LoRA training boundary | All seven datasets |
| `src/interactive/records.py::{TurnRecord,TrajectoryRecord,canonical_invoked_skill_ids}`; `rollout_collector.py::{AgentGraphRolloutCollector,ActiveSkillProvider}`; `skills/pipeline.py::SkillEvidencePipeline.active_skill_ids`; `scripts/train_agentgraph_smoke.py::LiveSmokeBackend.collect` | SkillFlow `rollout/context.py::CanonicalInitialContextAssembler.assemble`, `contracts/ttb_trajectory.py::{InitialContext,TrajectoryStep}`, and `contracts/skill_invocation.py::{parse_action_invocation,canonical_invoked_skill_ids,validate_trajectory_skill_invocations}`; FlowSteer `workflow_builder.py::{TurnRecord,Trajectory}` | The pinned, version-compatible ACTIVE Skill IDs are recorded independently from the ranked H0 retrieval; each Director turn records the exact ranked Skill IDs rendered into its public prompt; Skill invocation credit must be derived from the public structured action and must be a subset of ACTIVE and retrieved IDs | **Necessary adaptation:** the existing AgentGraph collector receives an independent ACTIVE-library provider and persists exposure IDs on FlowSteer turn/trajectory records while retaining legacy JSON compatibility. The current Executor rejects `ActionKind.SKILL`, so a rejected action records zero invocation and any successful-looking or environment-invented credit fails closed. This adds the missing receipt/admission surface only; it does not enable Skill execution | SkillFlow's trajectory is a single-Supervisor step sequence, while this project persists heterogeneous Agent calls inside FlowSteer turns. A thin projection is required to bind the same H0/ACTIVE/invocation invariants to AgentGraph without exposing hidden reasoning or creating another Skill runtime | All seven datasets |
| `scripts/train_agentgraph_smoke.py::_qa_tool_runtime_settings`; `scripts/run_hotpotqa_tool_availability_pair.py`; `config/development_hotpotqa_tool_availability_pair_v1.yaml` | Existing `run_joint_qa_mace_skill.py::_paired_probe`, `src/interactive/exploration/paired_probe.py::{randomize_probe_order,ProbeOrder}`, `evaluate_hotpotqa_round.py::{_select_tasks,_direct_resume_matches}`, and SkillFlow QA `RetrievalIndex` runtime already wired through `LiveSmokeBackend.collect` | Randomized arm presentation, one shared empty-Canvas prefix, exact task/version/sampling coordinates, independent condition receipts, forced-probe exclusion from GRPO, native HotpotQA EM/F1, behavior-adapter preflight, and the existing QA search/read Runtime are reused | **Necessary experiment adaptation:** one frozen experiment may name distinct `tool_off` and `tool_on` rollout conditions. OFF returns the existing base Runtime; ON returns the existing task-scoped SkillFlow QA Runtime. The dedicated two-task development CLI reuses two exact Direct receipts with zero new Direct calls, verifies the persisted Tool-catalog treatment, reads ToolReceipt only from actual executor calls, and records the Tool-availability intent-to-treat effect. It does not label availability as invocation, Tool usefulness, Skill evidence, benchmark accuracy, or GRPO data | The existing completion runner supports only one AgentGraph condition, while the joint-QA paired runner intervenes on a Skill prompt prior rather than Tool availability. Reusing either result as an OFF/ON pair would mix condition and protocol identities, so a thin coordinator over the existing collector is required | HotpotQA exposed development canary; not held-out or benchmark evidence |
| `scripts/report_multidataset_stable_zero.py::{_hotpot_tool_pair_report,_pair_topology}` | Existing receipt-only renderer in the same file; `AgentGraph.topology_statistics`; persisted output from `run_hotpotqa_tool_availability_pair.py` | Native evaluator receipts, Tool/Skill invocation receipts, Director/Executor telemetry, communication envelopes, and the runtime's exact topology statistics are read without model, Tool, evaluator, environment, or training calls | **NEW_PROJECT_EXTENSION:** render the completed OFF/ON pair as an independent JSON/Chinese Markdown diagnostic and append only a compact section to the Hotpot and total reports. The four forced trajectories are intentionally excluded from natural-policy Stable Zero, ToolReceipt and workflow aggregates | Neither FlowSteer nor SkillFlow contains this project-specific Tool-availability estimand or its report schema; a receipt-only renderer is required, while all execution and topology semantics remain upstream/local-runtime reuse | HotpotQA exposed development canary reporting only |

## QA semantic handoff, retrieval continuation, and terminal admission v6.2

The HotpotQA semantic-contract v5 development trajectories established that
all 43 directed receipts in the 34 EM failures matched their source artifacts;
the failure was not message transport.  The first divergence was instead
answer-span/qualifier drift, Format projection, free-contract target drift,
invalid retrieval continuation, or failure to submit an already valid terminal
artifact. V6.1 changed only the public contracts at those boundaries, but its
full development-128 result regressed. V6.2 therefore restores FlowSteer's
actual `Format` Operator call boundary: one problem plus one already-computed
solution under the directly imported `FORMAT_PROMPT`. The AgentGraph-specific
part is limited to identifying the immediate Format predecessor and requiring
an explicit semantic-answer/evidence handoff; the typed communication envelope
remains complete in the trajectory receipt but is not injected into the
extraction-only Format invocation.

| Current module | Classification | Upstream source | Reused boundary | Minimal adaptation and exclusion |
| --- | --- | --- | --- | --- |
| `agent_runtime.py::{AgentRequest,AgentRuntime._execute_component}`; `openai_gateway.py::build_agent_messages` | Direct FlowSteer Format reuse plus necessary free-AgentGraph adaptation | FlowSteer `scripts/operators.py::Format` and `scripts/prompts/prompt.py::FORMAT_PROMPT` | A distinct terminal Format node receives one completed semantic solution and performs extraction only; FlowSteer's `FORMAT_PROMPT` is imported directly and receives the clean `problem + solution` pair | Runtime marks only the immediate directed predecessor of the Format sink. That predecessor emits one explicit `Candidate answer` and separate evidence; the Format call receives the predecessor artifact without the persisted AgentGraph envelope and adds only the required `<answer>...</answer>` terminal wrapper. It does not verify, merge branches, consult the evaluator, or use benchmark answers. Directly replacing this handoff with FlowSteer's generic `ANSWER_GENERATION_PROMPT` was tested and rejected after a paired 16-task regression, so that failed intervention is not present in the effective source. |
| `qa_tool_adapter.py::{QARetrievalToolBackend,QARetrievalReactExecutionAdapter}` | Direct SkillFlow resource-domain reuse plus necessary admission adaptation | SkillFlow `benchmarks/retrieval.py::QARetrievalEnvironment` single `resource_id='qa-retrieval'` with `search/read` actions | One resource, exact StructuredAction domain, public search observation, opaque canonical passage ID, then read | The former two-resource projection is removed. A title-to-read action is rejected before the backend unless its exact passage ID occurred in this bounded execution's successful search observation. No query, ID or answer is synthesized. |
| `react_execution.py::ToolReactExecutionAdapter` | Direct continuation reuse and necessary multi-Agent projection | SkillFlow `scoring/rendering.py::_render_history`, `runtime/bounded_agent.py::execute_turn`, and its public Action/Observation records | Every prior sampled action and parse/schema/tool observation is available to the next bounded turn; completion remains explicit | Parse failures now return the exact public `action_text`; structured failures return the parsed action; an exact repeated dispatched Tool request is observed but not redispatched. Hidden reasoning is not stored or reconstructed. |
| `agent_workflow_env.py::finish_admissibility`; `director.py::_canvas_observation` | Direct terminal-validation reuse | FlowSteer `workflow_env.py::_check_finish_constraints` and SkillFlow `BoundedAgent._validate_completion` | A completion is terminal only after current-revision structural, Format, Tool and answer-protocol validation and an explicit `FINISH` action | Only a positive current-revision admissibility receipt is exposed. There is no automatic FINISH, round counter, shortest-path hint, topology preference, cached historical answer substitution, or relaxation of explicit terminal semantics. |

This revision does not activate a Skill, add a topology reward, force fan-in or
reciprocal communication, or update Director/Executor weights.  The earlier
global fan-in candidate remains rejected by its negative paired evidence.

### HotpotQA development evidence for v6.2

The frozen development-128 run in
`artifacts/qa_orchestration_tool_v6_2_development/hotpotqa` completed with 128
valid evaluator receipts and no operational collection failure. The local
Direct condition scored EM `70.3125%` / token F1 `78.6792%`; AgentGraph v6.2
scored EM `71.09375%` / token F1 `81.4453125%`, a change of `+0.78125` EM and
`+2.76616` F1 percentage points. There were 126 explicit `FINISH` submissions
and two `max_rounds` truncations. These are exposed development measurements,
not held-out test estimates, and no training or Skill injection occurred.

The v6.1 -> v6.2 comparison uses the same task IDs, sampling schedule,
sequence coordinates, model catalog and Tool version. V6.2 improved EM by
`3.90625` and F1 by `4.19461` percentage points over v6.1. The older semantic
contract v5 measured EM `73.4375%` / F1 `83.4037%`, but it used a different
sampling purpose and derived Director seeds, so that difference is descriptive
rather than a paired causal estimate.

Two additional development interventions were retained only as rejection
evidence. Directly using FlowSteer's generic `ANSWER_GENERATION_PROMPT` for the
reasoning predecessor reduced a seed-aligned first-16 panel from v6.2 EM
`56.25%` / F1 `71.25%` to EM `37.5%` / F1 `60.625%`; the effective code was
reverted. Increasing the episode horizon from 20 to 22 made a six-task panel
explicitly terminate `6/6` instead of `5/6`, but did not recover the target
answer and reduced panel EM from `66.67%` to `50%`; it was therefore not
accepted as an accuracy improvement.

V6.2 communication receipts show no systematic transport fault: 111 of 113
completed `Candidate answer` handoffs are preserved by Format under HotpotQA
normalization, and the one missing Output inbox belongs to a graph that never
created an Output Agent. The dominant remaining errors are semantic relation /
answer-type selection and answer-span canonicalization in the predecessor.
The final graphs are `single=1`, `serial_2=94`, and `serial_3_plus=33`; no
natural fan-in or reciprocal topology was committed. Descriptive performance
of three-Agent graphs is higher than two-Agent graphs, but there is no paired
topology intervention, so this result does not justify forcing a non-chain
topology or adding structural reward.

### TriviaQA development evidence for v6.2

The frozen TriviaQA development-128 run uses the same effective v6.2 source
boundaries: FlowSteer's progressive Canvas and imported `FORMAT_PROMPT`,
SkillFlow's unified `qa-retrieval` search/read action domain and public
continuation state, the heterogeneous Executor catalog, and explicit
current-revision `FINISH`. It completed with 128 valid evaluator receipts and
no final collection failure. Question-only local Qwen3.5-9B Direct scored EM
`35.15625%` / token F1 `40.81597%`; AgentGraph scored EM `51.5625%` / token F1
`60.10441%`, with 116 explicit `FINISH` submissions and 12 `max_rounds`
terminations. These are exposed development measurements, not held-out test
estimates, and no training or Skill injection occurred.

The older v3 run contains the same 128 task IDs, questions, accepted answers
and evaluator payloads, but it used a different Director condition, sampling
purpose, derived action seeds, Tool projection and handoff prompt. Its EM
`50.78125%` / F1 `63.36493%` and 125 explicit completions are therefore a
descriptive reference, not a single-variable paired comparison. V6.2 improves
EM by `0.78125` percentage points but reduces F1 by `3.26052` points and has
nine fewer explicit completions, so it is not reported as an overall v3
improvement.

TriviaQA v6.2 persisted 192 non-empty upstream artifacts; all 116 explicitly
finished trajectories have one non-empty semantic predecessor in the Output
inbox. The 186 actual QA Tool dispatch receipts (137 search, 49 read) all
completed at the backend. The dominant failures are instead retrieval
relevance, semantic answer/granularity, 6-turn ReAct exhaustion, and Director
action parsing (188 parse failures and 391 rejected turns). The final topology
distribution is `empty=4`, `single=3`, `serial_2=76`, `serial_3_plus=45`; no
natural fan-in or reciprocal topology was committed, so no topology reward or
forced graph pattern is introduced.

One necessary-adaptation candidate was tested and rejected. Some semantic
Agents execute before a later edit makes them the direct Format predecessor;
ordinary FlowSteer dirty-closure then reuses their old artifact even though
the project-specific `is_format_predecessor` flag changes the model-visible
protocol. A candidate invalidated those Agents and their descendants when the
execution role changed. On the 13 development tasks whose v6.2 terminal
artifact lacked the explicit handoff, the candidate held EM at `69.2308%` and
raised F1 from `71.4286%` to `73.6264%`, but reduced explicit `FINISH` from
13/13 to 12/13 and caused two task regressions. The effective source was
restored to v6.2; its fixed panel, trajectories and report remain under
`triviaqa_format_predecessor_v6_3_panel` as rejection evidence. This result
does not justify changing cache invalidation globally, adding an answer rule,
or letting Format re-solve the task.

### Coding source boundary

Codex is used only as a reference for a bounded model/tool/result loop,
verified patch application, command/test receipts, and iterative repair.  Its
session management, approvals, sandbox product policy, telemetry, user
interface, and workflow planning are not copied.  The Flow-Director remains
the sole graph editor and every structural change remains one FlowSteer Canvas
action.

## Runtime dataset registry source boundary

| Current module | Classification | Reference source | Reused boundary | Minimal adaptation | Scope |
| --- | --- | --- | --- | --- | --- |
| `config/datasets_runtime_v2.yaml`; `scripts/evaluate_completion_benchmark_round.py::{_runtime_dataset_registry_coordinates,_validate_runtime_dataset_registry}` | Necessary compatibility adaptation | Existing dataset-specific preparation catalogs and their published manifests; `task_dataset.py::{TASK_SCHEMA_VERSION,iter_task_records}`; completion runner `_select_tasks` frozen-selection comparison | Existing aligned JSONL files, task schema, split labels, dataset-specific preparation protocols, manifest provenance fields and frozen `selected_tasks` remain unchanged | A future condition may opt in only with both `data.registry_path` and `data.registry_dataset_key`. Before task selection, the runner fail-closes if its explicit train/validation/test paths, configured task schema, preparation manifest schema or available provenance disagree with the versioned registry. SWE-bench additionally requires the official evaluator dataset path/source to match the selected registry split. Missing optional manifest evidence is recorded in `skipped_checks` and is never inferred. Legacy conditions without the two coordinates retain their historical path and receipt semantics | All seven datasets; future conditions only |

## Coding completion freshness source boundary

| Current module | Classification | Reference source | Reused boundary | Minimal adaptation | Scope |
| --- | --- | --- | --- | --- | --- |
| `src/interactive/coding_execution.py::{CodingExecutionAdapter._completion_error,_completion_artifact}`; `config/evaluation_swebench_regular_dev_coding_agent_v2.yaml` | Necessary compatibility adaptation | SkillFlow `training/environment.py::{GenericTaskEnvironment.step,_generate_workspace_diff,_handle_run_tests}` and its one-action/observation iterative code-generation loop | Final output is derived from repository state rather than model prose; a public test pass/fail observation and a non-empty workspace diff are retained as execution evidence. SkillFlow does not require a passing test at terminal time, so this adapter does not invent that requirement | SkillFlow computes the current diff directly from monolithic episode state. The AgentGraph adapter instead receives ordered Tool receipts, so it requires `last changed edit < subsequent run_tests < subsequent changed diff < complete`; changed edits include exact replacement, mutating `str_replace_editor` commands, and successful Codex `apply_patch`. It submits only that fresh diff. The 9-turn/8-Tool-call budget bounds inspect/search/edit/test/revision/diff/complete without prescribing topology | SWE-bench Coding Agent |

## Unified Runtime failure, Tool admission, and Canvas search-space correction

| Current module | Classification | Upstream source | Reused boundary | Minimal adaptation and exclusion |
| --- | --- | --- | --- | --- |
| `tool_runtime.py::ToolCapability.argument_validation_error`; `react_execution.py::ToolReactExecutionAdapter.execute` | Necessary AgentGraph adaptation | SkillFlow `runtime/contracts.py::StructuredAction`, `runtime/bounded_agent.py::execute_turn`, and the task-published action schemas | One exact StructuredAction is parsed, admitted, executed, and followed by one public observation | Fixed-name Tool arguments are validated with Draft 2020-12 JSON Schema before dispatch. A schema failure is a public `schema_invalid` observation, consumes neither Tool budget nor backend call, and creates no ToolReceipt. Dynamic ALFWorld/WebShop action spaces retain their environment-supplied admissible-action path and are not forced through a static schema. |
| `react_execution.py::ToolReactExecutionAdapter.execute`; `coding_tools.py::create_swebench_repository_registration` | Necessary side-effect-aware adaptation plus schema completion | SkillFlow public Action--Observation history, episode-level repeated-Tool cache with edit/test exceptions, and Coding Tool argument domains | The next bounded turn receives the prior sampled action and observation; the backend remains authoritative for operation semantics | A global episode cache is unsafe for stateful AgentGraph Tools: the same `run_tests` request is valid after an intervening edit, and environment/repository state is revision-bound. The adapter therefore suppresses only an immediately repeated dispatched action in the same interaction state; Coding schemas reject unpublished properties before worktree mutation. |
| `qa_tool_adapter.py::QARetrievalReactExecutionAdapter`; `train_agentgraph_smoke.py::_qa_tool_runtime_settings` | Direct SkillFlow completion boundary plus explicit protocol adaptation | SkillFlow `training/environment.py::step` non-answer-Tool-before-answer rule and `benchmarks/retrieval.py` search/read environment | `required_tool_call` preserves the upstream historical behavior, including completion after a failed Tool receipt so an outage cannot erase a usable answer | `optional`, `required_tool_call`, and `required_evidence` are distinct configured completion-admission conditions. `required_evidence` requires a successful non-empty `read` receipt; scores from these conditions must not be merged. No answer key or evaluator state enters the Tool path. |
| `agent_runtime.py::{AgentFailureRecord,AgentRuntimeError,AgentRuntime.execute}`; `agent_workflow_env.py`; `rollout_collector.py` | Necessary free-AgentGraph adaptation | FlowSteer edit→execute→feedback and explicit FINISH; SkillFlow completed Action--Observation step materialization and nonterminal failure observations | Every accepted Canvas edit executes once; a failed execution remains nonterminal and returns public feedback | Because neither upstream has per-Agent cache reuse, AgentGraph computes the dirty closure from modified and missing artifacts, evicts stale outputs before execution, preserves only fully completed quotient-DAG blocks, records all public failure receipts, and keeps `pending_agent_ids` unresolved. A partial result has `final_answer=None`, is never written as a completed progressive execution, never reaches the evaluator, and cannot satisfy FINISH. |
| `environment_execution.py::EnvironmentExecutionError`; `agent_workflow_env.py::_environment_terminal_issue`; `evaluate_completion_benchmark_round.py::_graph_environment_terminal_receipt` | Direct SkillFlow terminal-observation reuse plus persistence adaptation | SkillFlow `RolloutEnvironmentSession`, `EnvironmentObservation`, and terminal completion validation; FlowSteer FINISH rejection | Real native action, transition, public observation, terminal flag, and evaluator replay are retained | An environment/provider failure carries the completed reset/transition/Tool/model prefix through AgentRuntime. WebShop/ALFWorld FINISH requires a measured terminal state-advancing Action--Observation edge. Their Stable Zero terminal artifact may have empty free text, but only when the typed terminal receipt and formal evaluator receipt are present; nonterminal prose cannot substitute for it. |
| `agent_workflow_env.py::{allowed_action_types,model_admissible_action_types,model_admissible_action_targets,step}`; `director.py::{_canvas_observation,action_schema_request}`; `rollout_collector.py::{SGLangReceiptDirectorClient._resolve_action_schema,_propose_hierarchical_action}`; `train_agentgraph_smoke.py`; `start_qwen35_director_server.sh` | FlowSteer action-set reuse plus necessary inference-wire adaptation | FlowSteer `workflow_env.py` BUILDING-state ADD/DELETE/MODIFY/FINISH handling, edit rejection/feedback, explicit terminal action, and the project's configured `agent_graph.actions`; SGLang 0.5.15 native singleton JSON-Schema constrained decoding and its `--constrained-json-disable-any-whitespace` runtime option | Canvas remains the authority for legal state transitions and returns rejection/execution feedback after every action | The configured scalar or `add_subgraph` profile remains the parser's full legal set. HotpotQA v19 derives a request-scoped action discriminator from the current model-admissible **action-type** subset and publishes exact target domains in the Canvas observation. Because the deployed xgrammar merges multi-action branch fields, v3 uses inference-only hierarchical constrained decoding: sample the legal action discriminator, then sample the complete action under that singleton live-domain schema; `modify_agent` additionally selects one mutable field and applies one atomic field patch, while relation edits are selected from Canvas-validated exact candidates. Current live-domain v5 `add_subgraph` first samples only bounded positional Agent roles, exposes that exact sampled receipt as the next chat turn, samples role-conditioned complete declarations, exposes those declarations, and finally samples the complete action with topology and Output still undecided. This follows FlowSteer's progressive committed-edit observation boundary and prevents a contract generated before the serialized `role_family` field from being semantically unconditioned. The final action is never assembled, spliced, or repaired in code. The versioned role-first strategy is mandatory only for current live-domain ADD receipts; historical decoding strategies retain their own receipt identity instead of being reinterpreted as v5. Every phase retains its native prompt/token/log-probability/seed/latency receipt. The Qwen3.5/SGLang generation receipt retains its sampled `<|endoftext|>` EOS text after a constrained JSON object; hierarchical phase parsing admits only that known transport-level suffix (and whitespace), retains it in the token receipt, and still rejects arbitrary trailing text. The task-owned SGLang process disables arbitrary JSON whitespace at the grammar backend so a sampled action cannot spend its token budget in an unbounded whitespace loop. No sampled final-action field is rewritten, and the unchanged parser/environment still validate every target and mutation. ADD is available only while capacity remains, target-bearing actions only when their exact domain is non-empty, DELETE only after an explicit node-unusable receipt plus same-responsibility artifact takeover, and FINISH only after the revision-local terminal gate passes. Once a verified terminal artifact exists, the HotpotQA action mask exposes only explicit FINISH; while a valid semantic lineage is preserved but another structural issue remains, MODIFY/SET_OUTPUT/SET_RELATION cannot invalidate its Agents, Output identity, or required edges. This inference factorization is disabled for training until the loss path represents every factorized decision. |
| `openai_gateway.py::build_agent_messages` | Direct FlowSteer Format separation | FlowSteer terminal `Format` operator and extraction prompt | Exact-answer datasets use a distinct Format sink; other Output Agents return the artifact required by their task contract | The ReAct Format predecessor wire uses `arguments.value`. Generic Output Agents no longer receive an unconditional short-answer/XML instruction, preventing HealthBench, Coding, and environment artifacts from being narrowed to a QA span. This does not add a dataset answer rule or let Format re-solve the task. |
| `environment_execution.py::{_alfworld_task_facts,_webshop_task_constraints,_public_state_feedback,_prompt_observation}`; `train_agentgraph_smoke.py::_environment_runtime_settings` | Direct SkillFlow public-state reuse with a neutral AgentGraph projection | SkillFlow `training/environment.py::{_parse_alfworld_task,_build_alfworld_task_brief,_build_alfworld_visible_memory,_build_webshop_visible_state_block}` and `GenericTaskEnvironment.max_obs_chars` | The task, current observation, admissible actions and completed public Action--Observation history are carried into the next bounded ReAct turn; model input may cap the observation while the receipt retains the full value | The AgentGraph projection keeps only task-visible target/destination/transform/count/coreference facts, WebShop instruction constraints, executed-action progress, format-repair state and an A-B-A-B no-progress signal. It does not import SkillFlow's action ranking, semantic action blocker, reward/`won`, hidden simulator state or benchmark-specific solution rule. `max_observation_chars` applies only to the model prompt; the complete observation remains in the environment/evaluator receipt. Scope: WebShop and ALFWorld. |

## HotpotQA unified_architecture_v1 verified answer-slot and recovery adaptation

This condition is selected only by
`agentgraph.director.hotpotqa-semantic-recovery.v22` and
`hotpotqa_verified_answer_slot_v1`.  Prompts v21 and earlier remain legacy,
explicitly versioned policies for old receipts, while the neutral v10 Director and every other
dataset retain their existing prompt/runtime behavior.  The semantic answer
slot, Verifier protocol, and recovery policy below are required by the user
design document.  They are not attributed to FlowSteer or SkillFlow.

| Current module | Classification | Upstream source | Reused boundary | Minimal adaptation and exclusion |
| --- | --- | --- | --- | --- |
| `director.py::{HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V22,AgentGraphOrchestrator.action_schema_request,AgentGraphOrchestrator._canvas_observation}`; `agent_workflow_env.py::{model_admissible_action_types,model_admissible_action_targets,finish_admissibility,step}`; `rollout_collector.py::{SGLangReceiptDirectorClient._resolve_action_schema,_propose_hierarchical_action}` | Direct FlowSteer Canvas reuse plus necessary request-wire adaptation | FlowSteer `workflow_env.py::{InteractiveWorkflowEnv.step,_step_internal,_check_finish_constraints}`, `workflow_graph.py::{add_operator,add_parallel}`, `workflow_builder.py::{TurnRecord,Trajectory,create_action_mask}`, progressive execution feedback, and explicit `FINISH`; deployed SGLang singleton constrained decoding | State-conditioned Canvas legality, one accepted edit/subgraph followed by one execution, public feedback, revision-local terminal validation, and trajectory/action-token records retain the FlowSteer boundary | v22 keeps the compact task-specific Director instruction and projects both current action types and parameter domains into an inference-only hierarchical JSON-Schema factorization. Required Agent semantic role, execution mode, Tool capability and catalog model ID are constrained before Canvas admission; relation actions use exact non-self/non-no-op candidates, including the unique terminal Formatter-predecessor constraint, and an empty candidate set removes the action from the mask. A staged `add_subgraph` decode is necessary because standard JSON Schema has no cross-field `$data` reference for “relation endpoint belongs to Agent IDs sampled in this same object”; current v5 samples count/roles before free contracts and carries each exact committed receipt into the next model-visible turn, while the final complete action remains independently sampled, parser/Canvas-authoritative, and bound to the exact declarations. The live target domain also carries the ordered existing-Agent role map, current Output identity, and `admitted_new_role_families`: a healthy Reasoner, Verifier, or Formatter is repaired rather than duplicated, while Evidence Retriever and Repair branches remain available and a same-role replacement is admitted only after typed node unusability. Single directed relations are encoded canonically as actual sender→receiver, semantic dataflow is admitted as Retriever/Repair→Reasoner→Verifier→Formatter, and reciprocal Reasoner–Verifier feedback remains available. Agent count, auxiliary roles, contracts, models, Tools, fan-in/fan-out, and reciprocal topology remain sampled choices. FlowSteer's training `action_mask` is not re-described as this inference protocol. |
| `react_execution.py::ToolReactExecutionAdapter`; `qa_tool_adapter.py::QARetrievalReactExecutionAdapter`; `openai_gateway.py::build_agent_messages`; `agent_runtime.py::{execute,_semantic_input_deferred_components}` | Direct SkillFlow bounded-execution reuse plus typed AgentGraph adaptation | SkillFlow `runtime/bounded_agent.py::BoundedAgent.execute_turn`, `runtime/contracts.py::StructuredAction`, `runtime/openai_provider.py` structured-response boundary, canonical public Action–Observation continuation, QA `search/read` Tool environment, and Supervisor/provider separation | ReAct remains the bounded `Thought -> Action(tool) -> Observation -> Thought -> Final` execution schedule and never becomes an Agent role. Tool action, public observation, subsequent model turn, bounded completion, and receipt boundaries follow SkillFlow | `required_evidence` binds the existing `qa-retrieval` resource to the semantic Reasoner; an optional direct Retriever predecessor may augment but cannot own the answer. HotpotQA uses the first read Observation to formulate a second missing-hop `search -> read` pair before completion. If provenance validation rejects a completion and the bounded Tool budget can still complete a pair, the same ReAct call reopens `search -> read` before retrying completion. Progressive execution defers a Verifier until it has exactly one routed Reasoner predecessor and defers a Formatter until it is the selected Output with exactly one routed Verifier predecessor; deferred nodes and descendants remain on the Canvas as unresolved work and are never invoked as independent solvers. A later relation/Output edit schedules them using preserved upstream artifacts. The AgentGraph adapter carries execution/deferred and Tool receipts into later Canvas feedback; it does not expose Ground Truth, evaluator state, or a benchmark answer to Director, Supervisor, or Executor. |
| `qa_tool_adapter.py`; `openai_gateway.py`; `agent_workflow_env.py::{_reasoner_candidate,_possessor_surface_issue,_contract_obligation_issue}`; `task_dataset.py::{hotpotqa_question_scope,hotpotqa_answer_type_constraint,hotpotqa_answer_cardinality_constraint}` | User-design-required HotpotQA semantic adaptation | Existing answer-free HotpotQA question rendering and the SkillFlow QA evidence receipts above; no upstream answer-slot implementation is claimed | Retrieved passages and their public evidence spans remain the only factual input to semantic completion | The Reasoner owns `candidate_answer`. It copies the unchanged question scope, represents evidence as subject–relation–object/attribute propositions with qualifiers, and binds the requested answer slot by `(proposition_index, answer_field)` to the selected proposition argument. Answer type/cardinality are derived from question text only. For a who-question licensed by a possessive construction, the candidate excludes the possessed attribute but retains the complete evidence-aligned possessor mention, including a title, honorific, or name suffix. New or modified contracts are transactionally rejected when they copy a concrete context/public-observation candidate, value, date, alias, or evidence span outside the question; this admission uses no Ground Truth or evaluator state and never rewrites a sampled action. The pointer may select any in-range proposition; evidence order is not an answer prior. No sample ID, query, passage, entity, candidate, Ground Truth, or evaluator result is hard-coded. |
| `openai_gateway.py::build_agent_messages`; `agent_workflow_env.py::{model_admissible_action_targets,_semantic_edit_issue_for,_semantic_protocol_issue}`; `director.py::_live_role_agent_schema`; FlowSteer `scripts/prompts/prompt.py::FORMAT_PROMPT` | Direct FlowSteer Format reuse plus user-design-required Verifier adaptation | FlowSteer's terminal Format Operator serializes an already computed solution and does not solve the task | Reasoner determines the semantic answer; Verifier is a separate evidence/contract gate; Formatter is the terminal extraction sink | The Verifier consumes only the Reasoner artifact and checks explicit evidence, entity–attribute/alias binding, answer-slot type and cardinality, complete multi-hop support, minimal answer surface, and unchanged question scope. It must preserve the identical candidate rather than select a replacement. Formatter receives only one supported Verifier artifact, omits the original question, and copies that candidate character-for-character into one `<answer>...</answer>` wrapper; it cannot reason, verify, canonicalize, or reselect. Its live ADD domain admits only the neutral formatting contract, the Canvas rejects any other Formatter contract, and MODIFY cannot mutate that contract; therefore the Director cannot place a sample-specific answer or reasoning instruction in the Formatter node. Semantic role and execution-mode admission rejects `ReAct` as a role and prevents Retriever-to-Verifier bypass. These checks are runtime protocol fields, not hidden chain-of-thought supervision. |
| `agent_workflow_env.py::{recovery_state,_mandatory_repair_agent_ids,_semantic_repair_attribution,_delete_admission_issue,_preservation_admission_issue,_execution_error_feedback,finish_admissibility}`; `agent_runtime.py::_public_failure_metadata`; `director.py::{_model_catalog,_canvas_observation}` | FlowSteer/SkillFlow failure-feedback reuse plus user-design-required recovery adaptation | FlowSteer returns execution/rejection/terminal diagnostics to the next Canvas turn; SkillFlow retains nonterminal provider/Tool failure as a public observation and keeps provider identity separate from model identity | Existing successful artifacts, directed communication, Output identity, failure receipts, and explicit terminal semantics remain visible and revision-bound | HotpotQA applies `preserve -> diagnose -> repair -> augment`: preserve valid evidence, semantic answer, lineage, and relations; attribute the measured failure to its Agent/relation/provider; repair the existing responsible Agent before any other edit while it remains usable; augment only after repair or typed node unusability. The same mandatory-repair predicate is enforced in the model action mask and the authoritative transactional Canvas admission, so callers cannot bypass it. A terminal semantic-artifact fault is attributed to Reasoner or Verifier and re-executes only that responsibility while preserving upstream artifacts. Disconnection, provider failure, ReAct exhaustion, Tool failure, timeout, and contract failure never by themselves make a node deletable. Deletion is admissible only after explicit typed `node_unusable=true` and completed same-role/same-artifact takeover. A verified semantic lineage is protected from MODIFY/SET_OUTPUT/edge mutation, and a verified terminal revision exposes only explicit FINISH. |

Prompt v22 is the current policy delta over the versioned v21 policy. It keeps
the same FlowSteer Canvas and live-domain decoding boundary. The Runtime now
enforces the pre-execution contract obligation transactionally instead of
relying only on prompt text. When typed Agent failures exist, MODIFY targets are
the actual `AgentFailureRecord` owners rather than blocked downstream nodes.
ReAct/schema failures retain the SkillFlow public Action--Observation trace and
Tool receipts, expose the generic public `repair_instruction`, and first target
the responsible Agent's `contract` or optional `completion_condition`; the
instruction may define only an output-schema obligation and may not carry task
candidate/value/evidence content. v22 additionally makes repair-first action
masking and the admitted ADD role domain authoritative: a usable failed Agent
must be modified before augmentation, healthy semantic responsibilities cannot
be duplicated, and SET_RELATION cannot violate the unique Formatter input
contract. Prompt v21 and older receipts retain their
versioned policy identity.

The corresponding Runtime continuation is phase-scoped: only the same Agent
and failed `single`, `draft`, or `revision` phase receives the retained public
history after a Canvas repair.  Evidence/provenance rejection may reopen an
admissible `search -> read` pair; answer-slot or semantic-schema rejection keeps
successful reads and admits only corrected completion.  The Director sees a
compact count/error-code diagnosis, not retrieved passage text or hidden
reasoning.  Recovery does not reduce the ordinary one-to-three-Agent ADD search
space after the measured repair boundary clears: connected retrieval, repair,
fan-in, reciprocal Reasoner–Verifier communication, and typed replacement
functional units remain available. Deletion is still separately gated on a
typed unusable-node receipt and completed replacement artifact takeover.

The current configuration is evaluation-only: GRPO, backward, optimizer
updates, LoRA publication, MACE/Bayesian exploration, and Skill retrieval are
all disabled.  Any measured development score is therefore an inference
architecture result, not a training or ACTIVE-Skill result.
