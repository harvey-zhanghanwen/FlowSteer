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
| `run_joint_qa_mace_skill.py` | Thin experiment adapter | Existing paired-probe runner, `JointQAPosteriorScheduler`, `SkillEvidencePipeline`, and SkillFlow retrieval observation adapter | Balanced cold start, UCB prefilter, EVSI selection, randomized paired-arm order, independent confirmation, deterministic gate, and delayed next-epoch activation | Old Step-2 paths and a fixed fan-in prior were incompatible with the new `add_subgraph` action profile.  The runner uses the disjoint `joint_qa_v2` train/Skill-confirmation partitions and two bounded, rejectable failure-derived priors without mutating Canvas. |
| `skills/pipeline.py::retrieval_snapshot` | SkillFlow lifecycle adaptation | SkillFlow frozen ACTIVE Skill library per rollout/training epoch | Every Director turn in one batch reads the same immutable Skill records | The local pipeline formerly reread its JSON store on every turn.  An optional typed snapshot freezes visibility while leaving the existing store and retriever unchanged. |
| `train_agentgraph_smoke.py` Skill-on joint micro boundary | Necessary SkillFlow training adaptation | Existing SkillFlow-style exact group admission, one-pass LoRA learner, frozen schedule/cursor, adapter publication and updated-policy canary | Terminal F1-only GRPO, one real `optimizer.step`, exact policy/version receipt, pause/drain/load/canary route switch | Skill-on is admitted only for a frozen store containing version-compatible ACTIVE Skills covering both datasets.  The manifest records posterior/library versions and first-turn Skill visibility; forced probes remain disabled.  Joint schedule resolution additionally includes the independent Skill-confirmation path in its held-out union. |
| `materialize_joint_qa_progressive_skill_training.py`, `freeze_joint_qa_training_schedule.py` | Necessary experiment materialization adaptation | Existing `freeze_joint_qa_training_schedule`, write-once cursor, `SkillStore`, YAML loader, and `validate_smoke_bounds` | Evidence gate is checked before a fixed train-only schedule and resolved Skill-on config can exist | The adapter selects one predeclared unused train position per dataset, binds exact ACTIVE Skill IDs/library/posterior/policy versions, and writes no rollout or model state.  The freeze CLI now forwards the optional Skill-confirmation path into the existing held-out-union check. |
| Progressive Step-0 and Skill-on Step-1 YAML files | Necessary configuration adaptation | Existing HotpotQA/TriviaQA evaluators, ten-arm v6 Executor catalog, formal zero-update LoRA, and joint-QA smoke trainer | Fixed development sample, local Qwen3.5-9B Director, `add_subgraph` search space, two train tasks x eight rollouts, one update, and two canaries | The training YAML remains a fail-closed template until evidence-gated ACTIVE Skill IDs and their exact library/posterior versions are materialized.  No placeholder is reported as an executed run. |

The SkillStore is frozen before natural GRPO collection.  A successful LoRA
update creates a new policy version, after which the prior ACTIVE Skills are
version-incompatible and must be suspended or independently revalidated before
another training epoch.  Development and final test tasks are never used for
posterior fitting, EVSI, Skill confirmation, or optimizer data.

The first progressive Step-0 stream showed that component transactions removed
the singleton collapse but still produced only serial depth-2/3 graphs.  The
fresh Skill epoch therefore replaces the obsolete transaction-construction
prior with a rejectable dependency-aligned topology prior: parallel branches
are suggested only for independent evidence, finite reciprocal revision only
for a draft/critique dependency, and serial dependencies otherwise.  This is a
paired Skill candidate, not a base Director template, topology quota, reward,
or direct Canvas mutation; it can become ACTIVE only through the unchanged
independent evidence gate.

The progressive Skill-on micro-training boundary reuses the existing
`SkillLifecycleManager.audit` transition after a successful LoRA policy
publication.  The runner changes only the policy coordinate, persists every
affected `ACTIVE -> SUSPENDED` transition in the configured `SkillStore`, and
records the lifecycle receipt before post-update canaries.  This is the
necessary runner integration for the design document's policy-drift rule; it
does not add a second lifecycle implementation.  The same boundary now derives
the manifest's post-update canary bound from `policy_sync.post_update_canary_count`
instead of reporting a stale constant.
