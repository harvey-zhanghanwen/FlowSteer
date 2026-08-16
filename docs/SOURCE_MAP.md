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
| Forward/backward LoRA profile | `training/gflownet_trainer.py::setup` | Configuration only: theta rank 64 and phi rank 16. No optimizer is connected. |
| Three-role GPU topology | `device`, `supervisor_gpu_id`, and `extra_device` in SkillFlow | Mapped to physical GPUs 3, 4, and 5 in `training_agent_graph.yaml`. |
| Split micro-batch backward | `GFlowNetTrainer._batched_logprob_backward` | Represented only by inactive OOM/micro-batch configuration. No backward code is claimed in this phase. |
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

SkillFlow's TTB objective, backward policy training, partition head, skill
evolution loop, benchmark environment, and local multi-executor launcher are
not copied into this phase.  The project continues to reserve terminal-only
GRPO as specified by the design note, but GRPO is disabled.

## Project-specific algorithm modules

The following are additions required by the attached design note rather than
claims of upstream FlowSteer or SkillFlow functionality:

- free-text Agent contracts, per-node model selection, and finite two-stage
  bidirectional Agent execution;
- the MACE feature/bandit baseline and joint Bayesian posterior primitives;
- same-prefix paired-probe and EVSI primitives;
- version-bound Skill evidence schemas, lifecycle, and persistence records.

Those modules are isolated from the runtime reward path.  In the checked-in
architecture configuration, exploration, Skills, GRPO, optimizer work, and
all GPU training are disabled.

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
