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
| Progressive execution cache | `src/interactive/workflow_env.py` (`execute_each_step`, `last_execution_result`) | Adapted as a revision-local result in `agent_workflow_env.py`. Every accepted Agent or functional-subgraph edit executes once and returns observation feedback; a later `finish` may reuse the same-revision result. A no-op edit is rejected. `rollout_collector.py` marks reuse and does not serialize old Agent calls as new executions. |
| AgentGraph search-space bounds | Project design note sections 3 and 4 plus `config/*agentgraph*.yaml` | The declared `max_agents` is consumed by the Canvas; the two-Agent reciprocal-block limit, unique output/reachability flags, seeded Executor selection, legacy six actions, optional FlowSteer-style `add_subgraph` transaction, and progressive execution mode are validated against runtime semantics rather than left as descriptive YAML. |

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
| Exact generation seed | `runtime/openai_provider.py` and `rollout/types.py::derive_generation_seed` | `openai_gateway.py` sends the fixed run seed at the provider edge. The native exact-receipt Director sends the deployed SGLang 0.5.15 equivalent, `sampling_seed`, and persists it per turn. |
| Existing adapter inference readiness | `training/external_sglang.py::publish_external_adapter` and `runtime/sglang_gateway.py` | `policy_sync.py::ensure_loaded_adapter` reuses only model-list, load, verification, and canary for evaluation. It neither trains nor publishes a new policy. |
| Multi-hop one-call contract | `training/task_prompts.py::MULTI_HOP_QA` | The paired local Direct path uses the upstream brief multi-hop contract through the existing Agent gateway; it bypasses Director, Canvas, and AgentGraph. |

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

## HotpotQA task-scoped dense retrieval condition

| Local boundary | Reused source | Adaptation and status |
| --- | --- | --- |
| `tool_runtime.py`, `react_execution.py`, Agent execution fields and receipt persistence | FlowSteer commits `fea7b2b` and `8ebba85`; SkillFlow `runtime/tools.py`, `runtime/bounded_agent.py`, and `training/environment.py` | Directly ports the structured action, immutable Tool registry, bounded per-Agent ReAct continuation, public Observation, and Tool receipt chain. ReAct remains an execution mode and does not define a role or workflow topology. |
| `hotpotqa_embedding_index.py` | SkillFlow `benchmarks/retrieval.py::{RetrievalIndex.open,search,read}` and `training/environment.py` BGE normalized-embedding path | Keeps immutable `open/search/read` and task-scoped read semantics; the minimal compatibility change replaces FTS5 with `BAAI/bge-base-en-v1.5` normalized embeddings and deterministic cosine top-k over each task's ten original public context documents. |
| `hotpotqa_embedding_tool.py` | SkillFlow QA retrieval environment; current FlowSteer `react_execution.py::_state_conditioned_action_domain` and `qa_tool_adapter.py` HotpotQA multi-hop state | Binds one task ID per registry, exposes only `search(query,k)` and `read(doc_id)`, persists query/rank/similarity/doc ID/Observation receipts, and admits the bounded `search → read → search → read → complete` continuation with duplicate-query and latest-unread-candidate gates. This is an action-domain mask inside any Tool-enabled Agent, not a fixed Agent role or AgentGraph template. |
| `agent_workflow_env.py::required_evidence_tool_id` | Existing unified architecture FINISH admission derived from FlowSteer terminal constraints | The topology-neutral subset is reused: FINISH requires successful search/read receipts on the routed Output ancestry. It requires no Reasoner, Verifier, Formatter, fixed chain, or role adjacency. |
| `select_hotpotqa_embedding_profile.py` | Project evaluation-isolation requirement | Uses only the original question from 32 disjoint train/architecture-development tasks to freeze top-k and Tool budget. Supporting-title labels are consumed only for aggregate development recall selection and are never written to the corpus, index, Tool observation, or validation runtime. Question-only recall ties at `k=4` and `k=5`, so the declared smallest-`k` tie-break freezes `k=4`. |

The corpus builder reads only source parquet columns `id` and `context` and
stores only `passage_id`, `document_id`, `title`, and `text`. Reference answers,
supporting-fact labels, evaluator receipts, and evaluator-private metadata are
absent. The evaluation condition supplies only the original question to the
Director and Agents; passage access occurs dynamically inside an Agent's
execution. Web Search is not registered. This condition is inference-only:
Skills, GRPO, LoRA, backward, optimizer updates, MACE, and Bayesian updates are
disabled.

`hotpotqa_embedding_retrieval_v4` is the first formal condition whose runtime,
development profile, deterministic rebuild smoke, and Tool smoke all share the
same question-only boundary.  Earlier v2/v3 conditions are retained only as
diagnostic evidence and are not the current next-run profile.
