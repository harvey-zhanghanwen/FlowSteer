# AIME 2026 initial-adaptation source map

This file records the executable sources used by the AIME 2026 initial
adaptation.  The attached papers and the project design document are design
references; they are not executable instructions.  Source priority for this
adaptation is:

1. `FlowSteer_MACE_Bayesian_Skill_Design.md` for the project contract;
2. SkillFlow / downstream SkillEval production code for the AIME 2026 data and
   evaluator contract;
3. FlowSteer for the progressive Canvas, execution-feedback, and trajectory
   boundaries; and
4. a minimal project adapter only where those interfaces do not meet directly.

## Frozen source identities

| Source | Fixed identity | AIME 2026 boundary used here |
| --- | --- | --- |
| Project design document | `FlowSteer_MACE_Bayesian_Skill_Design.md` supplied with this task | Free AgentGraph `G=(V,E,o)`, model-labelled Agents with free-text contracts, graph relations, one Canvas action per turn, explicit `FINISH`, terminal evaluation, and complete trajectory receipts. |
| Public SkillFlow source | revision `74be52bb6bd9f0e9e68dacb72636b75649197983` | Paper-level AIME 2026 benchmark/Accuracy context and general runtime conventions. Its checked-in `data/prepare_v3.py` is explicitly **not** the AIME 2026 loader/evaluator source used below. |
| Downstream SkillEval production source | local production tree `/home/test/SKILLEV/skillflow-bayesian-improve-deploy` | Exact AIME 2026 acquisition plan, row converter, public/private task split, empty Tool catalog, and strict integer terminal scorer. |
| MathArena AIME 2026 dataset | `MathArena/aime_2026` revision `d2de22f3c656b4f56cf8981212186377d1e23bc3` | One Parquet shard with the complete 30-row population and exact fields `problem_idx`, `problem`, and `answer`. |
| Upstream FlowSteer source | revision `1c9f2ab` | Progressive Canvas `edit -> execute -> feedback`, graph execution boundary, terminal evaluator timing, and trajectory concepts. This source contains AIME 2025 rather than AIME 2026. |

The fixed identities above are provenance declarations.  This task does not
perform an artifact-hash or repository-integrity audit.

## SkillFlow / SkillEval production mapping

| Production source | Reused semantic contract | Local target |
| --- | --- | --- |
| `packages/private-evaluation/src/skillev_private/benchmarks/non_process_preparation.py::LockedNonProcessSourcePlan` for `Benchmark.AIME_2026` | Dataset source is the `train` split of `aime-2026/data/train-00000-of-00001.parquet`; the source format is Parquet. | `config/datasets_aime2026_official_v1.yaml` and the official AIME 2026 runtime registry select that one explicitly named MathArena shard and record its fixed revision. |
| `packages/private-evaluation/src/skillev_private/benchmarks/production_catalog.py::PyArrowParquetRowReader.read_rows` | Read the named Parquet shard with `pyarrow`, preserve row order and scalar types, and reject empty/non-object records. | `scripts/prepare_aime2026_dataset.py::_read_official_parquet_rows` is a thin port of this reader; the official-only catalog writes no historical training or development records. |
| `packages/private-evaluation/src/skillev_private/benchmarks/converters.py::convert_matharena_aime_2026_row` | Exact row fields are `{answer, problem, problem_idx}`; indices are `1..30`; answers are integers in `0..999`; task ID is `aime-2026/{problem_idx:02d}`; task family is `aime-2026/integer-answer`; public context contains `answer_format`, `problem_index`, and `source_format`. | The AIME loader preserves those identities and exposes only the problem and legal public metadata to model-facing code. |
| `src/skillev/benchmarks/static.py::BenchmarkPublicItem.to_rollout_task` | Static AIME tasks have `available_tools=()`; the public rollout projection contains the query and public metadata, not the target. | AIME initial configuration disables QA retrieval, Web search, computation tools, and Skill retrieval. |
| `packages/private-evaluation/src/skillev_private/benchmarks/static.py::PrivateStaticTarget.score` | `StaticScoringRule.INTEGER` applies `str(int(prediction.strip()))` and exact comparison with equivalently canonicalized accepted answers. The primary metric is Accuracy. | `src/interactive/aime2026_adapter.py` ports this canonicalization and exact comparison. |
| `packages/private-evaluation/src/skillev_private/benchmarks/static.py::PrivateStaticBenchmarkEvaluator.evaluate` | The admitted terminal submission has exactly `{"answer": str}`; the target remains evaluator-only. | `src/interactive/task_evaluator.py` exposes the result as canonicalized integer `accuracy` and records parsing status. |
| `packages/private-evaluation/src/skillev_private/benchmarks/production_catalog.py` AIME route | AIME 2026 is a static benchmark workload, not a retrieval or interactive-environment workload. | Direct and AgentGraph evaluation use the same task population, extraction boundary, canonicalization, and evaluator. |

### Why public `prepare_v3.py` is not reused

The public SkillFlow `data/prepare_v3.py` recursively combines a general
`aime/**/*.parquet` pool, deduplicates by a question prefix, shuffles records,
and expands short pools before constructing a generic `500 train + 128 eval`
view.  It neither represents the fixed 30-problem MathArena AIME 2026
population nor preserves the downstream production public/private boundary.
Using it would conflict with the required official population, task identity,
and no-duplication constraints.  This incompatibility is why the downstream
production converter and scorer, rather than a newly invented equivalent
loader, are the source of truth.

## FlowSteer mapping

The concrete upstream reference points are
`src/interactive/workflow_env.py::InteractiveWorkflowEnv`,
`src/interactive/workflow_graph.py::WorkflowGraph`, and
`src/interactive/workflow_builder.py::{TurnRecord,Trajectory,InteractiveWorkflowBuilder}`.
The project core predates this AIME change and adapts those progressive
execution/trajectory boundaries to the MD's free AgentGraph; the AIME work
does not claim a direct import of those upstream classes.

| FlowSteer boundary | Status in AIME 2026 initial adaptation |
| --- | --- |
| Progressive Canvas with execution after an accepted edit and feedback before the next Director action | FlowSteer-derived boundary retained by the existing project `AgentWorkflowEnv` / Director loop. The AIME adapter does not copy a separate FlowSteer environment. |
| Canvas action, graph revision, Agent execution, and terminal trajectory records | FlowSteer-derived boundary retained by the existing project AgentGraph and rollout records. |
| Terminal evaluator timing after a legal terminal action | FlowSteer-derived boundary plus the project design document's stricter rule that only explicit legal `FINISH` admits an AIME answer to formal evaluation. |
| AIME 2025 / MATH data adapters | Not used as the AIME 2026 data source. |
| Fixed `Plan`, `Programmer`, `Verify`, or `Format` mathematical workflow templates | Not migrated. They would introduce an orchestration prior prohibited for the initial condition. |
| Fixed Solver/Verifier chains, parallel solvers, debate, voting, self-consistency, or mandatory Python use | Not migrated. Topology, contracts, model routing, and termination remain Director decisions within the legal search space. |
| `answer_extractor.py`, `eval_only.py`, `train_interactive.py`, and `scripts/evaluator.py` fallback/tolerance/symbolic or last-number scoring paths | Not used for AIME 2026 formal scoring. They are not equivalent to the SkillEval private integer scorer. |
| Historical-candidate or max-round answer fallback | Not admitted to formal AIME evaluation. A trajectory without explicit `FINISH` has no formal final answer. |

## Project design document mapping

The unified core remains `G=(V,E,o)`:

- each `V` entry retains `agent_id + model_id + free-text contract`;
- `E` retains independent, directed, and bounded bidirectional communication;
- `o` remains the unique Output Agent;
- no mathematical role enum is added;
- the Director retains the unified atomic action space `ADD_AGENT`,
  `MODIFY_AGENT`, `DELETE_AGENT`, `SET_RELATION`, `SET_OUTPUT`, and `FINISH`;
- every accepted Canvas edit executes the current graph and returns real
  execution feedback before the next Director turn;
- recovery follows `PRESERVE -> DIAGNOSE -> REPAIR -> AUGMENT` as a recovery
  order, not as a mathematical workflow; and
- only a legal explicit `FINISH` creates an evaluator-eligible terminal
  submission. `max_rounds` is a terminal failure, not an implicit finish.

## Project-specific thin adapters

| Local boundary | Classification | Necessary adaptation |
| --- | --- | --- |
| `scripts/prepare_aime2026_dataset.py` | Project-specific thin adapter over downstream SkillEval production schema | Port the PyArrow reader and convert the exact fixed Parquet rows into the existing `TaskRecord` schema while preserving source order, task identity, problem text byte-for-byte, public metadata, and private target separation. No problem is copied or rewritten. |
| `src/interactive/aime2026_adapter.py` | Project-specific terminal-envelope compatibility plus downstream SkillEval scorer port | The active Direct and AgentGraph lanes both submit the same model output to this boundary. A bare integer follows SkillEval's native submission rule; one optional existing FlowSteer `<answer>...</answer>` envelope is unwrapped identically in either lane. Multiple/malformed boundaries fail closed. This layer never solves or repairs an answer. |
| `src/interactive/task_evaluator.py` | Necessary evaluator interface adaptation | Return the SkillEval integer score through the unified `EvaluationOutcome` receipt with primary metric `accuracy` and explicit parsing diagnostics. Ground truth is accessed only here. |
| AIME evaluation configuration and completion runner | Existing unified-runtime wiring | Select the fixed 30 tasks, render the same public problem/answer-format metadata for Direct and AgentGraph, keep the Director prompt neutral, disable task-specific Tools/Skills/training, compare both lanes under the same extraction/canonicalization/evaluator, and persist paired and trajectory receipts. |
| Wrong-demo materialization | Existing trajectory-analysis boundary | Locate the first recorded failure in the actual Canvas/action/Agent/runtime/output/evaluator receipts. It does not synthesize a missing trace or add a task-specific workflow rule. |

## Model-visible and evaluator-only data

Model-visible AIME input is limited to:

- `problem` (project `question` alias); and
- legal public metadata such as benchmark identity, problem index, answer format,
  source format, and split.

Evaluator-only data includes:

- `ground_truth` / accepted answer; and
- canonicalized expected integer.

The evaluator target must not enter the Director prompt, Agent contract, Agent
input, Canvas feedback, recovery context, Tool observation, or model-visible
trajectory.  Direct and AgentGraph paths differ only in orchestration; they use
the same target-blind problem input and the same evaluator.

## Explicitly excluded from this initial adaptation

- HotpotQA/TriviaQA passages, supporting facts, entity linking, query
  normalization, evidence stores, retrieval databases, and Web search;
- Python, calculator, symbolic computation, sandbox execution, or any answer
  lookup path in the initial AIME condition (the production static task exposes
  no Tools);
- historical AIME solution databases or official-solution lookup;
- hard-coded mathematical workflows, fixed Agent counts, topology priors,
  role-to-model routing, or few-shot workflow examples;
- GRPO, backward, optimizer updates, LoRA publication, MACE exploration,
  Bayesian posterior/EVSI, Skill retrieval, Skill evolution, or artificial
  orchestration experience; and
- structural rewards or output-format topology rewards.

Those exclusions define the initial Stable Zero condition.  They are not
claims that the unified repository lacks optional implementations for later,
separately authorized experiments.

## Stable Zero runtime correction source boundary

The AIME canary confirmed that FlowSteer's accepted-edit execution boundary
requires later relation edits to invalidate and re-execute the affected
downstream closure. The existing project implementation already provides this
through `AgentGraph.dirty_closure()` and the progressive-output invalidation
path. A local predecessor-identity guard had been applied more broadly than its
verified semantic-lineage purpose and made free AgentGraph relation editing
unreachable after successful `ADD_AGENT` execution. The guard is now scoped to
semantic-lineage protocols; no AIME-specific graph operation or mathematical
workflow was added.

## Qwen3.5-9B Direct comparator source boundary

The checked public SkillFlow source and the downstream SkillEval production
tree do not provide an independent single-model Direct runner. SkillEval's
`InitialBaselineInferenceState` still executes
`RolloutEngine + BoundedAgent + StructuredJsonActionCodec` with explicit
reasoning/action phases and terminal actions. That is a pre-training bounded
policy episode, not the requested Qwen3.5-9B Direct comparator, and it is not
relabelled here.

The fallback therefore directly reuses FlowSteer revision `1c9f2ab`:

- `scripts/operators.py::AnswerGenerate` supplies the single-model,
  single-call execution boundary;
- `scripts/prompts/prompt.py::ANSWER_GENERATION_PROMPT` supplies the exact
  content-level step-by-step and XML response protocol; and
- `scripts/formatter.py::XmlFormatter.from_model(...).prepare_prompt(...)`
  appends FlowSteer's actual Pydantic-derived XML field contract; and
- `scripts/operator_analysis.py::AnswerGenerateOp` defines the `thought` and
  `answer` fields.

The project adaptation is limited to formatting the frozen public AIME problem
and public `answer_format` metadata into that upstream prompt, then submitting
the model's existing `<answer>` field to the same SkillEval-derived integer
extractor/canonicalizer used by AgentGraph. The Director prompt, action space,
AgentGraph search space, Tool catalog, and saved AgentGraph trajectories are
unchanged. The earlier bare-integer one-call outputs are retained only as a
pre-source-alignment diagnostic and are not reported as the final Direct
baseline.

## AIME runtime/artifact protocol v2 source map

The v2 repair changes the unified execution boundary, not the AIME search
space. Agent declarations remain `agent_id + model_id + free-text contract`,
relations keep the existing two-bit encoding, and the Director still chooses
the graph, Output pointer, and explicit `FINISH` action.

| v2 boundary | Upstream source | Reuse / adaptation |
|---|---|---|
| `SET_OUTPUT` and `FINISH` reuse | FlowSteer `src/interactive/workflow_env.py::_step_internal` reuses the last execution result at FINISH; `workflow_builder.py::create_aflow_executor_wrapper` caches a node by its actual input identity | **FlowSteer reused / project thin adaptation:** Output selection is pointer-only. A fresh node artifact is not resampled; FINISH consumes the current revision's Output artifact and rejects a missing/stale artifact. |
| Artifact identity and invalidation | Existing unified `AgentRuntime._response_output_metadata`, `UpstreamMessage`, `AgentGraph.dirty_closure`, and FlowSteer progressive execution cache | **Existing core reused:** `artifact_version=request_id` remains backward compatible and is also exposed as `artifact_id`. The receipt now explicitly projects agent, revision, model, contract, tool configuration, upstream dependencies, and raw output. Model/contract/tool edits and actual upstream relation/input changes retain the existing downstream dirty-closure behavior. |
| Fan-in provenance | Existing one-envelope-per-edge `UpstreamMessage` routing | **Existing core reused / project thin adaptation:** every source keeps `source_agent`, `artifact_id`, and `raw_output`; the model is told that each item is an unverified work product. A task adapter may deterministically extract public candidates and report only whether distinct candidates conflict. |
| Candidate anchoring during recovery | FlowSteer transactional Canvas admission plus the design document's target-isolation rule | **Project thin adaptation:** when a task supplies a deterministic public artifact extractor, Canvas rejects ADD/MODIFY contracts that copy a currently unverified artifact candidate into a candidate-bearing obligation. The check is target-blind, preserves the artifact for routed execution-time checking, and adds no Agent role or topology prior. |
| Provider retry | SkillFlow `src/executor/openai_request_policy.py::OpenAIRequestPolicy.create_chat_completion` retries transient connection/timeout, 408/409/429, and server failures without changing the request; the project already used the same-request retry boundary in `OpenAICompatibleGateway.generate` | **SkillFlow strategy reused / receipt adaptation:** the existing finite retry/backoff policy is retained, model/provider/payload stay fixed, and every attempt is now persisted in `retry_receipts`. |
| Partial failure state | Existing quotient-DAG scheduler preserves completed sibling blocks before fail-fast cancellation and records blocked descendants | **Existing core reused / project thin adaptation:** the same partial outputs are retained and an explicit per-node `SUCCESS`, `FAILURE`, or `BLOCKED_BY_UPSTREAM` projection is added. |
| Empty provider completion | SkillFlow bounded execution treats a provider/executor failure as a failed attempt rather than a semantic artifact; FlowSteer's executor cache is populated only by a completed node result | **Necessary runtime adaptation:** an HTTP-success response whose public text is empty or whitespace is recorded as `EmptyAgentResponse`, including its exact provider/retry receipt and input provenance. It is not admitted as a fresh replacement artifact, cannot enter `outputs` or be consumed by `SET_OUTPUT`/`FINISH`; the previous immutable receipt remains historical/stale and already completed upstream artifacts remain in the partial result. |
| Canvas rejection feedback | Existing `GraphValidationIssue` codes plus SkillFlow's repeated Action/Tool-call observation precedent | **Existing core reused / project thin adaptation:** typed `feedback_code` accompanies the legacy feedback string; an identical rejected action at an unchanged revision is reported as `repeated_rejected_action`. Scalar Director observation v2 exposes current Agent IDs, available model IDs, current relations, remaining rounds, and recent typed rejections without suggesting an edit. |
| AIME free-text answer extraction | SkillFlow's `training/reward.py::extract_math_answer` recognizes explicit `\\boxed{...}`, answer markers, and a final-lines numeric fallback; downstream SkillEval's private AIME scorer canonicalizes a structured integer with `str(int(...))` | **Necessary protocol adaptation:** AgentGraph emits free text rather than an already-structured terminal action. The v2.1 target-blind extractor admits a whole bare integer, one consistent set of explicit boxed/final-answer integer markers, or the narrower SkillFlow fallback where the final non-empty line is entirely one integer, then applies the private integer/range normalization. It omits arbitrary last-number and symbolic-equivalence fallbacks. |

The formal evaluator target remains isolated. Candidate extraction accepts only
prediction text, so neither fan-in conflict detection nor terminal parsing can
select a candidate by comparing it with the expected answer. A missing,
ambiguous, malformed, or out-of-range candidate is a parsing failure.

The v2 evaluation condition is
`config/evaluation_aime2026_runtime_v2.yaml`. It uses the same 30 tasks, base
weights, generation seed, and catalog presentation as v1, while assigning a
new scalar observation protocol version and separate artifact/report paths.
Tools, training, GRPO, MACE, Bayesian inference, and Skill functionality remain
disabled.

## Fan-in artifact visibility v2.4 source map

The fixed-30 Wrong Demos separate two boundaries that must not be conflated.
`AgentRuntime._upstream` and `openai_gateway._format_upstream` already route the
complete artifact body once per incoming edge; Task 03 confirms that a fan-in
Agent received both the bare `solver` candidate and the other source's full
derivation.  Task 08 was a reciprocal block rather than fan-in, and its
`max_rounds` termination was caused by rejected/no-op edits followed by a
last-round `SET_OUTPUT`, not by a truncated routed artifact.

| v2.4 boundary | Upstream source | Reuse / adaptation |
|---|---|---|
| Complete per-source artifact body | Existing `AgentRuntime._upstream` and `UpstreamMessage`; FlowSteer `Aggregate` keeps candidates in separate indexed blocks | **Existing core reused:** no new truncation or candidate-only projection is applied to Agent-to-Agent communication. Each direct source remains a separate envelope with its immutable full `raw_output`. |
| Source execution identity | Existing artifact metadata already persists producing model and contract; neither FlowSteer `Aggregate` nor SkillFlow's sequential Tool observations provide peer-Agent provenance | **Project-specific thin adaptation:** optional `source_model_id` and `source_contract` are copied into each direct `UpstreamMessage` and model-visible envelope. Ancestor raw artifacts are not recursively duplicated. |
| Public intermediate artifact content | FlowSteer operators pass a computed solution to later operators; SkillFlow retains each public Action--Observation message for the next bounded turn | **Necessary execution-protocol adaptation:** the generic, Output-pointer-invariant Agent protocol asks for contract-relevant public derivation, evidence, intermediate results, and checks instead of an unsupported bare candidate. This adds no Agent role, count, relation, or topology template. |
| AIME terminal-format scope | SkillFlow's target-blind math extraction accepts an explicit final candidate from free text | **AIME adapter correction:** `answer_format` is described as a terminal evaluator boundary. It no longer tells every intermediate Agent to discard its checkable work product. |
| Director artifact visibility | FlowSteer progressive Canvas feedback plus its input-identity execution cache; FlowSteer `ScEnsemble` uses a head--tail preview with an explicit truncation marker | **Project-specific Canvas adaptation:** every revision-live artifact exposes a target-blind candidate, character count, head--tail preview, and direct fan-in provenance/conflict. This compact receipt persists after rejected edits; the immutable Runtime artifact remains full. No evaluator target or candidate winner is exposed. |

This correction improves information preservation and diagnosis only.  It does
not auto-select a candidate, auto-finish a trajectory, relax explicit
`FINISH`, or change the frozen AIME model catalog and search space.

## AIME parameter-level action masking and termination lookahead v3

The v3 condition keeps AIME on the scalar, free-contract action set.  It does
not import the QA-specific `ADD_SUBGRAPH` role domain or any mathematical
workflow template.

| v3 boundary | Upstream source | Reuse / adaptation |
|---|---|---|
| Scalar `ADD_AGENT` live domain | FlowSteer `WorkflowGraph` neutral `node_N` allocation; existing unified `AgentWorkflowEnv._available_model_ids`; SkillFlow-style Runtime execution-profile registration | **FlowSteer/SkillFlow boundary reused, project thin adaptation:** v3 constrains only the next neutral ID, live model catalog, and registered `(execution_mode, allowed_tools)` pairs. `contract` remains free text; there is no mathematical role enum, Agent-count template, relation template, or topology prior. |
| Exact parameter schema | Existing `agentgraph.model-admissible-action-mask.v3` hierarchical action discriminator and exact parameter phase | **Existing core reused / compatibility adaptation:** add the missing scalar `ADD_AGENT` parameter branch. The sampled action is still parsed and consumed unchanged by the authoritative Canvas; malformed semantics are not repaired. |
| Candidate agreement/conflict, freshness, and provenance | Existing immutable Runtime artifacts, direct `UpstreamMessage` provenance, and AIME target-blind extractor | **Existing core reused / projection adaptation:** Canvas groups only fresh parseable public candidates and retains source Agent/artifact IDs, parsing status, and direct upstream provenance. Agreement is observable string equality, not correctness or independence; no target, reward, winner, or adjudication enters the observation. |
| Artifact-consumption ordering | FlowSteer's progressive execute-after-edit Canvas and input-identity artifact reuse | **Project-specific necessary adaptation:** when a fresh candidate exists, expose strict-progress relation consumption, then a fresh candidate-owning Output target, then explicit `FINISH`. `SET_OUTPUT` is hidden for parsing failure or unresolved conflict. Repair/augmentation remains available only when measured failure/conflict/no terminal artifact prevents this path. |
| Termination lookahead | Existing `AgentGraph.construction_progress()` over atomic `SET_RELATION`, `SET_OUTPUT`, and explicit `FINISH` edits | **Existing core reused / project-specific policy projection:** expose the state-conditioned structural lower bound and, when the remaining horizon reaches that bound, mask to legal actions that strictly reduce it. The bound does not auto-finish, recover a historical candidate, or predict Runtime/model success. |
| Output parsing gate | SkillFlow-derived target-blind AIME candidate extraction already used by the evaluator | **Existing adapter reused at the Canvas terminal boundary:** an ordered-artifact condition admits `FINISH` only when the current Output artifact has one deterministic public candidate. Failure is typed `output_parsing_failure`; no target, LLM repair, or candidate synthesis is used. |
| Empty live domain | FlowSteer's bounded Canvas natural termination and existing verified-QA `canvas_action_domain_exhausted` receipt | **Existing core generalized:** generic model-admissible conditions persist the same typed public terminal diagnosis when no exact action/parameter domain remains, instead of constructing an empty JSON schema. No implicit `FINISH` or historical candidate is created. |

The fixed evaluation condition is
`config/evaluation_aime2026_runtime_v3_artifact_termination.yaml`.  It preserves
the official 30-task split, Direct predictions, base Qwen3.5-9B Director
weights, model catalog presentation, seed, 20-round environment limit, and
target-blind integer evaluator from v2.3.  Tools, training, GRPO, MACE,
Bayesian inference, Skill retrieval, and Skill evolution remain disabled.

## SGLang auto-sized request-pool receipt compatibility

The deployed SGLang version preserves an omitted CLI
`--max-running-requests` as `max_running_requests: null` in `/server_info`.
Its scheduler publishes the resolved pool size in
`internal_states[].effective_max_running_requests_per_dp` (see the installed
SGLang scheduler `get_internal_state` implementation). The unified runtime
receipt now reuses that upstream effective field only when the configured
field is null, requires every DP state to expose the same positive integer,
and records which source supplied the value. It does not infer a limit from
GPU memory or substitute the evaluation concurrency.
## AIME runtime v4: durable turns and provenance-bound assessment

The v4 condition keeps the v3 Qwen3.5-9B Director prompt, scalar action set,
model catalog, seed, fixed 30-task split, and target-blind evaluator. It adds
no mathematical role, Agent count, relation, topology, Tool, or Skill prior.

| v4 boundary | Upstream source | Reuse / adaptation |
|---|---|---|
| Completed-turn persistence | FlowSteer's graph snapshots and input-identity execution cache; SkillFlow's public Action--Observation continuation boundary | **Existing boundaries reused / project-specific necessary adaptation:** neither checked upstream implementation persists a complete multi-Agent episode at every accepted Canvas turn. An append-only checkpoint now binds the exact TurnRecord, GraphSnapshotEvent, public Runtime state, next Director transcript, task/condition/policy/sampling identity, and completion status. Resume starts at the next Director round and does not replay fresh Agent artifacts. |
| Immutable artifact restoration | Existing unified artifact_id, full raw_output, per-source input_artifact_provenance, graph revision, and dirty-closure cache | **Existing core reused / serialization adaptation:** JSON receipt fields are restored exactly; opaque runtime-only objects are excluded using the existing trajectory receipt filtering semantics. SET_OUTPUT remains pointer-only and FINISH consumes the restored fresh Output artifact without a model call. |
| Provenance-bound candidate assessment | Existing full UpstreamMessage envelopes and AIME target-blind candidate extractor | **Project-specific thin protocol:** a downstream artifact may emit supported, insufficient_evidence, or refuted only for an exact fresh upstream artifact_id and the exact public candidate. refuted requires a public counterexample. The parser neither recomputes an answer nor receives the evaluator target. |
| Lineage-local recovery | Existing PRESERVE -> DIAGNOSE -> REPAIR -> AUGMENT admission and parameter-level action domains | **Existing core reused / target-domain adaptation:** a fresh negative assessment attributes MODIFY_AGENT to the assessed source Agent. Stale/unbound assessments cannot block a new artifact, and no fixed Verifier role or workflow is introduced. |
| Evidence-aware terminal gate | v3 artifact-consumption ordering, construction-progress lower bound, Output-pointer gate, and explicit FINISH semantics | **Existing core reused / lower-bound adaptation:** an unassessed, insufficient, refuted, conflicting, or stale candidate cannot be selected or finished. The public lookahead includes the minimum remaining assessment/selection/FINISH actions; a supported fresh artifact is consumed before unrelated graph growth. |

The evaluation condition is
config/evaluation_aime2026_runtime_v4_evidence_recovery.yaml. The outer task
boundary is 900 seconds and the inner Agent execution boundary is 480 seconds,
so a typed Runtime failure can be persisted before the task collector expires.
This is a runtime recovery setting, not an MD-defined mathematical horizon.
The Director remains at the unchanged configurable max_rounds: 20.


## AIME runtime v5 contract grounding and artifact completeness source map

The frozen candidate configuration is
`config/evaluation_aime2026_runtime_v5_contract_completeness.yaml`. It is a
mechanical derivative of the v4.9 condition: the official 30-task test slice,
seed, catalog namespace and order, base Qwen3.5-9B policy identity, paired
Direct predictions, target-blind integer evaluator, 20-round horizon, and
artifact/report isolation are unchanged. The new condition has its own output
namespace. The complete same-30 evaluation obtained 12/30 rather than v3's
14/30, so it did not replace the evidence-selected v3 best-profile pointer.

| v5 boundary | Upstream source | Reuse / adaptation |
|---|---|---|
| Transactional contract admission | FlowSteer `src/interactive/workflow_env.py::WorkflowEnv._step_internal` parses and validates a proposed edit before committing Canvas state | **FlowSteer transactional Canvas reused / project thin adaptation:** AIME `ADD_AGENT` and `MODIFY_AGENT` free-text contracts receive a target-blind task-specification guard at the existing admission boundary. It rejects question-external, constraint-bearing numeric assertions or precommitted terminal claims without reading the evaluator target. Agent IDs, models, contracts, relations, Output selection, topology and `FINISH` remain Director decisions. |
| Bounded length continuation | SkillFlow `training/batch_inference.py::_supervisor_call_unpaused` detects `finish_reason == "length"`, retains the first assistant prefix, makes one bounded continuation call to the same resolved model, and limits that continuation to 512 tokens | **SkillFlow execution schedule reused / necessary Agent-artifact adaptation:** the unified Runtime retains the exact original Agent request and partial assistant artifact, performs at most one same-model continuation with a neutral completion instruction and a 512-token bound, and records both generation segments. Unlike SkillFlow's Supervisor Tool-call retry, this path completes the same free-text Agent artifact and does not require or introduce a Tool call. It never changes model, contract, upstream inbox, Tool configuration, task or answer protocol. |
| Artifact completeness and terminal admission | FlowSteer progressive execution cache and pointer-only terminal reuse; existing immutable AgentGraph artifacts and provenance receipts | **Existing core reused / project thin adaptation:** `finish_reason=length` is incomplete until the bounded continuation completes. Incomplete artifacts remain diagnostic receipts but are excluded from candidate agreement, Output admission and explicit `FINISH`. A complete, fresh, parseable and unassessed artifact is terminal-admissible under `reject_negative`; conflict, `insufficient_evidence` or `refuted` remains blocking. No candidate is ranked against ground truth and no historical artifact is promoted. |
| Existing artifact consumption | v3 artifact-consumption ordering and termination lookahead; v4 provenance-bound assessment | **Existing unified core reused:** current candidate agreement/conflict, freshness, artifact provenance, strict-progress relation ordering, Output-pointer selection and explicit `FINISH` remain enabled. v5 changes only contract admission and completeness handling; it adds no fixed role, Agent count, chain, parallel pattern, verifier workflow or mathematical solving template. |

All v5 paths keep AIME Tools disabled. Training, backward, optimizer update,
LoRA publication, GRPO, MACE, Bayesian posterior/EVSI, Skill retrieval,
Skill evolution, retrieval databases, Web search and answer lookup are also
disabled. The completed v5 condition produced 30/30 evaluator-valid explicit
FINISH trajectories with zero operational, terminal, max-rounds, or parsing
failure, but strict Accuracy was 12/30 (40.00%), below v3's 14/30 (46.67%).
It remains a versioned rejected candidate rather than the default profile.

## AIME runtime v6-v8 reliability integration source map

The v6-v8 candidates reuse the v3 scalar free-Agent search condition and the
v5 runtime boundaries. They do not add an Agent role enum, fixed Agent count,
chain/parallel template, verifier workflow, mathematical method, Tool, or
Skill prior.

| Boundary | Upstream source | Reuse / adaptation |
|---|---|---|
| Truncated completion authority | SkillFlow `training/batch_inference.py::_supervisor_call_unpaused` treats `finish_reason == "length"` as a bounded continuation condition | **SkillFlow strategy reused / project receipt correction:** `finish_reason=length` overrides contradictory provider claims of completeness. The same-model bounded continuation remains the only completion path; an incomplete artifact cannot become a terminal candidate. |
| Artifact reuse at termination | FlowSteer progressive Canvas execution cache and pointer-only `SET_OUTPUT` / `FINISH` behavior | **FlowSteer core reused:** Output selection and explicit termination consume the current fresh immutable artifact. Neither action triggers a new Agent execution. |
| Timeout hierarchy | v4/v5 existing inner Runtime versus outer collection boundary | **Existing project adaptation retained:** Agent execution is bounded at 480 seconds and task collection at 900 seconds so a typed Runtime result can be persisted before the outer boundary. This fixes the equal-600-second race without changing mathematical search. |
| Contract task grounding | FlowSteer validate-before-commit Canvas transaction | **FlowSteer transaction reused / project target-blind guard:** free-text Agent obligations cannot inject question-external numeric assertions, derived conclusions, assumptions, or solution-method constraints. The guard reads only the public problem and never the evaluator target. |
| AIME public output marker | SkillFlow target-blind math extraction accepts explicit final-answer markers and boxed integers | **SkillFlow protocol reused / thin execution adaptation:** a non-format Agent that derives a terminal integer retains its public derivation and appends `Final Answer: <integer>`. The marker does not change the Output pointer and cannot invent or repair a candidate. |
| Partial trajectory persistence | Existing v4 append-only `rollout_checkpoints` stream | **Existing core reused:** completed turns survive later Director/provider failure. A remaining reporting gap is that an unfinished checkpoint is not yet materialized as a formal operational-failure trajectory. |

The same-30 results were v6 `8/30`, v7 `13/30`, and v8 `10/30`, all below
the v3 selected result `14/30`. Accordingly, the source-aligned reliability
code is retained while none of these evaluated conditions replaces the v3
best-profile pointer.
