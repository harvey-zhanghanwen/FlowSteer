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

## AIME runtime v11-v13: role-neutral live subgraphs and native Qwen thinking

The v11-v13 conditions retain the unified free AgentGraph and FlowSteer's
progressive Canvas boundary: one accepted Canvas edit is executed before the
next Director observation. They do not add a mathematical role enum, fixed
Agent count, fixed topology, solving method, Tool, Skill, or training path.

| Boundary | Upstream source | Reuse / adaptation |
|---|---|---|
| Progressive Canvas execution | FlowSteer `src/interactive/workflow_env.py::WorkflowEnv.step/_step_internal` records one parsed edit, its execution result, and feedback before the next round | **FlowSteer boundary reused:** `AgentWorkflowEnv` continues to consume one accepted action and execute the resulting graph before returning the next Canvas observation. `ADD_SUBGRAPH` is one functional Canvas edit even when it declares more than one Agent. |
| Strict action and terminal boundary | SkillFlow `src/skillev/rollout/codec.py::StructuredJsonActionCodec`, `src/skillev/runtime/openai_provider.py::OpenAIProvider._generate_sync`, `src/skillev/rollout/engine.py::RolloutEngine.run`, and `src/skillev/rollout/environment.py::{SubmittedTerminalValue,NoTerminalSubmission,TerminalEvaluationRequest}` | **SkillFlow protocol reused / interface adaptation:** each Director continuation is constrained to one strict JSON action receipt. Formal evaluation still requires an explicit legal `FINISH`; horizon exhaustion has no submitted value. The project maps these semantics to the existing Canvas action schema rather than importing a second environment. |
| Role-neutral live `ADD_SUBGRAPH` | FlowSteer's progressive multi-node Canvas edit plus SkillFlow's state-bound strict JSON schema | **Project-specific necessary adaptation:** `src/interactive/director.py::_live_free_subgraph_domain` and the v3 schema builders expose exact neutral Agent IDs, catalog model IDs, registered execution profiles, free-text contracts, legal relation endpoints, and Output choices. `src/interactive/rollout_collector.py` binds the Agent-declaration phase and final action phase to their exact prompts and receipts. The Director chooses one to three new Agents and their relations; no `Solver`, `Verifier`, chain, parallel, debate, or voting template is supplied. |
| Transitive candidate provenance | Existing immutable artifacts and one-envelope-per-edge `UpstreamMessage` provenance | **Existing core reused / projection correction:** `src/interactive/agent_workflow_env.py::candidate_state` resolves a copied candidate recursively to its root artifact IDs and source Agent IDs. Derived copies therefore do not count as independent agreement. The projection reports lineage and string agreement/conflict only; it never judges correctness or reads the evaluator target. |
| AIME extraction v2.2 | SkillFlow's target-blind math-answer markers plus downstream SkillEval integer canonicalization | **SkillFlow protocol reused / thin parser adaptation:** `src/interactive/aime2026_adapter.py` adds deterministic support for LaTeX text markers such as `\text{Final Answer: } 491` and `\text{Answer: }491`, while retaining bare integers, boxed integers, explicit answer markers, conflict detection, range checking, and fail-closed parsing. It performs no model call, answer repair, or target-based selection. |
| Native Qwen chat-template thinking | SkillFlow `src/executor/openai_request_policy.py::OpenAIRequestPolicy.budget` accepts `enable_thinking`; its checked `src/executor/m_exec.py` path explicitly invokes that interface with `enable_thinking=False` | **User-requested versioned condition, not an upstream default:** `src/interactive/rollout_collector.py::SGLangReceiptDirectorClient` passes the explicit boolean to Qwen's chat template and records it; `src/interactive/openai_gateway.py` passes the local Agent flag through `chat_template_kwargs`. v12/v13 set it to `true` for the local Qwen3.5 Director and local Qwen Agent. Those conditions recorded the chat-template flag, but did not yet prove that native SGLang JSON grammar was deferred until the end of reasoning; v14 closes and tests that transport boundary. Hidden reasoning is retained in the exact generation receipt and is not exposed as Canvas feedback. |

The v11 configuration is
`config/evaluation_aime2026_runtime_v11_live_subgraph.yaml`. The v12 native-
thinking condition is isolated in
`config/evaluation_aime2026_runtime_v12_live_subgraph_thinking.yaml` and
`config/model_catalog_multidataset_tool_v2_aime_thinking.yaml`; it does not
reuse the non-thinking Direct predictions.

The v12 two-task canary provides a concrete output-budget receipt rather than
an assumed model-capability claim. Both Direct requests reached
`finish_reason=length` with exactly 4096 completion tokens and returned an
empty public answer, which the v2.2 extractor recorded as `empty_answer`.
One Agent execution showed the same 4096-token/empty-output condition before
the free Canvas recovery produced a later terminal artifact. This evidence is
why v13 is a separate generation condition rather than an in-place rewrite of
v12.

The v13 condition is
`config/evaluation_aime2026_runtime_v13_live_subgraph_thinking_16k.yaml` with
`config/model_catalog_multidataset_tool_v2_aime_thinking_16k.yaml`. It raises
only the local Qwen Agent/Direct context and completion boundary to 32768 and
16384 tokens, respectively; remote catalog entries remain at their recorded
budgets, and the Director action budget remains separately bounded. v13 uses
a new Direct protocol label, artifact namespace, and paired-evaluation
condition, so v12 predictions or trajectories cannot be silently reused as
v13 results. No v13 Accuracy claim is made by this source map; only completed
v13 receipts may supply that result.

Training, backward, optimizer updates, LoRA publication, GRPO, MACE, Bayesian
posterior/EVSI, Skill retrieval/evolution, retrieval, Web search, and answer
lookup remain disabled in all three conditions.

## AIME runtime v14: reasoner-aware StructuredAction and computation runtime

The prepared v14 condition is
`config/evaluation_aime2026_runtime_v14_scalar_thinking_computation.yaml`, with
the versioned catalog
`config/model_catalog_multidataset_tool_v2_aime_thinking_32k.yaml`. It is an
inference-only candidate and has not replaced the evidence-selected v3
best-profile.

| v14 boundary | Upstream source | Reuse / adaptation |
|---|---|---|
| Reasoner-aware JSON Schema decoding | SGLang 0.5.15 `GenerateReqInput.require_reasoning` and `ReasonerGrammarBackend`; `sglang.launch_server` exposes `--reasoning-parser qwen3` and `--enable-strict-thinking` | **Necessary Qwen3.5/SGLang transport adaptation:** every native Director `/generate` request binds `require_reasoning` to the exact chat-template `enable_thinking` value. The project launch script enables the `qwen3` parser, strict thinking, and XGrammar. The runtime receipt records both client and server conditions. Hierarchical selectors and complete Canvas actions parse only after the required `</think>` boundary and reject malformed, duplicate-key, non-object, or inadmissible values without repair. |
| Schema-exact action serialization | SkillFlow strict StructuredAction codec and SGLang/XGrammar JSON Schema | **Protocol-compatible correction:** singleton action schemas now state `type=object` explicitly while retaining the same required fields and `additionalProperties=false`. The neutral scalar v8 prompt asks only for keys admitted by the live schema; it adds no task role or workflow prior. Schema-invalid keys are still rejected by `AgentActionParser`, never deleted or repaired. |
| Fail-closed runtime preflight | Existing SkillFlow-style Supervisor readiness and exact runtime receipts in `src/interactive/policy_sync.py` | **Existing boundary reused / necessary validation adaptation:** a thinking Director using JSON Schema cannot issue benchmark requests unless `/server_info` reports `reasoning_parser=qwen3`, `grammar_backend=xgrammar`, `enable_strict_thinking=true`, and a context length at least as large as `director.max_context_tokens`. The manifest records a typed preflight failure instead of silently accepting an incompatible reasoning/grammar deployment. |
| Progressive free AgentGraph search | FlowSteer Canvas step/execute/feedback continuation; MD six atomic actions and free-text Agent contract | **Existing core reused:** v14 exposes only `ADD_AGENT`, `MODIFY_AGENT`, `DELETE_AGENT`, `SET_RELATION`, `SET_OUTPUT`, and `FINISH`. A parseable artifact no longer collapses the remaining action domain to a one-Agent terminal path while budget remains. Only the formal termination lower bound narrows actions. No role enum, fixed Agent count, fixed chain, parallel solver, verifier, voting, or mathematical method is supplied. |
| Explicit Tool completion | SkillFlow StructuredAction / Action--Observation continuation and the existing project ReAct Tool Adapter | **Existing protocol reused / fail-closed adaptation:** calculator and bounded Python execution are available as computation Tools. Tool stdout is an Observation, not an answer artifact; an Agent must emit an explicit `COMPLETE` action. A failed Tool result remains a typed failure and never becomes a terminal artifact. No search, answer lookup, or AIME solution database is enabled. |

## AIME runtime v17: operational thinking-budget and artifact-preservation repair

| Boundary | Upstream source | Project status |
|---|---|---|
| Separate reasoning allowance | SkillFlow `training/batch_inference.py::supervisor_call` adds `thinking_budget` under `chat_template_kwargs` while retaining thinking mode and reserving visible output tokens | **SkillFlow reused / thin transport adaptation:** AIME ReAct requests carry a versioned per-turn action budget and a smaller reasoning budget. Remote OpenAI-compatible models receive the SkillFlow field only; an explicit local-SGLang capability admits the backend-specific `custom_params` transport. |
| Native Director boundary | SGLang 0.5.15 `GrammarManager._get_request_thinking_budget` reads `SamplingParams.custom_params.thinking_budget` and `ReasonerGrammarBackend` applies JSON grammar after the reasoning boundary | **Necessary Qwen3.5/SGLang adaptation:** the local Qwen3.5-9B Director sends `sampling_params.custom_params.thinking_budget`, preserves `require_reasoning=true`, and records the budget in the exact receipt. The parser remains strict and never fabricates `</think>` or recovers an action from reasoning text. |
| Serialization recovery | Existing project relation/parameter selector bounded regeneration, derived from SkillFlow's bounded structured-action continuation | **Existing boundary reused:** `action_selection` now receives the same one-request, same-model/schema/route/seed regeneration after a serialization or missing-boundary failure. Both exact phase receipts are retained; a semantic-invalid selector is not regenerated and no Canvas edit occurs before valid parsing. |
| Fresh artifact ordering | FlowSteer progressive Canvas `edit -> execute -> feedback`, plus the MD's `PRESERVE -> DIAGNOSE -> REPAIR -> AUGMENT` and explicit artifact-consumption ordering | **Project algorithm correction:** when a fresh, complete, non-conflicting terminal artifact is already admissible, the live action domain exposes `SET_OUTPUT` and then `FINISH` before augmentation. Failed/dirty ingress into a fresh target is masked and rejected; normal chain, parallel, reciprocal, fresh-to-fresh, removal, and repair relations remain available whenever the runtime state requires them. |

This v17 condition does not add a mathematical workflow, role enum, fixed Agent
count, topology prior, Skill, evaluator target, training update, answer lookup,
or relaxed terminal semantics.
| Generation budgets | SkillFlow separates Supervisor StructuredAction generation from Executor generation | **Versioned condition:** the Director action budget is 4096 tokens, while the local Qwen3.5 Agent/Direct completion budget is 32768 tokens within a 65536-token context. Remote catalog budgets are unchanged. These are configurable runtime limits, not MD-defined constants. |

The implementation preserves exact prompt IDs, output token IDs, behavior log
probabilities, generation seed, policy version, server weight version, and
per-phase receipts. Thinking text is diagnostic receipt data only; Canvas
consumes the parsed action, and no reasoning trace is injected into subsequent
feedback as a hidden orchestration prior. Training, backward, optimizer
updates, LoRA publication, GRPO, MACE, Bayesian posterior/EVSI, Skill
retrieval/evolution, retrieval, Web search, and answer lookup remain disabled.

## AIME runtime v18: bounded StructuredAction truncation regeneration

| Boundary | Upstream source | Project status |
|---|---|---|
| Parse-before-length recovery | SkillFlow `training/batch_inference.py::supervisor_call` parses native/text Tool calls before inspecting `finish_reason`; only an unparseable `finish_reason=length` response enters recovery | **SkillFlow reused:** a complete schema-valid StructuredAction is consumed even when the provider reports `length`. An incomplete response is retained verbatim in `react_trace` and `model_calls`, but only its typed public parse-error Observation is shown on the next turn. |
| Bounded same-model regeneration | The same SkillFlow function makes one retry with the same resolved model, `max_tokens=512`, and `enable_thinking=false` | **SkillFlow reused / protocol adaptation:** the unified StructuredAction adapter performs one short regeneration with the same model, provider, response schema, Tool domain, and public state. It disables thinking and the reasoning-trace requirement only for that serialization repair request. |
| Fail-closed recovery | SkillFlow logs and stops after its single retry cannot produce a Tool call | **Necessary unified-core adaptation:** the project cannot treat the truncated text as a direct answer because AgentGraph requires a valid StructuredAction and explicit completion. A second invalid serialization raises `structured_action_serialization_failure` with reason `output_truncation`; no model switch, answer inference, or further parse loop is allowed. |
| Partial execution preservation | FlowSteer progressive Canvas execution and the existing AgentRuntime `SUCCESS / FAILURE / BLOCKED_BY_UPSTREAM` boundary | **FlowSteer reused:** successful upstream artifacts remain immutable; only the failed node and its dependency closure are marked. Canvas exposes the typed failure through the existing `PRESERVE -> DIAGNOSE -> REPAIR -> AUGMENT` feedback path rather than misclassifying it as `react_turn_exhaustion`. |

The independent formal condition is
`config/evaluation_aime2026_runtime_v18_structured_action_recovery.yaml`.
The targeted condition
`config/evaluation_aime2026_runtime_v18_structured_action_canary.yaml`
selects the frozen operational-risk tasks 05, 09, 10, and 11 from the same
official AIME 2026 test slice. Both retain the v17 heterogeneous all-thinking
catalog, free AgentGraph search space, Director prompt, official evaluator,
and Direct protocol. Training, GRPO, MACE, Bayesian posterior, Skill
retrieval/evolution, Web search, and answer lookup remain disabled.

## AIME runtime v19: role-neutral multi-Agent search restoration

| Boundary | Upstream source | Project status |
|---|---|---|
| Progressive functional edit | FlowSteer `src/interactive/workflow_env.py::{step,_handle_add,_execute_workflow}` accepts one Canvas edit, executes it, and returns feedback before the next edit | **FlowSteer boundary reused:** the existing project `ADD_SUBGRAPH` transaction is selected instead of the v18 scalar `ADD_AGENT` profile. One accepted edit may declare one to three free-text Agents and at most one legal relation, then executes once. More complex relations remain later progressive `SET_RELATION` edits. |
| State-bound structured action | SkillFlow `training/environment.py`, `training/batch_inference.py`, and `training/trajectory.py` retain a short Supervisor instruction, one action--observation transition, bounded same-model retry, and per-turn receipts | **SkillFlow boundary reused / existing interface adaptation:** the Director receives only the live action/parameter domain. Agent count, contract, catalog model, relation, Output, and termination remain model choices; no mathematical role or topology template is added. |
| Unassessed artifact action domain | FlowSteer execute--feedback continuation plus the project MD's free `G=(V,E,o)`, target-blind artifact provenance, and `PRESERVE -> DIAGNOSE -> REPAIR -> AUGMENT` policy | **Project-specific necessary adaptation:** a positively supported fresh candidate still narrows to `SET_OUTPUT`, then explicit `FINISH`. A complete parseable but `unassessed` candidate keeps both `SET_OUTPUT` and ordinary legal Canvas edits visible, allowing the Director to consume sufficient work or construct further role-neutral collaboration. Conflict, refutation, incompleteness, and Runtime failure continue to block terminal consumption and expose measured repair/augmentation. This does not classify problem complexity or require multiple Agents. |

The independent v19 canary and formal configurations are
`config/evaluation_aime2026_runtime_v19_multiagent_search_canary.yaml` and
`config/evaluation_aime2026_runtime_v19_multiagent_search.yaml`. They retain
the v18 heterogeneous all-thinking catalog, StructuredAction truncation
repair, artifact assessment, completeness gate, termination lookahead,
official evaluator, and Direct protocol. Training, structural reward, GRPO,
MACE, Bayesian posterior, Skill retrieval/evolution, Web search, and answer
lookup remain disabled.

## AIME runtime v20: unassessed current-Output search boundary

The v19 canary showed that role-neutral `ADD_SUBGRAPH` alone was insufficient:
when the same transaction selected a one-Agent Output, a complete parseable but
`unassessed` artifact made `FINISH` legal and the ordering projection exposed
only that terminal action. v20 applies the same target-blind distinction at
the already-selected Output boundary. A current Output with a positive
`supported` assessment exposes only `FINISH`; an `unassessed` current Output
keeps `FINISH` and ordinary legal Canvas edits visible. If another fresh
artifact is supported, `SET_OUTPUT` is prioritized before termination.

This remains a project-specific parameter-level action-domain adaptation over
FlowSteer's edit--execute--feedback and SkillFlow's Action--Observation receipt
boundaries. It neither forces multiple Agents nor introduces a role, topology,
mathematical method, structural reward, or evaluator signal. The independent
configurations are
`config/evaluation_aime2026_runtime_v20_unassessed_output_search_canary.yaml`
and `config/evaluation_aime2026_runtime_v20_unassessed_output_search.yaml`.

## AIME runtime v21: public terminal-state semantics

The v20 diagnostic confirmed that `unassessed` current-Output state now exposes
both `FINISH` and ordinary legal graph edits. The Director nevertheless
interpreted `fresh + complete + finish_admissible` as correctness evidence and
terminated an incorrect single-Agent candidate. v21 reuses the neutral wording
already present in the scalar v7/v8 policy and makes the Canvas receipt semantics
explicit for the subgraph profile: terminal admissibility is protocol legality,
freshness/completeness/parsing are artifact-state properties, and `unassessed /
unverified_work_product` has no positive public assessment. The Director may
still finish sufficient work or choose one legal edit for a concrete public
gap. `output_agent_id` remains optional during nonterminal construction.

This is a prompt-level explanation of existing public state, not a complexity
classifier, role assignment, minimum Agent count, relation, topology, workflow,
or structural reward. The independent v21 configurations use
`agentgraph.director.minimal-neutral.v12` and otherwise retain the v20 runtime,
catalog, Tool, recovery, terminal, and evaluator conditions.

## AIME runtime v22: action availability versus termination lower bound

The v21 task-05 trace showed the Director reading
`minimum_remaining_breakdown.add_agent=0` as "ADD is unavailable", although
`admissible_action_types` still exposed legal graph edits. The raw breakdown is
an internal structural lower bound from `AgentGraph.construction_progress`; it
is not an action mask. v22 keeps that full receipt in the environment and
trajectory but omits the ambiguous per-action breakdown from the model-visible
projection. It exposes only the total lower bound plus explicit semantics that
action availability is defined by `admissible_action_types` and
`action_target_domains`.

This is a project-specific observation-serialization correction over the
existing FlowSteer Canvas and SkillFlow Action--Observation boundaries. It does
not add, remove, prefer, or reward any Agent, relation, topology, or method.
The independent v22 configurations use the neutral v13 prompt and preserve all
v21 execution/evaluation conditions.

## AIME runtime v23: provenance- and recovery-bound assessment gate

This section records source alignment for the v23 inference condition. It does
not report a model run, enable training, or perform a security, artifact-hash,
or repository-integrity audit.

### Exact FlowSteer Canvas control path

The upstream class name at revision `1c9f2ab` is
`src/interactive/workflow_env.py::InteractiveWorkflowEnv`. The synchronous
control path is
`workflow_builder.py::InteractiveWorkflowBuilder.run_loop ->
InteractiveWorkflowEnv.step -> InteractiveWorkflowEnv._step_internal`.
`step` records the parsed action, DSL, execution result, success, and feedback
returned by `_step_internal`; `run_loop` stores the corresponding `TurnRecord`
and appends the model response plus that feedback to the next prompt.

For an `ADD` or `MODIFY` that needs a custom prompt, upstream first enters
`AWAITING_PROMPT`; `_handle_prompt_input` later commits the node/prompt and, when
`execute_each_step=true`, calls `_execute_workflow`. `_execute_workflow` renders
the current graph DSL, passes the public problem and accepted prompts to the
executor, and stores `last_execution_result`; the formatted execution result is
then part of the returned feedback before the next building turn. Other
accepted executable edits use the same execute-then-feedback boundary.

Upstream `FINISH` is handled inside `_step_internal`: it checks the upstream
finish constraints, reuses `last_execution_result` when per-step execution is
enabled and a result exists, and otherwise calls `_execute_workflow`. It then
chooses an explicit action `final_answer`, else `last_solver_result`, else the
execution result, and returns `active=false`. Its separate max-round path says
the workflow was auto-finished. The project reuses the progressive
step/execute/feedback and result-reuse boundaries, but not the fixed operator
catalog, answer fallbacks, or max-round auto-finish: project `SET_OUTPUT` is
pointer-only, legal explicit `FINISH` consumes the fresh current-revision
Output artifact without a model call, and horizon exhaustion remains a
terminal failure.

### Current SkillFlow execution semantics and v23 mapping

| Boundary | Current upstream semantics | v23 reuse / thin adaptation |
|---|---|---|
| Truncated Supervisor action | `training/batch_inference.py::_supervisor_call_unpaused` checks a native Tool call and then a parseable text Tool call **before** inspecting `finish_reason`. A parseable Tool call is returned even when `finish_reason=length`. Only `length` with no parseable Tool call appends the truncated assistant content and a short Tool-call instruction to the messages, then makes one call to the same resolved model with the same Tool catalog and temperature, `max_tokens=512`, and thinking disabled. This is a changed recovery request, not an identical provider retry and not free-text artifact continuation. If the second response is still not a Tool call, upstream logs the failed retry and falls through with the original content and no Tool call. | The existing project StructuredAction recovery keeps the first exact response/receipt, parses before testing `finish_reason`, and only an unparseable `length` result receives one same-model/provider/schema regeneration capped at 512 tokens with thinking disabled. The project intentionally fails the Agent call with typed `structured_action_serialization_failure / output_truncation` if that regeneration is invalid; it does not inherit upstream's final fall-through. v23 raises the ordinary ReAct action and free-text length-continuation bounds to 8192, but the one structured regeneration remains capped at 512. |
| Repeated Tool action | `training/environment.py::GenericTaskEnvironment.step` keys a call by Tool name plus canonical sorted arguments. For a retained cached key, except `edit_file`, `run_tests`, `str_replace_editor`, and `verify_fix`, it does not dispatch the Tool again. It increments a repeat counter, returns a `[REPEATED xN]` observation containing or summarizing cached evidence, records the turn/messages, and still consumes an episode step. Thus the upstream repeat is cached Action--Observation state, not new Tool evidence; it need not be adjacent while the key remains cached. | Project ReAct suppresses only the same executable StructuredAction repeated in the unchanged Tool-interaction state. It does not invoke the backend or increment `tool_calls`; it consumes one ReAct turn and emits `schema_invalid / duplicate_tool_request` with `repeat_count`, the prior successful-or-error `cached_observation`, a repair instruction, and the sampled action in the receipt. A different dispatched Tool action resets the consecutive repeat count and makes the earlier action eligible again. |
| Public execution diagnostics | SkillFlow keeps sampled actions and their observations in the trajectory so parse failures, schema failures, truncation recovery, Tool failures, and cached repeats are observable state. It does not convert those observations into answer correctness. | `AgentRuntime._response_output_metadata` projects the distinct public ReAct error codes, or the public failure status when no code exists, as `execution_diagnostic_codes`; `execution_recovery_observed` is true exactly when that list is non-empty. `UpstreamMessage`, immutable artifact receipts, Canvas candidate state, and the downstream Agent prompt preserve this projection together with `artifact_complete`. It is diagnostic provenance, never a target, reward, or correctness label. |
| Dependency-bound unassessed candidate | Upstream Action--Observation state supplies the precedent for binding later decisions to the exact observed dependency; it does not define the project's multi-Agent candidate gate. | A fresh candidate is `dependency_assessment_required` only when its deterministic public candidate equals a candidate carried by an exact upstream artifact in its input provenance. Under the configured `reject_negative` policy, such an artifact cannot be selected or finished while its assessment status is `unassessed`; a `supported` assessment must bind the exact artifact ID and candidate. An upstream relation by itself, or a different downstream candidate, does not trigger this flag. |
| Recovery-observed unassessed candidate | Upstream truncation and repeated-action observations remain visible recovery receipts, not proof that the eventual answer is right or wrong. | A complete artifact that eventually succeeds but carries non-empty `execution_diagnostic_codes` is `recovery_observed`. While it remains `unassessed`, it is excluded from `SET_OUTPUT` and `FINISH` admission under `reject_negative`; a provenance-bound supported public assessment can admit it. The diagnostic does not refute the artifact and does not expose evaluator state. |
| Clean unassessed candidate | Neither upstream source requires a fixed verifier or a minimum multi-Agent graph merely because an artifact lacks a positive assessment. | A fresh, complete, parseable, non-conflicting `unassessed` artifact remains terminal-admissible when it has neither a matching-candidate upstream dependency nor an execution-recovery diagnostic. This preserves v22's free search space and avoids turning assessment into a mandatory role or topology template. Existing refutation, insufficient-evidence, conflict, completeness, freshness, and parsing gates still compose with this rule. |

The independent conditions are
`config/evaluation_aime2026_runtime_v23_provenance_assessment_gate_canary.yaml`
and `config/evaluation_aime2026_runtime_v23_provenance_assessment_gate.yaml`.
They retain the neutral v13 Director prompt, free-text contracts, catalog model
choice, role-neutral one-to-three-Agent `ADD_SUBGRAPH` declaration domain, free
relation and Output choices, and explicit `FINISH`. The condition enables only
the bounded computation ReAct Tool path; it adds no fixed role, fixed Agent
count, fixed topology, mathematical workflow, structural reward, retrieval,
Web search, answer lookup, Skill, training, backward pass, optimizer update,
LoRA publication, GRPO, MACE, or Bayesian update.

## AIME runtime v24: bounded compact Canvas history

The v23 operational-risk canary completed tasks 09 and 11, but tasks 10 and 05
failed before the next Canvas edit with SGLang HTTP 400. Exact Qwen3.5
tokenization of their last persisted Director transcripts established a finite
context overflow: task 10 requested `24743 + 8192 = 32935` tokens and task 05
requested `26371 + 8192 = 34563`, above the configured 32768-token context.
The current Canvas state itself remained valid; the excess came from replaying
full stale graph and artifact payloads in each retained historical observation.

FlowSteer's upstream loop keeps the sampled edit followed by its execution
feedback at every progressive boundary. SkillFlow likewise preserves the
sampled Action--Observation history needed for continuation. Neither source
requires every earlier observation to duplicate the latest revision-live graph
and full artifact previews. v24 therefore reuses the existing compact-history
projection already exercised by this project's QA and scalar Directors:

* the immutable first task/catalog observation is retained;
* exact sampled Director Actions remain in order;
* prior Canvas observations retain typed public feedback, failure, issue, and
  Tool/runtime receipts, but remove stale live graph, candidate, action-domain,
  and artifact-preview copies;
* the final Canvas observation remains exact and complete.

`agentgraph.director.minimal-neutral.v14` is byte-identical to v13 at the system
prompt level; the new version labels only this transcript-history policy. The
free AgentGraph, action space, model catalog, contracts, relation semantics,
Output semantics, artifact gates, evaluator, and sampling condition are
unchanged. Offline projection of the failed checkpoints gives task 10
`14480 + 8192 = 22672` and task 05 `12603 + 8192 = 20795`, both within the same
32768-token limit. The native SGLang client also performs a deterministic
pre-dispatch bound check on `prompt_tokens + max_new_tokens` so a future
overflow is reported as a typed Director context failure rather than an opaque
provider HTTP 400.

The independent conditions are
`config/evaluation_aime2026_runtime_v24_compact_canvas_history_canary.yaml` and
`config/evaluation_aime2026_runtime_v24_compact_canvas_history.yaml`. No Skill,
training, GRPO, MACE, Bayesian update, fixed role, fixed Agent count, fixed
topology, or mathematical workflow is introduced.

## AIME runtime v25: preserved artifacts and bounded truncation recovery

This source map describes the independent v25 configuration and its intended
unified-runtime boundary. It does **not** claim that a canary, formal run,
Accuracy measurement, or model call has completed.

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Progressive execution and cache reuse | FlowSteer `src/interactive/workflow_builder.py::InteractiveWorkflowBuilder.run_loop` and `src/interactive/workflow_env.py::{_step_internal,_execute_workflow}` execute accepted Canvas edits before returning feedback. With per-step execution, upstream `FINISH` consumes `last_execution_result` rather than unconditionally resampling the graph. | The unified runtime keeps immutable, input-identity-bound Agent artifacts and reuses only artifacts whose model, contract, upstream inbox, relation-derived dependency state, and Tool configuration remain fresh. Preserving a successful upstream artifact across a failed downstream repair is a thin cache/reuse adaptation for the free AgentGraph; it is not a second workflow engine and does not infer answer correctness. |
| Short truncation recovery | SkillFlow `training/batch_inference.py::_supervisor_call_unpaused` parses a native/text Tool call before considering `finish_reason`. Only an unparseable `length` response receives one bounded request to the same resolved model, with the same Tool catalog and temperature, a 512-token limit, and thinking disabled. | Project StructuredAction recovery directly follows this same-model, thinking-disabled, bounded-regeneration boundary. For a free-text reasoning artifact, the finite-context adaptation preserves the exact truncated prefix and all segment receipts, permits only the configured bounded continuation, and never treats a truncated or empty suffix as a complete terminal artifact. This extension is project-specific because SkillFlow's cited path repairs a Supervisor Tool action, not an arbitrary free-text Agent artifact. |
| Recovery after a failed dependent Agent | FlowSteer supplies the step/execution/feedback boundary and cached-result precedent; SkillFlow preserves sampled Action--Observation receipts across continuation. Neither upstream defines the project's free-graph repair action mask. | The project keeps successful upstream artifacts available while exposing typed downstream provider, truncation, and runtime failures. Recovery remains `PRESERVE -> DIAGNOSE -> REPAIR -> AUGMENT`; it must not invalidate a fresh successful artifact merely to make an unrelated parameter edit, and a bounded truncation failure must advance to a typed repair-exhausted state instead of repeating an identical accepted edit indefinitely. This is a finite-context/runtime adaptation, not a fixed Solver/Verifier workflow prior. |

The independent condition files are
`config/evaluation_aime2026_runtime_v25_preserve_truncation_recovery_canary.yaml`
and
`config/evaluation_aime2026_runtime_v25_preserve_truncation_recovery.yaml`.
They retain prompt v14, the free-text Agent contract, the same heterogeneous
model catalog, relation semantics, Output/explicit-`FINISH` protocol,
target-blind evaluator, four frozen canary task IDs, and official 30-task
formal population. All v25 artifact and report paths are disjoint from v24;
v24 checkpoints and results are neither overwritten nor relabeled. Skills,
training, GRPO, MACE, Bayesian inference, backward, optimizer update, and LoRA
publication remain disabled.

## AIME runtime v26: lineage-preserving assessment ordering

This source map records the independent v26 configuration boundary. It does
**not** claim that a model/API canary, formal 30-task run, or Accuracy result
has completed.

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Progressive artifacts and dependency receipts | FlowSteer `src/interactive/workflow_builder.py::InteractiveWorkflowBuilder.run_loop` and `src/interactive/workflow_env.py::{_step_internal,_execute_workflow}` preserve the edit--execute--feedback boundary and cache node results within progressive execution. | The v26 condition is prepared to exercise project-specific target-blind provenance closure, including deterministic multi-hop dependency lineage when an upstream natural-language artifact carries the downstream canonical integer but the official-compatible extractor leaves that upstream artifact unparsed. This metadata neither promotes upstream text to an evaluator answer nor chooses whether the candidate is correct. |
| Candidate assessment before terminal consumption | SkillFlow `training/environment.py` preserves sampled Action--Observation history for the next orchestration action, while FlowSteer exposes the current executed workflow result as Canvas feedback before the next edit. | The v26 condition is prepared to exercise assessment-preserving action ordering: when a fresh candidate has outstanding provenance-bound assessment, the admissible declaration boundary preserves that artifact and admits an independent assessment consumer without prescribing its contract or relation. Existing active runtime/dirty-Agent repair remains higher priority. Agent contract, model, relation, Output, Agent count, and topology remain Director choices; no Solver/Verifier role or fixed workflow is introduced. |
| AIME extraction and terminal evaluation | SkillFlow's AIME path supplies deterministic answer extraction/canonicalization and exact-match Accuracy; the unified AIME adapter remains the single Direct/AgentGraph evaluator boundary. | v26 does not change extraction, canonicalization, ground-truth isolation, explicit-`FINISH`, or evaluator semantics. Lexical lineage is internal runtime provenance metadata and is never an answer-recovery path. |

The disjoint profiles are
`config/evaluation_aime2026_runtime_v26_lineage_preserving_assessment_canary.yaml`
for frozen task IDs 05, 09, 10, and 11, and
`config/evaluation_aime2026_runtime_v26_lineage_preserving_assessment.yaml`
for the same official sequential 30-task population as v25. Apart from the
condition identity and artifact/report destinations, both retain v25's prompt
v14, heterogeneous model catalog, Tool protocol, evaluator, generation
condition, action space, free-text Agent contracts, graph constraints, and
disabled Skill/training/MACE/Bayesian settings.

The v26 canary later entered its AgentGraph stage but was safely interrupted
after a pre-canary architecture blocker was identified, before any formal
trajectory was persisted. The retained v26 state is `0/4` AgentGraph
trajectories and zero collection-failure records, with no evaluator metrics.
It is not resumed or relabeled by a later condition.

## AIME runtime v27: bounded assessment repair

This source map records the independent v27 configuration boundary. It does
**not** claim that a v27 model/API canary, formal 30-task run, or Accuracy result
has completed.

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Per-occurrence lexical provenance | FlowSteer's progressive execution preserves each node artifact and dependency receipt; SkillFlow's AIME path keeps answer extraction/canonicalization deterministic and separate from orchestration state. | v27 prepares a target-blind provenance boundary that evaluates each exact integer-token occurrence independently. An occurrence embedded in a fraction is excluded without suppressing a distinct standalone occurrence elsewhere in the same raw artifact. This is lineage metadata only: it does not alter canonicalization, generate an answer, consult ground truth, or rank candidates. |
| Bounded malformed assessment-consumer repair | FlowSteer's Canvas action space includes parameterized `MODIFY_AGENT` followed by real execution feedback. SkillFlow's Action--Observation execution records bounded recovery attempts rather than silently accepting malformed output. | If an assessment consumer is created but cannot produce a provenance-bound assessment artifact, the project may expose only the bounded parameter repair needed to modify that existing consumer. Exhaustion is typed and fail-closed; it cannot admit `SET_OUTPUT` or `FINISH`, fabricate support, or trigger an unbounded modify loop. This remains a runtime recovery boundary, not a predefined verifier role or workflow. |
| Assessment-aware termination lookahead | FlowSteer's progressive loop consumes one Canvas edit per round and terminates only at its explicit terminal boundary; the project retains the MD's explicit `FINISH` semantics rather than upstream max-round fallback. | The lower bound uses the actual live parameter domain: required ingress is `ADD_SUBGRAPH -> SET_OUTPUT -> FINISH`, a repairable malformed non-Output consumer is `MODIFY_AGENT -> SET_OUTPUT -> FINISH`, and repair exhaustion is terminal-unreachable. This masks only actions that cannot leave enough atomic rounds for legal explicit termination; it neither chooses an answer nor injects a role/topology. |
| Evaluation and experiment identity | The same SkillFlow-aligned AIME extraction/canonicalization and exact-match Accuracy remain the shared Direct/AgentGraph evaluator. | v27 changes no task, prompt, model catalog, Tool, evaluator, graph, or generation protocol. It receives disjoint condition, artifact, report, and unused-adapter namespaces so interrupted v26 state cannot be resumed under a changed runtime. |

The disjoint profiles are
`config/evaluation_aime2026_runtime_v27_bounded_assessment_repair_canary.yaml`
for the same frozen task IDs 05, 09, 10, and 11, and
`config/evaluation_aime2026_runtime_v27_bounded_assessment_repair.yaml`
for the same official sequential 30-task population. No Skill, training, GRPO,
MACE, Bayesian inference, backward, optimizer update, LoRA publication, fixed
role, fixed Agent count, fixed topology, or mathematical workflow is enabled.

## AIME runtime v28: task-agnostic detachable dead-branch recovery

The v27 four-task canary remains an incomplete condition. Direct reuse
completed for all four frozen tasks, while three of four AgentGraph tasks
formed evaluator-admitted trajectory records with zero collection-failure
rows. Task 11 did not form a terminal trajectory. Its retained checkpoints
cover rounds 0--15: after the round-10 bounded `MODIFY_AGENT` attempt left
`node_2` failed, repair-exhausted, artifact-free, and terminal-unreachable,
rounds 11--14 alternated the same `node_2`/`node_3` relation between
`node_3 -> node_2` and no relation. Round 15 returned to the same failed
parameter-repair boundary. Throughout that window, Output Agent `node_3`
retained the same fresh, complete artifact; the relation toggles changed only
`node_2`'s upstream inbox and therefore correctly caused the incremental
runtime to re-execute that failed node. These receipts are runtime evidence,
not an Accuracy result, and no v27 formal 30-task evaluation is claimed.

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Incremental execution and immutable fresh artifacts | FlowSteer `src/interactive/workflow_builder.py::InteractiveWorkflowBuilder.run_loop` and its node-level incremental cache retain an executed node result when that node's effective input identity is unchanged, while an accepted relation edit executes before the next Canvas feedback. | v28 leaves this cache contract unchanged. The observed `node_2` re-execution is correct because adding or removing `node_3 -> node_2` changes its inbox; the unchanged `node_3` Output artifact must remain fresh and must not be resampled. |
| Bounded execution recovery | SkillFlow `training/batch_inference.py::_supervisor_call_unpaused` gives an unparseable length-truncated Tool action one short same-model, thinking-disabled retry, then returns the unresolved result instead of replaying the same request indefinitely. | After an existing Agent's bounded parameter repair is exhausted without a successful artifact, v28 treats that measured state as a finite recovery boundary. It does not infer answer correctness, switch models silently, or repeat relation edits that cannot make terminal progress. |
| Relation action masking | FlowSteer's progressive Canvas exposes one parameterized edit and its real execution feedback at a time; strict reachability progress remains determined from the live graph rather than a prompt template. | The project-specific action-mask guard is task-agnostic: explicit evidence-ingress, replacement-routing, and strict terminal-reachability candidates retain priority, but a failed, repair-exhausted, terminal-unreachable Agent cannot fall through to generic relation candidates when none strictly improves recovery. This adds no role enum, fixed edge, chain, parallel pattern, or mathematical workflow. |
| Detachable dead-branch deletion | FlowSteer supplies the explicit `DELETE_AGENT` Canvas mutation and dirty-closure execution boundary. | A narrow project predicate may detach an Agent only when it is failed and repair-exhausted, has no current or preserved successful artifact, is not Output, has no downstream successor, and fork-delete validation proves that every retained fresh artifact keeps the same input identity. The deletion remains an explicit Director action; it does not auto-finish or select a candidate answer. |

The disjoint profiles are
`config/evaluation_aime2026_runtime_v28_detachable_dead_branch_recovery_canary.yaml`
for frozen task IDs 05, 09, 10, and 11, and
`config/evaluation_aime2026_runtime_v28_detachable_dead_branch_recovery.yaml`
for the official sequential 30-task population. They are protocol-equivalent
copies of v27 apart from condition identity, artifact/report destinations,
unused adapter namespace, and version comments. Prompt v14, task selection,
Direct reuse, model catalog, Tool protocol, evaluator, generation parameters,
free Agent contracts, relation semantics, explicit `FINISH`, and disabled
training/Skill/GRPO/MACE/Bayesian settings remain unchanged. This entry claims
no v28 model/API call, canary result, formal result, or Accuracy measurement.

## AIME runtime v29: authoritative dead-branch Canvas admission

The v28 canary is retained as an interrupted, incomplete condition. Its Direct
checkpoint contains four reused records, while AgentGraph contains zero formal
trajectories and the collection-failure stream is empty. One in-progress
round-0 checkpoint was persisted before an authoritative Canvas-admission
blocker was identified and the run was stopped. No Stable Zero result,
Accuracy, paired comparison, or formal 30-task result exists for v28. Its
checkpoint must not be resumed, relabeled, reused as v29 evidence, or compared
as a completed condition.

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Authoritative Canvas admission | FlowSteer's progressive environment keeps Canvas mutation/validation authoritative: a sampled edit is accepted transactionally, executed, and returned as feedback before the next Director action. A projected action mask cannot override that admission boundary. | v29 makes the same task-agnostic detachable-dead-branch predicate authoritative in action-type masking, parameter target domains, preservation admission, and DELETE admission. When strict-progress relation recovery exists it remains first; otherwise only the exact detachable leaf DELETE is accepted. An unrelated ADD, relation toggle, parameter edit, or Output edit cannot bypass this boundary. |
| Incremental artifact reuse | FlowSteer's node-level cache reuses an executed artifact only while its effective input identity is unchanged and invalidates the changed downstream closure after a relation edit. | Accepted dead-leaf DELETE has an empty retained dirty closure. It removes no current or preserved artifact, changes no retained Agent inbox, and therefore does not resample the fresh Output artifact. The following explicit `FINISH` remains a separate Canvas action. |
| Bounded failure state | SkillFlow `training/batch_inference.py::_supervisor_call_unpaused` performs one bounded same-model recovery for an unparseable truncated Tool action and preserves public Action--Observation receipts. | Successful computation Tool receipts remain lossless failure provenance, not automatically promoted downstream artifacts. A repair-exhausted Agent with no routable artifact may be detached only under the existing strict predicate; its historical receipts remain diagnostic evidence and never become a candidate or evaluator answer. |

The disjoint v29 profiles are
`config/evaluation_aime2026_runtime_v29_authoritative_dead_branch_admission_canary.yaml`
for frozen task IDs 05, 09, 10, and 11, and
`config/evaluation_aime2026_runtime_v29_authoritative_dead_branch_admission.yaml`
for the official sequential 30-task population. They are protocol-equivalent
copies of v28 apart from condition identity, artifact/report destinations,
unused adapter namespace, and comments. Prompt v14, task population, Direct
reuse, model catalog, Tool protocol, evaluator, generation parameters,
free-text contracts, graph constraints, explicit `FINISH`, and disabled
training/Skill/GRPO/MACE/Bayesian settings are unchanged.

The completed v29 fixed-task canary persisted four AgentGraph trajectories and
zero collection-failure rows. Only task 05 reached legal explicit `FINISH`, was
admitted to the formal evaluator, and was correct. Thus formal-evaluator
admission is `1/4`; the strict fixed-task correct/resolved rate is `1/4 =
25.00%`. The evaluator-valid subset is `1/1`, but that conditional value is not
reported as four-task architecture Accuracy. Tasks 09, 10, and 11 terminated
with `canvas_action_domain_exhausted` and were correctly excluded from formal
scoring. In particular, task 10 retained a fresh Output artifact whose
canonical candidate was the correct `156`, but no legal explicit `FINISH`
occurred, so that artifact received no evaluator reward.

## AIME runtime v30: failure-specific terminal recovery

v30 is a new condition and does not resume or relabel v29 evidence. Its source
classification is deliberately narrow:

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Assessment-protocol repair parameter domain and authoritative admission | FlowSteer's progressive Canvas supplies the parameterized `MODIFY_AGENT` edit and transactional validation/execution/feedback boundary. SkillFlow `training/batch_inference.py::_supervisor_call_unpaused` supplies the bounded same-model StructuredAction serialization-repair precedent. | **FlowSteer reused + SkillFlow reused + project-specific necessary adaptation:** while a live Agent specifically lacks a provenance-bound assessment, the projected parameter domain and authoritative Canvas admission expose only `model_id` or an answer-free `contract`. Edits to Tools, execution mode, artifact type, or completion condition cannot consume that bounded protocol-repair opportunity. No candidate, answer, role, topology, or mathematical method is injected. |
| Supported sink selection before unrelated recovery | FlowSteer's `SET_OUTPUT` is an explicit Output-pointer edit and its incremental cache reuses a fresh artifact when effective inputs are unchanged. | **FlowSteer reused / project-specific necessary action ordering:** when an already-supported, fresh sink artifact can legally become Output, `SET_OUTPUT` is exposed before unrelated repair or graph augmentation. The subsequent `FINISH` remains a separate Director action and does not resample the sink. |
| Preservation-safe removal of an exhausted non-candidate leaf | FlowSteer supplies `DELETE_AGENT`, fork validation, and dirty-closure execution. SkillFlow supplies the bounded-recovery principle after a failed request cannot be repaired indefinitely. | **FlowSteer reused + SkillFlow reused + project-specific necessary adaptation:** a protocol-exhausted leaf that owns no current/preserved candidate artifact may be exposed for explicit DELETE only when it is not Output, has no successor, and fork validation proves every retained fresh artifact keeps identical effective inputs. Receipts remain diagnostic evidence; DELETE neither selects nor evaluates an answer. |
| Explicit terminal boundary | FlowSteer's upstream execution may return a cached workflow result at its terminal boundary, while SkillFlow benchmark inference is not driven by this project's atomic Canvas `FINISH` contract. | **Project MD incompatibility / thin adaptation:** this project requires a distinct legal explicit `FINISH` before AIME evaluation. A correct historical or current artifact, including task 10's `156`, cannot be recovered by max-round fallback or evaluator-side answer salvage. v30 therefore improves reachability to `SET_OUTPUT -> FINISH` without weakening that rule. |
| AIME evaluator and experiment identity | The existing SkillFlow/SkillEval-aligned target-blind integer extraction, canonicalization, and exact-match Accuracy remain shared by Direct and AgentGraph. | **SkillFlow reused:** v30 changes no task population, prompt, model catalog, Tool protocol, generation condition, evaluator, or answer parser. It uses disjoint condition, artifact, report, and unused-adapter namespaces. |

The prepared profiles are
`config/evaluation_aime2026_runtime_v30_failure_specific_terminal_recovery_canary.yaml`
for the same frozen task IDs 05, 09, 10, and 11, and
`config/evaluation_aime2026_runtime_v30_failure_specific_terminal_recovery.yaml`
for the same official sequential 30-task population. Both retain prompt v14,
the heterogeneous all-thinking model catalog, computation Tool protocol,
`max_rounds: 20`, free-text Agent contracts, graph constraints, and disabled
training/Skill/GRPO/MACE/Bayesian settings. Configuration preparation does not
claim a v30 model/API call, canary result, formal result, or Accuracy.

## AIME runtime v31: candidate-artifact protocol consumption

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Incremental candidate-preserving recovery | FlowSteer `src/interactive/workflow_env.py::{_step_internal,_execute_workflow}` supplies accepted Canvas edit -> execution -> feedback, and `src/interactive/workflow_builder.py` supplies input-identity node caching. SkillFlow `src/skillev/rollout/{codec.py,engine.py,environment.py}` supplies strict StructuredAction, committed Action--Observation state, and a bounded/no-submission failure boundary. | **FlowSteer reused + SkillFlow reused + project-specific necessary adaptation:** a fresh, complete candidate-bearing assessment-protocol failure is not repaired by resampling its owner. Live action masking, parameter schema, and authoritative Canvas admission expose one local `ADD_SUBGRAPH`: one free-contract consumer receives exact directed ingress from the failed artifact owner, every missing candidate provenance owner, and all live candidate owners only when public candidates conflict. Model, contract and execution profile remain Director choices. Exact source artifact IDs persist across completed-turn checkpoints to prevent recursive consumers. This reads no evaluator target and makes no correctness decision. |
| Artifact/provenance resolution | FlowSteer's execution cache preserves unchanged source artifacts and invalidates only the affected downstream closure. SkillFlow keeps public Tool/action receipts attached to the producing trajectory. | A successor clears the old protocol failure only when its fresh artifact consumed the failed artifact and every missing candidate provenance artifact, and bound an assessment to every exact ID. Fan-in preserves each `source_agent`, `artifact_id`, and `raw_output`; it does not concatenate anonymous candidates or recover an answer from history. |
| Explicit terminal/cache boundary | FlowSteer reuses the last execution result at its terminal boundary; SkillFlow distinguishes submitted terminal values from no submission at horizon exhaustion. | `SET_OUTPUT` changes only the pointer and `FINISH` consumes the current-revision fresh Output without resampling. Under the project MD, max-round/action-domain exhaustion keeps `final_answer=null`; no historical candidate salvage is allowed. |

The local exact fan-in is a state-conditioned recovery subgraph, not a fixed
initial/global Agent count, role, topology, or mathematical workflow. The v31
profiles retain v30's task/evaluator/prompt/model/Tool/terminal protocols under
disjoint artifact and report namespaces. Training, Skill, GRPO, MACE, Bayesian
updates, backward, optimizer steps, and LoRA publication remain disabled.

The final v31 action ordering continues to use FlowSteer's pointer-only
`SET_OUTPUT` and `AgentGraph.dirty_closure`: after a provenance-complete sink is
materialized, Output selection does not execute an Agent; the subsequent
strict reachability domain prefers relations that do not change an
assessment-incomplete candidate owner's predecessor identity, then minimizes
the exact downstream invalidation closure. This is a project-specific
parameter-level projection over FlowSteer's legal live relation set, not a
fixed edge or topology. SkillFlow's bounded failure rule remains authoritative
when no provenance-preserving relation exists.

## AIME runtime v32: receipt validation follows the live fan-in domain

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Hierarchical action receipt | SkillFlow's StructuredAction discipline requires the committed action and its receipt to agree; FlowSteer's Canvas remains the authority for whether the parameterized edit is legal. | **FlowSteer reused + SkillFlow reused + project-specific compatibility fix:** `rollout_collector.py::_validate_v3_hierarchical_action_receipt` keeps the ordinary one-relation boundary. Only a non-verified-QA free-text live domain that explicitly requires all existing ingress and an integer relation count greater than one may carry a multi-source fan-in; the receipt must match the exact source set, target the one new consumer in the one-way direction, and continue to pass endpoint, allowed-relation, and duplicate-pair checks. This aligns collection with the existing Director v11 schema and Canvas admission without expanding the search space. |
| Evidence isolation | v31 task 05 proves that collector validation occurs after action sampling but before a completed trajectory/checkpoint can be committed. | v31 remains an incomplete failed condition. v32 uses disjoint condition/artifact namespaces and a clean fixed-task canary; no v31 historical candidate or partially sampled declaration is recovered as a terminal answer. |

## AIME runtime v33: explicit terminal admission and optional coding execution

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Bounded horizon and trajectory preservation | SkillFlow `src/skillev/rollout/engine.py` materializes each completed Action--Observation step and distinguishes `COMPLETED` from `HORIZON_EXHAUSTED`; `rollout/artifact.py` persists the complete artifact/trajectory. FlowSteer `src/interactive/workflow_env.py` persists each Canvas action, feedback, and execution result. | **FlowSteer/SkillFlow direct reuse:** all historical Agent output, Tool receipt, artifact ID, provenance, graph revision, and Canvas feedback remain in immutable turns/checkpoints after terminal failure. |
| No explicit `FINISH` means no terminal submission | SkillFlow's `NoTerminalSubmission(HORIZON_EXHAUSTED)` supplies the no-submission semantic boundary, while upstream FlowSteer may force-finish and retain its last solver result. | **User-MD-required project adaptation:** `AgentGraphOrchestrator.run` and `AgentGraphRolloutCollector` now set `final_answer=null`, retain the current terminal graph, and never promote historical lineage when `explicit_finish=false`. The existing AIME backend returns an invalid `not_evaluated_without_explicit_finish` receipt with `formal_evaluator_called=false`; historical candidates remain debug artifacts only. |
| Code-assisted AIME execution | SkillFlow's bounded StructuredAction/Tool loop is reused through `ToolReactExecutionAdapter`; `computation_tools.py` is the existing dependency-light port of SkillFlow calculator/Python tools. `AgentRuntime.registered_execution_profiles` and FlowSteer's live Canvas domains already correlate execution modes with exact Tool IDs. | **Direct reuse plus thin task adapter:** `aime_tool_runtime.execution_modes` defaults to legacy `[react]`; v33 explicitly enables `[react, coding]` and instantiates one adapter per mode over the same AIME-only calculator/Python registry. SWE-bench repository editing semantics are excluded. Coding remains an execution mode on `agent_id + model_id + free-text contract`, not a role or fixed workflow. |
| Search-space neutrality | The unified Director schema already supports `reasoning`, `react`, and `coding`, and FlowSteer's progressive Canvas executes each accepted edit before the next feedback. | v33 does not change the base prompt, define Solver/Programmer/Verifier roles, force a coding node, or prescribe chain/parallel/debate topology. Only live runtime profiles make coding legally selectable; Director actions, communication relations, Output selection, and `FINISH` remain autonomous. |

The disjoint v33 profiles retain v32's dataset/evaluator, neutral prompt v14,
heterogeneous model catalog, generation settings, free-text contracts, graph
constraints, artifact assessment policy, and disabled Skill/training/GRPO/MACE/
Bayesian features. They change only terminal submission semantics, the explicit
runtime-backed AIME execution-mode set, and versioned experiment destinations.

## AIME runtime v34: canonical execution profiles and assessment recovery

This entry records source classification for the v34 runtime/Canvas changes.
It does **not** claim that a v34 model/API canary, four-task run, official
30-task run, Accuracy measurement, or training update has completed.

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Independent reasoning and action budgets | SkillFlow `training/batch_inference.py::supervisor_call` keeps the reasoning allowance separate from the visible StructuredAction allowance; `src/skillev/runtime/contracts.py::StructuredAction` and `runtime/bounded_agent.py::BoundedAgent.execute_turn` preserve one schema-valid action followed by its public Observation. | **SkillFlow reused:** v34 retains distinct reasoning/action limits and strict StructuredAction parsing. A reasoning trace is not converted into an action, and a Tool Observation is not promoted to a terminal answer artifact without the existing explicit completion boundary. |
| Progressive Canvas execution | FlowSteer `src/interactive/workflow_builder.py::InteractiveWorkflowBuilder.run_loop` and `src/interactive/workflow_env.py::{_step_internal,_execute_workflow}` provide the accepted edit -> execute -> feedback sequence and trajectory boundary. | **FlowSteer reused:** each accepted atomic Canvas edit executes before the next Director observation. Agent artifacts, graph revision, communication provenance, execution receipts, Canvas feedback, and terminal state stay attached to the same trajectory. |
| Canonical execution-profile continuation | SkillFlow's StructuredAction continuation keeps the same resolved execution context across Action--Observation turns; FlowSteer's incremental execution reuses a node only while its effective input identity is unchanged. | **Project-specific necessary adaptation:** a continued or repaired Agent call is bound to the canonical `(model_id, execution_mode, allowed_tools)` profile that produced its current state. Runtime capability lookup, continuation, receipts, and Canvas target domains use that same profile identity; no silent mode, Tool, or model substitution is permitted. |
| Model/profile joint action masking | SkillFlow supplies executable StructuredAction/Tool profiles, while FlowSteer supplies the parameterized Canvas action domain. Neither upstream directly defines a heterogeneous free-AgentGraph model/profile catalog. | **Project-specific necessary adaptation:** ADD domains expose the exact flat union of registered `(model_id, execution_mode, allowed_tools)` tuples. MODIFY model candidates preserve the Agent's current mode/Tools, while mode or Tool candidates come only from the current model's registered profile union. Legacy aggregate fields remain serialization-compatible, but the joint domain is authoritative. |
| Provenance-aware assessment state | FlowSteer's progressive artifacts retain source-node identity and dependency state; SkillFlow preserves public Action--Observation receipts instead of treating diagnostics as evaluator evidence. | **Project-specific necessary adaptation:** assessment aggregation may ignore only a strictly defined null, noncandidate diagnostic artifact. It must retain candidate-bearing artifacts, conflicts, exact `source_agent`/`artifact_id` provenance, incomplete artifacts, and failed Tool/Runtime receipts. This exception neither extracts nor selects an answer and never consults ground truth. |
| Assessment ingress before unrelated structural edits | FlowSteer's Canvas exposes one legal atomic edit at a time and executes it before the next action. | **Project-specific necessary action ordering:** when a fresh artifact requires provenance-bound assessment and graph capacity permits, parameter-level masking exposes the legal `ADD_SUBGRAPH` ingress consumer before terminal-reachability or unrelated relation narrowing. `terminal_progress` and recovery feedback publish the same ingress. The Director still chooses the free-text contract, model/profile, relation endpoints, Output, and later explicit `FINISH`. |
| Fail-fast artifact-state preservation | SkillFlow bounds malformed StructuredAction recovery and retains the failed Action--Observation receipt; FlowSteer preserves successful incremental results outside the invalidated downstream closure. | **FlowSteer/SkillFlow reused plus project-specific necessary adaptation:** a Runtime, Tool, serialization, or assessment failure records typed failure state immediately while retaining every already-successful upstream artifact. Nested Tool results with `ok=false` are failure receipts, not successful Tool calls. Recovery does not recompute unaffected nodes, silently change profile, or synthesize a candidate from diagnostic text. |
| Optional coding execution | SkillFlow's Tool/StructuredAction runtime provides bounded computation actions; FlowSteer's free AgentGraph keeps model, contract, relation, and topology as Canvas parameters. | **Existing boundary retained:** `coding` is only an optional `execution_mode` when the selected model and exact Tool set have a registered profile. It is not a predefined Coding Agent, role enum, required node, task router, or fixed chain/parallel topology. |

The v34 work changes runtime capability projection, continuation consistency,
artifact/assessment state handling, and live Canvas feedback only. It does not
add a fixed mathematical workflow, initial Skill, Skill retrieval/evolution,
GRPO, MACE, Bayesian inference, backward, optimizer update, or LoRA
publication. Any v34 canary or official result must be documented only after a
separately completed run with persisted trajectories and evaluator receipts.

## AIME runtime v35: exact continuation identity and typed peer provenance

v34 remains a **prepare-only** architecture condition. No v34 model/API
canary, fixed-task Accuracy, official 30-task result, or training result exists.
Integration review found four interface gaps between continuation identity,
peer artifact typing, transitive computation provenance, and atomic execution-
profile modification. v35 supersedes v34 at those interfaces; it does not
resume or relabel any v34 experiment evidence. At the time this source entry
was created, no v35 run or metric had been claimed.

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Progressive execution and bounded bidirectional communication | FlowSteer's progressive Canvas and bounded bidirectional relation semantics execute each accepted edit before returning feedback, while its typed artifact lineage retains producer/dependency identity across the graph. | **FlowSteer reused:** execute-on-edit, bounded bidirectional communication, graph revision, typed artifact lineage, incremental invalidation, Output selection, and explicit terminal action remain the common AgentGraph runtime. v35 adds no alternate workflow engine or topology template. |
| Public continuation protocol | SkillFlow's public `StructuredAction -> Observation` history, bounded continuation, and receipt persistence keep every sampled action and resulting public observation in the next execution context. | **SkillFlow reused:** a continuation receives the committed public Action--Observation history and exact receipts. Reasoning text is not inferred as an action, failed observations remain failures, and continuation does not acquire evaluator or ground-truth state. |
| Exact declaration identity for continuation | FlowSteer preserves node declaration and dependency identity across progressive execution; SkillFlow preserves the resolved public execution context across continuation. Neither upstream directly defines continuation identity for this heterogeneous free AgentGraph. | **Project-specific necessary adaptation:** a continuation is reusable only under exact declaration identity comprising `model_id + contract + execution profile`, where execution profile is the canonical `(execution_mode, allowed_tools)` tuple. A change to any component invalidates that continuation boundary; no stale continuation or silent profile substitution is admitted. |
| Typed peer artifact envelope and assessment coverage | FlowSteer's typed artifact lineage supplies source/dependency identity, and SkillFlow supplies public Action--Observation receipts. | **Project-specific necessary adaptation:** every routed `peer_draft` uses one standard typed envelope carrying source Agent, artifact identity, declaration/profile identity, completeness/status, raw public payload, and its provenance references. Assessment coverage binds to those exact peer artifacts rather than anonymous concatenated text. The envelope is public execution state, not ground truth or evaluator evidence. |
| Source-bound transitive computation provenance | FlowSteer's lineage preserves directed upstream dependencies, while SkillFlow Tool receipts preserve the public StructuredAction, Observation, and execution result associated with a Tool call. | **Project-specific necessary adaptation:** nested computation provenance is propagated transitively only with its original source Agent and source artifact binding. A downstream artifact may cite inherited computation receipts, but cannot detach a number, Tool result, or diagnostic from the artifact lineage that produced it. This is provenance closure, not answer selection. |
| Atomic execution-profile modification | FlowSteer supplies atomic parameterized Canvas edits; SkillFlow supplies exact executable mode/Tool profiles. The earlier scalar-only MODIFY parameter branch could not express a valid correlated profile transition when both fields had to change together. | **Project-specific necessary adaptation:** one `MODIFY_AGENT` action may use a dedicated `execution_profile` parameter branch that changes `execution_mode` and `allowed_tools` as one correlated pair drawn from the selected model's registered union. The pair is one atomic declaration edit, not two Canvas actions and not a widened cross-product. Model, contract, relation, Output, and topology remain unchanged unless separately edited through their existing legal actions. |
| Optional coding | SkillFlow's bounded StructuredAction/Tool execution and FlowSteer's free AgentGraph remain the runtime boundary. | `coding` remains only an optional `execution_mode` in a registered model/profile tuple. It is not a predefined Coding Agent, role enum, required branch, fixed topology, or prescribed mathematical workflow. |

v35 remains a free AgentGraph with `agent_id + model_id + free-text contract`,
Director-selected relations, one Output Agent, and explicit `FINISH`. It adds
no fixed role, Agent count, chain/parallel/debate workflow, training, Skill,
Skill retrieval/evolution, GRPO, MACE, Bayesian update, backward, optimizer
step, or LoRA publication.

The subsequently attempted v35 fixed-four condition did not complete and is
not an Accuracy denominator. Task 05 produced one correct explicit-`FINISH`
trajectory (canonical answer `65`). Task 09 then recorded a 480-second
provider timeout for the exact `qwen3.5-flash + react + calculator` request;
provider recovery projected an empty `modify_agent.mutable_fields` domain and
collection failed. Tasks 10 and 11 were not completed. These receipts establish
the recovery-domain defect but do not establish v35 Accuracy, Stable Zero, or
an official 30-task result.

## AIME runtime v36: explicit provider/profile bridge recovery

v36 addresses only the deterministic provider-recovery domain exposed by the
partial v35 run. It retains the v35 continuation, provenance, coding, evaluator,
and explicit-terminal boundaries.

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Atomic recovery edit | FlowSteer's progressive Canvas consumes one legal atomic edit and executes before returning the next feedback; its `MODIFY_AGENT` action already owns declaration-parameter changes. | **FlowSteer reused:** v36 uses the existing atomic `MODIFY_AGENT` semantics. It adds no compound action, alternate runtime, fixed role, or fixed topology. |
| Exact executable profile domain | SkillFlow binds a StructuredAction/Tool continuation to an executable model/runtime context and does not silently change that context after provider failure. v35 already projects canonical registered `(model_id, execution_mode, allowed_tools)` tuples. | **Project-specific necessary compatibility adaptation:** when no admitted repair model supports the failed Agent's current profile, `AgentWorkflowEnv._provider_repair_bridge_execution_profiles` computes the exact intersection of profiles registered for the current model and profiles registered for admitted provider-repair models. Only an exact common tuple can populate the `execution_profile` parameter domain; no Cartesian product is synthesized. |
| Two-turn explicit takeover | FlowSteer's execute-on-edit ordering makes the post-edit Canvas state the next Director observation, while SkillFlow keeps the resolved model/provider explicit in every request and receipt. | **FlowSteer/SkillFlow boundary retained:** the first atomic edit changes only the correlated `execution_mode + allowed_tools` profile and preserves model, contract, and relations. Runtime does not call the already unavailable model for that bridge edit. The next turn exposes a compatible `model_id` candidate, and only an explicit second `MODIFY_AGENT` changes model/provider. There is no silent fallback. |
| Empty recovery domain | FlowSteer's typed natural-terminal boundary applies when no legal Canvas edit remains; SkillFlow retains the provider-failure receipt rather than fabricating an execution result. | **Project-specific fail-closed correction:** if neither a directly compatible repair model nor a common bridge profile exists, `MODIFY_AGENT` is omitted instead of serializing an empty `mutable_fields` domain. Existing successful artifacts and the exact failure receipt remain available. |
| Optional coding collaboration | SkillFlow's bounded StructuredAction/Tool loop and FlowSteer's free AgentGraph already admit executable computation where registered. | **Existing boundary retained:** `coding` is one optional `execution_mode`, not a Coding Agent type, forced branch, or mathematical workflow. The Director remains free to select any admitted reasoning/ReAct/coding profile, contract, relation, Output Agent, and terminal action. |

The v36 profiles are
`config/evaluation_aime2026_runtime_v36_provider_profile_bridge_canary.yaml`
and
`config/evaluation_aime2026_runtime_v36_provider_profile_bridge.yaml`. Skill,
Skill retrieval/evolution, GRPO, MACE, Bayesian inference, backward, optimizer
updates, LoRA publication, and all training paths remain disabled.

The later v36 fixed-four canary is retained as a **partial condition**, not an
Accuracy denominator. Only task 05 and task 09 completed collection; task 10
and task 11 were not run. Task 05 ended after nine Director turns with
`canvas_action_domain_exhausted`, no explicit `FINISH`, no evaluator
invocation, and 11 configured rounds still remaining. Task 09 exercised the
full explicit provider/profile bridge (`execution_profile` edit followed by a
separate `model_id` takeover), produced a fresh local artifact, and terminated
through legal explicit `FINISH`, but its canonical prediction was
mathematically incorrect. Consequently v36 did not reach Stable Zero, did not
complete the fixed-four denominator, did not start the official 30-task
condition, and establishes no formal Accuracy.

## AIME runtime v37: artifact-consumption consistency

v37 is a source-level correction for the task-05 failure exposed by the
partial v36 canary. It keeps v36's free AgentGraph, heterogeneous registered
model/profile catalog, progressive Canvas, explicit terminal semantics, and
evaluator boundary. The change does not encode an AIME solution method or a
preferred Agent count, relation motif, or topology.

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Progressive artifact consumption | FlowSteer's `InteractiveWorkflowEnv` / `WorkflowGraph` boundary consumes one Canvas edit, executes the affected graph state, and returns execution feedback before the next edit. Output selection and terminal evaluation remain separate transitions. | **FlowSteer reused:** existing immutable artifact identity, graph revision, dependency invalidation, one atomic Canvas action per turn, Output pointer, and explicit `FINISH` remain authoritative. `SET_OUTPUT` is pointer-only and `FINISH` consumes the current fresh Output artifact; neither transition resamples an unchanged Agent. |
| Existing-artifact ordering and reachability | FlowSteer preserves prior work products across progressive edits but does not directly define the MD's provenance-assessment recovery order for a heterogeneous free AgentGraph. | **Project-specific necessary adaptation:** `AgentWorkflowEnv._dependency_closed_artifact_output_agent_ids` can project an already fresh, complete, parseable, target-blind artifact whose dependency closure contains no active execution or assessment failure. When a recoverable failed branch is outside that closure, `SET_OUTPUT` may select the existing consumer before augmentation. `AgentWorkflowEnv._artifact_assessment_existing_output_relation_candidates` then exposes one exact missing source-to-Output relation per subsequent atomic edit. Full graph validity, all-Agent reachability, freshness, assessment policy, and a separately sampled explicit `FINISH` remain mandatory; the runtime neither auto-finishes nor compares a candidate with ground truth. |
| Exact recovery fan-in parameter domain | FlowSteer's Canvas remains the final graph-admission authority, while its progressive relation editing requires sampled parameters to represent the current live domain exactly. JSON Schema `uniqueItems` alone cannot express uniqueness by an unordered endpoint pair. | **Project-specific necessary parameter mask:** for a live `ADD_SUBGRAPH` domain that requires all existing recovery ingress, `director_live_action_parameter_json_schema_text` emits an exact ordered `prefixItems` array whose source, target, and direction fields are `const`, with exact `minItems` / `maxItems` and `items: false`. This prevents duplicate-source sampling or omission of another required artifact. Ordinary relation search is unchanged. |
| Public ReAct/coding action protocol | SkillFlow's `StructuredAction` and Action--Observation continuation use exactly `arguments`, `kind`, `name`, `resource_id`, and `skill_id`; Tool calls and completion are explicit public actions with receipts. | **SkillFlow reused:** `ToolReactExecutionAdapter` retains that five-field action wire protocol, bounded continuation, public observations, and Tool receipts for both ReAct and coding execution. `coding` remains only an optional registered `execution_mode`, never a predefined Agent type, role, required branch, or fixed topology. |
| Provenance-bound completion artifact | SkillFlow's explicit COMPLETE action carries the public work product, while FlowSteer's directed communication preserves the upstream artifacts consumed by a downstream Agent. Neither upstream alone defines the project's exact provenance-assessment admission rule. | **Project-specific necessary adaptation:** only when a request actually carries routed candidate-bearing provenance under `provenance_bound_candidate_assessment_v2`, `_provenance_bound_completion_admission` requires a non-scalar public derivation plus one assessment bound to every exact source `artifact_id`, with the routed candidate copied without alteration. Missing derivation, incomplete coverage, or identity/candidate mismatch stays a typed protocol failure and cannot become a complete artifact. Ordinary reasoning/ReAct/coding completion without those routed candidate bindings retains its existing behavior. No evaluator target or correctness judgment enters this admission. |
| Bounded failed-consumer recovery | FlowSteer's `PRESERVE -> DIAGNOSE -> REPAIR -> AUGMENT` ordering preserves successful work and prefers local declaration repair before unrelated expansion. | **Project-specific necessary adaptation:** the runtime binds the one accepted provenance-consumer Agent identity to the exact immutable source artifact IDs it consumed and persists that state through checkpoints. If that exact consumer returns a candidate-bearing protocol failure, one `MODIFY_AGENT` repair is admitted, with answer-free contract repair first and only a catalog-compatible model alternative when available. A second failed execution becomes typed repair exhaustion; the same source set cannot recursively seed an unbounded chain of equivalent consumers. This is a protocol/runtime repair boundary, not a verifier role or workflow hint. |

The disjoint v37 profiles are
`config/evaluation_aime2026_runtime_v37_artifact_consumption_consistency_canary.yaml`
and
`config/evaluation_aime2026_runtime_v37_artifact_consumption_consistency.yaml`.
They retain the neutral Director prompt and free-text Agent contracts. Skill,
Skill retrieval/evolution, GRPO, MACE, Bayesian inference, backward, optimizer
updates, LoRA publication, and all training paths remain disabled.

The completed v37 fixed-four receipt did **not** satisfy Stable Zero. Tasks 05
and 10 ended with `canvas_action_domain_exhausted` and no legal explicit
`FINISH`. Task 09 reached explicit `FINISH` but produced canonical answer `7`
against target `29`; task 11 reached explicit `FINISH` with the correct
canonical answer `896`. Thus only two of four trajectories were evaluator-valid
and one of those two was correct. This diagnostic `1 / 2` subset is not the
fixed-four Accuracy and does not establish an official 30-task result.

## AIME runtime v38: Output-closure consumer recovery

v38 retains every executable condition field from v37 except its independent
condition/path/version identity. The source change is restricted to artifact
dependency scope and bounded consumer recovery; it does not modify the neutral
Director prompt, model catalog, action set, free-text contract space, relation
space, explicit terminal protocol, or evaluator.

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Output dependency scope | FlowSteer's progressive Canvas caches each node result under its input/dependency identity and invalidates only the affected downstream closure. | **FlowSteer progressive-cache boundary reused:** terminal readiness and consumer repair are evaluated against the immutable dependency closure of the proposed Output artifact. An unrelated failed or stale branch cannot be mistaken for a failure of that Output artifact, while graph validity, provenance, freshness, Output selection and a separate explicit `FINISH` remain mandatory. No artifact is recomputed merely because the Output pointer changes. |
| Exact-consumer bounded repair | SkillFlow `training/batch_inference.py::_supervisor_call_unpaused` uses one bounded same-model StructuredAction repair and preserves the original public execution context and receipt. | **SkillFlow bounded-repair precedent reused / project thin adaptation:** provenance recovery is bound to the exact downstream consumer and exact upstream `artifact_id` set. Only that consumer receives the bounded protocol repair; successful upstream artifacts are preserved and an equivalent unbounded consumer chain cannot be created. Model/provider changes remain explicit Canvas edits, never silent fallback. |
| Mixed reasoning/coding provenance | SkillFlow's five-field StructuredAction and FlowSteer's typed communication artifacts preserve public Action--Observation content and producer/dependency identity independently of an Agent's execution implementation. | **Project-specific necessary conformance test:** reasoning and optional `coding` artifacts must pass through the same source-agent, artifact-id, raw-output, freshness and candidate-assessment checks at fan-in. The test checks provenance equivalence across registered modes; it neither forces coding nor defines a Coding Agent, role, relation motif, topology, mathematical method or candidate winner. |

The disjoint v38 profiles are
`config/evaluation_aime2026_runtime_v38_output_closure_consumer_recovery_canary.yaml`
and
`config/evaluation_aime2026_runtime_v38_output_closure_consumer_recovery.yaml`.
They preserve the v37 fixed task05/09/10/11 canary, official sequential 30-task
formal population, direct reuse receipt, v18 heterogeneous thinking catalog,
minimal-neutral v14 Director prompt, seed, explicit `FINISH`, and GPU0 rollout
placement. The independent configuration regression has `8 passed`; no v38
canary or formal evaluation has run, so v38 has no Stable Zero or Accuracy
claim.

### v38 execution evidence and v39 role-neutral admission

The v38 fixed-four condition was stopped after a model-visible schema defect
was observed.  Tasks 05, 09, and 11 each produced a legal explicit `FINISH`
and a correct evaluator receipt.  Task 10 remained incomplete and therefore
had no formal prediction or evaluator result.  Its initial generic
`ADD_SUBGRAPH` admitted `role_family` values even though the AIME protocol is
defined only by `agent_id + model_id + free-text contract`; those values were
then serialized back into Canvas and recovery observations.  Accordingly, the
three completed results are diagnostic coverage (`3 / 4`), not a fixed-four
Accuracy, and v38 is not Stable Zero.

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Role-neutral Agent declaration | FlowSteer's Canvas keeps operator parameters inside the live action domain; the project AIME adapter maps this to free Agent declarations. SkillFlow's Supervisor/Executor boundary does not introduce a generic Agent role enum. | **Project-specific thin AIME admission fix:** the live free-contract domain now publishes whether `role_family` is admitted. AIME publishes `false`, so constrained ADD declarations cannot sample it and the declaration parser rejects it. The global Agent schema remains unchanged for HotpotQA/TriviaQA semantic protocols and the legacy Format-Agent boundary. |
| Canvas fail-closed boundary | FlowSteer validates a sampled edit before mutating or executing the candidate Canvas. | **FlowSteer validation boundary reused:** raw/manual AIME `ADD_AGENT`, `ADD_SUBGRAPH`, or `MODIFY_AGENT` actions carrying `role_family` are rejected with typed `role_family_forbidden` feedback before graph revision or Runtime execution. Other dataset protocols retain their existing behavior. |
| Policy-state isolation | FlowSteer's next Director observation serializes the accepted graph and live parameter domains. | **Necessary isolation:** because AIME cannot accept role metadata, no role label can be persisted, echoed as a recovery field, or become a policy-active orchestration prior. Agent responsibility remains solely in the model-authored free-text contract. |

The independent v39 profiles are
`config/evaluation_aime2026_runtime_v39_role_neutral_schema_canary.yaml` and
`config/evaluation_aime2026_runtime_v39_role_neutral_schema.yaml`.  They retain
v38's fixed task05/09/10/11 population, official sequential 30-task population,
v18 heterogeneous thinking catalog, prompt v14, seed, evaluator, direct reuse,
GPU0 placement, explicit terminal semantics, and disabled training/Skill/MACE/
Bayesian paths.  Prepare-only validation reports exactly 4 and 30 samples.

## AIME runtime v45: provider/protocol admission and formal same-30 result

v45 is the formal successor of the role-neutral v39--v43 artifact/runtime work.
It preserves the unified orchestration core and introduces no task-specific
Agent role, fixed topology, solver/verifier workflow, answer lookup, training,
or Skill prior.

| Boundary | Upstream source | Reuse / necessary project adaptation |
|---|---|---|
| Progressive Canvas and immutable artifacts | FlowSteer's progressive Canvas executes each accepted edit before the next Director observation and invalidates only affected downstream state. | **FlowSteer reused:** graph revision, immutable artifact identity, dependency-aware reuse, pointer-only `SET_OUTPUT`, and explicit `FINISH` remain unchanged. |
| Structured Action--Observation continuation | SkillFlow's Supervisor/Executor path uses a bounded StructuredAction continuation and preserves provider/model receipts. | **SkillFlow reused:** ReAct/coding retain bounded continuation, typed Tool receipts, explicit completion, and no silent provider substitution. |
| Incomplete artifact recovery | FlowSteer preserves successful upstream work before local repair; SkillFlow treats incomplete StructuredAction completion as a recoverable execution failure. | **Project-specific thin adaptation:** `_incomplete_artifact_repair_domain` exposes only executable declaration repairs for the failed Agent and preserves fresh upstream artifacts. |
| Provider versus provenance-protocol recovery | SkillFlow requires the model/provider used by execution to remain explicit; FlowSteer's action mask must match Canvas admission. | **Project-specific thin adaptation:** `_provider_repair_agent_ids`, `_base_model_admissible_action_types`, `_model_admissible_modify_agent_ids`, and `_artifact_assessment_consumer_admission_issue` give a parameter-feasible explicit provider repair precedence over provenance-protocol repair. The same validated atomic action is accepted by the raw Canvas path; runtime never changes model implicitly. |
| Role-neutral mathematical task | The project MD defines a free AgentGraph `G=(V,E,o)` and AIME nodes as free-text contracts; neither FlowSteer Canvas nor SkillFlow Supervisor requires a generic Agent role enum. | **Project-specific compatibility boundary retained:** AIME omits `role_family`; all 55 reported nodes remain `unspecified`. Responsibility is expressed only by free-text contract. |
| AIME output and evaluator | SkillFlow AIME uses target-blind integer extraction/canonicalization and exact Accuracy. | **SkillFlow-aligned adapter retained:** AgentGraph uses `skillev.integer.target-blind-extraction.v2.2`; ground truth is evaluator-only and parsing cannot repair or recompute an answer. |

The final profile is
`config/evaluation_aime2026_runtime_v45_provider_protocol_admission.yaml`.
Its task10-only Stable Zero canary reached legal explicit `FINISH` with canonical
answer `156`. The formal run then evaluated the same fixed 30 AIME 2026 tasks
once: AgentGraph `24 / 30 = 80.00%` strict Accuracy, Direct `19 / 30 = 63.33%`,
27 explicit finishes, 3 terminal failures, 0 AgentGraph parsing failures, and
0 collection failures. Because the generated report records
`protocol_equivalent_to_direct=false`, the `+16.67` percentage-point difference
is descriptive rather than a strict causal estimate. Full evidence is under
`artifacts/aime2026_runtime_v45_provider_protocol_admission/evaluation/`; the
Chinese root-cause report is
`reports/aime2026_runtime_v45_provider_protocol_admission/root_cause_report_zh.md`.
