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
