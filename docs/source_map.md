# Dataset adaptation source map

## HealthBench v2.43 — declaration feedback and candidate refinement

- `rollout_collector.py`: retain the existing SkillFlow reasoning/action phase
  receipts and FlowSteer rejected-Canvas-turn continuation. The project-specific
  free-contract declaration validator runs before final relation generation;
  its actual error must reach the next Director input, rather than parsing a
  declaration as a complete edit and reporting an unrelated missing field.
  Partial declarations remain non-executable; no relation is synthesized.
- `react_execution.py`: reuse `StructuredAction.from_value`'s existing precise
  parser exceptions and SkillFlow's invalid-Action Observation/continuation.
  Retain the actual failure message for the next Agent turn, rather than
  treating every malformed action as an envelope wrapper. No parser relaxation,
  automatic JSON repair, new generation attempt or budget increase is added.
- `agent_workflow_env.py`: the existing public ReAct error summary and step
  Observation projection also preserve that error message, so the Director's
  next input receives the same diagnosis as the failed Agent. Existing bounded
  summaries are retained; sampled malformed JSON is not replayed as context.
- `healthbench_candidate_skills_v243.yaml`: reuse the v2.41 candidate helper and
  MD optional prompt-prior boundary. Three general procedures cover open
  questions before evidence, receipt-based repair, and final source consistency.
  No medical answer, fixed role or required topology is introduced.
- Versioned v2.43 configs directly reuse v2.42's model pool, sampling, tools,
  per-model token budget, 900-second task deadline and 360-second Agent limit.
  No training or Skill publication. Because rejection feedback also changes,
  comparisons are combined-condition observations, not a Skill-only effect.


## HealthBench v2.42 — per-model budgets and observed-source schema

- `react_execution.py`: directly retains SkillFlow-style StructuredAction and
  the existing HealthBench action union. The opt-in budget reads the existing
  ModelSpec metadata rather than replacing every Executor budget with the
  Director budget. A state-aware completion hook delegates to existing schemas.
- `healthbench_evidence_adapter.py`, `healthbench_clinical_react.py`: reuse actual
  Tool Observations, `_successful_search_evidence` and routed receipts; constrain
  only complete source-metadata tuples in the existing completion schema.
  Strict span validation still runs; no answer, quote or clinical claim is filled in.
- `openai_gateway.py`: add a small receipt from the existing payload passed to
  `_post_json`, following SkillFlow's response_schema interface. This records a
  client request, not proof of provider-side schema enforcement.
- `agent_workflow_env.py`: reuse `_provider_repair_catalog_domain`, the same
  repair target mask and admission checks; opt in to retaining compatible
  same-provider alternatives on a 429 of unknown scope. No new router or
  provider model is introduced, and the Director still chooses the edit.
- `train_agentgraph_smoke.py` and versioned configs: thin opt-in forwarding;
  reuse the existing AgentRuntime timeout setting (360 seconds per invocation,
  not per model turn). Task deadline remains 900 seconds.
- Candidate profile v2.42 reuses the v2.41 helper/collector and MD §§10–11
  boundaries. It remains unvalidated/rejectable; no ACTIVE publication or training.

See `docs/healthbench_v242_run_conditions.md` and
`docs/healthbench_v242_skill_changes.md` for the precise experimental changes.

## HealthBench v2.41 — inference-only bounded recovery and candidate conditions

| Changed module | Concrete source / reuse | Necessary adaptation |
| --- | --- | --- |
| `director.py`, `rollout_collector.py` | SkillFlow `runtime/bounded_agent.py::execute_turn`, `rollout/engine.py` two-phase reasoning/action; FlowSteer `workflow_env.py::step`; existing project artifact projection | Opt-in token-budgeted Director observation, with full original task/latest graph preserved and separate action-generation reserve. Complete runtime feedback remains in trajectory. |
| `react_execution.py`, `healthbench_evidence_adapter.py`, `healthbench_clinical_react.py` | SkillFlow bounded-agent invalid-action feedback and completion validation; project `_successful_search_evidence`, `_routed_evidence_receipts` and existing ReAct continuation | Return first failing artifact item and bounded real source metadata/excerpt; reject unobserved source IDs while retaining public conversation references. Strict evidence validators and tool limits unchanged. |
| `healthbench_candidate_skill_profile.py`, candidate YAML | Project `run_joint_qa_mace_skill.py::_prompt_condition` and `LiveSmokeBackend.collect(prompt_priors=..., forced_probe=True)`; MD §§10–11 | Three optional, unvalidated orchestration priors. No fixed medical roles, answer content, ACTIVE publication or posterior/optimizer update. |
| `train_agentgraph_smoke.py`, evaluation runners | Existing inference-only factory, shared HotpotQA collector and official HealthBench evaluator | Thin opt-in configuration/argument forwarding, candidate profile frozen in manifest, separate condition directories and explicit candidate labels. No new runner or evaluator. |

Source locations and candidate boundary: `docs/healthbench_v241_skill_plan.md`.
The input projection is not training-context reconstruction: exact transmitted
prompts continue to be recorded and this experiment is inference-only.
Candidate-conditioned results are not silently promoted to natural Skill-off
baselines or evidence-gated ACTIVE Skill results.

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

---

# HealthBench Professional initial-adaptation source map

This section records the sources selected for the HealthBench Professional
inference/evaluation adaptation. The implementation, two-case Stable Zero
smoke chain, and complete 525-case request execution are validated. The
full panel contains 503 evaluator-valid AgentGraph completions and 22
receipt-confirmed `max_rounds` terminal failures. The machine-readable and
concise reports are
`reports/healthbench_professional_official_v1/evaluation_report.json` and
`reports/healthbench_professional_official_v1/evaluation_report.md`.
The source order
is the project MD contract, SkillFlow/SkillEval execution boundaries,
FlowSteer's progressive Canvas boundaries, and only then the minimum
HealthBench Professional compatibility layer.

The exact user-provided references rechecked for this adaptation are:

- `/ssd1/iclr/1/.codex/attachments/d26515b1-d405-4a96-86f9-a611b9a8385c/FlowSteer_MACE_Bayesian_Skill_Design.md`;
- `/ssd1/iclr/1/.codex/attachments/d53b2d0d-5a02-4ecb-9b23-1769e59731a5/SkillFlow.pdf`; and
- `/ssd1/iclr/1/.codex/attachments/4efb630d-b545-4a0d-beac-28a3aa32d453/FlowSteer__Towards_Agents_Designing_Agentic_Workflows_via_Reinforced_Progressive_Canvas_Editing__3_(1).pdf`.

They are source references, not runtime instructions. Training, MACE,
Bayesian updates, and Skill evolution described by those sources remain
disabled in this evaluation-only round.

### Official public data and reference evaluator

| Source | Reused contract | Local boundary |
| --- | --- | --- |
| `openai/healthbench-professional` public dataset and the official `assets.zip` file `healthbench_professional_eval.jsonl` | The only public split is `test`, containing exactly 525 records. The checked local source is `/ssd1/iclr/2/datasets/healthbench_professional/healthbench_professional_eval.jsonl`. | The dataset adapter must preserve all 525 IDs and source order and must not invent train/development examples from this public test population. |
| Public HealthBench Professional record schema | Every row has `id`, `conversation`, `rubric_items`, `use_case`, `type`, `difficulty`, `specialty`, `physician_response`, and `canary_string`; `conversation` is an object with a `messages` array; each rubric item has `criterion_text` and `points`. | The full `conversation.messages` sequence becomes task input. Rubrics, the physician response, the canary, and analysis metadata remain evaluator/report-only. |
| OpenAI `simple-evals` commit `652c89d0ca9df547706735883097e9537d40dc47` | `healthbench_eval.py::{RubricItem,calculate_score,calculate_length_adjusted_score,HealthBenchEval.grade_sample,_aggregate_get_clipped_mean}` define the public reference grading and aggregation path. `simple_evals.py` supplies the Professional option bundle. | The project may thin-port these functions into the existing evaluator receipt interface; it must not substitute EM, token F1, string Accuracy, or a physician-response similarity score. |
| HealthBench Professional paper/evaluation protocol | The Professional primary score uses rubric-level grading, the reference grader `gpt-5.4-2026-03-05` with low reasoning effort, per-example length adjustment with center `2000` characters and penalty `0.0147` per 500 characters, and clipping of the final mean to `[0,1]`. | The paired runner must persist both unadjusted rubric score and length-adjusted score plus grader errors. A different grader condition is a local diagnostic, not a paper-comparable Professional score. |

OpenAI's internal production evaluator is not published in this repository.
Therefore the public implementation above is described as a
**HealthBench Professional reference-compatible evaluator**, not as the
unavailable internal evaluator. If the exact reference grader or its low
reasoning setting cannot be invoked, formal reference-compatible evaluation
is blocked rather than silently relabelled.

The reference score for one response is computed from rubric judgements as
follows: positive rubric points form the denominator; every rubric whose
`criteria_met` value is true contributes its signed points, so triggered
negative criteria reduce the numerator. Length adjustment is applied to that
per-example score. The reported primary aggregate is the clipped mean of the
length-adjusted per-example scores. Rubric text and grading responses never
cross the evaluator boundary.

### SkillFlow / SkillEval runtime and private-evaluator boundaries

The public SkillFlow code does not contain a HealthBench Professional task
adapter. The downstream SkillEval production tree provides the closest
reusable private-session and trusted-evaluator interfaces:

| Production source | Reused boundary | HealthBench Professional decision |
| --- | --- | --- |
| `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/skillev/rollout/session.py::{UnskilledRolloutSessionBundle,RolloutSessionBundle}` | An environment/session and its terminal evaluator are bundled explicitly; evaluator truth is not model state. | Reuse the separation semantically. This round is unskilled, so no retrieved-skill context is injected. |
| `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/skillev/runtime/bounded_agent.py::BoundedAgent.execute_turn` | Execute an already generated action, commit one measured public observation, and keep scoring authority outside the Agent. | Reuse the execution/observation boundary through the existing project runtime; do not move rubrics into Agent feedback. |
| `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/skillev/rollout/engine.py::RolloutEngine.run` | Bounded multi-turn rollout with an explicit terminal submission and a separate terminal evaluator. | Reuse the rollout/terminal separation through the project's existing collector and trajectory records; no SkillFlow training loop is activated. |
| `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/packages/private-evaluation/src/skillev_private/benchmarks/protocol_v10_official.py::{HealthBenchGrade,HealthBenchOfficialGrader,HealthBenchNativeBackend.evaluate_native}` | Rubric truth stays behind the grader protocol; only task ID and candidate answer cross into grading; failure is surfaced as evaluator failure. | Reuse the private-evaluator contract. The open public rubrics are still evaluator-only in this project and never enter the Director/Agent transcript. |
| `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/packages/private-evaluation/src/skillev_private/benchmarks/protocol_v10_workers.py::{PrivateJSONWorker,OfficialHealthBenchProcessGrader.grade}` | A private worker owns rubric data and the grader client and returns typed score evidence. | Use as the source for fail-closed grader isolation and receipt fields. The worker itself is not copied because the public reference evaluator grades each rubric and exposes richer rubric-level receipts. |
| `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/packages/private-evaluation/src/skillev_private/benchmarks/protocol_v10_evaluator.py::{ProtocolV10NativeResult,ProtocolV10TerminalEvaluator.evaluate}` | Trusted native fields are projected only after terminal evaluation. | Preserve the evaluator-only projection; training reward projection is outside this round. |
| `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/packages/private-evaluation/src/skillev_private/benchmarks/protocol_v10_population.py::{ProtocolV10TrainingSessionFactory,ProtocolV10PopulationSessionRegistry}` | Session factories bind public rollout tasks to private evaluators without copying private truth into tasks. | Source reference only. Protocol-v10 population building and training-session materialization are not enabled for this public 525-case evaluation. |

### FlowSteer progressive Canvas mapping

The concrete upstream FlowSteer revision `1c9f2ab` reference points remain:

- `src/interactive/workflow_env.py::InteractiveWorkflowEnv.step` for one
  admitted edit followed by execution and observable feedback;
- `src/interactive/workflow_env.py::InteractiveWorkflowEnv._execute_workflow`
  for workflow execution after an accepted Canvas change;
- `src/interactive/workflow_env.py`'s explicit `ActionType.FINISH` branch for
  terminal admission;
- `src/interactive/workflow_graph.py::WorkflowGraph` for graph state; and
- `src/interactive/workflow_builder.py::{TurnRecord,Trajectory,InteractiveWorkflowBuilder.run_loop,InteractiveWorkflowBuilder.run_loop_async}`
  for action/feedback turns and trajectory materialization.

The project already adapts these boundaries in
`src/interactive/agent_graph.py::AgentGraph`,
`src/interactive/agent_workflow_env.py::{AgentWorkflowEnv.step,AgentWorkflowEnv.execute}`,
`src/interactive/rollout_collector.py::AgentGraphRolloutCollector.collect`, and
`src/interactive/records.py::{TurnRecord,TrajectoryRecord}`. HealthBench
Professional must attach to these existing calls. It does not add another
Canvas, runtime, communication mechanism, or trajectory type.

The MD identity `Agent = agent_id + model_id + free-text contract` is the
graph-semantic identity used here. The shared serializer also retains optional
execution/receipt metadata (`role_family`, `allowed_tools`, `execution_mode`,
`artifact_type`, and `completion_condition`). Those fields do not define Agent
classes or topology semantics. In this HealthBench condition, every final node
used `execution_mode=reasoning` and `allowed_tools=[]`; any `role_family` label
was free text authored by the Director, not a predefined medical role.

### HealthBench Professional implementation classification

| Boundary | Classification | Required state |
| --- | --- | --- |
| Free AgentGraph, six Canvas actions, incremental execute/feedback loop, unique Output Agent, model routing, relation execution, and trajectory serialization | **Direct reuse** of the current project core and its FlowSteer-derived boundaries | No HealthBench-specific mutation semantics and no fixed topology. |
| Model gateway and Agent execution receipts | **Direct reuse** of the current project/SkillFlow-aligned interface | Preserve provider errors, token counts, latency, node/model identity, actual communication, and termination. |
| 525-row JSONL loader and full-conversation projection | **Necessary task-specific adaptation** | Convert the official schema to `TaskRecord` without changing conversation contents or exposing evaluator-only fields. |
| Rubric item conversion (`criterion_text` to the reference evaluator's criterion field), rubric-level grading, signed-point scoring, length adjustment, and final clipped aggregation | **Necessary task-specific adaptation** over OpenAI `simple-evals` commit `652c89d` | Return reference-compatible `overall_score_length_adjusted` and unadjusted rubric receipts; never EM/F1/Accuracy. |
| Direct versus AgentGraph paired runner configuration | **Necessary wiring** in the existing evaluation runner | Same 525 IDs, generation/model condition, Tool condition, evaluator, and grader condition. |
| Receipt-backed failure-demo reporting | **Necessary reporting adaptation** over the completion runner's `wrong_demos.jsonl` diagnosis and the existing multidataset CommunicationEnvelope/ReAct/Tool extractors | `scripts/report_healthbench_failure_demos.py` joins frozen paired/trajectory receipts with the evaluator-only case store. It writes a tracked redacted taxonomy plus an ignored evaluator-private full trace; it performs no model, Tool, grader, or training call. |
| Versioned HealthBench best-profile descriptor and current pointer | **Project-specific evaluation evidence metadata** | Selects only among completed same-split, same-denominator, same-evaluator AgentGraph conditions. It points to the existing executable official-v1 config and formal receipts; it does not add an orchestration module, implicit runner default, or new evaluated condition. |
| Medical retrieval or Web browsing | **Not enabled** | The public Professional base protocol declares no Tool interface. Direct and AgentGraph both run with an empty task Tool condition. Existing MedRAG code is not activated. |
| Doctor, Researcher, Reviewer, Verifier, or other medical role classes/templates | **Not implemented and prohibited as an initial prior** | Agents remain `agent_id + model_id + free-text contract`; the Director chooses Agent count, contracts, models, and relations from feedback. |
| GRPO, optimizer/backward, LoRA update, MACE, Bayesian posterior/EVSI, Skill retrieval, Skill injection, or Skill evolution | **Not enabled in this round** | No training or learning state may change. |
| OpenAI internal Professional evaluator | **Unavailable** | Do not claim official internal-evaluator equivalence; use the pinned public reference implementation or report the grader condition as blocked. |

### Model-visible and evaluator-only projections

The model-visible projection contains the complete ordered
`conversation.messages` and no benchmark answer key. `id` may be retained as
a receipt identity but is not task content. The following stay outside every
Director prompt, Agent contract, Agent input, Canvas feedback, Tool
observation, recovery message, and model-visible trajectory:

- `rubric_items`;
- `physician_response`;
- `canary_string`; and
- `use_case`, `type`, `difficulty`, and `specialty`, unless a later official
  protocol explicitly authorizes a particular metadata field as model input.

The terminal submission is the complete assistant response. No AIME-style
short-answer extractor, QA answer tag, answer compression, physician-response
rewrite, or task-specific Formatter is applied. The sole terminal processing
is whatever minimal transport normalization the pinned reference evaluator
requires.

Full failure demos follow `docs/failure_demo_reporting_protocol.md`. Complete
conversations, rubrics, physician completions, candidate responses, Director
prompts, Agent inputs/outputs, communication bodies, and grader explanations
are written only under the ignored `artifacts/.../evaluator_private/`
boundary. The tracked report retains aggregate taxonomy, task IDs, metrics,
topology, termination, and redacted receipt summaries only.

The preparation manifest's `model_visible_fields` label describes fields kept
in the public transport record; it does not mean every listed routing field is
concatenated into a model prompt. The executed boundary is recorded by
`run_manifest.json::model_visible_task_boundary`: the prompt source is
`TaskRecord.question`, which is the lossless rendering of the complete
conversation and is restored to native roles by the gateway. `task_id`,
`evaluator_route`, and `evaluator_source_id` are routing, private-evaluator
join, and receipt metadata. The top-level `conversation` copy is retained for
source preservation and is not injected a second time.

### Status boundary

Source/schema/evaluator mapping, code integration, and the two-case Stable Zero
chain are complete. Persisted receipts show 2/2 valid Direct grades, 2/2 valid
AgentGraph grades, 2/2 legal explicit `FINISH`, complete trajectories, and
successful bounded recovery from transient grader-provider failures. These
two cases are a chain-validation canary, not a benchmark estimate.

The complete public-test request population was executed under one frozen
condition. Direct completed and graded 525/525. AgentGraph produced 503 valid
reference-compatible grades with 503 explicit `FINISH`; the other 22
trajectories exhausted 20 Director turns without legal `FINISH` and are
reportable `max_rounds` terminal failures. Current operational/evaluator
failures are zero. With the full requested denominator of 525, the strict
raw/length-adjusted scores are 18.97%/19.17% for Direct and 22.65%/20.24% for
AgentGraph; the strict length-adjusted delta is +1.07 percentage points.
Among the 503 valid AgentGraph grades, the valid-only length-adjusted score is
21.12%. The result remains reference-compatible, not internal-official, and
the 22 non-submissions are not fabricated as valid grades. Consequently, the
two-case Stable Zero canary is confirmed, but the formal 525-case manifest's
all-task Stable Zero criterion is false because 22 workflows did not finish.

### HealthBench inference-loop v2 source classification

The versioned condition
`config/evaluation_healthbench_professional_inference_loop_v2.yaml` changes
only inference-time control and decoding. It preserves the same 525 public-test
tasks, full-conversation adapter, empty Tool condition, local Qwen3.5-9B arm,
and pinned HealthBench reference evaluator as official-v1.

| v2 boundary | Source and classification |
| --- | --- |
| Exact state-conditioned `SET_RELATION` candidates, one accepted Canvas edit followed by execution/feedback, and explicit `FINISH` | **Direct reuse** of the current FlowSteer-derived Canvas validation and progressive execution boundaries. Generic scalar `ADD_AGENT` binding to the existing v3 live domain is a **minimal compatibility adaptation**; it adds no medical role or topology. |
| Local SGLang `repetition_penalty` request field and receipt | **Thin SkillFlow compatibility adaptation** of the upstream Qwen3.5/SGLang request boundary. The configured `1.05` value is a project inference condition, not a claimed upstream optimum and not a training reward. |
| Task-scoped component reuse for an identical effective input | **Necessary AgentGraph adaptation** of FlowSteer's legacy per-environment node-cache boundary (`workflow_builder.py` / `vllm_workflow_generator.py`). It is not claimed as an upstream heterogeneous component cache. ReAct, coding, Tool use, failure continuation, empty output, and `finish_reason=length` are excluded. |
| Plain-language free-text contract admission | **Project-specific Canvas admission guard**. It rejects opaque labels and exact duplicate execution declarations without defining Doctor, Reviewer, or any other role class. |
| Current-revision rejected-relation exclusion and finish-only action mask after full terminal admission | **Necessary state-conditioned action-mask wiring** over existing Canvas candidates and `finish_admissibility`; graph validation and explicit `FINISH` remain authoritative. |
| SGLang `max_running_requests=null` server receipt | **Necessary version-compatible preflight adaptation** for SGLang's supported backend-default server argument. It is opt-in for this condition and records `backend_default`; it does not change the running service. |

This condition does not enable GRPO, backward, optimizer update, LoRA, MACE,
Bayesian inference, Skill retrieval/evolution, medical retrieval, or any new
model-visible evaluator field.

The 525-task inference evidence for this condition is complete: every raw
AgentGraph trajectory reached explicit `FINISH`. Reference grading is only
partially complete because the configured grader account returned an
insufficient-quota error after 510 Direct and 488 AgentGraph evaluations.
The report preserves those counts separately from inference completion and
does not promote the partial condition to the HealthBench best-profile.

### HealthBench Professional Artifact communication v3 source classification

The opt-in profile
`agent_graph.artifact_communication_profile=producer_context_exact_dedup_v1`
keeps the unified AgentGraph and the HealthBench task/evaluator adapters
unchanged.

| v3 boundary | Source and classification |
| --- | --- |
| Relation-scoped Artifact routing and incremental execution | **Direct reuse** of the current FlowSteer-derived `AgentRuntime._upstream` and progressive Canvas execution boundary. No broadcast channel, medical role, or fixed topology is added. |
| Producer contract/model/execution mode/completion condition, Artifact version, provider finish reason, and Tool receipt provenance | **Necessary typed-envelope adaptation** over the existing `UpstreamMessage`. The fields are existing Agent declarations or measured Runtime receipts; no model-generated metadata is fabricated. |
| Exact duplicate envelope suppression | **Thin adaptation** of SkillFlow's exact repeated Action--Observation reuse rule. Only a byte-identical serialized envelope with the same non-empty Artifact version is suppressed in one model input. Different producers, versions, bodies, Tool receipts, or contracts remain visible; no semantic-similarity deletion is used. |
| Version-aware component reuse | **Necessary adaptation** of FlowSteer's `operator + inputs` node-cache boundary. Under v3 only, a routed Artifact version change invalidates downstream reuse; legacy profiles retain the prior transport-version-insensitive behavior. |
| Reciprocal `peer_draft` and cache-reuse communication in Canvas feedback | **Receipt projection adaptation** over already persisted Runtime provenance. The Director receives compact 160-character previews plus versions and execution receipts; full Artifact bodies remain in the trajectory and are not duplicated into Canvas feedback. |
| HealthBench Direct evaluator-only retry | **Evaluation wiring adaptation**. A matching existing Direct response remains frozen when its reference grader receipt is invalid, so later attempts re-score the same text instead of regenerating the comparator. |
| Failed HealthBench evaluator preflight diagnosis | **Evaluation observability adaptation**. The runner preserves the grader error type/message and provider error type/status in its bounded manifest error and a structured failed `preflight_receipt.json`. It does not include any benchmark conversation/rubric/reference response, expose private cases to the model, weaken the evaluator gate, or continue into benchmark generation after a failed preflight. |

The v3 condition continues to use `repetition_penalty=1.05`. This is an
inference decoding parameter, not a reward. Skill retrieval/evolution, GRPO,
LoRA, backward, optimizer updates, MACE, Bayesian updates, and medical Tool use
remain disabled.

The fixed 525-task v3 selection is byte-for-byte aligned with v1 and v2, but
the 2026-08-29 live canary stopped before benchmark generation because the
pinned grader returned HTTP 403 `insufficient_quota` on all three bounded
provider attempts. Therefore v3 has no valid official score yet. The completed
v1 result remains the only full-denominator comparison; v2 remains explicitly
partial-evaluator evidence.

### HealthBench Professional retrieval-enabled paired condition source map

This is an explicitly separate **retrieval-enabled diagnostic protocol**. It
does not alter or replace the official/reference-compatible no-tool conditions
above, and it must not be described as an official HealthBench Professional
baseline. A repository search found no callable Web-search backend in the
checked FlowSteer or SkillFlow runtime. Consequently, the current executable
fallback is the already provisioned, frozen SkillFlow MedRAG textbook BM25
corpus, not a simulated Web service:

- runtime root:
  `/ssd1/iclr/.private/skillflow-resources/medrag-textbooks-runtime`;
- corpus identity: `MedRAG/textbooks`, recorded source revision
  `9c72838920a1323ffa867467d3f7aa7b36b0f994`;
- frozen files: `all_chunks.jsonl` and `bm25_index.pkl`; and
- checked corpus size: 125,847 chunks.

The corpus records contain public `id`, `title`, and text fields but no source
URL. The adapter therefore persists `document_id`, `title`, `chunk_id`, rank,
BM25 score, matched terms, and returned text in the Tool result/receipt; it
does not fabricate URLs or bibliographic provenance that the corpus does not
provide.

| Source | Reused boundary | Local retrieval-enabled boundary |
| --- | --- | --- |
| `/ssd1/iclr/2/SkillFlow/training/task_prompts.py::MULTI_HOP_QA` | Search with specific entities; after `[NO_MATCH]` or `[REPEATED]`, reformulate the query with synonyms. | **Direct semantic reuse** as model-driven ReAct query reformulation guidance. A standard synonym or expanded abbreviation may be authored by the acting model, but the adapter does not invent an automatic medical synonym table. |
| `/ssd1/iclr/2/SkillFlow/training/environment.py::{_search_passages,_search_external_corpus}` | BM25-backed retrieval, explicit no-match/repeated observations, and ranked evidence returned to the acting Agent. | **Thin compatibility reuse** through `src/interactive/healthbench_tool_adapter.py::{FrozenMedRAGBM25Corpus,open_healthbench_medrag_tool_registry}` and the existing Tool registry. |
| FlowSteer revision `1c9f2ab`, `workflow_env.py::{InteractiveWorkflowEnv.step,InteractiveWorkflowEnv._execute_workflow}` | One admitted Canvas edit followed by execution and observable feedback. | **Direct reuse** through the existing AgentGraph Canvas/runtime; enabling a Tool does not add a medical topology, role class, or alternate orchestration core. |
| Existing `src/interactive/react_execution.py::ToolReactExecutionAdapter` and `src/interactive/tool_runtime.py::{ToolRegistry.ainvoke_with_receipt,ToolReceipt}` | Per-Agent `Thought -> Action(tool) -> Observation -> Thought -> Final`, exact schema validation, and measured Tool receipts. | **Direct reuse**. ReAct remains an Agent execution mode, not an Agent role. |
| Existing `src/interactive/records.py::{TurnRecord,TrajectoryRecord}` and v3 Artifact communication | Canvas actions, Agent outputs, communication, Tool evidence, terminal state, and evaluator receipts remain reconstructible. | **Direct reuse**; retrieved chunks travel as receipt-backed Artifacts rather than an unrecorded knowledge channel. |
| `src/interactive/healthbench_tool_adapter.py` public corpus projection | Preserve existing search output and corpus provenance. | **Necessary adapter change**: carry source `id` and `title` into every ranked chunk and describe entity-specific/synonym-pivot query behavior in the Tool schema. |
| Existing `QARetrievalReactExecutionAdapter._state_conditioned_action_domain` | Mask Tool actions that cannot advance the measured public state and constrain the next generation to completion when evidence admission is satisfied. | **Thin HealthBench adaptation** in `HealthBenchMedRAGReactExecutionAdapter`: a successful non-empty textbook result makes `complete` the sole next action; an empty/error result admits a distinct reformulated query; the exhausted Tool budget also makes completion sole. Exact prior queries are rejected within the same Agent execution. This changes only ReAct action admission, not Agent roles, topology, medical content, or answer selection. |
| SkillFlow `scientific_sampling.py::{ScientificSamplingCoordinate,derive_generation_seed}` and the existing `LiveSmokeBackend.collect` coordinate construction | Per-task scientific sampling uses the frozen base seed, schedule purpose, ordered task identity, rollout ordinal, and anchor ordinal. | **Direct reuse plus evaluation wiring**: Single-Agent ReAct and free AgentGraph now receive the same coordinate protocol. Every ReAct model-call receipt is checked against the derived step seed before it can be resumed or reported. |
| Existing `ToolReactExecutionAdapter._state_conditioned_action_domain` and Tool dispatch boundary | The model-visible action schema reflects the latest public Action--Observation state. | **Necessary Runtime admission completion**: the same admitted Tool-action set is rechecked immediately before dispatch. A provider output that bypasses constrained decoding is retained in the public trace as `state_action_not_admitted`, but consumes no Tool budget and creates no ToolReceipt. |
| Existing completion-runner task checkpoint and version receipts | Resume only an artifact generated under the frozen evaluation condition. | **Necessary paired-evaluation adaptation**: `direct_generation_identity` binds the model catalog/provider model, Direct contract and completion condition, scientific sampling coordinate, Tool version/catalog, and MedRAG source revision/runtime limits. AgentGraph resume additionally revalidates its raw `director_sampling` receipt against the current seed, schedule purpose, task coordinate, and anchor. The paired label is emitted only when every executor call has a valid ReAct scientific-sampling receipt; a reasoning-only or mixed-mode Graph remains a valid architecture result but is reported as a separate-protocol comparison. |

No synonym/abbreviation lexicon exists in the checked MedRAG resource or the
referenced SkillFlow BM25 implementation. This condition therefore does not
claim deterministic automatic synonym expansion, construct paraphrased
HealthBench examples, or modify benchmark conversations. Any later curated
lexicon would require its own identified source and versioned receipts.

The paired comparison must expose exactly the same frozen MedRAG Tool catalog,
corpus revision, model catalog, generation settings, task IDs, and reference
evaluator to both arms. Since a plain one-shot Direct call cannot invoke the
Tool, the retrieval-enabled comparator is labelled **Single-Agent
ReAct+MedRAG**, compared with **free AgentGraph+MedRAG**; it is not silently
reported as the no-tool Direct baseline. The existing no-tool Direct versus
AgentGraph evidence remains a distinct protocol and artifact directory.

The HealthBench public test conversation may supply the clinical query to an
Agent, but `rubric_items`, `physician_response`, `canary_string`, grader output,
and reference responses remain evaluator-only. Neither those fields nor a
HealthBench case/answer database may enter a Tool query, corpus, Observation,
Director feedback, Agent Artifact, or trajectory visible to a model. This
retrieval condition adds no training, GRPO, LoRA, MACE, Bayesian update, Skill
retrieval/injection/evolution, or learned medical memory.

### HealthBench authoritative retrieval v1

`config/evaluation_healthbench_professional_authoritative_paired_gpt54_rubric_v1.yaml`
defines a new retrieval-enabled condition. It does not overwrite the
MedRAG-only condition or the reference-compatible Tool-free conditions.

| Module | Source classification | Exact boundary |
| --- | --- | --- |
| Frozen textbook search | **Direct SkillFlow reuse** | `FrozenMedRAGBM25Corpus` continues to use SkillFlow `training/environment.py::{_load_external_corpus,_search_external_corpus}` tokenization, BM25 constants, top-k projection, resource identity, and lifecycle. |
| Tool registration, dispatch, timeout, schema validation, and receipts | **Direct SkillFlow-compatible project reuse** | Existing `ToolRegistry`, `ToolCapability`, `ToolReactExecutionAdapter`, and `ToolReceipt` remain authoritative; no second Agent runtime is introduced. |
| Incremental AgentGraph execution and Artifact routing | **Direct FlowSteer-derived reuse** | Existing Canvas edit → execution → feedback, free Agent contracts, relation semantics, unique Output Agent, and explicit `FINISH` remain unchanged. Retrieval evidence travels through the existing ReAct trace and versioned Artifact envelope. |
| PubMed retrieval | **Necessary HealthBench adaptation** | The checked FlowSteer and SkillFlow trees contain no callable Web-search backend. `healthbench_evidence_adapter.py::PubMedEUtilitiesClient` therefore uses the official NCBI `ESearch` and `EFetch` endpoints with a bounded result count and structured source receipts. It is restricted to the official NCBI E-utilities host and cannot accept rubric, reference, ground-truth, sample-ID, or evaluator fields. |
| Aggregate evidence Tool | **Necessary HealthBench adaptation** | `healthbench-authoritative.search` interleaves the existing frozen textbook rank with PubMed results without comparing incompatible score scales. Each evidence item records source type, source, document ID, title, date, URL when provided by the source, excerpt, and rank. |
| English query normalization and bounded complementary retrieval | **Thin adaptation of SkillFlow query pivot plus FlowSteer state-conditioned action admission** | The Tool schema asks for concise English clinical terminology because the measured frozen corpus is English and non-ASCII queries had substantially higher empty retrieval. One bounded distinct supplemental query may cover another unresolved clinical concept; duplicate actions are masked from execution. No rubric or reference answer determines query admission. |
| Strict multi-branch StructuredAction schema | **Necessary provider-compatibility adaptation** | When both `search` and `complete` are legal, the HealthBench execution adapter supplies a strict JSON Schema `oneOf` over the existing five-field `StructuredAction` variants. It changes wire-format admission only, not the selected medical reasoning or topology. |
| Atomic Tool transition in Canvas | **Necessary general AgentGraph compatibility adaptation** | `execution_mode=react` and `allowed_tools` may be updated in one admitted Canvas transition so the Director is not offered an impossible intermediate reasoning-Agent state. The validator remains authoritative and no medical role or fixed workflow is added. |

The Web lane queries public medical literature only. HealthBench conversations
may be reduced by the acting model to clinical search concepts, but exact
benchmark-question lookup is not part of the Tool contract. `rubric_items`,
physician/reference responses, canary strings, grader output, and benchmark
answer stores remain evaluator-only. Direct and AgentGraph expose the same
aggregate Tool, corpus revision, Web provider, query budget, timeout, generation
condition, task IDs, and evaluator. This condition remains non-paper-comparable
while it uses the available `gpt-5.4` alias rather than the exact dated paper
grader identity.

### HealthBench authoritative retrieval scope-preservation v2

The v1 two-task canary exposed an orchestration-boundary regression rather
than a Dataset Adapter or communication-transport failure: `SET_OUTPUT`
invalidated and re-executed an Agent whose Artifact was already available.
The second execution did not receive the first execution as a new upstream
Artifact, so it repeated retrieval and replaced the prior response.  The v2
condition reuses the source-aligned behavior already present in the main
project tree rather than introducing a HealthBench-specific continuation path.

| v2 boundary | Source and classification |
| --- | --- |
| `SET_OUTPUT` changes only the unique Output pointer and returns no dirty component when the selected Agent already has an Artifact | **Direct reuse** of `/ssd1/iclr/1/FlowSteer/src/interactive/agent_workflow_env.py::AgentGraphWorkflowEnv._apply_action`; this is the FlowSteer Canvas edit/execution boundary. A missing Output Artifact is still executed by the existing Runtime missing-output boundary. |
| Ordinary Output and non-Output Agents receive the same generic execution contract; Output selection happens outside the model invocation | **Direct reuse** of `/ssd1/iclr/1/FlowSteer/src/interactive/openai_gateway.py`; this prevents pointer selection from silently changing Agent semantics. |
| Head--tail Artifact previews with measured character counts | **Direct reuse** of the current project's generic Canvas-feedback helper, derived from the same FlowSteer progressive feedback boundary. Full Artifacts remain in trajectory receipts; only bounded feedback is shown to the Director. |
| A free-text contract must preserve the original task scope and requested output form | **Minimal general compatibility adaptation** to the neutral Director/Canvas contract. It adds no medical role, fixed Agent count, topology, workflow template, rubric concept, or answer content. |
| `execution_role=output` unless a real Formatter protocol is enabled | **Receipt correction** over existing terminal-protocol state. It distinguishes the unique Output pointer from a Formatter role without requiring either role in the open search space. |

The new condition is
`config/evaluation_healthbench_professional_authoritative_scope_v2_gpt54_rubric.yaml`
with Director prompt
`agentgraph.director.minimal-neutral-scalar.v3`.  The prompt adds one generic
scope-preservation sentence only.  Retrieval, model, fixed 525-task ordering,
generation settings, Tool limits, evaluator, and GPU0 service remain identical
to authoritative v1.  The output namespace is independent, so v1 and v2
receipts cannot be resumed or reported as one condition.  Prepare-only has
validated the 525-task manifest; live canary and full metrics remain gated on
their own completed receipts.

### HealthBench registered execution profiles and Canvas feedback v3

The scope-v2 canary proved that free AgentGraph and reciprocal execution were
available, but its scalar `ADD_AGENT` domain omitted the Runtime execution
profile.  Contracts that described retrieval therefore instantiated as
`execution_mode=reasoning, allowed_tools=[]`.  The same canary also persisted
successful ReAct and reciprocal DRAFT/REVISION details in trajectory records
without projecting those public receipts into the next Director observation.

| v3 boundary | Source and classification |
| --- | --- |
| Scalar `ADD_AGENT` exposes and requires the exact `(execution_mode, allowed_tools)` pair registered by `AgentRuntime` | **Direct reuse** of `/ssd1/iclr/1/FlowSteer/src/interactive/agent_workflow_env.py::AgentWorkflowEnv.model_admissible_action_targets`. This is capability admission, not an Agent role or topology rule. |
| Constrained decoding uses one `oneOf` branch per registered execution profile | **Direct reuse** of `/ssd1/iclr/1/FlowSteer/src/interactive/director.py::director_live_action_parameter_json_schema_text`. The scalar action remains `agent_id + model_id + free-text contract + execution profile`. |
| The collector binds the sampled `ADD_AGENT` receipt to the same live profile | **Direct reuse** of `/ssd1/iclr/1/FlowSteer/src/interactive/rollout_collector.py::_validate_hierarchical_action_receipts`; missing or unregistered pairs fail closed. |
| Revision-live Artifact freshness and provenance remain visible after a rejected or pointer-only edit | **Thin generic reuse** of `/ssd1/iclr/1/FlowSteer/src/interactive/agent_workflow_env.py::current_artifact_receipts`; QA candidate and evaluator fields are deliberately omitted for HealthBench. |
| Every `AgentCallRecord` phase, CommunicationEnvelope and public ReAct `StructuredAction → Observation` is projected into the next Canvas observation | **Necessary compatibility adaptation** at `_accepted_feedback`, using the existing canonical Runtime/trajectory fields. Full Artifacts and retrieved passages stay in trajectory storage; Director feedback contains bounded previews, Tool/source receipts and evidence provenance only. |
| Persistent Director continuation keeps the full original task once, current receipts once, and compact prior Action--Observation feedback | **Direct SkillFlow/FlowSteer reuse** from SkillFlow `BoundedAgent.execute_turn` / `RolloutEngine.run` and FlowSteer Director compact-history/current-artifact logic. |

`execution_mode=react` is a working mode of an Agent and is never serialized as
an Agent role.  The v3 adaptation does not require retrieval, a minimum Agent
count, a medical role, a reciprocal relation, or any fixed topology.  It only
makes registered capabilities and measured execution feedback available to the
free Director search space.  Rubrics, physician/reference responses and grader
state remain evaluator-only.

### HealthBench Qwen3.5 thinking condition v4

The v4 condition changes the Qwen generation mode, its exact receipts, and the
per-Action total completion budget from 1,024 to 4,096 tokens. Hidden reasoning
and the public JSON action still share that single budget; the interrupted v4
formal run later observed length termination, so 4,096 must not be described
as a non-truncating allowance. It retains the v3 AgentGraph search space,
prompt profile, 525-task order, generation seed, Tool condition, grader, and
inference-only boundary. The result is therefore not a controlled
thinking-only ablation against v3.

| v4 boundary | Source and classification |
| --- | --- |
| Executor `chat_template_kwargs.enable_thinking` | **Direct reuse** of the existing `OpenAICompatibleGateway` Qwen/SGLang request path and SkillFlow `training/batch_inference.py` explicit `enable_thinking` switch. The registered model metadata is the single source of truth for Direct and AgentGraph Executors. |
| Director tokenizer thinking template plus native SGLang `require_reasoning` | **Necessary Qwen3.5/SGLang adaptation** in `SGLangReceiptDirectorClient`. The checked SkillFlow entry points default or hard-code thinking off, while Director Canvas actions require hidden reasoning followed by the existing JSON-Schema-constrained public action. |
| `meta_info.reasoning_tokens` action boundary | **Necessary provider-receipt adaptation**. Director hidden reasoning tokens are separated from the public Canvas action using the server-supplied token boundary; no heuristic scan for the first JSON brace is used. The public action alone reaches the parser and Canvas. |
| Thinking observability | **Receipt-only adaptation**. Trajectories record the thinking flag, available reasoning-token/character counts, presence flag, and public-action token offset. The Director's hidden reasoning is not stored as plaintext or routed between Agents or back to the Director, although the exact output token IDs/log-probabilities retained for the existing rollout receipt can reconstruct that prefix with the tokenizer. Executor `reasoning_tokens=0` means the OpenAI-compatible server omitted an exact token count when `reasoning_content_present=true`; it must not be read as evidence of zero reasoning. |
| ReAct scientific-sampling identity | **Necessary receipt wiring** over the existing SkillFlow scientific sampling coordinate. `chat_template_enable_thinking` is recorded in the pre-dispatch nested receipt and checked against the provider request/response receipt before a result can be resumed or reported. |

Thinking is a model generation mode. ReAct remains an Agent
`execution_mode`; neither is an Agent role. The condition does not add a
medical role, fixed topology, fixed Agent count, learned Skill, GRPO, LoRA,
MACE, Bayesian update, backward pass, or optimizer step.

### HealthBench two-phase Director and paired ReAct profile v5/v6

The v5 diagnostic and v6 paired condition keep the same unified Canvas,
Runtime, trajectory, Tool, dataset adapter, and official/reference evaluator
boundaries. They do not introduce a HealthBench-specific orchestration core.

| Boundary | Source and classification |
| --- | --- |
| Director `REASONING -> ACTION` generation with independent phase budgets | **Thin adaptation of direct SkillFlow reuse** from `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/skillev/rollout/engine.py::RolloutEngine.run`. The local SGLang client makes two conditioned calls, exposes only the JSON-Schema ACTION to Canvas, and stores exact per-phase prompt/token/log-probability receipts. The receipt explicitly records `p(reasoning\|base_prompt) * p(action\|base_prompt,reasoning)` and `single_autoregressive_receipt=false`; it is evaluation-compatible and is not represented as an action-masked one-pass training trajectory. |
| 512-token Director reasoning budget | **Direct parameter reuse** of the bounded Qwen Supervisor thinking budget in SkillFlow `training/batch_inference.py`. The paired Direct/Executor response budget stays at 4,096 because the existing evaluation identity uses the same field for both arms. |
| `FINISH` legal but not exclusive once the current Graph is admissible | **Direct reuse of the current FlowSteer-derived state-conditioned action domain** in `AgentWorkflowEnv.model_admissible_action_types`. v4 set `finish_only_when_admissible=true`, which forced every admissible singleton to terminate; v5/v6 set it to false without rewarding or requiring a larger Graph. |
| BM25 `score` and `matched_terms` retained in authoritative evidence | **Necessary projection repair over direct SkillFlow retrieval output**. `FrozenMedRAGBM25Corpus.search` already computes these fields; the HealthBench aggregation layer now preserves them in Tool Observation, trajectory receipt, and bounded Director evidence provenance. PubMed evidence does not fabricate BM25 fields. |
| `healthbench_tool_runtime.execution_profile_allowlist` | **Necessary project compatibility adaptation**, not an upstream module. SkillFlow filters task-scoped Tool resources and FlowSteer masks unavailable operators, but neither upstream represents heterogeneous `(execution_mode, allowed_tools)` pairs. The implementation narrows the existing `AgentRuntime.registered_execution_profiles -> Canvas action_target_domains -> validate_execution_contracts` path and rechecks the profile before dispatch. No Agent role, contract, count, relation, Output, or topology is selected by this allowlist. |
| Direct/Graph profile identity and resume boundary | **Evaluation provenance adaptation**. Preflight requires the v6 allowlist to equal Direct `(react, direct_allowed_tools)` exactly; manifest, Direct generation identity, paired receipt, and report persist the allowlist. A mismatch fails closed rather than being labelled protocol-equivalent. |

The v5 two-task run passed the execution/evaluator Stable Zero chain, but one
Graph selected Tool-free `reasoning`; its paired identity therefore failed and
v5 is retained only as rejected diagnostic evidence. v6 is accurately
described as a **fixed ReAct+Tool execution protocol with a free-topology
AgentGraph**. It is not compute-matched to the Single-Agent ReAct comparator,
because a freely selected multi-Agent Graph can make more total model and Tool
calls. ReAct remains an execution mode, never a role. Training, GRPO, LoRA,
MACE, Bayesian updates, and Skill retrieval/evolution remain disabled.

### HealthBench receipt-backed failure report v3

| Boundary | Source and classification |
| --- | --- |
| JSONL loading, trajectory communication extraction, ReAct trace extraction, Tool receipt extraction, and atomic Markdown output | **Direct project reuse** from `report_joint_qa_progressive_experiment.py` and `report_multidataset_stable_zero.py`; no second trajectory/report storage model is introduced. |
| Mutually exclusive first-observable failure taxonomy | **HealthBench evaluator adaptation** over the existing completion runner's `wrong_demo_diagnosis`, terminal receipt, and rubric-level grader receipt. It does not infer hidden reasoning or a fixed Verifier role. |
| Direct response population with one frozen strict-zero terminal | **Necessary reporting compatibility adaptation**. `scripts/report_healthbench_failure_demos.py::_validate_task_populations` requires exact paired/trajectory/private task-ID equality. A missing Direct response is admitted only when `run_manifest.direct_progress`, the paired unavailable/invalid zero-score record, and an append-only Direct ReAct exhaustion receipt agree exactly. No answer or evaluator record is synthesized. |
| Public versus evaluator-private reports | **Direct reuse of the repository's evaluator-only boundary**. Public output contains aggregate counts and redacted receipts; full conversation, signed rubric, physician response, candidate outputs, and full turns remain under ignored `artifacts/.../evaluator_private/` and never enter Director/Agent input. |

The v3 failure report is generated entirely offline from the frozen 525-task
evaluation. Its own manifest records zero model calls, zero Tool calls, zero
grader calls, and no training.

### HealthBench official-reference thinking/subgraph v2.1

This condition is derived from the highest fully completed condition that is
strictly comparable under the same 525-task public-test split and exact OpenAI
simple-evals reference evaluator: `healthbench_professional_official_v1`
(`overall_score_length_adjusted=0.2023946457`). It does not reuse the lower v6
retrieval condition or its non-reference grader alias.

| v2.1 boundary | Source and classification |
| --- | --- |
| Existing `ADD_SUBGRAPH` transaction, one-to-three free Agents, directed/reciprocal relations, incremental execute-on-edit, and next-turn Canvas feedback | **Direct FlowSteer/project reuse** from `src/interactive/agent_workflow_env.py::step`, `AgentWorkflowEnv.model_admissible_action_types`, and the existing `ADD_SUBGRAPH` parser/schema. No new graph builder, role inventory, or topology rule is added. |
| Hidden Qwen reasoning budget separated from the visible response budget | **Direct SkillFlow reuse plus necessary OpenAI-compatible adaptation** from SkillFlow `training/batch_inference.py::_supervisor_call_unpaused`, which sends `visible_max_tokens + thinking_budget`. `src/interactive/openai_gateway.py` now accepts a versioned `thinking_budget`, sends the sum as SGLang `max_tokens`, and records the visible and hidden components in `requested_sampling`. |
| Two-phase Director with thinking only in REASONING and JSON-Schema ACTION separately bounded | **Existing direct SkillFlow reuse** from `src/skillev/rollout/engine.py::RolloutEngine.run` through the already implemented `SGLangReceiptDirectorClient`. The public Canvas action remains schema constrained; the ACTION serialization phase is not a second reasoning Agent. |
| `agentgraph.director.minimal-neutral.v11` | **Minimal generic compatibility adaptation** over neutral v10. It states only distinct contract fields, producer-to-consumer relation direction, complete user-facing Output artifact, non-duplicated work, and FINISH/repair behavior. It contains no medical role, rubric item, fixed Agent count, fixed relation, topology template, or benchmark answer. |
| `require_informative_contracts`, `reuse_unchanged_agent_inputs`, and `producer_context_exact_dedup_v1` | **Direct reuse of existing project interfaces** derived from FlowSteer's Canvas admission/incremental execution and SkillFlow's bounded public Artifact boundary. Exact duplicate envelopes are removed; no semantic content is deleted or post-processed. |
| `repetition_penalty=1.10` for Director, Direct, and Executor | **Versioned local SGLang decoding condition**, not a learned reward and not a claimed SkillFlow optimum. It is symmetric across Direct/AgentGraph and must be judged only by the new canary/formal receipts. |
| Exact HealthBench reference evaluator, empty Tool condition, fixed public-test population, and evaluator-private rubrics | **Unchanged direct reuse** from `healthbench_professional_official_v1` and official OpenAI simple-evals revision `652c89d`. The grader's own reasoning setting remains fixed and is not part of the workflow model change. |

The first `v2` Stable Zero attempt failed before Agent execution because
hierarchical action-mask v3 requires a role-conditional semantic-lineage
domain, while HealthBench deliberately uses `semantic_protocol=none` and free
role metadata. Its failure receipts remain under the v2 namespace. The v2.1
condition uses the existing model-admissible v2 live action mask plus the
existing complete `ADD_SUBGRAPH` JSON schema; this is the free-AgentGraph
boundary used by non-semantic tasks and does not introduce a HealthBench role
template. No training, backward, optimizer update, LoRA, GRPO, MACE, Bayesian
update, Skill injection, medical memory, or Tool retrieval is enabled.

### HealthBench official-reference thinking/subgraph v2.4

The v2.4 condition remains an inference-only derivative of
`healthbench_professional_official_v1` on the same ordered 525-task public test,
empty Tool condition, and exact OpenAI simple-evals reference evaluator.  It
does not introduce a medical role inventory, fixed Agent count, topology,
training objective, memory, or benchmark-specific answer rule.

| v2.4 boundary | Source and classification |
| --- | --- |
| Generic Output Agent and intermediate-node execution protocols | **Direct project reuse** from the source revision evaluated by `healthbench_professional_official_v1` (`3f162564`), specifically `src/interactive/openai_gateway.py::build_agent_messages`. The only added sentence is a **necessary generic adaptation** that excludes AgentGraph identifiers, internal Artifact/provenance labels, repeated rationale, and intermediate-analysis headings from a user-facing response. |
| Atomic terminal `ADD_SUBGRAPH(..., output_agent_id=...)` | **Direct FlowSteer/project reuse** of `AgentWorkflowEnv._apply_mutation`: the relation set and Output pointer are installed before the accepted Canvas revision is executed, so the terminal node runs once with `is_output_agent=true`. `FINISH` retains FlowSteer's existing reuse-only submission boundary. |
| Pointer-only standalone `SET_OUTPUT` | **Current project compatibility boundary** retained to avoid repeating Tool or model work. Director v14 permits it only when the existing Artifact is already a complete user-facing response; otherwise the graph must be repaired or augmented before submission. |
| One relation object per unordered endpoint pair | **Direct reuse** of the existing two-bit Canvas relation encoding. `source_to_target` and `target_to_source` jointly encode a directed or bounded reciprocal relation; no topology is selected by this constraint. |
| Director budgets 2,048 REASONING / 4,096 ACTION | **Versioned SkillFlow two-phase generation parameter change**. REASONING uses Qwen thinking; JSON-Schema ACTION remains a non-reasoning serialization phase. Direct and Executor keep 4,096 visible plus 4,096 hidden tokens and `repetition_penalty=1.10`. |
| Executor sampling and trajectory receipt | **Direct SkillFlow reuse plus necessary SGLang adaptation** already introduced in v2.1. Provider `max_tokens=8192`, visible/hidden components, repetition penalty, thinking presence, and request status are retained in each `ExecutionRecord`; hidden reasoning text is not routed between Agents. |

The two-task v2.4 Stable Zero run completed 2/2 Direct and 2/2 AgentGraph
responses, with two explicit AgentGraph `FINISH` actions and no terminal or
collection failures.  AgentGraph raw score was `0.6208333333` and
length-adjusted score was `0.5936383333`; the same-condition Direct scores were
`0.2000000000` and `0.2389844000`.  These two samples validate the execution
chain only and are not a 525-task estimate.  Full evaluation remains required
before changing the best-profile pointer.

### HealthBench heterogeneous all-thinking Executor condition v2.5

The v2.5 condition changes only the Workflow Executor model search space and
its versioned storage namespace.  The Flow-Director remains the same local
Qwen3.5-9B policy, prompt v14, two-phase Canvas client, and action mask.  It
does not add a role-to-model mapping, medical role inventory, fixed Agent
count, relation, topology, Tool, memory, Skill, or training objective.

| v2.5 boundary | Source and classification |
| --- | --- |
| Per-node heterogeneous `model_id` selection | **Direct reuse** of the existing `ModelRegistry -> Director model catalog -> Canvas action.model_id -> AgentGraphValidator -> AgentRuntime provider dispatch` chain. `executor_selection=director_catalog_choice` is unchanged; no second router or random model assignment is added. |
| Four-entry thinking catalog | **Thin HealthBench configuration adaptation** of the receipt-backed AIME v15 heterogeneous catalog. The exact entries are local `qwen3.5-9b-local` plus remote `qwen3.5-flash`, `deepseek-v4-flash`, and `MiniMax-M3`; all have equal selection/cheap/fast weights and neutral text capability descriptions. |
| Remote thinking compatibility | **Direct reuse of existing capability receipts** in `artifacts/model_capability_canary/aime_all_thinking_remote_pool_20260831.json`. Every admitted remote returned `reasoning_content_present=true` with the gateway thinking toggle; candidates that returned HTTP 429 or timed out are not admitted. The current `/v1/models` endpoint was checked again before freezing the catalog. |
| Thinking request and receipt | **Direct SkillFlow/provider-boundary reuse** through `OpenAICompatibleGateway`. Every admitted ModelSpec sets `chat_template_enable_thinking=true`; the gateway stores the requested toggle and provider reasoning presence/token fields without routing hidden reasoning as an Artifact. Local SGLang alone keeps its supported `top_k` and repetition-penalty metadata; those fields are not invented for remote providers. |
| Direct checkpoint reuse | **Direct reuse** of `evaluate_completion_benchmark_round.py::_collect_direct`. The new condition declares `direct_reused_from` for the stopped v2.4 checkpoint; the runner verifies task/model/protocol/seed/evaluator/judge identity, records a reuse receipt, and only evaluates missing Direct tasks. Graph trajectories are never reused across the catalog/version boundary. |
| Comparison label | **Necessary evaluation provenance distinction**. Because Workflow Agents may use heterogeneous remote models while Direct remains local Qwen3.5-9B, `protocol_equivalent_to_direct=false`; the final Direct-versus-AgentGraph delta is a descriptive composite-system comparison, not an isolated causal estimate of orchestration. |

The Director REASONING phase keeps thinking enabled.  Its JSON-Schema ACTION
phase remains a constrained serialization phase with thinking disabled, as in
the existing SkillFlow two-phase boundary; it is not an Executor or a semantic
answer-generation turn.  Training, GRPO, backward, optimizer update, LoRA,
MACE, Bayesian update, and Skill evolution remain disabled.

### HealthBench heterogeneous all-thinking action-mask v3 condition v2.6

The v2.5 Stable Zero run completed two evaluator-valid Direct and AgentGraph
records, but its second task required twelve rejected ADD_SUBGRAPH attempts:
the unconstrained v2 relation strings repeatedly named task text, user/system
messages, and Output labels as if they were Canvas Agents. Prompt v14 already
states the endpoint rule, so v2.6 changes the constrained action domain rather
than adding a HealthBench workflow template.

| v2.6 boundary | Source and classification |
| --- | --- |
| Execution-profile-first free-contract ADD_SUBGRAPH | **Direct FlowSteer reuse** from commit 31b8c01 (director execution-profile selection, declaration factorization, and exact hierarchical receipt validation). The Director samples Agent count, registered execution profile, model_id, and free-text contract; no role family is introduced. |
| Tool-free Runtime profile admission | **Necessary HealthBench compatibility adaptation** over that FlowSteer source. The upstream stateful condition required one environment Tool owner; HealthBench has required_tool_id=null, so the live domain admits the Runtime's exact registered tool-free profiles and omits owner/count semantics. Stateful Tool conditions retain their prior path. |
| Same-action/existing-Agent endpoint domain and one-relation edit boundary | **Direct FlowSteer reuse** of live Canvas target domains and incremental edit semantics. Task, conversation messages, context labels, and Output labels cannot be relation endpoints because they are absent from the generated JSON Schema. One accepted edit may still add one to three Agents, zero or one relation, and an optional Output pointer. |
| Profile-first phase and trajectory receipts | **Direct FlowSteer reuse** from commit 31b8c01, including selection/declaration/parameter prompt binding, bounded serialization regeneration, action-schema identity, selected execution profiles, and fail-closed receipt validation. |
| Four-model all-thinking Executor catalog | **Unchanged direct reuse** of v2.5. Each node may select local Qwen3.5-9B, Qwen3.5 Flash, DeepSeek V4 Flash, or MiniMax M3. Flow-Director remains local Qwen3.5-9B; its REASONING phase uses thinking and its JSON-Schema ACTION phase remains non-reasoning serialization. |

The condition remains inference-only with the same 525-task public-test order,
empty Tool condition, Direct identity, reference evaluator, grader, seed, and
generation settings as v2.5. It uses an independent v2.6 artifact namespace;
v2.5 trajectories are not resumed into v2.6.

### HealthBench held-out architecture repair v2.7

The v2.7 condition is a fixed 20-task development check drawn from ordinal
63--82 of the frozen v2.6 public-test manifest.  These task IDs do not overlap
the 58 v2.6 AgentGraph trajectories used to derive the repair hypotheses.  It
retains the v2.6 heterogeneous all-thinking catalog, local Qwen3.5-9B
Flow-Director, no-Tool condition, generation seed, Direct receipts, and exact
OpenAI simple-evals reference evaluator.  It is not a 525-task benchmark score.

| v2.7 boundary | Source and classification |
| --- | --- |
| `agentgraph.director.minimal-neutral.v15` and compact historical Canvas observations | **Direct reuse plus minimal generic adaptation.** The compact Action--Observation projection already used by the SkillFlow/FlowSteer QA and scalar Director paths is enabled for v15. The added policy states only that a pre-execution contract is not evidence, unresolved ambiguity must not be silently collapsed, and a correction must be routed to the terminal producer. It does not name a medical role, require an Agent count, or fix a topology. |
| HealthBench Professional execution protocol in `build_agent_messages` | **Necessary task-adapter change.** It is applied uniformly to any free Agent receiving the official native conversation, and covers ambiguity, contradictions, clinical safety and requested response form. It contains no rubric, reference response, benchmark answer, role inventory, or evaluator state. |
| `generated_as_output_agent` / `generated_as_format_agent` Artifact metadata | **Necessary compatibility adaptation over direct FlowSteer pointer-only `SET_OUTPUT`.** `AgentRuntime` records the exact invocation protocol. When the v2.7 gate is enabled, an intermediate Artifact cannot be promoted by pointer-only `SET_OUTPUT`; the existing atomic `ADD_SUBGRAPH(..., output_agent_id=...)` executes a terminal consumer instead. No model output is rewritten. |
| Configurable `max_relations_per_subgraph=2` | **Direct FlowSteer transaction reuse plus necessary HealthBench adaptation.** Commit `31b8c01` supplied the profile-first atomic `ADD_SUBGRAPH` path but capped its ALFWorld Tool-owner transaction at one relation. HealthBench may add three free Agents, so v2.7 exposes up to two sampled relations in the same transaction. Zero relations, directed relations, reciprocal relations, Agent identities, and topology remain Director choices. |
| Reciprocal terminal Artifact-lineage coverage | **Necessary AgentGraph compatibility adaptation; not present in either upstream.** FlowSteer structural validation contracts a reciprocal pair for reachability, while Runtime produces two independent final revision Artifacts. The optional v2.7 FINISH gate follows existing `artifact_version` / `input_artifact_versions` receipts and reports any reciprocal final revision absent from Output's current lineage. It never broadcasts content, adds an edge, parses a contract for a role, or consults the evaluator; ordinary `SET_RELATION` plus dirty-closure execution performs any sampled repair. |
| Frozen task IDs and Direct reuse | **Direct reuse** of `evaluate_completion_benchmark_round.py` task-ID selection, immutable manifest validation, and strict Direct receipt matching. The comparator remains local Qwen3.5-9B Direct and the AgentGraph remains heterogeneous, so `protocol_equivalent_to_direct=false` is retained. |

Training, backward, optimizer update, GRPO, LoRA, MACE, Bayesian updates,
Skill retrieval/evolution, generic Web search, and benchmark-answer retrieval
remain disabled. The existing MedRAG/PubMed ReAct adapters are not enabled in
this no-Tool architecture ablation.

### HealthBench held-out Output closure and runtime admission v2.8--v2.15

Versions v2.8--v2.15 retain the exact v2.7 fixed 20-task population, local
Qwen3.5-9B Flow-Director, heterogeneous all-thinking Executor catalog, empty
Tool condition, generation seed, Direct receipts, and OpenAI simple-evals
HealthBench Professional reference evaluator.  They are incremental
architecture diagnostics on a development slice, not a 525-task benchmark
result and not a trained policy.

| Boundary | Source and classification |
| --- | --- |
| Accepted Canvas edit followed by execution, public feedback, and a later explicit `FINISH` | **Direct FlowSteer reuse** through `AgentWorkflowEnv.step`, `_apply_mutation`, and the existing trajectory collector. `ADD_SUBGRAPH` remains an atomic functional-unit edit; the adaptation does not change the six Canvas actions or impose an Agent count, role inventory, relation, or topology. |
| Provider call receipt followed by non-empty public completion admission | **Thin SkillFlow runtime adaptation** from the sampled-turn/error boundary in SkillFlow `src/executor/m_exec.py`. `AgentRuntime._invoke` first retains the actual provider receipt and then rejects whitespace-only visible content as the producer-scoped `CompletionArtifactEmpty`; hidden reasoning is not a semantic Artifact and an empty producer is not misattributed to a downstream consumer. |
| `artifact_version` and exact `input_artifact_versions` | **Direct reuse of the existing project Artifact receipt boundary**, aligned with SkillFlow's public Action--Observation state. The repair reads recorded provenance only; it does not infer hidden reasoning or synthesize missing content. |
| Quotient-sink Output closure | **Necessary AgentGraph compatibility adaptation** over the existing FlowSteer-derived `AgentGraph._quotient_structure`. Each current quotient-DAG sink must route one successful Artifact into an atomic new Output consumer. A reciprocal component is structurally contracted exactly as before; when final-revision lineage checking is enabled, each materialized reciprocal member is a required Artifact ingress. |
| Grouped ingress domains in constrained `ADD_SUBGRAPH` decoding | **Necessary constrained-decoding adaptation**. JSON Schema `prefixItems` expresses one candidate from each exact sink component and rejects non-sink, duplicate, missing, or extra ingress before execution. The schema exposes the live Canvas domain; it does not select a topology for the Director. |
| Pointer-only `SET_OUTPUT` and atomic Output construction | **Direct FlowSteer reuse plus the v2.7 provenance guard**. An existing user-facing Output Artifact may be selected without another model call; otherwise the existing atomic `ADD_SUBGRAPH(..., output_agent_id=...)` executes the terminal consumer once with all required ingress Artifacts. |
| Mandatory producer repair and raw-action admission | **Necessary generic adaptation** of the project MD's `preserve -> diagnose -> repair -> augment` policy. After a measured producer failure or incomplete Output provenance, both the constrained action mask and authoritative raw admission admit only the responsible live `MODIFY_AGENT`, exact lineage relation, or atomic Output closure. Correct Artifacts and valid graph structure are preserved. |
| Reciprocal final-revision lineage repair | **Necessary AgentGraph compatibility adaptation**. If one reciprocal member's current Artifact version is absent from Output lineage, the live domain exposes only a directed acyclic edge from that missing producer into Output or an already consumed Output ancestor. No edge is inserted automatically and no reciprocal topology is required. |
| Terminal-evaluator-only retry | **Direct reuse of the existing runner separation between trajectory collection and terminal evaluation**, implemented by `evaluate_hotpotqa_round.py::_retry_terminal_evaluator`. A transient grader failure reuses the frozen Director/Canvas trajectory and records `evaluation_retry_receipt.reused_director_canvas=true`; it never repeats Executor or Tool work. |

The completed v2.15 run produced 20/20 evaluator-valid explicit `FINISH`
trajectories, zero `max_rounds`, and zero terminal/runtime failure. Its strict
Official Overall Score is `0.4418114973` and its primary Length-Adjusted
Overall Score is `0.3998106573`, versus `0.2771519608` and `0.2155692508`
for the same 20 Direct records. The v2.14 namespace is an interrupted
diagnostic and is excluded from every v2.15 metric.

No training, backward pass, optimizer update, GRPO, LoRA, MACE, Bayesian
update, Skill injection/evolution, memory, Web search, or medical Tool was
enabled in v2.15. ReAct remains an optional per-Agent execution mode, not an
Agent role; this no-Tool condition happened to admit only `reasoning` profiles.

### HealthBench v2.27 provenance, scope, retrieval and Artifact quality repair

This inference-only condition starts from the completed fixed-20 v2.16
architecture profile because it has the highest completed same-panel score in
the local development evidence. It freezes that panel, sampling schedule,
Direct condition, four-model Executor catalog and official reference grader.
It does not replace the formal 525-task best-profile pointer before a complete
same-protocol evaluation exists.

| Boundary | Source and classification |
| --- | --- |
| Nested `input_artifact_provenance` on routed and reciprocal communication envelopes | **Direct FlowSteer reuse** from `/ssd1/iclr/1/FlowSteer/src/interactive/agent_runtime.py::_input_artifact_provenance_from_metadata`, `_artifact_envelope`, `_response_output_metadata`, and its generic multi-hop regression. Direct Tool receipts remain attributed to the producing Agent; earlier hops remain nested instead of being flattened into a later Agent's Tool use. |
| Model-visible nested provenance in `openai_gateway._format_upstream` | **Direct FlowSteer reuse** of the corresponding standard communication-envelope renderer. Exact duplicate receipts and envelopes are removed using FlowSteer's public-receipt normalization; no Artifact is summarized, rewritten, or semantically filtered. |
| `agentgraph.director.minimal-neutral.v17` | **Existing project reuse** of the already implemented compact v17 prompt. It preserves unresolved names, abbreviations, quantities, time points and answer slots without prescribing Doctor, Researcher, Verifier, Formatter, Agent count, relation or topology. |
| Entity-slot substitution admission | **Necessary project adaptation** inside the existing FlowSteer Canvas transaction guard. It rejects an unsupported replacement in public contract forms such as `patients with X` or `A stands for B` only when the replacement is absent from the initial task and all successful public Tool/Agent Artifacts. It reads no rubric, reference response, reward or evaluator result. |
| Complete named-entity retrieval admission | **Necessary HealthBench Tool-adapter adaptation** over the existing SkillFlow MedRAG plus NCBI PubMed ReAct path. A non-empty result is not counted as relevant when it matches only part of an unresolved acronym/hyphenated entity; the same Agent may reformulate within its existing Tool budget, and budget exhaustion permits an explicit insufficient-evidence completion. This is surface-form coverage, not a learned semantic relevance claim. |
| `public_text_quality_v1` Artifact admission | **Necessary generic Runtime adaptation over SkillFlow's typed completion-failure boundary**. It applies equally to reasoning and ReAct outputs and measures provider length termination, explicit incomplete status, character limit, invalid/control characters, abnormal tokens, exact paragraph repetition, repeated 24-token n-grams and extreme run-on text. Invalid Artifacts are rejected as producer-scoped `CompletionArtifactQualityError`; they are never truncated or routed downstream. |
| `preserve -> diagnose -> repair -> augment` after quality rejection | **Direct reuse of the existing project recovery policy and FlowSteer incremental execute-on-edit loop**. The failure receipt names the responsible Agent and exposes compact quality metrics, so the Director modifies or augments the current graph without deleting valid upstream Artifacts. |

The prepare-only run created the exact 20-task manifest with all dataset
registry checks true, `training_enabled=false`, and `optimizer_updates=0`.
No model, Tool, Web, grader, training, backward, optimizer, LoRA, GRPO, MACE,
Bayesian or Skill-evolution call was made during this verification. A new
official score is **not yet available**; the condition is prepared, not
evaluated.

### HealthBench v2.28 dependency-consistent structured-evidence flow

v2.28 retains the v2.27 provenance, scope, retrieval and Artifact-quality
boundaries while adding two opt-in information-flow controls. The fixed-20 and
full-525 profiles remain role- and topology-neutral: the Director still chooses
the Agent declarations, models, execution modes, relations and Output identity.

| Boundary | Source and classification |
| --- | --- |
| Atomic Canvas transaction and producer-to-consumer routing | **Direct FlowSteer reuse.** An accepted `ADD_SUBGRAPH` remains one validated Canvas transaction, and a directed `source -> target` relation routes the source Artifact to the target. No medical role, fixed Agent count or fixed graph motif is added. |
| Reciprocal initial-draft/revision exchange | **Direct FlowSteer reuse.** A reciprocal relation still means that both peers first produce independently and then each performs the existing bounded revision from the other's Artifact. Because both directed routes are present, a declared incoming peer dependency is compatible with this barrier; v2.28 does not reinterpret the reciprocal edge or impose one. |
| Action--Observation and Tool receipts | **Direct SkillFlow reuse.** ReAct retains the public Action--Observation loop and successful Tool receipts. Complete receipts, provider observations and nested Artifact provenance remain in the request/trajectory/backend record; they are not replaced by a summary. |
| Explicit incoming dependency admission | **Necessary project-specific adaptation.** With `require_declared_dependency_relations=true`, the free-text contract guard conservatively recognizes only explicit incoming references to a `node_*` Artifact/output/result/finding/evidence/analysis/response. Unknown or self dependencies and a producer without a directed path to the consumer are rejected. Outgoing forms such as `send ... to node_*` are not treated as incoming dependencies, so the guard does not infer a role or topology from general prose. |
| Receipt-bound structured-evidence Artifact | **Necessary project-specific adaptation.** Under `producer_context_structured_evidence_v2`, a non-Output HealthBench ReAct completion uses `healthbench.structured-evidence.v1` with `status`, a substantive `summary`, bounded `evidence_items`, and `uncertainties`. Each claimed evidence item carries the supported claim and qualifiers plus receipt-bound document identity and an evidence span; identifiers and metadata must exactly match a successful `healthbench-authoritative.search` receipt, and the normalized span must occur in its excerpt. Unsupported or invented evidence is rejected instead of being routed downstream. |
| Compact model-visible projection | **Necessary project-specific adaptation.** Downstream prompts receive the substantive structured Artifact plus only cited receipt rows and compact nested lineage. Uncited result bodies, duplicated Artifact aliases, timing, corpus statistics, Tool-version noise and raw nested receipts are omitted from this projection. A failed binding never falls back to replaying the whole receipt. This is a presentation boundary only: the complete receipt-bearing trajectory remains intact. |
| Direct and terminal Output boundary | **Unchanged behavior.** Direct remains a complete natural-language assistant response, and `request.is_output_agent=true` keeps the terminal Output as complete user-readable natural text. The internal structured-evidence schema is not exposed as the final answer. ReAct remains an execution mode, not an Agent role. Non-HealthBench communication keeps its prior renderer even if the profile name is present. |

The two isolated profiles are
`config/evaluation_healthbench_professional_mixed_all_thinking_v2_28_heldout20_declared_dependency_structured_evidence.yaml`
and
`config/evaluation_healthbench_professional_mixed_all_thinking_v2_28_full525_declared_dependency_structured_evidence.yaml`.
Implementation is present for the configuration/runtime admission, structured
schema and receipt binding, and compact projection paths. Targeted no-API tests
cover the v2.28 configuration invariants, producer-context profile routing,
intermediate-versus-Output schema boundary, exact receipt binding, cited-row
projection, nested-provenance compaction and non-HealthBench isolation. No live
model, Tool or grader evaluation has been run for v2.28, so this entry makes no
score claim. Training, backward, optimizer update, GRPO, LoRA, MACE, Bayesian
update and Skill retrieval/evolution remain disabled.

### HealthBench v2.29--v2.32 low-score regression and completion repair

The eight task IDs in these profiles are a fixed, post-development regression
panel selected from low-scoring v2.27/v2.28 cases. They are not an unbiased
test sample and are never reported as a replacement for the 525-task result.

| Boundary | Source and classification |
| --- | --- |
| `schema_invalid` completion feedback followed by same-Agent continuation | **Thin SkillFlow reuse.** The control flow follows SkillFlow `src/skillev/runtime/bounded_agent.py::_validate_completion` and `src/skillev/rollout/engine.py`: an invalid public completion becomes an Observation and the same Agent continues. The project keeps the equivalent hook in `ToolReactExecutionAdapter._completion_error`; hidden reasoning is never published as an Artifact. |
| Provider truncation, repetition, and incomplete heading/lead-in admission | **Necessary generic Runtime adaptation.** The new immutable `public_text_quality_v2` extends v1 by rejecting a one-line Markdown heading or a one-line lead-in ending in either `:` or Unicode `：`, and by recognizing CJK terminal punctuation. The published `public_text_quality_v1` behavior is unchanged. The check is surface-only, applies to reasoning and ReAct, and does not inspect the task rubric, reference response, Agent role, or topology. The Artifact remains in the call receipt but is not routed downstream. |
| Explicit removal of warnings/risks in a HealthBench contract | **Necessary task-adapter correction at the existing FlowSteer Canvas admission boundary.** The existing verb-led guard remains unchanged. v2.32 opts into `require_explicit_safety_scope_preservation`, which also covers the observed direct form `without disclaimers or warnings`; it reads only the public task and sampled contract. It does not supply a medical workflow or an answer. |
| Receipt-bound intermediate evidence | **Existing v2.28 project adaptation reused unchanged.** `producer_context_structured_evidence_v2` requires non-Output HealthBench ReAct completions to bind every cited `document_id`, source identity, and evidence span to an actual successful Tool receipt. Output Agents remain natural-language responders. |
| Relevant-evidence, task-query scope, and insufficient-evidence continuation | **Existing v2.27 evidence-anchor check plus a minimal HealthBench adaptation of FlowSteer's QA query-scope boundary.** `require_relevant_evidence=true` is restored after it was inadvertently omitted from v2.29--v2.31. The opt-in `require_task_query_anchor` check rejects a retrieval query with no substantive lexical anchor in the model-visible conversation and returns public same-Agent repair feedback; JSON scaffolding such as `role` is excluded, so tasks without comparable ASCII anchors remain admitted. If a receipt-bound Artifact reports `status=insufficient` while a distinct search remains available, v2.32 uses the same SkillFlow continuation boundary for one refinement instead of publishing the incomplete evidence state. None of these checks expands synonyms, infers a diagnosis, uses a rubric, or inspects ground truth. |
| State-conditioned completion admission | **Thin versioned completion of the existing FlowSteer/SkillFlow Runtime boundary.** Tool actions already fail closed when absent from the measured state-conditioned domain. v2.32 opts into the same post-parse check for completion, so a provider that bypasses constrained decoding cannot complete before the public retrieval state admits it. The historical default remains unchanged. |
| Incremental Canvas, model selection, graph relations, and terminal identity | **Direct FlowSteer reuse.** One accepted `ADD_SUBGRAPH` transaction is executed before the next Director action. Agent number, model, free-text contract, reasoning/ReAct mode, directed or reciprocal relation, and unique Output identity remain sampled choices. No Doctor/Researcher/Verifier/Formatter chain is required. |
| v18 neutral Director policy and compact history | **Versioned composition of existing project policies.** v18 combines v16's producer-to-consumer/conflict rule with v17's unresolved public-name and quantity preservation clause. It uses the compact historical-observation projection already enabled for v15/v16, without changing historical v16 or v17 behavior. Agent roles, Agent count, models, execution modes, relations and topology remain open. |

The v2.32 configuration is
`config/evaluation_healthbench_professional_mixed_all_thinking_v2_32_low8_receipt_bound_completion.yaml`.
Its prepare-only manifest freezes the same eight task IDs with all training,
GRPO, LoRA, MACE, Bayesian and Skill paths disabled. The official evaluator
preflight currently fails with provider `403 insufficient_quota`; therefore no
v2.32 model, Tool, Director or AgentGraph rollout has been started and no score
is claimed.

### 2026-09-05: HealthBench grader recovery and full525 replay

- `scripts/healthbench_professional_grader_worker.py::_BoundedResponsesSampler`
  continues to call the pinned simple-evals `HealthBenchEval.grade_sample`
  through the existing private worker. The HTTP retry classification follows
  the installed OpenAI SDK `BaseClient._should_retry`: retry transient
  408/409/429/5xx failures within the existing bound, but do not retry permanent
  client errors. Structured provider `insufficient_quota` errors stop without
  repeated requests even when delivered as HTTP 429. This is a necessary
  provider-transport adaptation, not a change to rubric grading, aggregation,
  grader model, AgentGraph, or model prompts.
- The v2.32 full525 config reuses v2.27's ordered public-test task panel, seed,
  and `sampling_schedule_purpose` through the existing FlowSteer/SkillFlow
  sampling path. The new condition and output directory remain separate.
  Catalog v7 selects Executor protocol v5 for both Direct and AgentGraph, so
  the new Direct comparator cannot reuse historical v2.27 predictions.
- The standalone quota diagnostic reuses
  `_attach_healthbench_reference_judge` and `_run_evaluator_preflight` with the
  existing synthetic fixture, without creating a model backend. Its one-call
  retry bound applies only to the diagnostic; the saved evaluation's transient
  retry bound remains three.

### 2026-09-05: v2.33 task-context and model-visible evidence correction

| Boundary | Actual source / reuse classification |
| --- | --- |
| Original task, one action, public Observation, next state | **Direct reuse:** SkillFlow `src/skillev/runtime/bounded_agent.py::BoundedAgent.execute_turn` and `src/skillev/rollout/engine.py::RolloutEngine.run`; FlowSteer `src/interactive/workflow_env.py::InteractiveWorkflowEnv.step/_step_internal`. No alternate Supervisor, scheduler or execution loop is introduced. |
| Free contracts, finite reciprocal exchange, unique Output | **Unchanged project adaptation required by MD sections 3 and 4.1.** `AgentRuntime`, Canvas admission and `ADD_SUBGRAPH` incremental execution are reused. Only the new communication profile is registered in the existing allowlists and runtime factory. |
| Compact Director v19 | **Necessary versioned prompt adaptation:** `src/interactive/director.py`. Reuses the existing action domains and compact observation history; keeps the requested relation/context separate from unverified premises, treats missing evidence as uncertainty, and routes grounded corrections to Output. v18 and other historical prompts remain unchanged; no fixed medical role or topology. |
| Public search evidence independent of producer completion | **Thin adaptation:** SkillFlow `training/environment.py::GenericTaskEnvironment.step/_search_external_corpus/_compress_memory_items` and its bounded public Observations; existing FlowSteer `_format_upstream` and receipt-binding helpers. `openai_gateway.py` v3 projects bounded, deduplicated successful search results even when the producer cites none. This repairs a model-input loss, not a missing transport channel. |
| Producer interpretation versus retrieved evidence | **Necessary HealthBench adaptation:** retain exact receipt-bound `supported_claim`, `conditions_or_qualifiers`, document provenance and evidence spans; distinguish these interpretations from independently verified facts. Uncited results remain explicitly unendorsed. Duplicate documents do not erase a later producer's distinct qualifiers. Projection truncation/omissions are explicit; full raw receipts remain in trajectory. |
| v3 applicability / Direct control | **Existing runtime reuse with explicit configuration:** only HealthBench + communication v3 receives the new prompt/projection. Non-HealthBench and historical v2 are unchanged. Direct is explicitly fixed to its original v2 communication profile; it does not receive the Graph treatment. |
| Historical Direct reference | **Necessary evaluation compatibility adaptation:** extends the existing `evaluate_completion_benchmark_round.py::_collect_direct`/HotpotQA `direct_reused_from` path. Source config, population, actual generation/evaluator settings and saved receipts must match before API preflight. Original v2.32 condition IDs are preserved, not relabeled as v2.33. A missing/incompatible control fails instead of silently issuing 525 repeated Direct calls. |

The raw Tool receipts were retained in v2.28/v2.32, but their existence did **not**
guarantee model visibility: an empty `evidence_items` list could erase relevant
results from downstream `rendered_messages`. This corrects the earlier broad
statement that the compact projection did not discard information. v3 preserves
bounded evidence visibility; it does not guarantee clinical correctness or that
the model will use the evidence correctly.

The 525 public tasks have informed development. New runs are explicitly
post-development diagnostic replays, not untouched held-out generalization
estimates. Rubrics/reference answers remain evaluator-only; no sample-specific
answer, medical template, extra judge Agent, training or Skill update is added.

### 2026-09-06: v2.34 evidence-bearing Director observation

| Boundary | Source and classification |
| --- | --- |
| Canvas edit → functional-unit execution → next Director observation | **Direct reuse:** FlowSteer `InteractiveWorkflowEnv.step/_step_internal/_execute_workflow`; current `AgentWorkflowEnv.step`, `AgentRuntime.execute`, `AgentGraphOrchestrator.continue_prompt`. MD §§2.1, 3, 4.1 require free AgentGraph and finite reciprocal execution. The unit remains one accepted subgraph edit, not a newly invented per-Tool Director scheduler. |
| Ordered public Agent/Tool feedback and persistent artifacts | **Thin adaptation:** SkillFlow `training/environment.py::GenericTaskEnvironment.step` records a public Observation alongside each action (lines 796–842), and `_compress_memory_items` folds exact duplicates (3186). Current `_agent_call_receipts`, `current_artifact_receipts` and `_compact_react_action_observations` are extended only for HealthBench communication v4. Hidden thinking and private evaluation data are not projected. |
| Evidence visible to Director | **Direct reuse of the project's already-tested v3 projection:** `openai_gateway._healthbench_v3_artifact/_healthbench_v3_receipts` supplies bounded, deduplicated excerpts, provenance and receipt-bound producer interpretations. `AgentWorkflowEnv._healthbench_artifact_feedback_v4` is a necessary adapter from those public envelopes to the existing Canvas observation. Total live artifact receipt budget is 24,000 characters, with explicit omissions; it is not lossless raw-context replay. |
| Failure recovery and artifact version attribution | **Necessary correction of the existing public feedback:** `_failure_continuations`, `_previous_revision_outputs`, `_latest_failure_record_by_agent` and stored input provenance remain the sources. A failed attempt's inputs cannot relabel an earlier successful artifact. Retained artifacts are marked stale; current node declarations are distinguished from the artifact's producing request. Existing recovery and FINISH admission are unchanged. |
| Directed/reciprocal communication and reused calls | **Existing runtime reused:** the public feedback now covers all seven incoming producers allowed by the eight-node graph limit, and distinguishes reciprocal draft versus revised artifact versions. The Director can inspect public graph state; no extra Agent-to-Agent broadcast channel or relation is introduced. |
| Director v20 | **Versioned task-neutral prompt adaptation of v19:** inspect the richer live receipt, distinguish executable completion from answer correctness, preserve the original responsibility while repairing Tool constraints, avoid inserting missing case-specific facts, and route grounded corrections to Output. No fixed roles, mandatory chain, minimum Agent count, evaluator-guided answer or new Skill. |
| V4 model interface / Direct | **Thin registration only:** runtime/config/factory/Gateway allowlists accept the explicit v4 profile; downstream Agent messages still use the v3 renderer. Direct remains the unchanged v2.32 control with its own v2 profile. No evaluator, generation setting, model catalog, Tool protocol or Direct answer is rewritten. |
| Monitoring and old-to-new evaluation handoff | **Project operational adapter, not an orchestration algorithm:** uses the existing completion runner's exact-resume and evaluator-only retry paths. An exit code or `completed_with_operational_failures` alone is not completion; fixed task IDs, admitted results, evaluator pending count and process locks determine handoff. No training, model calls for monitoring, or automatic source rewriting. |

The low-score panel used for this change is documented in the v2.34 report
directory. It includes cases where evidence **was already received by the
downstream model but was not used correctly**. Increased feedback visibility
does not prove semantic or clinical correctness. Scores on these public tasks
are post-development re-evaluation, not untouched held-out generalization.

## HealthBench Professional v2.35 optional clinical tools (2026-09-06)

| Local module | Source and adaptation classification |
| --- | --- |
| `healthbench_clinical_tools.build/open_healthbench_clinical_tool_registry` | **Direct reuse:** existing `build_healthbench_authoritative_tool_registry`, `build_healthbench_medrag_tool_registry`, and `OpenHealthBenchAuthoritativeToolRegistry` resource lifetime. Frozen BM25 remains SkillFlow `training/environment.py` external-corpus search (tokenization, ranking and excerpt protocol unchanged). No benchmark-derived memory. |
| Calculator registration | **Direct reuse:** `computation_tools.create_aime_computation_registry` and its action-bound backend, ported from SkillFlow `training/tools.py::execute_tool/_calculator`. **Necessary adaptation:** HealthBench dataset scope and tool ID only. No new clinical formula, training reward or arbitrary code runtime. |
| `HealthBenchSourceReadToolBackend` | **Necessary task adapter:** existing corpus has search snippets but no document-ID read API. Reuses the frozen corpus content/identity and `PubMedEUtilitiesClient` transport/date parser, reading the actual source at bounded, explicitly paginated offsets. PubMed abstracts are labeled as abstracts, not full articles. |
| `DailyMedClient` / `HealthBenchDrugLookupToolBackend` | **Necessary external-tool adapter:** NLM DailyMed official `services/v2/spls.json` schema (`setid`, `spl_version`, `title`, `published_date`) and `/spls/{SETID}.xml` HL7 SPL sections, `versionNumber` and `effectiveTime`. References: https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm and https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_setid_api.cfm. Real label text, not metadata-only evidence or an interaction diagnosis. |
| `HealthBenchClinicalReactExecutionAdapter` | **Direct reuse:** generic `ToolReactExecutionAdapter.execute`, derived from SkillFlow bounded Action–Observation/continuation/receipt semantics, and existing HealthBench completion/evidence binding. **Necessary adaptation:** old action domain hard-coded one search resource. A thin subclass admits only each node's declared optional tools, with the existing shared call/turn budget and no mandatory medical role or initial search. |
| `AgentRuntime._validate_execution_profile_allowlist/registered_execution_profiles` | **Necessary interface adaptation:** the existing executor accepts multiple tools, but its Director-facing profile enumeration exposed only singletons. Explicitly configured bundles of individually registered, available, dataset-scoped tools are now admitted. Default and historical singleton profiles are unchanged; no tool power set or graph template is introduced. |
| `openai_gateway` evidence projection | **Necessary task adaptation of existing V3/V4 projection:** accept MedRAG `ranked_chunks`, source-read and label evidence, retaining source/version/page provenance. New read/label dedup keys distinguish pages and document versions so an earlier search excerpt cannot erase new content. Existing authoritative-search-only V2.33/V2.34 input projection is unchanged. |
| Factory / evaluator condition validation | **Thin opt-in wiring:** `train_agentgraph_smoke._healthbench_tool_runtime_settings/_runtime_for_task` uses `toolset=optional_clinical_v1`; `evaluate_completion_benchmark_round.validate_completion_benchmark_config` checks matched Direct tool availability and explicit Graph profiles. Official grader, task adapter, Canvas step/FINISH, rubric isolation and terminal rules are reused. |
| Capability probe / sequential evaluation | **Operational reuse:** deployed Gateway, registered SkillFlow calculator, existing receipt writer and exact-resume completion runner; bounded two-call probes, no model fallback, and existing handoff script with a new target config. No new training/evolution algorithm. |

The model/tool capability change is versioned independently. DeepSeek V4 Flash
and MiniMax-M3 passed real calculator Action–Observation–Complete canaries;
Qwen3.5 Flash repeated an already dispatched action and is not promoted to ReAct.
Changing tools makes old Direct receipts a different condition: v2.35 must collect
its own matched Direct. Tool availability is a project extension, not native
HealthBench infrastructure and not evidence of a score improvement.

## HealthBench Professional v2.36：按来源分类的请求级知识库（2026-09-06）

本次是 `v2.35` 外部工具的必要数据接线，不重新实现检索器、ReAct
执行循环或 AgentGraph。知识库按**数据来源**分类，不按预定义医疗角色分类。

| 本项目边界 | 真实源码来源与复用分类 |
| --- | --- |
| `healthbench_knowledge_store.HealthBenchKnowledgeStore._build/search` | **直接复用 SkillFlow：**`/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/skillev/benchmarks/retrieval.py::DocumentPassage`、`build_retrieval_index`、`RetrievalIndex.search/read`。SQLite FTS5 schema、tokenizer、BM25 排序、完整 passage 读取均由原实现负责，不另造 ranker/schema。 |
| 模块加载、索引生命周期与线程亲和性 | **直接复用现有项目适配：**`src/interactive/qa_retrieval.py::_load_retrieval_module`、`src/interactive/qa_tool_adapter.py::_ThreadAffineRetrievalWorker`；构建与打开流程参考同文件 `open_provided_context_qa_tool_registry`。**必要适配：**同步 Tool 接口通过原 worker 的单线程 executor 调用原 index，避免在运行中的 event loop 内另开 `asyncio.run`。 |
| 三个独立数据分区 | **必要 HealthBench task-specific adaptation：**`conversation` 保存当前完整 `role/content/turn_index`；`medical_references` 仅保存真实 `MedRAG/textbooks`、`NCBI PubMed` 公开 Tool evidence；`drug_labels` 仅保存真实 `NLM DailyMed` evidence。没有将患者陈述、先前 assistant 陈述或模型总结认定为外部医学事实。 |
| 来源记录与不可变索引版本 | **必要 schema 兼容层：**原 `DocumentPassage` 仅有 passage/document ID、title、text，因此 source ID、日期、URL、content type、version、offset、truncated、next offset 等原来源字段保存在白名单 sidecar；`records.jsonl` 追加保存。首次实际查询时才构建 FTS5；追加内容后使用新的 `revision-N.sqlite3`，不覆盖已发布的不可变索引，也不把不同版本或分页误合并。 |
| 现有工具结果进入本地库 | **直接复用现有工具：**`healthbench_knowledge_tools._IndexingToolBackend.invoke` 调用原 `ToolRegistry.ainvoke`；MedRAG/PubMed/source-read/DailyMed/calculator 后端仍来自 v2.35。**薄适配：**只从实际成功 Tool result 中提取公开 evidence，再附加数量、路径和索引错误 receipt。计算结果不伪装为文献；索引错误不应丢弃原工具已成功取得的结果，不增加重复网络请求。 |
| 图关系限定的证据输入 | **复用现有 FlowSteer 通信边界：**`AgentRequest.prior_tool_receipts`、`upstream`、`peer_draft` 与公开 `input_artifact_provenance`；候选提取复用 `openai_gateway._healthbench_search_candidates`。**必要适配：**`healthbench_knowledge_tools._routed_receipts` 仅沿当前请求实际收到的 envelopes 遍历，并遵守 `UPSTREAM_MASKED`；不读取全图缓存，不共享其他任务/Agent 的目录，不把 Agent 自由文本解释追加为医学来源。 |
| 请求作用域、查询能力与 ReAct | **薄适配：**`HealthBenchKnowledgeReactExecutionAdapter.execute` 为每次 Agent invocation 创建独立 `request-*` 目录及 ContextVar，结束时关闭 worker；`_KnowledgeSearchBackend` 暴露可选 `healthbench-knowledge.search(database, query)`。继续复用 `HealthBenchClinicalReactExecutionAdapter` 和 `ToolReactExecutionAdapter.execute` 的 Action–Observation、调用预算、continuation、completion 与 receipt；不新增 Director 每工具调用一次的替代调度器。 |
| 自由编排与 evaluator | **保持原实现：**FlowSteer Canvas incremental execution、自由 `agent_id + model_id + free-text contract`、关系、唯一 Output 和 FINISH；ReAct 是 execution mode，不是 role。继续使用 `healthbench_professional_adapter.parse_model_visible_conversation` 的公开对话边界；rubric/reference 仅归官方 evaluator，不进入建库接口。没有固定医疗工作流、训练、GRPO、LoRA、MACE、Bayesian 更新或 Skill evolution。 |

`conversation` 查询返回 `conversation_matches` 且 `evidence=[]`；外部库没有
真实来源时返回 `status=empty`，不表示外界不存在证据。查询 metadata receipt
只含路径、版本、数量和检索后端；完整对话不会被重复塞进每次工具结果。
本次并未下载完整的新医学语料库，也没有构建 benchmark 答案库；这是对当前
节点实际获取或经图边传入的外部来源进行可查询索引，不等于跨任务的共享医学记忆。

知识库模块已用合成数据运行真实 upstream FTS5 的 10 项定向测试（另有 4 个
参数化子项），没有模型、HTTP 或 grader 调用。工具接线、运行配置与最终整合
由主线另行核验；此记录不声称 v2.36 已完成正式评测，也不声称分数提升。

## 2026-09-06 v2.35 parallel evaluation

- **Direct reuse:** `start_qwen35_director_server.sh` / `run_on_gpu_role.sh`
  retain SkillFlow Qwen3.5 SGLang startup, changing only GPU and port.
- **Necessary operational adapter:** `evaluate_completion_benchmark_round`
  adds `collection_arm`, calling the same `_collect_direct` or `_collect_graph`
  and exact-resume/evaluator. `_finish_collection_arm` projects only its arm
  into existing `_metrics`/`_aggregate`; no substitute grader or paired zero.
- **Necessary bug fix:** `_direct_one` compares exact Tool membership rather
  than declared order against sorted `ToolRegistry.resource_ids`; receipt
  retains config order for existing identity/resume checks and separately
  records registry order. No Tool capability is added or removed.
- FlowSteer/MD free AgentGraph, Canvas, per-node ReAct and all v2.35 sources
  remain unchanged. This does not include v2.36 databases or training.

## 2026-09-06 — HealthBench v2.37 五题开发回归

基于 v2.36 `fc655cc`，移植 v2.35 并行入口/Direct 工具顺序修复 `defb7ea`。
用户 MD 第 3 节自由 AgentGraph、第 4.1 节 edit→feedback、第 15.1 节轨迹
保持不变。本轮不训练、不更新 Skill，不以节点数或拓扑复杂度作为奖励。

| 修改边界 | 来源与必要适配 |
| --- | --- |
| 来源分类数据库 | **直接复用 v2.36**：`healthbench_knowledge_tools/store` → SkillFlow `benchmarks/retrieval.py::DocumentPassage/build_retrieval_index/RetrievalIndex`。对话、真实文献和药品标签分开；不是 benchmark 答案库。 |
| ReAct 有效搜索预算 | **复用原父类逻辑**：`healthbench_evidence_adapter._evidence_preserves_query_anchors` → `HealthBenchClinicalReactExecutionAdapter._state_conditioned_action_domain`。原多工具子类漏掉 `require_relevant_evidence`，现先用已有 MedRAG 投影统一 schema 再计数。仍最多 3 次实际调用、6 个回合；不强制检索或多 Agent。该检查只是词面相关性，不是医学正确性验证。 |
| 查询拼写边界 | **必要任务适配**：既有 `_query_preserves_task_surface` 精确匹配失败后允许纯字母长词的一次字符编辑；不改 SkillFlow `environment.py::_search_external_corpus` 的分词、BM25 或查询文本。短缩写、数字/代码不能模糊替换；没有真实样本映射表。 |
| 搜索片段完整性 | **复用既有 read-source schema**：`FrozenMedRAGBM25Corpus.search` 仍返回原 BM25 前三项/500 字符，但附真实长度、分页和 `source_id`；authoritative adapter、Gateway 透传，既有 V3/V4 与 FTS5 store 接收。不把残句当全文，也不强制每条来源再读一次。 |
| Canvas contract 的公开证据 | **薄适配 FlowSteer admission**：`AgentWorkflowEnv._public_contract_scope_grounding_texts` 对 HealthBench 检索 receipt 复用 `_healthbench_search_candidates`，只从来源标题/正文获取 literal；不再把 result 中回显的 query、目录、文档编号当证明。不是语义裁判，也没有新角色或 FINISH 门槛。 |
| 五题评测 | **直接复用原 runner/evaluator**：`selection=task_ids`、`stage=development`、`--collection-arm agentgraph`，官方 HealthBench rubric grading/聚合不变。旧五题结果仅离线比较；未新增 Direct 调用。 |

Director 仍为本地 Qwen3.5-9B、minimal-neutral.v20；模型池及 thinking、
唯一 Output、有限双向通信、Canvas 功能单元执行边界均不改。MD/两篇论文
没有提供这几条 HealthBench 修复的现成实现，故只在已有 task adapter 边界适配。

### 首轮暴露的 Director 上下文预算修复（评测后，尚无新分数）

`rollout_collector.SGLangReceiptDirectorClient` 继续复用 SkillFlow
`src/skillev/rollout/engine.py` 的 REASONING/ACTION 两阶段生成；原上游及本项目
固定阶段预算并未对最终 chat-template 后的输入计算剩余 context。真实失败
为 29102 + 4096 > 32768。**必要 SGLang 适配**：工厂传入已有 context 配置，
每阶段实际 input_ids 完整形成后将生成上限限制到剩余 token，并在 exact
receipt 同时记录 configured/effective budget。零余量在本地拒绝；不删任务、
证据或关闭 thinking。此修复在首轮结束后完成，不归因到首轮评分，也不称为
SkillFlow 原有动态预算实现。

## 2026-09-06 — v2.38：外部医学知识源（未评分）

用户要求增加医学知识并接入 Agent。HealthBench Professional 的构建来源是
医生实际使用/对抗测试对话，以及医生撰写、复核与裁定的 rubric；不是从
指定医学问答数据库直接抽取题目。官方论文 §3.2–3.3、§3.6 允许医生基线
查阅文献/指南/药品数据库，但没有给出一套可直接当作解题知识库的官方库清单。
来源：[官方论文](https://cdn.openai.com/dd128428-0184-4e25-b155-3a7686c7d744/HealthBench-Professional.pdf)。

| 模块 | 源码/协议来源与状态 |
| --- | --- |
| MedRAG、PubMed、DailyMed | 直接复用项目 v2.35–v2.37；MedRAG 对应 SkillFlow `training/environment.py::_load_external_corpus/_search_external_corpus`，不新增排名模型。 |
| `healthbench_europe_pmc.py` | 必要外部 API 适配，参考现有 `PubMedEUtilitiesClient`、`DailyMedClient` urllib 生命周期。使用 [Europe PMC REST](https://europepmc.org/RestfulWebService) 的 core search、精确 ID 摘要和 OA JATS 全文接口；上游没有此数据库客户端。 |
| `healthbench_clinical_trials.py` | 必要外部 API 适配，参考同一传输接口和 `_source_page/_source_receipt`。[ClinicalTrials.gov v2 数据结构](https://clinicaltrials.gov/data-api/about-api/study-data-structure)决定真实 `protocolSection/hasResults/resultsSection` 投影，不能把注册方案改写成已证实疗效。 |
| Tool 注册/执行 | `healthbench_clinical_tools` 新增两个可选 search capability，既有 `source.read` 增加精确来源分发；`HealthBenchClinicalReactExecutionAdapter` 复用既有 ReAct、查询约束、去重、预算、completion，不增加角色或调度器。旧配置默认关闭新增源。 |
| 证据入库与通信 | `healthbench_knowledge_tools/store` 直接复用 SkillFlow `benchmarks/retrieval.py::DocumentPassage/build_retrieval_index/RetrievalIndex.search/read`，只扩充来源与来源元数据。`openai_gateway._healthbench_search_candidates/_healthbench_v3_receipts` 识别新工具并保留 PMID/PMCID/DOI、文献类型、OA/预印本标记与全文入口。 |
| Canvas、Director、评分 | FlowSteer `workflow_env.py::step` 的 edit→execution→feedback/history 范式、用户 MD §3 自由节点/关系/唯一 Output、现有 Qwen3.5-9B Director、minimal-neutral.v20 及官方 rubric evaluator 均未改变。 |

两篇上游论文没有提供这两个数据库的特定协议实现；本版只扩展 Tool Adapter
和必要配置，不把 API 客户端称作上游现成功能。没有训练、Skill 或后验更新。

## 2026-09-06 — v2.39：Bookshelf / PDQ / AHRQ / MeSH

| 模块 | 来源、复用及必要适配 |
| --- | --- |
| `healthbench_bookshelf.py::BookshelfClient` | **必要 API 适配**：复用现有 PubMed 的 `_required_query/_node_text/_wait_for_pubmed_request_slot` 及 DailyMed/Europe PMC 的注入式 urllib 生命周期。官方 [Bookshelf search](https://www.ncbi.nlm.nih.gov/books/NBK45615/) 决定 ESearch books → ESummary accessionid；Entrez UID 不直接等于 NBK。官方 [Books-OAI](https://www.ncbi.nlm.nih.gov/books/about/oai/) 决定 GetRecord/nbk_ftext 与串行请求边界。metadata 不是正文，不能读取的来源报告 unavailable，不用网页片段伪装全文。 |
| PDQ / AHRQ 分类检索 | **同一客户端的官方集合参数**，不是三个独立全文库。[PDQ NBK82221](https://www.ncbi.nlm.nih.gov/books/NBK82221/) 使用 `pdqcis[book]`；[AHRQ NBK42934](https://www.ncbi.nlm.nih.gov/books/NBK42934/) 使用 `collection_hscompeffcollect[filter]`，不代表全部 AHRQ EPC 来源。两者均保留 `source=NCBI Bookshelf` 和可核验 NBK/集合，跨工具发现同一文献不当作独立证据。 |
| `healthbench_mesh.py::MeSHClient` | **必要 API 适配**：[官方 Swagger](https://id.nlm.nih.gov/mesh/swagger/ui) 的 descriptor-label contains 查询，及 [URI/JSON-LD 协议](https://hhs.github.io/meshrdf/sparql-and-uri-requests) 的 descriptor/preferred concept 读取。直接复用 `_required_query/_source_receipt`；定义不是治疗效果，术语 ID 不是已经取得同义词标签。 |
| Tool / ReAct / 证据通信 | **直接复用** `healthbench_clinical_tools` 注册、`_source_page`、`HealthBenchClinicalReactExecutionAdapter` 的动作与预算、`_healthbench_search_candidates/_healthbench_v3_receipts` 的真实来源投影。必要增量仅为四个工具 ID、source.read 分发、Bookshelf 元数据和显式配置开关。 |
| 请求级证据索引 | **直接复用 SkillFlow** `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/skillev/benchmarks/retrieval.py::DocumentPassage/build_retrieval_index/RetrievalIndex.search/read`；现有 `HealthBenchKnowledgeStore` 增加来源映射。没有新建持久化医学总库；空正文的 Bookshelf metadata 不作为医学正文入库。 |
| 工具能力配置 | **必要项目适配**：用户随后要求取消来源分类，故沿用 v2.35 `execution_profile_allowlist`，新配置仅保留无工具 reasoning 和一套全 12 项工具 ReAct；不再提供单一来源/分类执行 profile。现有 evaluator 配置验证接受这一既有 full-toolset 形式，仍严格匹配 Direct 工具集合。不是指定医疗职责或强制工作流，不改变角色、模型生成配置和每节点预算。 |
| 未改变部分 | MD §3 的自由 Agent、有限双向通信和唯一 Output；FlowSteer `workflow_env.py::step` 的 edit→execute→feedback/history；SkillFlow Qwen3.5/SGLang 接口；原 HealthBench 官方 evaluator。没有新增训练、Skill evolution、MACE 或后验逻辑。 |

**尚未接入：NIH/HHS HIV Clinical Guidelines。** 官方目录和真实章节在当前
urllib 运行环境分别返回 HTTP 403。网页阅读工具可展示页面不等于仓库客户端
可用；未注册猜测性接口或占位客户端。NICE 仍需官方 API 申请/许可，不在本次
可用目录中。新来源的搜索集合、正文可用性与来源日期通过 receipt 保留。

用户随后追加“优化并逐项修复 demo 后重跑同五题”：主线增加仅对新工具
条件生效的简短 ReAct 说明，强调原对话消歧、未命中不等于研究不存在、
metadata 不等于临床发现、contract 预设不等于来源结论；不改 Director
提示词，不使用具体题名/答案。正文完整性与跨阶段证据修复单独记录其来源。

### v2.39：错误 demo 的必要修复

- `healthbench_evidence_adapter`：参考真实 SkillFlow
  `src/skillev/runtime/bounded_agent.py::_validate_completion` 的非终局
  `schema_invalid` observation；直接复用项目 `react_execution.py` 的原
  completion-validation/恢复循环。新增的 HealthBench 正文外形检查是必要
  task adaptation，不是上游现成医学判断。用既有 ContextVar 模式向无 request
  参数的 hook 提供协程隔离的公开任务；明确要求正文时拒绝纯标题，保留短答、
  显式标题/提纲请求和中间 artifact，不新增模型 judge 或最小字数。
- `agent_workflow_env` → `agent_runtime`：复用 Canvas 已有
  `_previous_revision_outputs/_metadata`、UpstreamMessage/input provenance、
  MD §3.3 的有限双向 DRAFT/REVISION。旧答案继续失效，但旧来源通过当前图内
  原节点历史输入显式移交；不使用新的持久化服务或跨任务缓存。
  历史来源不进入 SkillFlow bounded-agent 控制 trace/当前 continuation
  输入版本，不重置工具预算。对方的新 DRAFT 仍只能在 REVISION 阶段读取；
  自身此前合法收到的来源可继续保留，不误当成提前读取对方新草稿。
- 定向测试使用真实 Canvas step→Runtime、build_agent_messages 和本地
  合成 Tool receipt；没有用 benchmark 答案编码规则，没有改官方 evaluator。

### v2.39 五题结果与离线报告适配（2026-09-06）

`scripts/report_healthbench_agentgraph_development.py` 为 AgentGraph-only
结果提供必要的离线报告适配，直接复用本项目既有
`report_healthbench_failure_demos._turn_view/_execution_view/_json_details`、
`report_multidataset_stable_zero._communication_envelopes/_react_trace_entries/
_tool_receipts` 与 `_atomic_text`。旧报告要求 Direct/paired 文件，本轮没有
这些 arm，故不伪造 paired 数据，改为只投影原生 AgentGraph metrics。
合并成功与失败执行中已有 receipt，避免重复计算 continuation 历史。
显式 operational-failure 模式保留固定5题及2题N/A；不重评、不失败置零。

真实运行3/5完成、2题collect超时，完整五题分数N/A；已完成子集不能代表
完整 benchmark。新增来源收益未证实，不能因定向测试通过而标为闭环全部修好。

## 2026-09-06 — v2.40：查询保真、阶段来源绑定、取消诊断

| 修改位置 | 真实复用与必要适配 |
|---|---|
| `healthbench_evidence_adapter._routed_evidence_receipts/_structured_evidence_artifact_error` | 将本项目 `healthbench_knowledge_tools._routed_receipts` 的既有16-envelope有界遍历移为共享函数；知识索引与completion校验使用相同真实上游、peer、历史receipt。只新增来源绑定入口，不把来源灌入Tool控制预算，不允许自由摘要/contract充当来源。 |
| `agent_runtime._execute_block/_request` | 用户MD §3.3要求有限DRAFT→REVISION；沿既有FlowSteer-derived Runtime四次调用与UpstreamMessage边界，给自身REVISION保留自身DRAFT真实来源。复用reference-only historical_evidence，不新增自边、不进入live input versions，不影响QA专用语义分支。 |
| `healthbench_clinical_react._tool_action_error/_model_visible_observations` | SkillFlow `training/task_prompts.py` 的具体实体query及 `runtime/bounded_agent.py::execute_turn` 的非法动作→Observation→继续。必要task适配：显式 `require_initial_query_fidelity` 仅保护单user、1–6关键词、可直接检索的原输入首搜，反馈原始query；无医学名称、答案或rubric模板。后续仍由Agent自由选择来源、refinement或completion。 |
| `healthbench_evidence_adapter._evidence_preserves_query_anchors` | 复用现有相关来源计数；空excerpt的标题/元数据不当成已取得正文，仍遵守原总Tool预算。 |
| `rollout_collector.collect/_emit_partial_trajectory`、completion runner | 复用FlowSteer edit→execution→history和既有TurnRecord/Runtime receipts；参考SkillFlow bounded_agent公开事件及rollout失败边界。必要诊断适配：默认关闭的callback仅将取消/异常前真实状态写独立evaluator-private文件，complete=false、non_scoreable=true、evaluation=null；原异常继续传播，绝不进入评分/训练/成功checkpoint。 |
| `train_agentgraph_smoke._healthbench_tool_runtime_settings/_runtime_for_task` | 只接线推理工厂中的opt-in布尔开关，不调用训练函数。 |

Director仍为minimal-neutral.v20，提示词没有修改。原5个开发task、模型目录、
generation、seed、并发4、900秒时限、20轮上限及官方grader均保持原值；
没有重新解释试验名或向模型提供评分目标。Europe PMC官方协议确认默认
relevance排序，未发现确定排序bug，因此未改其客户端。
