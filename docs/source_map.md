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
