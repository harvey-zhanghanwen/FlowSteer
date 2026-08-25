# AIME 2026 initial-adaptation log

## Scope

This log covers only the inference/evaluation adaptation needed to make the
following path complete on the existing unified orchestration core:

`AIME problem -> Director -> Canvas/AgentGraph -> terminal output -> evaluator -> trajectory`

No training or Skill phase is part of this change.  Runtime/evaluation results
belong in the corresponding experiment report and receipts; this document does
not treat configuration or source wiring as evidence that a model run passed.

## Source decision record

### 1. Dataset population

**Decision:** Use the complete 30-row `MathArena/aime_2026` Parquet population
at revision `d2de22f3c656b4f56cf8981212186377d1e23bc3`, in source order.

**Source:** downstream SkillEval
`non_process_preparation.py`, `production_catalog.py`, and
`converters.py::convert_matharena_aime_2026_row`.

**Required local adaptation:** project rows must use the existing `TaskRecord`
transport schema.  The converter therefore maps the exact source row to that
schema while preserving the production identity `aime-2026/{index:02d}` and
public metadata. The official-only catalog invokes a thin port of SkillEval's
`PyArrowParquetRowReader`, records the fixed dataset revision, preserves the
problem string byte-for-byte, and writes empty train/validation files plus the
complete 30-task test file. It does not read the historical or AIME 2025
compatibility sources.

**Rejected alternatives:**

- The public SkillFlow revision `74be52bb6bd9f0e9e68dacb72636b75649197983`
  `data/prepare_v3.py` mixes a general AIME pool, shuffles it, and expands it to
  a generic train/eval recipe. It is not the fixed AIME 2026 production loader.
- FlowSteer revision `1c9f2ab` contains AIME 2025, not AIME 2026.
- The project's historical-AIME train/development material is not used to
  replace, extend, or duplicate the 30 official AIME 2026 evaluation tasks.

### 2. Public/private target separation

**Decision:** The Director and Agents receive the problem and legal public
metadata only. `answer` / `ground_truth` remains evaluator-only.

**Source:** SkillEval `BenchmarkPublicItem.to_rollout_task`,
`PrivateStaticBenchmarkCase`, and `PrivateStaticBenchmarkEvaluator`.

**Required local adaptation:** the unified repository stores benchmark records
in `TaskRecord`; model-message construction must project only `question` and
public metadata. Evaluator receipts may store ground truth for offline paired
analysis, but it must not appear in model-visible trajectory context.

### 3. Answer extraction and Accuracy

**Decision:** Formal scoring is canonicalized integer Exact Match reported as
Accuracy. The underlying scorer is `str(int(prediction.strip()))` followed by
an exact comparison with canonicalized accepted answers.

**Source:** SkillEval
`packages/private-evaluation/src/skillev_private/benchmarks/static.py::PrivateStaticTarget.score`
and `PrivateStaticBenchmarkEvaluator.evaluate`.

**Required local adaptation:** SkillEval admits exactly `{"answer": str}`.
The AgentGraph terminal protocol does not require a tag; it may return a bare
decimal integer or one complete `<answer>...</answer>` envelope. The final
Direct comparator intentionally uses FlowSteer's `AnswerGenerate` XML
protocol. `src/interactive/aime2026_adapter.py` applies the same optional
single-envelope unwrapping to either lane before submitting the resulting
string to the production integer rule. Only explicit `FINISH` is mandatory for
AgentGraph evaluator admission.

The adapter:

- accepts at most one complete answer boundary;
- submits its inner text, or the complete response when no boundary exists;
- records an integer-conversion failure instead of selecting another number;
- rejects multiple or malformed answer boundaries; and
- never derives, corrects, or looks up an answer.

**Not reused:** FlowSteer's last-number fallback, numeric tolerance,
symbolic-equivalence, LLM-judge, and partial-match paths. They are not the
SkillEval AIME 2026 scorer.

### 4. AgentGraph and Canvas runtime

**Decision:** Adapt the dataset to the unified core; do not create an AIME-only
orchestration stack.

**Source:** the project design document's `G=(V,E,o)` contract and FlowSteer's
progressive Canvas execution/feedback loop.

The AIME condition keeps:

- free-text Agent contracts and per-node model selection;
- independent, directed, and bounded bidirectional relations;
- one unique Output Agent;
- existing graph validity checks and finite reciprocal execution semantics;
- one Canvas-consumed atomic action per Director turn; and
- execution feedback before the next turn.

**Not migrated:** fixed `Plan -> Solve -> Verify`, Solver/Verifier roles,
fixed three-Agent graphs, parallel-solver templates, debate, voting,
self-consistency, mandatory validation, mandatory Python, type-to-topology
rules, or few-shot workflow examples. Responsibilities may appear in a
Director-generated free-text contract but are not Agent types or initial-prompt
priors.

### 5. Director recovery and termination

**Decision:** The Director remains the unified neutral Director and retains the
atomic actions `ADD_AGENT`, `MODIFY_AGENT`, `DELETE_AGENT`, `SET_RELATION`,
`SET_OUTPUT`, and `FINISH`.

Recovery retains `PRESERVE -> DIAGNOSE -> REPAIR -> AUGMENT` only as an order
for responding to real Canvas/runtime feedback. It is not a mathematical
reasoning template. A recovery contract must not contain an evaluator target or
an unverified candidate answer.

Formal evaluation is admitted only after a legal explicit `FINISH`. If
`max_rounds` is reached first:

- the terminal status is a failure;
- the formal final answer is empty;
- historical candidate artifacts may be retained for diagnostics only; and
- no historical candidate is substituted into the evaluator.

### 6. Tool boundary

**Decision:** The initial AIME 2026 condition exposes no Tools.

**Source:** SkillEval's static public projection sets `available_tools=()` for
this workload.

This decision means there is no QA retrieval, Web search, Python, calculator,
symbolic execution, sandbox execution, historical solution database, answer
lookup, or official-solution lookup in the initial condition. Existing optional
computation adapters in the repository are not evidence that they are enabled
for this run. ReAct remains an execution strategy in the unified runtime, not
a predefined Agent role, and it has no Tool action to invoke in this condition.

### 7. Direct/AgentGraph paired protocol

**Decision:** Direct Qwen3.5-9B and the current AgentGraph use the same fixed
task IDs, public input, answer extraction, canonicalization, evaluator, and as
closely matched generation conditions as the runtime permits.

The primary reported quantity is:

- `correct / 30` and `Accuracy` for each lane; and
- the paired AgentGraph-minus-Direct difference on the same 30 task IDs.

HotpotQA token-F1 and QA retrieval metrics do not apply.

## Implementation classification

| Implementation boundary | Classification | Status and boundary |
| --- | --- | --- |
| Unified `AgentGraph`, Canvas step, execution feedback, communication, and trajectory serialization | Existing project unified core retaining FlowSteer-derived execution boundaries | No AIME-specific core semantics or role enum is added. |
| Fixed AIME 2026 Parquet source plan, PyArrow reader, and exact row schema | Downstream SkillEval reused/thin-ported | The official-only catalog retains the exact 30-row MathArena population, source order, question text, and dataset revision; train and validation files are empty. |
| Private integer scorer | Downstream SkillEval reused | Integer canonicalization and exact comparison define Accuracy. |
| `TaskRecord` dataset conversion | Project-specific thin adaptation | Necessary because the unified runtime consumes its existing dataset schema. |
| Optional `<answer>` to `{"answer": str}` conversion | Project-specific compatibility adaptation | Applied identically to Direct and AgentGraph. FlowSteer Direct uses its upstream XML protocol; AgentGraph remains free to emit a bare integer or one complete envelope and does not require a formatting topology. |
| Unified `EvaluationOutcome` receipt | Project-specific thin adaptation | Carries Accuracy and parsing diagnostics without changing the scoring rule. |
| Fixed FlowSteer mathematical workflows and loose answer extraction | Not migrated | Incompatible with the no-orchestration-prior and strict-evaluator requirements. |
| Public SkillFlow generic AIME train/eval materializer | Not used for official AIME 2026 evaluation | Incompatible with the fixed 30-task production population and private target boundary. |

## Failure taxonomy recorded by the initial condition

Wrong-demo analysis must locate the first observable failure in the persisted
trajectory and classify it using existing runtime/evaluator evidence:

- dataset/schema failure;
- invalid Director action;
- invalid graph;
- Agent execution failure;
- model/API failure;
- Tool execution failure (not expected while the Tool catalog is empty);
- output parsing failure;
- terminal failure, including `max_rounds` without `FINISH`;
- evaluator failure; or
- incorrect Agent reasoning after otherwise valid execution.

The report should record:

`first wrong turn -> first wrong action/Agent -> subsequent propagation -> terminal result`

This analysis may motivate a later hypothesis. It must not insert a workflow
template, answer, or task-specific solving rule into the initial prompt.

## Regression obligations

The initial adaptation is considered wired, but not empirically validated,
until tests or run receipts establish all of the following:

1. the loader produces exactly the fixed 30 distinct AIME 2026 tasks in source
   order and preserves the production task IDs;
2. source rows have exactly `answer`, `problem`, and `problem_idx`;
3. no QA passage, retrieval, evidence-store, search, or Tool catalog is exposed;
4. ground truth is absent from every Director/Agent/Canvas/recovery message;
5. Direct and AgentGraph call the same extraction/canonicalization/evaluator;
6. multiple/malformed answer boundaries and non-integer submissions fail
   closed without answer repair;
7. no predefined mathematical role or topology appears in the initial prompt;
8. actual relations are executed and recorded by the unified runtime;
9. only explicit legal `FINISH` is evaluated;
10. `max_rounds` produces an empty formal answer and terminal-failure receipt;
11. Director action serialization remains stable for all six atomic actions;
12. trajectory receipts preserve real Canvas actions, graph revisions, Agent
    calls, communications, token/latency/API telemetry, termination, parsing,
    and evaluator output.

## Capabilities intentionally inactive

The following remain disabled and must not be described as implemented or
executed by this adaptation:

- GRPO, backward, optimizer update, LoRA update/publication, or any training;
- MACE exploration;
- Bayesian posterior or EVSI;
- Skill retrieval, Skill injection, Skill evolution, or an initial Skill
  library;
- artificial orchestration experience or manually curated workflow priors; and
- structural/topology reward or output-format structure reward.

Any later activation requires a separate, explicitly authorized experiment and
must not retroactively change the Stable Zero source condition recorded here.

## Stable Zero canary correction: progressive relation editing

The first real AIME canary exposed a generic Canvas admission deadlock rather
than a mathematical-reasoning failure. With `execute_on_edit=true`, each
independent `ADD_AGENT` immediately produced a successful artifact. The local
PRESERVE guard then treated every successful Agent's predecessor identity as
immutable, so all later `SET_RELATION` candidates were removed. Once an Output
Agent was selected, the independent Agents could not reach it, `FINISH` could
not become admissible, and the trajectory ended at `max_rounds`.

**FlowSteer-aligned correction:** free AgentGraphs now reuse the existing
relation dirty-closure and downstream re-execution semantics after each
accepted Canvas edit. The stronger predecessor-identity guard remains limited
to the project's verified semantic-lineage protocols, where it protects
receipt-bound evidence lineage. This is a unified-core correction, not an AIME
special case, topology template, role prior, or answer-dependent repair.

The AIME bounded evaluation section now also supplies
`task_timeout_seconds=600` to the shared collector. The former root-level
execution timeout remains a model/runtime setting; it did not bound a complete
trajectory. The interrupted pre-correction canary is retained only as failure
evidence and is not reused as a post-correction result.

## Direct comparator source-alignment correction

The first 30-task Direct collection used the correct local Qwen3.5-9B model,
fixed tasks, generation seed, extractor, canonicalizer, and evaluator, but its
contract requested an unexplained bare integer. With the local catalog's
non-thinking chat template this produced almost no mathematical reasoning
content. That collection remains useful for diagnosing the protocol mismatch,
but it is not the final source-aligned Direct baseline.

Source inspection established that SkillFlow/SkillEval has no independent
single-call Direct implementation: its initial policy also runs through the
bounded action/terminal runtime. Reusing that code under the name Direct would
change the comparison condition. The corrected comparator therefore uses
FlowSteer's existing `AnswerGenerate` single-call protocol and imports
`ANSWER_GENERATION_PROMPT`, `AnswerGenerateOp`, and the
`XmlFormatter.prepare_prompt` boundary directly. Its one `<answer>` field then
enters the unchanged SkillEval-derived integer evaluator. This adjustment
affects only the Direct comparator; it does not alter the neutral Director
prompt, free Agent contracts, topology search space, Canvas actions,
AgentGraph execution, or terminal semantics.

To preserve the frozen AgentGraph condition, the unified evaluation runner now
has a bounded `--direct-only` mode. It collects/resumes Direct predictions and
rebuilds the paired report from the already admitted AgentGraph checkpoint; it
does not schedule any new Director or AgentGraph call. The report additionally
aggregates actual node/model/provider/runtime and collection-failure receipts
from the existing trajectories. No training, Skill, MACE, Bayesian update, or
Tool was introduced.

## Initial full-evaluation outcome

The final source-aligned condition ran the fixed 30 official tasks. The
Qwen3.5-9B Direct comparator produced `9/30 = 30.00%` Accuracy. Twenty Direct
responses ended with `finish_reason=length` at the upstream-aligned 4096-token
limit and had no complete admissible answer; all twenty correctly failed
integer conversion. Of the ten normally stopped Direct responses, nine were
correct.

The frozen AgentGraph checkpoint contains 29 trajectories: 25 legal explicit
`FINISH` trajectories, four reportable `max_rounds` terminal failures, and one
collection timeout. AgentGraph produced `4/30 = 13.33%` strict Accuracy, a
paired difference of `-16.67` percentage points from Direct. Five explicit
`FINISH` outputs failed strict terminal parsing. The two-task Stable Zero canary
passed, but the complete 30-task run did not satisfy full-panel Stable Zero.

These results are retained as the initial untrained architecture condition.
No loose parsing, historical-candidate recovery, answer lookup, workflow
template, training update, or Skill was added in response to the score.

---

# HealthBench Professional initial-adaptation log

### Scope and current status

This bounded adaptation covers:

`full conversation -> Direct or Director/Canvas/AgentGraph -> complete assistant response -> rubric evaluator -> trajectory/evaluation receipt`

It is an inference/evaluation adaptation only. The source and interface
decisions below were frozen before execution. The implementation and two-case
Stable Zero chain are complete; the 525-case paired request population has
been executed and the final reference-compatible report is persisted.
Canary scores are retained only as chain diagnostics and are not reported as a
benchmark estimate.

### 1. Official public population and schema

**Decision:** Use the complete 525-record public `test` population from
`openai/healthbench-professional`, represented locally by
`/ssd1/iclr/2/datasets/healthbench_professional/healthbench_professional_eval.jsonl`.
Preserve source IDs and source order.

**Observed official row schema:**

- `id`;
- `conversation = {messages: [...]}`;
- `rubric_items = [{criterion_text, points}, ...]`;
- `use_case`;
- `type`;
- `difficulty`;
- `specialty`;
- `physician_response`; and
- `canary_string`.

**Necessary local adaptation:** Convert each record to the existing
`TaskRecord` transport and retain the complete ordered conversation as model
input. The public test population is not reshuffled, cycled into 512 training
examples, or split to manufacture development data in this adaptation.

### 2. Public/private information boundary

**Decision:** Only the conversation is task content for Direct and AgentGraph.
The complete candidate assistant response is appended only at terminal
grading.

The following are evaluator/report-only: `rubric_items`,
`physician_response`, `canary_string`, `use_case`, `type`, `difficulty`, and
`specialty`. A task ID is allowed as a receipt key but not as medical
evidence. No evaluator field may enter:

- the Director observation or prompt;
- a free-text Agent contract;
- Agent input or Agent-to-Agent communication;
- Canvas feedback or failure recovery;
- Tool arguments/observations; or
- model-visible trajectory text.

This boundary follows SkillEval's session/evaluator separation in
`src/skillev/rollout/session.py::UnskilledRolloutSessionBundle` and the private
HealthBench contract in
`protocol_v10_official.py::{HealthBenchOfficialGrader,HealthBenchNativeBackend}`.

The dataset manifest's `model_visible_fields` name refers to fields retained
in the public transport record, not to automatic prompt concatenation. The
runtime uses `TaskRecord.question` as `_workflow_problem`; for Professional it
is the lossless conversation rendering that the gateway restores to native
roles. `TaskRecord.metadata`, `task_id`, `evaluator_route`, and
`evaluator_source_id` are dataset/evaluator routing and receipt fields and do
not enter the Director or Agent task text. The source-preservation
`conversation` copy is not injected a second time.

### 3. Evaluator and metric

**Decision:** Use OpenAI `simple-evals` commit
`652c89d0ca9df547706735883097e9537d40dc47` as the pinned public reference
implementation. The internal production evaluator is not public, so the
result is named **HealthBench Professional reference-compatible score**.

The reference path is:

`complete assistant response -> one grade per rubric -> signed-point per-example score -> Professional length adjustment -> clipped dataset mean`

Required implementation details from `healthbench_eval.py` are:

1. `HealthBenchEval.grade_sample` grades each rubric independently and records
   `explanation` and boolean `criteria_met`;
2. `calculate_score` divides achieved signed rubric points by total positive
   points; a met negative rubric contributes its negative point value;
3. `calculate_length_adjusted_score` uses center 2000 characters and penalty
   0.0147 per 500 characters for the Professional option bundle; and
4. `_aggregate_get_clipped_mean` clips the final mean to `[0,1]`.

The reference grader condition is `gpt-5.4-2026-03-05` with low reasoning
effort. A provider/model/grader error is persisted and reported separately
from valid grades; it is not converted to a fabricated score. The final report
must state both the requested and valid denominators. If the exact reference
grader condition is unavailable, the runner must stop the reference-compatible
lane or label a separate local diagnostic condition. It must not silently
substitute a guessed model ID.

**Rejected evaluators:** EM, token F1, string Accuracy, medical keyword match,
embedding similarity, LLM preference without per-rubric receipts, or
similarity to `physician_response`.

### 4. Unified AgentGraph and FlowSteer boundary

**Decision:** Adapt HealthBench Professional to the current unified core; do
not add a medical orchestration core.

The following existing semantics remain unchanged:

- `Agent = agent_id + model_id + free-text contract`;
- free Agent count and per-node model selection;
- independent, directed, or bounded bidirectional relations;
- one unique Output Agent;
- `ADD_AGENT`, `MODIFY_AGENT`, `DELETE_AGENT`, `SET_RELATION`, `SET_OUTPUT`,
  and `FINISH`;
- one admitted Canvas edit followed by current-graph execution and real
  feedback before the next Director turn; and
- explicit terminal admission plus complete trajectory receipts.

The first three fields are the Agent's graph-semantic identity. Optional
`role_family`, `allowed_tools`, `execution_mode`, `artifact_type`, and
`completion_condition` values are shared execution/receipt metadata rather
than predefined Agent classes. For this frozen condition, all final nodes used
reasoning mode and an empty Tool set; Director-authored free-text role labels
were neither required nor used by graph validity or topology rules.

The upstream reference calls are
`workflow_env.py::{InteractiveWorkflowEnv.step,InteractiveWorkflowEnv._execute_workflow}`
and
`workflow_builder.py::{InteractiveWorkflowBuilder.run_loop,TurnRecord,Trajectory}`.
The existing local counterparts are
`agent_workflow_env.py::{AgentWorkflowEnv.step,AgentWorkflowEnv.execute}` and
`rollout_collector.py::AgentGraphRolloutCollector.collect`.

**Prohibited initial priors:** `Doctor -> Researcher -> Reviewer`, mandatory
Doctor/Verifier/Researcher roles, a fixed medical chain, a minimum medical
Agent count, fixed role-to-model routing, or a task-type-to-topology table.
Medical responsibilities may emerge as ordinary free-text contracts selected
by the Director after real feedback; they are not Agent types.

### 5. Tool condition

**Decision:** Use no task Tool for the initial Professional base condition.
The official public base data and public reference evaluator specify no
medical-retrieval action protocol. Existing MedRAG or Web-search adapters are
therefore disabled rather than treated as official baseline behavior.

Direct and AgentGraph must share this same empty Tool condition. Neither lane
may retrieve a HealthBench case, rubric, physician/reference response,
benchmark answer database, or data-derived proxy. ReAct remains an optional
per-Agent execution mode in the unified runtime, not an Agent role; with an
empty Tool catalog it does not create a hidden retrieval path.

### 6. Direct versus AgentGraph protocol

**Decision:** Run a smoke test first, then the complete public 525-case test
population, pairing Direct and AgentGraph on identical task IDs and conditions.

Frozen comparison dimensions are:

- complete source conversation;
- candidate model condition and generation settings;
- empty Tool condition;
- rubric reference evaluator and grader condition;
- length-adjusted aggregation; and
- timeout/provider-error accounting.

AgentGraph alone additionally records every Director action, graph revision,
Agent ID/model/contract, executed relation and communication payload, Output
Agent input/output, terminal state, token usage, latency, and provider error.
The final report must include valid/invalid grading counts, `FINISH`,
`max_rounds`, runtime failures, natural Agent-count/topology distribution, and
the first observable failure layer for Wrong Demos. It must not use test-set
errors to insert a fixed medical workflow.

### 7. Implementation classification and evidence gate

| Boundary | Classification | Status at this log entry |
| --- | --- | --- |
| AgentGraph, Canvas action loop, communication, Output Agent, model interface, trajectory | **Direct reuse** | Existing core; no HealthBench-specific change planned. |
| SkillFlow bounded execution/session and private-evaluator separation | **Direct semantic reuse** | Mapped to existing runtime and evaluator boundary; no training session activated. |
| Official 525-row schema conversion and full-conversation rendering | **Necessary HealthBench adaptation** | Validated on all 525 source-ordered public records and by the two-case chain. |
| Public reference rubric grader, signed scoring, length adjustment, and clipped aggregation | **Necessary HealthBench adaptation** | Validated with the pinned reference grader and persisted rubric-level receipts; evaluator/provider failures remain invalid grades. |
| Paired Direct/AgentGraph configuration and report fields | **Necessary wiring** | Validated by the complete 525-request Direct/AgentGraph run. |
| Medical Tool/retrieval adapter | **Not enabled** | Empty Tool condition in both lanes. |
| Medical role or topology template | **Not implemented** | Explicitly excluded from Stable Zero. |
| GRPO/backward/optimizer/LoRA, MACE, Bayesian posterior/EVSI, Skill retrieval/injection/evolution | **Not enabled** | No training or learning-state mutation authorized. |
| OpenAI internal Professional evaluator | **Unavailable** | Do not claim internal-evaluator equivalence. |

The evidence gate is satisfied by schema tests, the two-case canary, the
complete paired request run, and the persisted final report. Raw conversations,
rubric text, physician responses, grader explanations, and full model answers
remain outside this public adaptation log.

### 8. Stable Zero evidence

The fixed two-case canary completed the full terminal path for both conditions:

- Direct: 2/2 model outputs and 2/2 valid reference-compatible grades;
- AgentGraph: 2/2 complete trajectories, 2/2 legal explicit `FINISH`, and 2/2
  valid reference-compatible grades;
- evaluator privacy: rubric/reference fields were joined only by task ID in
  the private worker after candidate generation;
- Tool condition: empty for both conditions;
- training state: no backward pass, optimizer update, LoRA publication, MACE,
  Bayesian update, Skill retrieval, or Skill evolution; and
- provider recovery: transient grader HTTP 500 responses were persisted and
  retried with the bounded exponential-backoff behavior sourced from the
  official `simple-evals` sampler. Frozen candidate responses and AgentGraph
  trajectories were reused; no Director/Agent generation was repeated for an
  evaluator-only retry.

The initial AgentGraph terminal-grader wiring omission was an implementation
bug: `LiveSmokeBackend.evaluate_final_graph` did not pass the attached private
Professional grader into `evaluate_task`. The fix adds only that callback at
the existing terminal evaluator boundary. It does not change Director prompts,
Canvas actions, Agent contracts, relations, topology search, or FINISH rules.

Receipt reconciliation after the full run found a second evaluator-routing
bug. Twenty-two frozen trajectories had `termination_reason=max_rounds`,
`explicit_finish=false`, and an empty terminal answer, but the completion
runner initially sent the empty value to the Professional grader and reported
the resulting local validation error as an operational evaluator failure.
The correction directly reuses the existing AIME non-interactive terminal
boundary: an empty submission without legal `FINISH` is retained as a
reportable terminal-failure trajectory, the formal grader is not called, and
no prior candidate is recovered. Historical append-only retry receipts remain
preserved. This routing correction does not resample any Director or Agent,
change a score, or alter the unified orchestration search space.

### 9. Full 525-case paired evaluation outcome

The final report is
`reports/healthbench_professional_official_v1/evaluation_report.json` with a
concise rendering at `evaluation_report.md`. Both lanes used the same complete
conversation, local Qwen3.5-9B model condition, generation settings, empty Tool
condition, and pinned reference grader.

| Condition | Requested | Evaluator valid | Strict raw | Strict length-adjusted | Valid-only length-adjusted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 525 | 525 | 18.97% | 19.17% | 19.17% |
| AgentGraph | 525 | 503 | 22.65% | 20.24% | 21.12% |

The AgentGraph minus Direct strict length-adjusted difference is +1.07
percentage points. There are 503 graded explicit `FINISH` trajectories and 22
reportable `max_rounds` terminal failures; current operational/evaluator
failures are zero. The 22 workflows were not regenerated and were not
converted to fabricated valid grades. The fixed two-case canary remains
Stable Zero-confirmed; the formal 525-case manifest does not pass the stronger
all-task Stable Zero criterion because those workflows never reached legal
`FINISH`.

Across all 525 terminal receipts, AgentGraph naturally produced 347
single-node, 80 serial-2, 17 serial-3-plus, 39 reciprocal, 18 fan-in, 4
fan-out, 4 parallel, and 16 mixed topologies. Agent counts ranged from one to
eight and relation counts ranged from zero to six. These are observed Director
choices, not medical role or topology templates. No training, GRPO, backward
pass, optimizer step, LoRA update/publication, MACE, Bayesian posterior/EVSI,
Skill retrieval, injection, or evolution ran; optimizer updates remain zero.
