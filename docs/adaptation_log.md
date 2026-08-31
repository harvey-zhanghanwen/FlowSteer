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

### 10. Receipt-backed failure-demo taxonomy

The offline reporter `scripts/report_healthbench_failure_demos.py` reuses the
completion runner's frozen `wrong_demos.jsonl` diagnosis, the saved
AgentGraph trajectories, and the existing multidataset
CommunicationEnvelope/ReAct/Tool receipt extractors. It made zero model,
Tool, grader, training, or architecture calls. Its Wrong Demo population is
the 496 of 525 AgentGraph outputs whose Professional length-adjusted score is
below the full-score threshold.

Terminal precedence produces the following mutually exclusive primary
taxonomy:

| Primary failure class | Count | Wrong Demo share | Full-set share |
| --- | ---: | ---: | ---: |
| Rubric-level response-quality shortfall | 358 | 72.1774% | 68.1905% |
| Professional response-length adjustment | 81 | 16.3306% | 15.4286% |
| Recovered Canvas/graph/relation edit anomaly before `FINISH` | 34 | 6.8548% | 6.4762% |
| Recovered Director action-parsing anomaly before `FINISH` | 1 | 0.2016% | 0.1905% |
| `max_rounds` terminal failure | 22 | 4.4355% | 4.1905% |

The rubric class separates into 271 unmet-positive-only, 25
triggered-negative-only, and 62 combined cases. The recovered Canvas class
separates into 27 rejected `SET_RELATION`, 5 rejected `MODIFY_AGENT`, and 2
rejected `ADD_AGENT` edits. The 22 terminal failures separate by first
observable receipt into 21 graph/Canvas anomalies and one Director parsing
anomaly. The report selects at least one deterministic frozen representative
for every non-zero subcategory; task IDs are not hard-coded.

Final Retrieval/Tool, Agent message transport/runtime, Formatter/output
parsing, evaluator/canonicalization, and provider/collection failure counts
are zero. Hidden reasoning, semantic use of a delivered message, and failure
of a mandatory Verifier are not independently observable under this open
search-space condition and are reported as `N/A`, not invented causes.
Historical recovered attempts remain separate from the final taxonomy.

The tracked redacted report is
`reports/healthbench_professional_official_v1/failure_taxonomy_report_zh.md`.
Full conversations, signed rubrics, the non-scoring physician completion,
candidate outputs, Director/Canvas/Agent traces, actual communication bodies,
terminal receipts, and rubric-level evaluator receipts are written only to
`artifacts/healthbench_professional_official_v1/evaluation/evaluator_private/`.
That private diagnostic output is excluded from Git and must not enter a model
prompt or training input.

### 11. Versioned best-profile selection

The same-split, same-denominator, same-evaluator comparison found one eligible
completed HealthBench Professional AgentGraph condition:
`healthbench_professional_official_v1`. The 128-case validation runs, the
two-case canary, prepared-only artifacts, and the separate MedRAG Tool
protocol were excluded rather than compared as though they were public-test
results.

`config/healthbench_professional_best_profile_v1.yaml` now records the exact
executable configuration, evaluated source commit, protocol/evaluator
versions, manifest/report evidence, architecture settings, formal metrics,
Direct comparator, and excluded conditions. Its `next_run` entry selects
`config/evaluation_healthbench_professional_official_v1.yaml`. The repository
current pointer `config/healthbench_professional_best_profile.yaml` selects the
same versioned descriptor and executable configuration. The repository has no
automatic global default resolver, so the runner still requires that config
path explicitly; neither pointer renames or fabricates a new evaluated
condition.

The selected strict full-denominator Professional length-adjusted score is
20.2395% on 525 public-test tasks, versus 19.1728% for matched Direct, a
+1.0667 percentage-point difference. Strict raw scores are 22.6451% and
18.9721%, respectively. The condition has 503 evaluator-valid `FINISH`
trajectories and 22 reportable `max_rounds` terminal failures. No rollout,
grader call, training, Tool, Skill, policy, model weight, or orchestration-core
change was made during profile selection.

## 2026-08-28: HealthBench Professional inference-loop v2

- Preserved official-v1 and created the recoverable branch
  `feature/healthbench-professional-inference-loop-v2-20260828` plus the
  pre-change backup ref `backup/pre-healthbench-inference-loop-fix-20260828`.
- Added a new inference-only paired-evaluation config and model catalog; old
  artifacts and reports are not resumed into the new condition.
- Connected generic scalar Canvas actions to the existing v3 live target
  domains so no-op, self-loop, cycle, oversized reciprocal block, and other
  validator-invalid relations are excluded before parameter generation.
- Added revision-local memory for rejected relation actions. The exclusion is
  automatically recomputed after an accepted edit changes the graph revision.
- Added an opt-in finish-only action mask after the existing complete graph,
  current Output artifact, terminal protocol, and environment gates all admit
  explicit `FINISH`; terminal validation itself was not weakened.
- Added opt-in admission for informative free-text contracts. Opaque labels
  and exact duplicate declarations are rejected transactionally; no medical
  role inventory or workflow template was introduced.
- Added task-local, Tool-free reasoning component reuse for exact semantic
  input identities and persisted explicit reuse receipts. No model call,
  token, latency, or artifact version is fabricated on a cache hit.
- Forwarded SGLang `repetition_penalty=1.05` only behind an explicit local
  backend capability and persisted the requested decoding value. No training
  reward, retry generation, or weight update was added.
- The deployed SGLang reports an unset `max_running_requests` as JSON `null`,
  which is an upstream-supported backend-default state. The new evaluation
  condition explicitly admits and records that state as `backend_default`;
  historical conditions keep the prior strict positive-integer preflight.
- Training, GRPO, backward, optimizer update, LoRA publication, MACE,
  Bayesian update, Skill evolution, and Tool use remain disabled.
- Evaluation status: all 525 Direct responses and all 525 AgentGraph raw
  trajectories were generated. AgentGraph reached explicit `FINISH` on
  525/525 tasks with zero `max_rounds` or terminal failures, compared with
  503/525 and 22 terminal failures in official-v1. The pinned reference grader
  returned valid receipts for 510 Direct and 488 AgentGraph responses before
  the configured grader account exhausted its quota. On the 473 same-task
  complete cases, `overall_score_length_adjusted` is 18.3629% for Direct and
  20.5584% for AgentGraph (+2.1955 percentage points). Fixed-denominator
  lower bounds are 17.2993% and 18.3264%, respectively. These are explicitly
  partial-evaluator results, not a fabricated complete 525-case official
  score; v2 therefore does not replace the completed official-v1 best-profile.
- The versioned Chinese status report is
  `reports/healthbench_professional_inference_loop_v2/evaluation_report_zh.md`.

## 2026-08-29: HealthBench Professional Artifact communication v3

- Created the independent condition
  `config/evaluation_healthbench_professional_artifact_routing_v3.yaml`; v2
  code, config, trajectories, and reports remain recoverable and are not mixed
  into the v3 AgentGraph evidence directory.
- Added an opt-in producer-context Artifact envelope containing the existing
  source Agent model, free-text contract, execution mode, optional role
  metadata, completion condition, provider finish reason, Artifact version,
  and existing Tool receipts.
- Reused the same renderer for directed and reciprocal communication. Exact
  duplicate suppression applies only to identical versioned envelopes; it
  never merges approximate medical text or independent producers.
- Made v3 component-cache identity depend on the routed Artifact version and
  producer context while retaining the legacy profile for historical configs.
- Repaired compact Canvas feedback so reciprocal `peer_draft` communication
  and cache-reused provenance remain observable. Artifact bodies stay capped
  at the existing 160-character preview in Director feedback.
- Preserved matching HealthBench Direct generations across evaluator failures;
  retries grade the same response instead of generating a replacement.
- Retained Qwen3.5-9B, the empty Tool condition, `repetition_penalty=1.05`, the
  fixed 525 public-test selection, and the pinned HealthBench reference
  evaluator. No Skill, training, GRPO, LoRA, backward, optimizer, MACE, or
  Bayesian path is enabled.
- Static verification completed: the focused Gateway/Runtime/Canvas suite
  passed 255 tests plus 56 subtests, and config/completion-runner tests passed
  51 tests plus 23 subtests. Live Stable Zero and full evaluator results are
  recorded only after their corresponding commands actually finish.
- Prepare-only selected the same ordered 525 public-test tasks as v1/v2. Two
  canary attempts were stopped at the synthetic evaluator preflight before any
  benchmark generation. After loading the existing project environment, the
  pinned grader returned HTTP 403 `insufficient_quota` on all three bounded
  provider retries. This is recorded in the v3 manifest; no v3 metric is
  inferred from historical results.
- Added a focused regression test for failed HealthBench preflight diagnosis.
  The completion runner suite now passes 38 tests. The failure remains a
  grader-account blocker, not an AgentGraph/GPU0 runtime failure.
- Failed evaluator preflights now also write a bounded, structured
  `preflight_receipt.json` before the runner exits. The receipt contains only
  synthetic-fixture status, evaluator identity, grader telemetry, and clipped
  provider diagnostics; it contains no benchmark conversation, rubric, or
  reference response. The failed preflight remains terminal and cannot fall
  through into benchmark generation.

## 2026-08-30: HealthBench Professional retrieval-enabled paired condition

- Preserved the official/reference-compatible no-tool conditions unchanged.
  Medical retrieval is a new versioned diagnostic protocol and is not labelled
  an official HealthBench Professional baseline.
- Confirmed that the checked FlowSteer and SkillFlow source trees do not expose
  a callable Web-search backend to runtime Agents. The executable fallback is
  the existing frozen SkillFlow MedRAG textbook BM25 resource at
  `/ssd1/iclr/.private/skillflow-resources/medrag-textbooks-runtime`; no Web
  result, URL, or freshness claim is synthesized.
- Reused SkillFlow's `MULTI_HOP_QA` query policy: search for specific entities
  and, after a no-match/repeated Observation, reformulate with a standard
  synonym or expanded abbreviation. Reused its external-corpus BM25 boundary
  instead of implementing another retrieval engine.
- Kept query reformulation model-driven inside each Agent's ReAct execution
  mode. No automatic synonym dictionary was added because neither the checked
  SkillFlow implementation nor the frozen MedRAG resource contains a sourced
  synonym/abbreviation lexicon.
- Extended the HealthBench MedRAG adapter's ranked result with the corpus's
  existing `document_id` and `title`. This is a necessary provenance/Tool
  receipt adaptation; it does not create new evidence, alter BM25 ranking, or
  fabricate source URLs.
- The first synthetic ReAct canary produced one valid search result but kept
  selecting search until its turn budget was exhausted. Added the same
  state-conditioned Action-domain boundary used by the existing QA adapter:
  non-empty evidence forces the next action to `complete`; empty/error results
  retain one distinct-query pivot; Tool-budget exhaustion also forces
  completion. An exact normalized query cannot be dispatched twice in one
  Agent execution. This is an inference-time action mask, not a reward,
  medical workflow template, hidden answer recovery, or weight update.
- Reused the existing FlowSteer-derived incremental Canvas execution,
  AgentGraph runtime, Artifact communication, trajectory serialization, and
  explicit terminal boundary. Reused the current `ToolRegistry`,
  `ToolReactExecutionAdapter`, and `ToolReceipt`; no medical Agent role,
  workflow template, fixed topology, or second orchestration core was added.
- Defined the fair retrieval-enabled pair as **Single-Agent ReAct+MedRAG**
  versus **free AgentGraph+MedRAG**, with the same frozen corpus, Tool catalog,
  model/generation condition, task IDs, and evaluator. Plain no-tool Direct is
  retained only in its separate official condition and is not mixed into this
  Tool-enabled comparison.
- Prohibited use of public-test rubrics, physician/reference responses,
  canary strings, grader output, benchmark answers, or HealthBench cases as
  retrieval documents or query-expansion material. No benchmark row is
  rewritten and no paraphrased dataset is constructed.
- No training, backward pass, optimizer update, LoRA publication, GRPO, MACE,
  Bayesian update, Skill retrieval/injection/evolution, or learned medical
  memory was enabled. Retrieval smoke/evaluation metrics remain evidence-gated
  until an actually completed, receipt-backed run exists.
- Closed the remaining Runtime admission gap: the current state-conditioned
  Tool-action set is now enforced again at the dispatch boundary. A different
  search emitted after non-empty evidence or after Tool-budget exhaustion is
  rejected without a Tool call or ToolReceipt; an empty result still permits
  one distinct query pivot. This is the execution-side enforcement of the
  existing action mask, not a new medical workflow or reward.
- Wired the HealthBench MedRAG adapter to the same SkillFlow scientific
  sampling coordinate already used by `LiveSmokeBackend.collect`. The
  Single-Agent ReAct comparator constructs the matching per-task coordinate,
  validates every step-derived generation receipt, and persists a versioned
  generation identity. Resume now fails closed when the catalog/provider,
  contract, completion condition, Tool version, MedRAG source revision,
  runtime limits, or sampling receipts differ.
- Direct responses with an evaluator-invalid receipt remain frozen for
  evaluator-only retry and are not counted as completed; the runner stops
  before AgentGraph collection until the paired Direct panel is evaluator
  complete. AgentGraph resume independently checks the current raw scientific
  sampling receipt, so a seed, schedule-purpose, task-coordinate, or anchor
  mismatch cannot reuse an earlier trajectory. The report also refuses the
  paired-comparison label when a free Graph selected reasoning or mixed
  executor modes that lack matching per-step ReAct sampling receipts.
- The retrieval Tool protocol is versioned as
  `skillflow.medrag-textbooks-bm25-react.state-conditioned.v4`. No completed
  formal v3 retrieval prediction or AgentGraph trajectory existed, so this
  version change does not mix or relabel a prior benchmark result. The frozen
  525-task selection remains unchanged.
- After the restored grader account passed credential routing but the exact
  grader endpoint returned transient HTTP 429/500 upstream-load failures, the
  existing bounded reference-worker retry setting was raised from three to
  six physical provider attempts. The grader model, reasoning effort,
  evaluator code, rubric aggregation, samples, generation condition, and
  score are unchanged; every retry remains visible in the grader receipt.

## 2026-08-30: HealthBench authoritative retrieval v1

- Kept the Tool-free and MedRAG-only HealthBench conditions recoverable and
  introduced a new artifact/config namespace for authoritative retrieval.
- Reused the frozen 125,847-chunk SkillFlow MedRAG textbook corpus and the
  existing ToolRegistry/ReAct/Canvas/trajectory path without adding a second
  orchestration core.
- Added a single aggregate `healthbench-authoritative.search` capability. It
  returns the existing textbook evidence plus bounded live PubMed evidence
  from NCBI E-utilities with source-level status and provenance receipts.
- Did not add HealthBench questions, task IDs, rubrics, physician responses,
  reference answers, grader output, or benchmark answer databases to the
  retrieval source or Tool schema.
- Generalized the HealthBench runtime condition parser so MedRAG-only and
  authoritative-retrieval modes remain separately versioned and fail closed on
  an incompatible Tool ID, source, runtime limit, or condition ID.
- Replaced the prior first-hit-forces-completion behavior only in the new
  condition: an Agent may issue one materially distinct supplemental clinical
  query before completion, while duplicate searches are rejected.
- Added a strict `oneOf` response schema for the simultaneous `search` versus
  `complete` state to reduce StructuredAction parsing failures without fixing
  an Agent role, medical workflow, or topology.
- Made the model-admissible Canvas Tool transition atomic so a Director can
  change `execution_mode` and `allowed_tools` together rather than repeatedly
  proposing a validator-invalid reasoning-Agent intermediate state.
- Preserved Qwen3.5-9B on GPU0, the simple-evals rubric aggregation, the fixed
  public-test task order, and inference-only execution. No backward pass,
  optimizer update, LoRA, GRPO, MACE, Bayesian posterior, Skill injection, or
  Skill evolution is enabled.
- Metrics are not recorded in this entry until a receipt-backed canary or
  complete evaluation actually finishes. A live PubMed Tool condition is
  reported separately from both the Tool-free and frozen MedRAG conditions.
# 2026-08-30 — HealthBench authoritative retrieval canary receipt repair

- The first paired canary completed both Direct responses and rubric grading,
  then failed before AgentGraph execution with
  `ReceiptValidationError: v3 MODIFY field/Agent receipt differs from the parsed atomic patch`.
- Root cause: the Director correctly sampled the Runtime-registered atomic
  `execution_mode + allowed_tools` profile, while the rollout receipt validator
  still expected a legacy one-field `MODIFY_AGENT` payload.
- Necessary project adaptation: receipt validation now accepts only a complete
  four-field execution-profile transaction and verifies that the coupled pair
  exactly matches one live JSON-Schema branch.  Legacy one-field MODIFY
  validation remains unchanged.
- No task text, rubric item, reference response, medical role, or topology was
  added to the orchestration core.

## 2026-08-30: HealthBench authoritative retrieval scope-preservation v2

- Completed the authoritative-v1 two-task canary under its frozen condition.
  Single-Agent ReAct+MedRAG+PubMed scored mean
  `overall_score_length_adjusted=0.6341221333`; free AgentGraph scored
  `0.4700098333`, a measured delta of `-0.1641123`.  Both tasks reached valid
  rubric grading and explicit `FINISH`.  These two cases validate the chain
  only and are not a 525-task benchmark estimate.
- The trajectory receipts showed that each free Graph used one Agent and that
  `SET_OUTPUT` re-executed it after evidence-backed output already existed.
  The second call repeated retrieval and replaced the earlier Artifact.  The
  complete HealthBench conversation was present at Runtime; the failure was
  not missing task input or broken Agent communication.
- Reused the main project implementation in which `SET_OUTPUT` is pointer-only.
  Existing Artifacts are preserved; a genuinely missing Output Artifact still
  executes through the generic Runtime boundary.
- Reused the same generic execution contract for ordinary Output and
  non-Output Agents.  Added one neutral Director constraint requiring every
  free-text contract to preserve the original task scope and output form.
  No medical role, fixed topology, fixed Agent count, rubric criterion, or
  workflow template was added.
- Corrected Canvas feedback so a unique Output Agent is labelled `output`, not
  `format`, unless a real Formatter protocol is enabled.  Added bounded
  head--tail Artifact previews and measured truncation receipts so the Director
  can observe response coverage without duplicating the full response.
- Added regression tests for pointer-only Output selection, Artifact identity
  preservation, true output/format roles, head--tail preview, the neutral v3
  prompt, and identical generic executor contracts.  Focused tests and Python
  compilation passed; no live benchmark metric is inferred from static tests.
- Added the independent v2 evaluation condition and completed prepare-only on
  the same ordered 525 public-test tasks.  The next evidence gate is a fresh
  two-task canary; the 525-task run is allowed only if the canary confirms no
  `SET_OUTPUT` re-execution, no evidence loss, valid rubric grading, and legal
  explicit `FINISH`.

## 2026-08-30: HealthBench registered execution profiles and Canvas feedback v3

- Confirmed the scope-v2 two-task Stable Zero canary at 2/2 valid rubric
  evaluations and explicit `FINISH`; its mean length-adjusted score was
  `0.5525903` for Single-Agent ReAct and `0.4759248` for AgentGraph. These are
  canary measurements, not a 525-task benchmark estimate.
- Identified a capability-admission defect: scalar `ADD_AGENT` did not require
  `execution_mode` or `allowed_tools`, so all Graph nodes silently used
  `reasoning + []` even when their free-text contract described retrieval.
- Directly ported the main FlowSteer Runtime-owned registered execution-profile
  domain, exact constrained-decoding branches and collector receipt binding.
  ReAct remains an `execution_mode`, not a role; no medical role, Agent count
  or topology is required.
- Added revision-live Artifact receipts and ordered `AgentCallRecord`
  projection. The next Director observation now contains the accepted Canvas
  action/result, current Graph and remaining rounds, `single/draft/revision`
  phases, CommunicationEnvelope previews, reciprocal block completion order,
  public ReAct `StructuredAction → Observation` receipts, Tool/source receipts,
  current Artifact freshness/provenance and explicit FINISH admissibility.
- Kept the complete HealthBench conversation in the immutable initial Director
  context. Each later observation carries only a bounded task-goal reference;
  evidence excerpts and complete Artifact bodies remain trajectory-only to
  avoid repeated context growth.
- Added the independent condition
  `healthbench_professional_authoritative_multiagent_feedback_v3_gpt54_rubric`.
  Prepare-only validated the same ordered 525 official public-test tasks. No
  training, backward pass, optimizer update, LoRA, GRPO, MACE, Bayesian update
  or Skill evolution is enabled.
- Completed the independent two-task canary. Both tasks reached explicit
  `FINISH`, valid rubric grading and zero collection/provider failures. The
  Single-Agent ReAct mean `overall_score_length_adjusted` was `0.6257137`; the
  AgentGraph mean was `0.8053899`. These values are Stable Zero evidence only,
  not a 525-task benchmark estimate.
- Both Graphs selected the exact registered
  `react + [healthbench-authoritative.search]` profile on their first
  `ADD_AGENT`. Across the two tasks, the Graph executions recorded three
  successful Tool Observations and non-empty ToolReceipt sets. Every sampled
  action-target-domain receipt used v12; `SET_OUTPUT` made zero executor calls.
  Model-visible Director/Agent prompts contained no rubric, physician response,
  reference response, grader output or canary field.

## 2026-08-30: HealthBench Qwen3.5 thinking condition v4

- Added an independent Qwen3.5-9B catalog and evaluation namespace with
  `chat_template_enable_thinking=true` for Direct, AgentGraph Executors, and
  the Director. The frozen 525 public-test tasks, seed, Tool condition, grader,
  and v3 free AgentGraph search space are unchanged. The total per-Action
  completion budget was increased from 1,024 to 4,096 tokens, but hidden
  reasoning and the public JSON action still share it; later v4 receipts
  observed length termination, so this is not a non-truncating guarantee. v4
  is also not a controlled thinking-only ablation.
- Kept ReAct strictly as an Agent `execution_mode`. No Agent role, medical role,
  topology, or minimum Agent count was introduced.
- Wired Director thinking through the tokenizer chat template and native
  SGLang `require_reasoning` flag while retaining the existing JSON Schema for
  the public Canvas action. The server `reasoning_tokens` boundary separates
  hidden reasoning from that action; hidden reasoning plaintext is not routed
  or stored as a text field. The existing exact output token IDs/log-probability
  receipt still contains the reasoning prefix and is tokenizer-reconstructible.
- Extended Executor receipts with the explicit thinking request and bounded
  reasoning presence/count metadata. The hidden `reasoning_content` body is
  deliberately excluded from receipts and Agent communication. On the current
  OpenAI-compatible SGLang response, `reasoning_content_present=true` and the
  character count demonstrate applied thinking, while `reasoning_tokens=0`
  denotes an unavailable provider token count rather than no reasoning.
- The first v4 two-task attempt generated both Direct calls but failed closed
  before grading because its nested scientific-sampling receipt omitted the
  already requested thinking flag. Added that missing result-affecting field to
  the existing receipt projection; no model, prompt, sample, Tool, or evaluator
  condition changed.
- Re-ran the same two-task canary after the receipt repair. Both Direct and
  AgentGraph arms completed model generation, Tool execution, valid rubric
  grading, explicit `FINISH`, terminal Artifact storage, and full trajectory
  receipts. Stable Zero passed 2/2 with zero terminal failures. These two
  samples validate the chain only and are not a 525-task score estimate.
- The 525-task v4 formal evaluation was started only after the successful
  Stable Zero gate. No training, GRPO, LoRA, backward pass, optimizer update,
  MACE, Bayesian update, Skill injection, or Skill evolution is enabled.

## 2026-08-30: HealthBench two-phase Director and paired ReAct v5/v6

- Diagnosed the interrupted v4 formal checkpoint without altering it. Among
  125 completed Graph trajectories, all used one Agent and no relation because
  `finish_only_when_admissible=true` reduced the post-Output action mask to
  `FINISH` only. This was an action-space defect, not evidence that free
  topology had been learned or rejected by reward.
- Added SkillFlow-style two-phase Director generation: a bounded 512-token
  REASONING call followed by a separately schema-constrained ACTION call.
  Per-phase exact receipts are retained, the Canvas receives only ACTION, and
  metadata states that the two calls are not one autoregressive training
  receipt. The condition is evaluation-only.
- Set `finish_only_when_admissible=false`. `FINISH` remains legal as soon as the
  current public Output is admissible, while ADD/MODIFY/relation edits remain
  available when they can change the Graph. No minimum Agent count, relation,
  reciprocal edge, medical role, or fixed workflow was introduced.
- Preserved SkillFlow BM25 `score` and `matched_terms` through the authoritative
  evidence aggregation and bounded Director provenance; PubMed records retain
  only their real source metadata.
- The v5 two-task Stable Zero chain passed 2/2 with evaluator-valid Direct and
  AgentGraph responses, explicit FINISH, terminal Artifact, Output inbox, and
  verified two-phase receipts. It was rejected as a paired comparison because
  one Graph chose Tool-free reasoning while Direct used ReAct+Tool.
- Created an independent v6 condition rather than relabelling or resuming v5.
  v6 adds a task-scoped execution-profile allowlist equal to Direct
  `react + [healthbench-authoritative.search]`. The allowlist reuses the
  existing Runtime capability/action-mask/dispatch path and leaves Agent IDs,
  models, free-text contracts, counts, relations, Output and topology free.
- Added fail-closed preflight and provenance: the v6 allowlist must exactly
  equal Direct execution mode and Tool IDs, and is recorded in manifest,
  Direct generation identity, paired receipt, and report. The comparison must
  be labelled fixed-protocol/free-topology and not compute-matched.
- No training, backward, optimizer update, GRPO, LoRA, MACE, Bayesian update,
  Skill retrieval/injection/evolution, medical memory, or benchmark-answer
  retrieval is enabled.

## 2026-08-31: HealthBench v6 formal public-test evaluation

- Completed the frozen 525-task public-test condition
  `healthbench_professional_react_paired_two_phase_artifact_v6_gpt54_rubric`.
  No task, model, Tool, seed, generation setting, evaluator, or topology rule
  changed while the batch was running.
- The Single-Agent ReAct comparator produced 524 evaluator-valid responses;
  one ReAct execution exhausted six turns without a valid completion and is
  retained as the manifest-declared strict-zero terminal failure. Its strict
  `overall_score_length_adjusted` is `0.2380870546` and strict raw score is
  `0.2213730706` over the full denominator of 525.
- Free AgentGraph produced 525/525 evaluator-valid responses, 525 explicit
  `FINISH` actions, zero `max_rounds`, zero Graph terminal failures, and zero
  terminal parsing failures. Its strict `overall_score_length_adjusted` is
  `0.1771755622` and strict raw score is `0.1759485462`.
- AgentGraph therefore trails the comparator by `0.0609114924` on the primary
  length-adjusted metric and by `0.0454245244` on raw score. This is a
  descriptive separate-protocol comparison, not a compute-matched causal
  estimate: AgentGraph adds Director calls and may add Executor calls.
- The untrained Director selected 522 singleton Graphs and only three
  two-Agent serial Graphs; it selected no three-Agent or non-chain topology.
  All three two-Agent cases scored below their corresponding Direct response.
  The action mask left graph editing available, so this is observed policy
  collapse rather than a forced singleton topology.
- The offline mutually exclusive taxonomy contains 478 below-full-score Graph
  cases: 350 rubric/response-quality shortfalls, 40 terminal character-length
  adjustments, 84 recovered Director action parsing/recovery anomalies, and 4
  recovered Canvas relation/edit anomalies. Final Tool execution, Agent
  runtime, output extraction, evaluator operation, and max-rounds categories
  are each zero; this does not prove that every retrieved passage was
  semantically sufficient.
- Updated the existing offline failure-demo adapter to accept an absent Direct
  response only when the run manifest, paired strict-zero record, and
  append-only ReAct exhaustion receipt all agree. Ordinary population mismatch
  still fails closed. Full conversation/rubric/private demos remain outside
  Git and model input.
- No training, backward pass, optimizer update, LoRA, GRPO, MACE, Bayesian
  update, Skill injection/evolution, or benchmark-answer retrieval ran.

## 2026-08-31: HealthBench official-reference thinking/subgraph v2.1

- Selected `healthbench_professional_official_v1` as the only fully completed
  same-split, same-525-denominator, exact-reference-evaluator AgentGraph base.
  Its strict length-adjusted score is `0.2023946457`; v6 was not relabelled as
  the base because it used a different Tool/generation condition and permitted
  a non-reference grader alias.
- Added `agentgraph.director.minimal-neutral.v11`, a short topology-neutral
  extension of v10. It requires distinct free-text contracts, correct
  producer-to-consumer relation direction, and a complete user-facing Output
  artifact. It does not prescribe Doctor, Researcher, Verifier, Formatter, a
  minimum Agent count, graph depth, direction, or fixed workflow.
- Reused the existing FlowSteer `ADD_SUBGRAPH` transaction so one accepted
  Canvas edit can add and execute a functional unit of one to three Agents.
  Multi-Agent structure remains selected by the Director from live feedback;
  no structural reward or forced topology was added.
- Added the SkillFlow-compatible `thinking_budget` boundary to the existing
  OpenAI-compatible SGLang gateway. The new catalog gives Direct and every
  Executor a 4,096-token visible budget plus a 4,096-token hidden reasoning
  budget; both components and the 8,192 provider total are persisted in the
  sampling receipt. Director REASONING is 1,024 tokens and structured ACTION is
  independently capped at 2,048 tokens.
- Retained exact Artifact deduplication and no-change Agent-input reuse, and
  applied the versioned SGLang decoding condition `repetition_penalty=1.10`.
  No hard character cap, semantic sentence deletion, post-hoc summarizer, or
  repetition reward was introduced. The official length-adjusted evaluator is
  unchanged.
- Prepare-only fixed the same ordered 525 public-test tasks. The first v2
  canary preserved two Direct results but rejected both Graphs before execution
  because role-conditional action-mask v3 was incompatible with
  `semantic_protocol=none`. Those failed artifacts remain immutable. v2.1
  switched to the existing v2 model-admissible action mask with the complete
  `ADD_SUBGRAPH` schema; no sample, seed, model, Tool, or evaluator field was
  changed.
- No training, backward pass, optimizer update, LoRA, GRPO, MACE, Bayesian
  update, Skill retrieval/evolution, medical memory, Web search, or benchmark
  answer database is enabled.

## 2026-08-31: HealthBench official-reference thinking/subgraph v2.2-v2.4

- Completed the v2.2 two-task Stable Zero chain with 2/2 explicit `FINISH` and
  no terminal failure, but rejected it for formal evaluation because its
  AgentGraph raw/length-adjusted scores (`0.2875`/`0.2427531333`) trailed the
  same-condition Direct scores (`0.3000`/`0.3358974`). The first causal issues
  were duplicate relation endpoint pairs, a contract that changed a task
  quantity, and an internal `Treatment Planning Artifact / Source Provenance`
  response promoted by pointer-only `SET_OUTPUT`.
- Compared the current terminal path with the exact source revision used by
  the highest completed official-reference condition. Reused that revision's
  generic Output-versus-intermediate execution protocols rather than adding a
  HealthBench-specific response template. Added only a generic exclusion of
  AgentGraph identifiers, internal Artifact/provenance labels, repeated
  rationale, and intermediate-analysis headings from the user-facing Output.
- Kept standalone `SET_OUTPUT` pointer-only so it cannot repeat a Tool call or
  overwrite an evidence Artifact. Director v14 instead uses the existing
  atomic `ADD_SUBGRAPH(..., output_agent_id=...)` when a functional subgraph
  contains the terminal response node; relation and Output identity are then
  present before its single incremental execution.
- Added the existing two-bit relation invariant to the neutral Director prompt:
  one unordered endpoint pair is represented by one relation object. This is
  an action-legality rule, not a topology preference.
- Increased the two-phase Director limits from 1,024/2,048 to 2,048 REASONING
  and 4,096 ACTION tokens. Direct and every Executor retain thinking enabled,
  4,096 visible plus 4,096 hidden tokens, and `repetition_penalty=1.10`; the
  structured ACTION serialization phase remains thinking-off.
- A v2.3 diagnostic was stopped before AgentGraph collection after the source
  comparison showed that its contract-only workaround did not restore the
  evaluated Output execution protocol. Its partial records remain under the
  independent v2.3 namespace and are not reported as a score.
- v2.4 passed the two-task Stable Zero chain: Direct 2/2, AgentGraph 2/2,
  explicit `FINISH` 2/2, terminal/collection failures 0. AgentGraph raw and
  length-adjusted scores were `0.6208333333` and `0.5936383333`; Direct was
  `0.2000000000` and `0.2389844000`. Mean Graph response length fell from
  3,522 characters in rejected v2.2 to 2,925 in v2.4. The two-task result is a
  gate only and does not replace the completed 525-task best profile.
- No training, backward pass, optimizer update, LoRA, GRPO, MACE, Bayesian
  update, Skill retrieval/evolution, memory, Web search, or benchmark-answer
  retrieval ran.

## 2026-08-31: HealthBench heterogeneous all-thinking Executor v2.5

- Safely stopped the incomplete v2.4 formal runner at 208/525 valid Direct
  records, two old canary Graph trajectories, and zero collection failures.
  The immutable v2.4 namespace remains an explicitly incomplete condition and
  is not reported as a 525-task AgentGraph result.
- Created an independent v2.5 condition, catalog namespace, policy version,
  artifact directory, and report directory.  No v2.4 Graph trajectory is
  admissible under the new catalog version.
- Reused the existing per-node Director catalog selection path rather than
  adding a ModelRouter.  The four equal-weight Executor choices are local
  Qwen3.5-9B, Qwen3.5 Flash, DeepSeek V4 Flash, and MiniMax M3.  Models are not
  bound to roles or topology positions.
- Rechecked the current VectorEngine `/v1/models` endpoint and reused the
  2026-08-31 capability receipts.  Every admitted remote model returned
  observable reasoning content under `chat_template_enable_thinking=true`.
  GPT-4o-mini, MiniMax M2.5, and GLM 4.5 Flash were excluded from this frozen
  pool because their existing probe ended in HTTP 429 or timeout; no guessed
  alias or unverified fallback was added.
- Preserved the v2.4 local Qwen Direct entry, protocol, contract, seed, empty
  Tool condition, official evaluator revision, and grader model.  Declared
  `direct_reused_from` so the runner verifies and reuses the 208 successful
  Direct receipts, then generates only the missing 317.
- Marked `protocol_equivalent_to_direct=false`: a heterogeneous AgentGraph
  versus local-Qwen Direct delta measures the composite orchestration/model
  selection system and cannot isolate architecture alone.
- No training, backward pass, optimizer update, LoRA, GRPO, MACE, Bayesian
  update, Skill injection/evolution, medical memory, Web search, or benchmark
  answer retrieval was enabled.

## 2026-08-31: HealthBench heterogeneous action-mask v3 v2.6

- Preserved the v2.5 Stable Zero evidence: 2/2 Direct and 2/2 AgentGraph
  responses were evaluator-valid, with AgentGraph raw score 0.8875 and
  length-adjusted score 0.909109; these are gate results, not a 525-task
  estimate.
- Diagnosed the second canary trajectory's twelve consecutive rejected
  ADD_SUBGRAPH attempts. The v2 schema accepted arbitrary non-empty relation
  endpoint strings, so the Director emitted input_data, user_message, task,
  system_prompt, and similar pseudo-nodes. Canvas correctly rejected them, but
  only after generation.
- Reused FlowSteer commit 31b8c01's execution-profile-first hierarchical
  ADD_SUBGRAPH path and generalized only its Tool-owner condition for
  required_tool_id=null. The generated schema now admits one to three
  role-free Agents, every available catalog model, Runtime-registered
  execution profiles, free-text contracts, zero or one exact live relation,
  and an optional live Output Agent.
- Kept the unified search space open: no Doctor/Researcher/Verifier/Formatter
  role inventory, fixed Agent count, topology, minimum depth, or medical
  workflow was added. ReAct remains an Agent execution mode.
- Added fail-closed profile-selection, declaration, final-action, regeneration,
  endpoint, and trajectory-receipt validation. A task/message/context label
  cannot enter the relation or Output endpoint enum.
- Added independent v2.6 config, policy identity, storage/report namespaces,
  and action-mask v3. Direct reuse remains subject to the runner's existing
  task/model/protocol/seed/evaluator checks.
- Added targeted tests for the four-model all-thinking catalog, disabled
  training/Tool/Skill boundaries, heterogeneous per-node model declarations,
  pseudo-node exclusion, one-relation limit, and complete profile-first
  hierarchical receipt binding.
- No API evaluation was started before these architecture tests. No training,
  backward pass, optimizer update, LoRA, GRPO, MACE, Bayesian update, Skill,
  memory, Web search, or benchmark-answer retrieval was enabled.
