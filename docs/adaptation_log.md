# Dataset adaptation log

## 2026-09-06 — HealthBench v2.45 candidate timing and execution-mode selection

v2.44 finished with two valid FINISH cases and three 900-second timeouts;
strict fixed-five scores remain N/A. WATERFALL raw increased from 0 to 50%,
while IBD raw stayed 46.6667%. This is not an overall Skill improvement.
Three stalled cases retained a failing producer while adding consumers; the
first model/profile repair opportunity was available but not selected, and
later MODIFY was disallowed by the existing repair-exhaustion gate.

Refine only the candidate profile: act on repeated parser/action errors at
the first legal repair boundary, preserve receipts, use evidence-aware
execution-mode selection and shorter contracts. Do not add irrelevant Tool
calls to reopen recovery or bypass strict completion. Runtime, masks, models,
sampling, time budgets and evaluator stay unchanged. No new evaluation score
is implied by preparing this configuration. The fixed five are development
material, not independent Skill confirmation or a 525-case benchmark.

## 2026-09-06 — HealthBench v2.44 candidate-only iteration

The completed v2.43 dev5 collection has four valid FINISH trajectories and one
900-second timeout; full-population metrics remain N/A. Actual feedback reaches
the Director after the v2.43 fixes, but edits can still target medical prose
instead of serialization errors, preserve unsupported conclusions, or add nodes
behind a failing prerequisite. New candidate instructions target those observed
action choices and existing Tool selection; no runtime/model/evaluator changes.

Output-ingress restrictions are recorded separately from model non-adherence.
This candidate does not remove those restrictions or fabricate a working edge.
The five examples are already observed development material, not confirmatory
Skill publication evidence. No new task score is implied by this entry.


## 2026-09-06 — HealthBench v2.43 candidate

The frozen v2.42 run exposed an engineering defect independent of grader
answers: intermediate Agent declarations rejected for Output-closure dependency
incompatibility were then parsed as full ADD actions, losing the actual error
and reporting missing relations. Direct replay, rather than inference from the
contract's number of node IDs, establishes the actual original exception.
Preserve the original declaration error through the existing Canvas
feedback continuation; do not generate edges or execute a rejected declaration.

The optional candidate profile is refined to discourage answer-bearing contracts
before evidence, to distinguish empty search results from proof of absence, and
to compare conclusions with real Tool receipts and delivered artifacts. These
are general orchestration procedures, not clinical recommendations or test
answers. Development examples have already been inspected, so their scores are
not unbiased held-out generalization results. Model weights remain unchanged.

Source, testing, completion and score evidence are recorded separately. This
entry does not imply that the candidate was evaluated or improved the score.

The correction reuses `AgentWorkflowEnv._reject`: one rejected turn, no Graph
mutation, original sampled text/token/log-prob receipts retained, actual error
present in the next native Director prompt. Four new mock tests and three
targeted existing regression instances passed, with no model/API requests.

MDT's saved prefix separately showed 18 MiniMax requests consuming 640.63 seconds
across three six-turn invocations. Twelve parsing errors received the same
wrapper warning despite different existing parser exceptions. v2.43 also passes
the real parser error through Agent Observation and continuation; it does not
infer duplicate-key support, rewrite JSON, or increase retries/time limits.
Nine new ReAct mock tests and two existing targeted regressions passed for
specific parser errors, observed length termination and cross-revision
continuation. Director-facing error projection is checked separately.
One additional targeted mock test passed for trace -> public error summary ->
Canvas feedback -> actual Director prompt, including existing 400/320-character
summary bounds and retention of the original full trace. No broad tests or
paid canaries were repeated.


## 2026-09-06 — HealthBench v2.42 candidate

The v2.41 five-task runs were kept frozen. Their failures justify per-model
generation budgets, bounded source-metadata choices and an Agent-invocation
budget that can accommodate multiple ReAct turns. v2.42 changes neither the
official evaluator nor the strict evidence-span validator; it does not repair
truncated JSON by inventing missing output. New candidate rules focus on
unverified premises, specific error recovery and faithful final synthesis.

Correction to the preliminary diagnosis: HealthBench's actual subclasses
already send full-domain oneOf schemas. The generic base-class multi-action
behavior was not the executed path. No replacement union-schema implementation
is retained. Provider enforcement remains unverified; the next run records the
client's actual structured-output request parameters compactly.

Tests: new budget/hook plus existing ReAct tests 39 passed/5 subtests passed;
source schema 22 passed; response-format receipt 5 passed; new and previous
factory tests 27 passed. These are offline checks, not new task scores.

The opt-in 429 repair domain now preserves other available, execution-compatible
models from the same provider when only a model-level transient failure is
observed. Its 18 targeted tests passed; the final factory/config integration
passed seven targeted checks. No additional provider probes were performed.

## 2026-09-06 — HealthBench v2.41

Scope: repair concrete v2.40 context exhaustion and repeated evidence completion
errors, then evaluate three optional candidate orchestration conditions. No
training, backward, optimizer, LoRA publication, MACE, Bayesian or Skill evolution.

- Context repair preserves original conversation and current graph, summarizes
  redundant history/tool schema and reserves ACTION budget during REASONING.
  Offline replay of Barrett round 7 fell from 32,423 to 19,702 tokens with the
  full task and all ten messages retained; this is not a generated answer/score.
- Evidence repair keeps strict metadata and exact-span validation, but shows the
  failing item and real receipt fields so the model can repair its submission.
  Unknown source IDs are rejected early rather than querying unrelated content.
- Candidate scope/evidence preservation, local recovery and complete response
  priors reuse the existing forced-probe prompt interface; no mandatory topology
  or role is introduced. Candidate results remain separately labelled.
- Reuse existing frozen five development tasks for engineering and paired
  candidate checks; only then run the authorized single full-525 Graph condition.
  The five previously inspected test examples are not an unbiased held-out score.
- N/A remains N/A for missing operational/evaluator results. No timeout answer
  salvage, implicit FINISH, zero substitution or evaluator relaxation is added.

Implementation tests and actual scores are recorded in versioned run reports;
this log alone does not claim that evaluation has completed or improved.

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

## 2026-08-31: HealthBench held-out architecture repair v2.7

- Froze a 20-task development population from v2.6 manifest ordinals 63--82.
  It has zero task-ID overlap with the 58 v2.6 AgentGraph trajectories used for
  error analysis. All 20 matching Direct receipts are reused under the runner's
  existing model/protocol/seed/evaluator/judge checks.
- Preserved v2.6 as an immutable baseline condition and created an independent
  v2.7 condition, policy identity, artifacts, report namespace, and prompt
  version. The old and new AgentGraph runs use the same catalog-order namespace
  for the fixed task IDs.
- Added only generic Director invariants observed in real trajectories: a
  contract cannot serve as evidence for its own conclusion; ambiguity and
  internal inconsistency cannot be silently collapsed; a correction must reach
  the user-facing response producer. Enabled the existing compact historical
  Canvas projection for v15 while retaining the exact latest observation.
- Added a HealthBench task execution protocol to every free Agent invocation.
  It checks ambiguity, contradictions, relevant red flags, contraindications,
  interactions, dosing, follow-up and urgent escalation while preserving the
  requested language and response form. It does not define Doctor, Researcher,
  Reviewer, Verifier, Formatter, or any fixed workflow.
- Repaired pointer-only Output promotion by recording whether an Artifact was
  generated with `is_output_agent=true`. Under the v2.7 gate, `SET_OUTPUT`
  cannot promote an intermediate Artifact; the existing atomic terminal
  subgraph path must execute a user-facing consumer.
- Raised only the v2.7 atomic subgraph relation capacity from one to two. This
  corrects the imported ALFWorld owner-ingress cap for a three-Agent free
  HealthBench functional unit without requiring any edge or topology.
- Added a version-receipt FINISH check for reciprocal blocks. It detects when
  one member's final revision never entered Output's current Artifact lineage,
  returns `artifact_lineage` feedback with the missing Agent IDs, and leaves the
  existing `SET_RELATION -> dirty closure -> execute -> feedback` path to the
  Director. No automatic edge, transitive broadcast, or artifact rewrite is
  performed.
- Prepare-only and targeted unit tests validate the fixed population, official
  evaluator identity, disabled training/Tool/Skill boundaries, role-neutral
  v15 policy, two-relation domain, Output provenance gate, reciprocal lineage
  gate, and native HealthBench message isolation.
- No training, backward pass, optimizer step, GRPO, LoRA, MACE, Bayesian
  update, Skill injection/evolution, generic Web search, or benchmark-answer
  retrieval was enabled.

## 2026-08-31: HealthBench held-out Output closure and runtime admission v2.8--v2.15

- Kept the v2.7 fixed 20-task development population and all comparison
  conditions unchanged: task IDs/order, local Qwen3.5-9B Director, four-model
  all-thinking Executor catalog, empty Tool condition, seed, frozen Direct
  records, reference evaluator, grader, and public-test source slice.
- Added exact Output-closure admission over the existing quotient graph. A new
  Output consumer must receive one current successful Artifact from every
  quotient-DAG sink; grouped live JSON-Schema domains reject non-sink,
  duplicate, missing, and extra ingress before Canvas mutation. This does not
  prescribe a topology or a medical role.
- Preserved FlowSteer's pointer-only `SET_OUTPUT`. It remains legal only for an
  Artifact already generated under the Output protocol; otherwise the existing
  atomic `ADD_SUBGRAPH(..., output_agent_id=...)` performs one terminal
  execution after relations and Output identity are installed.
- Made an empty visible completion a typed producer-scoped Runtime failure.
  The provider call receipt is retained, whitespace-only text is never
  published as a semantic Artifact, and the responsible producer becomes the
  mandatory repair target instead of the first blocked consumer.
- Applied mandatory repair symmetrically to constrained action domains and raw
  Canvas admission. Unrelated ADD, DELETE, SET_OUTPUT, FINISH, relation, or
  wrong-target MODIFY operations cannot bypass a measured repair state.
  Existing correct Artifacts and valid lineage remain preserved.
- Added exact reciprocal final-revision lineage admission. If a reciprocal
  peer's current Artifact is absent from Output lineage, the Director receives
  only legal directed acyclic repair edges into Output or its consumed
  ancestors. The environment does not create the edge or force reciprocal
  communication.
- Stopped the first v2.14 canary after source review identified incomplete
  sink/repair admission. Its partial namespace contains no completed public
  trajectory or grader result and is excluded from v2.15.
- Confirmed v2.15 Stable Zero on the first two fixed tasks. One grader request
  returned transient HTTP 500/HTML; the existing terminal-evaluator-only retry
  reused the exact frozen Director/Canvas trajectory, recorded
  `reused_director_canvas=true`, and succeeded without another model rollout.
- Completed all 20 fixed tasks: 20/20 evaluator-valid, 20/20 explicit
  `FINISH`, zero `max_rounds`, zero terminal/runtime failure. Strict Official
  Overall Score is `44.1811%`; the official primary Length-Adjusted Overall
  Score is `39.9811%`.
- The same-task Direct scores are `27.7152%` raw and `21.5569%`
  length-adjusted, so v2.15 is higher by `16.4660` and `18.4241` percentage
  points. Relative to strict v2.12, v2.15 is higher by `4.5145` and `5.1385`
  percentage points and removes its one terminal failure.
- The completed Graphs contain eight singleton, ten serial-three-plus, one
  fan-in, and one mixed topology. The 49 Executor calls select Qwen3.5 Flash
  25 times, local Qwen3.5-9B 13 times, MiniMax M3 six times, and DeepSeek V4
  Flash five times; all finish normally. No fixed role or topology template was
  introduced.
- Seven Director actions were rejected and recovered. No accepted graph
  contains a duplicate relation or cycle. The final batch did not trigger an
  empty-completion repair, sink-reduction edit, reciprocal-lineage edit, or
  lineage fallback; their correctness is established by targeted tests rather
  than claimed from occurrence in this sample.
- Remaining losses are primarily missing clinical facts, instruction
  following, redundant serial work, and one observed Output synthesis loss
  despite correct Artifact delivery. These are retained as evidence rather
  than converted into a fixed medical workflow or sample-specific rule.
- Broad regression verification completed with 430 tests plus 75 subtests
  passing, zero failures. No training, backward, optimizer update, GRPO, LoRA,
  MACE, Bayesian update, Skill injection/evolution, memory, Web search, or
  medical Tool call ran.

## 2026-09-02: HealthBench fixed-20 v2.27 public-state admission repair

- Preserved the v2.16 fixed panel, task order, seed, scientific sampling
  schedule, Direct condition, four-model catalog and official OpenAI
  simple-evals reference grader. v2.27 has an independent condition and
  artifact namespace; no earlier trajectory or score is relabelled.
- Directly ported FlowSteer's nested `input_artifact_provenance` envelope and
  renderer. A Tool receipt remains directly attributed to the Agent that made
  the call, while a later Artifact retains the earlier routed path as nested
  provenance. Exact duplicate public receipts and envelopes are removed.
- Extended the existing scope-neutral Canvas admission to reject unsupported
  entity-slot substitutions, including single lowercase replacements, unless
  the initial task or a successful public Tool/Agent Artifact establishes the
  surface. The rule is role- and topology-neutral and reads no evaluator data.
- Added an opt-in retrieval admission for complete acronym/hyphenated entity
  surfaces. A partial token match is no longer counted as relevant evidence;
  the same ReAct Agent may issue a non-duplicate reformulation within the
  unchanged budget, and exhaustion admits an explicit insufficient-evidence
  completion rather than an unsupported fact.
- Added opt-in public text quality admission to the common Agent Runtime for
  reasoning and ReAct outputs. Provider truncation, explicit incomplete
  output, over-limit text, invalid/control characters, abnormal tokens, exact
  paragraph repetition, repeated token n-grams and extreme run-on text create
  a producer-scoped `CompletionArtifactQualityError`. The invalid Artifact is
  retained in the call receipt but is not routed downstream, truncated, or
  rewritten.
- Kept `preserve -> diagnose -> repair -> augment`: valid upstream Artifacts,
  relations and Tool receipts remain in Canvas state; the existing failure
  recovery path identifies the producer for `MODIFY_AGENT` or augmentation.
  No fixed Doctor/Researcher/Verifier/Formatter workflow, Agent count, edge or
  topology was added. ReAct remains an execution mode.
- Added targeted regressions for multi-hop provenance, model-visible nested
  provenance, scope substitution, evidence-anchor coverage, budget exhaustion,
  quality rejection and structured JSON non-regression. The focused suites
  passed before prepare-only validation.
- Prepare-only succeeded for all 20 fixed tasks and recorded all registry
  checks true, `training_enabled=false`, and `optimizer_updates=0`. It made no
  model, Tool, Web or grader call. Consequently v2.27 has no new official
  metric yet and remains `prepared`, not `evaluated`.

## 2026-09-04: HealthBench v2.28 dependency-consistent structured-evidence flow

- Preserved the v2.27 provenance, scope, retrieval and Artifact-quality gates
  and the existing fixed-20 and full-525 populations. v2.28 changes only two
  opt-in information-flow controls; it does not prescribe a medical role,
  Agent count, model assignment, relation, topology or Output identity.
- Directly reused FlowSteer's atomic Canvas transaction, directed
  producer-to-consumer Artifact routing and reciprocal initial-draft/revision
  barrier. A reciprocal pair remains two independent initial Artifacts followed
  by one bounded peer revision; v2.28 does not create or reinterpret edges.
- Directly reused SkillFlow's public Action--Observation and Tool-receipt
  boundary. Complete Tool receipts, provider observations and nested Artifact
  provenance remain in the request, trajectory and backend record throughout
  the run.
- Added a conservative free-text incoming-dependency admission. When enabled,
  an explicit dependence on a named `node_*` Artifact/output/result/finding/
  evidence/analysis/response requires that producer to reach the consumer in
  the directed Canvas. Unknown, self and reverse-only dependencies are
  rejected; outgoing `send/route ... to node_*` prose is ignored. Reciprocal
  exchange remains legal because it contains the required direction.
- Added a receipt-bound `healthbench.structured-evidence.v1` Artifact for
  non-Output HealthBench ReAct completions under the v2 communication profile.
  It retains a substantive summary, evidence claims and qualifiers, exact
  document metadata and evidence spans, plus uncertainties. Evidence identity
  and normalized spans must bind to successful authoritative-search receipts;
  invented or mismatched evidence is rejected.
- Added a compact model-visible projection of the structured Artifact. Only
  evidence cited by the Artifact and compact nested lineage are shown to the
  next model; unrelated result bodies, duplicated Artifact bodies, timing,
  corpus statistics, Tool versions and raw nested receipts are not replayed.
  This does not discard information: the full receipts and provenance remain
  in the trajectory/backend record.
- Kept Direct and terminal Output as complete natural-language responses.
  `request.is_output_agent`, rather than a role or topology label, defines the
  Output boundary. ReAct is only an execution mode, and non-HealthBench
  communication remains unchanged.
- Added the isolated fixed-20 and full-525 v2.28 configurations. Targeted
  no-API tests cover their invariants, profile routing, intermediate/Output
  schema separation, exact receipt binding, compact cited-row and nested-
  provenance projection, and non-HealthBench isolation. This log records only
  the implemented and tested state: no v2.28 live model/Tool/grader evaluation
  or score is claimed.
- Training, backward, optimizer update, GRPO, LoRA, MACE, Bayesian update and
  Skill retrieval/evolution remain disabled.
- The v2.28 fixed-20 and full-525 manifests completed prepare-only validation
  with the same ordered task IDs as v2.27. A live Stable Zero run was not
  started on 2026-09-04 because physical GPU 0 had only about 21 GiB free while
  two task-external Qwen3.5 services remained resident. Three bounded startup
  checks of a task-owned 32K SGLang Director exited before readiness: standard
  loading lacked memory for the hybrid state cache, an explicit minimal cache
  still lacked prefill/KV headroom, and SGLang's CPU weight-offload path failed
  for Qwen3.5 with a cross-device LayerNorm error. Port 8025 was closed and GPU
  memory returned to its pre-check level after each failure. No selected task,
  Tool, Executor, Web search, provider or grader request was issued, so no new
  score is claimed and the v2.27 full-525 result remains the latest completed
  official-evaluator result.
- After the user authorized a different rollout GPU, v2.28 completed a two-task
  Stable Zero canary and the full fixed-20 development panel through the
  already-running GPU-2 SGLang `supervisor_theta`. The server receipt reported
  the same Qwen3.5-9B base-policy version (`default`), 32K context and native
  token receipts; the manifest records `effective_rollout_physical=2` rather
  than relabelling the run as GPU 0.
- The full fixed-20 v2.28 AgentGraph result is 20/20 evaluator-valid, 20/20
  explicit FINISH, zero max-rounds/terminal failures, raw official score
  `0.3008484848484848`, and primary length-adjusted score
  `0.2684364548484849`. Its strict Direct comparator is `0.08081736482471777`
  raw and `0.09263322482471775` length-adjusted on the 20-task denominator;
  17 Direct records are evaluator-valid and three ReAct turn-exhaustion cases
  are frozen strict zero under the declared protocol.
- On exactly the same 20 task IDs, the completed v2.27 full-525 trajectories
  score `0.33636363636363636` raw and `0.30554361636363636`
  length-adjusted. Thus v2.28 changes the matched-panel means by
  `-0.03551515151515152` and `-0.03710716151515152`, respectively. The
  structured evidence transfer is verified in live rendered requests, but the
  treatment does not pass the score-promotion criterion and is not promoted
  over v2.27.
- v2.28 naturally selected 7 one-Agent, 3 two-Agent, 8 three-Agent and 2
  four-Agent graphs. Twelve tasks carried at least one receipt-bound structured
  evidence Artifact, and two final graphs included a reciprocal relation.
  Multi-Agent graphs averaged `0.36130536130536134` raw versus
  `0.18857142857142858` for one-Agent graphs on this small panel; this is a
  descriptive post-hoc slice, not a causal or benchmark-wide claim.
- Live failure inspection separates transport correctness from answer quality.
  One structured Artifact was transferred once and in full, but retrieval about
  regional anaesthesia was over-generalized to interventional pain guidance;
  another case failed earlier when the Director expanded an ambiguous acronym
  without public evidence and then selected a one-Agent no-Tool repair. These
  remain evidence for a future scope/evidence-sufficiency treatment rather than
  a sample-specific medical workflow. No training or Skill update occurred.

## 2026-09-04: HealthBench v2.29--v2.32 low-score regression repair

- Froze eight low-scoring public-test task IDs as a post-development regression
  panel. This panel is used only to reject or promote architecture candidates;
  it is not an unbiased benchmark score and does not replace the completed
  v2.27 525-task result.
- v2.29 completed all eight AgentGraph trajectories and official evaluations.
  Relative to v2.28 on the same panel, signed raw mean moved from
  `-0.7173056723` to `-0.1469931723` and signed length-adjusted mean moved from
  `-0.7174747223` to `-0.1637327973`; official dataset-level clipping leaves
  the eight-task aggregate at zero. This is an overall-condition comparison,
  not an attribution to one guard.
- v2.31 preserved all eight complete Director/Canvas trajectories, but the
  official grader returned `403 insufficient_quota` on six terminal calls.
  Only two evaluator-valid Graph rows exist and both are negative
  (`-1.125`, `-1.142857` raw); the other six are N/A, not zero. The two valid
  failures expose retrieval/query scope drift and explicit suppression of
  warnings. A third saved trajectory collapses a complete 267-character
  translation into a 48-character Chinese lead-in ending in `：`.
- Added immutable `public_text_quality_v2` to reject lone headings/lead-ins in
  reasoning as well as ReAct, including full-width punctuation, while keeping
  published v1 behavior unchanged. Added an opt-in HealthBench contract guard
  for the observed `without disclaimers or warnings` form while keeping the
  prior verb-led scope guard unchanged. Both failures retain valid
  upstream Artifacts and return through `preserve -> diagnose -> repair ->
  augment`; no Agent is deleted automatically.
- Restored `require_relevant_evidence=true` and added the opt-in public-task
  query-anchor admission. A query such as the observed unrelated stent/diarrhea
  search cannot execute when it preserves no substantive surface from the
  supplied conversation; the same ReAct Agent receives a public repair
  Observation. The rule reads no rubric, reference answer, reward, or private
  metadata.
- Reused the already implemented v2.28 receipt-bound structured-evidence
  schema for non-Output ReAct Artifacts. This prevents a successful Tool call
  from being treated as sufficient evidence unless the claimed evidence span
  and provenance bind to that receipt. Agent communication still transfers the
  full Artifact; terminal Output remains natural language.
- Added opt-in enforcement of the measured state-conditioned completion
  domain, closing the case where a provider ignored constrained decoding and
  completed before the required public search. When a structured Artifact
  reports insufficient evidence and one distinct search remains, the same
  ReAct Agent now receives public continuation feedback instead of publishing
  the unresolved state.
- Selected new immutable v18, which composes the existing role-neutral v16
  conflict/relation policy and v17 public task-anchor clause without changing
  either historical prompt. No Agent
  role, Agent count, model, relation, topology, or medical workflow is fixed.
- Focused no-API verification passed through the current revision: 527 tests
  plus 114 subtests, including historical profile isolation, CJK completion,
  non-Latin task input, completion-domain enforcement, refined-search
  continuation and free-topology prompt checks. The v2.32 eight-task
  prepare-only manifest completed successfully.
- The final v2.32 official grader preflight made only the synthetic
  non-benchmark check and again failed on 2026-09-04 19:26 +08:00 with three
  `403 insufficient_quota` responses. Consequently Direct,
  Director, Agent, Tool, Web/PubMed and full evaluation were not started. No
  v2.32 score is predicted or fabricated; the next safe operation is the same
  preflight after quota recovery, followed by this fixed eight-task panel.
- No training, backward, optimizer update, GRPO, LoRA, MACE, Bayesian update,
  Skill injection/evolution, fixed medical role inventory, or fixed topology
  was introduced.

## 2026-09-05: v2.32 full525 retry preparation

- The user requested recovery and the complete 525-task rerun. Reused the
  existing v2.32 full525 profile and preserved the prepared 525 task records.
- A standalone reference-grader preflight at 09:50 +08:00 made one synthetic
  request to the configured VectorEngine `gpt-5.4-2026-03-05` endpoint and
  returned `403 local:insufficient_quota`. Receipt:
  `artifacts/healthbench_professional_mixed_all_thinking_v2_32_full525_receipt_bound_completion/evaluation/grader_connectivity_preflight_20260905.json`.
  No Direct, Director, Agent, Tool or benchmark scoring call followed it.
- Corrected the worker's retry handling for permanent provider errors and
  explicit exhausted quota. Transient failures retain bounded retries.
  Official rubric prompts, grading and aggregation are unchanged.
  All 13 tests in `tests/unit/test_healthbench_professional_grader.py` passed,
  including permanent errors, structured quota errors, transient recovery,
  and exhausted transient retry budgets; these checks made no API calls.
- Corrected the full525 sampling namespace to retain v2.27's per-task sampling
  coordinates. The previous candidate changed this field while claiming to
  retain the same generation settings. No v2.32 samples had been generated.
  Read-only checks confirmed identical ordered 525 task IDs, data/evaluator/
  Direct configuration, base seed and sampling namespace.
- Clarified that the v7 catalog changes the Executor protocol for both arms:
  v2.32 Direct must be evaluated under the new condition. Reusing v2.27 Direct
  outputs as the new paired comparator is not valid.
- GPU2 and port 8015 were free when checked. Server startup is deferred until
  the official grader quota recovers. The prepared manifest is not a completed
  evaluation and v2.32 scores remain N/A.
- After provider quota recovery, use the existing SGLang startup script on
  GPU2 with `--max-running-requests 8`, then run
  `scripts/evaluate_completion_benchmark_round.py` with the v2.32 full525
  configuration and `--canary-only`. On a passing Stable Zero result, invoke
  the same configuration without `--canary-only`; the existing runner resumes
  compatible completed predictions and trajectories and fills the 525-task
  panel. Do not use the low8 canary as a substitute for this panel.

## 2026-09-05: v2.33 evidence/context repair and isolated replay

- Preserved the v2.32 source at `ef0a48f` and pushed the new isolated branch
  `feature/healthbench-v2.33-evidence-context-20260905` before accepting edits.
  Worktree: `/ssd1/iclr/1/.tmp/FlowSteer-healthbench-v233`.
- The user explicitly requested switching to the new version without finishing
  the old evaluation. The task-owned v2.32 evaluator received SIGINT and exited;
  its 385 evaluator-valid Graph records, pending evaluator records, failures and
  original source remain in the old worktree. Its manifest is marked
  `interrupted_for_v233`; this is not a full525 result. A pre-stop manifest copy
  is retained in `runtime/switch_to_v233_20260905.json`. GPU5 SGLang on port8025
  remains available; no external project service was touched.
- Added immutable neutral Director v19 and communication v3. The treatments
  target public-task relation/context alignment, unsupported nonexistence
  conclusions, and loss of uncited retrieval evidence in model inputs. They
  preserve role/model/topology choice, per-Agent ReAct mode, functional-subgraph
  incremental execution and explicit FINISH. No additional model call is
  required by the new projection or prompt.
- Bounded evidence projection keeps successful source excerpts even when the
  producer rejects or omits citations. Receipt-bound producer claims/qualifiers
  remain distinct from raw unendorsed retrieved evidence. Exact duplicate
  documents are not repeated; new interpretations of a repeated source survive.
  Partial projections are explicitly marked rather than declared complete.
- Registered v3 in the existing configuration/runtime factory, retaining the
  existing structured-evidence completion validator for both v2 and v3.
- The new full525 config retains the public task order, seed, model catalog v7,
  thinking settings, generation budgets, concurrency4, task timeout900 and
  official grader/aggregation. Direct is an explicitly frozen historical
  control: 459 valid responses plus 66 strict-zero ReAct terminal failures.
  Its raw/length-adjusted full525 scores are 13.968705% / 16.756929%.
  No existing response is rewritten or regraded merely to rename the condition.
- Initial no-API verification: Director v19 tests 5 passed; evidence-projection
  v3 tests 16 passed; configuration/runtime/evidence-adapter integration tests
  109 passed plus 59 subtests. Baseline-reference and remaining legacy-renderer
  checks are recorded separately when completed. No new live score is claimed
  by this implementation entry.
- Training, backward, optimizer update, GRPO/LoRA, MACE, Bayesian updates and
  Skill retrieval/evolution remain disabled. New results use the separate
  `healthbench_professional_mixed_all_thinking_v2_33_full525_evidence_context`
  artifact/report namespace; the old interrupted run is never mixed into it.
- Final no-API checks: 94 legacy gateway/completion-runner tests passed (plus
  four subtests), and 17 new Direct-reference tests passed. The fixed525
  prepare-only run passed after the isolated worktree registry was pointed at
  the original preparation catalog recorded in the reused data manifest. No
  data, rubric, split or preparation manifest was regenerated or relabeled.

## 2026-09-06: v2.34 Director evidence feedback and completion handoff

- Candidate worktree: `/ssd1/iclr/1/.tmp/FlowSteer-healthbench-v234`, branch
  `feature/healthbench-v2.34-director-evidence-feedback-20260905`, based on
  accepted v2.33 source `3b18f7a`. v2.33 remains running from its separate,
  unchanged worktree; its scores are never relabeled as v2.34.
- After a stalled integration period, the user requested ongoing observation
  of v2.33 low-score demos and automatic v2.34 evaluation after the old run
  completes. The new work is now integrated and verified locally, rather than
  described as an already-running evaluation.
- Live Director feedback now uses bounded artifact/evidence projections rather
  than only 160–400-character previews. Ordered public ReAct observations,
  artifact/input versions, actual routing and retained failed-attempt evidence
  remain visible. A 24,000-character total budget and source deduplication
  bound repeated context; omissions are explicit and raw receipts remain in
  the original trajectory.
- Fixed projection bugs: failed-continuation input versions no longer overwrite
  the retained artifact's versions; failed node modification retains a visibly
  stale prior artifact; provider failure without a Tool trace does not hide it;
  indirect/peer evidence is visible even without a direct Tool call; seven-way
  fan-in is no longer cropped to six; draft/revision references are distinct.
- Director v20 additionally preserves a node's original responsibility when
  repairing query syntax/budget, distinguishes unknown case-specific data from
  evidence, and uses public evidence conflicts to guide free-form repair.
  These are prompt constraints, not formal guarantees of semantic correctness.
- The fixed 525-task `--prepare-only` passed with the same original data,
  evaluator, model catalog v7, thinking settings, sampling schedule, seed,
  concurrency4 and 900-second task budget. Actual resource binding remains
  GPU5 / port8025 / context32768 / mem_fraction0.82. No dataset was recreated.
- Local verification: 41 Director-v19/v20, v3 evidence and Direct-reference
  tests passed; 127 runtime/Director/config/feedback regression tests passed,
  with 23 subtests. The latter includes all 13 v4 feedback tests. No model,
  Tool, grader, backward or optimizer call was made by these tests.
- Direct stays an explicitly reused control: 459 valid responses plus 66
  strict-zero terminal failures, raw13.968705% / adjusted16.756929% on all525.
  No v2.34 AgentGraph score is claimed before it is actually evaluated.
- Operational handoff is separate from model orchestration: continuously read
  saved results, record diagnostics locally, wait for true old-run completion
  and lock release, and invoke the existing new runner only after readiness.
  Old pending evaluator results reuse their saved answers; missing old tasks
  use bounded exact resume, not unconditional resampling. Scripts do not
  autonomously modify architecture or generate Skills from test-set failures.
- Final handoff checks: 25 synthetic coordinator tests passed, including a
  harmless local-shell test of inherited flock lifetime. Ten v4 model-interface
  tests also passed, confirming unchanged downstream/peer payloads and runtime
  evidence-validator wiring. Across the four non-overlapping final test groups:
  203 tests plus 23 subtests passed, without model or evaluator calls.
- Accepted architecture commit `5b218f7` was pushed to the existing GitHub
  backup remote on the independent v2.34 branch. The handoff script, its tests,
  explicit host configuration and recovery notes are a separate operational
  commit; runtime readiness is recorded only after its push succeeds.

## 2026-09-06 — v2.35 optional clinical tools and full reevaluation request

- User requested optional search/source-read/medical retrieval/drug/calculation
  per Agent, then a complete score rerun. Independent branch:
  `feature/healthbench-v2.35-optional-clinical-tools-20260906`, based on v2.34
  `412886c`; v2.33 running source/model/sample conditions remain untouched.
- Reused SkillFlow-backed MedRAG and calculator, existing ToolRegistry and
  bounded ReAct loop, and FlowSteer Canvas/trajectory boundaries. Only source-ID
  reading, official DailyMed SPL parsing and required multi-tool registration/
  receipt projection adapters are new. Exact mapping is in `source_map.md`.
- Every compatible model can use optional ReAct tools; no fixed medical role,
  chain, mandatory retrieval, minimum Agent count, training or Skill evolution.
  Rubrics/reference responses stay evaluator-only. Tool results are external
  evidence, not a simulated patient or actual examination/prescription.
- Explicit budgets remain 6 turns and 3 tool calls per invocation, including
  continuation. Source pages preserve version/offset/truncation. Different
  pages are not lost under document-level duplicate suppression.
- Final non-overlapping local checks: 15 backend tests (+8 subtests), 66
  ReAct/projection regressions, 90 wiring/runner/V4/handoff tests passed.
  A test's expected exception class was corrected; implementation correctly
  rejected a mismatched Direct tool condition. No model calls in these tests.
- Real capability probes made 6 completion requests total, two per existing
  remote model, with thinking configured and no retry/fallback. DeepSeek and
  MiniMax passed; Qwen Flash successfully called calculator but repeated the
  same action instead of completion and remains reasoning-only. Initial probe
  setup failed before model calls because the authoritative superclass expects
  the search registration; the corrected probe reuses the complete registry
  while admitting only calculator, without loading/querying medical data.
- One real DailyMed search plus one SPL read succeeded, with body and version.
  This is an API connectivity check, not a patient-answer evaluation.
- Fixed public525 prepare-only succeeded. Because tool conditions changed,
  old Direct outputs are not reused: the unchanged completion runner collects
  matched-tool Direct followed by AgentGraph, using exact resume for already
  admitted rows. No v2.35 HealthBench score exists at preparation time.
- Per user's latest full-rerun request, the not-yet-started v2.34 evaluation is
  superseded by v2.35; only its waiting coordinator was stopped, not the live
  v2.33 process or SGLang service. New handoff config reuses the tested coordinator
  and launches after old completion and verified/pushed readiness. The public
  test collection has informed development and is reported as reevaluation.

## 2026-09-06 — v2.36：对话、医学文献与药品来源分库

- 用户要求在工具调用过程中结合健康对话和真实外部资料，分别建立可查询
  数据库。本次在独立 `/ssd1/iclr/1/.tmp/FlowSteer-healthbench-v236`
  worktree 开发，基于已备份的 v2.35 `a3dd5e6`；不修改正在执行的旧条件，
  也不将旧条件得分改名为新架构得分。
- 新增 `healthbench_knowledge_store.py`，直接使用 SkillFlow
  `skillev/benchmarks/retrieval.py::DocumentPassage/build_retrieval_index/`
  `RetrievalIndex.search/read`，复用本项目 QA 模块加载与
  `_ThreadAffineRetrievalWorker`。未新增数据库 schema 或排序算法；精确的
  源码调用与必要适配原因记录于 `docs/source_map.md` 的 v2.36 节。
- 分类为 `conversation`、`medical_references`、`drug_labels`。对话完整保留
  speaker、content、turn index，仅作当前任务上下文，绝不是医学事实依据。
  外部来源仅接收真实 MedRAG/textbooks、NCBI PubMed、NLM DailyMed 的公开
  Tool evidence；没有 rubric、reference response、人工答案或生成式改写。
- `healthbench_knowledge_tools.py` 在现有工具后端返回后追加索引；不重新发送
  检索请求。初始化仅接收当前 Agent 自己保留的 receipts，以及实际图关系
  传入的 upstream/peer evidence envelopes；不读取未连接节点输出或其他任务
  数据。Agent 总结与解释不是外部来源，不入医学证据库。
- 每个 invocation 独立目录。原公开记录追加到 `records.jsonl`，实际查询时
  lazy build SQLite FTS5；有新证据后构建新的不可变 revision 路径。原始版本、
  文档 ID、日期、URL、完整 excerpt 和分页 continuation 保留，不以 document ID
  粗粒度去重覆盖不同版本或后续页面。无证据的外部库显式为空，不生成替代内容。
- 查询使用可选 `healthbench-knowledge.search(database, query)`，不是固定角色
  或强制检索步骤。ReAct 仍为每 Agent 的执行模式；原调用预算、Action–Observation
  continuation、Canvas 执行顺序、图关系及唯一 Output 不变。按来源分库并不表示
  自动给每个 Agent 读取其他节点数据的权限。
- 这不是新下载的完整医学语料库，也不是 benchmark 答案库或跨任务 Skill/
  memory 系统。外部检索仍负责获取资料，本地索引只使本节点已有资料可复查。
  当前对话帮助约束查询中的实体、关系与适用条件，不补造患者事实。工具索引
  receipt 只报告路径、数量和状态，不反复复制完整对话。
- Store 定向验证已通过 10 项真实 upstream FTS5 测试（另有 4 个参数化子项），
  覆盖来源与请求隔离、线程亲和性、长内容保持、版本/分页保留及旧索引可读。
  测试使用临时目录和合成文档，无模型/API/grader 调用；工具接线及配置的最终
  验证由主线负责，本记录不将尚未运行的正式评测描述为完成。
- 本次没有训练、backward、optimizer update、GRPO、LoRA、MACE、Bayesian
  更新或 Skill evolution，也没有 v2.36 正式新分数。后续重评测必须使用匹配的
  Direct/AgentGraph 工具条件与官方 rubric evaluator，不能以索引可用或工程测试
  通过推断官方分数已经提高。
- 主线整合验证：工具与请求隔离/实际图关系接线、原有可选 ReAct、runtime
  allowlist 及评测协议合计 81 项测试、4 个子项通过；按数据库而非整个 store
  判断重复查询的定向测试再次通过。新增 v2.36 配置和模型目录，保持同批
  525 题、seed、并发、generation、Canvas 和 evaluator 不变；Direct/Graph
  均可使用六项工具，未复用不同工具条件的 Direct。525 题 prepare-only
  返回 `prepared`、`metrics=null`，未启动模型/API/grader。当前 v2.35 队列
  不变，v2.36 仅完成候选准备和独立备份。

## 2026-09-06 — v2.35 GPU5 Direct / GPU6 AgentGraph

User requested GPU6 AgentGraph alongside GPU5 Direct. An independent worktree
based on v2.35 a3dd5e6 retains the frozen scientific configuration and adds
single-arm collection/storage boundaries. Original sequential Direct is stopped
before Direct-only resumes, so it cannot later schedule a duplicate Graph.
GPU6 reuses existing SkillFlow SGLang startup and no downloaded model.

The original Direct entry rejected equal Tool sets because registry order was
sorted but config order was not. Repeated synchronous corpus reads then failed
before the first model call. Exact sorted-membership comparison plus retaining
config order in the stored receipt fixes both entry and strict resume; actual
registry order is additionally recorded. This is an engineering correction,
not a model/Tool/search-space change. Separate arm reports do not invent the
other arm or full-population scores for incomplete evaluations. Tests and
resource/output boundaries are documented in the parallel run report.

## 2026-09-06 — v2.37：停止 v2.35，来源分类检索与五题问题回归

- 用户要求停止旧评测，只优化现有 demo 并评测 5 题。旧 Direct/Graph PID
  6910/6912 均经 SIGINT 退出；保留 Direct 280 条有效结果、Graph 19 条有效
  结果及全部失败记录，不把中断状态记为完成。停止 receipt 随本版报告保存。
- 从 v2.36 独立 worktree 建立 v2.37；合入已验证的 v2.35 并行采集入口，
  文档冲突保留两边来源。旧版本/worktree/artifacts 不覆盖。
- 修复非空无关检索消耗有效搜索额度、长词轻微拼写导致 scope 误拒、500
  字符片段缺真实截断信息、查询 echo 被用于 contract literal grounding。
  不写临床结论、固定角色或样本答案；统一编排和官方 evaluator 不改。
- 运行前固定五个已观察低分 ID（WATERFALL、ASTRONAUT、共病管理、IBD/HIV、
  MDT），stage=development。旧版五题均分 raw 25.3561%、length-adjusted
  20.9078%；来自原始 receipt，不能外推 525 题，也不能说是 untouched test。
- 新版仅 `--collection-arm agentgraph`，沿用 GPU6/8026 的本地 Director 和
  已验证异构模型池、thinking、seed、每题预算、并发 4，不重复 canary/Direct。
  外部工具新增 v2.36 本地索引，所以旧版对照是端到端版本比较，不是单项消融。
- 无训练、GRPO、backward、optimizer、LoRA、MACE、Bayesian 或 Skill evolution。
  实测分数仅在本版五题完成后写入结果报告；不自动扩展到全量。

### 首轮真实结果与未评测的后续修复

首轮 c4fbacd 已结束：5 题中 3 题 evaluator-valid/FINISH，1 题 Director
context overflow HTTP400，1 题 900 秒 TimeoutError；五题 strict 指标 N/A。
同一有效三题：v2.35 raw 12.12%、length-adjusted 6.54%；v2.37 聚合后均为
0.00%（包含 MDT 的负分后按原 evaluator 裁剪）。没有证明提升。

已补上下文预算代码及 15 项离线定向测试；不裁剪输入，39102 等大于上下文
的输入仍本地失败，29102 输入对应最多 3666 生成 tokens。补丁没有重新运行
模型，因此不报告它的新得分，也不把首轮已完成三题改标为补丁版本。

## 2026-09-06 — v2.38 外部医学知识源接入

- 保留 v2.37 `df8adc4` 作为可恢复基线，在独立分支
  `feature/healthbench-v2.38-external-medical-sources-20260906` 开发；不覆盖旧评测。
- 用户请求扩充知识来源，本轮接入 Europe PMC 文献/可获取 OA 全文、
  ClinicalTrials.gov 注册试验/真实发布结果。沿用 MedRAG 125,847 个片段、
  PubMed、DailyMed；不是“新增下载了数百万篇全文”，也不是 benchmark 答案库。
- 通过显式 `external_medical_sources_enabled: true` 激活两个工具，扩充已具备
  ReAct 能力模型的 tool capability scope；Qwen Flash 原有 reasoning-only
  能力边界不改，不把它虚报为已通过新工具模型 canary。
- 新来源进入原 `medical_references` 索引，保留来源类型；药品标签仍独立。
  不把患者对话或 Agent 总结当外部医学事实；不检索 rubric/参考回答。
- ReAct 仍每次 Agent 调用最多 3 次真实 Tool dispatch，6 个回合；新增工具
  不独占额外预算，不强制 Agent 检索。Director 原提示词、图结构约束和
  execution feedback 边界保持不变。
- 两个接口各完成一次非 benchmark 免费 HTTP schema 检查；没有调用模型、
  grader 或重跑旧样本。定向离线测试包含查询、阅读、来源分页、去重、
  索引、实际关系传递、旧配置兼容和 Direct/AgentGraph 工具条件一致性。
- 既有五题 ID 原样保留在独立 v2.38 配置，仅 prepare-only；本版没有分数，
  不据此替换 best-profile，不自动启动全量评测。详细状态见本版中文报告。

随后用户明确授权用最新版评分同一组 5 题。离线 118 项测试、35 个子项通过
并完成 prepare-only 后，主线单独发起 AgentGraph 五题；无新 Direct、训练或
525 题运行。代码冻结在启动前备份提交，实测指标以单独结果报告为准。

五题已于 2026-09-06 完成：执行代码 `a65f9f2`，5/5 evaluator-valid、5/5
显式 FINISH；raw 35.3561%、length-adjusted 38.0050%。同题 v2.35 分别为
25.3561%、20.9078%，原始分的全部增加来自 WATERFALL。两题最终输出仅标题，
另有 IBD/HIV 在 SINGLE 转 reciprocal DRAFT 后丢失旧检索证据；本轮只定位，
没有修改冻结的 runtime。没有 Agent 实际调用全文阅读工具，不将接口可用
描述成已经利用全文。完整对照、错误分类和调用口径见
`reports/healthbench_professional_external_medical_sources_v2_38/evaluation_report_zh.md`。

## 2026-09-06 — v2.39 高质量来源与可选分类检索

- 从已保存 v2.38 的 `3b42012` 建立
  `feature/healthbench-v2.39-guidelines-sources-20260906`；旧配置和五题结果不改名、不覆盖。
- 用户要求接入已列出的高质量来源。本版接入 NCBI Bookshelf、其 PDQ/AHRQ
  指定集合和 MeSH；来源客户端由互不重叠的子任务实现，主线单点接 registry、
  schema、gateway、FTS5、配置与测试。NIH/HHS 两次真实 GET 403，明确未接入。
- 沿用 SkillFlow 的 search/read、Tool receipt、请求级 FTS5 和现有 ReAct；
  Bookshelf 只按需读取许可 XML，搜索元数据不冒充正文；MeSH 标为词表定义，
  不当成临床疗效证据。不检索 benchmark 原题、rubric 或参考回答。
- 分类工具仍由 Director/Agent 选择，不根据题目关键词硬路由或设置固定医疗
  角色。新分类检索 profile 同时允许阅读和查询已有证据，保留全工具集合选项，
  不强制全部检索，不增加调用预算。Direct 与 Graph 准备配置为同一工具条件。
- 对“分类检索是否提高评分”的结论：没有隔离该因素的配对消融。v2.35→v2.38
  同五题 raw 增加 10 个百分点且全部来自 WATERFALL，其他四题 raw 不变；
  不把长度项收益或接口接通描述成稳定临床质量提升。
- 当前只做定向离线验证和 prepare-only；本版没有新的模型/官方评分，未修改
  SINGLE→reciprocal DRAFT 证据丢失及标题式 FINISH 的既有 runtime 问题，
  不将这些问题标为已修复。没有训练、GRPO、LoRA 或全量评测。

随后用户更新本轮要求：取消分类检索执行配置，修复已知 demo 问题再重跑原
五题。新配置只保留 reasoning 与全工具 ReAct，不再按来源限制 Agent。
知识来源的出处/集合/药品标签与原对话的区别仍保存，不删除 provenance。
同五题评分已获授权，但只在修复完成及定向测试后启动，旧指标不作为新指标。

修复已完成：Output 公开任务要求正文却仅返回标题时，在原 ReAct 循环内
返回可修复的 completion 错误；SINGLE→reciprocal 的历史资料从 Canvas
已有 previous-revision 存储显式传 Runtime，旧输出保持失效，旧 evidence
供重核，不污染 control continuation。真实 env 交接、同阶段预算、并发、
合法短答及原 Runtime 定向回归已通过。测试中修正了两个不成立的假设：
调度首个请求不一定是节点 a；节点 b 先前已合法接收的 a 来源应继续保留，
只有 a 的本轮新草稿需要等待阶段屏障。这是测试断言修正，不是放宽通信边界。

2026-09-06 v2.39同五题运行已收尾，源码固定9fb27de，GPU6/8026，原并发4、
900秒单题时限不变。3题FINISH且官方grader有效，IBD/HIV与MDT各一次collect
TimeoutError。严格5题raw/length-adjusted均N/A；completed-only 3题为
33.33%/33.00%。未扩分母、未将缺失当0、未自动补跑、未训练。

已完成Barrett从标题变正文且raw100，但不是新增正文guard触发的重试；
ASTRONAUT/WATERFALL仍在Director初始contract发生语义漂移。双向阶段局部
索引保留IBD来源，但任务未收敛；超时collect路径没有独立保存完整未完成trace。
这些是剩余问题，不将v2.39称为最佳或Stable Zero完全通过。详见
`reports/healthbench_professional_clinical_reference_sources_v2_39/evaluation_report_zh.md`。

离线新增AgentGraph-only报告适配，复用现有I/O与receipt渲染；默认仍要求完整
收束，显式允许operational failure时固定五题、严格指标N/A，缺失题仅投影
本轮selected question/progress/collect失败，不构造假轨迹。没有新增API调用。

### v2.40（用户授权：修复后重跑同五题，不在提示词作弊）

从941f892创建独立候选分支，不覆盖v2.39。先定位到：上游receipt可见却不能
被completion绑定、双向自身DRAFT来源未进入自身REVISION两个确定接口缺口。
它们已用合成来源/虚构任务的真实ReAct→Runtime→Canvas→FINISH定向测试复现
并修正；不把缺失trace的旧两题超时单因归给它们。旧零分题的可见首错则为
contract把原命名对象换成推测含义，后续扩展query偏离；不向prompt提供正确解释。

本版给短原输入首搜加opt-in词面保真校验，首搜被拒时只返回原始公开query；
保持自由模型/Agent/拓扑/后续检索。metadata空正文不占有效正文搜索slot。
取消记录另存partial_trajectories.jsonl，明确不可评分，以便下次失败可直接
检查已有模型返回和Canvas记录；不延长900秒或放宽FINISH。

分组验证：检索/来源95 tests+43 subtests、Runtime相关100 tests+12subtests、
collector/runner114 tests；存在交叉，未相加成唯一总数。新增运行接线3项通过，
prepare-only通过。测试夹具中的condition_id不一致已改正，未放宽生产条件。
此时新真实评分仍N/A，等待同五题单次AgentGraph正式运行；没有训练或Direct重跑。

v2.40同五题已于2026-09-06 11:28:01 UTC结束，源码固定6b9bb5a。
仅2/5 FINISH且官方grader有效：WATERFALL raw50.00%、adjusted49.61%；
IBD/HIV raw53.33%、adjusted56.81%。ASTRONAUT与MDT各一次900秒Timeout，
Barrett因Director输入32798 tokens超过32768而拒绝。全五题两项均分N/A，
已完成两题描述性均分51.67%/53.21%，不与旧完成三题的均分宣称提升。
有效完成率从v2.39的3/5降为2/5，不提升为best-profile、不声称修复闭环完成。

partial diagnostics保存3条，可确认来源字段/原文span反复写错、JSON解析/
length和长feedback积累等失败；WATERFALL首搜反馈实际触发且找到相关文献，
但source.read使用了错误ID；IBD证据有异质性但回答省略。双向DRAFT/REVISION
在ASTRONAUT确实发生，但仍绑定失败。详见v2.40/evaluation_report_zh.md。

离线报告仅修正“没有终局轨迹等于没有任何中间证据”的过度断言，指向本轮
non-scoreable partial文件；不改变指标、运行源码、FINISH或官方评估。
报告生成不发出模型/Tool/grader请求，没有重新跑失败题或扩大到525题。
## 2026-09-07 — v2.46 固定新五题的 ReAct 工程迭代

从 v2.45 最终报告提交 `b8ffb2eba5508ee700475eb1763e044836adcba3` 建立
`feature/healthbench-v2.46-react-evidence-disambiguation-20260907`。
用户澄清是另外5道 HealthBench 题，并允许通过候选 Skill 调整职责。按官方
文件顺序仅看 ID，排除上一轮五题，选定 ordinal 0–4；后续修复不更换样本。
公开全集有历史评测，故不声称这五题是从未接触的独立测试集。

首次prepare-only在模型调用前发现配置不兼容：runner的医学工具allowlist
检查仅接受带reasoning的组合。沿既有检查增加完整ReAct-only形式；不删除
工具匹配检查、不声称与Direct等价。此失败没有调用模型或grader。
修复后prepare-only已通过，实际选择固定5题；尚未开始生成或评分。
runner真实入口37项定向测试通过，旧配置兼容且不放宽工具条件；新增测试
没有额外模型调用。各实现子任务均已结束，主线单点冻结并运行。

旧 demo 中的确定问题：单节点修改后历史 evidence 未入请求；ReAct 切到
reasoning 时公开检索记录丢失；scope guard 把普通引用要求当临床实体；
已反复发生协议错误的节点在既有恢复耗尽时不能尝试一个未用兼容模型。
本次沿既有接口作最小修复并通过定向测试，来源逐项写入 source_map。
新条件只允许 ReAct execution mode，避免通过关闭工具模式跳过错误；
保留跨版本薄适配的证据交接能力，不强制每个节点重复查库。

候选 Skill 的职责建议仍可拒绝：先识别未确定实体，保留真实证据和不确定性，
明确当前节点产物与实际依赖；按缺口核对或综合。没有预设医疗角色或强制
Reasoner→Verifier→Formatter，没有 sample/rubric/答案写入提示词。

已通过32项新增scope/recovery、21项既有scope回归、14项evidence测试及
4个参数子例、4项配置和2项工厂接线测试。本记录写入时尚未运行v2.46模型
或grader，尚无新分数；无训练、权重更新、ACTIVE发布或自动Skill evolution。
完整525只有在同一固定五题5/5 FINISH且evaluator-valid、length-adjusted
均分严格超过60%后，先备份接受版本再启动；不能将缺失计零或回收中间答案。
