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

## Runtime/artifact/output protocol v2

The 30-task Wrong Demo receipts exposed four execution-boundary confounders
that preceded mathematical reasoning quality: Output-pointer edits resampled a
fresh Agent, FINISH could sample on a cache miss, free-text boxed/final-answer
forms failed before integer comparison, and repeated invalid Canvas actions
were visible only as untyped text.

The following minimal unified-core changes were accepted:

1. `SET_OUTPUT` changes the Output pointer and reuses immutable fresh artifacts.
   Generic non-semantic Agent prompts are Output-pointer invariant; a declared
   Format contract is likewise materialized from its role/inputs before Output
   selection rather than being created by the pointer edit.
2. In execute-on-edit conditions, `FINISH` never invokes a model. It consumes
   the current revision's fresh Output artifact or returns
   `stale_output_artifact`.
3. Output metadata explicitly persists artifact identity, producing Agent,
   graph revision, model/provider, free-text contract, tool configuration,
   exact upstream artifact provenance, and raw output. Relation edits continue
   to invalidate only their affected target/downstream closure.
4. Fan-in feedback retains every source and reports
   `candidate_conflict=true` only when a target-blind task extractor obtains
   different public candidates. It does not rank or resolve them.
5. AIME terminal extraction now accepts a bare integer, `\\boxed{integer}`, or
   an explicit `Final Answer: integer`/`Answer: integer` marker. Conflicting
   explicit candidates, malformed boundaries, missing candidates, and values
   outside `0..999` fail closed. No LLM, expected answer, symbolic solver, or
   last-number heuristic participates.
6. The existing same-request transient provider retry is unchanged in policy,
   but all failed and successful attempt receipts are now saved. The Runtime
   never changes provider/model during a retry.
7. Partial Runtime results now publish per-node `SUCCESS`, `FAILURE`, and
   `BLOCKED_BY_UPSTREAM` state while retaining every already successful
   artifact.
8. Rejections carry typed feedback codes. An identical rejected action at an
   unchanged graph revision is blocked as `repeated_rejected_action`.
   Scalar Director observation v2 exposes only live IDs, model IDs, relations,
   remaining rounds, feedback, and legal actions; it contains no mathematical
   workflow recommendation.

The directed regression suite covers Output-pointer and FINISH reuse,
relation-scoped invalidation, fan-in provenance/conflict, AIME extraction and
fail-closed parsing, same-model provider retry, partial failure state,
repeated-action rejection, and evaluator-target isolation. No training,
optimizer, LoRA, MACE, Bayesian, Skill, retrieval, or answer lookup path was
enabled.

The v2 Wrong Demo for `aime-2026/08` also showed candidate anchoring: after
an Agent emitted `244`, later sampled contracts copied that unverified value
into new Agent obligations and the Director spent the remaining turns editing
relations without issuing `FINISH`. Canvas now rejects such target-blind
candidate copying as `unverified_candidate_in_contract`; the artifact remains
available through provenance for execution-time checking. The guard neither
judges the candidate nor prescribes an Agent role, count, relation, or topology.

The same-30 v2 evaluation exposed one output-protocol false negative:
`aime-2026/09` ended with an isolated final line `29`, but the initial v2
projection rejected all explanation-bearing responses. SkillFlow's real
`training/reward.py::extract_math_answer` already includes a final-lines
numeric fallback. v2.1 ports only its deterministic narrow subset—an entire
final non-empty line containing one legal AIME integer—and also admits
SkillFlow's explicit `The answer is N` marker. It still rejects arbitrary last
numbers, out-of-range values, conflicts, and any target-dependent repair.

## Empty-artifact runtime correction

The first candidate-guard canary exposed a separate runtime defect on
`aime-2026/02`: a reciprocal revision returned HTTP 200 with
`finish_reason=length` and an empty public response. The runtime had classified
that response as a successful fresh artifact, replacing the same Agent's prior
non-empty `62`; pointer-only `SET_OUTPUT` and no-resampling `FINISH` then
faithfully consumed the invalid empty artifact.

The unified runtime now records such a completion as `EmptyAgentResponse`,
retains the exact call/retry and input-provenance receipt, leaves the failed
node without a fresh artifact, and preserves already successful upstream
artifacts in the partial result. It does not retry with or route to another
model. The corrected two-task canary completed both tasks with legal explicit
`FINISH`; task 02 again terminated with `62`.

## Runtime/artifact protocol v2.3 fixed-30 outcome

The final condition
`config/evaluation_aime2026_runtime_v2_3_artifact_guard.yaml` preserves the
same 30 tasks, catalog order, base Director weights, generation seed, neutral
prompt, action space, and 600-second task timeout. It reuses the frozen Direct
responses and reruns only AgentGraph under the accepted candidate-anchoring and
empty-artifact guards.

- Direct: `6/30 = 20.00%` strict Accuracy.
- AgentGraph v2.3: `10/30 = 33.33%` strict Accuracy, `+13.33` percentage
  points over the paired Direct condition.
- Initial AgentGraph v1: `4/30 = 13.33%`; v2.3 therefore changes eight tasks
  from incorrect to correct, two from correct to incorrect, keeps two correct,
  and keeps eighteen incorrect (`+20.00` percentage points net).
- Twenty-eight AgentGraph trajectories completed. Twenty-six issued legal
  explicit `FINISH`; two reached `max_rounds`; tasks 18 and 27 reached the
  unchanged collection timeout. Missing/invalid outcomes remain in the fixed
  denominator of 30.
- One explicit terminal output failed the fail-closed parser. Seven execution
  turns recorded structured runtime failures (`EmptyAgentResponse` six,
  `OpenAICompatibleGatewayError` one); successful local recovery remained
  visible in the same trajectory rather than being hidden or relabelled.

No optimizer step, backward pass, LoRA update, training, Tool, answer lookup,
GRPO, MACE, Bayesian update, Skill retrieval, or Skill evolution occurred.
