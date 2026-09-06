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

## Fan-in artifact visibility correction (v2.4)

The AIME Wrong Demos did not show a general Agent-to-Agent truncation bug.
Routed fan-in inputs already contain each source's complete immutable
`raw_output`.  Task 08 was not a multi-source fan-in and therefore cannot be
used as evidence that fan-in truncation caused its `max_rounds` failure.

The accepted correction addresses the observed information boundaries:

1. `UpstreamMessage` now carries the producing `source_model_id` and
   `source_contract` in addition to source/target IDs, artifact ID, complete
   artifact body, revisions, and Tool receipts.
2. The generic Agent execution protocol remains Output-pointer invariant but
   explicitly preserves contract-relevant public derivation, evidence,
   intermediate results, and checks for downstream assessment. It does not
   request or persist hidden reasoning.
3. The AIME workflow problem labels the single-integer requirement as a
   terminal evaluator protocol instead of applying “no explanation” to every
   intermediate artifact.
4. Canvas exposes compact `current_artifact_receipts` on every Director turn,
   including after a rejected edit. Each receipt contains the current
   artifact ID, model, contract, target-blind candidate, character count,
   head--tail preview, direct upstream provenance, and per-node
   `candidate_conflict` without ranking candidates.
5. Accepted execution feedback now provides the same source-bound fan-in
   summary for non-Output nodes; full routed artifacts remain in Runtime and
   trajectory receipts.

This is a runtime/artifact/feedback correction only. No fixed mathematical
workflow, Agent role, topology prior, automatic candidate adjudication,
automatic `FINISH`, Tool, training, GRPO, MACE, Bayesian update, or Skill was
added.

## AIME runtime v3: parameter mask, artifact ordering, and termination lookahead

The same-30 v2.3 Wrong Demos showed that a parseable artifact could coexist
with unrelated graph growth and that a last-round `SET_OUTPUT` left no action
for the required explicit `FINISH`.  The correction is confined to live
Canvas state and constrained decoding:

1. AIME remains on `ADD_AGENT`; a generic v3 live domain now constrains the
   neutral `node_N`, available `model_id`, and registered execution profile.
   It does not use QA role selection or introduce Solver/Verifier labels.
2. `current_artifact_receipts` now exposes target-blind parsing status,
   freshness, source Agent/artifact identity, model/contract, and direct fan-in
   provenance. `candidate_state` reports observable agreement/conflict only.
3. With a fresh parseable candidate, the live mask consumes existing artifacts
   through a relation that strictly reduces structural terminal distance,
   selects a fresh candidate-owning Output, and then exposes only explicit
   `FINISH`. Parsing failure and unresolved candidate conflict cannot be
   converted into an Output-pointer decision.
4. The Director receives the existing neutral
   `AgentGraph.construction_progress()` lower bound and remaining rounds. At a
   tight horizon, only the next legal atomic edit that preserves the explicit
   terminal path remains in the v3 domain. No action is synthesized and a
   trajectory that still lacks legal `FINISH` remains `max_rounds` with a null
   formal answer.
5. `FINISH` now reuses the same target-blind AIME extractor as the public
   artifact state. A non-empty but unparseable Output is a typed
   `output_parsing_failure`, not an evaluator-eligible terminal artifact.
6. If the exact model-admissible action domain is empty, the generic Canvas
   records `canvas_action_domain_exhausted` with public recovery/model/
   terminal state before any schema request; it does not sample outside the
   domain or synthesize `FINISH`.
7. The new prompt version is a short policy/environment contract. It names the
   public artifact and horizon fields but contains no fixed Agent number,
   mathematical role, chain/parallel/debate/voting pattern, Tool requirement,
   or Skill.

Directed regression tests cover scalar v3 domain/schema validation, neutral
IDs and live model/profile constraints, single-artifact `SET_OUTPUT -> FINISH`
ordering without re-execution, fan-in provenance and strict-progress relation
targets, target-blind conflict, parsing-failure exclusion from Output targets,
termination lookahead, unparseable-sink and terminal-parsing counterexamples,
generic empty-domain termination, and compatibility with existing
QA/runtime/collector paths. The complete unit suite reports 1064 passing tests
and 197 passing subtests. The evaluation configuration freezes a new condition and output
namespace; Direct remains the same frozen 30-response comparator.

## AIME runtime v3 fixed-30 outcome

The v3 canary completed both frozen tasks through explicit `FINISH` with valid
evaluator and full-turn receipts. The full run reused those two trajectories
and the frozen Direct predictions, then checkpointed every completed task.
Three first-pass collection timeouts were resumed without resampling the 27
successful tasks; tasks 03 and 24 completed on resume. Task 28 reached the
unchanged 600-second task boundary on two further isolated resume attempts and
therefore remains an operational failure rather than a recovered answer.

- Direct: `6/30 = 20.00%` strict Accuracy.
- AgentGraph v3: `14/30 = 46.67%` strict Accuracy; 29 evaluator-valid
  trajectories and one operational failure. Completed-only Accuracy is
  `14/29 = 48.28%` and is not used as the primary score.
- AgentGraph v2.3: `10/30 = 33.33%`; v3 changes the strict score by `+4/30`
  or `+13.33` percentage points.
- No answer lookup, Tool, training, optimizer step, LoRA update, GRPO, MACE,
  Bayesian update, Skill retrieval, or Skill evolution ran.

The local SGLang preflight additionally exposed a version-compatible runtime
receipt case: an auto-sized request pool reports configured
`max_running_requests=null` while the scheduler reports the actual positive
`effective_max_running_requests_per_dp`. The receipt now consumes that real
upstream field only when all DP states agree; it does not guess a value or
change scheduling.
## AIME runtime v4 implementation status

This pass applies only the approved runtime/artifact/output/Canvas/recovery/
termination corrections on top of the same-30 v3 condition.

Implemented:

1. AgentGraphSnapshot, Canvas history, Runtime results, failure records, and
   progressive execution state have JSON restoration paths.
2. EvidenceStore owns an append-only rollout_checkpoints stream. The collector
   persists each completed turn before the next Director request and writes a
   terminal-pending marker before evaluation plus a completed marker after the
   final trajectory.
3. Resume validates the exact task, condition, policy versions, Director
   sampling coordinate, problem, Skill condition, contiguous turn sequence,
   graph snapshot IDs, and next-round boundary. It restores the cached artifact
   and transcript instead of repeating successful model/API calls.
4. The AIME adapter parses a target-blind artifact_assessments JSON block with
   exact provenance binding and fail-closed status/counterexample rules.
5. Canvas exposes fresh candidate agreement/conflict, provenance, assessment
   status, and terminal lower bound. SET_OUTPUT/FINISH require a supported fresh
   artifact; negative assessment restricts the next parameter domain to
   lineage-local repair under the existing recovery policy.
6. The generic Agent execution protocol asks an Agent with routed AIME
   artifacts to assess every public upstream candidate while retaining a free
   contract. It explicitly supplies no role, count, relation, topology, target,
   evaluator result, or benchmark solution.
7. The v4 config keeps the v3 Director prompt and frozen model/catalog/seed/
   evaluator condition, disables all Tools, training, GRPO, MACE, Bayesian,
   Skill retrieval/evolution, backward, optimizer, and LoRA paths, and selects
   GPU0 for inference.

Directed validation completed before the paid run:

- 18 AIME adapter/runtime tests passed, including supported, insufficient,
  refuted, pointer/FINISH no-resampling, checkpoint restoration, target-blind
  parsing, legacy v3 ordering, fan-in provenance, parsing failure, and
  termination lookahead.
- The normal collector test and a forced completed-turn interruption/resume
  test passed; the latter restored all three turns while making exactly one
  Agent call total, proving that resume did not replay the first fresh artifact.
- The v4 YAML loaded through validate_agent_graph_config with the fixed
  sample_count: 30 and provenance_bound_candidate_assessment_v1 protocol.

The same-30 AgentGraph v4 evaluation result is intentionally not recorded here
until the complete fixed-denominator run has finished.


## AIME runtime v5 frozen candidate: contract grounding and artifact completeness

Status: **formal same-30 evaluation completed; candidate not selected**.

The v5 candidate is frozen in
`config/evaluation_aime2026_runtime_v5_contract_completeness.yaml`. It is
derived mechanically from v4.9 and preserves the same official 30 AIME 2026
tasks, sequential selection, seed `20260825`, catalog namespace/order, base
Qwen3.5-9B policy identity, Direct prediction source, target-blind extraction
and evaluator protocol, concurrency, and 20-round environment limit. All
artifact, evidence, manifest and report paths use a distinct v5 namespace.

The configuration enables only the approved architecture/runtime boundaries:

1. `task_specification_contract_guard=true` applies a target-blind semantic
   admission check to free-text contracts at FlowSteer's existing
   validate-before-commit Canvas transaction. It does not read ground truth or
   introduce a role enum, topology template, solving plan or Agent-count prior.
2. `artifact_completeness_gate=true` prevents a length-truncated artifact from
   entering candidate agreement, Output admission or explicit `FINISH`.
3. `max_length_continuations=1` and
   `length_continuation_max_tokens=512` reuse SkillFlow's bounded same-model
   continuation schedule. The project adaptation continues the exact Agent
   artifact rather than SkillFlow's Supervisor Tool-call request; model,
   contract, task, upstream inbox, Tool configuration and protocol stay fixed.
4. `artifact_assessment_terminal_policy=reject_negative` permits a complete,
   fresh, parseable, unassessed artifact to be consumed while preserving the
   existing blocks for conflict, `insufficient_evidence` and `refuted`.
5. Existing v3 artifact-consumption ordering and termination lookahead plus v4
   provenance-bound assessment remain enabled.

Static validation used the repository's
`config_loader.validate_agent_graph_config` and passed, including explicit
checks for the frozen 30-task count and seed, the five v5 flags above, and
disabled Tool/training/Skill paths. This check started no model service, API
request, rollout, training or evaluator.

The complete official-test evaluation used the same 30 tasks, Direct records,
target-blind extraction, evaluator version, seed and frozen catalog condition:

- Direct: 6/30 = 20.00%.
- AgentGraph v5: 12/30 = 40.00%; 30/30 evaluator-valid trajectories,
  30 explicit FINISH, zero max-rounds, parsing, terminal or operational failure.
- AgentGraph v3: 14/30 = 46.67%; v5 is lower by 2/30, or 6.67 percentage points.
- v3 to v5 paired transfer: 4 correct-to-wrong, 2 wrong-to-correct, 10
  both-correct and 14 both-wrong.

The v3 best-profile therefore remains authoritative. v5 is retained as a
completed, reproducible but unselected architecture candidate. A post-run
target-blind notation fix for public \sqrt2 versus contract sqrt(2) removes
one observed contract-admission false positive; no Accuracy claim is attached
to that unrerun code correction.
GRPO, backward, optimizer updates, LoRA, MACE, Bayesian inference, Skill
retrieval/evolution, retrieval, Web search and answer lookup remain disabled.

## AIME runtime v6-v8: v3 search with v5 reliability boundaries

Status: **three formal same-30 evaluations completed; none selected over v3**.

The v6-v8 passes separated the three observed architecture defects from
mathematical reasoning errors without adding a mathematical workflow prior:

1. `finish_reason=length` now has authoritative precedence over contradictory
   provider metadata. A truncated response remains an incomplete artifact and
   cannot enter candidate agreement, Output admission, or explicit `FINISH`.
2. The inner Agent execution timeout remains 480 seconds while the outer task
   boundary is 900 seconds. This removed the v3 task-28 timeout race: v7 saved
   a formal trajectory and issued explicit `FINISH` for task 28.
3. The AIME contract guard rejects target-blind task-external numeric
   assertions and pre-execution conclusions while leaving Agent count, model,
   contract, relation, Output pointer, and topology to the Director. The five
   historical semantic-drift contracts were absent from the v7 executed
   contracts.
4. A non-format AIME Agent that derives a terminal integer is asked to retain
   its public derivation and append the extractor-supported `Final Answer:
   <integer>` marker. The extractor remains deterministic and target-blind;
   it never solves or repairs a candidate.

Formal strict results under the same official 30-task denominator, frozen
Direct comparator, catalog, and evaluator were:

- Direct: `6/30 = 20.00%`.
- v3 selected best: `14/30 = 46.67%`; 29 evaluator-valid trajectories and one
  operational failure.
- v5 reliability candidate: `12/30 = 40.00%`; 30 evaluator-valid trajectories.
- v6 combined prompt candidate: `8/30 = 26.67%`; 30 evaluator-valid
  trajectories.
- v7 v3-prompt plus v5-reliability candidate: `13/30 = 43.33%`; 29
  evaluator-valid trajectories and one terminal failure.
- v8 contract/output candidate: `10/30 = 33.33%`; 28 evaluator-valid
  trajectories, one operational failure, and one terminal failure.

The target defects were materially isolated in v7: five of five historical
contract-drift cases no longer executed the drifted contract, both historical
length-truncated artifacts were prevented from becoming the terminal
artifact, and task 28 produced a formal trajectory. The eight target tasks
changed from `0/8` correct in v3 to `3/8` correct in v7. The remaining five
target errors were complete mathematical reasoning errors rather than the
original contract, truncation, or timeout failure.

The reliability fixes did not establish a higher-Accuracy policy. v6-v8
collapsed toward single-node graphs, and v8 routed 20 of 30 final nodes to
`gpt-4o-mini`. v7 and v8 used the same seed, prompt, catalog, and evaluator,
but the deployed inference receipt reports nondeterministic generation; their
paired delta is therefore confounded by Director sampling and model-routing
variation. In v8, task 25 terminated through a fresh, parseable artifact with
no resampling but produced the wrong integer 425 instead of 850. Task 03
retained rounds 0-12 in an append-only checkpoint before repeated HTTP 400
responses, and task 15 ended in `canvas_action_domain_exhausted` after
candidate conflict without a legal explicit terminal path.

The evidence-selected v3 best-profile remains authoritative. v6-v8 are kept
as reproducible rejected candidates; no model weights were changed and no
training, optimizer step, LoRA, GRPO, MACE, Bayesian update, Skill, Tool,
retrieval, Web search, or answer lookup ran.

## AIME runtime v11-v13: live `ADD_SUBGRAPH` and native thinking conditions

### v11 role-neutral live parameterization

v11 adds a state-conditioned, role-neutral `ADD_SUBGRAPH` branch to the
existing unified Canvas. The edit may declare one to three Agents with neutral
IDs, catalog-selected models, free-text contracts, registered execution
profiles, legal relations, and an optional Output pointer. The Director still
chooses every Agent, contract, model, relation, topology, Output Agent, and
`FINISH`; no mathematical workflow template or role enum was added.

The implementation retains two upstream boundaries:

1. FlowSteer's progressive Canvas executes an accepted functional edit before
   the next Director observation.
2. SkillFlow's strict structured-action convention binds each generated JSON
   action to the exact live schema and prompt, and its explicit terminal
   convention admits a value only after legal completion.

The local two-phase receipt is a necessary interface adaptation: the first
phase declares the free Agents, and the second phase completes the same
`ADD_SUBGRAPH` action with exact legal relations and Output selection. The
declaration receipt is rooted in the original Canvas prompt while the final
receipt is rooted in the continuation prompt. Validation of that distinction
is restricted to the v3 model-admissible schema so legacy conditions retain
their original receipt contract.

The first isolated v11 canary in
`artifacts/aime2026_runtime_v11_live_subgraph_v3_mask/evaluation` failed before
any AgentGraph trajectory was created: both tasks recorded
`ReceiptValidationError: Director final receipt is bound to a different
prompt`. The failed condition remains evidence and carries no Accuracy claim.
The prompt-root correction was then applied without reusing or relabelling
those failed receipts.

Candidate agreement was also corrected to resolve transitive provenance.
When a downstream artifact repeats a candidate received from an upstream
artifact, both now resolve to the same root lineage instead of being counted
as independent support. Truly independent fan-in roots remain distinct. This
changes only the public lineage projection; it does not select a candidate or
access ground truth.

The AIME target-blind extractor is versioned as
`skillev.integer.target-blind-extraction.v2.2`. It deterministically accepts
the SkillFlow-compatible LaTeX form `\text{Final Answer: } 491` in addition to
the existing bare, boxed, and explicit final-answer forms. Conflicting or
malformed markers still fail closed, and parsing never invokes a model or
uses the evaluator target.

### v12 native Qwen thinking and the 4096-token canary

At the user's explicit request, v12 enables Qwen's native chat-template
`enable_thinking=true` for the local Qwen3.5-9B Director and the local Qwen
Agent. This is an isolated inference condition, not a new prompt instruction
and not a change to SkillFlow's checked executor default, which explicitly
uses `enable_thinking=False`. The Director receipt records the template mode;
the local Agent request passes it through SGLang's `chat_template_kwargs`.
The heterogeneous model catalog remains available, and remote provider models
retain their provider-specific generation behavior.

Because thinking mode changes generation, v12 collects a new Direct baseline
under `target_blind_aime_integer_extraction_v2_qwen35_thinking`; it does not
reuse non-thinking Direct predictions. The two fixed canary tasks completed
the full AgentGraph chain with legal explicit `FINISH` and evaluator-valid
receipts, so the manifest records Stable Zero for the execution pipeline.
This two-task canary is not a formal 30-task Accuracy estimate.

It also exposed a generation-budget failure:

- Direct task 01: `completion_tokens=4096`, `finish_reason=length`, empty
  public output, parser reason `empty_answer`.
- Direct task 02: the same 4096-token length termination and empty-answer
  receipt.
- AgentGraph task 02: an initial local Agent execution also produced an empty
  4096-token length-limited artifact; subsequent Canvas recovery produced a
  complete parseable artifact and legal terminal result.

These are observed receipts, not an inference that the mathematical answer was
wrong. The 4096-token condition can consume its public-answer budget in native
reasoning before emitting answer content, so it is not compared as if it were
a complete Direct prediction.

### v13 independent 16k condition

v13 is a separate condition rather than an overwrite of v12. Its local-Qwen
catalog sets `context_window=32768` and `max_tokens=16384`, retains native
thinking, and leaves remote model budgets unchanged. The Direct protocol is
versioned as `target_blind_aime_integer_extraction_v2_qwen35_thinking_16k`;
all selected-task, Direct, trajectory, failure, paired-result, manifest, and
report paths use the v13 namespace. No `direct_reused_from` field is present,
so a v13 paired result must use newly collected v13 Direct receipts.

The 16k change is only a generation-condition correction for the observed
length/empty-answer failure. It does not change the official 30 tasks,
evaluator, Director action schema, free AgentGraph semantics, maximum Canvas
rounds, model weights, Tools, Skills, or mathematical orchestration prior.
No final v13 Accuracy is recorded here until the independent condition has a
complete, formally evaluated manifest.

No training, backward, optimizer update, LoRA, GRPO, MACE, Bayesian update,
Skill retrieval/evolution, retrieval, Web search, or answer lookup was enabled
for v11, v12, or v13.

## AIME runtime v14: native thinking transport and free-graph correction

v14 is a prepared, unevaluated inference condition. It retains the same
target-blind AIME adapter/evaluator and unified orchestration core, but fixes a
transport gap in the earlier thinking experiments: Qwen's chat-template
`enable_thinking=true` alone does not establish that SGLang defers JSON Schema
grammar until reasoning has ended. The native `/generate` payload now sends
`require_reasoning=true`, while `/server_info` preflight requires the deployed
`qwen3` reasoning parser, XGrammar, strict thinking, and at least the configured
65536-token context. These values, together with the exact generation receipt,
are persisted. The project launch script now uses the same conditions by
default.

The strict hierarchical selector and complete-action decoders accept reasoning
text only before a required `</think>` boundary. They reject a missing boundary,
malformed first object, non-object first value, duplicate key, non-finite value,
or value outside the live Canvas domain. They do not search for a later usable
action and do not repair sampled values. Consumed offsets remain bound to the
exact raw generation receipt, while the Canvas receives only the parsed atomic
action.

The scalar Director prompt is versioned to v8. Relative to v7 it adds one
serialization boundary only: emit exactly the keys in the current action
schema and do not add explanation/reason/rationale fields. It does not add an
Agent role, Agent count, relation, topology, mathematical method, or action
ordering rule. Singleton action schemas also state `type=object` explicitly;
the strict parser contract itself is unchanged.

The local Qwen3.5 Agent/Direct budget is raised to 32768 completion tokens in a
65536-token context. The Director remains separately bounded at 4096 action
tokens, and its request timeout is 300 seconds. This follows SkillFlow's
separation of Supervisor StructuredAction and Executor completion budgets; it
does not change `max_rounds`, introduce a mathematical workflow, or define an
Agent role taxonomy.

The Canvas action domain is also corrected so a first parseable artifact does
not force a shallow single-Agent terminal path while rounds remain. The
Director may still add or modify Agents, set relations, change the Output
pointer, or finish according to the current graph and typed feedback. Only the
formal minimum-action termination boundary narrows the domain. This preserves
the MD free AgentGraph search space rather than rewarding depth or multi-Agent
count directly.

Finally, the existing calculator/Python ReAct path is enabled only as a
bounded computation Tool. Tool stdout remains an Observation and cannot be
promoted to an answer artifact without an explicit `COMPLETE` StructuredAction.
Provider or Tool failures remain typed receipts, and the runtime never switches
models implicitly. No Web Search, retrieval database, AIME solution lookup,
training, GRPO, MACE, Bayesian update, or Skill path is enabled.

The v14 condition is not an Accuracy claim and does not replace the v3
best-profile. A formal score requires a frozen run under the v14 condition;
smoke/canary results establish only runtime compatibility.

A live local control on the already-running GPU0 SGLang 0.5.15 service isolated
the deployment boundary. With `require_reasoning=true`, `reasoning_parser=qwen3`
and XGrammar but `enable_strict_thinking=false`, a singleton `FINISH` schema
returned an extra `reason` field after `</think>`; the same schema in
non-thinking mode returned only `{"action":"finish"}`. The strict parser
correctly rejected the former. That external 32768-context service was not
restarted or modified; v14 therefore remains blocked by runtime preflight on
that process. After applying scalar v8 and the reasoning-aware parser, a second
synthetic request on the same service returned a 169-token thinking trace,
`</think>{"action":"finish"}`, and an exact verified receipt; parsing consumed
the post-boundary `FINISH` action successfully. This establishes the client
transport and serialization correction, but it does not override the formal
64K/strict-thinking preflight and carries no task Accuracy.

## AIME runtime v17: operational failure repair

The interrupted v16 fixed-30 run preserved 30 Direct predictions, eight valid
AgentGraph trajectories, and four collection failures. Three failures reached
the 3600-second task boundary; the affected trajectories repeatedly generated
ReAct requests whose completion consumed the shared 4096-token allowance and
often ended with `finish_reason=length`. The fourth failure occurred before a
Canvas edit because the Director action selector had no required `</think>`
boundary. This is an operational structured-action failure, not an evaluator
or `max_rounds` implementation error.

v17 makes four minimal unified-core corrections:

1. `director.thinking_budget_tokens` is transported through native SGLang
   `sampling_params.custom_params` and recorded in the exact receipt. The
   Director remains local Qwen3.5-9B with thinking enabled.
2. AIME Tool/ReAct uses explicit `max_action_tokens_per_turn` and
   `thinking_budget_tokens_per_turn`; it no longer inherits the Director token
   limit. Every Executor remains in the v16 all-thinking heterogeneous pool.
3. The initial hierarchical action selector reuses the existing one-request
   structured regeneration boundary. It retains both exact receipts, never
   infers an action from reasoning, and performs no Canvas mutation until the
   selector is valid.
4. With a fresh, complete, non-conflicting and terminal-admissible artifact,
   the state-conditioned action domain consumes it through `SET_OUTPUT` then
   explicit `FINISH` before unrelated augmentation. A new edge from a measured
   failed/dirty source into a fresh target is masked and authoritatively
   rejected. Conflict, runtime/artifact failure, and terminal unreachability
   still expose the ordinary recovery/augmentation domain.

The v17 configuration uses a new condition ID, model-catalog version, artifact
directory and report directory. It selects the identical official 30-task test
population and reuses the complete v16 Direct lane only because its task,
model, protocol, contract and seed are unchanged. AgentGraph trajectories are
not resumed from v16. `max_rounds` remains 20 and the 3600-second task timeout
is unchanged. No Skill, training, backward, optimizer update, LoRA, GRPO,
MACE, Bayesian update, fixed mathematical role, fixed Agent count, topology
template, retrieval, Web search, or answer lookup is added.

Prepare-only validation completed for all 30 frozen tasks. Accuracy remains
unreported until the v17 canary and independent AgentGraph rerun complete.

## 2026-08-31: AIME runtime v18 StructuredAction truncation repair

The v17 formal run was interrupted and preserved after eight completed
AgentGraph trajectories. All eight were correct and explicitly finished, but
task 05 exposed a remaining operational regression: six of eight Executor
model calls ended at the 4096-token boundary with
`finish_reason=length` and an unparseable StructuredAction. The ordinary
ReAct parse-error loop eventually recovered, but repeated full-budget
serialization attempts retained the same timeout risk as v16. This partial
prefix is not reported as formal Accuracy.

The implementation now follows SkillFlow
`training/batch_inference.py::supervisor_call`:

1. parse the StructuredAction before testing `finish_reason`;
2. on `length + parse failure`, preserve the exact response and receipt;
3. make one same-model/provider/schema regeneration with at most 512 output
   tokens and thinking disabled;
4. if regeneration is still invalid, fail immediately with
   `structured_action_serialization_failure / output_truncation`.

AgentRuntime now carries the typed category, reason, regeneration count, and
exhaustion flag into the failure record. Canvas classifies this separately
from provider failure, terminal answer parsing, and ReAct turn exhaustion,
while retaining the existing local repair domain. No Agent role, Agent count,
topology, mathematical method, Tool, evaluator rule, or answer lookup was
added.

The targeted regression suite covers successful bounded regeneration,
two-call fail-fast behavior, complete JSON with `finish_reason=length`, and
Canvas failure attribution. Formal score remains pending the v18 targeted
canary and independent same-30 rerun.

## 2026-08-31: AIME runtime v19 multi-Agent search profile

The v18 formal process was stopped on request after 10/30 trajectories. The
preserved prefix contains 10 evaluator-valid explicit `FINISH` trajectories,
6 correct predictions, no collection failure, no `max_rounds` termination,
and one multi-Agent graph. This is an incomplete prefix, not a formal 30-task
Accuracy result.

The single-Agent concentration was traced to the frozen v18 action profile and
artifact ordering rather than a Canvas execution failure: v18 exposed scalar
`ADD_AGENT`, and its first complete parseable artifact was usually still
`unassessed` but nevertheless collapsed the next action domain to
`SET_OUTPUT`. v19 is a new condition and does not resume or relabel v18 data.
It selects the already implemented role-neutral `ADD_SUBGRAPH` profile and
keeps legal graph edits available beside `SET_OUTPUT` until public artifact
assessment is positive. A supported artifact still follows
`SET_OUTPUT -> FINISH`; negative/conflicting/incomplete artifacts and typed
Runtime failures retain local recovery.

No fixed Agent number, role enum, topology, mathematical method, complexity
classifier, structural reward, Skill, or training update was introduced. The
Director must still select one to three free-text Agents, catalog models,
relations, Output, and explicit `FINISH` from the current state-conditioned
domain. Unit tests cover neutral 1--3 Agent admission, unassessed/supported/
refuted candidates, candidate conflict, and Runtime failure before any model
canary is run.

## 2026-08-31: v19 canary stop and v20 current-Output correction

The v19 canary was stopped after its first preserved trajectory. Task 05 was
correct (`65`) and explicitly finished, but its action sequence was a
one-Agent `ADD_SUBGRAPH` with that Agent already selected as Output, followed
by `FINISH`. The artifact remained `unassessed`; nevertheless the live action
domain had removed every graph edit. This was a second action-ordering gap,
not evidence that the Director had freely rejected collaboration.

v20 keeps `FINISH` and ordinary legal graph edits jointly visible for that
exact state. A positively supported current Output still narrows to `FINISH`,
and a supported non-Output artifact narrows to `SET_OUTPUT`. The v19 trajectory
is retained only as a diagnostic and is not resumed or counted in v20. The
new regression test covers the selected-Output case; the AIME AgentGraph suite
passes before the independent v20 canary.

## 2026-08-31: v20 diagnostic and v21 neutral terminal semantics

The v20 canary was stopped after one completed trajectory. Its second Director
turn correctly exposed `add_subgraph`, `modify_agent`, and `finish` for an
`unassessed` current Output, proving the v20 action-domain repair worked. The
Director still selected `FINISH` after reasoning that fresh and complete meant
"everything is in order"; task 09 then failed the official evaluator. The
failure is preserved as a Director state-semantics Wrong Demo, not mixed into
v21.

The v21 prompt adds only definitions already present in Canvas receipts and in
the earlier neutral scalar policy: `finish_admissibility` is not correctness
evidence, and `unassessed / unverified_work_product` has no positive public
assessment. It asks the Director to inspect the public derivation, provenance,
and checks, then either finish sufficient work or address one concrete gap.
It does not request a second Agent, Verifier, chain, parallel branch, reciprocal
edge, mathematical method, or minimum topology.

## 2026-08-31: v21 diagnostic and v22 lookahead projection

The v21 canary was stopped after one completed trajectory. Task 05 was correct
after `ADD_SUBGRAPH -> MODIFY_AGENT -> MODIFY_AGENT -> FINISH`, but remained a
single-Agent graph. Its Director receipt explicitly said that
`minimum_remaining_breakdown.add_agent=0` meant no more Agents could be added,
even though the live action mask still allowed graph growth. This was a
model-visible field-semantics ambiguity, not a Canvas admission constraint.

v22 removes only that per-action breakdown from the Director observation and
states that the total `minimum_remaining_actions` is a lower bound to explicit
termination. The internal lookahead calculation, horizon mask, complete
trajectory receipt, AgentGraph actions, contracts, model catalog, relations,
Output semantics, and evaluator are unchanged. v21 data is retained only as a
diagnostic and is not resumed or counted as v22.

## 2026-08-31: AIME runtime v23 provenance-assessment gate

v22 correctly distinguished action availability from the termination lower
bound, but its `reject_negative` terminal policy still admitted every complete
unassessed artifact. That class included two materially different public
states: a clean dependency-free derivation, and an artifact that either copied
the same candidate from an upstream dependency or reached completion only
after a recorded ReAct recovery event. Treating all three states identically
allowed provenance and execution-recovery gaps to disappear at `SET_OUTPUT` or
`FINISH`.

Source inspection fixed the boundary precisely. FlowSteer's actual upstream
path is `InteractiveWorkflowBuilder.run_loop -> InteractiveWorkflowEnv.step ->
_step_internal`; an accepted executable edit calls `_execute_workflow`, returns
the execution result in feedback, and that feedback is appended before the
next controller turn. Upstream `FINISH` reuses `last_execution_result` when
available under per-step execution, otherwise executes once, then stops. The
project retains that progressive execution and reuse model while keeping its
stricter pointer-only `SET_OUTPUT`, current-revision artifact `FINISH`, and no
max-round answer fallback.

The current SkillFlow sources also establish two narrower precedents than the
earlier shorthand suggested:

1. `training/batch_inference.py::_supervisor_call_unpaused` first accepts any
   parseable native or textual Tool call, even if the provider reports
   `finish_reason=length`. Only an unparseable length-truncated response gets
   one new 512-token same-model Tool-call request with thinking disabled. The
   request includes the truncated assistant prefix plus a corrective user
   instruction; it is not the original payload retried unchanged. If that
   second response is also invalid, upstream returns the original direct
   content with no Tool call rather than raising.
2. `training/environment.py::GenericTaskEnvironment.step` caches a dispatched
   non-mutating Tool observation by exact Tool name and canonical arguments.
   A cached repeat increments its visible repeat count, returns cached or
   summarized evidence without another dispatch, records an ordinary
   Action--Observation turn, and consumes the episode step. Mutating/test
   actions named by the source are excluded from this deduplication path.

v23 therefore makes four thin unified-runtime changes without importing a
second environment:

1. Every completed Agent artifact now carries deduplicated public
   `execution_diagnostic_codes` derived from its ReAct trace. The immutable
   artifact receipt, `UpstreamMessage`, Canvas candidate projection, and
   downstream prompt preserve those codes together with artifact completeness.
   `execution_recovery_observed` is only a convenience boolean over the code
   list; neither field judges mathematical correctness.
2. For an `unassessed` candidate, the terminal gate now checks exact public
   dependency provenance. If the candidate equals one carried by an upstream
   input artifact, `dependency_assessment_required=true` and `SET_OUTPUT` /
   `FINISH` require a supported assessment bound to the exact upstream artifact
   ID and candidate. Merely having an upstream edge does not trigger the gate.
3. An `unassessed` candidate with any recorded execution recovery is likewise
   nonterminal until a provenance-bound supported assessment is present. The
   diagnostic is not treated as refutation; it only prevents a recovered error
   from being silently interpreted as clean evidence. A fresh, complete,
   parseable, non-conflicting, dependency-free artifact with no recovery code
   remains admissible under `reject_negative`.
4. The computation ReAct loop now detects the same executable StructuredAction
   repeated in the unchanged Tool state. It does not redispatch or charge
   another Tool call; it emits `duplicate_tool_request` with consecutive
   `repeat_count` and the prior `cached_observation`, while consuming the
   sampled ReAct turn. A different dispatched Tool action resets this local
   repeat state. This is the project state-safe projection of SkillFlow's
   broader retained-cache behavior, not a claim that all identical calls are
   globally interchangeable.

The existing structured-action truncation repair remains source-aligned but
deliberately fail-closed: parse first; on `length + parse failure`, retain the
exact first receipt and make at most one 512-token same-model/provider/schema
regeneration with thinking disabled; if that is invalid, raise the typed
serialization failure immediately. v23's normal ReAct action and free-text
continuation limits are 8192 tokens; they do not enlarge this bounded retry.

The v23 formal and four-task canary configurations retain the official AIME
population, target-blind evaluator, neutral v13 Director prompt, free-text
Agent contracts, catalog model choice, role-neutral one-to-three-Agent
`ADD_SUBGRAPH` declaration domain, freely chosen relations and Output, and
explicit `FINISH`. They do not require a Verifier, minimum Agent count, chain,
parallel branch, debate, vote, or other topology. Computation Tools remain
bounded; retrieval, Web search, answer lookup, Skills, training, backward,
optimizer updates, LoRA publication, GRPO, MACE, and Bayesian updates remain
disabled. This entry records the prepared inference architecture and tests; it
makes no canary or full-panel Accuracy claim.

## 2026-08-31: AIME runtime v24 compact-history transport repair

The completed v23 canary is retained as an operational-failure diagnosis. It
produced two evaluator-valid explicit-FINISH trajectories: task 09 was correct
and task 11 was an evaluator-valid mathematical reasoning error (`584` versus
`896`). Tasks 10 and 05 produced no formal trajectory because their fifth and
sixth Director requests, respectively, exceeded the local SGLang 32768-token
context and returned HTTP 400. They are not scored as mathematical failures.

The failure was reproduced target-blind from persisted prompts with the same
Qwen3.5 tokenizer. Full historical Canvas replay yielded 32935 total requested
tokens for task 10 and 34563 for task 05. Applying the existing compact Canvas
history policy reduced the same requests to 22672 and 20795 tokens. This keeps
the latest revision-live observation exact and keeps prior sampled Actions plus
typed public feedback/receipts, while removing stale duplicate graph, candidate,
action-domain, and artifact-preview state from earlier observations.

v24 versions that policy as `agentgraph.director.minimal-neutral.v14`; its
system prompt is exactly v13, so it adds no role, topology, mathematical method,
or orchestration prior. A client-side context bound now rejects any future
`prompt_tokens + max_new_tokens > max_context_tokens` before dispatch with a
typed Director error. Unit tests cover the v14 projection, v13 non-mutation,
latest-observation preservation, diagnostic-receipt preservation, and
pre-dispatch context check. v24 canary and formal paths are independent of v23;
no failed checkpoint is relabeled or resumed across conditions. Training,
Skills, GRPO, MACE, Bayesian inference, backward, optimizer update, and LoRA
publication remain disabled.

## 2026-08-31: AIME runtime v25 preserved-truncation recovery condition

The interrupted v24 four-task canary is retained unchanged. Three tasks formed
evaluator-valid explicit-`FINISH` trajectories; task 10 did not form a formal
trajectory. Its public runtime receipts showed the recovery boundary that v25
is intended to isolate: a successful upstream artifact survived the first
downstream provider timeout, a later parameter repair invalidated that
successful Agent, and the replacement local reasoning request reached
`finish_reason=length` without a non-empty final artifact. Repeating another
accepted parameter edit produced the same bounded failure instead of a new
recovery state. These v24 receipts motivate v25; they are not counted or
relabeled as v25 results.

The source boundary is deliberately narrow:

1. **FlowSteer cache/reuse.** The upstream progressive Canvas loop executes an
   accepted edit before returning its feedback, and its per-step `FINISH` path
   reuses the last execution result. The project applies that precedent to an
   input-identity-bound immutable artifact cache so an unrelated downstream
   failure does not erase a still-fresh successful upstream artifact.
2. **SkillFlow bounded retry.** The actual
   `training/batch_inference.py::_supervisor_call_unpaused` implementation
   parses first and gives only an unparseable length-truncated Supervisor Tool
   action one short same-model request with thinking disabled. That exact
   bounded same-model/thinking-disabled rule remains the direct source for
   StructuredAction recovery.
3. **Project finite-context/runtime adaptation.** A free-text Agent artifact is
   not the SkillFlow Supervisor Tool action above. The project-specific path
   therefore preserves the exact partial prefix and segment receipts, admits
   only bounded continuation, marks an unresolved truncation as incomplete,
   and advances repeated repair failure to typed recovery state. It neither
   reconstructs a candidate from a truncated tail nor introduces a fixed
   mathematical role, Agent count, topology, or workflow.

Two new, independent configuration profiles were added:

* `config/evaluation_aime2026_runtime_v25_preserve_truncation_recovery_canary.yaml`
  keeps the frozen task IDs 05, 09, 10, and 11;
* `config/evaluation_aime2026_runtime_v25_preserve_truncation_recovery.yaml`
  keeps the same official sequential 30-task population.

Apart from condition identity and disjoint artifact/report destinations, the
profiles retain v24's prompt v14, evaluator, model catalog, free AgentGraph,
action space, generation condition, and disabled training/Skill/MACE/Bayesian
settings. At the time of this entry, these are prepared conditions only: no
v25 canary, formal evaluation, model/API call, Accuracy result, GRPO/LoRA
update, or Skill evolution is claimed.

## 2026-09-01: AIME runtime v26 lineage-preserving assessment condition

The completed v25 four-task evidence is retained at its original condition and
paths. It exposed two separable orchestration-runtime gaps without changing the
AIME evaluator: (1) a downstream bare integer could copy an exact integer token
from an upstream natural-language artifact that the strict canonical extractor
left unparsed, causing dependency lineage to appear empty; and (2) a fresh
candidate requiring provenance-bound assessment could be invalidated by later
parameter edits before an independent assessment artifact was produced.

The v26 condition is prepared to exercise a project-specific thin adaptation
while keeping source and evaluator boundaries unchanged. Target-blind lexical
lineage supplements provenance metadata, including deterministic multi-hop
dependency closure; it does not canonicalize an upstream artifact, recover a
final answer, consult ground truth, or rank candidates. Assessment-preserving
action ordering keeps a fresh candidate available while admitting an
independent assessment consumer whose contract and relation remain free.
The Director still freely selects contracts, models, relations, Output, Agent
count, and topology. No fixed verifier, chain, parallel branch, mathematical
method, or workflow template is injected.

Two independent profiles were added by copying v25's protocol exactly and
changing only condition identity, disjoint artifact/report paths, adapter-name
prefix, and version comments:

* `config/evaluation_aime2026_runtime_v26_lineage_preserving_assessment_canary.yaml`
  retains frozen task IDs 05, 09, 10, and 11;
* `config/evaluation_aime2026_runtime_v26_lineage_preserving_assessment.yaml`
  retains the official sequential 30-task population.

Both profiles retain prompt v14, model catalog, Direct reuse, evaluator, Tool
protocol, generation parameters, max-round boundary, free AgentGraph, and all
disabled training/Skill/MACE/Bayesian settings. After prepare-only, the v26
canary entered its AgentGraph stage but was safely interrupted when a
pre-canary architecture blocker was identified. At interruption it had
persisted `0/4` formal AgentGraph trajectories and zero collection failures;
there is no v26 Accuracy or other evaluator metric. No v26 formal 30-task run,
training, optimizer update, LoRA publication, Skill, MACE, or Bayesian update
is claimed, and the checkpoint is not resumed under a later architecture.

## 2026-09-01: AIME runtime v27 bounded assessment-repair condition

v27 is a new condition rather than a continuation of the interrupted v26
checkpoint. It is prepared to isolate two target-blind runtime fixes:

1. **Per-occurrence fraction boundary.** Exact integer-token provenance is
   evaluated per occurrence. A candidate occurrence embedded in a fraction is
   excluded, while another standalone occurrence in the same raw artifact can
   still establish lineage. This changes neither AIME extraction nor evaluator
   canonicalization and never reads ground truth.
2. **Bounded malformed assessment-consumer recovery.** When an admitted
   assessment consumer is malformed and has not produced a valid
   provenance-bound assessment artifact, the runtime may expose the bounded
   `MODIFY_AGENT` repair for that existing consumer. If the repair budget is
   exhausted, recovery ends in a typed fail-closed state rather than admitting
   terminal actions, fabricating assessment support, or repeating indefinitely.
3. **Assessment-aware termination lookahead.** The lower bound to explicit
   `FINISH` now follows the live parameter domain: required assessment ingress
   needs `ADD_SUBGRAPH -> SET_OUTPUT -> FINISH`; a repairable malformed
   consumer needs `MODIFY_AGENT` followed by the remaining pointer/terminal
   actions; an exhausted consumer has no reachable terminal path. Near
   `max_rounds`, the Director therefore sees only a parameter-feasible next
   action, or an empty domain when no legal explicit-termination sequence fits.
   This does not add automatic termination or a workflow prior.

The new profiles are:

* `config/evaluation_aime2026_runtime_v27_bounded_assessment_repair_canary.yaml`;
* `config/evaluation_aime2026_runtime_v27_bounded_assessment_repair.yaml`.

They are exact protocol copies of v26 apart from condition identity, disjoint
artifact/report destinations, unused adapter-name prefix, and version comments.
The canary retains task IDs 05, 09, 10, and 11; formal retains the official
sequential 30-task population. Prompt v14, model catalog, Direct reuse,
evaluator, Tool protocol, generation condition, max-round boundary, free
AgentGraph, and disabled training/Skill/MACE/Bayesian settings are unchanged.
Targeted and full offline regression cover these horizon boundaries and pass.
This remains a prepared inference condition only: no v27 model/API call,
canary Accuracy, formal result, training, optimizer, LoRA, Skill, MACE, or
Bayesian claim is made.

## 2026-09-01: AIME runtime v28 detachable dead-branch recovery condition

The v27 canary is preserved in place as incomplete evidence rather than being
resumed or relabeled. Its manifest remains in the AgentGraph stage: Direct has
four reused records, AgentGraph has three admitted trajectory records, and the
collection-failure stream is empty. Task 11 has no terminal trajectory. In its
round-10--15 recovery window, `node_2` remained a failed,
repair-exhausted, artifact-free, terminal-unreachable node while `node_3`
retained one fresh, complete Output artifact. Rounds 11--14 alternated
`node_3 -> node_2` on and off; each inbox change correctly invalidated and
re-executed only `node_2`, reproducing the same bounded StructuredAction
failure. Round 15 attempted the same node parameter repair again. No terminal
evaluator result or v27 four-task Accuracy is inferred from this incomplete
run, and no v27 formal 30-task run is claimed.

The first observable architecture fault is the parameter domain, not the
incremental cache. Existing recovery projections fail closed for particular
semantic role families, but AIME's free-contract `computation_engine` label
fell through to generic relation candidates after strict reachability recovery
returned no candidate. At the same time, preservation policy rejected DELETE
because no same-role replacement had taken over, even though the failed node
owned no artifact, was not Output, and had no downstream responsibility. This
left the Director with edits that could not reduce the terminal diagnosis.

v28 records the following minimal, role-neutral adaptation boundary:

1. Keep FlowSteer's node-cache/input-identity behavior. Relation changes still
   invalidate the changed downstream closure; unrelated fresh artifacts remain
   immutable and reusable.
2. Keep SkillFlow's bounded recovery principle. A no-progress repair-exhausted
   Agent does not receive an unbounded sequence of equivalent requests.
3. Before generic relation fallback, suppress relation candidates whenever a
   failed, repair-exhausted, terminal-unreachable node remains and no already
   defined evidence-ingress, replacement-routing, or strict-reachability edit
   makes progress.
4. Admit explicit `DELETE_AGENT` only for a provably detachable dead branch:
   no current or preserved artifact, not Output, no directed successor, valid
   graph after deletion, and unchanged input identity for every retained fresh
   artifact. The following `FINISH`, if any, remains a separate Director
   action subject to the ordinary terminal gates.

Two independent protocol-equivalent profiles were copied from v27 with only
condition/output/report/unused-adapter namespaces and comments changed:

* `config/evaluation_aime2026_runtime_v28_detachable_dead_branch_recovery_canary.yaml`
  retains task IDs 05, 09, 10, and 11;
* `config/evaluation_aime2026_runtime_v28_detachable_dead_branch_recovery.yaml`
  retains the official sequential 30-task population.

No prompt, model catalog, Tool protocol, evaluator, task split, Direct reuse,
generation condition, max-round boundary, Agent contract, graph constraint,
or terminal semantics changed. No model/API call, evaluation, training,
optimizer update, LoRA publication, Skill, GRPO, MACE, or Bayesian update is
claimed by this configuration/documentation step.

## 2026-09-01: AIME runtime v29 authoritative dead-branch admission condition

The v28 canary was stopped before any formal AgentGraph trajectory was
persisted. Its retained state is intentionally incomplete: Direct has four
reused records, AgentGraph has `0/4` admitted trajectories, collection failures
are zero, and the append-only checkpoint store contains one in-progress
round-0 record. The manifest remains in the AgentGraph stage with no
`completed_at`, Stable Zero receipt, metrics, paired report, or Accuracy.
Because the source admission contract changed after that condition started,
v28 is neither resumed nor used as a comparator for v29.

The pre-run review found that a projected detachable-dead-branch DELETE must
also pass the environment's authoritative preservation admission. Updating
only `model_admissible_action_types` and action targets is insufficient: it can
advertise a legal-looking action that the transactional Canvas later rejects,
or allow another generic edit to bypass the intended recovery boundary. The
minimal source adaptation therefore uses the same state-derived predicate at
all four boundaries:

1. parameter-level action masking exposes strict-progress `SET_RELATION` first;
2. when no such relation exists, target domains expose only the exact
   artifact-free, non-Output, successor-free repair-exhausted leaf;
3. `_preservation_admission_issue` authoritatively accepts only that DELETE and
   rejects unrelated Canvas edits while the dead-branch diagnosis remains;
4. `_delete_admission_issue` accepts the same predicate, after fork-delete
   validation has proved that retained fresh artifact inputs are unchanged.

This remains a task-agnostic Canvas recovery rule. It does not inspect ground
truth, infer answer correctness, prescribe a role or topology, auto-delete an
Agent, or auto-emit `FINISH`. FlowSteer's incremental cache and explicit
edit--execute--feedback sequence remain unchanged. SkillFlow's bounded
Action--Observation recovery remains the source for treating repeated
no-progress StructuredAction failure as finite while retaining Tool receipts
as provenance rather than downstream answer artifacts.

Two new profiles were copied from v28:

* `config/evaluation_aime2026_runtime_v29_authoritative_dead_branch_admission_canary.yaml`
  retains frozen task IDs 05, 09, 10, and 11;
* `config/evaluation_aime2026_runtime_v29_authoritative_dead_branch_admission.yaml`
  retains the official sequential 30-task population.

Only condition/output/storage/report/unused-adapter namespaces and comments
changed. The subsequently completed fixed-task canary persisted four formal
trajectory records and zero collection failures. Only task 05 reached legal
explicit `FINISH`, entered the formal evaluator, and was correct: evaluator
admission was `1/4`, and strict fixed-task correct/resolved rate was `1/4 =
25.00%`. Tasks 09, 10, and 11 ended in
`canvas_action_domain_exhausted` without formal scoring. Task 10 nevertheless
retained a fresh Output artifact with the correct canonical candidate `156`;
under the MD terminal contract it correctly received no reward because no
explicit `FINISH` followed.

## 2026-09-01: AIME runtime v30 failure-specific terminal recovery condition

The v29 receipts isolate terminal reachability rather than AIME evaluator
failure. v30 keeps FlowSteer's progressive Canvas and immutable incremental
artifacts, SkillFlow's bounded StructuredAction recovery, and the existing
SkillFlow/SkillEval-aligned AIME integer evaluator. It adds only three
project-specific necessary adaptations:

1. During an identified provenance-bound assessment protocol failure, both
   parameter-level action masking and authoritative Canvas admission permit
   only `model_id` or an answer-free `contract` repair. `allowed_tools`,
   `execution_mode`, `artifact_type`, and `completion_condition` are not valid
   substitutes for repairing the malformed assessment artifact.
2. If a supported fresh sink already exists, expose its pointer-only
   `SET_OUTPUT` before unrelated recovery or augmentation. This consumes the
   existing artifact; it does not infer correctness, resample an Agent, or
   combine `SET_OUTPUT` with termination.
3. If an exhausted protocol-failure leaf owns no candidate artifact, expose
   explicit `DELETE_AGENT` only when the existing preservation predicate proves
   that it is non-Output, successor-free, and removable without changing any
   retained fresh artifact's effective input identity.

The explicit terminal rule remains intentionally stricter than upstream
workflow-return conventions: only a later legal Director `FINISH` enters the
formal AIME evaluator. Correct artifacts are never salvaged from history or
from max-round termination. This is the necessary incompatibility adaptation
for the project's MD semantics, not a workflow prior.

The independent profiles are:

* `config/evaluation_aime2026_runtime_v30_failure_specific_terminal_recovery_canary.yaml`,
  retaining task IDs 05, 09, 10, and 11;
* `config/evaluation_aime2026_runtime_v30_failure_specific_terminal_recovery.yaml`,
  retaining the official sequential 30-task population.

They are field-for-field copies of v29 except for comments and disjoint
condition/output/storage/report/unused-adapter namespaces. Prompt v14, model
catalog, Tools, evaluator, task split, Direct reuse, generation settings,
`max_rounds`, free AgentGraph, and terminal semantics are unchanged. No
training, optimizer update, LoRA publication, Skill, GRPO, MACE, Bayesian
update, model/API call, or v30 Accuracy is claimed by this preparation step.

### v30 fixed-task result and v31 candidate-artifact consumption

The completed v30 canary persisted four AgentGraph trajectories, 34
append-only round checkpoints, and zero collection failures. Tasks 05 and 09
reached legal explicit `FINISH`, entered the shared target-blind AIME evaluator,
and were correct (`65` and `29`). Tasks 10 and 11 ended with
`canvas_action_domain_exhausted`; their formal evaluator was not called and
their Accuracy is therefore `N/A`, not zero. The strict fixed-task
correct/resolved rate is `2/4 = 50.00%`; the evaluator-valid subset is `2/2`.
No v30 official 30-task Accuracy exists.

Both terminal failures exposed the same runtime boundary: a fresh, complete,
candidate-bearing consumer omitted the required provenance-bound assessment.
v30 repaired that consumer by `MODIFY_AGENT`, which resampled and replaced the
live candidate artifact. After the bounded repair reproduced a bare candidate,
the candidate-bearing leaf could neither be deleted safely nor reach
`SET_OUTPUT -> FINISH`.

v31 is a disjoint condition. It keeps the task IDs/official population, seed,
prompt v14, model catalog, Tool protocol, evaluator, `max_rounds: 20`, free
Agent contracts, and explicit terminal semantics unchanged. The necessary
runtime adaptation is state-conditioned and bounded:

1. a candidate-bearing assessment-protocol failure never modifies/resamples
   its owner;
2. one `ADD_SUBGRAPH` transaction adds one free-contract consumer and routes
   every exact required source artifact into it as a directed fan-in;
3. source artifact IDs are persisted in the completed-turn checkpoint, so the
   same recovery cannot recursively grow another consumer after deletion or
   restore;
4. a successor resolves the old protocol failure only after it consumes the
   failed artifact plus every missing candidate provenance artifact and emits
   exact provenance-bound assessments for all of them;
5. non-candidate protocol failure retains the v30 bounded
   `MODIFY_AGENT -> DELETE_AGENT` path, and full capacity uses an explicit
   preservation-safe `DELETE_AGENT` before the fan-in ADD.

This is one local recovery subgraph after a measured failure, not a fixed
initial/global topology, role, model, mathematical method, or workflow. It
does not inspect ground truth or select a candidate. The prepared profiles are
`config/evaluation_aime2026_runtime_v31_candidate_artifact_consumption_canary.yaml`
and
`config/evaluation_aime2026_runtime_v31_candidate_artifact_consumption.yaml`.
No v31 Accuracy is claimed until its fixed-task canary and formal run complete.

The final no-model regression additionally covers mixed candidate/non-candidate
protocol failures. Its measured path is `ADD_SUBGRAPH -> MODIFY_AGENT ->
SET_OUTPUT -> SET_RELATION -> FINISH`: the pointer-only Output edit precedes
the reachability relation, and the live relation domain prefers the
provenance-compatible strict-progress edit with the smallest FlowSteer dirty
closure. Mixed pre-fan-in horizons whose exact lower bound depends on a later
protocol repair are reported as `minimum_remaining_actions=null` with
`assessment_minimum_deferred=true`, not as a falsely small integer. After the
repair, the lower bound is recomputed from the live graph. The final related
regression set is 354/354 passing; this does not itself constitute a model
evaluation or an Accuracy result.

### v31 canary freeze and v32 receipt-domain alignment

The v31 canary is frozen as an incomplete failed condition. Tasks 09 and 11
formed evaluator-valid explicit-`FINISH` trajectories and were correct (`29`
and `896`). Task 05 formed no trajectory: after round 8, its live action domain
correctly required one new free-contract consumer with exact two-source fan-in,
but the collector's legacy v3 receipt check rejected every free-text
`ADD_SUBGRAPH` containing more than one relation. Task 10 remained in progress
at round 3 when the runner exited with signal status 143. Consequently v31 has
no four-task Stable Zero result and no official Accuracy; the two valid rows
must not be reported as a four-task denominator.

v32 changes only the receipt validator and experiment identity. Ordinary
free-text and verified-QA `ADD_SUBGRAPH` receipts retain the one-relation
boundary. A multi-relation receipt is admitted only when the frozen live domain
is non-verified-QA free text and explicitly declares
`require_all_existing_ingress=true` with `required_relation_count>1`; validation
then requires the exact cardinality, one new consumer, every required source,
one-way source-to-consumer direction, live endpoints, allowed relation values,
and no repeated unordered pair. Canvas admission, Director schema, prompt,
model catalog, Tools, evaluator, terminal semantics, and all disabled training
features remain unchanged. The disjoint profiles are
`config/evaluation_aime2026_runtime_v32_receipt_domain_aligned_fanin_canary.yaml`
and `config/evaluation_aime2026_runtime_v32_receipt_domain_aligned_fanin.yaml`.

### v32 operational freeze and v33 explicit terminal/coding condition

The v32 fixed-task process is frozen as an operationally non-comparable partial
run. Its long-lived tmux server did not inherit the host HTTP(S) proxy, so all
three selected remote model families produced bounded `URLError` receipts. The
retained evidence contains 18 completed-turn checkpoints, one formal task 09
trajectory (`FINISH`, canonical prediction `29`, correct), and zero collection
failures. It is not resumed with a different provider route and does not define
a four-task Accuracy or Stable Zero result.

A subsequent read-only terminal audit found two independent historical-answer
promotion paths: `AgentGraphOrchestrator.run` and the formal
`AgentGraphRolloutCollector` could each promote
`last_valid_evidence_lineage` after max-round or Canvas-domain termination.
v33 removes both producer paths. Without legal explicit `FINISH`, the current
terminal graph and all immutable turn/artifact receipts remain diagnostic
evidence, while `final_answer` and `final_runtime` are null and the existing
AIME no-submission gate records `formal_evaluator_called=false`. This is the
MD-required incompatibility adaptation; upstream FlowSteer may force a final
workflow result, while SkillFlow represents horizon exhaustion as a distinct
no-terminal-submission state.

The user also allowed code-assisted Agent collaboration. The existing generic
`ToolReactExecutionAdapter` already supports both `react` and `coding`, and the
AIME registry already supplies bounded calculator/Python computation. v33 adds
an optional, versioned `aime_tool_runtime.execution_modes` field (legacy
default `[react]`) and explicitly selects `[react, coding]`. Both modes share
the same task-scoped computation registry and separate adapter instances. The
SWE-bench repository edit/test/diff completion adapter is not reused for AIME.
No Coding Agent type, fixed role, fixed topology, task-type router, or prompt
template is introduced; the Director sees only runtime-backed correlated
mode/Tool profiles and chooses through the existing free AgentGraph action.

The prepared disjoint profiles are
`config/evaluation_aime2026_runtime_v33_explicit_finish_coding_canary.yaml` and
`config/evaluation_aime2026_runtime_v33_explicit_finish_coding.yaml`. The
canary keeps task IDs 05, 09, 10, and 11; the formal profile keeps the official
sequential 30-task population. No v33 model/API result or Accuracy is claimed
until the proxy-consistent fixed-task canary completes.

### 2026-09-01: v34 runtime/Canvas execution-profile alignment

This is a source-classified architecture entry, not an experiment result.
No v34 model/API canary or official 30-task evaluation is claimed here.

Direct upstream reuse remains unchanged:

1. **SkillFlow reused:** Supervisor reasoning and visible action generation
   retain independent budgets. The strict `StructuredAction` boundary, one
   Action followed by its public Observation, bounded serialization recovery,
   and lossless failure receipts remain the execution protocol.
2. **FlowSteer reused:** progressive Canvas keeps one atomic edit -> execution
   -> feedback transaction per turn. The graph revision, executed Agent
   artifacts, communication provenance, execution result, Canvas feedback,
   and terminal state remain in the same trajectory boundary.

The following items are **project-specific necessary adaptations** for the
heterogeneous free AgentGraph and do not replace either upstream runtime:

1. Canonical execution-profile continuation binds a live Agent and any bounded
   continuation/repair to its exact `(model_id, execution_mode,
   allowed_tools)` profile. Receipts and capability checks use the same tuple;
   Runtime cannot silently switch a model, mode, or Tool set.
2. ADD action targets publish the exact flat
   `registered_model_execution_profiles` union. MODIFY `model_id` candidates
   are filtered by compatibility with the current execution mode and Tool set;
   MODIFY mode/Tool candidates are derived only from the current model's
   registered profiles. Existing aggregate target fields remain for backward
   serialization compatibility, but do not widen the authoritative joint
   domain.
3. Provenance-aware assessment ignores only the explicitly defined null,
   noncandidate diagnostic artifact. Candidate-bearing, conflicting,
   incomplete, or failed artifacts remain visible with exact source and
   artifact identity; evaluator ground truth is never consulted.
4. When a fresh candidate artifact needs provenance-bound assessment and
   capacity exists, assessment ingress `ADD_SUBGRAPH` precedes unrelated
   terminal-reachability relation narrowing. Recovery state and terminal
   progress expose the same next action without prescribing the new Agent's
   contract, model, relation, Output, or topology.
5. Fail-fast Runtime/Tool/serialization/assessment handling preserves all
   successful upstream artifacts and writes the typed failed state. A nested
   Tool result with `ok=false` is not counted as a successful Tool receipt;
   diagnostics cannot become candidate artifacts.

`coding` remains an optional Runtime `execution_mode` on
`agent_id + model_id + free-text contract`. It is available only through a
registered model/profile tuple. v34 introduces no predefined Coding Agent,
role enum, forced coding node, fixed Agent count, fixed relation, or fixed
chain/parallel/debate topology. Skill, GRPO, MACE, Bayesian inference,
backward, optimizer update, and LoRA publication remain outside this change.

### 2026-09-01: v35 integration-contract supersession of prepare-only v34

v34 did not enter model/API evaluation. It is retained as **prepare-only**
source/configuration state with no canary, Accuracy, official 30-task result,
training run, or weight update. An integration review found four concrete
interface gaps, so v35 supersedes v34 rather than resuming or relabeling it:

1. continuation cache identity did not yet require the complete declaration
   identity `model_id + contract + execution profile`;
2. routed `peer_draft` artifacts did not yet have one standard typed envelope
   on which exact assessment coverage could be based;
3. nested computation provenance did not yet require transitive binding to
   the source Agent and source artifact;
4. scalar MODIFY parameter decoding could not express a correlated
   `execution_mode + allowed_tools` change as one atomic profile edit.

The v35 source classification is:

- **FlowSteer reused:** bounded bidirectional Agent communication, progressive
  Canvas execute-on-edit, transactional graph revision, typed artifact
  lineage, incremental invalidation, Output selection, and explicit terminal
  action remain unchanged.
- **SkillFlow reused:** public `StructuredAction -> Observation` continuation,
  bounded continuation, and lossless action/observation/Tool receipts remain
  the model-visible execution protocol.
- **Project-specific necessary adaptation:** continuation reuse now requires
  exact `model_id + contract + (execution_mode, allowed_tools)` declaration
  identity. Any declaration/profile change invalidates the old continuation.
- **Project-specific necessary adaptation:** each `peer_draft` is routed in a
  standard typed envelope with source Agent, artifact ID, declaration/profile
  identity, completeness/status, raw public payload, and provenance. Exact
  assessment coverage binds to these artifacts and never to anonymous merged
  text or evaluator state.
- **Project-specific necessary adaptation:** nested computation receipts are
  propagated transitively only while bound to their source Agent and source
  artifact. Downstream use cannot detach a candidate or diagnostic from its
  producing lineage.
- **Project-specific necessary adaptation:** `MODIFY_AGENT` gains one
  `execution_profile` parameter branch whose value is an admitted correlated
  pair of `execution_mode` and `allowed_tools`. This performs one atomic
  profile edit from the selected model's registered union; it does not expose
  an invalid Cartesian product or consume two Canvas turns.

Coding remains an optional `execution_mode` of a free-text Agent declaration,
not a Coding Agent type. v35 keeps the free AgentGraph and does not prescribe a
role, Agent count, relation, topology, chain, parallel branch, verification
workflow, or mathematical method. Training, Skill, Skill evolution/retrieval,
GRPO, MACE, Bayesian inference, backward, optimizer update, and LoRA
publication remain disabled.

The later v35 fixed-four execution is retained as a **partial, invalid
four-task condition**, not as an Accuracy or Stable Zero result. Task 05 alone
committed a formal trajectory with legal explicit `FINISH`, canonical answer
`65`, and a correct evaluator receipt. On task 09,
`qwen3.5-flash + react + calculator` exhausted its bounded 480-second provider
request and produced the typed provider-failure receipt. The recovery domain
then removed the unavailable model, found no repair model compatible with the
unchanged profile, exposed an empty `modify_agent.mutable_fields` domain, and
failed collection. Tasks 10 and 11 were not completed. Therefore no
`correct / 4`, four-task Accuracy, Stable Zero status, or official 30-task
metric may be reported from v35; its task-05 trajectory and task-09 failure are
diagnostic evidence only.

### 2026-09-01: v36 provider/profile bridge after the invalid v35 condition

v36 is a minimal recovery-domain compatibility adaptation. It keeps
FlowSteer's existing one-atomic-`MODIFY_AGENT` Canvas transaction and does not
introduce an alternate action, workflow engine, role, or topology. For a
provider-failed Agent, recovery proceeds as follows:

1. If a legal repair model directly admits the current execution profile,
   `model_id` remains the one exposed repair field.
2. Otherwise, the live domain computes the exact intersection between the
   failed Agent model's registered execution profiles and the registered
   profiles of the admitted provider-repair models. A single
   `execution_profile` edit may select only one tuple in this intersection.
3. The bridge edit changes only `execution_mode + allowed_tools` atomically,
   preserves model/contract/relations, and does not execute the already
   unavailable model. On the next Canvas turn, the compatible repair
   `model_id` is exposed for an explicit takeover.
4. If neither a direct compatible model nor a common bridge profile exists,
   `MODIFY_AGENT` fails closed instead of publishing an empty mutable-field
   domain. Runtime never switches model, provider, mode, or Tool set silently.

This two-edit sequence is parameter recovery inside the existing free
AgentGraph. It does not prescribe the Agent's mathematical contract or create
a fixed Solver, Coding, or Verifier role. `coding` remains an optional
registered `execution_mode`; the Director may also select reasoning or ReAct
when admitted by the exact model/profile domain. Skill, Skill retrieval or
evolution, GRPO, MACE, Bayesian inference, backward, optimizer updates, LoRA
publication, and all other training paths remain disabled.

The disjoint profiles are
`config/evaluation_aime2026_runtime_v36_provider_profile_bridge_canary.yaml`
and
`config/evaluation_aime2026_runtime_v36_provider_profile_bridge.yaml`.

The subsequent fixed-four v36 canary stopped as a **partial condition** after
two collected trajectories. Task 05 consumed nine Director turns and then
terminated with `canvas_action_domain_exhausted`; it had no legal explicit
`FINISH`, no evaluator receipt, and still reported 11 remaining rounds. Task
09 demonstrated the intended explicit bridge end to end: after the original
provider failure, one atomic `execution_profile` edit selected the common
`reasoning + []` profile without calling the unavailable model, the following
turn exposed and accepted the compatible local `model_id`, and the resulting
fresh artifact was consumed by an explicit `FINISH`. That terminal result was
evaluator-valid but mathematically wrong. Task 10 and task 11 were not run.
Therefore v36 has no complete `correct / 4` denominator, no Stable Zero result,
no official 30-task run, and no formal Accuracy claim.

### 2026-09-01: v37 artifact-consumption consistency after partial v36

The partial v36 task-05 trajectory separated a Canvas/runtime failure from a
mathematical failure. Three fresh, complete artifacts already exposed the same
parseable public candidate, including an independent artifact outside the
protocol-failed branch. The live action mask nevertheless admitted only a new
assessment consumer. Rejected contracts and semantically duplicated fan-in
relations then consumed turns; the accepted consumer returned a bare scalar
without provenance-bound assessment, after which no legal action remained.
The trajectory ended with `canvas_action_domain_exhausted`, not `max_rounds`.

v37 therefore makes the following source-classified adaptations:

1. **FlowSteer reused — progressive consumption and explicit termination.**
   Every accepted Canvas edit still executes before the next Director
   observation. Artifacts remain immutable and revision-bound. `SET_OUTPUT`
   changes only the Output pointer, and `FINISH` consumes the current fresh
   Output artifact without resampling it. A formal evaluator call still
   requires a separate legal explicit `FINISH`.
2. **Project-specific necessary adaptation — existing artifact before
   augmentation.** A fresh, complete, parseable artifact can enter the
   `SET_OUTPUT` target domain when its entire dependency closure is free of
   current execution and provenance-protocol failures. Failures wholly outside
   that closure may be deferred only for pointer selection. Subsequent
   observations expose one exact missing directed relation into that Output
   consumer per atomic `SET_RELATION` edit. Complete-graph reachability and all
   terminal guards are still required before `FINISH`; no candidate is ranked
   against evaluator ground truth and the runtime does not auto-finish.
3. **Project-specific necessary adaptation — exact fan-in parameter masking.**
   When the current live `ADD_SUBGRAPH` domain requires all existing ingress,
   the relation array is serialized as positional `prefixItems` with constant
   source, target, and direction fields, exact length, and no additional
   items. This makes duplicate-source or missing-source relation samples
   unrepresentable at the Director parameter layer while keeping ordinary
   relation search and Canvas validation unchanged.
4. **SkillFlow reused — five-field StructuredAction.** ReAct and coding
   execution keep SkillFlow's public `arguments / kind / name / resource_id /
   skill_id` action, bounded Action--Observation continuation, and Tool
   receipts. The project does not introduce a second action format.
5. **Project-specific necessary adaptation — complete provenance-bound
   artifact admission.** When routed upstream state contains candidate-bearing
   artifacts under the assessment protocol, a COMPLETE action must preserve a
   public derivation and an assessment for each exact `artifact_id`, with each
   public candidate binding copied unchanged. A bare scalar, incomplete
   assessment set, or mismatched provenance/candidate binding is returned as
   typed execution feedback rather than accepted as a complete downstream
   artifact. Requests without those routed candidate bindings keep the
   existing completion rule. This check is target-blind and cannot determine
   which candidate is correct.
6. **Project-specific necessary adaptation — bounded consumer repair.** The
   runtime binds the accepted recovery-consumer Agent identity to the exact
   source artifact IDs it consumed and persists that state in checkpoints. If
   that exact consumer returns a candidate-bearing protocol failure, one
   `MODIFY_AGENT` repair is admitted: the answer-free contract is the first
   mutable field, with only a catalog-compatible model alternative when one
   exists. A second failed execution is recorded as typed repair exhaustion;
   the same source set cannot recursively create another equivalent consumer
   chain.

`coding` in every item above is an optional catalog-registered
`execution_mode`, not a Coding Agent, role enum, fixed subgraph, required
branch, or mathematical workflow. The Director remains free to choose Agent
count, model, free-text contract, relations, Output, and explicit termination
within the current live domain.

The prepared canary and formal configuration identities are respectively
`aime2026_runtime_v37_artifact_consumption_consistency_canary` and
`aime2026_runtime_v37_artifact_consumption_consistency`. Skill, Skill
retrieval/evolution, GRPO, MACE, Bayesian inference, backward, optimizer
updates, LoRA publication, and training remain disabled.

The subsequent fixed-four v37 run completed collection but failed Stable Zero.
Task 05 and task 10 ended in `canvas_action_domain_exhausted` without legal
explicit `FINISH`. Task 09 explicitly finished with canonical prediction `7`
against target `29`; task 11 explicitly finished with the correct canonical
prediction `896`. The run therefore contains two evaluator-valid trajectories,
one correct and one wrong, but it does not supply a valid `correct / 4`
Accuracy denominator. No official 30-task v37 evaluation was run.

### 2026-09-01: v38 Output-closure consumer recovery after v37 fixed-four

v38 is a minimal runtime/artifact correction for the two v37
`canvas_action_domain_exhausted` trajectories. It preserves v37's fixed-four
and official-30 populations, v18 heterogeneous thinking catalog,
minimal-neutral v14 Director prompt, seed, direct reuse, free-text contracts,
free AgentGraph action/relation search space, GPU0 rollout placement and
explicit `FINISH`. It does not add a fixed Agent count, role enum, coding
branch, relation motif, topology, solution template or automatic terminal
action.

The source-classified boundaries are:

1. **FlowSteer reused — progressive-cache Output dependency scope.** Runtime
   readiness is determined from the proposed Output artifact's immutable
   dependency closure, using the same incremental cache/invalidation boundary
   as progressive Canvas execution. Unrelated failed branches do not become
   false failures of that Output artifact; pointer-only `SET_OUTPUT` and
   explicit `FINISH` still do not resample a fresh Agent.
2. **SkillFlow reused / project thin adaptation — bounded exact-consumer
   repair.** SkillFlow's one bounded StructuredAction serialization-repair
   precedent is applied only to the exact provenance consumer associated with
   the exact upstream artifact set. Existing successful upstream artifacts and
   receipts are preserved. Repair exhaustion remains typed feedback; runtime
   does not silently substitute a model/provider or recursively manufacture
   equivalent consumers.
3. **Project-specific necessary conformance test — mixed reasoning/coding
   provenance.** Registered reasoning and optional coding executions are
   checked against the same `source_agent + artifact_id + raw_output +
   freshness + assessment` boundary. This validates collaboration through
   coding without turning coding into a predefined Agent type or preferred
   workflow, and it never consults ground truth or selects a candidate.

The independent conditions are
`aime2026_runtime_v38_output_closure_consumer_recovery_canary` and
`aime2026_runtime_v38_output_closure_consumer_recovery`. The configuration
regression verifies versioned paths/condition IDs, fixed task05/09/10/11,
official sequential 30, direct reuse, v18 catalog, prompt v14, seed, explicit
terminal action, GPU0, and disabled Skill/GRPO/MACE/Bayesian/training paths;
it passes `8` tests. No v38 canary or formal run has been started, so this entry
makes no v38 Stable Zero, `correct / total`, or Accuracy claim.

### 2026-09-01: v38 stopped evidence and v39 AIME role-neutral schema

The v38 fixed-four runner was safely stopped after task 10 demonstrated an
AIME action-schema leak.  Completed evaluator-valid trajectories were task 05
(`65 / 65`), task 09 (`29 / 29`), and task 11 (`896 / 896`), all with legal
explicit `FINISH`.  Task 10 had five partial turns, no Output artifact, no
`FINISH`, and no evaluator call.  Its generic `ADD_SUBGRAPH` carried
`role_family=output_agent` and `role_family=calculator_agent`; these labels did
not directly control AIME execution, but they were persisted and returned to
the Director as policy-active state.  Therefore v38 coverage is `3 / 4`, not a
formal Accuracy denominator, and the condition is not Stable Zero.

v39 makes only the following compatibility correction:

1. The AIME free-contract ADD domains set `role_family_admitted=false`.
2. The role-neutral constrained schema omits `role_family`, and the exact
   declaration parser rejects payloads that add it.
3. Canvas rejects raw AIME role-bearing ADD/MODIFY actions before mutation or
   execution, returning typed `role_family_forbidden` feedback.
4. HotpotQA/TriviaQA semantic role schemas, legacy `require_format_agent=true`,
   and non-AIME generic recovery retain their previous role behavior.

No role, Agent count, relation, topology, chain, parallel branch, verifier,
coding branch, or mathematical method is prescribed.  Coding remains an
optional registered `execution_mode`.  No training, Skill retrieval/evolution,
GRPO, MACE, Bayesian update, backward, optimizer step, or LoRA publication was
performed.  Regression evidence is: Director `56 passed`; AgentGraph/Canvas
`246 passed, 72 subtests passed`; the combined AIME role-neutral, v39-config,
coding-collaboration, and v38-config selection has `21 passed, 5 subtests
passed`.  Prepare-only receipts contain 4 fixed-canary tasks and 30 formal
tasks.  v39 runtime results must be recorded separately after the live canary
finishes.

### 2026-09-02: v43--v45 incomplete-artifact and provider/protocol admission

The v43--v45 work was restricted to runtime, artifact, feedback, recovery and
termination boundaries. It retained the v18 heterogeneous model catalog,
role-neutral free-text Agent contracts, the existing AIME evaluator, the
fixed same-30 task set and explicit `FINISH`. No fixed mathematical workflow,
training, GRPO, MACE, Bayesian posterior, Skill retrieval/evolution, backward,
optimizer update, or LoRA publication was introduced.

1. **Incomplete artifact repair.** A fresh artifact with incomplete-generation
   diagnostics cannot become terminal evidence. Only declaration changes that
   correspond to a registered executable model/profile are admitted for the
   failed Agent; successful upstream artifacts remain immutable and reusable.
2. **Exact fan-in recovery.** A failed provenance consumer remains bound to
   the exact source artifact IDs it consumed. The recovery domain does not
   create an unbounded chain of equivalent consumers and does not copy an
   unverified answer into a pre-execution contract.
3. **Provider repair precedence.** The task10 v44 canary showed that a valid
   explicit model repair was present in the constrained action domain but was
   rejected by a lower raw-Canvas provenance guard. v45 aligns these two
   admission paths: a parameter-feasible provider repair of the same failed
   Agent is exposed and accepted before artifact-protocol repair. Model changes
   stay explicit and receipt-bearing; no runtime fallback is hidden.
4. **Targeted verification.** The focused AgentGraph regression covering
   exact-fan-in provider repair passed, and the independent v44/v45 config
   regression passed. The v45 task10 canary produced `156`, explicit `FINISH`,
   and a valid exact-match evaluator receipt, establishing Stable Zero for the
   blocker before the formal run.
5. **Formal same-30 result.** AgentGraph completed `30 / 30` collection records,
   with `24 / 30 = 80.00%` strict Accuracy, `24 / 27 = 88.89%` on evaluator-valid
   trajectories, `27 / 30` explicit finishes, 3 terminal failures, 0 parsing
   failures and 0 collection failures. Direct produced `19 / 30 = 63.33%` with
   11 terminal-output parsing failures caused by `finish_reason=length`.
   `protocol_equivalent_to_direct=false`, so the `+16.67` percentage-point
   paired difference is reported as descriptive, not causal.
6. **Remaining causal failures.** Tasks 09, 15 and 29 are mathematical
   reasoning errors after valid terminal execution. Task 17 had the correct
   candidate `243`, but an answer-bearing assessment contract was rejected at
   the horizon. Task 22 had four agreeing `754` artifacts and complete fan-in
   assessment, but stale recovery state plus an Output pointer that was no
   longer the graph sink exhausted the action domain. Task 28 never produced a
   legal AIME candidate after structured-action and incomplete-artifact
   failures, and premature graph augmentation amplified blocked dependencies.

The reproducible profile is
`config/evaluation_aime2026_runtime_v45_provider_protocol_admission.yaml`.
Machine-readable evidence and the full trajectories are in
`artifacts/aime2026_runtime_v45_provider_protocol_admission/evaluation/`.
The formal reports are in
`reports/aime2026_runtime_v45_provider_protocol_admission/`.
