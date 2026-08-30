# MBPP+ v0.2.0 initial adaptation source map

This adaptation replaces the incomplete SWE-bench repository path with the
MBPP+ function-generation protocol.  It does not change the unified
orchestration core and does not enable training.

## Upstream reuse

| Boundary | Upstream implementation | Local use |
|---|---|---|
| Public/private task materialization | SkillFlow `packages/private-evaluation/src/skillev_private/benchmarks/external_materialization.py::materialize_evalplus` | Public records contain only the model-visible prompt and task identity; evaluator inputs and reference code remain evaluator-only. |
| Evaluation population | SkillFlow `configs/evaluation/protocol_v10.yaml`, `mbpp-plus-fixed-100` | First 100 tasks in ascending canonical EvalPlus task-ID order; the result is named `mbpp-plus-fixed-100@1`, not `hard-100`. |
| Dataset loading | EvalPlus 0.3.1 `evalplus/data/mbpp.py::get_mbpp_plus` | Load MBPP+ v0.2.0 and apply the official input deserialization. |
| Complete-source submission | EvalPlus 0.3.1 `evalplus/evaluate.py::evaluate` and SkillFlow `skillev_private/benchmarks/evalplus_official.py::run_mbpp` | Pass the complete Python submission unchanged to the evaluator. EvalPlus sanitization is an optional code-generation utility and is not part of either formal evaluation path. |
| Ground-truth execution | EvalPlus 0.3.1 `evalplus/evaluate.py::get_groundtruth` | Compute expected outputs only inside the evaluator boundary. |
| Candidate evaluation | EvalPlus 0.3.1 `evalplus/evaluate.py::check_correctness` and `evalplus/eval/__init__.py::untrusted_check` | Execute base and plus tests with the official status and timeout semantics. |
| AgentGraph and relation semantics | FlowSteer `src/interactive/agent_graph.py` | Preserve `agent_id + model_id + free-text contract` and the existing relation search space. |
| Progressive Canvas editing and execution feedback | FlowSteer `src/interactive/agent_workflow_env.py` | Execute the current graph after each accepted Canvas edit and return execution feedback to the Director. |
| Agent execution and communication | FlowSteer `src/interactive/agent_runtime.py` | Preserve independent, directed, and bounded bidirectional execution blocks and typed Agent communication. |
| Trajectory receipts | FlowSteer `src/interactive/workflow_builder.py`, `src/interactive/records.py` | Persist Director actions, Canvas revisions, Agent calls, communication, terminal output, and evaluator receipt. |

## Necessary compatibility adaptation

MBPP+ has no repository snapshot, worktree, file-edit action, patch artifact, or
SWE-bench `Resolved` evaluator.  Its terminal artifact is a complete Python
source candidate. EvalPlus reports base pass@1 and MBPP+ pass@1 separately;
MBPP+ pass@1 is determined by the Plus-test status. The local adapter therefore adds only:

1. an EvalPlus-to-`TaskRecord` materializer;
2. an EvalPlus 0.3.1 evaluator callback;
3. MBPP+ registration in the existing completion benchmark runner; and
4. a versioned evaluation-only configuration.

The SkillFlow public MBPP+ tasks expose no repository Tool, so both Direct and
AgentGraph use the same empty Tool surface in this initial condition.  ReAct
remains an optional Agent `execution_mode`; it is not an Agent role.

## Model-visible and evaluator-only fields

Model-visible fields are limited to the official prompt, official task ID, and
public entry point.  The following fields never enter the Director prompt,
Agent input, Agent communication, Tool observation, or public trajectory:

- `canonical_solution`;
- `base_input`;
- `plus_input`;
- expected outputs;
- failed hidden test inputs.

## Evaluation condition

Direct and AgentGraph use the same fixed tasks, public prompt, one greedy
candidate per task, empty Tool surface, complete-source submission, and official
EvalPlus evaluator.  AgentGraph may use more model calls because the Director
selects the graph; therefore total model compute is reported rather than
claimed equivalent.  No result from a smoke subset is labelled as the formal
fixed-100 score.

Training, GRPO, LoRA, MACE, Bayesian updates, Skill retrieval, and Skill
evolution are disabled.

## Runtime-contract v2 correction

The initial fixed-100 receipts exposed three adapter defects. The successor
condition `config/evaluation_mbppplus_runtime_contract_v2.yaml` corrects them
without adding a role inventory, fixed Agent order, topology prior, Tool, or
training path.

| Correction | Classification | Source and boundary |
|---|---|---|
| Remove mandatory `sanitize` before `check_correctness` | Direct upstream alignment | EvalPlus 0.3.1 `evaluate.py` submits `solution` directly, and SkillFlow's sealed `run_mbpp` executes the complete submission. The old adapter could discard a later legal definition and change Python semantics. |
| Validate raw Python syntax and the exact public module-level `entry_point` before `FINISH` | Necessary MBPP+ task adapter | Uses only the public `metadata.entry_point`; it never executes code or reads base/plus inputs. The callback extends FlowSteer's existing FINISH terminal-constraint feedback so the Director can repair the current Output artifact before evaluator submission. |
| Use action mask v3 with `add_subgraph` | Direct reuse of existing FlowSteer unified runtime | The v3 live target domain is populated from `AgentRuntime.registered_execution_profiles()`. With no MBPP+ Tool runtime, only the registered `reasoning` profile is admissible; unavailable `react`/`coding` declarations are no longer proposed. One accepted `add_subgraph` remains one progressively executed functional-unit Canvas edit. |
| Freeze the next condition to authenticated model catalog v8 | Direct reuse of existing receipt-pruned catalog plus necessary local-service adaptation | Existing permanent HTTP 403 receipts already excluded the unavailable remote model IDs. v8 preserves v7's sole local Qwen3.5-9B model and adds only the existing `SGLANG_API_KEY` environment-variable reference required by the current local Supervisor; it does not add a routing abstraction or retry paid endpoints. |

The initial configuration and results remain preserved as the measured v1
condition. The v2 condition writes to separate artifact and report paths.

## Runtime-contract v3 relation-schema correction

The completed v2 fixed-100 trajectories contained 416 Director turns. The
generic v3 parameter schema admitted actions that the existing strict parser
then rejected: 79 `add_subgraph` self-loops and 40 repeated unordered endpoint
pairs. These are action-domain projection defects, not benchmark failures or
model capability failures.

The versioned `complete_source_live_targets_v3` condition keeps the same fixed
tasks, EvalPlus evaluator, Qwen3.5-9B policy, sampling purpose, seed, execution
budget, and free-Agent protocol. Its generic `add_subgraph` parameter phase now
uses the exact-candidate JSON Schema pattern already used by the existing
FlowSteer verified-QA path:

- every candidate has two different endpoint IDs;
- every candidate is incident to at least one Agent declared in the current
  `add_subgraph` transaction;
- relations solely between existing Canvas Agents remain `set_relation`
  actions;
- both directed orientations remain available, and a reciprocal relation is
  available between two Agents declared in the same transaction;
- the transaction samples at most one unordered endpoint pair, while later
  Canvas-validated `set_relation` actions may extend the topology after
  execution feedback.

This correction removes invalid actions from the live search space without
introducing fixed roles, a fixed Agent count, a fixed topology, a coding Tool,
or any task-specific solution template. The v3 condition writes to separate
artifact, evaluator-cache, and report paths.

The completed v3 fixed-100 run has full evaluator coverage. Direct obtained
83/100 Base and 71/100 Plus passes; AgentGraph obtained 88/100 Base and 73/100
Plus passes. All 100 AgentGraph trajectories explicitly finished with zero
terminal failures. Across 348 Director turns, the two corrected relation
rejection classes were both zero. The first collection attempt timed out on
one task; the existing checkpoint/resume path reused all completed receipts
and collected only that missing task, while preserving the timeout receipt.

## Stepwise ReAct Action--Observation feedback (v12)

The MBPP+ ReAct path is a thin bridge between two upstream execution
boundaries; it does not add a fixed role sequence or topology:

| Boundary | Reused implementation | Local adaptation |
|---|---|---|
| Outer progressive edit | FlowSteer `InteractiveWorkflowEnv.step`, `_format_feedback`, `TurnRecord`, and `InteractiveWorkflowBuilder.run_loop` | One accepted Director edit is executed and its execution feedback is appended to the next Director observation. |
| Inner bounded action | SkillFlow `BoundedAgent.execute_turn`, `_execute_action`, `EnvironmentObservation`, `StructuredAction`, and `materialize_trajectory_step` | One structured Agent action produces one public observation and one committed trajectory step. |
| Existing local runtime | `ToolReactExecutionAdapter.execute`, `AgentRequest.action_history`, `prior_tool_receipts`, and `ReactExecutionError` | Reuse parsing, Tool dispatch, public receipts, continuation, and failure metadata unchanged. |
| MBPP+ Tool surface | `MBPPPlusReactExecutionAdapter` and `mbpp-plus.python_exec::run_public_test` | Expose only prompt-visible public assertions; EvalPlus Base/Plus inputs and expected outputs remain evaluator-only. |

The v11 fixed-100 audit found that live `public_react_state` described only
currently resumable Agents. In a parallel Runtime tick, a ReAct Agent that had
already completed could therefore be absent while a sibling remained pending.
The v12 compatibility layer records only the current Canvas step's new
`react_trace` entries (entries restored from a prior revision are explicitly
excluded), and publishes them as `react_events` in the same FlowSteer execution
feedback. Each event carries the original task objective, Agent contract,
completion objective, execution status, exact structured Action, public
Observation, and remaining Tool budget. The event outbox is cleared when the
next Director action begins; the full trajectory continues to retain every
historical event.

This corrects an execution-feedback transport defect. It does not prescribe an
Agent count, contract, role, relation, Output Agent, or terminal sequence.
Stepwise ReAct Agents remain incompatible only with a reciprocal FlowSteer
block because the latter requires a complete draft/revision response before
returning control; directed and non-linear AgentGraph topologies remain in the
search space. Training, GRPO, LoRA, MACE, Bayesian updates, Skill retrieval,
and Skill evolution remain disabled in v12.

## Lossless current-step feedback and pending-only repair admission (v17)

The current-step compatibility layer keeps a lossless `react_events` outbox in
the persisted FlowSteer feedback receipt while projecting one compact merged
`react_state` to the next Director prompt. The merged state includes pending,
completed, and exhausted parallel ReAct siblings, so every newly produced
public Action--Observation transition carries the original task objective,
Agent completion objective, execution status, exact Action, Observation, and
remaining budget across the Canvas boundary. Full history remains in the
trajectory and is not replayed into the one-turn Director history window.

The v16 audit exposed one action-mask defect after widening state visibility:
the pending-boundary success check included already completed siblings, whose
legal `observation_status=completed` was not equal to `success`. That could
incorrectly admit `modify_agent`. v17 retains all sibling observations for the
Director but restricts repair admission to Agents in
`_stepwise_react_pending_agent_ids()`. A unit test covers a successful pending
Agent coexisting with a completed sibling.

The fixed-100 v17 run has 100/100 official EvalPlus evaluator coverage. Of 809
public ReAct events, all 801 events followed by another Director decision were
present in that decision's prompt with the task objective, final objective,
status, Action, and Observation; the remaining eight occurred on the last
allowed round of max-rounds trajectories and remain persisted. Direct obtained
83/100 Base and 72/100 Plus passes. AgentGraph obtained 81/100 Base and 69/100
Plus passes, with 91 explicit FINISH trajectories and nine max-rounds
trajectories. The engineering correction therefore closes the feedback
transport and repair-admission defects but does not establish an accuracy
improvement. No evaluator-only field was exposed and no training or Skill path
was enabled.
