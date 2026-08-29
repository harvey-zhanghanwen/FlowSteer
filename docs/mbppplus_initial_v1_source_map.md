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
| Candidate post-processing | EvalPlus 0.3.1 `evalplus/sanitize.py::sanitize` | Apply the official Python candidate sanitizer identically to Direct and AgentGraph outputs. |
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
candidate per task, empty Tool surface, official sanitizer, and official
EvalPlus evaluator.  AgentGraph may use more model calls because the Director
selects the graph; therefore total model compute is reported rather than
claimed equivalent.  No result from a smoke subset is labelled as the formal
fixed-100 score.

Training, GRPO, LoRA, MACE, Bayesian updates, Skill retrieval, and Skill
evolution are disabled.
