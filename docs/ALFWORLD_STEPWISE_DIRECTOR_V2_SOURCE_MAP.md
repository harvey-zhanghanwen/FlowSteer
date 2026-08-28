# ALFWorld Stepwise Director v2 source map

## Scope

This version changes only the ALFWorld task/environment adapter, execution
control, public feedback, trajectory receipts, and evaluation configuration.
The attached design document and papers are implementation references; the
user request defines the required behavior.

Training, GRPO, MACE, posterior sampling, LoRA updates, and Skill evolution are
disabled. The unified AgentGraph data model and orchestration core are not
replaced by an ALFWorld-specific workflow.

## Upstream mapping

| Local boundary | Upstream source | Classification | Retained behavior |
| --- | --- | --- | --- |
| `AgentGraph`, two-bit relations, unique Output Agent, progressive Canvas | `FlowSteer_MACE_Bayesian_Skill_Design.md` sections 3–4 and FlowSteer `InteractiveWorkflowEnv.step` | Direct reuse | Free Agent identifiers, model selection and contracts; one atomic Canvas edit followed by execution feedback; no fixed role inventory or topology. |
| ALFWorld task-scoped episode | SkillFlow `alfworld_official.py::_OfficialEpisodeState.create/execute` and `ragen_adapter.py::RAGENAdapter.reset/step` | Direct semantic reuse through the existing adapter | One pinned task, one reset, serialized native state transitions, and one close boundary. |
| Native environment action | SkillFlow `_embodied.py::EmbodiedTextEnvironment.execute/_command_from_action` | Direct reuse | `act(command)` with exactly one current ALFWorld admissible command; no WebShop `search` or `click` grammar. |
| One Action–Observation turn | SkillFlow `_embodied.py::EmbodiedTextEnvironment.execute` and `runtime/bounded_agent.py::BoundedAgent.execute_turn` | Direct reuse | One policy action, one environment step at most, then one public observation. |
| Stepwise Director control | WebShop stepwise implementation in `/ssd1/iclr/1/.tmp/FlowSteer-webshop-initial-v1` | Necessary adaptation | `continue`, graph-revision-preserving execution, public environment state, and a short topology-neutral Director prompt. The WebShop action schema is not copied. |
| Stateful Tool namespace | SkillFlow public `resource_id="alfworld"` plus the project's request-scoped Tool registry | Necessary adaptation | Canvas capability `alfworld.environment`; the backend still dispatches SkillFlow `ToolRequest("act", {"command": native_action})`. Legacy bounded-episode conditions retain Tool ID `alfworld`. |
| Terminal evaluator | SkillFlow `PrivateALFWorldTerminalEvaluator` and local `task_evaluator.py` | Direct semantic reuse | Success is read only from a real terminal environment receipt and boolean `won`; a complete nonterminal action-budget closure is valid failure with success `false`. |
| `continue` Canvas action | FlowSteer progressive state/feedback boundary plus SkillFlow single-turn execution | Necessary adaptation | Execution control only: it does not mutate the AgentGraph and does not receive graph-edit semantics or structural credit. |

## Runtime sequence

```text
task
→ create one task-scoped ALFWorld session
→ reset once
→ Director selects one atomic Canvas edit
→ execute current admissible graph boundary
→ unique ReAct Tool owner emits at most one native ALFWorld action
→ environment returns public observation and next admissible commands
→ Director receives environment revision and remaining action budget
→ Director selects another atomic edit, continue, or finish
→ native terminal receipt or measured action-budget closure
→ official evaluator reads terminal outcome
→ close task-scoped session
```

`continue` executes the unique environment Tool owner and its directed dirty
closure while preserving the current graph revision. An accepted graph edit
may also execute the graph once, but one Agent invocation cannot consume the
rest of the ALFWorld episode.

## Authoritative stateful-resource constraints

The following constraints are enforced before a candidate Canvas mutation can
invoke the environment:

1. at most one Agent can own `alfworld.environment`;
2. the owner uses `execution_mode="react"` and allows exactly that Tool;
3. the owner cannot participate in a reciprocal execution block;
4. all world mutations are serialized in the same task-scoped session; and
5. a second owner or reciprocal owner relation is rejected transactionally,
   without an environment action.

These are stateful-resource execution constraints, not Agent roles. The
Director remains free to choose the owner's identifier, model, free-text
contract, directed relations, Output Agent, and every Tool-free Agent.

## Public/private boundary

The Director-visible environment state is restricted to:

- original task instruction;
- environment episode identifier and public environment identifier;
- last native action;
- current public observation;
- current admissible actions;
- environment revision;
- consumed and remaining action budget;
- terminal/truncated status; and
- public state-advance/parse status.

Reward, `won`, episode score, simulator hidden state, private `info`, and
evaluator output are not copied into Agent prompts, Canvas feedback, or the
Director observation. They remain only in the evaluator replay receipt.

## Terminal semantics

A real environment terminal transition makes only explicit `finish`
admissible. When the fixed stepwise action budget is exhausted before a native
terminal, `continue` is unavailable and explicit `finish` closes an
unsuccessful rollout. This closure cannot prove success: the official
evaluator returns success only when the replay ends with a real terminal
receipt whose `won` field is `true`.

## Versioned condition

`config/evaluation_alfworld_stepwise_director_v2.yaml` selects:

- prompt `agentgraph.director.minimal-neutral-scalar-stepwise.v1`;
- Tool version `skillflow.alfworld.native-stepwise-director.v2`;
- `stepwise_director: true`;
- native ALFWorld action budget 20;
- Director Canvas budget 32, leaving room for graph edits and explicit
  `finish`;
- free AgentGraph actions including `continue`;
- the existing paired Qwen3.5-9B Executor catalog for a controlled
  Direct/AgentGraph comparison;
- no required named Agent role or fixed chain; and
- all training and Skill subsystems disabled.

The previous ALFWorld v1 configuration and legacy Tool ID are unchanged and
remain replayable.

## Verification boundary

The deterministic tests cover shared-episode reset/close, consecutive native
steps, graph-revision-preserving `continue`, the exact public Director state,
reward/evaluator isolation, unique Tool ownership, owner capability
preservation, reciprocal-relation rejection before `env.step`, native action
syntax, terminal submission, and nonterminal budget closure. The evaluation
runner's `--prepare-only` mode freezes all 140 `valid_seen` task identities and
validates the v2 configuration without starting a model, API, GPU service, or
environment episode. No Success Rate is produced by prepare-only validation.
