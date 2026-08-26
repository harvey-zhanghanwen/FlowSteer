# ALFWorld Initial Adapter v1 source map

## 1. Scope and source priority

This document records the executable sources and compatibility boundaries for
the ALFWorld initial adaptation. The attached papers and the project design
document are references; the implementation requirements come from the user's
request.

The source priority for this adaptation is:

1. `FlowSteer_MACE_Bayesian_Skill_Design.md` for the unified free-AgentGraph
   contract;
2. SkillFlow / SkillEval production code for the ALFWorld task, session,
   action, observation, termination, and evaluator contracts;
3. the official ALFWorld `AlfredTWEnv` implementation and data release for
   environment semantics; and
4. FlowSteer for progressive Canvas editing, execution feedback, `FINISH`, and
   trajectory records.

Only a thin Dataset / Environment Adapter is in scope. The unified
orchestration core must not be replaced or specialized into a fixed embodied
workflow.

The status terms in this source map have exact meanings:

- **Direct reuse**: the upstream interface or semantics is retained.
- **Necessary adaptation**: a thin conversion is required to connect an
  upstream interface to the existing project contract; it must not change the
  benchmark semantics.
- **Not enabled**: the capability may exist elsewhere in the repository but is
  outside this evaluation-only adaptation.

## 2. Authoritative source locations

| Source | Location | Boundary used by this adaptation |
| --- | --- | --- |
| Project design document | `/ssd1/iclr/1/.codex/attachments/b382a86f-bca2-4607-9b56-f4eba78f50a3/FlowSteer_MACE_Bayesian_Skill_Design.md` | `Agent = agent_id + model_id + free-text contract`, two-bit relations, finite bidirectional execution, one Output Agent, progressive Canvas state, explicit `FINISH`, and terminal task reward. |
| FlowSteer paper | `/ssd1/iclr/1/.codex/attachments/10523847-8765-4128-8ff5-31b02e7ed265/FlowSteer__Towards_Agents_Designing_Agentic_Workflows_via_Reinforced_Progressive_Canvas_Editing__3_(1).pdf` | Progressive `Canvas edit -> execute -> feedback` interaction and terminal evaluation. |
| FlowSteer code retained in this repository | `src/interactive/workflow_builder.py::InteractiveWorkflowBuilder.run_loop/run_loop_async`, `src/interactive/workflow_env.py::InteractiveWorkflowEnv.step`, and `src/aflow_executor.py::AFlowExecutor.execute_workflow` | Mutable Canvas, execution of the current workflow after an accepted edit when `execute_each_step` is enabled, execution feedback, trajectory records, and bounded workflow execution. |
| SkillFlow paper | `/ssd1/iclr/1/.codex/attachments/d99c93d9-b586-4028-9eae-0ad940dc2c2b/SkillFlow.pdf` | ALFWorld as a text-interactive benchmark and Success Rate as the native terminal metric. |
| SkillFlow public ALFWorld adapter | `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/skillev/benchmarks/alfworld.py::ALFWorldPublicItem/ALFWorldEnvironment` and `src/skillev/benchmarks/_embodied.py::EmbodiedPublicItem.to_rollout_task/EmbodiedTextEnvironment.execute/_command_from_action` | Public item, ordered task provider, `resource_id="alfworld"`, `name="act"`, exactly `arguments={"command": ...}`, contiguous environment steps, public observations, terminal state, and action budget. |
| SkillFlow official ALFWorld bridge | `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/packages/private-evaluation/src/skillev_private/benchmarks/alfworld_official.py::_OfficialEpisodeState.create/execute/_create_pinned_env/OfficialALFWorldEpisodeFactory.create` | Pinned task/game/seed/step limit, one official episode, reset validation, serial native steps, terminal-only success, and public/private separation. |
| SkillFlow official environment worker | `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/packages/private-evaluation/src/skillev_private/benchmarks/official_environment_worker.py::ALFWorldWorker.reset/step/create_alfworld_env/create_alfworld_builder/reset_alfworld_env` | `AlfredTWEnv` construction, split routing, batch size 1, reset, native `step(action)`, `admissible_commands`, observation, terminal flag, and terminal `won`. |
| SkillFlow private evaluator | `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/packages/private-evaluation/src/skillev_private/benchmarks/alfworld.py::PrivateALFWorldTerminalEvaluator.evaluate/PrivateALFWorldSessionFactory.create` | Native `alfworld-success`, reward `float(success)`, one episode/evaluator bundle per task, empty retrieved-Skill set, and no LLM judge. |
| SkillFlow deployed compatibility adapter | `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/ragen_adapter.py::AlfredEnvConfig/ALFWorldEnv.reset/ALFWorldEnv.step/RAGENAdapter.reset/RAGENAdapter.step` | ALFWorld inventory, reset/session state, native environment step, validity/effectiveness fields, terminal `won`, available-action updates, and TextWorld registration with `max_episode_steps=50`. |
| SkillFlow bounded ReAct execution | `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/training/task_prompts.py::ALFWORLD`, `training/gflownet_trainer.py::GFlowNetTrainer._run_react_episode`, and `training/react_prompts.py::ALFWORLD_TEMPLATE` | One ReAct Agent execution runs a bounded Action--Observation loop of at most 20 ALFWorld policy turns; each accepted turn emits one native command and consumes the next public observation. |
| SkillFlow evaluation protocol | `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/configs/evaluation/protocol_v10.yaml` and `protocol_v10_sources.yaml` | Dataset version `data-2.1.1`; train-main, train-holdout, full `valid_seen`, full `valid_unseen`; native seen/unseen Success Rate; official environment evaluator. |
| Official ALFWorld environment | `/home/test/datasets/ALFWorld/repo/alfworld/agents/environment/alfred_tw_env.py::AlfredTWEnv.__init__/collect_game_files/init_env` and `AlfredExpert.step/reset` | Official split inventories, native state transition, admissible actions, `won`, and maximum episode-step enforcement. |
| Official ALFWorld configuration and data | `/home/test/datasets/ALFWorld/repo/configs/base_config.yaml` and `/home/test/datasets/ALFWorld/data/json_2.1.1` | Six official task types, `train`, `valid_seen`, `valid_unseen`, `AlfredTWEnv`, and the 50-step episode limit. |

## 3. Unified AgentGraph contract

The ALFWorld adapter retains the design document's graph

\[
G=(V,E,o), \qquad v_i=(\mathrm{id}_i,m_i,p_i),
\]

where `p_i` is a free-text contract. The adapter does not add a role enum or a
dataset-specific graph template.

The following are prohibited in the initial condition:

- mandatory `Navigator`, `Manipulator`, `Planner`, or `Verifier` roles;
- a required chain, parallel graph, debate graph, or reciprocal graph;
- a fixed Agent count;
- a fixed role-to-model mapping;
- an ALFWorld-specific Output Agent identity; and
- a task-family-specific topology rule.

The Director retains authority over Agent count, `agent_id`, `model_id`,
free-text contract, relations, Output Agent, and `FINISH`. Existing graph
validity constraints remain execution constraints, not rewards: unique Agent
IDs, registered model IDs, non-empty contracts, no self-loop, bounded
two-Agent reciprocal components, acyclic quotient graph, one Output Agent, and
reachability to Output.

ReAct is an Agent **execution mode**, not an Agent role. When selected, the
runtime semantics are the conventional
`Thought -> Action(tool) -> Observation -> Thought -> Final` loop. Selecting
ReAct does not rename an Agent and does not prescribe a graph topology.

## 4. Two distinct action spaces

The two action layers must never be merged.

### 4.1 Director actions

Director actions edit the AgentGraph Canvas, for example adding or modifying an
Agent, changing a relation, selecting the Output Agent, or issuing `FINISH`.
They do not directly change the ALFWorld world state.

### 4.2 ALFWorld environment actions

Environment actions are issued by an executing Agent to the `alfworld` Tool.
The SkillFlow public contract is defined by
`ALFWorldPublicItem.RESOURCE_ID`, `EmbodiedPublicItem.to_rollout_task`, and
`EmbodiedTextEnvironment._command_from_action`:

```text
resource_id = "alfworld"
action      = "act"
arguments   = {"command": <native command>}
```

The legal command set is the current `admissible_commands` returned by the
official environment. Commands such as navigation, object pickup, object
placement, heating, cooling, cleaning, or examination are not redefined by the
project. Locally, `build_environment_execution_resources` registers that same
public Tool shape; `EnvironmentExecutionAdapter.execute` places the selected
native command in `ToolRequest("act", {"command": command})`, and
`EnvironmentToolBackend.invoke` performs only the thin conversion from that
structured request to `RAGENEnvironmentSession.step(command)`. The adapter does
not introduce a second command grammar or rename the public resource.

For every environment decision, the model-visible state contains the task,
current public observation, current admissible commands, and public
Action--Observation history. Goal predicates, terminal `won`, reward, and any
private evaluator state are not model-visible.

An output that cannot be parsed as exactly one current admissible command is an
`invalid_action` attempt. It does not become a successful Tool call and does
not receive a synthetic environment reward. A provider exception, environment
reset failure, and environment step failure remain separate operational
failure classes.

## 5. One session per rollout and shared world state

SkillFlow's `PrivateALFWorldSessionFactory.create` pairs one public task with
one official episode and one private terminal evaluator. The local adaptation
therefore uses this lifecycle:

```text
one rollout task
  -> create one pinned ALFWorld session
  -> reset exactly once
  -> execute contiguous native actions against that session
  -> reach native terminal or the environment action limit
  -> read terminal outcome through the paired evaluator
  -> close the session
```

An accepted Canvas edit may execute the current AgentGraph and return feedback,
but it must not create a fresh ALFWorld game for the same rollout. Consistent
with FlowSteer's `execute_each_step` boundary, the execution after an accepted
edit is the complete currently executable workflow, not merely the Agent just
added by that edit. Within that workflow, selecting ReAct for one Agent invokes
SkillFlow's bounded episode loop: `GFlowNetTrainer._run_react_episode` iterates
up to `task_prompts.ALFWORLD["max_episode_steps"] == 20`, and each iteration
passes one native command through `react_step`. Thus one ReAct Agent execution
may consume multiple native ALFWorld actions (up to the remaining shared
20-turn policy budget); there is no invented rule that one Canvas edit equals
one environment action. Later graph executions continue from the same shared
world state and from the first unconsumed policy turn. The initial adapter does
not reconstruct a failed mid-episode simulator from a model-generated replay;
such a failure remains an operational failure.

The one environment-capable Agent in a rollout observes the complete ordered
environment history. Other Agents receive only artifacts routed to them by the
Director-selected relations. Native environment mutations are serialized
because the official bridge creates a single-game, batch-size-1 environment.
The initial adapter follows SkillFlow's one bounded environment-Agent episode:
exactly one graph Agent may own the stateful Tool, that owner uses ReAct and is
not placed in a reciprocal execution block, while Tool-free Agents use the
ordinary reasoning runtime. This is a stateful-resource legality boundary, not
a named role or a prescribed global topology.

The trajectory must preserve, in order:

```text
task identity
-> reset observation and admissible commands
-> AgentGraph revision / executing Agent
-> native action
-> public observation
-> updated admissible commands
-> state-advance / validity information
-> terminal flag
-> terminal native success or operational failure
```

## 6. Canvas, policy-action, and simulator limits

These limits govern different processes and are not interchangeable.

| Limit | Unit | Source and meaning |
| --- | --- | --- |
| `director.max_rounds: 20` | Director/Canvas turns | Project evaluation condition consumed by `src/interactive/director.py::AgentGraphOrchestrator.run`; `AgentWorkflowEnv.step` applies one Canvas action and, when `execute_on_edit=true`, invokes `AgentRuntime.execute` for the current workflow and returns that execution feedback. It is not an ALFWorld command counter. |
| `action_policy_budget: 20` | ReAct policy turns inside the rollout-owned episode | SkillFlow `training/task_prompts.py::ALFWORLD` sets `max_episode_steps=20`, and `GFlowNetTrainer._run_react_episode` performs that bounded loop. The local `EnvironmentExecutionAdapter.execute` continues from `len(receipts) + 1`, so all ReAct Agent executions in the rollout share the same 20-turn budget. Parse failures consume a policy turn even when no world mutation occurs. |
| `simulator_step_cap: 50` | Accepted native environment actions | Official ALFWorld `configs/base_config.yaml` (`max_nb_steps_per_episode: 50`) and SkillFlow `ragen_adapter.py::ALFWorldEnv.reset`, which registers TextWorld with `max_episode_steps=50`. This is an independent simulator horizon. |

Therefore:

- the two numerical values `20` for Canvas rounds and policy turns remain
  separate counters even though this initial condition uses the same value;
- one accepted Canvas edit triggers execution of the complete current workflow
  (with unchanged upstream outputs eligible for progressive reuse), whereas one
  ReAct Agent execution may perform several serialized ALFWorld actions before
  returning its execution artifact and feedback;
- the 50-step simulator hard cap must not replace SkillFlow's 20-turn policy
  budget or permit 50 additional Canvas edits;
- environment parse failures that do not advance the state are recorded
  separately from accepted native steps; and
- `max_rounds` is a Director terminal failure, while environment step-limit
  exhaustion is a valid native episode with success `false` unless the
  simulator reports terminal success.

## 7. Dataset splits and evaluation protocol

The formal ALFWorld protocol is SkillFlow protocol v10, materialized locally by
`config/datasets_alfworld_protocol_v10.yaml` and
`scripts/prepare_alfworld_protocol_v10.py::prepare`:

| Role | Population | Official split | Use in this initial adaptation |
| --- | --- | --- | --- |
| training | `alfworld-train-main` | `train` | Not used; training is disabled. |
| validation | `alfworld-train-holdout` | deterministic holdout from `train` | May be used for adapter development and smoke tests; it is not a formal final score. |
| final evaluation, seen | `alfworld-valid-seen` | `valid_seen` | Run full split when performing formal evaluation; report its SR separately. |
| final evaluation, unseen | `alfworld-valid-unseen` | `valid_unseen` | Run full split when performing formal evaluation; report its SR separately. |

The repository's historical `data/alfworld_v2` 128-validation / 512-training
view was produced from the deployed **train inventory**. It is a project
train-holdout view and must not be relabelled as `valid_seen`, `valid_unseen`, or
an official ALFWorld final score. The initial adapter may retain that view for
development, but the requested formal evaluation must materialize and report
the two official final populations independently.

Smoke tests use a small, explicitly labelled subset from the same pinned
protocol solely to verify reset, native action, observation, state transition,
termination, evaluator, and trajectory wiring. Smoke results are not merged
with full-split results.

## 8. Native evaluator and reported metrics

The formal reward comes only from the official environment outcome:

```text
episode_success = 1.0 if terminal and won is true else 0.0
Success Rate     = successful terminal episodes / valid evaluated episodes
```

`PrivateALFWorldTerminalEvaluator` reads `final_success()` only after a native
terminal state and emits `native_metric_name = "alfworld-success"`. No LLM
judge, answer string match, Agent self-report, or Output Agent prose may create
success.

The deployed RAGEN bridge also returns the upstream TextWorld `info["score"]`.
This adaptation preserves and reports that value as `episode_score` without
renormalizing or using it in place of terminal success. The adapter's
configurable positive reward (for example `10.0` on a win) remains reward
shaping and must not be presented as an independent official metric.

Results must distinguish:

- valid native episode success/failure;
- explicit Director `FINISH` and Director `max_rounds`;
- native environment action count;
- invalid action count and repeated action count;
- environment reset/step failure;
- provider/model failure; and
- evaluator failure or unavailable terminal outcome.

The Direct and AgentGraph conditions must share the exact same task identities,
environment snapshot, reset seed, 20-turn policy budget, model/tool condition, and
native evaluator. Their only intended difference is orchestration.

### 8.1 Model-visible environment-prompt parity

Both conditions use
`src/interactive/task_evaluator.py::_environment_prompt`, the thin local copy of
SkillFlow `training/react_prompts.py::ALFWORLD_TEMPLATE`:

- Direct calls it from `task_evaluator._evaluate_environment` before each
  `run_graph` policy decision;
- AgentGraph calls it from
  `environment_execution._action_prompt`, which is used by
  `EnvironmentExecutionAdapter.execute`; and
- the AgentGraph ALFWorld branch does **not** append
  `_public_state_feedback` to `model_request.problem`.

`_public_state_feedback` remains in stored `model_calls` and transition receipts
for trajectory diagnostics, but it is not an additional model-visible ALFWorld
prompt block. Consequently task instruction, current observation, admissible
commands, bounded Action--Observation history, and native action syntax are
rendered by the same function in Direct and AgentGraph. The Director-authored
free-text contract and graph communication remain the intended orchestration
difference; no ALFWorld role or topology text is injected by the environment
adapter.

## 9. Mapping to the existing project

| Local boundary | Upstream source | Classification | Required behavior |
| --- | --- | --- | --- |
| `config/datasets_alfworld_protocol_v10.yaml` and `scripts/prepare_alfworld_protocol_v10.py::_default_task_provider/_record/prepare` | SkillFlow `protocol_v10.yaml`, `protocol_v10_sources.yaml`, and deployed `RAGENAdapter`/`ALFWorldEnv` inventory | Necessary adaptation | Materialize one independent train preflight task plus the complete pinned `valid_seen` and `valid_unseen` inventories into the existing `TaskRecord`/JSONL contract without changing instruction, official split, game identity, seed, 20-turn policy budget, or 50-step simulator cap. `prepare_alfworld_dataset.py` is reused only for existing identity/record helpers; it is not the protocol-v10 entry point. |
| `src/interactive/environment_execution.py::build_environment_execution_resources/EnvironmentExecutionAdapter.execute/EnvironmentToolBackend.invoke` | SkillFlow `ALFWorldEnvironment`, `_embodied.EmbodiedTextEnvironment._command_from_action`, bounded ReAct execution, and deployed `RAGENAdapter` | Necessary adaptation | Register the exact public resource `alfworld` and sole action `act(command)`; map each accepted command through the thinnest native-command bridge; bind all Canvas executions in one rollout to one task-scoped session; run the remaining bounded ReAct policy turns inside one ReAct Agent execution; retain the Director-authored free-text Agent contract; serialize mutations; and save every transition receipt. |
| `src/interactive/task_evaluator.py` | SkillFlow official outcome view and `PrivateALFWorldTerminalEvaluator` | Direct semantic reuse through a project receipt adapter | Lock exact task/game/seed/budget, replay only a frozen complete action trace, use only terminal `won`, return binary episode success, and keep infrastructure/evaluator failure separate from task failure. A complete nonterminal trace remains evaluator-valid with zero success; replay never requests a new model action. |
| `src/interactive/agent_workflow_env.py::AgentWorkflowEnv.step` | FlowSteer `InteractiveWorkflowBuilder.run_loop/run_loop_async` plus progressive Canvas and execution feedback | Existing core, unchanged by dataset adapter | Apply accepted graph edits, execute the complete currently executable graph through `AgentRuntime.execute` (reusing unaffected progressive artifacts), return feedback, and admit only legal `FINISH`; do not add an ALFWorld role or topology requirement. |
| `src/interactive/agent_graph.py` and `agent_runtime.py` | Design document free AgentGraph plus FlowSteer execution boundary | Existing core, unchanged by dataset adapter | Preserve free contracts, model selection, directed/independent relations, bounded reciprocal execution, unique Output Agent, and complete Agent communication receipts. |
| `scripts/evaluate_completion_benchmark_round.py` | Existing project evaluation collector plus FlowSteer bounded execution | Necessary evaluation wiring | Freeze paired tasks and conditions, run Direct and AgentGraph independently, checkpoint exact receipts, select the longest task-scoped environment ledger across evaluator and executor receipt locations, rescore historical complete traces without model calls, aggregate native SR, distinguish historical/recovered/unresolved collection failures, retain the first observable typed failure, and assign a separate mutually exclusive receipt-causal primary failure class. |

The existing interactive FINISH capability check may require an Agent that can
invoke the configured environment Tool. `run_task` binds that requirement to
the unique resource ID exposed by the task's actual `ToolRegistry`; therefore
ALFWorld uses `alfworld`, while the existing WebShop adapter keeps
`webshop.environment`. That is a runtime capability constraint, not a named
role, fixed contract, fixed Agent count, or fixed topology. The Director
remains free to choose which Agent receives that capability and how it
communicates with the Output Agent.

### 9.1 Known initial limitation

The current model-admissible action mask constrains Canvas action types but
does not yet encode the correlated environment execution profiles
`(reasoning, [])` and `(react, [alfworld])`, Tool-owner cardinality, or the
reciprocal-block exclusion as structured parameter domains. The initial
adapter states these existing Runtime legality rules in the concise execution
interface and the Runtime rejects violations, which is sufficient for the
independent train-split Stable Zero smoke. It is not an action-mask-level
guarantee: formal evaluation may still record repair turns or `max_rounds`
caused by an illegal profile sampled before Runtime feedback. Changing that
generic Canvas/Director contract is intentionally deferred because this round
must not modify the unified orchestration core.

## 10. Direct reuse, necessary adaptation, and not enabled

### Direct reuse

- official ALFWorld `AlfredTWEnv` reset, state transition, observation,
  `admissible_commands`, terminal, and `won` semantics;
- SkillFlow task/session identity and public/private separation;
- SkillFlow `alfworld` / `act(command)` execution boundary;
- SkillFlow terminal-only success evaluator and Success Rate aggregation;
- FlowSteer progressive Canvas, execution feedback, explicit `FINISH`, and
  trajectory structure; and
- the project's existing free AgentGraph model and finite communication
  semantics.

### Necessary adaptation

- official task records to the project's unified dataset schema;
- `valid_seen` / `valid_unseen` protocol materialization without mixing the
  historical train-holdout records;
- one task-scoped environment session exposed to progressive AgentGraph
  execution;
- serial native state mutation across multiple Agents;
- typed transition, failure, model-call, Tool-call, and evaluator receipts;
- a shared evaluation harness for paired Direct and AgentGraph conditions; and
- Wrong Demo reporting with both the first observable typed failure and a
  mutually exclusive receipt-causal primary failure taxonomy; early errors
  repaired before a complete native episode are not relabelled as terminal
  root causes.

### Not enabled in ALFWorld Initial Adapter v1

- GRPO or any other reinforcement-learning update;
- backward pass, optimizer step, LoRA update, adapter publication, or weight
  synchronization;
- MACE exploration;
- Bayesian posterior, EVSI, or paired-probe scheduling;
- Skill retrieval, Skill injection, Skill validation, or Skill evolution;
- structural rewards or topology rewards;
- an LLM judge; and
- a fixed ALFWorld workflow inferred from smoke-test or evaluation errors.

`PrivateALFWorldSessionFactory` returns `retrieved_skills=()` in the referenced
SkillFlow implementation, which is consistent with this initial no-Skill
condition.

## 11. Historical paths that must not be relabelled

The following existing material remains historical evidence only:

- `data/alfworld_v2/{validation,train}.jsonl`: deterministic train-inventory
  holdout/training records, not official final splits;
- `config/evaluation_alfworld_ragen_environment_stable_zero.yaml`: an earlier
  128-task train-holdout condition;
- `reports/alfworld_ragen_required_actor_v2_stable_zero/`: earlier runtime
  results under that condition; and
- `docs/backups/alfworld_round_01.md`: a draft backup description for the
  earlier condition.

Those tasks, prompt/runtime versions, and scores must not be merged with the
new `valid_seen` / `valid_unseen` results. A new condition/version and a new
artifact directory are required for the initial official-split adaptation.

## 12. Evaluation acceptance checklist

The initial adaptation is complete only when all of the following are backed by
actual receipts:

1. a pinned task resets to its recorded instruction, game identity, seed, and
   initial public state;
2. exactly one task-scoped environment session is used per rollout;
3. each accepted native action belongs to the current admissible-command set;
4. observation and admissible-command updates are forwarded to the next Agent
   decision without exposing reward or `won`;
5. multi-Agent execution shares one ordered world state and never performs
   concurrent mutations;
6. native terminal and the 50-action limit are enforced independently of the
   20-round Director limit;
7. only the paired native evaluator assigns binary success;
8. Direct and AgentGraph use identical task/environment/action-budget/model/
   evaluator conditions;
9. `valid_seen` and `valid_unseen` are reported separately with
   `success / valid / total / SR` and operational failures;
10. every Wrong Demo preserves the first observable typed failure and also
    identifies one primary causal failure from the complete stored environment
    ledger, with `max_rounds` reported separately as a terminal manifestation;
    and
11. no fixed role, Agent count, topology, or ALFWorld workflow has been added to
    the unified orchestration core.

## 13. Completed validation record

The train-split Stable Zero canary completed the full chain with native
`won=true`, score `1`, four accepted actions, zero invalid/repeated actions,
and explicit `FINISH` in both Direct and AgentGraph conditions.

The full official paired evaluations are complete:

| Official split | Direct | AgentGraph | AgentGraph minus Direct | Evaluator valid | Operational failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `valid_seen` | 43/140 (30.71%) | 46/140 (32.86%) | +2.14 percentage points | 140/140 in both conditions | 0 |
| `valid_unseen` | 28/134 (20.90%) | 40/134 (29.85%) | +8.96 percentage points | 134/134 in both conditions | 0 |

AgentGraph explicit `FINISH` / `max_rounds` counts are 46/94 on
`valid_seen` and 38/96 on `valid_unseen`. Two `valid_unseen` episodes reached
native `won=true` without a Director `FINISH`; deterministic replay of their
complete executor-side ledgers therefore corrects the native SR without any
model call. Three historical SGLang connection
failure attempts on `valid_unseen` were recovered through exact checkpoint
resume; all three receipts remain append-only while unresolved failures are
zero.

The receipt-backed reports are:

- `reports/alfworld_initial_adapter_v1/valid_seen_report.json` and `.md`;
- `reports/alfworld_initial_adapter_v1/valid_unseen_report.json` and `.md`; and
- `reports/alfworld_initial_adapter_v1/ALFWORLD_INITIAL_ADAPTER_V1_REPORT_ZH.md`.
