# ALFWorld stepwise multi-Agent v10 source map

## 1. Scope

This document records the implementation boundary for
`config/evaluation_alfworld_stepwise_multiagent_v10.yaml`. The condition keeps
the v9 frozen `valid_seen` tasks, model catalog, Tool implementation,
evaluator, action budget, seed and Direct protocol. It aligns preterminal
Output targets and relation candidates with the same complete-graph and
multi-Agent collaboration gates; old v7/v8/v9 artifacts remain immutable.

This is an evaluation-only condition.  GRPO, optimizer updates, LoRA, MACE,
Bayesian updates, Skill retrieval and Skill evolution are disabled.

The architecture remains:

```text
Agent = agent_id + model_id + free-text contract
```

For this ALFWorld condition, the live free-profile action schema does not
accept `role_family`; semantic responsibility is expressed by the free-text
contract.  `execution_mode` is one of `reasoning`, `react`, or `coding`.  In
particular, **ReAct is an execution mode, not an Agent role**.

## 2. Source precedence and reuse classification

| v10 boundary | Primary source | Classification | Retained or adapted behavior |
|---|---|---|---|
| Progressive Canvas transaction | FlowSteer `src/interactive/workflow_env.py::InteractiveWorkflowEnv.step` and `src/interactive/workflow_builder.py::InteractiveWorkflowBuilder.run_loop_async` | Direct reuse | The Director selects one admissible Canvas action; an accepted edit executes once and returns execution feedback before the next Director turn. |
| Atomic functional subgraph | FlowSteer-compatible local `src/interactive/agent_action_parser.py::AgentActionType.ADD_SUBGRAPH`, `src/interactive/agent_workflow_env.py::AgentWorkflowEnv.step`, and `src/interactive/director.py` live `add_subgraph` domains | Direct reuse | One `add_subgraph` action declares one to three Agents and zero or more two-bit relations as one transactional Canvas edit, then executes the resulting revision once. A rejected internal mutation commits nothing. |
| Agent communication relation | Project MD sections 3–4 and FlowSteer's executable graph/Canvas boundary | Direct reuse | Independent, directed and bounded reciprocal relations remain in the open search space. The Director selects endpoints and direction; no fixed chain or topology is injected. |
| ALFWorld task and task-scoped episode | SkillFlow `src/skillev/benchmarks/alfworld.py`, `src/skillev/benchmarks/_embodied.py`, private `alfworld_official.py::_OfficialEpisodeState`, and deployed `src/ragen_adapter.py::RAGENAdapter.reset/step` | Direct semantic reuse | One pinned task uses one reset and one serialized environment session. Every accepted native action produces one public observation and state revision. |
| Native Action--Observation protocol | SkillFlow `_embodied.py::EmbodiedTextEnvironment.execute/_command_from_action` and deployed `ragen_adapter.py::ALFWorldEnv.step` | Direct semantic reuse | The environment Tool accepts one native ALFWorld admissible command. WebShop `search/click` actions are not introduced. |
| Terminal Success Rate | SkillFlow `PrivateALFWorldTerminalEvaluator` and local `src/interactive/task_evaluator.py` receipt adapter | Direct semantic reuse | Success comes only from replay of the frozen native action trace and the real terminal environment receipt. Agent text, Director claims, and an LLM judge never determine reward. |
| Public task-complexity projection | SkillFlow/ALFWorld public task instruction plus local `src/interactive/environment_execution.py::_alfworld_task_facts` | Necessary compatibility adaptation | The existing parser projects public target count, requested transform and task family into `task_facts`. It does not read simulator hidden state, terminal reward, evaluator data or the reference action plan. |
| Multi-Agent admission for complex public tasks | Project MD's free AgentGraph plus FlowSteer's `add_subgraph` transaction | Necessary compatibility adaptation | A complex ALFWorld task initially admits a functional subgraph with at least two Agents. Exactly one Agent owns `alfworld.environment`; at least one auxiliary artifact must be routed to that owner. Contracts, optional roles, models, Agent IDs, relation direction and Output identity remain Director-selected. |
| Functional decomposition guidance | FlowSteer's free-text Agent contracts and SkillFlow's compact Supervisor instruction | Necessary compatibility adaptation | When the live domain requires multiple Agents, the Director assigns distinct complementary contracts and routes a task-relevant artifact to its consumer. No semantic role inventory, fixed chain or fixed topology is introduced. |
| Shared-world serialization | SkillFlow's single task-scoped environment session and the project Runtime's Tool capability admission | Necessary compatibility adaptation | Exactly one Agent is the stateful Tool owner and uses `execution_mode="react"`. Other Agents are stateless collaborators using `execution_mode="reasoning"` and no environment Tool. The owner cannot be in a reciprocal execution block, so one Canvas execution cannot apply concurrent mutations to shared world state. |
| Public feedback to collaborators | FlowSteer execute-and-feedback plus SkillFlow public Action--Observation state | Necessary compatibility adaptation | The next execution receives only the task instruction, current public observation, admissible actions, environment revision, remaining budget, public transition history and collaborator artifacts. Hidden simulator state, `info`, `won` and evaluator reward are excluded. |
| Output/FINISH closure | FlowSteer `FINISH` reuse of the last execution result and SkillFlow step-limit terminal submission | Necessary compatibility adaptation | Before and after environment closure, only prospectively complete-graph-valid Output targets are advertised. After terminal/truncation, a pure `SET_OUTPUT` pointer change preserves the original artifact provenance receipt, Output may be selected once, and the next action is `FINISH`. |
| Relation action mask | FlowSteer's prospective Canvas mutation validation plus the ALFWorld collaboration admission gate | Necessary compatibility adaptation | Every advertised relation candidate preserves the unique Tool owner and at least one collaborator artifact ingress; an edge reversal that the authoritative step boundary would reject is never exposed to the Director. |

## 3. Complex-task boundary

The collaboration gate is scoped by configuration:

```yaml
agent_graph:
  complex_task_collaboration:
    enabled: true
    dataset_scope: ["alfworld"]
    minimum_agents: 2
    require_artifact_delivery: true
```

The Runtime derives complexity only from `task_facts` parsed from the public
ALFWorld instruction.  Tasks requesting more than one target instance or an
explicit transform such as cleaning, heating, cooling, or examination under a
desk lamp require the multi-Agent subgraph.  A simple single-object
pick-and-place task may still use one to three Agents.  This condition does not
inspect a hidden expert plan or a test answer.

The gate constrains capabilities and artifact delivery, not semantic roles:

- exactly one Agent declares
  `execution_mode="react"` and
  `allowed_tools=["alfworld.environment"]`;
- every other Agent declares `execution_mode="reasoning"` and
  `allowed_tools=[]`;
- for a complex task, at least one directed relation delivers a collaborator
  artifact to the Tool owner before the environment action;
- the Tool owner cannot participate in a reciprocal relation because the
  task-scoped environment has one serialized writer;
- no `Navigator`, `Manipulator`, `Planner`, `Verifier`, `ReAct`, or other fixed
  role is required;
- no fixed chain, fan-in, fan-out, or role order is required.

Thus the Director continues to choose free-text contracts and natural
topology.  The compatibility layer only makes multi-Agent collaboration
executable without allowing two Agents to mutate the same ALFWorld session.

## 4. Stepwise execution semantics

For a complex task, the intended control flow is:

```text
public task instruction
  -> Director selects one add_subgraph Canvas transaction
  -> stateless collaborator artifact(s) flow through declared relation(s)
  -> unique Tool owner performs one ReAct execution turn
  -> one native ALFWorld action mutates the task-scoped episode
  -> public Observation + admissible actions + revision + remaining budget
  -> Director selects one next Canvas edit, continue, or finish
```

`continue` is an execution-control action, not a semantic role or graph edge.
On a nonterminal complex episode it re-executes the collaborating graph against
the latest sanitized public state while the single Tool owner remains the only
Agent permitted to emit a native action.  Each resulting native action is
therefore followed by a Director-visible Action--Observation receipt.

## 5. Evaluation invariants retained from v9

The following values are unchanged:

- official split: `valid_seen`;
- ordered sample count: 140;
- Stable Zero canary count: 2;
- seed: `20260825`;
- action budget: 20 native ALFWorld actions;
- Director budget: 32 Canvas turns;
- inference device: GPU0 through the existing local SGLang endpoint on port
  `8015`;
- model catalog: `config/model_catalog_alfworld_paired_qwen35_v1.yaml`;
- Direct protocol: `skillflow_ragen_alfworld_react_20step_v1`;
- terminal evaluator: native ALFWorld terminal replay / Success Rate;
- Tool implementation version:
  `skillflow.alfworld.native-stepwise-recovery.v5`.

AgentGraph still uses constrained Canvas and admissible-action decoding while
Direct uses SkillFlow's raw/tag parser.  Consequently, the Direct/AgentGraph
SR difference is reported descriptively and is not labelled a paired causal
effect.

All v10 artifacts and reports use new versioned paths under
`artifacts/alfworld_stepwise_multiagent_v10/` and
`reports/alfworld_stepwise_multiagent_v10/`. Failed v7/v8/v9 canary results must not be
merged into v10 metrics.

## 6. Acceptance criteria

A v10 Stable Zero or formal evaluation is reportable only if:

1. selected task IDs are the frozen v9 `valid_seen` sequence;
2. every complex task AgentGraph contains at least two Agents;
3. exactly one Agent owns `alfworld.environment` and its ReAct declaration is
   recorded only in `execution_mode`;
4. at least one declared relation delivers a collaborator artifact to the
   Tool owner before an environment action;
5. every environment mutation has one native action and one public observation
   receipt;
6. public state contains no reward, `won`, private `info`, evaluator output or
   hidden simulator state;
7. success is scored only by the terminal ALFWorld evaluator receipt; and
8. no training, optimizer update, LoRA publication, MACE, Bayesian update or
   Skill activation occurred.

## 7. Stable Zero result

The frozen two-task canary completed on 2026-08-30 with status
`stable_zero_confirmed`:

- AgentGraph: 2/2 environment successes, canary SR 100%, episode score 1 for
  both tasks, 0 terminal failures and 0 collection failures;
- Direct: 2/2 environment successes under its retained SkillFlow protocol;
- both AgentGraphs contained exactly two Agents, no node declared
  `role_family="react"`, and each graph contained exactly one
  `execution_mode="react"` owner of `alfworld.environment` plus one stateless
  collaborator;
- 20/20 stepwise Tool-owner execution records contained a current upstream
  collaborator artifact; the two episodes used 12 and 8 native environment
  actions respectively;
- both trajectories ended with an explicit `FINISH`; the v9 preterminal
  Output/edge-reversal dead state did not recur.

The 100% value above is only the two-task Stable Zero canary SR.  It is not a
140-task `valid_seen` evaluation result and must not be reported as formal
benchmark accuracy.

Evidence is stored in
`artifacts/alfworld_stepwise_multiagent_v10/valid_seen/run_manifest.json`,
`direct_predictions.jsonl`, and `agentgraph_trajectories.jsonl`.
