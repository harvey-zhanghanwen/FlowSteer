# WebShop stateful AgentGraph v34 source map

## Scope

v34 is a minimal evaluation-only correction after the v33 fixed-20 gate. It
keeps the same free AgentGraph, Director prompt, Qwen3.5-9B condition, model
catalog, sample order, seed, WebShop action budget, and official evaluator. It
does not train or impose a role, Agent count, relation, or topology.

## Reused execution boundaries

- FlowSteer's progressive Canvas executes every accepted edit before returning
  execution feedback to the Director:
  `/ssd1/iclr/icassp/code/FlowSteer/src/interactive/workflow_builder.py`.
- SkillFlow's serialized WebShop ReAct loop remains the source of native
  Action--Observation transitions, legal actions, terminal state, Average
  Score, and Success Rate:
  `/ssd1/iclr/2/SkillFlow/training/environment.py`.
- One stateful environment owner receives only directed predecessor artifacts;
  all other Agents are tool-free reasoning/verification collaborators selected
  by the Director.

## Necessary compatibility fixes

- A gender-scoped task size such as `wife ... size 5.5` now retains an exact
  public `click[5.5]` option. Gender disambiguation is applied only when a
  visible option itself contains women/men segments.
- A paginated WebShop result page containing `< Prev>` is no longer
  misclassified as a product/detail page. Candidate history remains available,
  but stale product purchase preconditions are not presented as current state.
- The native public `configuration` label starts its own option group. Its
  values are no longer merged into the preceding `size` group, so every
  instruction-matched group retains an independent selection precondition.
- When `Buy Now` is visible, the feedback states whether public purchase
  preconditions are admissible while retaining the native evaluator as the
  authority.

These fixes use only the public instruction, observation, native actions, and
Action--Observation receipts. They do not read reward, hidden targets, or
evaluator state.

## Evaluation gate

- `config/evaluation_webshop_stateful_multi_agent_component_v34_20.yaml`
- `config/evaluation_webshop_stateful_multi_agent_component_v34_128.yaml`

The 128-task condition may run only if the fixed-20 Success Rate is strictly
greater than 50%.

To avoid duplicate inference, the unchanged Direct comparator is reused by
exact task ID, model ID, protocol, and generation seed. The v34 fixed-20 gate
reuses v33 Direct receipts; the frozen 128 condition reuses the complete v31
Direct panel. v34 AgentGraph trajectories are always newly collected.
