# WebShop stateful multi-Agent component v37 source-map delta

## Delta from v36

This v37 profile is a fresh formal execution identity for the same controlled
128-task WebShop validation panel documented in
`webshop_stateful_multi_agent_component_v36_source_map.md`. It retains the v36
orchestration core and adds only three defects found by read-only replay of the
completed v34 panel. The interrupted v36 AgentGraph run is diagnostic-only;
none of its trajectory, receipt, paired-result, wrong-demo, manifest or report
artifacts may be reused by v37.

The minimal environment-adapter changes are:

- bind a measurement as a hard option constraint only within the corresponding
  public measurement-bearing option dimension (or an explicitly named option
  group); compatibility measurements such as `for our 60 inch TV` remain
  candidate evidence and do not become a `style name` assignment;
- parse only the bounded color phrase adjacent to `color`, `colour`,
  `colored`, or `coloured`, while retaining SkillFlow's exact-first and
  same-visible-option word-containment behavior for composite/prefixed labels;
- define purchase `admissible` as a property of the current executable state,
  so it additionally requires a retained product context and a currently
  visible `Buy Now` action. Cross-page reachability remains represented by
  `minimum_actions_to_purchase`.

These checks consume only the original public instruction, current WebShop
observation, admissible native actions, and prior public Action--Observation
receipts. They do not inspect the hidden goal, native reward, or evaluator
state, and they do not prescribe an Agent role, count, relation, or topology.

The v37 configuration changes only these versioned identities:

- experiment name and condition: `webshop_stateful_multi_agent_component_v37_128`;
- Tool receipt version: `skillflow.structured-action.webshop.stepwise-director.stateful-multi-agent.v37`;
- behavior-policy receipt: `qwen35-9b-base-webshop-stateful-multi-agent-v37-128`;
- AgentGraph profile: `webshop.stateful-multi-agent-component.v37`;
- storage schema and all artifact/report paths: fresh v37 namespaces;
- policy-sync adapter prefix: fresh, unused v37 namespace (policy sync remains disabled).

## Frozen scientific condition

The following scientific-control fields are equivalent to v36 and preserve the
v34 comparison coordinates:

- dataset/stage/split/selection/sample count:
  `webshop/development/validation/sequential/128`;
- seed and Direct generation seed: `20260825`;
- sampling schedule purpose:
  `webshop_native_validation_stateful_multi_agent_v34`;
- catalog-order namespace:
  `webshop_stepwise_v34_nonthinking_32768_catalog`;
- rollout count/concurrency/task timeout: `1/2/900s`;
- environment action budget/Director round budget: `10/20`;
- Director model/backend/served model: local Qwen3.5-9B, SGLang,
  `supervisor_theta`;
- prompt, decoding, action-mask, model-catalog and AgentGraph constraints;
- native WebShop environment and official reward/evaluator protocol;
- training, GRPO, LoRA, exploration and Skills remain disabled.

## Direct baseline reuse

The completed Direct arm is reused losslessly from
`artifacts/webshop_stateful_multi_agent_component_v34_128/validation/direct_predictions.jsonl`.
No Direct inference is requested. AgentGraph output, receipts and reports are
written only under the v37 namespaces, so neither v34 nor interrupted v36
AgentGraph artifacts can enter the formal result.

## Upstream provenance

All implementation provenance and WebShop compatibility details remain exactly
those recorded by the v36 source map:

- SkillFlow WebShop reset/step/Action--Observation interface and public
  observation parsers;
- native WebShop terminal state, reward and evaluator;
- FlowSteer Canvas edit--execute--feedback boundary, directed relations,
  progressive outputs and trajectory receipts;
- free Agent declarations (`agent_id + model_id + free-text contract`) with no
  fixed shopping role or topology.

No hidden target, reward or evaluator field is exposed to the Director or any
Agent.
