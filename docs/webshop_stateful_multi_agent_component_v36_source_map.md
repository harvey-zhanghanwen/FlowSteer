# WebShop stateful multi-Agent component v36 source map

## Evaluation boundary

- Dataset/split/evaluator/model/action budget: unchanged from
  `evaluation_webshop_stateful_multi_agent_component_v34_128.yaml`.
- Sampling schedule purpose and catalog-order namespace: deliberately retain
  the v34 values so the 128-task comparison uses the same scientific sampling
  coordinates. The v36 condition/storage identifiers are new, so no v34
  AgentGraph trajectory is reused.
- Direct arm: reused losslessly from the completed v34 128-task run; no new
  Direct inference is issued.
- Training, GRPO, LoRA, MACE, Bayesian update and Skill evolution: disabled.

## Direct upstream reuse

### SkillFlow WebShop

- `training/environment.py::_reset_react` / `_react_step` /
  `_build_react_prompt`: one environment Action, one real Observation, then a
  new policy decision.
- `training/environment.py::_webshop_tokens`,
  `_webshop_parse_product_options`, `_webshop_parse_results`,
  `_webshop_product_price`: public observation parsing only.
- Native WebShop environment terminal state and graded reward remain the sole
  evaluator authority.

### FlowSteer

- Canvas edit -> execute -> feedback boundary.
- Progressive outputs, directed relations, unique Output Agent and trajectory
  receipts.
- Free Agent declarations remain `agent_id + model_id + free-text contract`;
  v36 adds no fixed Searcher/Reviewer/Buyer role or topology.

## Necessary WebShop compatibility fixes

1. Accept the renderer's public `style name` option label as the `style`
   dimension.
2. Preserve natural public `size is/of/should be ...` and labelled color
   surfaces when the requested value is absent from a candidate; absence is
   `no_visible_match`, not permission to infer a value from that candidate.
3. Normalize quote-mark measurement spellings and bind WebShop W x H title
   positions to explicitly labelled width/height requirements.
4. Retain unranked public result rows and task-only requirement/measurement
   projections in every Action--Observation feedback state.
5. Mask already-inspected product tabs and, only at the public completion
   lower-bound boundary, expose actions on the feasible current-candidate
   purchase path.
6. Treat a tool-free Agent's structured action text as an unverified routed
   artifact. Capability violations require an actual Tool receipt; artifact
   syntax alone is not an execution event.
7. Allow terminal-safe `SET_OUTPUT` when the environment finishes before an
   Output pointer is selected; this pointer-only edit must not re-execute the
   stateful environment.
8. When terminal reward does not prove an earlier causal error, Wrong Demo
   output records the last action as `terminal_observation_action` and leaves
   `first_error_action` null.

All compatibility logic consumes only the immutable public instruction,
current observation, native admissible actions and prior Action--Observation
receipts. Hidden target fields, reward and evaluator details are excluded from
Director and Agent input.
