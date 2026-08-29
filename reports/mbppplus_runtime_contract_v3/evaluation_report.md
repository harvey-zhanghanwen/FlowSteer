# MBPP+ Architecture Validation

Fixed test samples: **100**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **pass_at_1** (`EvalPlus_MBPP_plus_official_pass_at_1_with_base_pass_at_1_auxiliary`). AgentGraph explicit FINISH: **100/100**; terminal failures: **0**; operational/evaluator failures: **0**.

Terminal-output parsing failures: **Direct 0**, **AgentGraph 0**.

| Condition | Completed | Evaluator valid | Strict pass_at_1 |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 100 | 100 | 71.00% |
| AgentGraph | 100 | 100 | 73.00% |

AgentGraph - Direct: **+2.00 percentage points**.

Direct and AgentGraph are separate protocols; their delta is descriptive, not a paired causal estimate.

## Failure types

- `agentgraph_higher_pass_at_1`: 7
- `direct_higher_pass_at_1`: 5
- `equal_pass_at_1`: 88
