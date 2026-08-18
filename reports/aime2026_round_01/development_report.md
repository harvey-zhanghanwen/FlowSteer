# AIME 2026 Architecture Validation

Fixed validation samples: **30**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill publication ran. No Skill was injected.

Primary metric: **exact_match** (`SkillFlow_exact_answer_extraction_and_exact_match`). AgentGraph explicit FINISH: **30/30**; terminal failures: **0**; operational/evaluator failures: **0**.

| Condition | Completed | Evaluator valid | Strict exact_match |
|---|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 30 | 30 | 0.00% |
| AgentGraph | 30 | 30 | 66.67% |

AgentGraph - Direct: **+66.67 percentage points**.

## Failure types

- `agentgraph_exact_match_gain`: 20
- `both_incorrect`: 10
