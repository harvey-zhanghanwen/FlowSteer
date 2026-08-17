# TriviaQA 第一轮架构验证

固定项目 validation：**128** 题。Direct 与 AgentGraph 使用同一批题、同一 Qwen3.5-9B、同一公开检索观察和同一终局 evaluator。本轮未执行训练、GRPO、反向传播、优化器更新、LoRA 发布或 Skill 注入。

评测采用 SkillFlow Formal Protocol 10 兼容的答案归一化：对 accepted answers 取最大 token F1，并报告 normalized exact match。检索复用 SkillFlow `RetrievalIndex.search/read`；当前适配采用确定性问题查询预取，不等同于 SkillFlow 的模型驱动多轮 `search/read/complete` 完整协议。

| 条件 | 完成 | evaluator 有效 | 严格 EM | 严格 F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 51.56% | 57.90% |
| Progressive AgentGraph | 128 | 128 | 52.34% | 61.80% |

AgentGraph − Direct：**+0.78 EM**，**+3.90 F1**。

## Failure Types

- `architecture_gain`：6
- `architecture_regression_candidate`：5
- `correct`：61
- `director_max_rounds`：12
- `partial_or_overlong_answer`：17
- `shared_reasoning_or_model_failure_candidate`：27
