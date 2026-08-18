# TriviaQA 第一轮架构验证

固定项目 validation：**32** 题。Direct 与 AgentGraph 使用同一批题、同一 Qwen3.5-9B、同一公开检索观察和同一终局 evaluator。本轮未执行训练、GRPO、反向传播、优化器更新、LoRA 发布或 Skill 注入。

评测采用 SkillFlow Formal Protocol 10 兼容的答案归一化：对 accepted answers 取最大 token F1，并报告 normalized exact match。检索复用 SkillFlow `RetrievalIndex.search/read`；当前适配采用确定性问题查询预取，不等同于 SkillFlow 的模型驱动多轮 `search/read/complete` 完整协议。

| 条件 | 完成 | evaluator 有效 | 严格 EM | 严格 F1 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 32 | 32 | 50.00% | 57.29% |
| Progressive AgentGraph | 32 | 32 | 43.75% | 53.12% |

AgentGraph − Direct：**-6.25 EM**，**-4.17 F1**。

## Failure Types

- `architecture_gain`：1
- `architecture_regression_candidate`：3
- `correct`：13
- `director_max_rounds`：1
- `partial_or_overlong_answer`：4
- `shared_reasoning_or_model_failure_candidate`：10
