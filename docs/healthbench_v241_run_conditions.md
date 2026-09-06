# HealthBench Professional v2.41 运行条件

## 范围与来源

三份配置均直接复制 `config/evaluation_healthbench_professional_query_recovery_v2_40_dev5.yaml`，仅修改版本化输出路径、显式启用本轮修复，以及下表声明的样本选择或候选 Skill 上下文。未改写统一 AgentGraph、Director system prompt、医学职责或固定拓扑。

三份配置均已通过 prepare-only（5/5/525 条固定样本），尚未据此声称模型采样或官方评分完成。没有预测分数。

整合验证：新增 context 8 项、evidence repair 19 项、candidate helper 22 项、factory 20 项、candidate runner 13 项；原 Director/context-budget 59 项、completion/HotpotQA runner 61 项，全部通过。均为离线合成测试，不代表真实任务得分。

| 配置 | 样本 | 候选 Skill | 用途 |
|---|---|---|---|
| `evaluation_healthbench_professional_budgeted_recovery_v2_41_dev5.yaml` | 原 v2.40 五题 | 不注入 | 确认上下文预算、证据修复及终局路径 |
| `evaluation_healthbench_professional_candidate_skill_v2_41_dev5.yaml` | 同一五题 | 注入候选提示 | 与同版本修复后无 Skill 条件比较 |
| `evaluation_healthbench_professional_candidate_skill_v2_41_full525.yaml` | 官方 public test 顺序 525 题 | 同一候选提示 | 冻结条件下完整运行 |

## 不变条件

- Director 固定为本地 Qwen3.5-9B，沿用 `agentgraph.director.minimal-neutral.v20`；Executor 继续使用既有 v11 模型目录，自主选择模型与 execution mode。
- 推理使用 GPU 6、现有 SGLang 端口 8026；配置不启动其他 GPU 上的模型或训练服务。
- `sampling_schedule_purpose` 保持 `healthbench_professional_public_test_full525_v2_27`，seed 保持 20260825，catalog order namespace 不变。
- thinking、temperature、top-p、top-k、模型目录、工具目录、官方 grader、对话输入与 rubric 隔离保持 v2.40 条件。
- 每题 900 秒，concurrency 4，Director 最多 20 轮；每次 Agent ReAct 最多 6 轮、3 次工具调用及 2 个成功查询，未额外增加工具预算。
- Direct 配置保留用于接口复现，不代表重新调用 Direct；本轮 AgentGraph 执行应显式使用 `--collection-arm agentgraph`。
- 无训练、backward、optimizer update、LoRA 发布、GRPO、MACE 或 Bayesian 更新，`skills.enabled` 仍为 false。

## 本轮明确变化

全部配置开启：

```yaml
director:
  context_projection: true
  max_prompt_tokens: 24000
  max_context_tokens: 32768
  reasoning_context_reserve_tokens: 4608
healthbench_tool_runtime:
  enable_evidence_repair_feedback: true
```

Director 使用真实 tokenizer 计算 prompt 长度。投影只移除冗余工具 schema、重复或旧正文；原始完整问题、当前图、合法动作、当前 evidence/source identity 和失败分类仍保持可见，原始记录留在 trajectory。REASONING 阶段给 ACTION 和模板保留 4608 tokens，因此实际阶段输出上限可能小于原配置，真实值必须读取 phase receipt。不能将此称为生成条件完全不变。

两份候选配置另外开启：

```yaml
candidate_skill_evaluation:
  enabled: true
  profile_path: config/healthbench_candidate_skills_v241.yaml
skills:
  enabled: false
```

这是显式的、版本化的 candidate prompt evaluation，不是 ACTIVE Skill，也不是通过 MACE/贝叶斯证据门槛自动发布的 Skill。候选上下文只能包含通用编排经验，不含具体题目答案、rubric、reference response 或特定医学结论；自由 Agent 数量、自由 contract 和模型选择保留。

## 样本与指标边界

dev5 沿用此前已经用于诊断的公开测试样本，不是无偏 held-out 估计：

1. `healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a`
2. `healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181`
3. `healthbench-professional:4f08ae480b16ef825cf098eca6530e68`
4. `healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3`
5. `healthbench-professional:37101607e2947481e85e8fe3597a1acf`

full525 为 `stage: final_evaluation`、`split: test`、`benchmark_slice: public_test`、`selection: sequential`、`sample_count: 525`。它不重切数据；同一公开测试集已被开发过程反复观察，因此即使全量完成也不能宣称完全独立、无偏的泛化估计。

报告使用官方 rubric aggregation 的 raw 与 length-adjusted score，不使用 EM/F1 或二元 Accuracy 替代；同时给出固定分母、evaluator-valid、FINISH、max_rounds、timeout/provider/grader failure。缺少有效 evaluator receipt 的题目保留 N/A，不伪造成 0 或从总体分母消失。候选 Skill 效果只能由已完成同样本对照确认，不能因为注入候选就声称提升。

## 输出隔离与恢复

每份配置的 condition ID、`artifacts/<condition>/evaluation`、`artifacts/<condition>/evidence_indexes`、`reports/<condition>` 均独立。不得将前一条件的成功题、共享运行态 evidence index 或后续补跑结果静默混入另一条件。

复现时读取对应配置中的 `selected_tasks.jsonl`、`run_manifest.json`、`agentgraph_trajectories.jsonl`、`collection_failures.jsonl`、`evaluator_private/partial_trajectories.jsonl` 和官方 `evaluation_report.json`。完整 model/provider、实际 prompt、candidate profile、token budget 与 source receipt 应随运行 manifest/trajectory 保存；已有历史结果不覆盖。
