# HealthBench Professional v2.37：五题开发回归

状态：工程修复已完成，5 题 prepare-only 已通过；本文件不预报实测分数。
用户要求已将 v2.35 两端停止，原结果完整保留。停止记录见
`v235_stop_receipt.json`，固定旧版五题评分见 `baseline_v235_dev5.json`。

## 来源与修复

基于 v2.36 `fc655cc`，移植 v2.35 并行 runner 的 `defb7ea`。保留 MD 自由
`agent_id + model_id + free-text contract`、FlowSteer Canvas 增量执行及
SkillFlow ReAct Action–Observation 和现有官方 evaluator。

1. **搜索预算接线**：原多工具子类没有使用已配置的相关性检查，两个无关
   非空结果也会关掉搜索。现复用父类词面锚检查，所有实际请求仍计总预算。
2. **查询拼写**：长词一次字符编辑可通过既有 scope 检查；短缩写/代码仍
   精确。没有错字词典或真实样本替换表，查询与检索排序不改。
3. **残缺片段**：500 字符片段标出真实截断与可读取 source_id，沿现有
   Agent/Director/知识库通路传递；不根据残句自动生成补全内容。
4. **公开证据边界**：query echo、索引路径和文档编号不再为后续 contract
   提供事实依据。真实来源标题/正文与原对话仍保留。
5. **v2.36 索引**：保留对话、文献、药品标签分区；只索引本节点获取或由
   实际图关系传入的来源。患者陈述不认定为外部医学证据。

这些修复不保证临床推理正确。未解析名称被误解、无依据 summary 被下游
相信、把备选治疗误写成顺序方案仍需要真实回归结果判断。Director 不增加
医疗长提示词，不强制多 Agent 或固定 topology；ReAct 不是 role。

## 固定评测口径

- 五题 ID 在 config 中冻结，全部来自已经用于开发的公开 test 集低分案例。
- 只执行 AgentGraph；旧版相同五题均分：raw **25.36%**、length-adjusted
  **20.91%**。不额外花费额度重跑 Direct。
- Director 仍本地 Qwen3.5-9B，GPU6/8026；原模型池、thinking、20 Director
  回合、每节点 3 次 Tool 调用/6 次 ReAct 回合、并发 4 不变。
- 官方 evaluator 与长度调整保持不变；单题调整分可为负，聚合后才裁剪。
- 新增来源分类索引改变了工具条件，前后比较不是纯粹单一修复的因果消融。
- 不训练，不运行 MACE/Bayesian/Skill evolution，不自动进入 525 题。

配置：`config/evaluation_healthbench_professional_grounded_retrieval_v2_37_dev5.yaml`。
复用及适配清单：`docs/source_map.md`、`docs/adaptation_log.md`。

最小入口（使用现有 provider 环境配置）：

```bash
FLOWSTEER_SUPERVISOR_PORT=8026 FLOWSTEER_ROLLOUT_GPU=6 python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_professional_grounded_retrieval_v2_37_dev5.yaml \
  --collection-arm agentgraph
```

有效结果和完整 trajectory 在独立目录
`artifacts/healthbench_professional_grounded_retrieval_v2_37_dev5/evaluation/`。
