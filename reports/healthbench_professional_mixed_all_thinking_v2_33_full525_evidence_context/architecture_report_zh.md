# HealthBench Professional v2.33 架构修复与评测说明

## 本轮改动

1. **任务语义和约束识别。** Director 使用新的、较短的中性 v19 提示词，区分原始请求与未经验证的前提，保留问题询问的关系及必要上下文；翻译、摘要、格式化不取消矛盾和风险处理。没有固定医疗职责或工作流。
2. **证据判断。** Graph 内 Agent 的执行协议明确区分“未找到证据”与“不存在”，要求结合来源、日期、适用人群和问题关系判断，不能仅以模型记忆否定检索结果。这是提示词约束，不是保证判断正确的形式验证。
3. **模型可见的信息传递。** communication profile v3 保留上游结论，同时传入有界、去重的成功检索摘要；即使上游没有引用，也不再把检索证据完全隐藏。保留来源标识、日期、摘录，以及准确绑定来源的 claim 和 qualifier；区分检索结果与 Agent 的解释，不把“检索成功”当成“事实已验证”。重复文档不重复展开，但不同 Agent 新增的限定条件仍保留。超限会明确标记，完整 receipt 留在 trajectory。

## 保持不变

- MD 的自由 `agent_id + model_id + free-text contract`；自由 Agent 数量、每节点模型选择、单向/有限双向关系和唯一 Output。
- SkillFlow/FlowSteer 的动作、公开 Observation、逐步执行、反馈与 trajectory 边界。一个功能子图作为一次编辑，执行完成后才进入下一次 Director 编排。
- 本地 Qwen3.5-9B Director；既有异构模型目录 v7、thinking、采样预算、seed、并发 4、每题 900 秒。
- 同一公开 test 的固定 525 题，以及官方 simple-evals HealthBench Professional rubric grader 和聚合方法。
- 不训练，不运行 backward/optimizer/LoRA/GRPO，不运行 MACE、Bayesian 或 Skill 更新。

## 版本和对照

- 起点：`ef0a48f`，旧版代码与结果保留于 `/ssd1/iclr/1/.tmp/FlowSteer-healthbench-v1`。
- 新版分支：`feature/healthbench-v2.33-evidence-context-20260905`。
- 用户要求直接切换新版，旧版评测已停止；其 385 条有效 Graph 结果是部分结果，不能作为完整 525 题成绩。
- Direct 作为未改变的历史 control 显式复用：459 条有效回答，66 条按原协议计零的 ReAct 终局失败；完整 525 分母原始评分 **13.968705%**，长度调整后 **16.756929%**。
- 复用前检查来源配置、任务、模型、实际采样/工具/evaluator 条件和原始 receipt，保留旧 condition，不重新生成或评分；不兼容则阻断，不静默重跑 Direct。
- 新版 AgentGraph 尚无正式成绩。历史公开题已用于开发，新版完整结果属于开发后的公开 test 重评，不能宣称为未接触的 held-out 泛化成绩。

## 验证与运行入口

已完成纯本地检查：109 项配置/runtime/evidence 集成测试、94 项旧 gateway/runner 回归、5 项 Director v19 测试、16 项 evidence v3 测试、17 项 Direct 复用测试；另有 63 项 subtests。固定 525 题 prepare-only 已通过。

现有服务：GPU5、端口 8025，本地 `supervisor_theta`；不启动第二套模型，不触碰其他项目服务。

在新 worktree 中，导出项目已有 `.env`（不得打印内容），使用以下入口：

```bash
export FLOWSTEER_ROLLOUT_GPU=5 FLOWSTEER_SUPERVISOR_PORT=8025
export FLOWSTEER_SUPERVISOR_CONTEXT_LENGTH=32768 FLOWSTEER_SUPERVISOR_MEM_FRACTION=0.82
/ssd1/iclr/gpf/venvs/skillflow/bin/python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_professional_mixed_all_thinking_v2_33_full525_evidence_context.yaml
```

恢复该独立 worktree 时，`data/healthbench_professional_official_v1` 应指向原已准备数据目录；保留配置中声明的 v2.32 Direct 来源四个文件。模型、数据库、私有 rubric、凭据和大型 trajectory 不包含在代码备份中。具体源码来源见 `docs/source_map.md`，改动和验证见 `docs/adaptation_log.md`。

正式运行状态、每题执行和评分保存于新命名空间：
`artifacts/healthbench_professional_mixed_all_thinking_v2_33_full525_evidence_context/evaluation/`。
官方汇总由同一 runner 写入本目录的 `evaluation_report.json` 和 `evaluation_report.md`；没有实际运行出的分数不填充、不预测。
