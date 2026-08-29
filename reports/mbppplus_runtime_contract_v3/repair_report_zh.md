# MBPP+ 架构适配缺陷修复与固定 100 题评测报告

## 结论

初版 AgentGraph 相对 Direct 的下降包含真实的适配缺陷，不能仅归因于
Qwen3.5-9B 的代码生成能力。修复 evaluator submission、Runtime action
domain、公开入口终止约束、model catalog 和 `add_subgraph` relation schema
后，固定 100 题已获得完整官方 EvalPlus evaluator coverage：

| Condition | Base passed | Base pass@1 | MBPP+ passed | MBPP+ pass@1 | Evaluator valid |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B Direct | 83/100 | 83.00% | 71/100 | 71.00% | 100/100 |
| AgentGraph runtime-contract v3 | 88/100 | 88.00% | 73/100 | 73.00% | 100/100 |
| AgentGraph − Direct | +5 | +5.00 pp | +2 | +2.00 pp | — |

主指标是 **MBPP+ pass@1**；Base pass@1 是 EvalPlus 同时报告的辅助指标。
所有通过数均由 EvalPlus 0.3.1 `check_correctness` 产生，不使用 LLM judge，
也不根据生成文本中的 `PASSED` 字符串判分。

## 为什么初版下降

初版固定 100 题的测量值为 Direct Base/Plus 86%/72%，AgentGraph
81%/68%。该条件存在以下可复现缺陷：

1. adapter 在正式 evaluator 前强制调用 `evalplus.sanitize`。EvalPlus 正式
   `evaluate.py` 和 SkillFlow `run_mbpp` 都直接执行完整 submission；旧实现
   可能丢弃同名函数的后续定义并改变 Python 语义。
2. action mask 向没有 Tool Runtime 的 MBPP+ Agent 暴露 `coding` 和
   `react` execution mode。旧轨迹产生 80 次 `coding` 与 24 次 `react`
   Runtime rejection。
3. 不可用的远程模型在不同 task 之间被重新尝试，旧轨迹包含 62 次 HTTP
   403。v2/v3 使用既有 receipt-pruned local catalog，只保留本地
   Qwen3.5-9B。
4. `FINISH` 前没有用公开 `entry_point` 检查完整源码是否定义了精确公开
   函数，因而可能提交改名后的实现。
5. v2 虽已修复以上问题，但 generic v3 `add_subgraph` parameter schema
   仍以两个独立 endpoint enum 表示 relation。416 个 Director turns 中，
   Parser 随后拒绝了 79 个 self-loop 和 40 个重复 unordered endpoint pair。

## v3 修复

v3 保留统一自由 AgentGraph：`Agent = agent_id + model_id + free-text
contract`。没有预设 `Coder / Reviewer / Tester`，没有固定 Agent 数量、
固定 topology 或固定 role sequence；`reasoning` 是当前 Runtime 实际注册的
execution mode，不是角色。

generic `add_subgraph` 的最终 parameter schema 改为枚举当前 Canvas 上的
精确合法 relation candidate：

- relation 的两个 endpoint 必须不同；
- relation 必须连接至少一个本次 transaction 新增的 Agent；
- 两个已有 Agent 之间的关系编辑继续由 `set_relation` 表达；
- 保留两个有向 orientation；同一 transaction 的两个新 Agent 之间保留
  reciprocal relation；
- 单次 `add_subgraph` 只采样一个 unordered endpoint pair，后续仍可依据
  execution feedback 使用 `set_relation` 扩展 topology。

该修改把 `AgentActionParser`/Canvas 已有合法性约束前移到 guided JSON
Schema，不添加 task-specific solution、reference answer 或 hidden tests。

## 正式运行收据

- 固定 test population：`mbpp-plus-fixed-100@1`，100 题。
- Director 与 Agent：本地 Qwen3.5-9B，GPU0 SGLang。
- Direct 与 AgentGraph：相同公开 prompt、任务、模型、EvalPlus 0.3.1
  evaluator；每题一个 candidate。
- Training、GRPO、backward、optimizer update、LoRA、MACE、Bayesian、
  Skill retrieval/evolution：均未执行。
- AgentGraph：100/100 explicit `FINISH`，terminal failure 0，API fallback 0。
- 348 个 Director turns：self-loop rejection 0，duplicate endpoint-pair
  rejection 0，malformed JSON 1；另有 3 次 no-op edit rejection 和 1 次在
  未变化 graph revision 上重复已拒绝动作。
- 第一次正式 collection 中 `Mbpp/223` 达到 600 秒 task timeout；resume
  复用了 100 条 Direct 和 99 条 AgentGraph，只重试该缺口。恢复后
  evaluator-valid 为 100/100；第一次 timeout receipt 仍保留在
  `collection_failures.jsonl`。

最终 Agent 数量分布为：1 Agent 59 题、2 Agents 22 题、3 Agents 12 题、
4 Agents 2 题、5 Agents 3 题、8 Agents 2 题。这是 Director 在开放 search
space 中产生的自然分布，不是固定模板。

## 定向验证

- Python syntax compilation：通过。
- v3 config prepare-only：通过，冻结 100 题。
- Stable Zero canary：Direct 与 AgentGraph 均为 3/3 Base、3/3 Plus，
  evaluator-valid 均为 3/3。
- relation candidate、MBPP+ config、公开入口 validator、terminal artifact
  repair 的 7 个定向单元测试：通过。
- 内存 Scripted SGLang client 的一个 rollout collector 测试在当前 Python
  环境中超过 60 秒且没有产生 pytest 结果，按超时终止；这不是 assertion
  failure。对应 live SGLang 路径已由 100/100 正式 evaluator-valid
  trajectories 覆盖。

## 历史条件对照

| Condition | Direct Base/Plus | AgentGraph Base/Plus | 说明 |
|---|---:|---:|---|
| initial v1 | 86% / 72% | 81% / 68% | 存在强制 sanitizer、Runtime mode 和 provider domain 缺陷 |
| runtime-contract v2 | 83% / 72% | 84% / 72% | 官方 complete-source evaluator；relation schema 仍产生 119 次关系拒绝 |
| runtime-contract v3 | 83% / 71% | 88% / 73% | 官方完整 coverage；self-loop/duplicate-pair rejection 均为 0 |

v2 与 v3 是独立正式 generation condition；即使任务和科学采样坐标固定，
并发 GPU generation 也不保证 bitwise-identical output。因此历史条件之间的
百分点变化是测量对照，不宣称为单一代码修改的严格因果效应。v3 内部
AgentGraph 与 Direct 的差值同样按 runner 定义属于 descriptive comparison。

## 证据

- 正式报告：`reports/mbppplus_runtime_contract_v3/evaluation_report.json`
- 人类可读报告：`reports/mbppplus_runtime_contract_v3/evaluation_report.md`
- Run manifest：`artifacts/mbppplus_runtime_contract_v3/evaluation/run_manifest.json`
- AgentGraph trajectories：`artifacts/mbppplus_runtime_contract_v3/evaluation/agentgraph_trajectories.jsonl`
- Direct predictions：`artifacts/mbppplus_runtime_contract_v3/evaluation/direct_predictions.jsonl`
- 首次 timeout receipt：`artifacts/mbppplus_runtime_contract_v3/evaluation/collection_failures.jsonl`
