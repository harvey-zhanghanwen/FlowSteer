# WebShop v13 架构验证报告

## 评测条件

- 数据集：WebShop validation，固定样本 `webshop:00500`–`webshop:00627`，共 128 条。
- Orchestrator：本地 Qwen3.5-9B，GPU0 SGLang Supervisor。
- 任务指标：WebShop 官方环境 `Average Score` 与 `Success Rate`。
- 训练状态：未运行训练、GRPO、backward、optimizer update、LoRA、MACE、Bayesian update 或 Skill injection。
- AgentGraph 保持自由搜索空间；没有固定 Searcher、Buyer、Reviewer 等角色，也没有固定 chain。

## 本版修复

1. WebShop Tool owner 的 `execution_mode=react` 与 `allowed_tools=["webshop.environment"]` 作为一个原子 execution profile 提交，避免半配置状态。
2. 每次 WebShop Action–Observation 后，将原始任务目标、最新动作结果、当前公开环境状态、剩余动作预算和 public progress 写入下一轮 Director observation。ReAct 是 Agent 的 execution mode，不是 role。
3. failure recovery 使用 `preserve → diagnose → repair → augment`；保留有效 Agent、episode 和 artifact，不再因一次失败直接删除 Agent。
4. v13 的 typed action mask 不再向 Director 暴露“持有 stateful `webshop.environment` Tool 的 Agent 进入 reciprocal relation”这一 runtime 不支持的组合；independent、unidirectional 和其他合法 topology 保持开放。
5. 报告聚合器补计 Canvas `execution_error=<json>` receipt，避免没有 `runtime_summary.failure_records` 时漏报 `AgentRuntimeError`；不改变推理或 evaluator。

## 正式结果

| Condition | 样本数 | Evaluator valid | Average Score (/100) | Success Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 33.87 | 19/128 = 14.84% |
| AgentGraph v11 | 128 | 128 | 51.68 | 32/128 = 25.00% |
| AgentGraph v12 | 128 | 127 | 46.74 | 23/128 = 17.97% |
| **AgentGraph v13** | **128** | **128** | **52.70** | **34/128 = 26.56%** |

v13 相对 v12：Average Score **+5.96**，Success Rate **+8.59 个百分点**。  
v13 相对 v11：Average Score **+1.02**，Success Rate **+1.56 个百分点**。  
v13 相对 Direct：Average Score **+18.83**，Success Rate **+11.72 个百分点**。

Direct 与 AgentGraph 的 protocol 不等价，因此 Direct 差值仅作 descriptive comparison，不解释为 paired causal effect。v11、v12、v13 使用相同的 128 条 validation 样本与官方 evaluator，可用于版本间架构比较。

## 执行闭环

- 128/128 explicit `FINISH`；terminal failure 0；`max_rounds` 0。
- 751 个正式环境动作，751 个 state-advancing action，invalid action 0。
- 95 个 environment terminal episode；33 个 environment step-limit episode。
- `AgentRuntimeError` 0；stateful reciprocal error 0；partial execution-profile error 0；collection failure 0。
- 自然 topology：126 条 single-Agent；2 条 serial two-Agent；没有强制固定 topology。
- 37 条任务触发 public no-progress 诊断，说明反馈传输已接通，但 action policy 仍未总能有效利用历史状态。

## 重复动作诊断

基于官方 environment trace 的精确 `(observation, action, next_observation)` 重复统计：

| Version | 涉及任务 | 额外重复 transition | no-change transition | 已知零结果 query 重搜 |
|---|---:|---:|---:|---:|
| v11 | 49 | 129 | 117 | 0 |
| v12 | 52 | 143 | 153 | 0 |
| **v13** | **45** | **113** | **134** | **0** |

v13 的精确重复 transition 低于 v11/v12，但 no-change transition 仍高于 v11。剩余问题主要是 candidate revisit、属性验证不足和动作预算分配，而不是 Director 没收到 Action–Observation。

## 定向案例

- `webshop:00515`：v12 在建立不受支持的 reciprocal stateful relation 后连续发生 4 次 `AgentRuntimeError`，最终 Score 0；v13 使用单 Agent、4 个环境动作、6 个 Director rounds 合法完成，Score 1.0。
- `webshop:00540`：v12 连续发生 8 次 reciprocal runtime error，最终 `max_rounds`、无 `FINISH`、evaluator invalid；v13 无 runtime error，12 个 Director rounds 后合法 `FINISH` 且 evaluator valid，但因 no-progress 用满 10 个环境动作，Score 0。

上述结果说明 search-space/runtime capability mismatch 已修复；`00540` 的残余失败属于 action policy 与 no-progress recovery 质量，不再是状态传输或 reciprocal runtime bug。

## 已知边界

- 当前冻结 model catalog 只有已 canary 的本地 Qwen3.5-9B；统一架构支持 model choice，但本轮没有异构 Agent 模型对照。
- 本轮没有 Skill 或训练，因此结果不能解释为 Skill/LoRA 增益。
- 126/128 个最终图为 single-Agent，表明 WebShop 当前策略仍偏向浅 topology；这不是 action mask 强制结果。
- 34/128 完全成功仍然较低，下一步应针对 step-limit、candidate revisit、属性覆盖与 no-progress repair 做开发集上的最小策略调整，不能根据 validation target hard-code workflow。

## 版本与证据

- 分支：`feature/webshop-stateful-relation-mask-v13-20260830`
- 架构修复：`86e6b74 architecture: mask WebShop stateful reciprocal relations`
- telemetry 修复：`e551100 report: count Canvas runtime error receipts`
- 正式报告提交：`a4647f9 eval: record WebShop stateful-safe v13 results`
- 正式 manifest：`artifacts/webshop_stepwise_director_v13/development/run_manifest.json`
- 完整 trajectory：`artifacts/webshop_stepwise_director_v13/development/agentgraph_trajectories.jsonl`
- 机器可读报告：`reports/webshop_stepwise_director_v13/development_report.json`

