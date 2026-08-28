# ALFWorld Stepwise Director v2 架构完成报告

## 结论

ALFWorld 已完成 stepwise Director 的代码接线与无模型定向验证。新条件在同一任务内保留一个 task-scoped environment episode；每次 ReAct Agent 调用最多执行一个 ALFWorld 原生 action，并立即向 Director 返回公开 observation、admissible actions、environment revision 和 remaining action budget。`continue` 不修改 AgentGraph revision。

本轮没有启动模型服务、GPU、API、GRPO、LoRA、MACE、Bayesian posterior 或 Skill evolution，也没有产生 Success Rate。正式 ALFWorld Stable Zero 环境 canary 与完整 evaluation 尚未执行。

## 完成的边界

- 新增版本化条件：`config/evaluation_alfworld_stepwise_director_v2.yaml`。
- 新增简短、topology-neutral 的 stepwise scalar Director prompt。
- 新增 `continue` execution control；它不是 AgentGraph mutation，也不是 Agent role。
- 复用同一个 ALFWorld session；task reset 与 runtime close 有明确生命周期。
- ALFWorld 保留 SkillFlow `act(command)` 和原生 admissible command，不使用 WebShop `search/click`。
- Canvas capability 使用 `alfworld.environment`；旧 bounded-episode 条件继续使用 `alfworld`。
- 唯一 Tool owner、`execution_mode="react"`、exclusive toolset、非 reciprocal block 在 Canvas mutation 提交前校验。
- Owner 的 Tool capability 在 episode 开始后受 preservation gate 保护；模型可继续修改其 contract 或 model。
- `continue` 执行 Tool owner 及其 directed dirty closure，graph revision 保持不变。
- Director 可见状态使用白名单；reward、`won`、episode score、hidden simulator state 和 evaluator `info` 不进入 Director observation 或 Agent communication。
- 真实 terminal receipt 后只开放显式 `finish`；stepwise action budget 耗尽可显式结束失败 rollout，但不能被判定成功。
- trajectory metadata 保存 environment episode ID、execution boundary、current public state、terminal/truncated 和 action budget。
- Director round budget 为 32，独立于 20 个 ALFWorld policy action budget。

## 验证结果

- 定向测试：`285 passed`，包含 50 个 subtests。
- Python 编译检查：通过。
- `git diff --check`：通过。
- prepare-only：通过；冻结官方 `valid_seen` 的 140 个 task identity。
- prepare-only manifest：`artifacts/alfworld_stepwise_director_v2/valid_seen/run_manifest.json`。
- training enabled：`false`；optimizer updates：`0`。

完整 unit suite 试运行中，唯一与本次改动相关的兼容性失败已修复并纳入上述定向回归测试；其余失败来自当前隔离 worktree 缺少其他 benchmark 的数据 fixture，与 ALFWorld stepwise 代码无关。

## 尚未执行

- 真实 ALFWorld environment smoke episode；
- Qwen3.5-9B Director Stable Zero canary；
- Direct 与 AgentGraph 配对 evaluation；
- 官方 Success Rate、episode score、invalid/repeated action 和 Wrong Demo 报告；
- 任何训练或 Skill 路径。

因此当前状态是：**架构接线完成、prepare-only 与无模型定向测试通过；真实环境 Stable Zero 和准确率仍为 N/A。**
