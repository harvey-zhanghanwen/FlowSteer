# SWE-bench 初版适配 v7：真实 smoke evaluation 报告

## 结论

SWE-bench 的 Dataset、task-scoped repository workspace、SkillFlow repository Tools、
workspace patch publication、official SWE-bench harness 与 receipt-based reporting 已接入
统一 AgentGraph。固定 test panel 前 2 个 canary task 已完成 Direct vs AgentGraph 的同题
evaluation，两个条件都有 2/2 official-evaluator-valid receipts；真实结果均为
`Resolved 0/2，Resolved Rate 0.00%`。因此 evaluator 记账阻塞已经解除，但 Stable Zero
未通过。当前失败来自没有生成有效 patch，而不是 evaluator coverage 不足。

这只是 2-task canary，用于验证端到端协议；不能称为 128-task 正式准确率，也不能外推为
SWE-bench Verified benchmark 水平。

## 协议与实现边界

- 数据：SkillFlow v3 `test_iid_v3.json` 的 128 个 `code_generation` task，按 source
  order 冻结；本轮只执行前 2 个 canary。
- Repository runtime：每个 task、每个 paired arm 使用相同 base commit 的独立 detached
  worktree；同一个 AgentGraph 内的 Agents 共享该 task workspace。
- Director：仍只有统一 Canvas actions；没有增加 repository Tool action，也没有预设
  `Coder -> Reviewer -> Tester` topology。
- Agent：`execution_mode=coding` 时调用 SkillFlow deployed action vocabulary：`bash /
  list_files / search_code / view_file / str_replace_editor / run_tests`。
- Completion：从 worktree 物化真实 `git diff`；Output Agent prose 不得替代 patch。
- Evaluator：只使用 SkillFlow `evaluate_patch` 到 official SWE-bench harness 的
  `Resolved / Failed` receipt；没有 LLM judge，也没有解析输出中的 `PASSED`。
- 训练：未启用 GRPO、MACE、Bayesian、Skill injection/evolution、backward、optimizer
  update 或 LoRA publication。

## 真实指标

| Condition | Tasks | Evaluator valid | Resolved | Resolved Rate | FINISH | max_rounds |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B Direct | 2 | 2 | 0 | 0.00% | N/A | N/A |
| AgentGraph | 2 | 2 | 0 | 0.00% | 0 | 2 |

AgentGraph - Direct 为 `+0.00` percentage points。两臂的 task、repository snapshot、Tool
surface 与 task-global repository episode budget 相同；AgentGraph 额外包含 Director 和
多 Agent inference，所以这里只作 descriptive paired comparison。

## Repository Tool 与 AgentGraph 运行

| Condition | search | view | edit | test | command | invalid structured action | Tool budget exhausted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 0 | 1 | 0 | 0 | 19 | 2 | 2 |
| AgentGraph | 2 | 2 | 1 | 0 | 22 | 6 | 2 |

AgentGraph 的 1 次 `str_replace_editor` 返回失败，最终没有 workspace change；两题都没有
运行 `run_tests`。AgentGraph 的最终 topology 均为自然产生的 parallel topology：

- `astropy__astropy-14182`：8 Agents、4 个 reciprocal relations、structural depth 1；
- `django__django-13417`：3 Agents、1 个 directed relation、structural depth 2。

这说明自由 AgentGraph search space 确实被使用，但本轮 policy 把 rounds 花在持续扩展和
修改图上，没有收敛到可提交 patch 与 explicit `FINISH`。

## Failure taxonomy

AgentGraph wrong-task denominator 为 2。按互斥 primary observable category 统计：

| Category | Count | Share |
|---|---:|---:|
| terminal / budget failure | 2 | 100.00% |
| collection / provider / repository environment failure | 0 | 0.00% |
| Agent communication / repository Tool / local validation failure | 0 | 0.00% |
| patch apply / official target test / regression failure | 0 | 0.00% |
| evaluator runtime failure | 0 | 0.00% |

中间 receipt 仍记录到 1 次 recoverable provider failure、6 次 invalid structured actions、
1 次 failed edit，但它们没有被伪装成 task-level root cause。没有 intervention evidence 时，
`first observable failure` 不等于 causal failure。

## Wrong Demo 1：Astropy RST header rows

- Task ID：`swe-bench:astropy__astropy-14182`
- Issue：`ascii.rst` writer 不接受 `header_rows=["name", "unit"]`。
- Repository state：`astropy/astropy@a5917978be39d13cd90b517e1de4e7a539ffaa48`。
- Direct：耗尽 12 个 coding turns；没有 patch；official evaluator receipt 为
  `empty_patch -> resolved=false`。
- AgentGraph：第 0 个 Canvas turn 的首个可观察 failure 是 Director 产生了 endpoints
  相同的 `add_subgraph` relation，Canvas 拒绝该 action。随后 Director 最终构造 8 Agents
  与 4 个 reciprocal pairs，但 20 rounds 后仍未产生 patch，也未 `FINISH`。
- 最终结果：`empty_patch -> Resolved 0`。
- 错误传播：最早 receipt 是 Director action parsing/validation failure；后续持续图编辑、
  coding execution budget exhaustion 与 failed edit 共同出现。当前证据不足以把首个 receipt
  认定为唯一 root cause。

## Wrong Demo 2：Django QuerySet.ordered

- Task ID：`swe-bench:django__django-13417`
- Issue：带 `Meta.ordering` 的 model 在 GROUP BY query 中，`QuerySet.ordered` 与实际 SQL
  是否包含 `ORDER BY` 不一致。
- Direct：耗尽 12 个 coding turns；没有 patch；official evaluator receipt 为
  `empty_patch -> resolved=false`。
- AgentGraph：第 0 个 `add_subgraph` 被 Canvas validation 拒绝；后续最终图为 3 Agents、
  1 个 directed relation、structural depth 2。共有 7 个 accepted execution turns / 15 个
  executor calls，但 20 Canvas rounds 后仍没有 patch 和 explicit `FINISH`。
- 最终结果：`empty_patch -> Resolved 0`。
- 错误传播：最早可观察 layer 是 Canvas action validation；终局 primary category 是
  terminal / budget failure。没有运行 local tests，因此不存在 official target-test 或
  regression failure receipt。

## Stable Zero 判定与后续边界

Stable Zero 未通过。当前已不再被 repository environment、Docker transport 或 evaluator
coverage 阻塞；下一阶段若继续，应优先处理 policy 在 bounded Canvas 中的收敛、有效
repository Tool plan、edit/test/patch publication 和 explicit FINISH，而不是修改 official
evaluator 或加入固定代码 Agent workflow。按照本轮范围，本报告之后不启动训练或 Skill
evolution，也不自动运行剩余 126 个 task。
