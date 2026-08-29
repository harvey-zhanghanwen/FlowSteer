# MBPP+ 初版适配与正式评测报告

## 评测条件

- 数据：EvalPlus MBPP+ v0.2.0，共 378 题。
- 正式任务集合：SkillFlow `mbpp-plus-fixed-100@1`，按 canonical numeric task ID 升序取前 100 题，范围为 `Mbpp/2` 至 `Mbpp/224`。
- 模型可见信息：公开 prompt、task ID、entry point。`canonical_solution`、`base_input`、`plus_input`、expected outputs 和 failed hidden tests 均留在 evaluator boundary 内。
- 终局 evaluator：EvalPlus 0.3.1 `sanitize → get_groundtruth → check_correctness`。
- 主指标：MBPP+ pass@1，由 Plus-test status 决定；辅助指标为 Base pass@1。
- Direct 与 AgentGraph 各生成一个 candidate/task；两者使用相同 100 题与同一 evaluator。
- 本轮没有训练、GRPO、backward、optimizer update、LoRA、MACE、Bayesian update、Skill retrieval 或 Skill evolution。

## Stable Zero

3 题 canary 中，Direct 与 AgentGraph 的 Base pass@1 和 MBPP+ pass@1 均为 3/3；3 条 AgentGraph trajectory 均显式 `FINISH`，没有 terminal、provider collection 或 evaluator failure。Stable Zero 因而确认通过，但该 3 题结果不作为正式准确率。

## 正式结果

| Condition | Valid / total | Base pass@1 | MBPP+ pass@1 | Explicit FINISH | Terminal failure |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B Direct | 100 / 100 | 86 / 100（86%） | 72 / 100（72%） | 不适用 | 0 |
| AgentGraph | 100 / 100 | 81 / 100（81%） | 68 / 100（68%） | 100 / 100 | 0 |

AgentGraph 相对 Direct：Base pass@1 下降 5 个百分点，MBPP+ pass@1 下降 4 个百分点。两种协议的模型调用预算不同，因此该差值是同题描述性对照，不是等计算量下的因果估计。

## AgentGraph、执行与通信

- 最终 topology：`single` 94 题、`serial_2` 5 题、`fan_in` 1 题；reciprocal/bidirectional relation 为 0。
- 最终 Agent 数量：1 Agent 94 题、2 Agents 5 题、4 Agents 1 题，平均 1.08 Agent/task。
- Director 共 540 turns，平均 5.4 turns/task；动作计数为 `add_agent=216`、`modify_agent=115`、`set_relation=9`、`set_output=100`、`finish=100`。
- 只有 6/100 题形成实际上游 artifact communication，共 9 条 artifact messages、8 次接收方 Agent execution。
- 成功执行的 117 次 Agent call 全部使用 `qwen3.5-9b-local / supervisor_theta`。虽然 model catalog 包含远程候选，62 个 trajectory 的远程 Agent request 返回 HTTP 403，随后按 `preserve → diagnose → repair → augment` 修改为本地 Qwen 并完成任务。失败分布：GPT-4o-mini 29、Qwen3.5-Flash 13、DeepSeek-V4-Flash 11、GLM-4.5-Flash 5、MiniMax-M2.5 2、MiniMax-M3 2。
- Canvas 拒绝 161 次 edit，其中未注册 `coding` execution adapter 80 次、未注册 `react` execution adapter 24 次。其余主要是 provider repair action 与当前 action constraints 不一致。最终任务虽均完成，但该不一致造成额外 Director turns，并使最后 graph 大多退化为本地 Qwen `reasoning` mode 的单 Agent。
- 统计到的 model request：Direct 100；AgentGraph 878 次 Director request 加 117 次成功 Agent execution。AgentGraph 输入 1,465,782 tokens、输出 71,967 tokens；Direct 输入 24,844 tokens、输出 12,189 tokens。

## Failure taxonomy

### Direct

- Base 与 Plus 均通过：71。
- Base 通过、Plus 失败：15，属于 robustness / edge-case failure。
- Base 失败且 source 可解析、entry point 正确：13，属于 semantic implementation failure。
- entry-point binding failure：1（`Mbpp/126`）。
- 清洗后 syntax error、empty output、provider failure、evaluator-invalid：均为 0。

### AgentGraph

- Base 与 Plus 均通过：68。
- Base 通过、Plus 失败：13，属于 robustness / edge-case failure。
- Base 与 Plus 均失败：19，其中 1 题为明确 entry-point binding failure，其余候选均可解析并包含所需 entry point，首个正式可观察失败为 Base evaluator。
- syntax error、empty output、terminal failure、evaluator-invalid、collection failure：均为 0。

## 代表案例

### 1. `Mbpp/14`：Agent contract 改变公开函数签名

- 问题：实现三参数 `find_Volume(10, 8, 6) == 240`。
- Direct：实现 `find_Volume(base, height, length)`，Base/Plus 均通过。
- AgentGraph 链路：`add_agent`（多次 rejection）→ `set_output(prism_solver)` → `FINISH`；最终 topology 为 `single`。
- `prism_solver` contract 把输入解释成“三条三角形边加棱柱高度”，输出四参数函数。最终 candidate 与公开三参数调用不一致，Base/Plus 均失败。
- 首个可观察 failure layer：Director 生成的 free-text contract 扩大了原问题 scope；没有下游 Agent communication 或 executable test receipt 纠正该签名。

### 2. `Mbpp/9`：semantic implementation failure

- 问题：求字符串恢复自身所需的最小正旋转次数。
- 链路：`add_agent(miniwriter) → set_output(miniwriter) → FINISH`；topology 为 `single`。
- Candidate 只检查 `1..n-1`，没有较小周期时返回 `0`；但旋转完整字符串长度 `n` 仍是满足要求的正旋转。
- Base/Plus 均失败。首个可观察 failure layer 是 Agent 对任务语义的实现错误；没有独立验证或实际 code execution。

### 3. `Mbpp/137`：`fan_in` 通信有效，但没有可执行验证

- 问题：计算整数数组中零元素与非零元素的比值。
- 最终 topology：4 Agents、3 relations、`fan_in`，structural depth 3；Output Agent 为 `validator`。
- `validator` 的 inbox 确实收到 `executor` artifact，说明 artifact routing 成功；但所有 Agents 均为 `reasoning` mode 且 `allowed_tools=[]`，`validator` 基本原样返回上游代码，没有 test execution receipt。
- Base 通过、Plus 失败。首个运行异常是远程 model request HTTP 403，随后恢复到本地 Qwen；首个正式结果失败层为 Plus evaluator。
- 该案例说明复杂 topology 本身没有带来额外验证证据，不是 Agent communication 丢失。

### 4. `Mbpp/223`：`serial_2` 改善 Plus 结果

- 问题：判断 sorted array 中指定元素是否为 majority element。
- Direct：Base 通过、Plus 失败。
- AgentGraph：`agent-0 → agent-1`，`agent-1` 收到上游 Python artifact 后输出基于左右边界 binary search 的实现；Base/Plus 均通过。
- 该题是 9 个 AgentGraph 相对 Direct 改善 MBPP+ pass@1 的任务之一，但 100 题总体仍低于 Direct。

## 结论与已知缺口

Dataset loader、公开/私有边界、AgentGraph Canvas、execution feedback、trajectory、终局 Python source artifact 与官方 EvalPlus evaluator 已完整接通，Stable Zero 和 100 题正式 evaluation 均可运行。因此当前不再是“没有准确率”或“链路没通”的状态。

初版 AgentGraph 未超过 Direct。公开 evidence 指向三个通用工程缺口：Director search space 暴露了当前 MBPP condition 未注册的 `coding`/`react` execution modes；远程 model provider 全部返回 HTTP 403；空 Tool surface 使验证型 contract 不能产生真实 code-execution receipt。上述问题应在新的 development condition 中修复并验证，不能根据 fixed-100 hidden-test outcome 为具体题目加入 hard-coded workflow。

## 证据

- `reports/mbppplus_initial_v1/evaluation_report.json`
- `reports/mbppplus_initial_v1/evaluation_report.md`
- `artifacts/mbppplus_initial_v1/evaluation/run_manifest.json`
- `artifacts/mbppplus_initial_v1/evaluation/preflight_receipt.json`
- `artifacts/mbppplus_initial_v1/evaluation/direct_predictions.jsonl`
- `artifacts/mbppplus_initial_v1/evaluation/agentgraph_trajectories.jsonl`
- `artifacts/mbppplus_initial_v1/evaluation/paired_results.jsonl`
- `artifacts/mbppplus_initial_v1/evaluation/wrong_demos.jsonl`
