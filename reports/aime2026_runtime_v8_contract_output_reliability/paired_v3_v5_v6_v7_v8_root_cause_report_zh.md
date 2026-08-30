# AIME 2026 v3/v5 组合修复与固定 30 题复评报告

## 结论

本轮完成了 v3 搜索条件与 v5 运行时可靠性边界的组合验证，但没有得到
高于 v3 的 Accuracy。当前证据支持的最高 AgentGraph 仍为 v3：
`14/30 = 46.67%`。组合候选中最好的是 v7：`13/30 = 43.33%`，因此
没有晋升为 best-profile。

## 同协议结果

所有条件使用同一 AIME 2026 官方 30 题、固定分母 30、同一 Direct
预测、同一 target-blind integer extraction、同一 evaluator 和冻结模型
目录。未完成或 evaluator-invalid 的题仍保留在严格分母中。

| 条件 | AgentGraph | Accuracy | evaluator-valid | 状态 |
|---|---:|---:|---:|---|
| Direct | 6/30 | 20.00% | 30 | completed |
| v3（当前 best） | 14/30 | 46.67% | 29 | 1 operational failure |
| v5 | 12/30 | 40.00% | 30 | completed |
| v6 | 8/30 | 26.67% | 30 | completed |
| v7 | 13/30 | 43.33% | 29 | 1 terminal failure |
| v8 | 10/30 | 33.33% | 28 | 1 operational + 1 terminal failure |

v7 相对 Direct 提升 `+7/30 = +23.33` 个百分点；相对 v3 下降
`-1/30 = -3.33` 个百分点。v8 相对 Direct 提升 `+4/30 = +13.33`
个百分点；相对 v3 下降 `-4/30 = -13.33` 个百分点。

## 三类目标缺陷的修复结果

### 1. Director contract semantic drift

Canvas 的 validate-before-commit 边界增加 target-blind task-specification
guard。它只阻止 Agent contract 把原题之外的数值、推导结论、假设或
求解方法写成预执行义务，不读取 ground truth，也不规定 Agent 数量、
角色或 topology。

v7 中五个历史 semantic-drift contract 均未进入最终执行 contract。
其中 task04、task11 得到正确答案；task07、task10、task13 仍是完整
artifact 内的数学推理错误。说明原架构缺陷已从这些样本中剥离，但
模型能力问题尚未解决。

v8 又覆盖了类似 task03 的越界候选 contract。该题首轮漂移 edit 被
typed feedback `task_specification_drift` 拒绝且 graph revision 保持不变。

### 2. artifact truncation

`finish_reason=length` 现在强制令 `artifact_complete=false`，即使 provider
metadata 错误声明 `artifact_complete=true`。截断 artifact 不再进入
candidate agreement、`SET_OUTPUT` 或 `FINISH`；只允许同一 model、同一
contract、同一 upstream inbox 的一次有界 continuation。

两道历史截断题均不再以截断尾部整数作为正式答案。task17 在 v7
修复为正确；task29 的 continuation 形成完整但数学错误的 artifact，
因此后者已从 parsing/runtime failure 转化为 reasoning failure。

### 3. operational timeout

内部 Agent execution timeout 为 480 秒，外部 task timeout 为 900 秒，
不再使用相同 600 秒边界竞争。task28 在 v7 形成正式 trajectory 并显式
`FINISH`，消除了 v3 的 collection-timeout 缺口；最终答案 7、参考答案
107，属于数学错误而不是运行时错误。

## 为什么组合后没有超过 v3

修复提高了运行时可诊断性，不等于提高 Director policy 或数学模型能力。
v6-v8 出现明显单节点化：v7 的 30 题中 28 题为单节点图；v8 的 29 条
trajectory 中 28 条为单节点图。v8 的最终节点有 20/30 路由到
`gpt-4o-mini`。v7 与 v8 的 seed、Director prompt、catalog 和 evaluator
一致，但部署 receipt 表明 generation 未开启 deterministic inference，
所以两次 paired 变化同时包含 Director sampling 与 model routing 方差，
不能把分数下降归因于 evaluator 或某一条可靠性修复。

v8 的 task25 已证明 output/runtime 路径正确：artifact 以
`finish_reason=stop` 完成，`\boxed{425}` 被 target-blind extractor 解析；
`SET_OUTPUT` 和 `FINISH` 都复用同一 artifact，没有重新采样。它仍然错误，
因为 425 不是参考答案 850。

## 剩余架构问题

1. **Director observation/context growth**：v8 task03 在 round 0-12 的
   append-only checkpoint 后两次 HTTP 400；prompt 从约 74k 字符增长到
   131k 字符。现有 receipt 没保存 provider response body，因此只能定位
   为 request/context-size 相关的 operational failure，不能进一步断言。
2. **action-domain exhaustion**：v8 task15 已有冲突候选 252/288，但没有
   Output pointer，后续 relation churn 导致
   `canvas_action_domain_exhausted`。需要在不替 Director 选答案的前提下，
   保证 conflict-local repair 与合法 terminal path 不被 action mask 同时
   消除。
3. **single-node topology collapse**：后续需要通过 policy learning 或
   有证据的 search-space 调整学习何时扩图，而不是人工写入固定
   Solver/Verifier/parallel 模板。
4. **partial-checkpoint reporting**：task03 已保存 13 个 turn checkpoint，
   但 paired report 将其记为 trajectory missing。应把未完成 checkpoint
   materialize 为正式 operational-failure trajectory 证据；不能从历史
   candidate 回收答案。

## 验证与训练状态

- 定向回归覆盖：length completion precedence、contract task grounding、
  `SET_OUTPUT`/`FINISH` artifact reuse、v6/v7 配置边界和 AIME output marker。
- 完整单元测试：`1097 passed, 1 warning, 206 subtests passed`。
- 没有进行 backward、optimizer update、LoRA、GRPO、MACE、Bayesian、
  Skill、Tool、检索、Web Search 或答案查询。

## 版本选择

`config/best_profiles/aime2026_agentgraph_best_v1.yaml` 继续指向
`aime2026_runtime_v3_artifact_termination`。v6、v7、v8 均作为已完成、
可复现但未晋升的候选保留。
