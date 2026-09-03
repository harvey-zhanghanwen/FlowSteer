# WebShop v41/v42 架构修复与小样本验证报告

## 结论

本轮确认并修复了一个 stateful AgentGraph 的 artifact freshness 缺陷：
Canvas edit 新增 fan-in 分支并使 environment owner 重新执行时，旧的
directed ancestors 仍可能复用上一 environment revision 的缓存 artifact。
修复后，只要 edit 已经影响 environment owner，就使用现有
`dirty_closure` 重新执行 owner 的全部有向祖先；与 owner 无关的独立
Agent 仍可复用缓存，避免额外环境动作和模型调用。

非链式 topology 的创建、路由和执行并未被架构阻塞。v40 的
`webshop:00525` 已实际形成 6-Agent `mixed` topology，包含 parallel、
fan-in 和 fan-out。其失败原因是 fan-in 中混入旧 revision artifact，
而不是 graph executor 不能运行复杂 topology。

## 真实指标

所有分数均来自原始 WebShop evaluator；Average Score 以下按百分制展示。
Direct 记录复用同一 validation task 的既有结果，没有重复调用模型。

| Condition | Task | AgentGraph topology | AgentGraph Average Score | AgentGraph Success Rate | Direct Average Score | Direct Success Rate | Stable Zero |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| v40（修复前诊断） | `webshop:00525` | 6-Agent mixed | 0.00 | 0.00% | 0.00 | 0.00% | 未通过：environment terminal receipt invalid |
| v41 | `webshop:00525` | 2-Agent serial | 50.00 | 0.00% | 0.00 | 0.00% | 通过 |
| v42 | `webshop:00558` | single | 25.00 | 0.00% | 0.00 | 0.00% | 通过 |
| v42 | `webshop:00625` | 2-Agent serial | 50.00 | 0.00% | 50.00 | 0.00% | 通过 |

三个 post-fix canary 的 AgentGraph Average Score 为 **41.67/100**，
Success Rate 为 **0/3 = 0.00%**；对应 Direct Average Score 为
**16.67/100**，Success Rate 为 **0/3 = 0.00%**。这是定向 canary，不能
替代 128-task 正式 validation 指标。

当前同协议、完整 128-task validation 的最高已验证版本仍为 v37：

- AgentGraph Average Score：**62.8242/100**；
- AgentGraph Success Rate：**46/128 = 35.9375%**；
- Direct Average Score：**32.6940/100**；
- Direct Success Rate：**18/128 = 14.0625%**；
- terminal failure：**0**。

因此未把 v41/v42 小样本配置替换为全量 best profile，也未启动更大
评测。用户所指的 `webshop:00525` 已完成重跑；若“525”指 525 条样本，
当前 0/3 canary Success Rate 不满足扩大运行的证据门槛。

## 三个 canary 的执行结果

### `webshop:00525`

目标：xx-small、polyester hoodie、价格低于 50 美元。

执行动作：

`search[xx-small polyester hoodie]`
→ `click[b09jzcz5nw]`
→ `click[features]`
→ `click[< prev]`
→ `click[description]`
→ `click[< prev]`
→ `click[xx-small]`
→ `click[buy now]`

环境正式终止，Average Score 为 50/100。公开证据支持 polyester、
xx-small 和价格条件，但官方 human goal 还包含没有在公开句子中明确
表达的 attribute（例如 `machine wash`）；该 evaluator-private 信息未
泄露给 Director 或 Agent。

### `webshop:00558`

目标：gift set、8 ounce cinnamon dip、价格低于 50 美元。

执行动作：

`search[gift set eight ounce bottle cinnamon dip]`
→ `click[b09h69jk1y]`
→ `click[buy now]`

Average Score 为 25/100。首个可观察失败层是 evidence acquisition：
购买前没有打开 Description/Features，也没有确认 cinnamon 和 8 ounce
绑定。Director 选择 single-Agent，属于 policy selection；Runtime 没有
拒绝 multi-Agent topology。

### `webshop:00625`

目标：兼容 Apple Watch 的 size 42 white smartwatch band，价格低于
20 美元。

执行动作：

`search[size 42 white smartwatch band apple watch under 20]`
→ `click[b09ktfgsyj]`
→ `click[white | yellow | red]`
→ `click[42 | 44 | 45mm]`
→ `click[buy now]`

Average Score 为 50/100。size `42` 到可见 `42mm` 的 option binding 已
成功，主要缺口是对 `compatible apple` 的 candidate evidence 未验证；
同时复合 color option 不能等同于唯一 white variant。

## 剩余问题分类

1. **Policy selection**：post-fix 三题中 Director 选择 single/serial
   topology；这不是 action mask 或 Runtime 禁止 non-serial topology。
2. **Retrieval ranking**：首个搜索结果不一定是与 evaluator target
   product type/attributes 最匹配的商品。
3. **Evidence acquisition**：Agent 可能在 Description/Features 未核验
   前购买，尤其是自然语言 attribute 未进入 SkillFlow 固定公开短语
   projection 时。
4. **Public semantics vs evaluator-private labels**：官方 human goal 可能
   含公开 instruction 未明确表达或措辞顺序不同的 attribute。不能把
   hidden goal、reward breakdown 或 target ASIN 注入编排输入。
5. **Training/Skill**：本轮没有训练、LoRA、GRPO、MACE、Bayesian update
   或 Skill injection；未训练 Director 对复杂 topology 的选择偏好没有
   因本轮架构修复而改变。

## 是否继续全量评测

不继续。v41/v42 已证明 Stable Zero 和终局 evaluator 闭环正常，但三题
Success Rate 均为 0%。在没有训练或来源明确的通用 evidence parser 前，
立即扩大到 128 或 525 条只会消耗推理额度，不能验证新修复带来稳定的
Success Rate 增益。
