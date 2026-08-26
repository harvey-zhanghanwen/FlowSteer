# WebShop v5 Failure Taxonomy 与可复现 Wrong Demos

## 1. 统计口径

- Condition：`webshop_initial_adapter_v5`。
- Split：固定 WebShop validation，`webshop:00500`–`webshop:00627`，共 128 个 episode。
- 正式指标：WebShop native `Average Score` 与 `Success Rate`；不使用 EM、F1、Accuracy 或 LLM judge。
- Full success：native reward `== 1.0`。Wrong Demo：native reward `< 1.0`。
- 128/128 AgentGraph episode evaluator-valid、128/128 explicit `FINISH`；Wrong Demo 共 110 个。
- “主要结果类别”互斥，110 个 Wrong Demo 每个只计入一次。“机制性标签”允许交叉，用于描述同一 episode 中出现但可能已经恢复的错误。
- “首个因果失败点”只依据可观察的 Director/Canvas/Agent/Tool/evaluator receipt；不从模型隐状态推断原因。已恢复的 provider、Canvas 或 capability 事件不自动认定为最终低分原因。
- WebShop 没有文本 reference answer；数据记录中的 reference target 是 `environment_success`，即购买满足 instruction 全部约束的商品并得到 native reward `1.0`。下文在存在 Direct full-success trajectory 时同时给出该 action chain 作为可复现参照，但它不是 hard-coded ground truth。

## 2. 互斥主要结果类别

| 类别 | 判定规则 | 数量 | 占全部 128 | 占 110 个 Wrong Demo |
|---|---|---:|---:|---:|
| Full success（非错误） | terminal reward `== 1.0` | 18 | 14.06% | — |
| Purchase constraint mismatch / insufficient constraint verification | terminal purchase，`0 < reward < 1` | 91 | 71.09% | 82.73% |
| Native-action parsing loop → truncation | `environment_step_limit` 且 episode 含 `<INVALID>` | 15 | 11.72% | 13.64% |
| Search/navigation budget exhaustion | `environment_step_limit` 且 10 个 action 全部合法 | 4 | 3.13% | 3.64% |
| 合计 | — | 128 | 100.00% | 100.00% |

这三个错误类别完整覆盖 110 个 Wrong Demo，不重复计数。

## 3. 可交叉的机制性标签

| 机制或运行事件 | 全部 128 | 110 个 Wrong Demo | 与正式失败的关系 |
|---|---:|---:|---|
| 至少一次 invalid native action | 28（21.88%） | 25（22.73%） | 共 153/610 次 action invalid（25.08%）；3 个 episode 仍 full success |
| 三步完成的 partial-credit purchase | 72（56.25%） | 72（65.45%） | 占 91 个 partial purchase 的 79.12%；严格链路为 `search → click[ASIN] → buy now` |
| Partial purchase 未访问 `description/features/reviews` | 89（69.53%） | 89（80.91%） | 89/91 没有 product-detail verification |
| Option-bearing page 未执行 option click | 49（38.28%） | 49（44.55%） | Product page 明示 color/size 控件，但 purchase receipt 没有 option binding |
| 执行过 option click 但约束仍不完整 | 4（3.13%） | 4（3.64%） | 说明局部 option selection 不等于完整 constraint satisfaction |
| 已查看 detail evidence 但未绑定到最终购买 | 2（1.56%） | 2（1.82%） | 已观察 evidence 没有转化为最终 ASIN/option action |
| Environment 10-step truncation | 19（14.84%） | 19（17.27%） | 15 个伴随 invalid action，4 个为合法检索/导航耗尽 |
| Provider HTTP 429 | 6（4.69%） | 6（5.45%） | 全部切换到 `qwen3.5-9b-local` 后恢复；0 个 provider-caused invalid trajectory |
| Director action JSON parsing failure | 3（2.34%） | 2（1.82%） | 下一轮均成功解析并继续；另一个发生在 full-success episode |
| 其他 Canvas edit rejection | 8 episodes / 13 turns | 7 episodes / 12 turns | 均继续并 explicit `FINISH`；0 个 rejection-caused terminal failure |
| Stateful Tool capability staging feedback | 128（100.00%） | 110（100.00%） | `execution_mode=react` 与挂载 `webshop.environment` 是两个 atomic edits；下一 edit 均修复，不能当作任务失败 |

Invalid action 是交叉标签而不是单独覆盖全部 28 个 episode 的互斥结果类别：其中 15 个最终 truncation、10 个最终 partial credit、3 个最终 full success。153 次 invalid action 全部有 native action-parser failure receipt；其中 136 次 raw output 为空，17 次 raw output 非空但 grammar 错误。110 个 Wrong Demo 内为 25 个 episode、145 次 invalid action。

## 4. 专业 failure layer 核对

| Failure layer | 确认的因果失败数 | 说明 |
|---|---:|---|
| Retrieval / Tool action | 19 个 truncation；另有 91 个 terminal selection mismatch | 15 个 parsing loop、4 个合法 search/navigation budget exhaustion；partial purchase 的首个可观察错误通常是错误 product/option action |
| Orchestration / relation | 0 | 11 个 episode 出现 Director JSON parsing 或 Canvas edit rejection，均恢复；128 个 final graph 均为 single-agent、0 relation，不能从本轮建立 topology 对性能的因果结论 |
| Agent communication | N/A；适用分母为 0 条有 relation 的 trajectory | 所有 final graph 均无 relation，所有代表 demo 的 `upstream=[]`；没有发生 inter-agent message routing，因此不能写成已测试过的 `0/128`，也不虚构 communication failure demo |
| Reasoning | 不单独计数 | WebShop receipt 只观察 action policy；错误 product action 可证实，但不能把不可见的内部原因强行标为 reasoning failure |
| Constraint verification / option selection | 91 | terminal reward 位于 `(0,1)`，表明已购买但没有满足完整约束；72 个仅三步完成 |
| Output formatting | 0 / not applicable | WebShop 由 environment state 与 native reward 评分，不按文本答案格式评分；native action serialization failure 已归入 Tool action parsing |
| Terminal / max_rounds | Environment step-limit 19；Director `max_rounds` 0；AgentGraph terminal failure 0 | 19 个为有界 environment truncation，均仍有合法 Canvas `FINISH` receipt |
| Evaluator / canonicalization | 0 | 128/128 native evaluator valid；没有 LLM judge 或文本 canonicalization |
| Provider | 0 个最终因果失败；6 个恢复事件 | 6 个 HTTP 429 全部恢复，不能把其后 partial score 归因给 provider |
| Collection | 0 | `collection_failures.jsonl` 为 0 条 |
| Director parsing | 0 个最终因果失败；3 个恢复事件 | malformed Director JSON 后均重试成功 |
| Environment timeout | 0 | 没有 timeout episode |

## 5. Wrong Demo 1：native action serialization/parsing loop

### Task 与目标

- Task ID：`webshop:00581`。
- Instruction：`i need a-5 pairs of false eyelashes. it should be easy to apply, and price lower than 40.00 dollars`。
- Reference target：`environment_success`；购买 5 pairs、easy to apply、低于 `$40` 的 false eyelashes，native reward `1.0`。

### Director / Canvas

1. `ADD_AGENT researcher(gpt-4o-mini)`；contract：研究 5 pairs、easy application、price `<40`。
2. `MODIFY_AGENT researcher.execution_mode=react`；Canvas 返回缺少 stateful Tool capability。
3. `MODIFY_AGENT researcher.allowed_tools=[webshop.environment]`；Canvas 接受并立即执行一个 request-scoped episode。
4. `SET_OUTPUT researcher`；复用现有 artifact。
5. `FINISH`；Canvas 返回 `workflow finished`。

### Agent input / output / communication

- Agent input：原始 instruction、free-text contract、当前 public WebShop observation 与动态 admissible action；初始页面只允许 `search[...]`。
- Inter-agent communication：无；final graph `relations=[]`，Agent request `upstream=[]`。
- Agent raw output：step 0–6 均为 `<action>search>false eyelashes easy to apply under 40.00</action>`；step 7–9 改为 `<action>search>false eyelashes easy to apply price under 40.00</action>`。
- Output Agent 最终 artifact：仍是 WebShop Search 首页，没有环境状态推进。

### ReAct Tool Action–Observation receipt

| Step | Raw action output | Parsed action | Observation / feedback |
|---:|---|---|---|
| 0 | `<action>search>false eyelashes easy to apply under 40.00</action>` | `<INVALID>` | `[INVALID] No valid <action> tag found.`；状态不变 |
| 1–6 | 与 step 0 相同 | `<INVALID>` × 6 | 同一 format repair feedback；状态不变 |
| 7–9 | `<action>search>false eyelashes easy to apply price under 40.00</action>` | `<INVALID>` × 3 | 同一 feedback；状态不变 |

Native grammar 要求 `search[query]`；该输出把 `[` 写成了 `>`。

### Terminal / evaluator receipt 与错误传播

- Environment attempts：10；state-advancing actions：0；invalid actions：10。
- `environment_terminal=false`，`environment_truncated=true`，evaluator reason=`environment_step_limit`。
- AgentGraph explicit `FINISH=true`、evaluator valid=`true`。
- Final output：未变化的 Search 页面。
- Average Score=`0.0`，Success Rate=`0.0`。
- 首个因果失败点：environment step 0，Agent `researcher` 的 native action serialization failure。
- 传播：parser 拒绝 action → environment state 不推进 → repair observation 未改变输出模式 → 10 次 budget 全部消耗 → truncation → reward 0。

## 6. Wrong Demo 2：合法 search/navigation action 耗尽预算

### Task 与目标

- Task ID：`webshop:00516`。
- Instruction：购买 beauty-salon folding mattress，`wine red`、`75*190cm`、price `<130`。
- Reference target：`environment_success`。
- 同题 Direct full-success 参照：`search[folding mattress wine red 75*190cm] → click[b09bzmbwnf] → click[wine red] → click[75*190cm] → click[buy now]`，reward `1.0`。

### Director / Canvas 与 Agent

- Director：`ADD_AGENT webshop_user(glm-4.5-flash) → MODIFY_AGENT execution_mode=react → MODIFY_AGENT allowed_tools=[webshop.environment] → SET_OUTPUT webshop_user → FINISH`。
- Contract：`Find a wine red folding mattress, 75x190cm, under $130.`
- Agent input：原始 instruction、contract、每一步 public page 与动态 admissible actions。
- Inter-agent communication：无，`relations=[]`、`upstream=[]`。
- Agent final output：第 3 页 search results 页面，没有购买确认。

### ReAct Tool Action–Observation receipt

`search[folding mattress wine red 75x190cm beauty salon]`

`→ next → next → next`

`→ click[b08zxgz9zt]`

`→ back to search → click[search]`

`→ search[folding mattress wine red 75x190cm beauty salon under 130]`

`→ next → next`

关键 observation：第一次 search 的 Page 1 已出现 `B09BZMBWNF`，标题明确为 beauty-salon folding mattress、`75×190cm`；Direct 随后在该 product page 选择 `wine red` 并 full success。AgentGraph 在 step 1 没有检查该高相关候选而直接翻页；之后点击的 `B08ZXGZ9ZT` 页面又显示 `Stars, 80×190cm`，与目标不匹配。

### Terminal / evaluator receipt 与错误传播

- 10/10 action 合法并推进状态；invalid action=0；没有 `buy now`。
- `environment_terminal=false`、`environment_truncated=true`、reason=`environment_step_limit`。
- AgentGraph explicit `FINISH=true`、evaluator valid=`true`。
- Average Score=`0.0`，Success Rate=`0.0`。
- 首个可证实、对结果有直接影响的失败点：environment step 1 的 `click[next >]`，没有检查当前 observation 中已经出现的高相关候选 `B09BZMBWNF`。
- 传播：连续翻页 → step 4 检查不匹配候选 → back/reset → 第二次 search 和翻页 → 未在 step 10 前 purchase → truncation → reward 0。

## 7. Wrong Demo 3：product selection 与 option omission

### Task 与目标

- Task ID：`webshop:00501`。
- Instruction：iPhone 13 Pro Max 6.7-inch heavy-duty/dust-proof tempered-glass case，size=`case+4 protectors`、color=`redblack`、price `<50`。
- Reference target：`environment_success`。
- Direct full-success 参照：`search[...] → click[b09p572dp9] → click[case+4 protectors] → click[redblack] → click[buy now]`，reward `1.0`。

### Director / Canvas

1. `ADD_AGENT executive(MiniMax-M2.5)`；provider HTTP 429。
2. `MODIFY_AGENT execution_mode=react` 被 action mask 拒绝：provider repair 必须先只修改 `model_id`。
3. `MODIFY_AGENT model_id=deepseek-v4-flash` 被拒绝：该 provider 已不可用，live domain 只允许本地模型。
4. `MODIFY_AGENT model_id=qwen3.5-9b-local`，接受并恢复执行。
5. `MODIFY_AGENT execution_mode=react`，Canvas 返回缺少 Tool capability。
6. `MODIFY_AGENT allowed_tools=[webshop.environment]`，接受并执行 episode。
7. `SET_OUTPUT executive`。
8. `FINISH`。

这里的 HTTP 429 与两个 rejected edits 都被恢复；它们不是最终 partial reward 的因果失败点。

### Agent input / output / communication

- Agent：`executive`，最终模型 `qwen3.5-9b-local`。
- Input：完整原始约束、free-text contract、动态 search results/product-page actions。
- Communication：无，`relations=[]`、`upstream=[]`。
- Final output：`Thank you for shopping with us!`

### ReAct Tool Action–Observation receipt

1. `search[heavy duty dust proof tempered glass for iphone 13 pro max 6.7 inch case+4 protectors redblack]`。
2. `click[b09m9lzs2m]`。Product observation 显示 Metal Case、color 只有 `black/camo/red`，size 只有 iPhone model variants；没有 `redblack` 与 `case+4 protectors`。同一 search results 中存在 Direct 使用的 `B09P572DP9`。
3. 未选择任何 size/color，直接 `click[buy now]`。

### Terminal / evaluator receipt 与错误传播

- Environment actions=3，state-advancing=3，invalid=0，terminal purchase=true。
- Evaluator reason=`evaluated`，native reward/Average Score=`0.5`，Success Rate=`0.0`。
- AgentGraph explicit `FINISH=true`、evaluator valid=`true`。
- 首个因果失败点：environment step 1 的 product selection，选择了无法满足目标 option 的 ASIN。
- 传播：错误 ASIN → product page 明示 option 不匹配 → 未执行 option selection/verification → `buy now` → partial reward 0.5。

## 8. Wrong Demo 4：product-category mismatch 与 premature purchase

### Task 与目标

- Task ID：`webshop:00620`。
- Instruction：anti-slip water shoes，size `7.5`、color `khaki`、price `<40`。
- Reference target：`environment_success`。
- Direct full-success 参照：`search[...] → click[b07p9zlx4q] → click[nets-khaki] → click[7.5-8.5 women | 6.5-7.5 men] → click[buy now]`，reward `1.0`。

### Director / Canvas 与 Agent

- Director：`ADD_AGENT shopping_agent(gpt-4o-mini) → MODIFY_AGENT execution_mode=react → MODIFY_AGENT allowed_tools=[webshop.environment] → SET_OUTPUT shopping_agent → FINISH`。
- Contract：购买满足 water shoes、size、color、price 的商品。
- Input：原始 instruction、contract、dynamic admissible actions。
- Communication：无，`relations=[]`、`upstream=[]`。
- Final output：`Thank you for shopping with us!`

### ReAct Tool Action–Observation receipt

1. `search[anti-slip water shoes size 7.5 khaki price<40.00]`。
2. `click[b09t3hrk9l]`；product observation 明示商品为 `Slip On Loafers ... Mary Jane Shoes ... Ballet Flats`，价格 `$26.99`，并列出 `khaki` 和 `7.5` options。
3. Agent 没有选择 `khaki` 或 `7.5`，直接 `click[buy now]`。

### Terminal / evaluator receipt 与错误传播

- Environment actions=3，state-advancing=3，invalid=0，terminal purchase=true。
- Native reward/Average Score=`0.05`，Success Rate=`0.0`，evaluator valid=`true`。
- 首个因果失败点：step 1 的 product-category mismatch；页面明确是 loafers 而非 water shoes。
- 传播：错误 product category → option omission → premature `buy now` → terminal partial reward 0.05。

## 9. Wrong Demo 5：已取得正确 evidence，但遗漏 option binding

### Task、AgentGraph 与目标

- Task ID：`webshop:00616`。
- Instruction：适配 60-inch TV、huge storage、Ashland Pine、price `<320` 的 TV stand。
- Reference target：`environment_success`；Direct 在本题 10 步内未 terminal，score 0。
- Director：`ADD_AGENT shop_agent(qwen3.5-9b-local) → MODIFY_AGENT execution_mode=react → MODIFY_AGENT allowed_tools=[webshop.environment] → SET_OUTPUT shop_agent → FINISH`。
- Agent input：原始 instruction、free-text contract、public observation 与 admissible actions；communication=`relations=[]`、`upstream=[]`。
- Agent final output：`Thank you for shopping with us!`

### ReAct Tool Action–Observation receipt

`search[60 inch tv stand ashland pine color huge storage space price lower than 320.00]`

`→ click[b073hd55fj] → click[features] → back to search`

`→ 重复 search → click[b073hd55fj] → click[buy now]`

Product observation 明示 color options 包含 `ashland pine`；`features` observation 明示支持 up to 60-inch、additional storage、Ashland Pine finish。Agent 已检索到正确 evidence，却没有执行 `click[ashland pine]`，purchase receipt 中 `options={}`。

### Terminal / evaluator receipt 与错误传播

- 7 个 action 全部合法，terminal purchase=true，explicit `FINISH=true`，evaluator valid。
- Average Score=`0.0667`，Success Rate=`0.0`。
- 首个因果失败点：最终 product page 上没有把已确认的 Ashland Pine evidence 转化为 option-binding action。
- 传播：正确 evidence 已存在 → back/search 重复执行 → 再次进入同一商品 → 跳过 option click → purchase receipt 丢失目标 variant → partial reward。

## 10. Wrong Demo 6：evidence 没有绑定到最终 purchased ASIN

### Task、AgentGraph 与目标

- Task ID：`webshop:00571`。
- Instruction：complete、easy-to-carry orthodontic storage case，price `<40`。
- Reference target：`environment_success`；Direct score=`0.3333`，AgentGraph score=`0.6667`，两者均未 full success。
- Director 在 provider recovery 后形成：single Agent `researcher(qwen3.5-9b-local)`、`execution_mode=react`、`allowed_tools=[webshop.environment]`、0 relation、`SET_OUTPUT → FINISH`。
- Contract：`Identify correct product, then checkout and complete purchase with proper tool calls`。
- Agent input：原题、contract、public observation/admissible actions；`upstream=[]`，无 Agent communication。
- Final output：`Thank you for shopping with us!`

### ReAct Tool Action–Observation receipt

`search → click[b09lmjb75p] → click[features]`

`→ back to search → 第二次 search → click[b09c31l8tl] → click[buy now]`

`B09LMJB75P` 的 features 明示 lightweight、small size、easy carrying and storage、enough capacity。Agent 随后放弃该已验证 ASIN，购买另一个没有查看 `features` 的 `B09C31L8TL`。

### Terminal / evaluator receipt 与错误传播

- 7 个合法 action，terminal purchase=true，explicit `FINISH=true`，evaluator valid。
- Average Score=`0.6667`，Success Rate=`0.0`。
- 首个因果失败点：step 5 切换到未经 detail verification 的新 ASIN，导致先前 evidence 与最终 purchase entity 失去绑定。
- 传播：正确 evidence 绑定 `B09LMJB75P` → 返回 search 后选择 `B09C31L8TL` → 没有重新验证完整性/easy-carrying → purchase → partial reward。

## 11. Wrong Demo 7：局部 option selection 但完整约束未验证

### Task、AgentGraph 与目标

- Task ID：`webshop:00503`。
- Instruction：purple、machine washable、52-inch width、price `<60` 的 drapes。
- Reference target：`environment_success`；Direct score 0，AgentGraph score 0.75，均未 full success。
- Director 经 provider recovery 后形成：`shop-agent(qwen3.5-9b-local)`、`execution_mode=react`、`allowed_tools=[webshop.environment]`、0 relation、`SET_OUTPUT → FINISH`。
- Agent input：原题、free-text contract、public observation/admissible actions；`upstream=[]`，无 Agent communication。
- Final output：`Thank you for shopping with us!`

### ReAct Tool Action–Observation receipt

`search[purple machine washable drapes 52 inch width]`

`→ click[b07t53mdnr] → click[purple] → click[buy now]`

Product title 明示 `52W×63L`，且 color option 包含 `purple`。Agent 正确执行了 `click[purple]`，但没有访问 `description/features` 来核验 `machine washable`，随即购买。

### Terminal / evaluator receipt 与错误传播

- 4 个合法 action，terminal purchase=true，explicit `FINISH=true`，evaluator valid。
- Average Score=`0.75`，Success Rate=`0.0`。
- 首个因果失败点：完成 color binding 后直接购买，没有验证仍未被 evidence 支持的 `machine washable` constraint。
- 传播：局部 option 正确 → attribute verification 缺失 → terminal purchase → partial reward 0.75。

## 12. Recovered control-plane receipts（不计作最终因果失败）

### Provider recovery：`webshop:00501`

`MiniMax-M2.5` HTTP 429 后，Canvas live action domain 强制只修改 `model_id`，并拒绝同 provider 的替代模型；Director 最终切换到 `qwen3.5-9b-local`，trajectory 完整 `FINISH` 且 evaluator-valid。最终 reward 0.5 的直接证据来自后续错误 product/option action，不来自 provider failure。

### Director parsing recovery：`webshop:00542`

- Round 0：Director JSON malformed，Canvas feedback=`invalid action: ... Unterminated string ...`。
- Round 1：成功 `ADD_AGENT searcher(deepseek-v4-flash)`，之后完成 capability edits、`SET_OUTPUT` 与 `FINISH`。
- 该 episode 最终 reward 0 的直接原因是后续 10 个 environment attempts 中 7 个 invalid native actions并导致 truncation，不是已经恢复的 round-0 Director parsing failure。

## 13. 后续报告统一格式

后续每个数据集沿用以下结构，不把 WebShop taxonomy 生搬硬套到静态 QA、代码修复或 embodied environment：

1. 数据集原生 evaluator、正式分母与 Wrong Demo 判定。
2. 互斥主要失败类别：数量、占全部样本比例、占 Wrong Demo 比例。
3. 可交叉机制标签：单独声明允许重叠。
4. 不适用、为 0、已恢复但非因果的类别明确分开。
5. 每个非零因果类别至少一个可复现 demo：task ID、输入、reference target、Director/Canvas、Agent input/output、communication、Tool Action–Observation、terminal/evaluator receipt、首个因果失败点和传播链。
6. 只使用已落盘 trajectory/receipt；没有证据的内部推理原因标为不可识别，不预测、不伪造。

本报告只分析既有 128 条 WebShop evaluation artifacts，没有启动训练、模型调用或新的评测。
