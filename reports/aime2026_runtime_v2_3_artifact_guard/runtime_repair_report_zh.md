# AIME 2026 统一编排架构 Runtime v2.3 修复报告

## 1. 实验边界

- 固定任务：`aime-2026/01` 至 `aime-2026/30`，分母始终为 30；缺失 trajectory、`max_rounds` 和 collection timeout 均不从分母中删除。
- Director：未训练的本地 Qwen3.5-9B；保持简洁、中性 prompt 与统一原子动作空间。
- AgentGraph：`agent_id + model_id + free-text contract + relation + unique Output Agent`；Director 自主选择 Agent 数量、模型、contract、relation、Output 和 `FINISH`。
- 工具：本条件未启用 Tool；没有 Web Search、检索数据库、AIME 题解库或答案查询。
- 学习：没有训练、backward、optimizer step、LoRA、GRPO、MACE、Bayesian posterior、Skill retrieval 或 Skill evolution。
- `max_rounds=20`、Agent `max_tokens=4096` 和 task timeout 600 秒均为本次冻结实验配置，不是设计文档规定的普遍边界。

## 2. 已接受的统一 core 修复

1. `SET_OUTPUT` 只改变 Output pointer；fresh artifact 不重新执行。
2. `FINISH` 只消费当前 graph revision 的 fresh Output artifact；不重新采样。
3. 每次成功执行保存不可变 artifact identity、Agent、graph revision、model/provider、free-text contract、tool config、上游依赖与 raw output。
4. relation/model/contract/upstream/tool config 的实际变化只 invalidate 受影响节点及其 downstream closure。
5. fan-in 为每个上游分别保留 `source_agent_id / artifact_id / raw_output`；确定性抽取到不同 candidate 时只报告 `candidate_conflict=true`，不判断谁正确。
6. AIME v2.1 parser 支持 bare integer、`\boxed{N}`、`Final Answer: N`、`The answer is N` 和末行独立整数；不调用 LLM、不读 ground truth、不重新解题。
7. 同一 provider/model/request 的 transient failure 使用有限 retry/backoff，并保存每次 attempt receipt；runtime 不暗中换模型。
8. partial execution 明确记录 `SUCCESS / FAILURE / BLOCKED_BY_UPSTREAM`，保留已经成功的上游 artifact。
9. Canvas 提供 typed feedback；同一 graph revision 的相同 rejected action 返回 `repeated_rejected_action`。
10. recovery contract 不得把当前未验证 artifact candidate 写成预承诺答案；artifact 仍可经真实 relation 路由给 Agent 使用。
11. HTTP 200 但 public text 为空的 completion 现在记录为 `EmptyAgentResponse`，不再登记为 fresh semantic artifact；完整 provider/retry/input-provenance receipt 仍被保存。

## 3. 回归验证

最终单元测试结果：`1051 passed`，另有 `197 subtests passed`。

覆盖以下边界：

- `SET_OUTPUT` 与 `FINISH` 不重执行；
- relation-scoped invalidation；
- fan-in provenance 与 target-blind conflict；
- boxed/final/bare AIME extraction 及 fail-closed parsing；
- retry 不换模型；
- 空 completion 失败与成功 upstream preservation；
- partial block status；
- repeated rejected action；
- ground-truth isolation；
- trajectory/turn/execution receipt round trip。

两题 Stable Zero canary 均通过合法显式 `FINISH`。完整 30 题运行状态为 `completed_with_operational_failures`，因此不能把完整 panel 宣称为全通过 Stable Zero。

## 4. 固定 30 题结果

| 条件 | correct / 30 | Accuracy | 说明 |
|---|---:|---:|---|
| Qwen3.5-9B Direct（v2.3 同批复用） | 6/30 | 20.00% | 30 条均为冻结复用，0 次新采样 |
| AgentGraph v1 | 4/30 | 13.33% | 初版；29 条 trajectory |
| AgentGraph v2.3 | 10/30 | 33.33% | 28 条 trajectory；26 条 evaluator-valid |

- v2.3 相对同批 Direct：`+4/30`，即 `+13.33` percentage points。
- v2.3 相对 v1：`+6/30`，即 `+20.00` percentage points。
- v1→v2.3：修复 8 题（02、07、14、19、20、21、24、26）；回退 2 题（03、11）；持续正确 2 题（01、16）；持续错误 18 题。
- v1 Direct 的另一次运行为 9/30，v2.3 Direct 来源运行是 6/30。SGLang receipt 显示 deterministic inference 未启用，因此不能把两次 Direct 的随机差异归因于 runtime patch；正式 v2.3 paired comparison 使用同批 6/30 Direct。
- Direct 的 30 次调用中有 23 次 `finish_reason=length`，22 次 parsing failure；20.00% 不应解释为纯数学推理能力。

## 5. Termination、Parsing 与 Runtime

- 合法显式 `FINISH`：26/30。
- `max_rounds`：2/30（08、09）；正式 `final_answer=null`，历史 candidate 未补入 evaluator。
- collection timeout：2/30（18、27）；没有伪造或补采 trajectory。
- Output parsing failure：1/30（25）。
- trajectory 内部 failed execution turns：7 次，其中 `EmptyAgentResponse` 6 次、`OpenAICompatibleGatewayError` 1 次；涉及 04、10、13、14、28。题 14 recovery 后正确，其余四题 recovery 后合法 `FINISH` 但数学答案仍错。
- 30 个 rejected turns 分布在 9 条 trajectory。只有 08、09 的 invalid/no-op action loop 直接持续到 `max_rounds`；其他 rejected action 后仍继续执行，不能自动解释为最终数学错误的因果根源。

## 6. AgentGraph 与模型使用

- Agent 数量分布：1 Agent ×17；2 Agent ×7；3 Agent ×2；4 Agent ×1；5 Agent ×1。
- topology：single 17；serial-2 6；serial-3+ 1；fan-in 1；reciprocal 1；parallel 1；mixed 1。
- 最终图节点模型：GPT-4o-mini 22；本地 Qwen3.5-9B 12；DeepSeek-V4-Flash 7；Qwen3.5-Flash 5。
- 实际 executor calls：GPT-4o-mini 58；DeepSeek-V4-Flash 16；本地 Qwen3.5-9B 15；Qwen3.5-Flash 11。

这说明统一 model catalog 与非链式 relation 均被真实执行，但当前未训练 Director 仍较常选择 single topology；本轮没有用固定多 Agent 或固定 verifier workflow 干预这一分布。

## 7. 典型 Wrong Demo

以下分类是“首个可观察 failure layer”，不是对隐藏因果的推断。Ground truth 只在终局 evaluator/事后报告中出现。

### 7.1 Agent mathematical reasoning / candidate propagation：Task 22

题目要求计算一个计数问题，ground truth 为 754。

1. `ADD_AGENT solver`（本地 Qwen3.5-9B）→ immutable artifact `100`。
2. `ADD_AGENT verifier`（独立执行）→ artifact `107`。
3. `SET_RELATION solver → verifier` → 只 invalidate/re-execute verifier；其新 artifact 变成 `100`。
4. `SET_OUTPUT verifier` → execution count 0。
5. `FINISH` → execution count 0；parser 得到 100，Accuracy=0。

通信、invalidation、pointer reuse 与 termination 均按协议工作；错误候选从 solver 经真实 relation 传播，属于可观察的 Agent reasoning/candidate adoption failure，不是消息丢失。

### 7.2 Fan-in candidate conflict 后选择错误：Task 03

题目是半球与内切小球的面积比问题，ground truth 为 79。

1. `ADD_AGENT solver`（本地 Qwen3.5-9B）→ `49`。
2. `ADD_AGENT verifier`（GPT-4o-mini）→ 推导后给出 `1`。
3. `ADD_AGENT finalizer` → 独立初始 artifact。
4. relation 将 solver 路由到 finalizer → finalizer 重执行为 `49`。
5. `SET_OUTPUT finalizer` → 0 次执行。
6. 一次重复 relation 被 `no_graph_change` 拒绝。
7. 再加入 verifier→finalizer 后，fan-in provenance 分别保留两个 source；finalizer 仍输出 `49`。
8. `FINISH` → 0 次执行；49≠79。

runtime 没有丢来源，也没有用 ground truth 仲裁冲突；错误发生在 Agent 对冲突候选的数学判断。该题是 v1→v2.3 的回退之一。

### 7.3 EmptyAgentResponse 被显式恢复，但数学答案仍错：Task 28

题目 ground truth 为 107。

1. `ADD_AGENT math_solver`（本地 Qwen3.5-9B）→ `404`，状态 `SUCCESS`。
2. `ADD_AGENT verifier`（DeepSeek-V4-Flash）→ HTTP 200、`finish_reason=length`、4096 completion tokens、空文本；状态为 `FAILURE/EmptyAgentResponse`，solver 的 `404` artifact 保留。
3. 后续同类空响应仍不进入 outputs。
4. Director 显式 `MODIFY_AGENT`，选择 GPT-4o-mini → verifier 产生非空 artifact。
5. 建立 solver→verifier relation 后 verifier 输出 `404`。
6. `FINISH` → 404≠107。

该链证明空 completion 不再覆盖成功 artifact，provider receipt 和 recovery 均可见；终局错误仍是数学推理错误。Task 13 的 429/重试也保持同一 MiniMax 请求，失败后由 Director 显式换到本地模型，不是 runtime 暗中换模型。

### 7.4 正确 artifact 已存在但未显式 FINISH：Task 08

题目 ground truth 为 244；ground truth 未进入 Director 或 Agent 输入。

1. 独立 verifier artifact 已输出 `244`。
2. 后续 `ADD_AGENT extractor` 试图把未验证 candidate `244` 写入 contract，被 `unverified_candidate_in_contract` 拒绝。
3. relation 执行中出现 458/244 等冲突 candidate；provenance 分别保留。
4. 后续出现 `no_graph_change`、`bidirectional_block_too_large` 与 `repeated_rejected_action` typed feedback。
5. 第 16 轮局部修改后 verifier/answer 再次产出 `244`。
6. 第 19 轮 `SET_OUTPUT verifier`，execution count 0，fresh artifact 仍为 `244`。
7. 20 轮预算耗尽，Director 没有执行 `FINISH`；正式 `final_answer=null`，Accuracy=0。

这是 20 个 Wrong Demo 中唯一确认“历史/当前 artifact 与事后 ground truth 相同，但未合法终结”的样本。协议按要求没有自动捞取 candidate；剩余问题是 Director termination/action-budget behavior。

### 7.5 Fail-closed output extraction：Task 25

1. `ADD_AGENT solver`（GPT-4o-mini）→ `The sum of all possible values of BC is 425.`。
2. `SET_OUTPUT solver` → 0 次执行。
3. `FINISH` → 0 次执行。
4. v2.1 parser 返回 `aime_integer_not_found`，没有重新求解或猜答案。

该输出没有遵守“唯一十进制整数”协议；而且 425 与 ground truth 850 不同，因此不是“正确答案因 parser 丢失”的 Accuracy false negative。当前 parser 对 SkillFlow last-number fallback 采用了已在 source map 记录的更窄、fail-closed 适配。

### 7.6 Collection timeout：Tasks 18、27

两题均仅有 `TimeoutError` collection receipt，没有完整 trajectory。它们说明 task-level timeout 仍缺少可报告的 partial trajectory checkpoint；这与单次节点失败时的 upstream artifact preservation 是两个不同边界。

## 8. 当前结论

本轮已将 Output-pointer resampling、FINISH resampling、无来源 fan-in、空 artifact 覆盖、untyped rejection 和 boxed/final-answer false negative 从数学推理失败中分离出来。20 个 Wrong Demo 的首个可观察层分布为：8 个纯 Agent reasoning、3 个 graph rejection 后继续合法终结但答案错误、4 个节点级 runtime failure 后恢复并合法终结但答案错误、2 个 invalid-action loop 导致的 `max_rounds`、2 个 collection timeout，以及 1 个 fail-closed parsing failure。没有证据支持继续用人工固定 Agent 数量、固定角色或固定 topology 修改 initial Director prompt；后续若继续，应先分别处理 task-level partial checkpoint、termination budget behavior 与模型数学能力，而不是将它们混为同一种架构故障。
