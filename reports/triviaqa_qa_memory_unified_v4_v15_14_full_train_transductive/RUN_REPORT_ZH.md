# TriviaQA 全量 Q–A memory v15.14 运行报告

## 1. 评测口径

本报告按项目约定将主结果称为：**全量 TriviaQA Q–A memory 条件下的直接准确率**。

- 固定样本：128 条 TriviaQA validation task。
- 数据库：TriviaQA 全量 76,523 条 Q–A 记录；问题与答案共同进入每条 embedding document。
- 语义保持改写：76,153 条通过生成与语义约束；其余 370 条采用确定性的原始问题—canonical answer 配对回退。固定 128 条中，127 条属于前者，1 条属于后者。
- Embedding：BGE base English v1.5，768 维，L2 normalization，dot-product similarity，TopK=3。
- 评测范围：`in_database_transductive`；数据库包含本次评测对应的 Q–A 记录。
- 正式 evaluator：`triviaqa.official.answer.v1`，TriviaQA official normalization 后计算 Exact Match 与 token-level F1。
- 无 Web Search；无训练、GRPO、LoRA、backward、optimizer update、MACE/贝叶斯更新或 Skill 注入。

仅输入 question、完全不使用数据库的 Qwen3.5-9B closed-book Direct 是独立对照，不与上述主条件混写。

## 2. 架构与数据流

执行链为：

`Question → Qwen3.5-9B Director → progressive Canvas editing → Retriever worker (ReAct + triviaqa.qa_memory) → explicit AgentGraph relation → Reasoner → Verifier → Formatter → FINISH → TriviaQA evaluator → trajectory`

关键边界：

- Director 只执行 Canvas action，不拥有 Tool，也不接收 TopK Q–A payload。
- Retriever worker 在自己的 ReAct execution 中执行 `search(query, k=3)` 和 `read(memory_id)`。
- Tool observation 只沿显式 AgentGraph relation 进入下游 artifact lineage。
- Reasoner、Verifier 与 Formatter 不直接访问 QA-memory。
- Formatter 仅把经过验证的 candidate answer 序列化为 `<answer>...</answer>`。

## 3. 正式 128 条结果

| Condition | 有效样本 | 正确数 | EM | F1 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B closed-book Direct（question-only） | 128 | 45 | 35.1563% | 40.8160% |
| 全量 TriviaQA Q–A memory + AgentGraph Tool 检索直接准确率 | 128 | 122 | **95.3125%** | **95.3125%** |

相对 closed-book Direct：EM **+60.1563** percentage points，F1 **+54.4965** percentage points。该差值是两个明确协议的描述性对照，不是 held-out generalization 的因果估计。

运行状态：128/128 trajectory 完成，128/128 evaluator-valid，122 次显式 `FINISH`，6 次 `canvas_action_domain_exhausted`，0 次 collection/evaluator failure。

## 4. Tool、控制面与检索统计

- `director_tool_calls=0`。
- `director_request_allowed_tools=[]`；Director data plane isolation 通过。
- `retrieval_tool_calls_by_worker=612`；worker ownership violation 为 0。
- 正式 trajectory 使用的 Tool 只有 `triviaqa.qa_memory`；Web Search 调用为 0。
- 127/128 任务实际执行 search；`tc_138` 在 Tool 调用前被 query admission gate 拒绝。
- exact-source Top1：122/128；exact-source TopK：127/128。
- 在实际执行 search 的 127 条中，exact-source TopK 为 127/127。
- 155/155 个 native TopK batch 均完整返回 3 条记录。
- 126 个 retrieval artifact 均通过显式 relation 路由；全局 `retrieval_artifact_routed_via_relation=true`。
- Output inbox receipt lineage 为 124/127；该全任务断言未通过，原因是 terminal failure 的有效 artifact 没有全部到达 Output Agent，不能把它写成全通过。

## 5. 六个错误案例与首个因果失败点

### `triviaqa:tc_117` — ReAct structured-output/schema-conformance failure

- 问题：Who had a 70s No 1 hit with Let Your Love Flow?
- 参考答案：Bellamy Brothers。
- 检索：Top1 exact-source `tc_117`，similarity 0.718275；数据库 read 返回 canonical answer `Bellamy Brothers`。
- 首个失败：Retriever 第一次 `complete` 只报告 Top1，违反 `memory_ids == latest TopK`；修正三条 ID 后又遗漏必填 `retrieval_query`，随后重复至 ReAct turn exhaustion。
- 传播：replacement Retriever 重复同一 schema error → 无 evidence artifact → 未创建 Reasoner/Verifier/Formatter 与 relation → `canvas_action_domain_exhausted` → final answer 为空 → EM/F1 0/0。

### `triviaqa:tc_138` — Director semantic scope loss / conjunctive-constraint loss

- 问题：Who wrote The Turn Of The Screw in the 19th century and The Ambassadors in the 20th?
- 参考答案：Henry James。
- 数据库：存在 `tc_138` exact-source Q–A memory。
- 首个失败：Director 给 Retriever 的 contract 只保留 `The Turn Of The Screw`，丢失 `The Ambassadors` 这一并列约束；所有 query 在 Tool 调用前被 entity-anchor / named-constraint gate 拒绝。
- 传播：Director 只修改 completion condition，未恢复语义 scope → search/read receipt 均为 0 → 无后续 AgentGraph → terminal failure → EM/F1 0/0。

### `triviaqa:tc_154` — Formatter null-sentinel collision

- 问题：How many home runs did Ty Cobb hit in the three World Series?
- 数据集参考答案：`None` / `None (disambiguation)`。
- 检索：Top1 exact-source，Retriever、Reasoner、Verifier 均正确；Reasoner candidate 为字符串 `"None"`。
- 首个失败：Formatter 将合法字符串 `"None"` 当作空值，输出 `<answer></answer>`。
- 传播：严格 format lineage 拒绝 → recovery exhaustion → final answer 为空 → EM/F1 0/0。

### `triviaqa:tc_211` — Formatter whitespace-sensitive serialization failure

- 问题：In which country are Tangier and Casablanca?
- 参考答案：Morocco。
- 检索：Top1 exact-source；Retriever、Reasoner、Verifier 与 Agent communication 均正确。
- 首个失败：Formatter 输出 `<answer>\nMorocco\n</answer>`，与 candidate `Morocco` 的 character-for-character contract 不一致。
- 传播：terminal false rejection → recovery exhaustion → final answer 为空 → EM/F1 0/0。

### `triviaqa:tc_220` — answer-slot cardinality binding failure

- 问题：Name the East African country which lies on the equator.
- 参考答案：Kenya。
- 检索：Top1 exact-source，Reasoner candidate 与 Formatter output 均为 Kenya。
- 首个失败：Reasoner 把单数 country answer slot 错标为 `multiple`；Verifier 正确指出 cardinality 不一致。
- 传播：Director 的 repair 没有修正出错字段 → 正确 Kenya 虽已进入 Formatter，仍被 terminal semantic protocol 拒绝 → final answer 为空 → EM/F1 0/0。

### `triviaqa:tc_223` — Formatter exact-serialization failure

- 问题：In which country did King Hassan II ascend the throne in 1961?
- 参考答案：Morocco。
- 检索：Top1 exact-source；Retriever、Reasoner、Verifier 与 relation routing 均正确。
- 首个失败：Formatter 输出 `<answer>\nMorocco\n</answer>`，违反 exact serialization。
- 传播：Director 只修改 completion condition，没有修复 serialization → terminal/action-domain exhaustion → final answer 为空 → EM/F1 0/0。

综上：6 个错误中，没有一个是向量数据库 TopK recall failure。主要剩余问题是 ReAct artifact schema-conformance、Director 并列语义约束保持、answer-slot cardinality，以及 Formatter/terminal serialization contract；不是 Agent communication 普遍不通。

## 6. 验证与版本状态

- terminal2 canary：`tc_100`、`tc_171`，2/2 通过。
- QA-memory adapter / gateway / runtime / completion benchmark 定向测试：212 passed，另有 40 个 subtests 通过。
- AgentGraph selection 测试中 20 passed；14 个失败来自历史 fixture/profile 未注册，不属于本次 TriviaQA v15.14 变更，未进行无关重构。
- 本版本只做 inference-time architecture adaptation；模型权重未更新，Skill 未注入。

