# TriviaQA fact-memory AgentGraph 最终报告

## 结论

本轮选择 **non-thinking** 作为当前默认推理配置。Reasoner-only thinking 在固定 19 题上与 non-thinking 完全持平，但在 fixed128 的前 16 个同题样本上出现明确回归，因此已按 accuracy gate 停止，不再续跑。

当前可恢复默认配置：

`config/evaluation_triviaqa_fact_memory_unified_v4_v16_4_full_native_transductive.yaml`

本轮没有执行 GRPO、LoRA、backward、optimizer step 或权重更新，也没有使用 Web Search。

## 评测口径

这是 **in-database transductive fact-only development diagnostic**，不是 held-out TriviaQA benchmark result：评测的 128 个 task 对应事实也存在于由全量数据构建的 fact-memory 中。因此下列 EM/F1 只能用于比较当前检索与 AgentGraph 链路，不能作为官方泛化成绩。

Evaluator 为 `triviaqa.official.answer.v1`，固定分母为 128。

## Fact-memory 与检索配置

| 项目 | 值 |
|---|---:|
| fact records | 76,523 |
| unique sources | 76,523 |
| semantic-preserving rewrite records | 76,523 |
| `fact_text` records | 76,523 |
| exact original-question substring | 0 |
| embedding model | BGE-base-en-v1.5 |
| embedding dimension | 768 |
| normalization | L2 |
| similarity | dot product |
| frozen Top-K | 5 |
| Top-K selection split | architecture development |

原始 Question/Answer 只作为数据库外 provenance/evaluation metadata；Agent-facing embedding input 是自包含 `fact_text`。

## fixed128 结果

| 条件 | 样本 | EM | token F1 | FINISH | terminal failure |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B Direct | 128 | 35.16% | 40.82% | N/A | N/A |
| AgentGraph non-thinking | 128 | 82.81% | 84.94% | 110/128 | 18/128 |

AgentGraph 相对 Direct 的描述性差值为 **EM +47.66 个百分点、F1 +44.12 个百分点**。

但是，128 题中只有 123 题通过完整 Output lineage 断言，且有 18 个 terminal failure。因此 formal protocol 状态为 `partial_or_protocol_invalid`，82.81/84.94 是固定分母 diagnostic，不应写成 protocol-valid official result。

## thinking accuracy gate

| 条件 | 同题样本 | EM | F1 | FINISH | terminal failure |
|---|---:|---:|---:|---:|---:|
| non-thinking | 19 | 94.74% | 98.25% | 19/19 | 0 |
| Reasoner-only thinking | 19 | 94.74% | 98.25% | 19/19 | 0 |
| non-thinking | 16 | 93.75% | 97.92% | 16/16 | 0 |
| Reasoner-only thinking | 16 | 87.50% | 91.67% | 15/16 | 1 |

fixed128 前 16 个同题样本上，thinking 的 EM 与 F1 均下降 **6.25 个百分点**。新增回归为 `triviaqa:tc_11`：thinking 以 `canvas_action_domain_exhausted` 终止且答案为空；non-thinking 输出 `<answer>Norway</answer>`，EM/F1 均为 1。因此关闭 thinking，保留其配置和 16 条 trajectory 仅用于诊断。

## Tool ownership 与 Agent communication

| 断言 | 结果 |
|---|---:|
| Director Tool calls | 0 |
| worker search calls | 189 |
| worker read calls | 939 |
| first data-plane action is search | 128/128 |
| complete Top-K read by rank | 128/128 |
| fact artifact routed via explicit relation | 128/128 |
| Output lineage | 123/128 |
| Web Search calls | 0 |
| non-fact-memory search calls | 0 |
| Agent-facing data-plane violations | 0 |

检索只由 `evidence_retriever` worker 在 ReAct execution 中通过 `triviaqa.qa_memory` Tool 完成；Director 只编辑 Canvas，不接收 fact payload。检索结果沿显式 AgentGraph relation 传递给 Reasoner、Verifier 和 Formatter。

## AgentGraph 与终止状态

- 128/128 题均实际执行 AgentGraph。
- 119 题使用 4 个 Agent；其余使用 3、5、6 或 8 个 Agent。
- topology：`serial_3_plus` 121、`fan_in` 4、`fan_out` 2、`parallel` 1。
- explicit FINISH：110/128。
- terminal failure：18/128，其中 `canvas_action_domain_exhausted` 16，`max_rounds` 2。
- collection、provider、evaluator failure：0。

## 错误分类

共有 22 个 evaluator-wrong samples。按首个可观测因果失败点分类：

| 类别 | 数量 | 占错误样本 |
|---|---:|---:|
| worker execution | 10 | 45.45% |
| reasoning / answer selection | 5 | 22.73% |
| terminal | 5 | 22.73% |
| Agent communication / relation | 2 | 9.09% |

### 典型错误 1：Retriever ReAct turn exhaustion

- Task：`triviaqa:tc_55`
- Question：Who became US Vice President when Spiro Agnew resigned?
- Reference：Gerald Ford
- Output：`null`，EM/F1=0/0
- 链路：Director `ADD_SUBGRAPH` → Retriever `search` 1 次、`read` 5 次 → Reasoner blocked → Verifier/Formatter 未形成有效 lineage → terminal。
- 首个失败点：Retriever 虽有 6 个成功 Tool receipts，但第 7 个 ReAct turn 出现 `react_turn_exhaustion`；此前还有一次 `qa_retrieval_query_entity_anchor_loss` schema rejection。
- 错误传播：Retriever 未产生完成态 evidence artifact，Reasoner 被阻塞，最终 Canvas action domain 耗尽。

### 典型错误 2：accepted-answer canonicalization mismatch

- Task：`triviaqa:tc_17`
- Question：Which William wrote Lord of the Flies?
- Reference：Golding
- Output：`<answer>William Golding</answer>`，EM=0，F1=0.6667
- 链路：Retriever `search` 1 次、`read` 5 次 → Reasoner → Verifier → Formatter → FINISH。
- 首个失败点：语义答案正确，但 Formatter 保留完整人名，没有与 accepted-answer surface form 对齐。

### 典型错误 3：terminal failure

- Task：`triviaqa:tc_47`
- Question：Who directed 2001: A Space Odyssey?
- Reference：Stanley Kubrick
- Output：`null`，EM/F1=0/0
- 链路：Retriever → Reasoner → Verifier → Formatter 已构建，随后 Director 继续修改 Canvas。
- 首个失败点：没有在合法 evidence lineage 可用时 FINISH，最终 `canvas_action_domain_exhausted`。

### 典型错误 4：AgentGraph relation 缺失

- Task：`triviaqa:tc_104`
- Question：Which James Bond film features a song by Louis Armstrong?
- Reference：On Her Majesty's Secret Service
- Output：`null`，EM/F1=0/0
- 链路：Retriever 成功 `search`/`read` → Reasoner → Director 添加 Formatter，但 Verifier 没有收到一个直接 Reasoner semantic-candidate artifact。
- 首个失败点：Canvas edit 被 relationship validator 拒绝；反馈为 Verifier direct Reasoner input count=0。
- 错误传播：Director 后续没有修复该 relation，最终 Canvas action domain 耗尽。

### 典型错误 5：答案 surface form

- Task：`triviaqa:tc_58`
- Question：Which George invented Kodak roll-film camera?
- Reference：Eastman
- Output：`<answer>George Eastman</answer>`，EM=0，F1=0.6667
- 链路：Retriever `search` 1 次、`read` 5 次 → Reasoner → Verifier → Formatter → FINISH。
- 首个失败点：semantic answer 正确，但 final answer 没有按 evaluator accepted-answer canonical form 输出。

每个案例的完整 Director action、Canvas snapshot、Agent execution、communication、ReAct Tool receipt、terminal receipt 和 evaluator receipt 均保存在 `formal_result_analysis.json`，没有为报告补造 trajectory。

## 当前状态

- thinking：关闭；默认 next-run 指向 non-thinking profile。
- fixed128：采用已完成的 non-thinking 128/128，不重跑。
- 数据库：使用已完成 full-native-v1；停止 strict-v2 后续构建。
- GPU4 数据构建服务：已关闭。
- 训练：未发生。
- 后续优先修复项：terminal early stopping、Verifier relation repair、accepted-answer canonicalization；本轮不自动修改架构或重跑评测。
