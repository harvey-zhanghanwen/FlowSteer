# TriviaQA `unified_architecture_v2`：worker-only QA-memory 固定 128 条验证报告

## 结论

本轮完成了 inference-only 的 worker-only QA-memory 接线、固定 128 条
held-out validation 运行和离线诊断。Director 与 retrieval 数据面实现了隔离，
worker Agent 也确实执行了动态 `search/read`；但本条件没有通过 Stable Zero：
128/128 条 trajectory 均未形成合法 evidence artifact，因而没有 downstream
relation routing、Output inbox lineage 或 `FINISH`。最终 AgentGraph EM/F1 都是
0%，不能替换当前历史 best-profile。

## 评测口径与指标

- 数据：项目冻结的 128 条 TriviaQA held-out validation，阶段为
  `development`，不是 TriviaQA 公共 test leaderboard。
- Evaluator：`triviaqa.official.answer.v1`。
- 分母：固定 128；terminal failure 产生空答案时按 0 计入，不从分母剔除。
- Director：GPU0 上本地 Qwen3.5-9B，`behavior_policy_version` 为
  `qwen35-9b-base-triviaqa-qa-memory-unified-v2`。
- 本条件的 admissible model catalog 实际只暴露
  `qwen3.5-9b-local`，因此本轮所有 worker Agent 也使用该模型；没有 API fallback。
- 本轮未使用 Web Search，未进行 Skill、MACE、Bayesian、GRPO、LoRA、
  backward、optimizer update 或 policy synchronization。

| 条件 | evaluator-valid | EM | F1 | terminal failure |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B Direct，question-only | 128/128 | 45/128 = **35.16%** | **40.82%** | 0 |
| worker-only QA-memory AgentGraph | 128/128 | 0/128 = **0.00%** | **0.00%** | 128/128 |
| AgentGraph − Direct | — | **−35.16 pp** | **−40.82 pp** | +128 |

历史 `triviaqa_round_01_stable_zero` 的 AgentGraph EM 52.34%、F1 61.80%
使用 deterministic DPR/Wikipedia prefetch，且历史报告的 protocol label 与本轮
question-only Direct / dynamic train QA-memory 条件不同，因此只能作为历史参考，
不能与本轮分数作同协议因果比较。

## 架构实现与来源边界

| 边界 | 状态 | 来源与实现 |
| --- | --- | --- |
| Qwen3.5-9B Director / Supervisor | 已实现 | 保留 SkillFlow 的本地模型服务与 bounded execution 边界；Director 只编辑 Canvas。 |
| Progressive Canvas editing | 已实现 | 复用 FlowSteer 的 `edit → execute → feedback`、显式 `FINISH` 和 trajectory receipt。 |
| AgentGraph | 已实现 | 使用 `ADD_AGENT/ADD_SUBGRAPH`、`MODIFY_AGENT`、`SET_RELATION`、`SET_OUTPUT`、`FINISH`；本轮没有固定串行 topology。 |
| Worker ReAct retrieval | 已实现 | `triviaqa.qa_memory` 仅绑定到 `execution_mode=react` 的 worker Agent；ReAct 是 Agent execution policy，不是 Agent role。 |
| QA-memory embedding index | 已实现，TriviaQA 必要适配 | 512 条冻结 train QA 的 semantic-preserving paraphrase，BGE 768 维、L2 normalization、dot-product、冻结 `top-k=3`。 |
| Evidence provenance / relation routing | 接口已实现，运行未通过 | provenance validator 和 explicit relation receipt 已接线，但 128 条均未生成合法 evidence artifact。 |
| Skill / training | 未接入本条件 | 配置明确关闭；本轮结果不是 Skill gain 或训练增益。 |

详细源码映射见 [SOURCE_MAP.md](../../docs/SOURCE_MAP.md) 和
[恢复入口](../../docs/TRIVIAQA_QA_MEMORY_UNIFIED_V2.md)。

## Split isolation 与 QA-memory manifest

- memory 数：512；unique train source：512；cycled：0；paraphrase：512。
- held-out validation isolation：128/128；`validation_content_indexed=false`。
- validation question、answer、accepted aliases、supporting facts 和 evaluator
  receipt 均未写入 index。
- train QA 的 `canonical_answer` 只存在于 worker Tool 数据面；accepted answers
  只在运行结束后用于离线 Answer Recall 诊断。

证据：[index manifest](../../data/triviaqa_qa_memory_v1/index/manifest.json)。

## Control plane / data plane 断言

| 断言 | 实测 | 结果 |
| --- | ---: | --- |
| `director_tool_calls=0` | 0 | 通过 |
| `retrieval_tool_calls_by_worker>0` | 308，ownership violation=0 | 通过 |
| `retrieval_artifact_routed_via_relation=true` | artifact task=0，routed task=0 | **未通过** |

Director provider request 的 `allowed_tools=[]`，没有 `tools/tool_choice`，也没有
top-k、query、similarity、memory record 或 read observation。对 128 条 trajectory
的来源感知检查确认实际 retrieval payload exposure 为 0。报告另保留 11 条 bare
lexical collision 作为诊断：7 条来自公开 question，1 条来自 Director 自生成的
Agent contract，2 条是 `76` 与 `context=32768` 的子串重合，1 条是 `H/V` 与协议
文本的单字符重合；这些不是 worker receipt 注入。

第三项断言为 false 不足以单独证明 relation adapter 损坏：前置 worker evidence
artifact 从未成功生成，因此没有 artifact 可供 relation routing。它仍然是本条件的
Stable Zero admission failure，不能标记为通过。

## Retrieval Tool 使用与离线 Answer Recall

以下 accepted-answer match 仅是运行后的离线诊断，不进入模型或 Tool 请求；它也
不保证 held-out question 的 target relation 与 train-memory relation 一致。

| 指标 | 结果 |
| --- | ---: |
| 使用 QA-memory Tool 的任务 | 128/128 |
| 唯一物理 Tool receipts | 308 |
| `search` / `read` | 130 / 178 |
| 成功 Tool receipts | 308/308 |
| 非空 search / 返回候选 | 130/130 / 390 |
| 至少一次 read | 108/128 = 84.38% |
| 多条成功规范化 query | 2/128 = 1.56% |
| 512-memory 全库 accepted-answer 覆盖上限 | 21/128 = 16.41% |
| 实际 Answer Recall@1 | 5/128 = 3.91% |
| 实际 Answer Recall@3 | 8/128 = 6.25% |
| corpus 可覆盖任务内 Recall@3 | 8/21 = 38.10% |
| 正确候选被 read | 6/128 = 4.69% |
| 正确 read 转为合法 evidence artifact | 0/6 |

这组数值把失败分成两个前后相继的瓶颈：首先，512 条 train QA-memory 对固定
validation 的答案覆盖上限仅 16.41%，实际 Recall@3 进一步降到 6.25%；其次，
即便 6 条样本已经 read 到 accepted-answer match，evidence span、entity identity、
target relation 或 provenance validation 仍拒绝 completion，导致 artifact conversion
为 0。

## Failure taxonomy

按首个持久化因果失败点：

| 类别 | 数量 | 占 128 条错误样本 |
| --- | ---: | ---: |
| worker execution / ReAct completion | 101 | 78.91% |
| Agent communication / relation | 27 | 21.09% |

Runner 的互斥 operational 分类为：relation or answer-slot binding 84、retrieval
strategy 23、structured output or format 21。全部 128 条 termination reason 都是
`canvas_action_domain_exhausted`，`explicit_finish=false`，`final_answer=null`。

主要根因：

1. **Corpus coverage 与 retrieval recall**：107/128 条 validation 的 accepted
   answer 不在 512-memory corpus；实际 search 仅命中 8 条。
2. **ReAct action admission**：query rewrite 经常被 entity anchor、named scope、
   relation semantics 或 duplicate-query guard 拒绝；只有 2 条产生多条成功 query。
3. **Artifact schema / semantic provenance**：正确 read 后，模型生成的
   `evidence_span`、predicate 或 entity binding 与 read receipt 不满足严格约束。
4. **AgentGraph relation**：部分 Director action 缺失 semantic-answer → Formatter
   relation，或 repair 后 Retriever 脱离通向 Output 的 active lineage。
5. **Terminal semantics**：没有合法 evidence lineage 时 terminal gate 正确拒绝
   `FINISH`，但 recovery 持续消耗 action domain，最终 128/128 terminal failure。

## 典型 Wrong Demo

### Demo 1：`triviaqa:tc_5` — retrieval miss 与 query rewriting admission failure

- Question：In which decade did Billboard magazine first publish an American hit chart?
- Reference：`30s` / `1930s`。
- 链路：Retriever `node_1` → Reasoner `node_2`；Verifier 未获得合法输入。
- Tool：search `billboard magazine first publish american hit chart`；top-3 为
  Wham!、Elton John、Mick Jagger 相关 QA；没有 read。
- 首个失败：Retriever `react_turn_exhaustion`；后续 query 分别触发
  `query_strategy_semantics_mismatch`、`entity_anchor_loss` 和
  `duplicate_normalized_query`。
- 传播：Reasoner / Verifier blocked → repair 未产生新 receipt →
  `canvas_action_domain_exhausted`。
- Final：`null`；EM/F1 = 0/0。

证据：[trajectory line 3](../../artifacts/triviaqa_qa_memory_unified_v2/agentgraph_trajectories.jsonl#L3)。

### Demo 2：`triviaqa:tc_52` — Recall@1 命中但没有 read

- Question：Who was the first woman to make a solo flight across the Atlantic?
- Reference：`Amelia Earhart`。
- 链路：Retriever `node_2` → Reasoner `node_1`。
- Tool：search `first woman solo flight across the Atlantic`；top-1 为
  `Amelia Earhart`，similarity 0.7521；随后没有 read。
- 首个失败：后续 action 被 `duplicate_normalized_query`、
  `named_scope_loss[atlantic]`、`scope_modifier_loss` 和
  `strategy_semantics_mismatch` 拒绝。
- 传播：正确 SearchHit 未转为 read receipt/evidence artifact → Reasoner 未执行 →
  无 Verifier / Formatter 输出。
- Final：`null`；EM/F1 = 0/0。

证据：[trajectory line 35](../../artifacts/triviaqa_qa_memory_unified_v2/agentgraph_trajectories.jsonl#L35)。

### Demo 3：`triviaqa:tc_8` — 正确答案已 read，但 relation/provenance validation 拒绝

- Question：From which country did Angola achieve independence in 1975?
- Reference：`Portugal`。
- Tool：search `Angola independence 1975 country`；top-1/read 得到同事实 train
  memory：`Which European nation granted Angola its independence in 1975?`。
- 首个失败：`qa_semantic_evidence_provenance_invalid`；问题关系
  `achieve independence` 与 evidence predicate `granted ... independence` 没有通过
  controlled relation alignment。
- 附加 topology 问题：repair 后 Retriever `node_3` 没有通向
  Reasoner/Verifier/Formatter active lineage 的出边。
- 传播：合法 artifact 未生成 → 无 relation routing → Output inbox 为空。
- Final：`null`；EM/F1 = 0/0。

证据：[trajectory line 4](../../artifacts/triviaqa_qa_memory_unified_v2/agentgraph_trajectories.jsonl#L4)。

### Demo 4：`triviaqa:tc_1` — Canvas relation rejection 后 retrieval miss

- Question：Which American-born Sinclair won the Nobel Prize for Literature in 1930?
- Reference：`Sinclair Lewis`。
- Round 0：Director 创建 Reasoner → Verifier 并设置 Formatter 为 Output，但缺少
  semantic-answer → Formatter relation，Canvas 拒绝 edit。
- 后续 Tool：search `Sinclair Nobel Prize literature American-born`；top-3 为
  Ralph Johnson Bunche、Hammond Innes、Alan Paton；read 了 Bunche memory。
- 首个因果失败：`orchestration_relationship`；替代 Retriever 仍未生成合法
  evidence artifact，也没有形成完整 Output lineage。
- Final：`null`；EM/F1 = 0/0。

证据：[trajectory line 1](../../artifacts/triviaqa_qa_memory_unified_v2/agentgraph_trajectories.jsonl#L1)。

## 验证与版本坐标

- Feature branch：`feature/triviaqa-qa-memory-unified-v2-20260827`。
- Pre-QA-memory backup：`backup/triviaqa-unified-v2-v63-pre-qa-memory-20260827`。
- QA-memory data commit：`9aae022`。
- Worker-only architecture commit：`91326fb`。
- 最终定向回归：19 passed；只有既有 Pydantic deprecation warning。
- 正式 manifest：`completed_with_terminal_failures`，128/128 evaluator-valid。

关键 artifacts：

- [run manifest](../../artifacts/triviaqa_qa_memory_unified_v2/run_manifest.json)
- [完整 trajectories](../../artifacts/triviaqa_qa_memory_unified_v2/agentgraph_trajectories.jsonl)
- [paired results](../../artifacts/triviaqa_qa_memory_unified_v2/paired_results.jsonl)
- [正式分析 JSON](formal_result_analysis.json)
- [正式分析 Markdown](formal_result_analysis.md)

本轮结果否决了“仅把 dynamic QA-memory Tool 接入 worker 就会提高准确率”的假设。
下一轮若继续，应先在 train/architecture-development 上分别验证 corpus coverage、
SearchHit→read continuation 和 read→evidence artifact conversion；不能用 held-out
validation 调 top-k 或放宽 evidence provenance，也不能把 retrieval payload 移到
Director。
