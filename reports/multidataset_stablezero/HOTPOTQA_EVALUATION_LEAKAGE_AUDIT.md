# HotpotQA evaluation leakage protocol audit

## 结论

当前 HotpotQA Stable Zero 结果是 **2/2 exposed development canary**。它能证明这两题上的 Direct、AgentGraph、Tool、evaluator 与 trajectory 链路可以完成，但不能报告为 100% held-out accuracy、128 题验证结果或 HotpotQA benchmark estimate。

本次核对区分两个不同问题：

1. **Prompt leakage**：没有证据表明 `ground_truth`、accepted answers 或 evaluator payload 被写入本次模型请求。
2. **Evaluation contamination**：两条 canary task ID 曾反复用于架构开发、诊断和 progressive evaluation；它们已经不是 unseen held-out samples。

因此，“模型输入中没有 Ground Truth 字段”是成立的，但不能据此把这两题恢复为 held-out evidence。

## 当前证据范围

- Stable Zero receipt：
  - `artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/protocol_separated_results.jsonl`
  - `artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/tool_agentgraph_trajectories.jsonl`
  - `artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/direct_closed_context_predictions.jsonl`
- 两条 task ID：
  - `hotpotqa:5a7a06935542990198eaf050`
  - `hotpotqa:5a879ab05542996e4f30887e`
- 两题 Direct 与 AgentGraph 的 EM/F1 都是 2/2；这个分数只属于上述 exposed development canary。
- 相同 task ID 可在既有 `reports/hotpotqa_multiagent_skill/`、`reports/joint_qa_progressive/` 和 `reports/joint_qa_curve/` artifacts 中找到，说明它们曾参与多轮架构开发与诊断。

## 模型可见输入边界

仓库内真实调用链如下：

- `scripts/evaluate_completion_benchmark_round.py::_direct_one` 构造 `AgentRequest` 时使用 `problem=task.question`。
- `scripts/train_agentgraph_smoke.py::_workflow_problem` 对静态 QA 返回 `task.question`；该值进入 Flow-Director 与 AgentGraph execution request。
- `artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/tool_agentgraph_trajectories.jsonl` 的 recorded Director prompt 和 execution request 包含题目、给定 passages、Canvas state、execution feedback 与 upstream artifacts；未记录 `ground_truth` 或 `evaluator_payload` 字段进入这些模型输入。
- `scripts/evaluate_completion_benchmark_round.py` 在生成完成后调用 `_evaluate_prediction`；`ground_truth` 与 accepted answers 在 evaluator 边界使用。
- 同一脚本的 known-answer preflight 用 `selected[0].ground_truth` 或 accepted answer 调用 `_evaluate_prediction`，目的是验证 evaluator 可正确给出满分；该 preflight 位于模型生成路径之外。对应 receipt 为 `artifacts/qa_tool_react_exact_wire_v3_stable_zero/hotpotqa/preflight_receipt.json`。

需要注意，prediction/result artifacts 会为了可复现评测而保存完整 `task` 和 `evaluation`，其中可包含 Ground Truth。这些落盘字段不等于模型 prompt；判断 prompt leakage 必须查看 request/prompt receipt，而不能只搜索结果文件中的 `ground_truth` 字段。

## HotpotQA supplied passages 与 RetrievalIndex

HotpotQA distractor protocol 的问题输入按设计包含若干 passages，其中包括支持答案的事实和 distractor passages。因此，模型从 `TaskRecord.question` 中读取支持事实是该协议本身，不是 Ground Truth 字段泄漏。

第二条 canary 的 AgentGraph trajectory 还记录了一次 `qa-retrieval.search`：

- query：`Oberoi Group head office location`
- source：`atlas-dpr-wikipedia-psgs-w100`
- backend：`sqlite-fts5-lexical`
- 返回公开 Wikipedia passage，内容指出 The Oberoi Group 的 head office 在 Delhi。

该 Tool 调用由 `src/interactive/qa_tool_adapter.py` 执行 SkillFlow `RetrievalIndex.search/read`，索引生命周期由 `src/interactive/qa_retrieval.py` 管理；它没有 evaluator access。这个 receipt 证明检索结果来自公开 Atlas DPR Wikipedia 语料，而不是按 task ID 或 accepted answer 查询 evaluator 数据。

## Evaluation contamination 的处置

`data/joint_qa_v2/manifest.json` 已把数据划分为独立的 `development`、`train`、`quarantine`、`skill_confirmation` 与 `test` partitions：HotpotQA 每个 partition 分别为 128、512、32、64、128 条。上述两个 canary task ID 位于 `data/joint_qa_v2/development.jsonl`，而不是 `test.jsonl`。

后续报告规则：

- 现有 v3 结果只标记为 exposed development canary。
- Stable Zero canary 只用于验证端到端链路，不外推为 benchmark accuracy。
- 最终 benchmark estimate 必须使用未参与架构选择、prompt 调整、failure analysis 或 Skill confirmation 的固定 `test` partition，并在运行前冻结 task IDs、evaluator、model catalog、policy/adapter version 与 runtime condition。
- 不得使用 development canary、训练样本、behavior reward、known-answer preflight 或 forced Tool probe 替代 held-out metric。

## 判定

```text
GROUND_TRUTH_FIELD_IN_MODEL_PROMPT = NOT_OBSERVED_IN_CURRENT_RECEIPTS
EVALUATOR_PAYLOAD_IN_MODEL_PROMPT = NOT_OBSERVED_IN_CURRENT_RECEIPTS
SUPPLIED_PASSAGES_CONTAIN_SUPPORTING_FACTS = YES_BY_HOTPOTQA_PROTOCOL
RETRIEVAL_SOURCE = PUBLIC_ATLAS_DPR_WIKIPEDIA
DEVELOPMENT_SAMPLE_REUSE = YES
UNSEEN_HELDOUT_CLAIM_ALLOWED = NO
BENCHMARK_ACCURACY_CLAIM_ALLOWED = NO
```
