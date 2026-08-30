# AIME 2026 AgentGraph v3 → v5 同 30 题配对评测与根因报告

## 1. 评测口径

- 数据：AIME 2026 官方 test split，共 30 题；v3 与 v5 的 selected_tasks 完全一致。
- Director：本地 Qwen3.5-9B，冻结相同 policy identity、seed 与 model catalog order。
- Direct：两组均复用同一批 Qwen3.5-9B Direct predictions。
- Evaluator：canonicalized integer Exact Match / Accuracy，版本 skillev.integer.target-blind-extraction.v2.1。
- v5 未启用 Tool、检索、训练、GRPO、LoRA、MACE、Bayesian inference 或 Skill。

## 2. 正式结果

| 条件 | Correct / Total | Accuracy | Evaluator-valid | Explicit FINISH | Operational failure |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B Direct | 6 / 30 | 20.00% | 30 | N/A | 0 |
| AgentGraph v3（best） | 14 / 30 | 46.67% | 29 | 29 | 1 |
| AgentGraph v5（candidate） | 12 / 30 | 40.00% | 30 | 30 | 0 |

v5 相对 Direct 提升 6/30，即 20.00 个百分点；相对 v3 下降 2/30，即 6.67 个百分点。v5 因此不能替换 v3 best-profile。

## 3. v3 → v5 逐题转移

- Correct → Wrong：4 题，task16、task20、task24、task27。
- Wrong → Correct：2 题，task07、task18。
- Both Correct：10 题，task01、02、03、06、12、19、21、23、25、26。
- Both Wrong：14 题，task04、05、08、09、10、11、13、14、15、17、22、28、29、30。

## 4. Runtime、artifact 与 termination 结果

- v5：30/30 显式 FINISH，0 max_rounds，0 output parsing failure，0 terminal failure，0 operational failure。
- v3：29/30 显式 FINISH；task28 为 collection timeout，固定分母中仍计为失败。
- v5 共观察到 24 个初始 finish_reason=length receipt；12 个进入同模型 bounded continuation，其中 3 个完成为 stop，9 个转为 typed IncompleteAgentArtifact。
- v5 的 30 个正式终局 artifact 全部 fresh、complete、parseable；不完整 artifact 未进入 Output admission 或 evaluator。
- v5 共 154 个 Director/Canvas turns，v3 为 205；v5 executor calls 为 93，v3 为 123。
- v5 的 19 次 Canvas rejection 与 1 次 Director action parse failure 均被 recovery 收束；未发生 repeated_rejected_action loop。

这些结果说明 bounded same-model continuation、artifact completeness gate、pointer-only SET_OUTPUT、explicit FINISH 与 durable recovery 修复了运行时完整性问题。

## 5. AgentGraph 搜索行为

| 条件 | Single | Serial-2 | Reciprocal | Serial-3+ |
|---|---:|---:|---:|---:|
| v3 | 16 | 11 | 1 | 1 |
| v5 | 28 | 0 | 1 | 1 |

v5 所有 FINISH 前均只有一个 fresh candidate 和一个 provenance root；30 个终局 candidate 都是 unassessed，未出现 candidate conflict、refuted 或 insufficient_evidence receipt。reject_negative terminal policy 因此在这批任务上主要表现为：完整、可解析、未被否定的单一 artifact 可以立即 SET_OUTPUT → FINISH。该机制提高了收束率，但显著减少了 relation construction、downstream assessment 与局部 repair 的实际发生机会。

这不是 evaluator 或 Canvas transaction bug，而是 artifact-consumption ordering 与 terminal admission 共同导致的搜索行为变化。由于 v5 Accuracy 回退，当前不把该条件设为默认 best-profile。

## 6. 18 个 v5 Wrong Demo 的首个可观察 failure layer

| 类别 | 数量 | 占错误题比例 | 典型 task |
|---|---:|---:|---|
| Agent mathematical reasoning | 11 | 61.11% | 04、05、08、10、11、13、15、16、20、27、29 |
| Runtime execution failure receipt | 6 | 33.33% | 14、17、22、24、28、30 |
| Graph action rejection | 1 | 5.56% | 09 |
| Artifact selection / terminal / evaluator / parsing | 0 | 0% | 无 |

Runtime failure receipt 是首个可观察异常，不等于最终数学错误的唯一因果。例如 task24 同时存在 contract method drift、length/incomplete artifact 与后续错误级数变换。

## 7. 代表性 paired 根因

### task07：修复

v3 的 contract 将题目中的六次迭代改写为 π^7，最终预测 1。v5 保留 π^6，预测 396，与 ground truth 一致。该题支持 target-blind task-specification contract admission 的必要性。

### task16：回退

v5 选择 gpt-4o-mini 单节点。执行器将 d=10/k 的合法性错误化为 k | 90，而正确约束来自 d | gcd(20,30)=10，预测 282，ground truth 为 178。首个因果层是 model routing 后的数学推理错误；Runtime、artifact 与 termination 均正常。

### task20：回退

v5 在组合计数展开中错误得到 5(b-2)=r-5，正确关系应为 5(b-2)=3(r-4)，预测 245，ground truth 为 190。首个因果层是数学推理，不是 communication 或 terminal failure。

### task24：回退

初始 contract 预先限定将无限 Lambert series 转换成有理数，形成 method drift；DeepSeek 输出多次 length/incomplete 后切换为 gpt-4o-mini，后者又把 1/(10^n-1) 错误等同于几何级数，最终预测 370，ground truth 为 669。首个因果层是 Director contract method constraint，随后叠加 Runtime truncation 与数学推理错误。

### task27：回退

contract 仅写 Tetrahedron Analysis，缺少可验收的 task-specific completion condition。执行器在 Heron 公式中把 sqrt(25×225) 错算为 375（正确为 75），随后无充分推导跳到 58；ground truth 为 223。首个因果层是 contract underspecification，随后是数学推理与 unsupported conclusion。

### task18：运行恢复成功，同时暴露 admission 假阳性

v5 最终预测 503 并答对，且 timeout/continuation recovery 正常。但原题写作 14\sqrt2，contract 写作 14*sqrt(2)，旧 guard 把等价公共数值 2 误判为 question-external。评测后已加入 target-blind radical-notation equivalence 和定向回归；该代码修复未重跑 30 题，因此不附加新的 Accuracy 声明。

## 8. 版本选择

- 当前 best-profile：aime2026_runtime_v3_artifact_termination，14/30 = 46.67%。
- v5：完整、可恢复、运行稳定，但 12/30 = 40.00%，记录为 unselected candidate。
- 统一 core 的 contract admission、bounded continuation、artifact completeness 和 typed recovery 代码保留为显式配置能力；默认 v3 profile 不启用造成回退的 v5 terminal condition。
- 后续若继续实验，应在 development split 上验证 terminal admission 与 downstream assessment 的平衡，不能根据这 30 题 ground truth 注入固定 Solver/Verifier、固定 Agent 数或 topology template。

## 9. 证据

- v3 report：reports/aime2026_runtime_v3_artifact_termination/evaluation_report.json
- v5 report：reports/aime2026_runtime_v5_contract_completeness/evaluation_report.json
- v3 paired results：artifacts/aime2026_runtime_v3_artifact_termination/evaluation/paired_results.jsonl
- v5 paired results：artifacts/aime2026_runtime_v5_contract_completeness/evaluation/paired_results.jsonl
- v3 trajectories：artifacts/aime2026_runtime_v3_artifact_termination/evaluation/agentgraph_trajectories.jsonl
- v5 trajectories：artifacts/aime2026_runtime_v5_contract_completeness/evaluation/agentgraph_trajectories.jsonl

本轮未执行训练、backward、optimizer update、LoRA、GRPO、MACE、Bayesian inference、Skill injection、Tool 或答案检索。
