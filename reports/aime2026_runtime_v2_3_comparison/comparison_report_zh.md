# AIME 2026 Runtime v1 → v2 离线配对报告

固定同序题目：**30**；task_id 对齐：**True**。缺失、无效或 collection_failed 的 AgentGraph 行仍保留在固定分母中。

v1：**4/30**；v2：**10/30**；v2-v1：**+20.00 pp**。
同批 Direct：v1 run **9/30**，v2 run **6/30**；主配对结论使用 AgentGraph v1→v2，不把两次随机 Direct run 的差异归因于 runtime patch。

修复 **8**，回退 **2**，持续正确 **2**，持续未正确 **18**。

故障计数为非互斥标记；首个 failure layer 仅投影已有 receipt，不推断隐藏因果。

| Task | v1 | v2 | 变化 | v1 failure | v1 首层 | v2 failure | v2 首层 |
|---|---:|---:|---|---|---|---|---|
| aime-2026/01 | 1 | 1 | unchanged_correct | - | - | - | - |
| aime-2026/02 | 0 | 1 | repaired | graph | graph | - | - |
| aime-2026/03 | 1 | 0 | regressed | - | - | graph | graph |
| aime-2026/04 | 0 | 0 | unchanged_incorrect | - | agent | runtime | runtime |
| aime-2026/05 | 0 | 0 | unchanged_incorrect | terminal,graph | graph | graph | graph |
| aime-2026/06 | 0 | 0 | unchanged_incorrect | parsing | output_extraction | - | agent |
| aime-2026/07 | 0 | 1 | repaired | - | agent | - | - |
| aime-2026/08 | 0 | 0 | unchanged_incorrect | parsing | output_extraction | terminal,graph | graph |
| aime-2026/09 | 0 | 0 | unchanged_incorrect | parsing | output_extraction | terminal,graph | graph |
| aime-2026/10 | 0 | 0 | unchanged_incorrect | - | agent | runtime | runtime |
| aime-2026/11 | 1 | 0 | regressed | - | - | - | agent |
| aime-2026/12 | 0 | 0 | unchanged_incorrect | parsing | output_extraction | - | agent |
| aime-2026/13 | 0 | 0 | unchanged_incorrect | runtime | runtime | runtime | runtime |
| aime-2026/14 | 0 | 1 | repaired | parsing | output_extraction | - | - |
| aime-2026/15 | 0 | 0 | unchanged_incorrect | terminal,runtime | runtime | - | agent |
| aime-2026/16 | 1 | 1 | unchanged_correct | - | - | - | - |
| aime-2026/17 | 0 | 0 | unchanged_incorrect | terminal,runtime | runtime | - | agent |
| aime-2026/18 | 0 | 0 | unchanged_incorrect | - | agent | runtime | runtime |
| aime-2026/19 | 0 | 1 | repaired | runtime | runtime | - | - |
| aime-2026/20 | 0 | 1 | repaired | - | agent | - | - |
| aime-2026/21 | 0 | 1 | repaired | - | agent | - | - |
| aime-2026/22 | 0 | 0 | unchanged_incorrect | - | agent | - | agent |
| aime-2026/23 | 0 | 0 | unchanged_incorrect | - | agent | - | agent |
| aime-2026/24 | 0 | 1 | repaired | runtime | runtime | - | - |
| aime-2026/25 | 0 | 0 | unchanged_incorrect | - | agent | parsing | output_extraction |
| aime-2026/26 | 0 | 1 | repaired | - | agent | - | - |
| aime-2026/27 | 0 | 0 | unchanged_incorrect | - | agent | runtime | runtime |
| aime-2026/28 | 0 | 0 | unchanged_incorrect | terminal | director | runtime | runtime |
| aime-2026/29 | 0 | 0 | unchanged_incorrect | - | agent | - | agent |
| aime-2026/30 | 0 | 0 | unchanged_incorrect | - | agent | graph | graph |

## 来源映射

- `alignment`：scripts/report_joint_qa_progressive_experiment.py: strict same-order task_id comparison
- `paired_row_schema`：scripts/evaluate_completion_benchmark_round.py::_paired_rows
- `aime_failure_type`：scripts/evaluate_completion_benchmark_round.py::_failure_type
- `first_observable_failure`：scripts/evaluate_completion_benchmark_round.py::_aime_wrong_demo_diagnosis
