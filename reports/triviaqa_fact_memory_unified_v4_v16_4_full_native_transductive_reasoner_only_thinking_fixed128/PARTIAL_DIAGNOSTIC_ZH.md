# TriviaQA fixed128 Reasoner-only thinking 部分诊断

## 状态

- 状态：`stopped_after_paired_regression`
- 已完成：16/128
- 正式 fixed128 指标：N/A
- collection failure：0
- 数据条件：76,523 条 full-native-v1 fact-memory，in-database transductive
- Evaluator：`triviaqa.official.answer.v1`

该运行按 accuracy gate 主动停止，不能按 16 条结果外推或冒充完整 128 条指标。

## 同题 paired comparison

| 条件 | 完成数 | EM | F1 | FINISH | terminal failure |
|---|---:|---:|---:|---:|---:|
| Reasoner-only thinking | 16 | 87.50 | 91.67 | 15/16 | 1 |
| non-thinking | 16 | 93.75 | 97.92 | 16/16 | 0 |

Thinking 相对 non-thinking：EM -6.25 个百分点，F1 -6.25 个百分点。

首个新增回归是 `triviaqa:tc_11`：

- thinking termination：`canvas_action_domain_exhausted`
- thinking final answer：`null`
- thinking EM/F1：0/0
- non-thinking final answer：`<answer>Norway</answer>`
- non-thinking EM/F1：1/1

另一处 `triviaqa:tc_17` 是两个条件共有的 accepted-answer canonicalization mismatch：`William Golding` 对 `Golding`，EM=0、F1=2/3，不是 thinking 新增错误。

## 决策

关闭 thinking，默认使用：

`config/evaluation_triviaqa_fact_memory_unified_v4_v16_4_full_native_transductive.yaml`

该 non-thinking 条件已完整收束 128/128：EM 82.81、F1 84.94。Reasoner-only thinking fixed128 不再续跑，已落盘的 trajectory 仅作为失败诊断保留。
