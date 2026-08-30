# TriviaQA v16.4 可恢复诊断 checkpoint

## 结果口径

- 固定样本数：128
- AgentGraph 诊断 EM：82.81%
- AgentGraph 诊断 F1：84.94%
- Qwen3.5-9B Direct EM：35.16%
- Qwen3.5-9B Direct F1：40.82%
- 显式 `FINISH`：110/128
- 合法输出 lineage：123/128
- Director Tool 调用：0
- worker `search` / `read`：189 / 939
- Web Search：0

## 状态

该批次 128 条样本已经完整收束，但协议校验未完全通过，因此以上 EM/F1
是完整批次诊断指标，不是正式 protocol-valid 指标。主要阻塞是 5 条输出
lineage 不完整，以及旧事实语料中存在未消解指代的陈述，例如
`The company called it Frosted food.`。后续版本必须先以严格的自包含陈述事实
准入规则重建语料和索引，再使用同一批 128 条样本重新评测。
