# Skill Evidence Epoch 5 废弃报告

Epoch 5 在完成 4/174 条 trajectory 后停止，不能用于估计 Skill prompt-prior
visibility 的 intent-to-treat effect，也不能进入 Bayesian posterior、Skill gate
或 GRPO。

原因是 treatment arm 特有的接口不兼容：候选 Skill 要求首个
`ADD_SUBGRAPH` 不设置 Output，Qwen3.5-9B Director 合法地生成了显式 JSON
`"output_agent_id": null`；当前 parser 只接受字段缺省，不接受 present-null。
候选条件共出现 10 次该形式，其中 9 次被
`output_agent_id must be a non-empty string` 拒绝；incumbent 条件为 0 次。
因此已观测 raw reward 受到条件相关的工程故障污染，不能解释为 Skill 效果。

已保留全部 4 条 trajectory 和 44 条 TriviaQA retrieval receipt 供故障诊断，
但不发布 Skill、不更新 posterior、不启动训练。Epoch 5 的 discovery 与
confirmation coordinates 全部隔离；后续 epoch 使用新的任务坐标和新的 tool
version，并对 `ADD_SUBGRAPH` 的可选 Output 字段做最小 JSON 兼容适配。
