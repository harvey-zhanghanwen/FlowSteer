# HotpotQA–TriviaQA 渐进式编排架构完成报告

日期：2026-08-18  
分支：`experiment/joint-qa-progressive-skill-rl-02`

## 已完成

- 按 FlowSteer structure-level `ADD` 的真实调用边界，为自由
  AgentGraph 增加 `add_subgraph`：一个 Director action 可原子加入 1–3 个
  Agent、two-bit relations 和可选 Output Agent；整体验证、整体回滚、成功后
  只进行一次 progressive execution 和一次 Canvas feedback。
- 直接复用现有 AgentGraph validation、dirty closure、quotient-DAG 调度、
  fan-in/fan-out、双 Agent reciprocal `draft → revision`、Agent communication
  receipt、terminal evaluator、trajectory 和 action-masked token span。
- 将初始 Director prompt 收敛为合法 action schema、模型目录、Canvas 状态和
  execution feedback；未加入固定 role、固定 topology、最小 Agent 数、结构奖励
  或未经 evidence gate 的 Skill。
- Executor search space 使用此前 canary-backed 的 10 个模型臂；2026-08-18
  重新读取 `/v1/models` 得到 526 项，9 个远程 exact model ID 均仍存在。
  Flow-Director 仍固定为本地 Qwen3.5-9B。
- 冻结新的联合 QA 数据协议：每个数据集 development 128、train 512、
  quarantine 32、Skill-confirmation 64、final test 128；完整 ordered task IDs
  写入 manifest，四个可用分区两两不相交。
- HotpotQA/TriviaQA evaluation loader 已支持显式 development/test 路径和
  frozen task-ID selection；训练 schedule 将 Skill-confirmation 与 test 一并
  纳入 held-out 集，并检查循环训练样本的 `base_task_id`。

## 已验证模块

- strict action parser 与完整 sampled-action span；
- subgraph transaction 的原子 rollback；
- 单 Agent、三 Agent fan-in、双 Agent reciprocal block；
- solver → Format Agent → exact-answer terminal protocol；
- legacy/new action profile 配置兼容；
- joint-QA partition 顺序、数量、不相交与 quarantine 隔离；
- train schedule 对 confirmation/test 及 `base_task_id` 泄漏的拒绝；
- HotpotQA/TriviaQA split loader。

无模型单元测试结果：`299 passed，90 subtests passed`。唯一 warning 是既有
Pydantic v2 class-based config deprecation，与本轮改动无关。

## 预留接口与尚未完成项

- `SkillEvidencePipeline`、paired intervention、联合贝叶斯后验、MACE-style
  UCB、EVSI 数值原语和 ACTIVE Skill retrieval 已分别存在；新
  `add_subgraph`/多模型 regime 的 fresh discovery、confirmation 与 delayed
  activation 尚未运行。
- 真实 one-pass GRPO、`optimizer.step()`、LoRA checkpoint、SGLang adapter
  sync 和 updated-policy canary 已有实现；本轮新 regime 尚未启动训练。
- final test 必须在 architecture/policy/Skill 全部冻结后只运行一次；此前使用过
  的 validation128 仅作为 development/diagnostic，不报告为 final test。

## Stable Zero 状态

静态接口与无模型回归已通过；live Stable Zero 尚未确认。确认条件是本任务自己的
GPU4 Qwen3.5-9B Supervisor 启动后，固定 HotpotQA 与 TriviaQA development task
能够完成 `Question → Director → add_subgraph → Runtime/communication → Format
Agent → FINISH → evaluator → trajectory`，且完整 receipt 可重放。在完成该检查前
不把本阶段描述为 live Stable Zero。
