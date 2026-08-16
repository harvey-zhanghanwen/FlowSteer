# Architecture Completion Report

## 结论

Round-01 Stable Zero 推理链已完成。Training-ready Step 0 的协议修正、版本
配置、独立通信消融和 untouched confirmation 也已完成真实复验：

`Question → local Qwen3.5-9B Director → Canvas/AgentGraph → AgentRuntime`
`→ Agent communication → Output Agent → Hotpot evaluator → Trajectory`

固定 development-128 与一次性 untouched-32 共 160/160 条 AgentGraph 都显式
`finish`，均保存 Output Agent 实际收件箱、完整 Director/Executor 回执和有效
evaluator receipt；没有 collection failure 或 `max_rounds` 终止。因此 Stable Zero
仍成立。任务表现、通信利用率和 GRPO 放行判断见
`reports/hotpotqa_training_ready_step0/TRAINING_READY_STEP0_REPORT.md`。

## 1. 已完成

- FlowSteer 式 progressive Canvas：一次一个原子编辑、事务校验、执行反馈、
  有界历史、显式 `finish` 与 `max_rounds` 终局。
- 自由文本 Agent contract、catalog 内模型选择、两比特关系、唯一 Output Agent
  和全节点可达 Output 的约束。
- AgentRuntime 的有向上游传递、fan-in、并行独立 block，以及有限两阶段双向
  通信；Output Agent 的最终输出是唯一 task answer。
- SkillFlow 式本地 Qwen3.5-9B/SGLang Director 路径、既有 LoRA 的
  evaluation-only load → model-list → canary 边界。
- validation trajectory 采集；训练资格仍由 `split == train` 独立门禁，
  validation 不可进入 GRPO。
- HotpotQA normalized EM + SkillFlow token F1；F1 仍是既有 reward 字段，EM
  作为并列评测指标。
- Director/Executor 的生成 seed、真实 request ID、attempt count、token、latency；
  保存实际 rendered messages、upstream、peer draft、所有 Agent 输出、runtime
  outputs 与 block completion order。
- HotpotQA 旧对齐文件从其完整 `context` 恢复十篇 passage；后续数据准备也不再
  截断每篇前 300 字符。ground truth 与 supporting facts 没有进入模型输入。
- 固定 128 条 project-held-out validation、one-call local Direct paired baseline、
  AgentGraph 可续跑采集、严格失败计 0、Wrong Demo 与报告入口。
- Runtime 从既有唯一 `output_agent_id` 派生 Intermediate/Output 身份；只有
  Output 节点拥有 task-level `<answer>`，不增加固定角色或 workflow 模板。
- Director continuation 改为可定位问题驱动；已有执行候选后不再无条件展示新的
  weighted preferred model，但 `finish` 仍由 Director 显式选择。
- `normal` / `upstream_masked` 双臂通信诊断保留原始 upstream receipt，只遮蔽
  模型实际看到的跨 Agent 内容；诊断记录结构上不可进入 GRPO。
- 已冻结 raw candidate 640--671 的 32 条 untouched HotpotQA confirmation，且与
  既有 128 validation 和 512 training 候选无重叠。

## 2. 已验证模块

- 固定 development-128：128/128 完成全链，AgentGraph EM/F1 为
  73.44/81.62；复用的同题 Direct 为 72.66/82.08。
- 一次性 untouched-32：32/32 完成全链，AgentGraph EM/F1 为
  71.88/83.62；同题 Direct 为 78.12/87.77。该集合在看到结果后没有被重跑或
  用于调 prompt/search space。
- 独立 communication diagnostic：16 个冻结多 Agent 图、32 个执行 arm 全部
  完成；遮蔽 upstream 后 EM 不降、F1 反而增加 1.11 个百分点。它证明遮蔽入口和
  记录边界工作，但没有提供下游行为依赖 upstream 的证据。
- 全量 unit test：170 passed；本轮修改文件的 Ruff 检查通过。
- 固定数据：128 条、128 个唯一 task ID，输入标记为 `full_passages_v1`。
- SGLang base model `supervisor_theta` 可服务；既有
  `theta_smoke_step_000001` adapter 加载成功且 canary 通过。
- 两条真实 Hotpot canary：两条均完成 3 个 Director turn、1 个 Executor call、
  显式 finish、有效 evaluator receipt 和可定位 Output Agent inbox。
- 首次 canary 暴露的 native SGLang seed 字段不兼容已最小修复：SkillFlow
  OpenAI 边界的 `seed` 在部署的 SGLang 0.5.15 `/generate` 中对应
  `sampling_seed`。修复后只补采失败的 AgentGraph 条目，没有重调成功 Direct。

Round-01 运行证据保存在 `artifacts/hotpotqa_round_01/`；新的复评、通信消融和
untouched confirmation 证据保存在 `artifacts/hotpotqa_training_ready_step0/`。

## 3. 仅预留/本轮关闭

- MACE feature/bandit、Bayesian posterior/EVSI、Skill 发现/验证/生命周期接口仍是
  后续方法模块；本轮不注入 Skill。
- GRPO、backward、optimizer、LoRA 更新与 policy publish 全部关闭。本轮仅加载
  已存在的 step-1 adapter 进行推理。
- 其他六个数据集及其正式 environment/harness 本轮不进入。

## 4. 已知问题

- Director 或 API 在 terminal trajectory 创建前异常时，collector 仍可能无法
  保存已完成的 partial turns；runner 会独立保存 task/stage/error/timestamp，
  但失败调用的 token/latency 未必可得。
- Direct 是绕过 Director/Canvas/AgentGraph 的一次本地 Qwen 调用，但复用了项目
  Agent gateway 的轻量 node system wrapper；它是本项目 paired one-call baseline，
  不等同于论文公开 Qwen Direct 的原始 prompt/protocol。
- 项目 validation 来自原生 HotpotQA train 候选中的固定 held-out 128 条；论文
  reference 只能旁注，主要比较必须使用这批样本上的 Local Direct。
- persistence replay 的旧 legacy revision 假设尚未用于本轮 runner；本轮续跑以
  exact task/condition/VersionBundle 和 EvidenceStore payload 为准。

## 5. Stable Zero

**Stable Zero 仍达到；当前尚不放行 GRPO。** 工程链、版本冻结与可观测性已经形成
可复现的 `step_000000 / training_ready_step0` 基线，但通信遮蔽没有显示多 Agent
结果依赖 upstream，且 untouched-32 的 AgentGraph 成对低于 Direct。最终阻塞项和
判定以 `TRAINING_READY_STEP0_REPORT.md` 为准。
