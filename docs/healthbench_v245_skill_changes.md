# HealthBench v2.45：提前执行协议修复，而不是继续加长职责

## 来源和修改范围

仅修改候选 Skill profile 和独立开发评测配置；执行代码仍为 v2.44/v2.43
runtime。SkillFlow `src/skills/format.py::SkillEntry` 的 trigger/plan/pitfall/
constraint、`src/skills/workspace.py::format_skills_for_prompt` 和
`training/environment.py::_handle_skill_invoke` 提供策略表达依据；继续复用
项目 `healthbench_candidate_skill_profile.py`、`LiveSmokeBackend.collect`
与 FlowSteer `InteractiveWorkflowEnv.step` 派生的逐编辑执行/反馈边界。
不新建 Skill runtime、不修改 action mask 或终局准入。

用户 MD §§10.1、10.3、11.3 要求策略可拒绝、建议可执行、独立确认后才能
发布。本版仍为手工预声明 candidate，不是 ACTIVE 或 learned Skill；没有
配对因果增益、训练、MACE/Bayesian 更新或 Skill evolution。

## v2.44 已观察问题与本版假设

- Barrett 首个失败 invocation 已包含多次协议错误；第一次 MODIFY 仍只改
  医学职责。随后 `_repair_exhausted_agent_ids` 将其从 MODIFY 域剔除，
  v2.44 所建议的“再次失败后换模型”可能错过唯一合法机会。本版建议在
  首次收到重复协议错误时就考虑切换当前准入模型，不等待另一次无效改写。
- 三个超时案例中新 Agent 多为失败根节点的下游，图变大却没有可执行的新
  修复。候选明确要求新增修复单元在实际合法域内能够独立于故障 prerequisite
  执行；不能声称一般 free-role 跨节点接管已自动继承原始 receipts。
- 已有足够输入证据的综合节点可选择已有 reasoning profile；不是每个节点
  都需要再次进入 ReAct 的动作/证据 JSON 生成循环。缺证据时仍需 Tool，
  不通过关闭检索或吞掉证据校验强行 FINISH。
- 缩短 candidate 文本及建议的 contract，但不压缩必需临床内容、来源或
  不确定性；不向候选加入样本 ID、疾病答案、rubric、参考回复或指定查询。

以上是可检验的修复假设，不是保证。Director 可能不采纳；采纳也未必提高
官方分数。已知 sink-only Output 输入域、修复耗尽域和 scope guard 的
词面误判仍存在，本轮没有修改它们，也不声称 Skill 消除了这些硬限制。

## 固定条件和验收

同五个已观察开发样本、同 seed/模型池/工具/evaluator、并发 4、Agent
360 秒、任务 900 秒。只跑 AgentGraph，不重复 Direct，不新增 canary
或模型探测，不发布权重。单独 condition 与 artifact 目录，不覆盖 v2.44。
先检查候选 schema 和 prepare-only，再执行一个五题批次。

完整五题只要有缺失，严格总体评分仍为 N/A；逐题比较 raw 与 length-adjusted，
同时报告 FINISH/有效数。五题已经参与开发，不充当 held-out 泛化或 ACTIVE
发布证据。全 525 不在本候选开发批次中自动启动。
