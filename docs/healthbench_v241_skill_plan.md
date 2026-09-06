# HealthBench v2.41：推理阶段候选编排经验的最小接线方案

状态：源码对照后的方案；本文件不代表 Skill 已接入正式 runner、已执行、已验证或已发布。后续有界任务已实现独立候选配置 helper 并运行 22 项合成单测；未调用模型、评分器或训练。共享 runner/runtime 的生产接线由主线单点修改。

## 1. 方法边界

用户 MD《FlowSteer_MACE_Bayesian_Skill_Design.md》§10.1–10.4 和 §11.2–11.3 要求：Skill 是有适用条件、动作、配对证据及失效边界的规则；候选经验不能因为来自一条成功/失败 demo 就成为 ACTIVE。发布需要独立 problem 的确认数据、预先规定的效应/伤害门槛与版本兼容。公开 test 上的当前调试结果不应当作新 Skill 发布证据。

本轮不训练、不开 MACE/Bayesian/Skill evolution，因此可以明确试验**预声明的、未验证的候选 prompt condition**，不能伪造有效配对数、置信区间或 gate receipt，不能声称已经完成 MD 的 Skill 闭环。

候选只是 Director 可拒绝的提示，不添加医疗答案、题目 ID、rubric 或参考回答；不强制医疗角色、Agent 数量、串行骨架或图形。Director 仍按当前 Canvas 状态自主采样原有合法动作。

## 2. 已核对的真实上游与现有调用链

| 来源 | 已有函数/类 | 可复用部分与边界 |
| --- | --- | --- |
| 公开 SkillFlow `/ssd1/iclr/2/SkillFlow/src/skills/workspace.py` | `SkillWorkspace.retrieve`、`format_skills_for_prompt`、`get_by_id` | 按任务内容/类型检索少量策略并提供原内容；没有项目 MD 所要求的 ACTIVE 统计门槛，不能直接替换本项目 gate。 |
| 公开 SkillFlow `training/environment.py` | `_handle_skill_invoke`、`_auto_inject_best_skill` | invocation 返回 plan/pitfall/constraint，保留实际调用；`_auto_inject_best_skill` 会记录自动注入，这不等于模型自主学会调用。 |
| 公开 SkillFlow `src/skills/format.py` | `GENERAL_SKILL` | 无特定策略时自行分析、按需用工具、提交答案。它没有医学事实或固定多 Agent 流程，不能凭借 `general` 名称认定任务得分提升。 |
| 下游 SkillFlow/SkillEval `/home/test/SKILLEV/skillflow-bayesian-improve-deploy/src/skillev/runtime/skill_library.py` | `require_seed_library`、`SkillLibraryState.from_seed_documents` | 冻结 seed documents 的机制存在；此上游会把 seed 文档置为 active，而本项目 MD 有额外发布门槛，所以不能照搬激活语义。 |
| 同一下游 `runtime/skills.py`、`runtime/full_skill_context.py`、`runtime/bounded_agent.py` | `SkillApplicability`、`model_visible_skill_content`、`FullRetrievedSkillContext`、`BoundedAgent.execute_turn` | 所需工具/适用条件、完整模型可见内容，以及 retrieved/active/invoked 身份分离；不是单靠提示里出现 skill 字样就算 invocation。 |
| 本项目 `src/interactive/skills/{schema,store,retrieval,pipeline}.py` | `SkillRecord`、`SkillStore`、`SkillRetriever.retrieve`、`SkillEvidencePipeline.retrieve_prompt_priors` | 现成正式 Skill 路径：冻结 store → 版本/epoch/gate/applicability → 可拒绝的 `PromptSkillPrior`。本轮保留，不放松。 |
| 本项目 `scripts/train_agentgraph_smoke.py` | `LiveSmokeBackend.from_config`、`_skill_query`、`_visible_skill_priors`、`collect` | 文件名含 train，但 `evaluation_only=True` 的工厂、runtime 和 collector 已供正式评测复用；不调用 `train` 或 `publish`。 |
| 本项目 `scripts/run_joint_qa_mace_skill.py` | `_prompt_condition`、`_arm`、`_paired_probe` | 已有候选条件序列化与调用方法，薄适配到 HealthBench；不运行此联合 QA 实验主程序，它还会做 posterior/证据发布等本轮不允许的工作。 |
| 本项目 `src/interactive/rollout_collector.py` | `_retrieved_skill_ids`、`AgentGraphRolloutCollector` | `application_mode=forced_probe_condition` 不计入 ACTIVE Skill receipt；普通 Skill 必须来自独立 ACTIVE library。保留此隔离。 |

上述属于真实源码定位，不需要新造 Skill runtime、检索器、工具或者训练框架。

## 3. 现有可直接使用的候选入口

`LiveSmokeBackend.collect` 已支持：

- `prompt_priors`：预声明的静态候选条件；
- `stage_conditioned_prompt_prior`：一个按阶段检索的候选条件；
- `forced_probe=True`、独立 `condition_id`、`sampling_schedule_purpose`、`sampling_anchor_ordinal`；
- 既有 `_forced_probe_condition_matches` 的 task family、graph stage、tags、所需 tools、model 判断；
- 既有 `_prompt_prior_exposure_receipt` 同时记录当前 observation 暴露轮次与历史 transcript 保留轮次。

候选字典沿用 `_prompt_condition`：`condition_id`、`application_mode=forced_probe_condition`、`condition`、`action`、`content`、`rejectable=True`。此路径名称是现有代码保守的干预记账；它并不替换 Director 采样出的 Canvas 动作，但会把候选试验轨迹与正常 Skill-on/自然 baseline 隔离，且不能进入 GRPO。

阶段仅支持 `empty_graph`、`construction`、`before_final_answer`。最后一个只说明图的结构校验通过，**不等于**完整回答已完成或 FINISH 可用；不能在这个阶段无条件要求结束。运行时错误也可能发生在结构合法的图上，因此仅在 `construction` 注入修复经验会漏掉这类错误。

若本轮希望“几步编排经验”，最小方案是一个有界、条件化的通用候选条目，其内部描述几个操作步骤；不要新增角色、强制执行链或新的阶段状态机。也可使用已有静态候选列表，但应控制长度，避免每轮重复放入大段经验再次引起上下文超限。

## 4. 可预声明的通用候选内容范围

以下是待验证内容边界，不是声称经过统计验证的 ACTIVE Skill：

1. 出现失败时，保留已经产生的有效结果与来源记录，先辨认错误属于输入、执行、证据格式还是依赖，再修改受影响的局部节点或关系；不要在无变化输入下反复执行相同失败步骤。
2. 下游声明依赖某个结果时，先检查实际信息流方向及收到的 artifact；引用保留来源 ID 和忠实摘录，推理/改写与证据原文分开。
3. 在完整用户要求已被满足且当前 FINISH 确实合法时结束；如仍有明确缺口，则按实际缺口修复或加入协作单元，不因为已有节点数量少而强制扩图，也不因为已有文本就强制结束。

内容应是一般软件协议与编排经验，不写入疾病、试验名、问题答案、数字、样本特征或 grader 判定；不保证医学结论正确，也不对测试题指定检索词。

## 5. 主线必须处理的最小兼容点

`scripts/evaluate_completion_benchmark_round.py::_skill_evaluation_mode` 当前只接受关闭 Skill 或 ACTIVE-only deployment；其 `_collect_graph` 直接复用 `evaluate_hotpotqa_round`，没有候选条件参数的专属正式入口。因此不能仅把 `skills.enabled` 改成 true 就完成本轮要求。

若主线在现有 completion runner 接候选试验，需保持普通正式评测行为不变，并在明确命名的 development-condition 分支薄传已有 `collect` 参数。模型、样本、评分器、预算保持冻结。必须同时在 manifest、resume identity 和报告中区分：候选条件版本、暴露轮次、`forced_probe=True`、`grpo_eligible=False`、`ACTIVE Skill 数=0`、`Skill publication=False`。

特别注意现有报告文案：`skills.enabled=true` 会写出 “Only evidence-gated ACTIVE Skills were retrieved …”。候选分支不能沿用这一句，也不能因为最终得分改善就把 `skill_injection_performed` 当作证据门控 Skill 已发布。报告应明确是“候选编排提示条件试验”，与无候选的自然 AgentGraph baseline 单列，不宣称 learned Skill 增益。

如果主线暂不增加该明确分支，可先通过既有 backend.collect 做独立候选 development 试验；正式全 525 自然评测保持 skill_off。不得悄悄把候选 forced-probe 结果混入最高正式架构口径。不要为了取得 ACTIVE 状态重切当前公开 test、编造配对证据或临时启用训练/后验更新。

## 6. 验证与报告（共享 runner 部分由主线执行）

- 关闭候选时，原配置、collector 和官方 evaluator 行为不变。
- 候选进入模型 observation，但不进入 `active_skill_ids`、`retrieved_skill_ids` 或伪造的 `invoked_skill_ids`。
- Director 可以忽略建议、保留单 Agent、采用自由图形；不存在固定角色、固定拓扑或额外动作 mask。
- 当前 observation 与历史保留暴露分开记录；仅看到建议不等于采纳，也不等于新增调用。
- 仅有一条完整 525 的新条件结果时，可报告该条件绝对评分及完成率，不能把代码修复和候选共同改变后的差值单独归因于 Skill。
- 超时、provider/context/解析失败、FINISH 缺失与 evaluator-invalid 分开；未评分样本不得伪装为官方 grader 判零。

## 7. 已完成的独立 helper

新增 `scripts/healthbench_candidate_skill_profile.py`、`config/healthbench_candidate_skills_v241.yaml`、`tests/unit/test_healthbench_candidate_skill_profile.py`。

- `load_candidate_skill_profile(path, *, run_config, dataset_key='healthbench_professional') -> dict`：复用现有 YAML loader（不展开环境变量），检查本轮无训练、无 LoRA/optimizer、无 exploration/ACTIVE Store/policy publication。
- `validate_candidate_skill_profile(profile, *, dataset_key='healthbench_professional') -> None`：限定三个短 candidate，无效应/gate 或答案字段，明确 task family、阶段和触发条件。
- `build_candidate_prompt_priors(profile, *, dataset_key='healthbench_professional') -> tuple[dict, ...]`：直接沿用既有 `forced_probe_condition` wire，不带 ACTIVE `skill_id`，不触发 API。
- `validate_candidate_skill_run_config(config) -> None`：供主线配置接线复用。

单独运行本文件的合成测试：22 passed；只出现已有 Pydantic 配置弃用警告。没有运行其他测试或模型评测。共享 runner、factory、Director、SkillStore 均未由本子任务修改。

本子任务到此结束。冻结配置、模型/API 调用、最终整合均由主线负责。
