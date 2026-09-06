# HealthBench Professional v2.35：可选医学工具

本版继承 v2.34（源码提交 `412886c`）的统一 AgentGraph，仅扩展 Tool Adapter
及其必要的执行配置。没有训练、GRPO、LoRA 更新或 MACE/Bayesian/Skill 演化。
目前完整 525 题仅完成 prepare-only；下列兼容测试不是 HealthBench 得分。

## 已实现

| 能力 | 实际实现与边界 |
| --- | --- |
| 医学搜索 | 原 MedRAG BM25 + NCBI PubMed 聚合搜索原样复用；不是通用互联网搜索。 |
| 本地知识检索 | 原 MedRAG/textbooks 冻结的 125,847 个片段与 BM25 检索；没有用 benchmark 或 rubric 构建数据库。 |
| 读取资料 | 按实际 MedRAG document ID、PubMed PMID、DailyMed SETID 读取。PubMed 返回完整摘要而非论文全文。每页最多 16,000 字符，返回 next_offset 和截断标记。 |
| 药品查询 | NLM DailyMed 药名检索后实际读取 SPL 标签正文；保留产品、来源、版本及日期。不等同于完整的药物相互作用检查器。 |
| 计算 | 直接复用 SkillFlow calculator 的已有移植；不预置临床公式、不代填缺失患者信息。 |

每个已通过工具兼容测试的模型，可以被 Director 选为普通 reasoning 节点或
ReAct 节点。ReAct 节点可选单个工具，也可选完整工具集合；不强制先搜索，
不要求工具必须全部调用。保持自由 contract、每节点模型选择、有限双向通信、
唯一 Output 和增量 Canvas 执行，不增加固定医疗角色或工作流。

原执行器只支持单个搜索工具，原 Runtime 的 action domain 只列单工具配置。
本版复用同一个 Action–Observation 循环，仅新增显式多工具配置的能力校验。
每次 Agent 调用仍最多 6 个回合、3 次实际工具调用；两次成功搜索后仍可读取
已找到的资料或调用计算器。重复请求被拒绝，历史 receipt 不因 continuation 清零。

完整 Tool receipt 保存在 trajectory 中。传给下游及 Director 的证据仍有明确
的上下文预算，不宣称无损无限传递；来源、版本、分页、截断标记均保留。
新资料读取按文档版本及分页去重，避免已有摘要把后续新页面误删。
计算结果不会被标为医学文献证据。最终回复为完整 assistant response，非短答案。
rubric、医生参考回复和 evaluator 状态不进入解题 Agent 的工具输入。

## 已验证与限制

- 工具后端：15 项离线测试及 8 个子测试通过。
- 可选 ReAct、证据投影及旧搜索兼容：66 项定向/回归测试通过。
- Runtime 接线、评测协议、V4 反馈和顺序启动：90 项定向/回归测试通过。
- DailyMed 实际连通测试：一次药名检索并读取一个真实 SPL 标签成功，含版本；不是临床答题评测。
- 远程模型实际兼容探测共 6 次生成，无重试或模型替换：DeepSeek-V4-Flash、MiniMax-M3
  均完成 calculator → Observation → complete；Qwen3.5-Flash 首次计算成功，
  第二回合却重复相同请求，未完成 canary。因此 Flash 暂保持 reasoning-only，
  **不能声称四个模型都已通过完整 ReAct 测试**。
- 本地 Qwen3.5-9B 沿用已验证的 SGLang 工具执行能力；Director 始终是本地 9B。
  所有模型保留原 thinking 配置。简单计算器 canary 不代表医疗推理能力已验证。

## 完整重评测

新条件：`healthbench_professional_optional_clinical_tools_v2_35`。
固定公开 test set 525 题，sample/order/seed、模型生成参数、20 个 Director 回合、
并发 4、任务超时 900 秒与 v2.34 一致，工具和 Agent 工具权限单独版本化。
继续用官方 HealthBench Professional reference evaluator，报告 raw overall score
和主指标 length-adjusted overall score，以及有效数、FINISH、超时和 provider/grader 失败。

因工具条件改变，**不得复用 v2.32 Direct 当作新版同条件对照**。
现有评测程序会先生成并评测同工具条件的 Direct，再评测 AgentGraph；已有合规结果
断点续跑，不重复已完成调用。异构 AgentGraph 与单模型 Direct 仍是描述性比较，
不是仅改变拓扑的因果消融。

按用户最新要求，尚未运行的 v2.34 不再另做一次完整评测；v2.33 当前批次保持冻结，
其正常收束后由原有监控/顺序启动脚本启动 v2.35。不会同时开启两个全量 GPU 评测。
此公开集合已用于开发，因此结果属于开发后重评测，不是未使用过的 held-out 泛化估计。

当前**没有 v2.35 实测 HealthBench 分数**，不得预测提升。评测结束后按真实低分案例
区分检索、实体/适用条件、证据传递、推理、终局与 provider/grader 失败，不预设医疗答案。

配置：`config/evaluation_healthbench_professional_optional_clinical_tools_v2_35.yaml`。
启动配置：`config/healthbench_v233_to_v235_handoff.json`。
运行证据：`artifacts/healthbench_professional_optional_clinical_tools_v2_35/evaluation/`。
兼容探测原始记录：`artifacts/model_capability_canary/healthbench_optional_clinical_tools_v235.json`。

上游来源详见 `docs/source_map.md`、`docs/adaptation_log.md`。
DailyMed 官方接口：[药名检索](https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm)、
[SPL 标签读取](https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_setid_api.cfm)。
