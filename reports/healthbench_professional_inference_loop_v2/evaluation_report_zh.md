# HealthBench Professional inference-loop v2 评测报告

## 结论

本轮只修改推理与编排控制，没有训练，没有执行 GRPO、backward、optimizer update、LoRA、MACE、Bayesian update 或 Skill evolution。

HealthBench Professional 不使用 Accuracy、EM 或 F1；本报告采用 OpenAI simple-evals HealthBench Professional reference protocol 的 `overall_score_length_adjusted` 作为主指标。

525 个公开测试任务的 Direct 和 AgentGraph 回答都已生成。AgentGraph 为 **525/525 显式 `FINISH`**，`max_rounds=0`，terminal failure 为 **0**。旧条件为 503/525 显式 `FINISH`、22 个 `max_rounds` terminal failure，因此本轮修改消除了已观察到的终止失败。

评分阶段后段出现 grader 配额不足：Direct 有 510/525 个 evaluator-valid receipt，AgentGraph 有 488/525 个 evaluator-valid receipt。未评分条目没有被伪造成有效分数；因此当前不能宣称已得到完整 525 题的最终官方主指标。

## 当前可报告指标

| 口径 | Direct | AgentGraph | AgentGraph - Direct |
| --- | ---: | ---: | ---: |
| evaluator-valid 子集的 length-adjusted score | 17.8081%（510 条） | 19.7159%（488 条） | 不作直接配对差值 |
| 固定 525 分母的严格下界 | 17.2993% | 18.3264% | **+1.0271 个百分点** |
| 双方同题均 evaluator-valid 的 complete-case 配对 | 18.3629%（473 条） | 20.5584%（473 条） | **+2.1955 个百分点** |

同题 complete-case 配对的未调整 `overall_score` 为 Direct 17.9061%、AgentGraph 22.9469%，差值 **+5.0408 个百分点**。

“固定 525 分母的严格下界”把缺失 grader receipt 计为 0，只能作为保守下界；“evaluator-valid 子集”与“complete-case 配对”都受评分缺失影响，不能替代完整 525 题的最终官方分数。

## 本轮推理控制改动

- 在 Canvas action mask 中提前排除当前 revision 下的无变化、self-loop、cycle 和 validator-invalid relation 候选。
- 保存当前 revision 最近被拒绝的 relation action，避免在图状态没有变化时重复选择同一非法操作。
- 要求 `ADD_AGENT` 的 free-text contract 明确描述责任、输入和输出 artifact，同时保持开放角色与开放 topology，不预设医疗 Agent 模板。
- 对 Tool-free reasoning component 的完全相同 semantic input 启用 task-local execution reuse；本轮记录 4 条 reuse receipt，对应 2 个任务、3 个独立 semantic cache reuse event。
- 只有完整 terminal admissibility 已满足时才把下一步 action mask 收缩为 `FINISH`；没有降低 terminal validator 的要求。
- 在本地 SGLang 明确支持时，为 Direct 与 AgentGraph 同时使用 `repetition_penalty=1.05`，避免改变两条评测条件的对称性。

## 运行结构与截断

- AgentGraph 生成完成：525/525。
- 显式 `FINISH`：525/525。
- `max_rounds` / terminal failure：0 / 0。
- 最终图：520 个单 Agent；3 个双 Agent；2 个三 Agent。SCC condensation 后为 520 个 `single` 和 5 个 `serial_2`；其中 2 个图包含 reciprocal relation。最大结构深度为 2，全部图的 `max_width=1`。
- Agent execution 共 1109 次，其中 `finish_reason=length` 为 4 次；其余 1105 次为 `stop`。
- Canvas rejected action：6 次，分别为 reasoning Agent 配置 Tool 4 次、no-op modification 1 次、未注册 coding execution adapter 1 次；`SET_RELATION` 没有 rejection。
- 4 次长度截断都发生在首轮 `ADD_AGENT`：2 次具有明显重复循环，另外 2 次是正常长文本达到 4096-token 上限；随后的 `SET_OUTPUT` 重执行都以 `stop` 结束，最终输出没有保留明显重复循环。

本轮证据表明，重复生成是部分长度截断的原因，但不是全部原因，也不是 terminal failure 的主要来源：只有 4/1109 次 Agent execution 被长度截断，而全部 525 条 trajectory 都合法 `FINISH`。当前主要质量瓶颈仍是回答对 rubric criteria 的覆盖；同时，Director 仍高度偏向单 Agent（520/525），没有自然形成带分支宽度的 DAG。

## 证据与状态

- 自动聚合报告：`reports/healthbench_professional_inference_loop_v2/evaluation_report.json`
- 配对结果：`artifacts/healthbench_professional_inference_loop_v2/evaluation/paired_results.jsonl`
- 全量原始 trajectory：`artifacts/healthbench_professional_inference_loop_v2/evaluation/evidence/trajectories.jsonl`
- evaluator failure receipt：`artifacts/healthbench_professional_inference_loop_v2/evaluation/collection_failures.jsonl`

自动聚合 JSON 中的 `explicit_finished_count=488` 是从 evaluator-valid AgentGraph materialization 统计的；全量 inference evidence 中的真实显式 `FINISH` 数为 525。两种口径在本报告中已明确拆分。

由于 grader 配额不足，当前 best-profile 仍保留完整收束的 `healthbench_professional_official_v1`，没有把未完成评分的 v2 错标成新的正式最佳条件。
