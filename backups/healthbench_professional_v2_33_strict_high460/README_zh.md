# HealthBench Professional v2.33：较高strict评分、460/525有效的独立备份

**本分支保存v2.33源码、配置及460/525评分快照；不表示完整525题评测已完成。**
它与v2.27完整运行证据分支分开保留，不互相覆盖、不混为同一最高成绩口径。

## 成绩快照

原报告时间：2026-09-06T04:39:03.151664+00:00。
固定525分母，AgentGraph evaluator-valid=460、FINISH=460，
terminal failure=0、max_rounds=0，另有65题无有效评分。

| 指标 | 原报告保存值 |
|---|---:|
| strict overall_score_length_adjusted | 27.455919823062575% |
| strict overall_score | 29.521328623062576% |
| 460条完成子集length-adjusted | 31.33556066762576% |
| 460条完成子集raw | 33.692820711104027% |

原strict值高于v2.27完整525题的26.6597763574%/28.1879043574%，
但v2.33此报告只有460/525有效，不能替代“完整运行最高”的v2.27。
保留原报告聚合口径，不重新评测、不将未完成题补为合法医学评分。
HealthBench采用rubric加权评分，不是二元Accuracy、EM或F1。

## 65题运行缺口

- 53题：collect TimeoutError。
- 3题：Canvas actions must be a non-empty sequence。
- 9题：terminal evaluator healthbench_grader_error。

这65个task ID互不重复，来自该时点report内实际collection failure记录。
逐题ID、stage、时间和原错误见
`receipts/agentgraph_failures_460_snapshot.jsonl`。
原报告operational_failure_count=117是Direct或Graph任一失败的题目联合数，
不是Graph失败65的替代值；failure_types中的51也不是全部Graph缺口。

## 源码与配置边界

- 本分支从完整v2.33源码提交
  `3b18f7a866b189254b387dd2ccc406888744b9ad`直接创建，保留其完整Git树。
- 主配置：
  `config/evaluation_healthbench_professional_mixed_all_thinking_v2_33_full525_evidence_context.yaml`。
- 数据准备/registry、mixed-authoritative-thinking v7模型目录、grader v1目录、
  Canvas/runtime、Director v19、evidence communication v3、runner/evaluator、
  测试、source_map/adaptation_log均来自该已有提交，未手工重写或回退。
- 没有导入v2.34、v2.35或以后修改，也没有修改其他工作树。
- 历史架构说明仍在
  `reports/healthbench_professional_mixed_all_thinking_v2_33_full525_evidence_context/architecture_report_zh.md`。
  其中“尚无正式成绩”是架构提交时的状态；本次附加评分报告描述其后460条时点。
- 原配置已说明公开test参与过开发，这不是未接触的held-out泛化分数。
  Direct来自独立旧协议，差值仅是描述性对照，不是同条件配对因果收益。

## 保存的报告和评分证据

1. 原 `reports/.../evaluation_report.json`、`.md`原样归档；这是现存460条
   时点的汇总，不把文件中的completed_at解释成全525题完成。
2. `receipts/agentgraph_score_rows_460_snapshot.jsonl`保存原paired_results中
   525条评分字段投影，恰460条available/valid。字段顺序：
   `task_id, available, valid, explicit_finish, overall_score,`
   `overall_score_length_adjusted, evaluator_version, termination_reason`。
   仅汇总已有分数即可对应原strict数值；没有调用grader。
   此投影不含问题、答案、rubric或Agent prompt，原始约54MB paired文件留在原工作树。
3. `receipts/later_run_manifest.json`是04:40:37Z开始的**后续续跑**记录，
   status仍为agentgraph、completed=505、failed_attempts=79。
   **它不是460条报告的同一时点manifest，不能混合使用。**
4. `receipts/preflight_receipt.json`同样来自后续续跑覆盖后的准备记录，不能
   冒充04:39时点preflight。没有找到对应时点原manifest时不补造。
5. `backup_status.json`记录上述数值、范围、原件位置及区别。

本次不根据后续505条轨迹重算或替换用户指定的460条评分快照。

## 复现资源

Git保存代码、配置和必要报告，不打包模型权重、tokenizer、MedRAG数据库、
数据对话/私有rubric、完整trajectory或凭据。原配置使用外部路径及v2.32
Direct控制四个文件；恢复运行环境时须另外提供，不能声称模型和数据已随Git打包。
该限制不同于v2.27的未提交源码缺失：本分支完整保留了已提交v2.33源码。

## 分支与操作

分支：`backup/healthbench-v233-strict-high460-of525-20260906`。
remote：`backup` → `https://github.com/harvey-zhanghanwen/FlowSteer.git`。
v2.27归档仍在`backup/healthbench-v227-best525-evidence-20260906`，未修改。

本次只备份：不启动/恢复评测或训练，不调用解题模型/grader，不改main，
不强推、不改写历史。凭据只通过临时进程环境提供，不进入文件、URL或提交。
推送是否成功以执行后的真实push回执为准。
