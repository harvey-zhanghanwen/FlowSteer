# HealthBench Professional v2.27：最高完整成绩的证据归档

> **SOURCE_SNAPSHOT_INCOMPLETE：不是已恢复的可执行 v2.27 源码快照。**
> 已知基础源码和可保留配置/报告已归档；历史运行时未提交的源码差量缺失。
> 不得用本备份提交声称重新得到了26.66%的成绩，也不得自动运行评测。

## 最高成绩与口径

完整公开test的AgentGraph 525/525 evaluator-valid、525/525 FINISH，
terminal failure=0。官方主指标overall_score_length_adjusted为
**26.6597763574%**，原始评分overall_score为**28.1879043574%**。
这些是2026-09-03既有运行的实测结果，不是本备份新增评测。

condition:
`healthbench_professional_mixed_all_thinking_v2_27_full525_bestbase_provenance_scope_quality`。
reference evaluator为OpenAI simple-evals HealthBench Professional，
来源revision `652c89d`，实际grader为`gpt-5.4-2026-03-05`。
Direct为不同协议，且有27条operational/evaluator failure；它们不属于本轮
AgentGraph的525条有效终局，不以其差值作严格配对因果结论。

排除尚未完成的525题批次、5/20题开发子集、单题满分及Direct。
项目旧best_profile指针仍指official_v1；本归档没有修改默认运行条件。

## 本分支保存了什么

分支：`backup/healthbench-v227-best525-evidence-20260906`。
目标remote：`backup`，地址
`https://github.com/harvey-zhanghanwen/FlowSteer.git`。

- 分支父提交为运行manifest记录的
  `f680d137cf2187045682c439bf5a7a032cb1d1da`。其完整Git树原样继承，
  保存可确定的基础源码，包括当时已提交的Canvas/runtime、Director、
  model interface、task adapter、evaluator、runner及来源说明。
  **这只是已知baseline，不是完整v2.27实际工作树。**
- `config/`保存目前保留下来的v2.27 full525 YAML及其4个直接配置依赖，
  位于独立归档目录，不替换根目录运行配置。主配置首次入库为后续
  `a4bf83217002cd653ac0eea7c9b0ce34f9df35c5`；原manifest没有
  resolved-config快照，不能宣称已证明所有字段在整次历史运行中完全不变。
- `reports/evaluation_report.json`和`.md`为原始必要成绩报告，未重评或改分。
- `receipts/run_manifest.json`保存任务ID、分母、运行身份和协议记录。
- `receipts/preflight_receipt.json`保存已有模型/评估器准备回执。
- `receipts/dataset_manifest.json`保存已准备数据的来源和切分说明。
- `backup_status.json`为机器可读的归档范围、分数和未解决缺口。

新增文件仅在本目录。没有复制后续v2.28–v2.40运行源码，也没有将别的
工作树未提交内容带入本备份；既有基础Git树中的共享代码未改写。

## 为什么暂时不能精确复现

1. 原run_manifest的git_start/git_end都只记录f680d13；没有未提交源码patch。
2. 该baseline不包含v2.27配置，且不具有配置启用的minimal-neutral.v17和
   public_text_quality_v1等实现，直接组合baseline和归档配置并非原版。
3. 下一个可用源码提交a4bf832同时纳入v2.16–v2.32的大量修改，不能凭猜测
   回删成v2.27；本分支没有这样操作。
4. 本次窄范围检查没有发现可恢复上述差量的已有提交、独立worktree或源码存档。

恢复完整执行入口前，需要找到**真实的v2.27运行时源码差量或完整文件快照**。
找到后应在独立恢复提交中补齐并核对接口；不能通过改名或放宽评估器解决。
目前仍能恢复基础源码、查看原配置、任务分母和实际成绩，但不能保证原样重跑。

## 外部复现依赖

- 本地Qwen3.5-9B及tokenizer；历史服务端口8015、GPU0，具体参数见配置。
- Agent模型池：本地Qwen、Qwen3.5 Flash、DeepSeek V4 Flash、MiniMax M3。
  历史目录仅本地Qwen提供此版本ReAct工具能力，其他条目为reasoning。
- MedRAG textbooks资源和NCBI PubMed EUtils；本版不是v2.39的12工具条件。
- 官方525条test及evaluator-private cases；simple-evals来源与grader解释器。
- 模型、tokenizer、数据库、完整trajectory、公开对话和原始rubric、凭据
  未加入本次提交。其既有本地位置见配置/manifest，未下载或重新生成。

## 本次操作边界

只做独立分支和证据备份。不训练、不评测、不调用解题模型或grader、不改main、
不force-push、不改写历史。Git认证仅由临时进程凭据提供，不写入本目录、
remote URL或提交。是否push成功以执行后的真实回执为准。
