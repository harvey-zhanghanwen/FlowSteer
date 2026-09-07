# HealthBench Professional：失败经验候选 Skill 与全量复跑

## 版本与成绩口径

用户选择的运行源码为 `c5ae00310ca642ab22ac51ee7375f128365ae19b`；本分支从
报告提交 `619c8347677918a402badef03e88367d8a5cb2ba` 建立，保留相同执行链路。
其历史成绩仅为**单题 raw 50.00%、length-adjusted 24.70424%**，不是525题均分。
后来的 contract-artifact v4 单题 raw 为0，本轮没有采用或覆盖该版本。

备份分支：`feature/healthbench-failure-skills-full525-20260907`。
远端：`backup`，`https://github.com/harvey-zhanghanwen/FlowSteer.git`。
本报告建立时新运行尚未完成，不预填分数。

## 本次注入的三条候选经验

| 观察到的失败 | 候选策略 | 保留的边界 |
| --- | --- | --- |
| 非检索文本产物被医学 evidence schema 拒绝，反复提交仍失败 | 根据交付内容选择兼容的执行模式和产物协议，保留有效证据后修复节点 | ReAct仍可独立选择；没有修改或掩盖v3接口限制 |
| 消费节点收到“已完成/已检查”的说明，而非它需要的原文或结论 | 检查实际输入，将必要产物与具体修正传给下游，保留出处和限定条件 | 不要求固定角色、链式或非链式模板 |
| 标题/承诺被当作完整答案；检查发生在最终产物产生之前 | 对照公开任务检查实际Output；根据剩余时间修复具体缺口，充分时FINISH | 不读rubric、不预置答案、不强制增加Verifier |

三条通过既有 `prompt_priors` 接口呈现给Director，可拒绝；替换旧三条，不叠加扩张。
候选状态为 `candidate`，并非已验证 `ACTIVE` Skill。没有训练、权重更新、
MACE、Bayesian、Skill evolution 或自动发布。

## 冻结条件

- 数据：官方公开test的原顺序525题，每题1条AgentGraph轨迹。
- 数据已用于开发及公开问题语义检索建库，准确标为开发后复评，不是未使用的测试集。
- Director：本地Qwen3.5-9B，GPU6现有SGLang，端口8026；节点自主从现有模型池选择。
- 保留thinking、生成参数、20轮上限、900秒任务预算、并发4和v3信息交接配置。
- 复用现有外部证据库/BGE语义索引与医学工具；不新下载、不新建答案数据库。
- 官方reference evaluator与grader不变；rubric只在评分阶段可见。
- 本轮只采集AgentGraph，不重复Direct、canary或既有已完成模型调用。
- 后续恢复必须使用同一配置和manifest，已有合格轨迹按既有checkpoint逻辑复用。
- 记录原生raw、length-adjusted、有效数量、FINISH、失败/超时和完整私有trajectory；
  无效评分保持N/A。完成子集均分不得冒充525题均分。

## 恢复入口

工作目录为本分支worktree。复用本机已有环境文件，不将凭据写进代码或报告：

```bash
set +x
set -a
source /ssd1/iclr/1/FlowSteer/.env
set +a
export FLOWSTEER_SUPERVISOR_PORT=8026 FLOWSTEER_ROLLOUT_GPU=6
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
/ssd1/iclr/gpf/venvs/skillflow/bin/python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_failure_skills_full525.yaml \
  --collection-arm agentgraph --prepare-only
# 首次真实运行/原条件恢复时移除 --prepare-only；不要并发重复启动。
```

本地数据链接复用 `/ssd1/iclr/1/.tmp/FlowSteer-healthbench-v1/data/healthbench_professional_official_v1`；
证据库及索引的绝对位置已在配置明确记录。不将链接、rubric、模型、凭据及大型运行数据提交。

本轮独立过程目录：`artifacts/healthbench_professional_failure_skills_full525/evaluation/`；
最终采集汇总：其中的 `run_manifest.json`，以及
`reports/healthbench_professional_failure_skills_full525/evaluation_report.json`。
完整输入输出和evaluator receipt保留本地，不进入解题提示词或候选Skill。

## 验证与归因限制

24项离线定向测试已通过，核对配置保持、525题选择和三条可拒绝候选接口。
prepare-only已冻结525题，没有调用模型。正式评测启动后以同目录manifest记录状态。
未进行候选开/关配对消融，因此结果即使提升，也不能单独归因于Skill。
本轮不根据运行中的低分再修改候选或冻结条件。
