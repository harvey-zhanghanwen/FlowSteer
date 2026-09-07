# HealthBench Professional：逐题 rubric 可见实验与独立备份

## 实验口径

用户在暂停旧评测后明确要求按逐题 rubric 构建资料库，并重新运行525题。
本条件为 **rubric-aware evaluation / oracle-information condition**：
原测试题的评分标准在解题阶段可见。这是有意改变信息条件，不是未见rubric的
官方盲测，也不是独立测试集泛化成绩。即使得分上升，也不能单独归因为编排改进。
后续Direct对照必须使用同一逐题条件库、样本、工具和evaluator；本轮不额外支付
525题Direct调用，也不重复旧25题的评分。新条件与旧条件分别采样、保存。

没有训练、GRPO、backward、optimizer、LoRA、MACE/Bayesian或Skill evolution。
仍使用三条可拒绝的failure.v2候选提示，不声称它们已通过消融或成为ACTIVE Skill。

## 已完成的构建

- 原公开医学证据库v2：4505条，MedRAG3966、PubMed511、Europe PMC26、DailyMed2。
  3446条由短摘录扩为本地真实原文passage；不声称整本书或论文全文完整。
- 独立逐题rubric条件库：525题、1135条criterion；最长单题补充上下文2089字符。
  正分项标为`satisfy`、负分项标为`avoid`，零权重项若有则标为`informational`。
- 直接投影官方case中的真实criterion和points；没有生成参考回答或伪造临床资料。
  physician_response、其余评测metadata没有导入。构库付费模型/评分调用均为0。
- rubric作为明确标记的实验补充上下文进入Director与Agent输入；原健康对话保持原文、
  原角色和顺序。原始Task ID、评分路由与官方grader的原对话完全不变。
- rubric不是外部医学证据，不进入MedRAG/PubMed/药品库，也不作为患者说过的话
  进入conversation知识库。需核实时，Agent仍可通过现有ReAct工具查询医学来源。

## 复用与最小适配

| 模块 | 来源与修改 |
| --- | --- |
| AgentGraph/Canvas/Runtime/轨迹/预算 | 保留原SkillFlow/FlowSteer适配；不改自由contract、关系、逐步执行、唯一Output和FINISH边界。 |
| `healthbench_rubric_context.py` | 用户新授权的信息条件适配。复用官方case loader、TaskRecord和对话renderer；仅构库与按Task ID连接上下文，不另造执行器/评分器。 |
| `healthbench_professional_adapter.py` | 原renderer/parser默认行为不变；新增显式`rubric_aware_context`扩展字段。官方原始数据schema不改。 |
| `openai_gateway.py` | 实验上下文并入现有单条system消息，避免多system消息的provider格式问题。 |
| clinical ReAct / knowledge tools | 原名首查和患者对话索引只读原对话；rubric不冒充医学证据或患者消息。 |
| completion benchmark runner | 复用两arm共同的`_select_tasks`冻结入口；续跑比较真实question，禁止混入旧条件。manifest/报告记录rubric可见，官方评分不变。 |

## 备份边界

1. 原rubric不可见修复版：`feature/healthbench-public-task-repair-20260907`，
   commit `4d8e5f3d01501bc59e00930c03a2e14b443f004d`，已推送至`backup`。
2. 本实验：`feature/healthbench-rubric-aware-full525-20260907`，基于上述提交，
   包含新增适配、配置、测试和说明；最终提交与push状态在本次交付中给出。
3. 更早已暂停的条件：`feature/healthbench-failure-skills-full525-20260907`，
   commit `1385368013eeb9f7f78ee868a0be9f086bea488b`，其25条结果独立保留。

远端：`backup`，`https://github.com/harvey-zhanghanwen/FlowSteer`。
不改main、不强推、不覆盖旧条件；凭据不进文件、remote URL或提交。
大型外部库、模型、运行中间文件和私有逐题rubric原文保留本机，不复制进Git。
Git保留构建脚本、精确配置和本机资源/复现入口。异机运行仍需同样的数据、
模型、上游源码及provider配置，不声称只clone即可离线重放全部API。

## 复现入口

工作树：`/ssd1/iclr/1/.tmp/FlowSteer-healthbench-failure-skills-full525`。

```bash
python scripts/healthbench_rubric_context.py \
  --private-cases data/healthbench_professional_official_v1/private_cases.jsonl \
  --output artifacts/healthbench_rubric_context_v1
python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_rubric_aware_full525.yaml \
  --collection-arm agentgraph --prepare-only
python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_rubric_aware_full525.yaml \
  --collection-arm agentgraph
```

构库只对新目录执行；已有冻结库直接复用，不覆盖。
复用GPU6现有8026端口Qwen3.5-9B Supervisor、现有混合Agent池和thinking。
保留原525题顺序、seed20260825、并发4、每题900秒/最多20轮；不替换Director。
配置依赖原公开证据库和CPU构建的4538窗口BGE语义索引：

- `artifacts/healthbench_public525_evidence_corpus_v2/manifest.json`
- `artifacts/healthbench_public525_semantic_v2/manifest.json`
- `artifacts/healthbench_rubric_context_v1/manifest.json`

新输出：`artifacts/healthbench_professional_rubric_aware_full525/evaluation/`。
manifest、selected tasks与trajectory保留实验信息；不复用旧rubric-hidden回答。
评分沿用OpenAI reference evaluator `652c89d@1`、`gpt-5.4-2026-03-05`
与原生raw/length-adjusted aggregation，不夹紧负分或超100%分值。
缺失/超时不伪造有效评分；报告需同时列有效数、终局和运行失败数。

## 验证与当前指标

- 条件库/对话边界、共同冻结入口、原名检索、知识库及completion runner测试：
  **80 passed**；无模型/HTTP/评分调用。
- 另有gateway、官方grader及v4产物交接回归：**65 passed、25 subtests passed**。
- 正式525题prepare-only通过，状态`prepared`，指标为空；这不是已经跑完。
- 运行前先提交并推送源码；启动/采集状态以独立run manifest为准。
- 不预报rubric-aware分数，不将暂停的25题6.31%/6.94%当作525题成绩。

## 限制

条件库不是独立验证过的医学答案库；criterion可含错误行为描述，必须区别正负方向。
没有强制医疗role或固定拓扑，也未解决所有模型语义错误与临床矛盾。
旧版任务完成/证据交接修复与新增rubric信息同时改变，一次差值不能识别各自贡献。
数据库、候选提示、编排的影响需后续同信息条件消融，不能凭本轮得分推断。
