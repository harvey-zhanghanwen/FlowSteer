# HealthBench v2.48：新五题开发评测，修复证据协议和漏答

## 版本与选择

父版本v2.47源代码f42e0575c7944dfca52c4d85e7dd35648945c965，报告6c8e75a。
上一轮4/5有效、1题900秒超时，完整五题原始/长度调整均分N/A。
用户本轮明确改为另外五题：按官方test.jsonl的文件顺序，只读task_id，
排除当前版本库已有dev5配置的10个唯一ID，选择最先剩余5个。没有按问题、
rubric、历史得分或模型可解性选题；官方公开全集525题曾用于历史评测，
因此这些题不能称为从未见过的独立盲测。

固定ID：

- healthbench-professional:3533d9bfd2d32f8c465e7af62aec9781
- healthbench-professional:9a160f86c59743692e46fab89aae42f2
- healthbench-professional:fa30f3f57c7219130345f5c2e6d03d65
- healthbench-professional:2014ab7a9d8865f0da483817843ccbc5
- healthbench-professional:cd132a0c7cde74c0242aa8ef3850c9b9

不能用新五题与v2.47的旧五题均分差异宣称同题提升。后续本批迭代保持这五题，
不为了过线轮换容易题。上轮rubric仅用于报告；修复和候选建议只来自公开任务、
真实Tool/Canvas协议错误和通用完整回应义务，不携带医学参考答案。

## 必要适配与不变部分

复用SkillFlow BoundedAgent.execute_turn的非法动作反馈与completion校验，
FlowSteer InteractiveWorkflowEnv.step的edit→execute→feedback/history边界，
以及现有state-conditioned JSON Schema、证据receipt和候选profile。

1. 把已有supported/insufficient终局字段约束同步到采样schema，避免合法采样
   却必然被已有validator拒绝；不放宽引文/来源检查。
2. 引文错误反馈给出对应实际来源的局部文本和位置，供Agent自己重交；
   不自动改写artifact，不把近似文本当准确引文。
3. 修正“12 clinical terms”这类工具查询预算被误识别为临床数值；剂量、
   年龄、分母等未由公开输入/证据给出的具体值仍不能预填。
4. 三条可拒绝候选先验强调证据协议、实体范围、按原始请求检查漏答。
   职责/模型/关系由Director自由选择，不设置固定医疗角色或串行模板。

不改变模型池、种子、并发4、900秒任务时限、360秒执行时限、thinking、
token/Tool/ReAct预算、工具集合、唯一Output/FINISH、官方reference evaluator。
不训练，不更新权重、MACE/Bayesian或发布ACTIVE Skill；当前是候选prompt prior。

## 验收

先离线定向测试/prepare-only，再运行一次五题AgentGraph，不重复Direct/canary。
5/5显式FINISH且5/5 evaluator-valid，官方原生length-adjusted均值严格超过60%，
方可备份通过版本并进入全集525。缺失=N/A，不裁负分、不回收草稿评分。
五题是开发验收，不代表全集评分；即使超过门槛也不保证全部医学陈述正确。
截至本文件创建，v2.48五题和full525均尚未运行；真实状态以独立manifest/report为准。
