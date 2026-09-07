# HealthBench v2.47：同五题，修复参数反馈和职责边界

沿用v2.46协议的同一5个task ID、模型池、thinking、工具集合、900秒任务时限、
360秒Agent时限、并发4、六轮ReAct/三次Tool预算及官方reference evaluator。
无训练、权重更新、MACE、Bayesian、ACTIVE发布或Skill evolution。
v2.46源版本8ca968750b08170c6c935d31e1350dc5e4553727，结果3/5有效、
2超时，完整五题raw/length-adjusted为N/A；详细报告在v246分支提交e64978b。

## 允许调整的范围

- 角色仍由Director通过free-text contract自主定义。候选Skill改为强调
  原问题范围、来源identifier namespace、明确产物和真实依赖、局部检索
  缺口与整个答案覆盖的区别。独立解释/核对/综合只是可选职责，不强制数量、
  模型、固定角色或topology。
- ReAct仍为唯一admitted execution mode。收到足够证据可直接complete；
  不关闭工具来绕过协议、不无意义重查、不扩预算。
- 只修复已观察且可离线复现的接口问题，来源见source_map。不得把这5题的
  rubric、参考回答、医学数值或具体处置写入Skill/提示词。

## 通过与停止边界

本轮是公开test中的已观察开发批次；保留固定五题，不轮换到容易题。
进入完整525的门槛仍为5/5显式FINISH、5/5 evaluator-valid，且原生官方
overall_score_length_adjusted算术均值严格超过60%；原始分同时报告。
缺失为N/A；不裁负分、不用中间答案补终局。通过后备份接受版本再冻结运行
525；未通过继续按最早失败的通用机制诊断，不把候选先验当成已验证Skill。

v247 full525仅配置准备，尚未启动。不同版本各自独立源码目录和artifacts，
不改动仍在执行的条件、不重复Direct/模型池探测或额外付费canary。
