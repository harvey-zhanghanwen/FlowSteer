# HealthBench Professional v2.40：修复与验证边界

## 旧版为何超时、为何零分

v2.39原五题只有3条完整trajectory：两个试验名在Director第一条contract中就被
解释成了错误的人群或设计方式，检索查询跟随这些假设扩展，最后没有回答目标
试验；两题原始rubric为0。不能将搜索为空当“研究不存在”。

IBD/HIV与MDT分别提交14、9次Canvas编辑，多次MODIFY后达到原定900秒时限。
旧取消路径只存progress，不存完整未完成trajectory，所以不能杜撰每次修复
失败的具体Agent原因。已有日志显示provider429/节点超时影响运行，但不是
已证明的两个任务唯一原因。本轮排查另外确认两个代码级来源交接缺口。

## 最小修复

1. 完成校验与知识索引共享同一条有界来源遍历，认可真实上游/peer/历史
   Tool receipts；仅文本自述、伪造同ID摘录和被屏蔽的上游均不予绑定。
2. 双向REVISION拿到自身DRAFT的来源，不止文本；有限四次调用/阶段屏障
   不变，不重放旧控制trace、不增加预算、不形成自依赖。
3. 仅对单条、简短、可直接检索的公开关键词输入，首搜不许擅自添删词。
   不预先选择来源，不强制检索；首搜Observation或上游真实来源出现后可
   继续refinement。该限制不判断实体真实含义，也不保证临床相关性。
4. 标题/元数据不是正文，不记为内容有效搜索；总Tool调用上限仍为3。
5. 超时/异常前已有动作、Canvas、模型返回和失败回执另存诊断文件；原异常
   继续传播，标non_scoreable=true，不能成为假FINISH或假评分。

来源和必要不兼容说明见docs/source_map.md、docs/adaptation_log.md。
没有新增数据库、分类路由、医疗角色、固定workflow、训练或权重更新。

## 防止针对样本的提示词调优

- Director提示词仍为minimal-neutral.v20，完全未改。
- 没有将试验解释、具体药物方案、正确答案、rubric或参考回答写进新增规则。
- 新反馈只使用原始公开query及真实来源；规则测试使用虚构名称与合成资料。
- 保持原五个样本、模型池、thinking、采样参数、seed、并发4、900秒、20轮
  以及相同官方grader；不增加模型canary、不重跑Direct。
- 这五题是反复使用的开发样本，不声称独立held-out或525题准确率。

## 验证与运行

定向/回归分组通过：来源95、Runtime100、collector/runner114，部分交叉，
不合计成唯一测试数；工厂接线3项通过。prepare-only确认配置和原五题。

```bash
FLOWSTEER_ROLLOUT_GPU=6 FLOWSTEER_SUPERVISOR_PORT=8026 \
  /ssd1/iclr/gpf/venvs/skillflow/bin/python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_healthbench_professional_query_recovery_v2_40_dev5.yaml \
  --collection-arm agentgraph
```

结果目录独立：`artifacts/healthbench_professional_query_recovery_v2_40_dev5/evaluation`。
`evaluator_private/partial_trajectories.jsonl`仅在异常时写出，不存在不代表遗漏。
正式分数以run_manifest和evaluator receipt为准；此说明不预测改动的收益。
