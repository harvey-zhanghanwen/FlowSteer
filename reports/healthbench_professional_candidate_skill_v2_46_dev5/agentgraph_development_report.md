# HealthBench Professional AgentGraph-only development report

仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：5 / 3；operational failures without trajectory：2
- evaluator valid：3；explicit FINISH：3；terminal failure：0
- 严格全五题 native mean raw：N/A；length-adjusted：N/A；固定 denominator=5，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：0 (3/5)；length-adjusted：-0.035917 (3/5)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：51；Director recorded phase calls：38；observed Tool calls（题内去重后求和）：14；全五题 actual Tool calls：N/A

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:9566084de89c416408691006a6f06f9c | 0 | 0.051597 | true | true | false | 2 / 2 | node_1 → node_2 | 21 | 5 | 1 |
| healthbench-professional:c19c2113ba68bb3c4a3e63836e31b558 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| healthbench-professional:a5778c7ecdb4eeccf9d252631e18a274 | 0 | -0.207711 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 23 | 6 | 0 |
| healthbench-professional:e339f34a3a35f3f067422b5768287f7c | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| healthbench-professional:c42bd4fc760487ac7b5e70fbb41a8edc | 0 | 0.048363 | true | true | false | 2 / 2 | node_1 → node_2 | 7 | 3 | 0 |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 3/5 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:9566084de89c416408691006a6f06f9c

- Agent model calls：`{"deepseek-v4-flash": 1, "qwen3.5-9b-local": 20}`；missing-source receipts=0
- Director turns=7；recorded phase calls=14；turn attempts=32 (7/7)；turn latency_ms=100444.51 (7/7)
- Tool calls：`{"healthbench-knowledge.search": 2, "healthbench-literature.search": 1, "healthbench-medrag.search": 2}`；raw receipts=8；duplicates=3；latency_ms=1536.5191 (5/5)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 1, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 3, "runtime_failure_types": {"ReactExecutionError": 2, "ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 107539 (20/21) | 18375 (20/21) | 125914 (20/21) | 135793.55 (20/21) |
| Director recorded phases | 161028 (14/14) | 5845 (14/14) | N/A (0/14) | 43824.531 (14/14) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 3, "api_calls": 3, "latency_ms": 7870.272357016802, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 1790, "output_reasoning_tokens": 116, "output_tokens": 260, "total_tokens": 2050}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:c19c2113ba68bb3c4a3e63836e31b558

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

### healthbench-professional:a5778c7ecdb4eeccf9d252631e18a274

- Agent model calls：`{"MiniMax-M3": 21, "deepseek-v4-flash": 2}`；missing-source receipts=0
- Director turns=8；recorded phase calls=16；turn attempts=34 (8/8)；turn latency_ms=104735.65 (8/8)
- Tool calls：`{"healthbench-bookshelf.search": 1, "healthbench-literature.search": 5}`；raw receipts=12；duplicates=6；latency_ms=2504.0204 (6/6)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 4, "runtime_failure_types": {"ReactExecutionError": 2, "ReactGenerationError": 2}}`；tool errors=`{"ValueError": 4}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 85442 (21/23) | 25779 (21/23) | 112460 (21/23) | 225948.95 (21/23) |
| Director recorded phases | 188654 (16/16) | 4933 (16/16) | N/A (0/16) | 40272.315 (16/16) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 1, "api_calls": 1, "latency_ms": 5293.24165917933, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 2724, "output_reasoning_tokens": 65, "output_tokens": 181, "total_tokens": 2905}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:e339f34a3a35f3f067422b5768287f7c

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

### healthbench-professional:c42bd4fc760487ac7b5e70fbb41a8edc

- Agent model calls：`{"deepseek-v4-flash": 1, "qwen3.5-9b-local": 6}`；missing-source receipts=0
- Director turns=4；recorded phase calls=8；turn attempts=18 (4/4)；turn latency_ms=51080.07 (4/4)
- Tool calls：`{"healthbench-knowledge.search": 1, "healthbench-literature.search": 1, "healthbench-medrag.search": 1}`；raw receipts=3；duplicates=0；latency_ms=1815.1509 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 1, "runtime_failure_types": {"ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 37782 (6/7) | 3903 (6/7) | 41685 (6/7) | 33631.143 (6/7) |
| Director recorded phases | 95338 (8/8) | 3347 (8/8) | N/A (0/8) | 25901.264 (8/8) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 1, "api_calls": 1, "latency_ms": 5747.205764055252, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 1540, "output_reasoning_tokens": 28, "output_tokens": 81, "total_tokens": 1621}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
