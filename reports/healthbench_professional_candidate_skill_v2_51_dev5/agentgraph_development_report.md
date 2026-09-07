# HealthBench Professional AgentGraph-only development report

仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：5 / 4；operational failures without trajectory：1
- evaluator valid：4；explicit FINISH：4；terminal failure：0
- 严格全五题 native mean raw：N/A；length-adjusted：N/A；固定 denominator=5，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：0 (4/5)；length-adjusted：0.0444528 (4/5)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：82；Director recorded phase calls：66；observed Tool calls（题内去重后求和）：23；全五题 actual Tool calls：N/A

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:3533d9bfd2d32f8c465e7af62aec9781 | -1 | -0.9467272 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 6 | 0 | 0 |
| healthbench-professional:9a160f86c59743692e46fab89aae42f2 | 1 | 1.0464226 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 27 | 7 | 0 |
| healthbench-professional:fa30f3f57c7219130345f5c2e6d03d65 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| healthbench-professional:2014ab7a9d8865f0da483817843ccbc5 | 0 | 0.0523614 | true | true | false | 4 / 4 | node_2 → node_1; node_3 → node_1; node_1 → node_4 | 28 | 7 | 0 |
| healthbench-professional:cd132a0c7cde74c0242aa8ef3850c9b9 | 0 | 0.0257544 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 21 | 9 | 1 |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 4/5 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:3533d9bfd2d32f8c465e7af62aec9781

- Agent model calls：`{"MiniMax-M3": 3, "deepseek-v4-flash": 1, "qwen3.5-9b-local": 2}`；missing-source receipts=0
- Director turns=5；recorded phase calls=10；turn attempts=24 (5/5)；turn latency_ms=79841.907 (5/5)
- Tool calls：`{}`；raw receipts=0；duplicates=0；latency_ms=N/A (0/0)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 1, "runtime_failure_types": {"ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 20794 (5/6) | 5868 (5/6) | 26839 (5/6) | 65415.826 (5/6) |
| Director recorded phases | 99903 (10/10) | 3719 (10/10) | N/A (0/10) | 31101.178 (10/10) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 4278.97821739316, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 2122, "output_reasoning_tokens": 126, "output_tokens": 295, "total_tokens": 2417}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:9a160f86c59743692e46fab89aae42f2

- Agent model calls：`{"MiniMax-M3": 12, "deepseek-v4-flash": 2, "qwen3.5-9b-local": 13}`；missing-source receipts=0
- Director turns=12；recorded phase calls=24；turn attempts=54 (12/12)；turn latency_ms=174812.19 (12/12)
- Tool calls：`{"healthbench-authoritative.search": 5, "healthbench-literature.search": 1, "healthbench-source.read": 1}`；raw receipts=16；duplicates=9；latency_ms=14897.432 (7/7)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 5, "runtime_failure_types": {"ReactExecutionError": 3, "ReactGenerationError": 2}}`；tool errors=`{"TimeoutError": 1}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 370401 (25/27) | 27092 (25/27) | 398201 (25/27) | 395680.8 (25/27) |
| Director recorded phases | 344404 (24/24) | 6016 (24/24) | N/A (0/24) | 71277.631 (24/24) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 1, "api_calls": 1, "latency_ms": 4580.090372823179, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 838, "output_reasoning_tokens": 56, "output_tokens": 136, "total_tokens": 974}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:fa30f3f57c7219130345f5c2e6d03d65

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

### healthbench-professional:2014ab7a9d8865f0da483817843ccbc5

- Agent model calls：`{"MiniMax-M3": 18, "deepseek-v4-flash": 2, "qwen3.5-9b-local": 7, "unknown": 1}`；missing-source receipts=0
- Director turns=9；recorded phase calls=18；turn attempts=40 (9/9)；turn latency_ms=232492.32 (9/9)
- Tool calls：`{"healthbench-authoritative.search": 1, "healthbench-knowledge.search": 1, "healthbench-literature.search": 4, "healthbench-source.read": 1}`；raw receipts=15；duplicates=8；latency_ms=13418.919 (7/7)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 5, "runtime_failure_types": {"CancelledError": 1, "ReactExecutionError": 2, "ReactGenerationError": 2}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 335558 (25/28) | 46901 (25/28) | 383521 (25/28) | 589743.59 (25/28) |
| Director recorded phases | 291903 (18/18) | 6946 (18/18) | N/A (0/18) | 83247.553 (18/18) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 3, "api_calls": 3, "latency_ms": 7015.28337225318, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 2408, "output_reasoning_tokens": 247, "output_tokens": 457, "total_tokens": 2865}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:cd132a0c7cde74c0242aa8ef3850c9b9

- Agent model calls：`{"MiniMax-M3": 16, "deepseek-v4-flash": 1, "qwen3.5-9b-local": 4}`；missing-source receipts=0
- Director turns=7；recorded phase calls=14；turn attempts=36 (7/7)；turn latency_ms=247901.83 (7/7)
- Tool calls：`{"healthbench-authoritative.search": 1, "healthbench-knowledge.search": 1, "healthbench-literature.search": 2, "healthbench-medrag.search": 2, "healthbench-source.read": 1, "healthbench-trials.search": 2}`；raw receipts=12；duplicates=3；latency_ms=6868.0208 (9/9)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 1, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 2, "runtime_failure_types": {"ReactExecutionError": 1, "ReactGenerationError": 1}}`；tool errors=`{"RuntimeError": 1}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 214934 (20/21) | 38360 (20/21) | 254238 (20/21) | 547873.76 (20/21) |
| Director recorded phases | 169353 (14/14) | 6615 (14/14) | N/A (0/14) | 94868.069 (14/14) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 4, "api_calls": 4, "latency_ms": 6415.057795122266, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 2912, "output_reasoning_tokens": 103, "output_tokens": 299, "total_tokens": 3211}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
