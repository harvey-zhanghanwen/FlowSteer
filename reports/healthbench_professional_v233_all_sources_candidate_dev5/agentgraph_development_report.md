# HealthBench Professional AgentGraph-only development report

仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：5 / 4；operational failures without trajectory：1
- evaluator valid：4；explicit FINISH：4；terminal failure：0
- 严格全五题 native mean raw：N/A；length-adjusted：N/A；固定 denominator=5，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：0.26923077 (4/5)；length-adjusted：0.26479137 (4/5)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：51；Director recorded phase calls：34；observed Tool calls（题内去重后求和）：14；全五题 actual Tool calls：N/A

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:3533d9bfd2d32f8c465e7af62aec9781 | 1 | 1.046158 | true | true | false | 2 / 2 | node_1 → node_2 | 4 | 0 | 0 |
| healthbench-professional:9a160f86c59743692e46fab89aae42f2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| healthbench-professional:fa30f3f57c7219130345f5c2e6d03d65 | 0.69230769 | 0.58661469 | true | true | false | 4 / 4 | node_1 → node_4; node_2 → node_3; node_3 → node_4 | 31 | 6 | 0 |
| healthbench-professional:2014ab7a9d8865f0da483817843ccbc5 | -0.61538462 | -0.62091182 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 10 | 4 | 0 |
| healthbench-professional:cd132a0c7cde74c0242aa8ef3850c9b9 | 0 | 0.0473046 | true | true | false | 2 / 2 | node_1 → node_2 | 6 | 4 | 2 |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 4/5 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:3533d9bfd2d32f8c465e7af62aec9781

- Agent model calls：`{"qwen3.5-9b-local": 4}`；missing-source receipts=1
- Director turns=4；recorded phase calls=8；turn attempts=18 (4/4)；turn latency_ms=69822.303 (4/4)
- Tool calls：`{}`；raw receipts=0；duplicates=0；latency_ms=N/A (0/0)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 1, "runtime_failure_types": {"CompletionArtifactEmpty": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 10091 (4/4) | 13541 (4/4) | 23632 (4/4) | 213873.31 (4/4) |
| Director recorded phases | 132298 (8/8) | 1669 (8/8) | N/A (0/8) | 27323.029 (8/8) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 6566.462128423154, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 2370, "output_reasoning_tokens": 141, "output_tokens": 323, "total_tokens": 2693}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:9a160f86c59743692e46fab89aae42f2

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

### healthbench-professional:fa30f3f57c7219130345f5c2e6d03d65

- Agent model calls：`{"MiniMax-M3": 2, "qwen3.5-9b-local": 27, "unknown": 2}`；missing-source receipts=0
- Director turns=7；recorded phase calls=14；turn attempts=34 (7/7)；turn latency_ms=188721.06 (7/7)
- Tool calls：`{"healthbench-authoritative.search": 3, "healthbench-computation.calculator": 1, "healthbench-source.read": 2}`；raw receipts=18；duplicates=12；latency_ms=11829.372 (6/6)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 0, "runtime_failure_receipts": 4, "runtime_failure_types": {"ReactExecutionError": 2, "TimeoutError": 2}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 168482 (29/31) | 57898 (29/31) | 226498 (29/31) | 878225.46 (29/31) |
| Director recorded phases | 264325 (14/14) | 4139 (14/14) | N/A (0/14) | 63882.564 (14/14) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 3, "api_calls": 3, "latency_ms": 13377.546487376094, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 7137, "output_reasoning_tokens": 298, "output_tokens": 576, "total_tokens": 7713}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:2014ab7a9d8865f0da483817843ccbc5

- Agent model calls：`{"qwen3.5-9b-local": 10}`；missing-source receipts=0
- Director turns=3；recorded phase calls=6；turn attempts=14 (3/3)；turn latency_ms=84424.943 (3/3)
- Tool calls：`{"healthbench-literature.search": 2, "healthbench-source.read": 2}`；raw receipts=4；duplicates=0；latency_ms=6280.1675 (4/4)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 0, "runtime_failure_receipts": 0, "runtime_failure_types": {}}`；tool errors=`{"LookupError": 1}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 61429 (10/10) | 17763 (10/10) | 79192 (10/10) | 284469.31 (10/10) |
| Director recorded phases | 111229 (6/6) | 1727 (6/6) | N/A (0/6) | 28064.261 (6/6) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 3, "api_calls": 3, "latency_ms": 11796.639979816973, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 3572, "output_reasoning_tokens": 179, "output_tokens": 431, "total_tokens": 4003}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:cd132a0c7cde74c0242aa8ef3850c9b9

- Agent model calls：`{"qwen3.5-9b-local": 6}`；missing-source receipts=0
- Director turns=3；recorded phase calls=6；turn attempts=14 (3/3)；turn latency_ms=68014.509 (3/3)
- Tool calls：`{"healthbench-ahrq.search": 1, "healthbench-knowledge.search": 1, "healthbench-literature.search": 2}`；raw receipts=4；duplicates=0；latency_ms=5815.3799 (4/4)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 2, "provider_identified_runtime_failure_receipts": 0, "runtime_failure_receipts": 0, "runtime_failure_types": {}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 31238 (6/6) | 8954 (6/6) | 40192 (6/6) | 142602.21 (6/6) |
| Director recorded phases | 101601 (6/6) | 1394 (6/6) | N/A (0/6) | 23198.332 (6/6) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 5, "api_calls": 5, "latency_ms": 10902.535342611372, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 2402, "output_reasoning_tokens": 68, "output_tokens": 259, "total_tokens": 2661}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
