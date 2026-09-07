# HealthBench Professional AgentGraph-only development report

仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：5 / 3；operational failures without trajectory：2
- evaluator valid：3；explicit FINISH：3；terminal failure：0
- 严格全五题 native mean raw：N/A；length-adjusted：N/A；固定 denominator=5，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：-0.33333333 (3/5)；length-adjusted：-0.32705153 (3/5)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：33；Director recorded phase calls：40；observed Tool calls（题内去重后求和）：7；全五题 actual Tool calls：N/A

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:3533d9bfd2d32f8c465e7af62aec9781 | -1 | -0.9473152 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 7 | 0 | 0 |
| healthbench-professional:9a160f86c59743692e46fab89aae42f2 | 0 | -0.0223734 | true | true | false | 2 / 2 | node_1 → node_2 | 10 | 3 | 0 |
| healthbench-professional:fa30f3f57c7219130345f5c2e6d03d65 | 0 | -0.011466 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 16 | 4 | 0 |
| healthbench-professional:2014ab7a9d8865f0da483817843ccbc5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| healthbench-professional:cd132a0c7cde74c0242aa8ef3850c9b9 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 3/5 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:3533d9bfd2d32f8c465e7af62aec9781

- Agent model calls：`{"qwen3.5-9b-local": 7}`；missing-source receipts=0
- Director turns=9；recorded phase calls=18；turn attempts=42 (9/9)；turn latency_ms=268503.55 (9/9)
- Tool calls：`{}`；raw receipts=0；duplicates=0；latency_ms=N/A (0/0)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 0, "runtime_failure_receipts": 0, "runtime_failure_types": {}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 26887 (7/7) | 7557 (7/7) | 34444 (7/7) | 117431.98 (7/7) |
| Director recorded phases | 138093 (18/18) | 6795 (18/18) | N/A (0/18) | 109378.52 (18/18) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 5660.784458741546, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 2120, "output_reasoning_tokens": 148, "output_tokens": 306, "total_tokens": 2426}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:9a160f86c59743692e46fab89aae42f2

- Agent model calls：`{"qwen3.5-9b-local": 10}`；missing-source receipts=0
- Director turns=5；recorded phase calls=10；turn attempts=24 (5/5)；turn latency_ms=160415.98 (5/5)
- Tool calls：`{"healthbench-bookshelf.search": 1, "healthbench-literature.search": 1, "healthbench-medrag.search": 1}`；raw receipts=6；duplicates=3；latency_ms=3453.6493 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 0, "runtime_failure_receipts": 1, "runtime_failure_types": {"ReactExecutionError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 56566 (10/10) | 6478 (10/10) | 63044 (10/10) | 103710.81 (10/10) |
| Director recorded phases | 72406 (10/10) | 4404 (10/10) | N/A (0/10) | 66133.506 (10/10) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 1, "api_calls": 1, "latency_ms": 8571.91033102572, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 1153, "output_reasoning_tokens": 44, "output_tokens": 126, "total_tokens": 1279}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:fa30f3f57c7219130345f5c2e6d03d65

- Agent model calls：`{"qwen3.5-9b-local": 15, "unknown": 1}`；missing-source receipts=0
- Director turns=6；recorded phase calls=12；turn attempts=28 (6/6)；turn latency_ms=121430.29 (6/6)
- Tool calls：`{"healthbench-authoritative.search": 1, "healthbench-literature.search": 2, "healthbench-source.read": 1}`；raw receipts=7；duplicates=3；latency_ms=8776.41 (4/4)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 0, "runtime_failure_receipts": 2, "runtime_failure_types": {"ReactExecutionError": 1, "TimeoutError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 83352 (15/16) | 33854 (15/16) | 117206 (15/16) | 521651.7 (15/16) |
| Director recorded phases | 107920 (12/12) | 4277 (12/12) | N/A (0/12) | 65093.744 (12/12) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 3, "api_calls": 3, "latency_ms": 5643.620582297444, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 3459, "output_reasoning_tokens": 91, "output_tokens": 337, "total_tokens": 3796}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:2014ab7a9d8865f0da483817843ccbc5

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

### healthbench-professional:cd132a0c7cde74c0242aa8ef3850c9b9

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
