# HealthBench Professional AgentGraph-only development report

仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：5 / 3；operational failures without trajectory：2
- evaluator valid：3；explicit FINISH：3；terminal failure：0
- 严格全五题 native mean raw：N/A；length-adjusted：N/A；固定 denominator=5，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：0.12121212 (3/5)；length-adjusted：0.12095732 (3/5)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：47；Director recorded phase calls：32；observed Tool calls（题内去重后求和）：16；全五题 actual Tool calls：N/A

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a | 0 | -0.0597702 | true | true | false | 4 / 4 | node_1 → node_2; node_2 → node_3; node_3 → node_4 | 24 | 9 | 0 |
| healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181 | 0.36363636 | 0.36834036 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 13 | 3 | 0 |
| healthbench-professional:4f08ae480b16ef825cf098eca6530e68 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3 | 0 | 0.0543018 | true | true | false | 2 / 2 | node_2 → node_1 | 10 | 4 | 0 |
| healthbench-professional:37101607e2947481e85e8fe3597a1acf | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 3/5 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a

- Agent model calls：`{"MiniMax-M3": 6, "deepseek-v4-flash": 2, "qwen3.5-9b-local": 16}`；missing-source receipts=0
- Director turns=7；recorded phase calls=14；turn attempts=30 (7/7)；turn latency_ms=97941.325 (7/7)
- Tool calls：`{"healthbench-authoritative.search": 1, "healthbench-knowledge.search": 2, "healthbench-literature.search": 2, "healthbench-medrag.search": 1, "healthbench-trials.search": 3}`；raw receipts=11；duplicates=2；latency_ms=10263.449 (9/9)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 2, "runtime_failure_receipts": 3, "runtime_failure_types": {"ReactExecutionError": 1, "ReactGenerationError": 2}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 240169 (22/24) | 26470 (22/24) | 266993 (22/24) | 250189.41 (22/24) |
| Director recorded phases | 191339 (14/14) | 4153 (14/14) | N/A (0/14) | 31622.489 (14/14) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 4603.415636345744, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 3004, "output_reasoning_tokens": 48, "output_tokens": 226, "total_tokens": 3230}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181

- Agent model calls：`{"MiniMax-M3": 2, "deepseek-v4-flash": 1, "qwen3.5-9b-local": 10}`；missing-source receipts=0
- Director turns=5；recorded phase calls=10；turn attempts=22 (5/5)；turn latency_ms=86518.68 (5/5)
- Tool calls：`{"healthbench-computation.calculator": 1, "healthbench-trials.search": 2}`；raw receipts=3；duplicates=0；latency_ms=2153.4434 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 2, "runtime_failure_types": {"ReactExecutionError": 1, "ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 65805 (12/13) | 6511 (12/13) | 72434 (12/13) | 75262.2 (12/13) |
| Director recorded phases | 89187 (10/10) | 2355 (10/10) | N/A (0/10) | 17578.565 (10/10) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 3, "api_calls": 3, "latency_ms": 7137.595695443451, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 3584, "output_reasoning_tokens": 134, "output_tokens": 385, "total_tokens": 3969}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:4f08ae480b16ef825cf098eca6530e68

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

### healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3

- Agent model calls：`{"MiniMax-M3": 5, "deepseek-v4-flash": 1, "qwen3.5-9b-local": 4}`；missing-source receipts=1
- Director turns=4；recorded phase calls=8；turn attempts=16 (4/4)；turn latency_ms=43224.812 (4/4)
- Tool calls：`{"healthbench-authoritative.search": 2, "healthbench-literature.search": 1, "healthbench-source.read": 1}`；raw receipts=4；duplicates=0；latency_ms=10644.673 (4/4)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 2, "runtime_failure_types": {"CancelledError": 1, "ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 47995 (9/10) | 9486 (9/10) | 57776 (9/10) | 105080.19 (9/10) |
| Director recorded phases | 73954 (8/8) | 2856 (8/8) | N/A (0/8) | 21834.202 (8/8) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 6254.128944128752, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 1521, "output_reasoning_tokens": 85, "output_tokens": 219, "total_tokens": 1740}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:37101607e2947481e85e8fe3597a1acf

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
