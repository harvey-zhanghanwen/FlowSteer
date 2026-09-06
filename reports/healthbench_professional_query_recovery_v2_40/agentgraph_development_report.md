# HealthBench Professional AgentGraph-only development report

仅五题开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：5 / 2；operational failures without trajectory：3
- evaluator valid：2；explicit FINISH：2；terminal failure：0
- 严格全五题 native mean raw：N/A；length-adjusted：N/A；固定 denominator=5，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：0.51666667 (2/5)；length-adjusted：0.53210167 (2/5)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整五题均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：11；Director recorded phase calls：12；observed Tool calls（题内去重后求和）：5；全五题 actual Tool calls：N/A

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a | 0.5 | 0.4961486 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 7 | 3 | 0 |
| healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| healthbench-professional:4f08ae480b16ef825cf098eca6530e68 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3 | 0.53333333 | 0.56805473 | true | true | false | 1 / 1 | no directed relations | 4 | 2 | 0 |
| healthbench-professional:37101607e2947481e85e8fe3597a1acf | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 2/5 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:f056cdb489e3636b0b51afb8fd6b3a8a

- Agent model calls：`{"qwen3.5-9b-local": 5, "qwen3.5-flash": 2}`；missing-source receipts=0
- Director turns=3；recorded phase calls=6；turn attempts=14 (3/3)；turn latency_ms=65181.43 (3/3)
- Tool calls：`{"healthbench-literature.search": 1, "healthbench-source.read": 1, "healthbench-trials.search": 1}`；raw receipts=3；duplicates=0；latency_ms=4498.397 (3/3)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 0, "runtime_failure_receipts": 0, "runtime_failure_types": {}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 65167 (7/7) | 4115 (7/7) | 69282 (7/7) | 42993.821 (7/7) |
| Director recorded phases | 116403 (6/6) | 1336 (6/6) | N/A (0/6) | 10775.53 (6/6) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 4208.8994504883885, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 2422, "output_reasoning_tokens": 171, "output_tokens": 360, "total_tokens": 2782}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:ed8b3ca0a4dabfd0827c17a08513a181

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

### healthbench-professional:4f08ae480b16ef825cf098eca6530e68

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

### healthbench-professional:dadbebd3dce1b5928cac5a44dde095d3

- Agent model calls：`{"deepseek-v4-flash": 1, "qwen3.5-9b-local": 3}`；missing-source receipts=0
- Director turns=3；recorded phase calls=6；turn attempts=12 (3/3)；turn latency_ms=19667.215 (3/3)
- Tool calls：`{"healthbench-authoritative.search": 1, "healthbench-literature.search": 1}`；raw receipts=2；duplicates=0；latency_ms=4164.9524 (2/2)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 1, "runtime_failure_receipts": 1, "runtime_failure_types": {"ReactGenerationError": 1}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 14378 (3/4) | 4502 (3/4) | 18880 (3/4) | 35257.135 (3/4) |
| Director recorded phases | 88329 (6/6) | 931 (6/6) | N/A (0/6) | 8128.666 (6/6) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 2, "api_calls": 2, "latency_ms": 4866.786736994982, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 1811, "output_reasoning_tokens": 142, "output_tokens": 296, "total_tokens": 2107}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

### healthbench-professional:37101607e2947481e85e8fe3597a1acf

本轮 collection failure 有 task/condition 依据，但 terminal trajectory 未落盘，终局分数和终局统计均 N/A。原始 selected question、progress 与 collect failure 见 evaluator-private demo；若 evaluator_private/partial_trajectories.jsonl 存在，还可查看同次运行的中间诊断，但不得据此补造终局或评分。

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
