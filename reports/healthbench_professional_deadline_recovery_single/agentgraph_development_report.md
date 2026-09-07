# HealthBench Professional AgentGraph-only development report

仅本轮固定样本的开发观测；不是正式 benchmark 估计，不提供 Direct/paired 比较。
分数直接读取 evaluation.metrics；不重评、不裁剪、不做失败置零。完整输入、回答与 evaluator receipt 仅在 evaluator_private。

- selected / persisted terminal trajectories：1 / 1；operational failures without trajectory：0
- evaluator valid：1；explicit FINISH：1；terminal failure：0
- 严格全体固定样本 native mean raw：0.5；length-adjusted：0.2470424；固定 denominator=1，任一缺失则 N/A，绝不置零。
- completed-only native mean raw：0.5 (1/1)；length-adjusted：0.2470424 (1/1)
- completed-only 是实际已观测子集的描述性均分，不能与旧版本完整样本均分直接当作提升或下降比较。
- 已落盘 trajectory 内 Agent recorded model calls：6；Director recorded phase calls：6；observed Tool calls（题内去重后求和）：0；全体固定样本 actual Tool calls：0

| Task | Raw | Length-adjusted | Valid | FINISH | Terminal failure | Nodes / max | Topology | Agent model receipts | Tool calls (dedup) | Grader provider errors |
|---|---:|---:|---|---|---|---:|---|---:|---:|---:|
| healthbench-professional:38ed97e78292dfcf0805ac924311fa14 | 0.5 | 0.2470424 | true | true | false | 3 / 3 | node_1 → node_2; node_2 → node_3 | 6 | 0 | 0 |

## 模型、调用与覆盖范围

- Agent：合并成功 execution 与 runtime failure metadata 中的模型 receipt，按 request identity 去重；未保存调用不推算。
- Director：分别报告 turn、已保存的 reasoning/action phase 调用与 turn attempt_count；phase tokens 不覆盖被覆盖/未保存的重试。实际 model_id 未单独记录时为 N/A，不从 policy 名称猜测。
- Tool：成功和失败执行合并后按 tool_id + 单调时钟起止去重，避免 continuation/history 重算。无事件 identity 时 actual_calls=N/A，另保留 distinct receipt count。
- Tokens/latency 后括号为有数值的 receipt 数 / 该类 receipt 总数；缺失是 N/A 而非零。Provider total_tokens 原样保留，不假设等于输入+输出。
- 延迟分别是模型/工具/Director turn 的累计服务时间与单题 grader 遥测，不是端到端 wall-clock；并发、重试和工具时间可重叠，不跨类相加。
- provider-identified runtime failures 仅按结构化 provider_id/http_status 标记；与 runtime 类型计数重叠，不根据错误消息或题目内容推断原因。未保存 provider errors 不等于没有发生。
- 调用/token/latency 只覆盖已落盘的 1/1 条终局 trajectory 及其内嵌失败；不是整轮计费总量。缺轨迹题只读取本轮 selected task、progress、collection failure 作为 operational receipt；不据此推算完整调用。进程日志、未落盘尝试不在范围。

### healthbench-professional:38ed97e78292dfcf0805ac924311fa14

- Agent model calls：`{"deepseek-v4-flash": 1, "qwen3.5-9b-local": 5}`；missing-source receipts=0
- Director turns=3；recorded phase calls=6；turn attempts=16 (3/3)；turn latency_ms=74766.608 (3/3)
- Tool calls：`{}`；raw receipts=0；duplicates=0；latency_ms=N/A (0/0)
- Terminal reason：finish；errors=`{"execution_error_types": {}, "grader_error_present": false, "grader_provider_errors": 0, "provider_identified_runtime_failure_receipts": 0, "runtime_failure_receipts": 0, "runtime_failure_types": {}}`；tool errors=`{}`

| Receipt scope | Prompt tokens | Completion tokens | Provider total tokens | Latency ms |
|---|---:|---:|---:|---:|
| Agent recorded calls | 30969 (6/6) | 11872 (6/6) | 42841 (6/6) | 196500.14 (6/6) |
| Director recorded phases | 60873 (6/6) | 1926 (6/6) | N/A (0/6) | 24348.471 (6/6) |

- Grader model=gpt-5.4-2026-03-05；saved telemetry=`{"api_call_receipts": 3, "api_calls": 3, "latency_ms": 7053.816992789507, "model_id": "gpt-5.4-2026-03-05", "token_usage": {"input_cached_tokens": 0, "input_tokens": 21210, "output_reasoning_tokens": 139, "output_tokens": 331, "total_tokens": 21541}, "usage_scope": "saved grader_telemetry for this terminal evaluator receipt; no unrecorded retries"}`

## 解释边界

图是末次保存的 Canvas 节点与方向边，不代表真实语义依赖或通信被正确使用。失分原因不自动分类；请读 evaluator-private demos 与原生 rubric receipts。
