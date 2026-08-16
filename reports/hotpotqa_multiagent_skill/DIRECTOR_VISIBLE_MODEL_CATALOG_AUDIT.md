# HotpotQA Architecture-v6：Director 可见 Executor 模型目录审计

审计日期：2026-08-16（UTC）。本报告只覆盖模型目录、Director 可见性和
Executor 静态可调用边界；没有启动训练、权重更新或新 HotpotQA rollout。

## 结论

Architecture-v6 的 Executor search space 现在包含 **10 个**等权模型臂：

- 本地 Qwen3.5-9B：1 个；
- VectorEngine：9 个；
- 模型家族：Qwen、DeepSeek、GPT、MiniMax、GLM、Kimi，共 6 类。

所有远程 `model_id`/`model_name` 都逐字匹配本轮真实 `/v1/models` 返回的
大小写敏感 ID。新增的 `glm-4.5-flash` 和 `kimi-k2` 各做且只做了一次最小
文本 canary，两者均通过；此前已经成功的模型没有重复付费 canary。

## 本轮真实请求与版本化 receipts

- `/v1/models`：1 次，HTTP 200，返回 524 个对象；
- 新模型 text canary：2 次；
- retry：0；
- 本轮外部 API 总调用：3 次。

保存位置：

- `artifacts/hotpotqa_multiagent_skill/model_catalog_v6/model_list_receipt.json`
- `artifacts/hotpotqa_multiagent_skill/model_catalog_v6/canary_receipt.json`
- `artifacts/hotpotqa_multiagent_skill/model_catalog_v6/evidence_manifest.json`
- `artifacts/hotpotqa_multiagent_skill/model_catalog_v6/director_visible_catalog_receipt.json`
- `config/model_catalog_hotpotqa_deep_v6.yaml`

完整 prompt、expected output、真实 output、request ID、token、latency 和错误字段
均保存在 canary receipt；文件不含凭据。

| 新候选 exact ID | Request ID | Input / Output tokens | Latency | Attempt | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| `glm-4.5-flash` | `chatcmpl-89DHC97MsxC9bRQdCR2t54itDEvdG` | 193 / 264 | 28.49 s | 1 | 精确输出 `<answer>Paris</answer>`，通过 |
| `kimi-k2` | `chatcmpl-3vh4skncd4q` | 252 / 6 | 5.62 s | 1 | 精确输出 `<answer>Paris</answer>`，通过 |

## Director 可见目录

“Director-visible”不是只检查 YAML。本轮使用现有
`AgentGraphOrchestrator.build_prompt()` 真正渲染无模型初始状态，再解析其中的
`model_catalog`。测试确认 10 个模型各出现一次，并且 Director 可以看到中性的：

- provider / local-or-remote；
- general reasoning class；
- canary 或历史执行 latency 事实；
- context capability（仅保留 provider receipt 明确声明的内容）；
- chat instruction / exact-tag compatibility；
- 当前 availability evidence。

这些字段通过现有 renderer 已允许的 `profile`、`family`、
`text_qa_canary` 和 `canary_source` 传入，没有修改 Director，也没有加入题型到模型
的人工映射。所有模型 `selection_weight`、`cheap_weight`、`fast_weight` 均为 1.0。

| Model | Provider | Model-list | Canary / evidence | Director-visible | Executor callable | Attributes |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `qwen3.5-9b-local` | local SGLang | N/A | 既有成功 Executor execution | 是 | 静态 graph/provider validation 通过 | Qwen；local；8,192 context；exact-tag |
| `qwen3.5-flash` | VectorEngine | 是，text/openai | 既有成功 Executor execution | 是 | 是 | Qwen；remote；provider-tagged thinking；exact-tag |
| `qwen3.5-plus` | VectorEngine | 是，text/openai | 已有 canary pass，8.28 s | 是 | 是 | Qwen；remote；provider-tagged thinking；exact-tag |
| `deepseek-v4-flash` | VectorEngine | 是，text/openai | 已有 canary pass，10.62 s | 是 | 是 | DeepSeek；remote；provider 声明 1M context；exact-tag |
| `deepseek-v4-pro` | VectorEngine | 是，text/openai | 已有 canary pass，14.37 s | 是 | 是 | DeepSeek；remote；provider-tagged thinking/long context；exact-tag |
| `gpt-4o-mini` | VectorEngine | 是，text/openai | 既有成功 Executor execution | 是 | 是 | GPT；remote；general；exact-tag |
| `MiniMax-M2.5` | VectorEngine | 是，text/openai | 既有成功 Executor execution | 是 | 是 | MiniMax；remote；general；exact-tag |
| `MiniMax-M3` | VectorEngine | 是，text/openai | 已有 canary pass，19.37 s | 是 | 是 | MiniMax；remote；provider-described agent/long context；exact-tag |
| `glm-4.5-flash` | VectorEngine | 是，text/openai | 本轮 canary pass，28.49 s | 是 | 是 | GLM；remote；provider-tagged thinking；exact-tag |
| `kimi-k2` | VectorEngine | 是，text/openai | 本轮 canary pass，5.62 s | 是 | 是 | Kimi；remote；provider-described reasoning；exact-tag |

这里的 “Executor callable” 表示：exact model ID 能被 registry 解析到 provider，且用该
模型构成的单节点完整 AgentGraph 能通过现有运行前校验；本轮没有为了重复验证而再次
调用已成功模型。

## 纳入与排除边界

- 新增 GLM 和 Kimi 是为了增加真实家族异构性，而不是为了凑满 12 个模型。
- fresh receipt 中仍没有任何 Gemini exact ID，因此没有猜测 Gemini alias。
- Grok 的旧 model-list 条目不能覆盖最近 canary 的 HTTP 429；本轮没有重试，也没有纳入。
- embedding、reranker、audio/video/image-only、realtime、TTS、transcribe 等非当前
  text-QA contract 模型均未进入目录。
- 本轮 fresh list 为 524 个对象，而同日较早 receipt 为 552 个，进一步说明 provider
  列表会变化；实际运行必须绑定本目录版本和 exact receipt，不能把一次列表当成永久能力。

## 验证

定向无模型测试：

```text
tests/unit/test_model_catalog_v6.py
5 passed
```

测试覆盖：10-arm 等权 catalog、9 个远程 exact ID 在 fresh list 中、text/openai
过滤、新候选一次 canary、全目录 evidence、真实 Director rendered-state 可见性，以及
每个模型臂的完整 AgentGraph 静态可调用性。

## 限制

最小 canary 只证明当前 text transport 和 exact Output contract 可用，不证明模型在
HotpotQA 上优于其他模型，也不证明一次测得的 latency、价格或长期 availability。
模型选择规律仍应由后续同条件 trajectory、evaluator 和训练证据学习；本目录没有预埋
“某题型必须调用某模型”的规则。Flow-Director 仍固定为本地 Qwen3.5-9B，新增远程模型
只属于 Executor 候选池。
