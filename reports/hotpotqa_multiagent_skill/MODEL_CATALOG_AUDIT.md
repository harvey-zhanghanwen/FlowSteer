# HotpotQA Multi-Agent Model Catalog Audit

Audit time: 2026-08-16 UTC. The provider endpoint was read from the existing
VectorEngine configuration. No credential is stored in this report, the model
catalog, or either receipt.

## Receipts and catalog freeze

- Complete non-secret `/v1/models` response:
  `artifacts/hotpotqa_multiagent_skill/model_catalog/model_list_receipt.json`
  (HTTP 200, 552 returned model objects at 2026-08-16T03:17:46Z).
- Output-protocol canaries:
  `artifacts/hotpotqa_multiagent_skill/model_catalog/canary_receipt.json`.
- Frozen Executor catalog:
  `config/model_catalog_hotpotqa_multiagent_v1.yaml`.

A second model-list lookup performed immediately before the canaries returned
534 objects rather than 552. This is provider-catalog instability, so the
canary receipt and exact configured model IDs are the operative evidence; the
raw list is not treated as a permanent provider capability claim.

## Candidate decisions

`reasoning` below means only that a separately named reasoning endpoint was
found or tested. The simple two-hop canary validates text generation and the
global Output protocol; it is not a reasoning benchmark or a price test.

| Exact model ID | Provider | Family/type | Text QA | Reasoning separately verified | In prior catalog | Added to v1 | Canary/status | Decision reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.5-9b-local` | local SGLang | Qwen, local text | Prior Hotpot receipts pass | No | Yes | Yes | Existing real run evidence | Required local Director; also remains an allowed Executor arm. |
| `qwen3.5-flash` | VectorEngine | Qwen text | Prior Hotpot receipts pass | No | Yes | Yes | Existing real run evidence | Exact ID present in model list and already used successfully. |
| `qwen3.5-plus` | VectorEngine | Qwen text | Pass | No | No | Yes | Pass, request `chatcmpl-fce39709-8b50-9389-9c89-2e3e9d1b3990`, 8.28 s | Exact returned ID and exact Output wrapper passed. |
| `deepseek-v4-flash` | VectorEngine | DeepSeek text | Pass | No | No | Yes | Pass, request `30e00394bf94411cb61851f5079e1018`, 10.62 s | Exact returned ID and exact Output wrapper passed. |
| `deepseek-v4-pro` | VectorEngine | DeepSeek text | Pass | No | No | Yes | Pass, request `751b13e261514131b2b39a6be48ec928`, 14.37 s | Exact returned ID and exact Output wrapper passed; no extra reasoning claim is made. |
| `gpt-4o-mini` | VectorEngine | GPT text | Prior Hotpot receipts pass | No | Yes | Yes | Existing real run evidence | Exact ID present and already used successfully. |
| `MiniMax-M2.5` | VectorEngine | MiniMax text | Prior Hotpot receipts pass | No | Yes | Yes | Existing real run evidence | Exact provider model name remains mapped by stable local ID `minimax-m2.5`. |
| `MiniMax-M3` | VectorEngine | MiniMax text | Pass | No | No | Yes | Pass, request `58ff45496d3146968fde15efc6522dd4`, 19.37 s | Exact returned ID and exact Output wrapper passed. |
| `grok-4-1-fast-non-reasoning` | VectorEngine | Grok text | Not established | No | No | No | HTTP 429 | A model-list entry is insufficient; failed compatibility canary excludes it. |
| `grok-4-1-fast-reasoning` | VectorEngine | Grok reasoning-labelled text | Not established | No | No | No | HTTP 429 | Failed compatibility canary excludes it. |
| Gemini family | VectorEngine | Gemini | Not available | No | No | No | No exact ID returned | No Gemini ID appeared in the actual list; no alias was guessed. |

The four new passed canaries used one fixed two-hop question, temperature 0,
seed 20260816, and required exactly `<answer>Paris</answer>`. Full prompt,
response, request ID, tokens, latency, finish status, and errors are in the
canary receipt.

## Routing boundary

All eight active arms have equal numeric routing priors in v1. The Director sees
only the exact model ID plus family/profile/canary facts and chooses the model in
its normal `add_agent`/`modify_agent` action. No role-to-model table, required
family count, or preferred-model suggestion is injected. Consequently, later
model-frequency statistics demonstrate actual use but do not by themselves
establish causal model superiority; that requires a same-graph diagnostic.
