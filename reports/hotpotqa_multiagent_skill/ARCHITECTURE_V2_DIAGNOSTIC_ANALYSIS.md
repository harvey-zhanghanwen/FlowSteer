# HotpotQA Multi-Agent architecture-v2 diagnostic analysis

## Scope and outcome

This is a fixed 14-task architecture-development diagnostic.  It reused the
already completed Round-01 Qwen3.5-9B Direct outputs and performed no GRPO,
backward pass, optimizer step, LoRA update, policy publish, MACE update, or
Skill lifecycle operation.

The full `Question -> local Qwen3.5-9B Director -> Canvas -> AgentGraph ->
Executor/communication -> Output -> evaluator -> trajectory` chain completed
for 14/14 tasks with zero collection failures.  This establishes execution
Stable Zero for this diagnostic, not formal training Step 0.

| Condition | Valid | EM | F1 |
| --- | ---: | ---: | ---: |
| Reused local Qwen3.5-9B Direct | 14/14 | 42.86 | 48.57 |
| architecture-v2 AgentGraph | 14/14 | 50.00 | 61.84 |
| AgentGraph - Direct |  | +7.14 | +13.27 |

On the same task IDs, v1 AgentGraph was 35.71 EM / 52.86 F1.  The v2 absolute
change is +14.29 EM / +8.98 F1, but this is not a causal A/B estimate because
the Director seed, prompt/tool version, action-token limit, and sampled
Executor choices changed.

## Observed workflow behavior

- Final topology: 13 singleton graphs and one two-node, one-edge graph.
- The sole multi-Agent graph sent one real upstream artifact from `retriever`
  to `accountant` and answered its task correctly.
- Executor nodes used seven exact catalog IDs: `gpt-4o-mini` 7,
  `qwen3.5-flash` 3, and one each of `qwen3.5-9b-local`,
  `deepseek-v4-flash`, `deepseek-v4-pro`, `minimax-m2.5`, and `minimax-m3`.
- The 71 Director turns contained 2 invalid actions and 9 rejected FINISH
  actions.  All 14 trajectories eventually ended with one accepted exact
  answer wrapper.
- There were 19 saved Executor records and 20 Executor attempts, so one
  successful record followed a retry.

The model-pool collapse seen in v1 is no longer present.  The graph-shape
distribution did not materially deepen, so the gain is evidence for usable
heterogeneous routing and terminal repair, not evidence that multi-Agent
collaboration has been learned.

## Result and failure classification

The evaluator labelled seven Graph answers exact and seven non-exact.  The
paired failure categories are:

- architecture gain: 3;
- architecture regression candidate: 2;
- both conditions correct: 4;
- shared reasoning/model failure candidate: 4;
- partial or overlong answer: 1.

The complete Wrong Demo objects, including the task, both answers, metrics,
final graph, Output inbox, telemetry, and trajectory ID, are stored in
`artifacts/hotpotqa_multiagent_skill/architecture_v2_diagnostic/wrong_demos.jsonl`.
Low absolute accuracy on this deliberately difficult development slice is not
used as a reason to add fixed roles, graph templates, Agent-count rewards, or
communication rewards.

### Representative first-error analysis

| Task suffix | Ground truth / Graph answer | First saved defect | Classification |
| --- | --- | --- | --- |
| `5ac2a912` | Wolfhounds / Hole | The Executor first found the useful years `1989` and `1985` but emitted two wrappers.  After two rejected FINISH actions, a destructive contract repair reversed the comparison. | Director continuation plus Executor reasoning |
| `5ab93287` | plant / herbaceous | The Director contract anchored the common property as `herbaceous`, although that property was not shared by all relevant species. | Director semantic anchoring |
| `5ab345db` | Hawaii / California | The contract contained the correct CEO-to-company-to-store chain; the Executor bound the corporation's California description to store location and ignored the Hawaii store evidence. | Executor evidence binding; regression from Direct |
| `5abee5e2` | The Joshua Tree / The Chimes | The contract explicitly contained the correct two-hop chain, but the Executor stopped at the vocalist's group rather than the requested album. | Executor stopped one hop early |
| `5a7a5274` | Sir Francis Nethersole / Francis Nethersole | Entity and comparison were correct; the answer omitted the honorific required by strict EM. | Answer-span precision |

Two exact-match successes also have process defects: one trajectory repaired a
false birth-year premise before answering correctly, while another retained an
incorrect person mapping but happened to produce the correct nationality.
Consequently terminal EM alone is not used to certify contract quality.

## Telemetry

- 91 total saved API attempts: 71 local Director turns and 20 Executor
  attempts represented by 19 final execution records.
- 245,705 input tokens and 13,904 output tokens were recorded for AgentGraph.
- Summed per-call latency was 423.29 seconds; manifest wall time was 304.87
  seconds because calls ran concurrently.
- Two long Executor calls consumed 72.94% of output tokens and 81.50% of summed
  latency.  This is recorded as a routing/cost observation, not added to task
  reward.
- The Direct receipt reports its original historical calls.  New Direct calls
  made by this v2 run were zero.

## Receipt completeness

The saved artifacts contain:

- task ID, question, ground truth, final answer, per-task EM/F1 and paired
  failure class;
- every Director prompt, raw policy response, parsed action, Canvas feedback,
  request ID, sampling seed, tokens/log-prob receipt, latency, and graph
  snapshot;
- every executed Agent's free-text contract, exact model/provider, rendered
  input messages, upstream envelope, output, tokens, latency, attempt count,
  and final provider response receipt;
- the actual Output inbox, evaluator receipt, version bundle, complete
  trajectory, Stable Zero checks, and aggregate report.

Role semantics intentionally remain `agent_id + free-text contract`; there is
no fixed role enum.  Two receipt limitations remain: reused Direct records do
not carry a per-record reuse marker, and retries retain the final response plus
attempt count rather than a separate record for every failed HTTP attempt.

## Bugs and formal-Step0 blockers exposed by v2

1. Catalog presentation order currently uses the same seed as Director
   generation.  Because generation seed varies by rollout, different rollouts
   of one same-task/same-condition GRPO group could see different first prompts.
   Formal Step 0 must use a task/condition-stable catalog-order seed while
   retaining rollout-varying sampling seeds.
2. The FINISH parser rejects ordinary multiple wrappers, but its current regex
   can accept a nested `<answer>` wrapper.  Feedback and terminal acceptance
   must share one strict parser that requires exactly one opening tag, one
   closing tag, non-empty content, and no nested answer tag.
3. Final manifest materialization drops the `direct_reused_from` field that is
   present during collection.  The completed manifest must preserve that
   provenance so historical Direct API attempts are not mistaken for calls
   made in this run.
4. The diagnostic YAML contains `max_context_tokens` and `live_adapter_name`
   declarations that are not consumed by the evaluation path; the actual
   service context recorded by preflight/manifest is authoritative.  A formal
   config must remove or validate such fields instead of presenting them as
   active controls.

## Decision

```text
V2_EVALUATION_PIPELINE_COMPLETE = YES
V2_DIAGNOSTIC_STABLE_ZERO = YES
MULTI_MODEL_ROUTING_OBSERVED = YES
MULTI_AGENT_COLLABORATION_VALIDATED = NO
FORMAL_POLICY_STEP_000000 = NO
FORMAL_STEP0_READY = NO
```

The next justified architecture version is limited to the four concrete
compatibility/receipt fixes above.  Deeper graphs, fixed roles, and method-level
MACE/Bayesian/Skill changes are not justified by this diagnostic.
