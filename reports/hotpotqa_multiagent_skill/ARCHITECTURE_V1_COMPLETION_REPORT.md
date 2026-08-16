# Architecture Completion Report — HotpotQA Multi-Agent v1

## Result

The end-to-end chain is executable and receipt-complete:

`Question → local Qwen3.5-9B Director → progressive Canvas → AgentGraph Runtime → Agent communication → Output → Hotpot evaluator → Trajectory`

The two-task runtime canary passed Stable Zero on physical GPU 4, local port
8015, with the existing cross-dataset smoke adapter explicitly labelled as a
warm-start diagnostic policy. This is not yet the formal HotpotQA
`policy_step_000000`.

## Completed and verified

- FlowSteer-style six-action progressive Canvas remains atomic and supports
  arbitrary valid serial, parallel, fan-in/out and finite reciprocal graphs.
- The Director remains local Qwen3.5-9B. Its prompt is still short and neutral;
  it now explains only free-contract fields and existing relation semantics.
- The old sampled preferred-model hint was removed after saved evidence showed
  it matched 92.1% of `add_agent` choices. Eight exact Executor IDs are now
  exposed with equal priors and bounded canary metadata.
- Communication keeps a free artifact body while adding runtime-known source,
  target, type, dependency/request and graph revision. The same envelope is
  present in rendered prompts and trajectory receipts.
- Canvas execution feedback includes bounded per-Agent artifacts and upstream
  source IDs, without evaluator correctness or gold evidence.
- HotpotQA FINISH is rejected unless the latest Output is exactly one non-empty
  `<answer>...</answer>` wrapper. This is config-scoped and does not change
  WebShop/ALFWorld terminal actions.
- Explicit ordered task-ID selection freezes the architecture-development
  diagnostic slice without changing the evaluator or aligned dataset.
- 172 unit/regression tests passed (the unrelated pandas-only dataset test was
  not run in the serving environment); Ruff and `git diff --check` passed.

## Stable Zero receipt

| Check | Result |
| --- | --- |
| Fixed canary tasks | 2/2 |
| Director/Canvas/Runtime completed | 2/2 |
| Explicit FINISH | 2/2 |
| Valid evaluator receipt | 2/2 |
| Full Director token/logprob/request receipts | 2/2 |
| Output inbox saved | 2/2 |
| Provider/runtime failures | 0 |
| Malformed FINISH accepted | 0 |

Both canary questions were nevertheless wrong (0/2 EM) and both final graphs
were singleton graphs using `deepseek-v4-flash`. Stable Zero therefore means
the chain is operational, not that the architecture is performance-ready.

## Reserved interfaces, not claimed complete

- MACE/Bayesian exploration and forced probes remain outside reward and were
  not run.
- Skill schema/store/gate/retrieval components exist, but the full
  candidate→paired evidence→independent validation→ACTIVE→retrieval→ON/OFF
  coordinator is not connected.
- Formal deterministic initial LoRA `policy_step_000000`, immutable per-step
  checkpoints, optimizer-state continuation and a Hotpot-only Step runner are
  still required before controlled GRPO.
- No `role_family` field was added: the current evidence does not require a
  typed role schema, and neither FlowSteer nor SkillFlow provides one for this
  free AgentGraph.

## Known issues and current judgment

- v1 removes confirmed protocol/routing biases, but the two canaries show that
  the warm-start Director still stops at the first valid singleton solution.
- The canary validates transport and terminal semantics only; the fixed 14-task
  development diagnostic is required before judging graph behavior.
- Multi-Agent collaboration and causal model routing are not yet validated.

`STABLE_ZERO = YES`

`ARCHITECTURE_PERFORMANCE_READY = NOT_YET_EVALUATED`
