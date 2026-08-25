# HealthBench Professional Architecture Completion Report

## Completed

- Materialized the 525-row official public `test` population in source order.
- Preserved the full ordered conversation as the model-visible task input.
- Separated rubrics, physician response, benchmark metadata, and canary into a
  task-ID-keyed evaluator-only store.
- Reused the unified FlowSteer-derived Canvas/AgentGraph/runtime/trajectory
  path without adding medical roles, a fixed topology, or a second runtime.
- Added a reference-compatible Professional terminal evaluator over pinned
  OpenAI `simple-evals` revision `652c89d`, exact grader
  `gpt-5.4-2026-03-05`, low reasoning effort, signed rubric score, and the
  Professional character-length adjustment.
- Added paired Direct-versus-AgentGraph configuration with the same local
  Qwen3.5-9B, generation condition, empty Tool condition, and evaluator.
- Added evaluator/provider/token/latency receipts and evaluator-only retry
  semantics that never resample an already frozen candidate or trajectory.

## Validated

- 108 focused unit/integration tests plus 13 subtests pass.
- Official schema conversion and public/private join are deterministic.
- Native conversation roles reach Direct and Agent executions.
- Rubrics and reference fields do not enter Director or Agent messages.
- Two-case Stable Zero: Direct 2/2 valid; AgentGraph 2/2 valid; explicit
  `FINISH` 2/2; complete trajectory and Output Agent inbox 2/2.
- No Tool, training, GRPO, backward, optimizer step, LoRA update/publication,
  MACE, Bayesian posterior, Skill retrieval, or Skill evolution ran.

## Reserved interfaces

- The unified per-node model selector remains available, but this paired
  comparison intentionally freezes one Executor model to avoid a model-pool
  confound.
- Unified Tool and per-Agent execution-mode interfaces remain available, but
  the official public Professional base condition declares no Tool protocol,
  so this evaluation exposes no Tool.
- Training, MACE, Bayesian, and Skill modules remain repository capabilities;
  they are outside this round and were not activated.

## Known limitations

- OpenAI's internal production HealthBench Professional evaluator is not
  public. Results are therefore labelled reference-compatible rather than
  internal-official.
- The paper's repeated-sampling aggregate is not reproduced by this one-sample
  paired architecture comparison.
- The two-case canary validates the chain only; it is not a benchmark score.
- Full 525-case runtime, natural topology distribution, and Wrong Demo analysis
  remain pending.

## Stable Zero

**Confirmed** for the fixed two-case chain:

`conversation -> Direct or Director/Canvas/AgentGraph -> complete assistant response -> private rubric grader -> evaluation receipt/trajectory`
