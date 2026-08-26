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

- 110 focused unit/integration tests plus 13 subtests pass.
- Official schema conversion and public/private join are deterministic.
- Native conversation roles reach Direct and Agent executions.
- Rubrics and reference fields do not enter Director or Agent messages.
- Two-case Stable Zero: Direct 2/2 valid; AgentGraph 2/2 valid; explicit
  `FINISH` 2/2; complete trajectory and Output Agent inbox 2/2.
- Complete public-test request run: Direct 525/525 evaluator-valid;
  AgentGraph 503/525 evaluator-valid with 503 explicit `FINISH`, 22 reportable
  `max_rounds` terminal failures, and zero current operational/evaluator
  failures.
- No Tool, training, GRPO, backward, optimizer step, LoRA update/publication,
  MACE, Bayesian posterior, Skill retrieval, or Skill evolution ran.

## Reserved interfaces

- The unified per-node model selector remains available, but this paired
  comparison intentionally freezes one Executor model to avoid a model-pool
  confound.
- Unified Tool and per-Agent execution-mode interfaces remain available, but
  the official public Professional base condition declares no Tool protocol,
  so this evaluation exposes no Tool.
- The repository contains reserved or experimental training, MACE, Bayesian,
  and Skill interfaces; this round neither validated nor activated them.

## Known limitations

- OpenAI's internal production HealthBench Professional evaluator is not
  public. Results are therefore labelled reference-compatible rather than
  internal-official.
- The paper's repeated-sampling aggregate is not reproduced by this one-sample
  paired architecture comparison.
- The two-case canary validates the chain only; it is not a benchmark score.
- Twenty-two AgentGraph workflows exhausted the 20-turn Director budget
  without legal `FINISH`. Strict metrics use 525 as the denominator;
  valid-only metrics are reported separately and no terminal grade is
  fabricated.
- All 22 terminal failures are multi-Agent graphs with three to eight nodes;
  together they consumed 1,195 generation attempts. This is evidence of a
  complex-graph termination tail, not a reason to add a fixed medical
  topology from public-test cases.

## Full public-test evaluation

The machine-readable and concise reports are `evaluation_report.json` and
`evaluation_report.md` in this directory.
The receipt-backed, terminal-first Wrong Demo classification is
`failure_taxonomy_report_zh.md`; its complete conversations, signed rubrics,
candidate responses, Director/Canvas/Agent traces, communication bodies, and
evaluator receipts remain in the ignored evaluator-private artifact boundary.

| Condition | Evaluator valid | Strict raw | Strict length-adjusted | Valid-only length-adjusted |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B Direct | 525/525 | 18.97% | 19.17% | 19.17% |
| AgentGraph | 503/525 | 22.65% | 20.24% | 21.12% |

AgentGraph improves the strict length-adjusted score by +1.07 percentage
points. All 525 terminal receipts contain 347 single-node, 80 serial-2, 17
serial-3-plus, 39 reciprocal, 18 fan-in, 4 fan-out, 4 parallel, and 16 mixed
topologies. Agent counts range from one to eight; relation counts range from
zero to six. These are natural Director outputs rather than fixed medical
roles or templates.

## Stable Zero

**Confirmed** for the fixed two-case chain:

`conversation -> Direct or Director/Canvas/AgentGraph -> complete assistant response -> private rubric grader -> evaluation receipt/trajectory`

The full public-test request run confirms the path for 503 workflows, but it
does not pass the all-task Stable Zero criterion because 22 workflows reached
`max_rounds` without `FINISH`; they were not sent to the rubric evaluator.
Stable Zero is a chain-validity claim, not a score threshold.
