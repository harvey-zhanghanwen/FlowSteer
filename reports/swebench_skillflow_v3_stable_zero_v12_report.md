# SWE-bench Verified Architecture Validation

Fixed test tasks: **2**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill evolution ran. No Skill was injected.

Official primary metric: **Resolved Rate** (`SWE_bench_Verified_official_Docker_harness_resolved_rate`). Only the official SWE-bench harness result is a valid task label; model prose and local `run_tests` output are not rewards.

| Condition | Completed | Evaluator valid | Resolved | Resolved Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 2 | 2 | 0 | 0.00% |
| AgentGraph | 2 | 2 | 0 | 0.00% |

AgentGraph - Direct official Resolved Rate: **+0.00 percentage points**. Direct and AgentGraph share tasks, repository snapshots, repository Tools, and the task-global repository episode budget; AgentGraph has additional Director/Agent inference, so the delta is descriptive.

AgentGraph explicit FINISH: **2/2**; max_rounds: **0**; terminal failures: **0**; operational/evaluator failures: **0**.

Repository Tool action groups (`search/view/edit/test/command`): Direct **{'search': 3, 'view': 2, 'edit': 0, 'test': 0, 'command': 46}**; AgentGraph **{'search': 0, 'view': 0, 'edit': 0, 'test': 0, 'command': 0}**. First-observable Wrong Demo diagnoses: **2**.

Agent count distribution: **{'2': 1, '8': 1}**. Natural topology distribution: **{'reciprocal': 1, 'serial_3_plus': 1}**.

## Receipt-based Wrong Demo failure taxonomy

Wrong-task denominator: **2**. Each wrong task contributes to exactly one primary observable category. Full Director/Canvas/Agent/communication/ReAct/Tool/terminal/evaluator receipts for every listed representative demo are stored in report JSON at `swebench_offline_receipts.wrong_demo_failure_taxonomy.representative_demos`.

| Primary observable category | Count | Share of wrong tasks | Representative demo |
|---|---:|---:|---|
| `collection_receipt_failure` | 0 | 0.00% | None |
| `provider_failure` | 0 | 0.00% | None |
| `repository_environment_failure` | 0 | 0.00% | None |
| `orchestration_failure` | 0 | 0.00% | None |
| `agent_communication_failure` | 0 | 0.00% | None |
| `repository_tool_failure` | 0 | 0.00% | None |
| `local_validation_failure` | 0 | 0.00% | None |
| `terminal_budget_failure` | 0 | 0.00% | None |
| `patch_publication_application_failure` | 2 | 100.00% | `swe-bench:astropy__astropy-14182:patch_publication_application_failure` |
| `official_target_test_failure` | 0 | 0.00% | None |
| `official_regression_failure` | 0 | 0.00% | None |
| `official_test_failure_unclassified` | 0 | 0.00% | None |
| `evaluator_runtime_failure` | 0 | 0.00% | None |
| `unclassified_receipt_failure` | 0 | 0.00% | None |

`first_observable_failure` is never renamed as root cause. Without explicit causal or intervention evidence, `first_causal_failure` and causal propagation remain null. SWE-bench uses patch application and official tests; answer-string canonicalization and LLM judging are not part of this evaluator.

## Failure types

- `equal_resolved`: 2
