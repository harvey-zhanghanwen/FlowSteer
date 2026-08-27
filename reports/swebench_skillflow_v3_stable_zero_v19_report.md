# SWE-bench Verified Architecture Validation

Fixed test tasks: **128**. No training, GRPO, backward pass, optimizer update, LoRA publication, Bayesian update, or Skill evolution ran. No Skill was injected.

Official primary metric: **Resolved Rate** (`SWE_bench_Verified_official_Docker_harness_resolved_rate`). Only the official SWE-bench harness result is a valid task label; model prose and local `run_tests` output are not rewards.

| Condition | Completed | Evaluator valid | Resolved | Resolved Rate |
|---|---:|---:|---:|---:|
| Qwen3.5-9B Direct Local Baseline | 128 | 128 | 4 | 3.12% |
| AgentGraph | 128 | 128 | 4 | 3.12% |

AgentGraph - Direct official Resolved Rate: **+0.00 percentage points**. Direct and AgentGraph share tasks, repository snapshots, repository Tools, and the task-global repository episode budget; AgentGraph has additional Director/Agent inference, so the delta is descriptive.

AgentGraph explicit FINISH: **128/128**; max_rounds: **0**; terminal failures: **0**; operational/evaluator failures: **0**.

Repository Tool action groups (`search/view/edit/test/command`): Direct **{'search': 132, 'view': 65, 'edit': 76, 'test': 9, 'command': 2802}**; AgentGraph **{'search': 204, 'view': 167, 'edit': 86, 'test': 24, 'command': 2094}**. First-observable Wrong Demo diagnoses: **124**.

Agent count distribution: **{'1': 19, '2': 69, '3': 39, '4': 1}**. Natural topology distribution: **{'fan_in': 2, 'mixed': 1, 'reciprocal': 20, 'serial_2': 60, 'serial_3_plus': 26, 'single': 19}**.

## Receipt-based Wrong Demo failure taxonomy

Wrong-task denominator: **124**. Each wrong task contributes to exactly one primary observable category. Full Director/Canvas/Agent/communication/ReAct/Tool/terminal/evaluator receipts for every listed representative demo are stored in report JSON at `swebench_offline_receipts.wrong_demo_failure_taxonomy.representative_demos`.

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
| `patch_publication_application_failure` | 96 | 77.42% | `swe-bench:astropy__astropy-14182:patch_publication_application_failure` |
| `official_target_test_failure` | 0 | 0.00% | None |
| `official_regression_failure` | 0 | 0.00% | None |
| `official_test_failure_unclassified` | 28 | 22.58% | `swe-bench:sympy__sympy-13852:official_test_failure_unclassified` |
| `evaluator_runtime_failure` | 0 | 0.00% | None |
| `unclassified_receipt_failure` | 0 | 0.00% | None |

`first_observable_failure` is never renamed as root cause. Without explicit causal or intervention evidence, `first_causal_failure` and causal propagation remain null. SWE-bench uses patch application and official tests; answer-string canonicalization and LLM judging are not part of this evaluator.

## Failure types

- `agentgraph_higher_resolved`: 1
- `direct_higher_resolved`: 1
- `equal_resolved`: 126
