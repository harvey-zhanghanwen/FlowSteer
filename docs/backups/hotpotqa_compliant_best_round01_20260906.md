# HotpotQA compliant-best Round-01 backup

This branch preserves the highest complete HotpotQA AgentGraph result under
the project's fixed held-out validation and official-compatible EM/F1
protocol. Transductive QA/fact-memory conditions are excluded.

## Result

- Condition: `hotpotqa_round_01_stable_zero`
- Samples: 128/128 completed and evaluator-valid
- AgentGraph EM: 0.75 (96/128)
- AgentGraph token F1: 0.8444010416666666
- Direct EM: 0.7265625 (93/128)
- Direct token F1: 0.8207758884803922
- Optimizer updates in this evaluation: 0

## Version boundary

- Stable Zero source commit: `deac8b9bbbba84d35ead2b8d96e94db90ea16e22`
- Evaluation result commit: `46083d6b83f64348964f34ee8dbabba031e388b8`
- Best-profile selection commit: `5243b7b24cf3186b2504fde15c7a3f177302c9e0`

The branch starts from the best-profile selection commit, whose direct
ancestry contains the source and result commits above.

## Reproduction records

- Entrypoint: `scripts/evaluate_hotpotqa_round.py`
- Evaluation configuration: `config/evaluation_hotpotqa_round_01.yaml`
- Frozen model catalog: `config/model_catalog.yaml`
- Dataset catalog: `config/datasets_agentgraph.yaml`
- Best-profile receipt: `config/best_profiles/hotpotqa_agentgraph_v1.yaml`
- Run manifest: `artifacts/hotpotqa_round_01/run_manifest.json`
- Machine-readable report: `reports/hotpotqa_round_01/report.json`
- Human-readable report: `reports/hotpotqa_round_01/report.md`
- Diagnostic report: `reports/hotpotqa_round_01/diagnostic_report.md`

The tracked Round-01 artifact directory also contains the fixed task IDs,
Direct predictions, AgentGraph trajectories, paired results, wrong demos,
preflight receipt, and collection receipts used by the report.

## External runtime dependencies

The prepared `data/agentgraph_v1` records, the Qwen3.5-9B base model, and the
`theta_smoke_step_000001` LoRA adapter are external runtime assets and are not
duplicated in this Git branch. Their expected locations and policy identifiers
are recorded in the evaluation configuration and run report. Provider
credentials must be supplied through environment variables and are not stored
in this branch.

This result uses the project's held-out validation split derived from the
HotpotQA native training source. It is not the HotpotQA official development
split and is not the later transductive fact-memory condition.
