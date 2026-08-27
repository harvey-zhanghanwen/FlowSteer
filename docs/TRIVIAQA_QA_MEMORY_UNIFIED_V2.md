# TriviaQA train QA-memory unified-v2

This document is the recovery and execution entry point for the inference-only
TriviaQA QA-memory condition. It records interfaces and artifact locations,
not a formal accuracy claim.

## Versioned coordinates

- Feature branch: `feature/triviaqa-qa-memory-unified-v2-20260827`.
- Pre-QA-memory restore branch:
  `backup/triviaqa-unified-v2-v63-pre-qa-memory-20260827`.
- Evaluation config:
  `config/evaluation_triviaqa_qa_memory_unified_v2.yaml`.
- Frozen index manifest:
  `data/triviaqa_qa_memory_v1/index/manifest.json`.
- QA-memory source/index implementation:
  `src/interactive/triviaqa_qa_memory.py`.
- Tool adapter and bounded ReAct execution:
  `src/interactive/qa_tool_adapter.py` and
  `src/interactive/react_execution.py`.
- Progressive Canvas and Director control-plane projection:
  `src/interactive/agent_workflow_env.py` and
  `src/interactive/director.py`.
- Runtime wiring:
  `scripts/train_agentgraph_smoke.py`.
- Formal result analysis entry point:
  `scripts/analyze_triviaqa_qa_memory_results.py`.

## Data and execution contract

The main index contains 512 semantic-preserving paraphrases of frozen train QA
records. Its current manifest records 512 unique source tasks and no cycled
rows. The separate 128-record validation view is used only for split-isolation
checks and later evaluation; its questions, answers, accepted aliases,
supporting facts and evaluator receipts are absent from the index.

The local Qwen3.5-9B Director edits the Canvas and never acts as a Tool
principal. Its execution profile has `allowed_tools=[]`; it receives only
content-free control-plane receipts. A Director-created worker Agent owns the
`triviaqa.qa_memory` capability, runs bounded ReAct, and performs dynamic
`search`/`read` calls. The resulting artifact and Tool provenance travel to
reasoning, verification and Output Agents only through explicit AgentGraph
relations. No static retrieval prefetch or Web Search path is enabled.

Every accepted evaluation must establish all three runtime assertions:

1. `director_tool_calls=0`;
2. `retrieval_tool_calls_by_worker>0`;
3. `retrieval_artifact_routed_via_relation=true`.

It must also retain worker `agent_id`, query, rank, similarity, `memory_id`,
Tool Action--Observation receipt, communication envelope and Output provenance
in the lossless trajectory. A failed assertion is an architecture failure and
must not be replaced by a Director-side retrieval payload.

## Execution order

Use the project's existing Python environment and provider configuration; do
not put credentials in command arguments or reports.

```bash
python -m pytest \
  tests/unit/test_triviaqa_qamemory_v2_adapter.py \
  tests/unit/test_triviaqa_qamemory_v2_control_plane.py \
  tests/unit/test_triviaqa_qamemory_v2_wiring.py -q

python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_triviaqa_qa_memory_unified_v2.yaml \
  --prepare-only

python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_triviaqa_qa_memory_unified_v2.yaml \
  --canary-only

python scripts/evaluate_completion_benchmark_round.py \
  --config config/evaluation_triviaqa_qa_memory_unified_v2.yaml
```

The fixed-128 run is admitted only after focused tests, prepare-only selection
and the Stable Zero canary pass. Direct predictions are reused from the
versioned same-batch question-only baseline path declared in the config; they
must not be regenerated implicitly.

## Disabled paths

This condition is inference-only. Skill injection, MACE/Bayesian exploration,
GRPO, backward, optimizer updates, LoRA publication and policy synchronization
remain disabled. No result should be described as an ACTIVE-Skill or training
gain. Formal EM/F1, terminal failures and Tool/provenance statistics belong in
the completed version-bound report, not in this source/recovery document.

## Completed diagnostic run

The user-requested fixed-128 diagnostic run completed on 2026-08-27 with all
128 evaluator receipts valid, after retrying only one operationally missing
trajectory under the unchanged frozen condition. It did **not** pass Stable
Zero: all 128 AgentGraph trajectories ended in
`canvas_action_domain_exhausted`, so this condition is not a best-profile
candidate. Metrics, worker Tool ownership, post-hoc Answer Recall, failed
relation routing and reproducible wrong demos are recorded in
`reports/triviaqa_qa_memory_unified_v2/FINAL_REPORT_ZH.md` and
`formal_result_analysis.json`.
