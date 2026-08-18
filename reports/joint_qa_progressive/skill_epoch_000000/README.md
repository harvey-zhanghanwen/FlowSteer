# Joint-QA Skill evidence epoch 0

This frozen MACE--Bayesian--Skill epoch used the initial
`qwen35-9b-hotpot-step-000000` Director policy, the fixed ten-arm Executor
catalog, train-only discovery, and the first 20 independent
`skill_confirmation` tasks per dataset.  All 40 paired interventions started
from an empty Canvas; their 80 forced arms are excluded from GRPO and benchmark
EM/F1.

| Dataset | Selected candidate | Mean F1 effect | Calibrated interval | Harm probability | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| HotpotQA | `evidence_to_format_handoff` | -0.0300 | [-0.1600, 0.1167] | 0.6741 | rejected |
| TriviaQA | `dependency_aligned_topology` | +0.0467 | [-0.1367, 0.2400] | 0.30785 | rejected |

Neither candidate became `ACTIVE`, so this epoch was not injected into natural
Skill-on evaluation or LoRA/GRPO training.  The TriviaQA candidate trajectories
remained serial; its score effect is a full-trajectory prompt-prior
intent-to-treat estimate, not evidence of fan-in, fan-out, parallel, or
reciprocal topology adoption.

The complete local receipts are under
`artifacts/joint_qa_progressive/skill_epoch_000000/`, including
`publication_results.json`, `paired_observations.jsonl`, posterior snapshots,
selection/EVSI receipts, trajectories, and the candidate-only SkillStore.
